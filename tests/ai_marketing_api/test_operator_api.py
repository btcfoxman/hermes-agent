from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest

os.environ["HERMES_API_KEY"] = "operator-test-secret"

import ai_marketing_api.main as marketing_api


app = marketing_api.app


@pytest.fixture(autouse=True)
def _operator_api_key(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    operator_auth = kwargs.pop("operator_auth", True)
    if operator_auth and path.startswith("/api/v1/"):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Authorization", f"Bearer {os.environ.get('HERMES_API_KEY', '')}")
        kwargs["headers"] = headers

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def _base_payload() -> dict:
    return {
        "objective": "Propose today's AI industry topic",
        "topic": "New model release",
        "audience": "AI product teams",
        "channels": ["wechat_mp"],
        "as_of": "2026-07-16T08:00:00Z",
        "authorized_context": [
            {
                "record_id": "source-1",
                "space": "industry",
                "record_type": "source_item",
                "title": "Official release",
                "content": "The vendor released version 2.",
                "structured_data": {},
                "status": "approved",
                "authorized_roles": ["industry"],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": "https://vendor.test/news",
                "source_tier": "official"
            }
        ],
    }


def _revision_payload() -> dict:
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Focus on product teams.", "previous": proposal})
    return payload


def _protected_endpoints() -> list[tuple[str, str, dict | None]]:
    return [
        ("GET", "/api/v1/operators/health", None),
        ("GET", "/api/v1/operators/industry/probe", None),
        ("POST", "/api/v1/operators/industry/propose", _base_payload()),
        ("POST", "/api/v1/operators/industry/revise", _revision_payload()),
        ("POST", "/api/v1/topic/analyze", {}),
        ("POST", "/api/v1/brief/generate", {}),
        ("POST", "/api/v1/draft/generate", {}),
        ("POST", "/api/v1/draft/humanize", {}),
        ("POST", "/api/v1/risk/review", {}),
    ]


def test_operator_health_and_role_probes_cover_all_profiles():
    response = _request("GET", "/api/v1/operators/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert set(data["roles"]) == {"commercial", "industry", "personal_ip"}
    for role in data["roles"]:
        probe = _request("GET", f"/api/v1/operators/{role}/probe")
        assert probe.status_code == 200
        assert probe.json()["direct_tools"] == []
        assert probe.json()["knowledge_source"] == "request_authorized_context_only"


def test_propose_returns_the_strict_versioned_contract(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    response = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "operator.proposal.v1"
    assert data["role_id"] == "industry"
    assert data["action"] == "propose"
    assert data["requires_human_review"] is True
    assert data["evidence_refs"] == ["source-1"]
    assert data["system_prompt_sha256"] == _request(
        "GET", "/api/v1/operators/industry/probe"
    ).json()["system_prompt_sha256"]


def test_revise_stays_in_the_same_profile(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Focus on product teams.", "previous": proposal})

    response = _request("POST", "/api/v1/operators/industry/revise", json=payload)

    assert response.status_code == 200
    assert response.json()["role_id"] == "industry"
    assert response.json()["action"] == "revise"
    assert "Focus on product teams" in response.json()["proposal"]["angle"]


def test_system_prompt_bytes_do_not_change_between_propose_and_revise(monkeypatch):
    prompts: list[str] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        prompts.append(system)
        return fallback

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Change the audience value only.", "previous": proposal})
    revised = _request("POST", "/api/v1/operators/industry/revise", json=payload)

    assert revised.status_code == 200
    assert len(prompts) == 2
    assert prompts[0].encode("utf-8") == prompts[1].encode("utf-8")
    assert hashlib.sha256(prompts[0].encode("utf-8")).hexdigest() == proposal["system_prompt_sha256"]


def test_revise_rejects_cross_profile_previous_output(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = {
        "objective": "Revise commercial proposal",
        "change_request": "Add a CTA.",
        "previous": proposal,
        "authorized_context": [],
        "as_of": "2026-07-16T08:00:00Z",
    }

    response = _request("POST", "/api/v1/operators/commercial/revise", json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "operator_role_mismatch"


def test_cross_role_context_is_rejected_before_model_execution(monkeypatch):
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not receive cross-role context")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    payload = _base_payload()
    payload["authorized_context"][0]["space"] = "company_internal"
    payload["authorized_context"][0]["record_type"] = "business_offer"

    response = _request("POST", "/api/v1/operators/industry/propose", json=payload)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "knowledge_space_denied"
    assert called is False


def test_bearer_auth_uses_the_existing_marketing_api_secret(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")

    denied = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        operator_auth=False,
    )
    allowed = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        headers={"Authorization": "Bearer operator-test-secret"},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "needs_input"


def test_only_public_health_is_available_without_a_service_secret(monkeypatch):
    assert _request("GET", "/health", operator_auth=False).status_code == 200

    protected_endpoints = _protected_endpoints()
    monkeypatch.delenv("HERMES_API_KEY", raising=False)

    for method, path, payload in protected_endpoints:
        kwargs = {"operator_auth": False}
        if payload is not None:
            kwargs["json"] = payload
        response = _request(method, path, **kwargs)
        assert response.status_code == 503, path


def test_every_protected_endpoint_rejects_the_wrong_bearer_key():
    for method, path, payload in _protected_endpoints():
        kwargs = {
            "operator_auth": False,
            "headers": {"Authorization": "Bearer definitely-wrong"},
        }
        if payload is not None:
            kwargs["json"] = payload
        response = _request(method, path, **kwargs)
        assert response.status_code == 401, path


def test_operator_authentication_fails_closed_when_secret_is_missing(monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)

    response = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        operator_auth=False,
    )

    assert response.status_code == 503


def test_unknown_role_is_rejected_by_the_router():
    response = _request("POST", "/api/v1/operators/chief/propose", json={"objective": "Do everything"})

    assert response.status_code == 422
