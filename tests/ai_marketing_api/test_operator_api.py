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


def _compose_payload() -> dict:
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update(
        {
            "approved_proposal": proposal["proposal"],
            "claims": proposal["claims"],
        }
    )
    return payload


def _role_compose_payload(role: str) -> dict:
    if role == "commercial":
        content = "已批准能力：内容交接后必须经过人工终审。"
        space, record_type, kind = "company_public", "capability", "fact"
    elif role == "personal_ip":
        content = "第一次带团队出海时，我暂停了未经验证的投放。"
        space, record_type, kind = "personal_approved", "experience", "experience"
    else:
        return _compose_payload()
    return {
        "objective": "Compose the approved proposal",
        "topic": "Approved topic",
        "audience": "business owners",
        "channels": ["wechat_mp", "xiaohongshu"],
        "as_of": "2026-07-16T08:00:00Z",
        "approved_proposal": {
            "title": content,
            "angle": "Use only approved evidence.",
            "audience_value": "Provide a traceable explanation.",
            "key_points": [content],
            "suggested_formats": ["wechat_mp", "xiaohongshu"],
            "cta": None,
            "first_person": role == "personal_ip",
        },
        "claims": [
            {
                "text": content,
                "kind": kind,
                "evidence_ids": ["record-1"],
                "verification_status": "verified",
            }
        ],
        "authorized_context": [
            {
                "record_id": "record-1",
                "space": space,
                "record_type": record_type,
                "title": "Approved record",
                "content": content,
                "structured_data": {},
                "status": "approved",
                "authorized_roles": [role],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": None,
                "source_tier": "primary",
            }
        ],
    }


def _publishable_model_candidate(payload: dict) -> dict:
    required = [
        {"block_ref": block["block_ref"]}
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]

    def editorial(label: str) -> list[dict]:
        return [
            {
                "kind": "transition",
                "text": f"First, {label} opens with the concrete approved tension.",
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": f"What matters for {label} is the reader's changed choice.",
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": f"The implication for {label} is a clearer decision boundary.",
                "evidence_ids": [],
            },
        ]

    return {
        "_model": "publishable-test-model",
        "blocks": [*required, *editorial("the master story")],
        "platform_variants": [
            {
                "platform": channel,
                "blocks": [*required, *editorial(channel)],
            }
            for channel in payload["channels"]
        ],
    }


def _protected_endpoints() -> list[tuple[str, str, dict | None]]:
    return [
        ("GET", "/api/v1/operators/health", None),
        ("GET", "/api/v1/operators/industry/probe", None),
        ("POST", "/api/v1/operators/industry/propose", _base_payload()),
        ("POST", "/api/v1/operators/industry/revise", _revision_payload()),
        ("POST", "/api/v1/operators/industry/compose", _compose_payload()),
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
    assert data["schema_version"] == "operator.proposal.v1"
    assert data["schema_versions"] == {
        "proposal": "operator.proposal.v1",
        "content": "operator.content.v1",
    }
    assert set(data["roles"]) == {"commercial", "industry", "personal_ip"}
    for role in data["roles"]:
        probe = _request("GET", f"/api/v1/operators/{role}/probe")
        assert probe.status_code == 200
        assert probe.json()["direct_tools"] == []
        assert probe.json()["knowledge_source"] == "request_authorized_context_only"
        assert probe.json()["supported_actions"] == ["compose", "propose", "revise"]
        assert probe.json()["schema_versions"]["content"] == "operator.content.v1"


def test_api_directory_lists_the_compose_contract():
    response = _request("GET", "/", operator_auth=False)

    assert response.status_code == 200
    assert response.json()["endpoints"]["operatorCompose"] == "/api/v1/operators/{role_id}/compose"


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


def test_compose_without_model_never_promotes_safe_fallback(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]

    response = _request("POST", "/api/v1/operators/industry/compose", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "operator.content.v1"
    assert data["role_id"] == "industry"
    assert data["action"] == "compose"
    assert data["status"] == "quality_insufficient"
    assert data["requires_human_review"] is True
    assert data["critic"]["passed"] is False
    assert data["master_content"] is None
    assert data["platform_variants"] == []
    assert "model_generation_required" in data["critic"]["errors"]


@pytest.mark.parametrize("role", ["commercial", "personal_ip"])
def test_compose_endpoint_blocks_fallback_for_each_role_profile(role, monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    payload = _role_compose_payload(role)

    response = _request("POST", f"/api/v1/operators/{role}/compose", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["role_id"] == role
    assert data["status"] == "quality_insufficient"
    assert data["critic"]["passed"] is False
    assert "model_generation_required" in data["critic"]["errors"]
    assert data["blocks"] == []


def test_compose_retries_shallow_model_copy_once(monkeypatch):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]
    calls: list[dict] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            required = [
                {"block_ref": block["block_ref"]}
                for block in payload["canonical_block_registry"]
                if block["required"]
            ]
            evidence_id = next(
                block["evidence_ids"][0]
                for block in payload["canonical_block_registry"]
                if block["required"] and block["evidence_ids"]
            )
            return {
                "_model": "shallow-test-model",
                "blocks": [
                    *required,
                    {
                        "kind": "opinion",
                        "text": "What matters is how this rule changes the reader's available choice.",
                        "evidence_ids": [evidence_id],
                    },
                ],
                "platform_variants": [
                    {"platform": channel, "blocks": required}
                    for channel in payload["channels"]
                ],
            }
        return _publishable_model_candidate(payload)

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "content_ready"
    assert len(calls) == 2
    assert "quality_retry" not in calls[0]
    assert calls[1]["quality_retry"]["critic_errors"]
    assert calls[1]["quality_retry"]["required_shape"]
    assert any(
        "kind=opinion" in item
        for item in calls[1]["quality_retry"]["required_shape"]
    )
    assert "Do not describe the review process" in calls[1]["quality_retry"]["instruction"]
    assert "evidence_ids to []" in calls[1]["quality_retry"]["block_binding_rule"]
    assert any(
        diagnostic.endswith(":evidence_forbidden")
        for diagnostic in calls[1]["quality_retry"]["discarded_block_diagnostics"]
    )


def test_compose_keeps_good_surfaces_while_retry_fills_other_platforms(monkeypatch):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        candidate = _publishable_model_candidate(model_payload)
        required = [
            {"block_ref": block["block_ref"]}
            for block in model_payload["canonical_block_registry"]
            if block["required"]
        ]
        if len(calls) == 1:
            # The first attempt has a good master and WeChat article, but its
            # Xiaohongshu body is only a fact wrapper.
            next(
                variant
                for variant in candidate["platform_variants"]
                if variant["platform"] == "xiaohongshu"
            )["blocks"] = required
        else:
            # The retry repairs Xiaohongshu but regresses surfaces that were
            # already good. Whole-surface aggregation must retain the earlier
            # copy instead of making the second response all-or-nothing.
            candidate["blocks"] = required
            next(
                variant
                for variant in candidate["platform_variants"]
                if variant["platform"] == "wechat_mp"
            )["blocks"] = required
        return candidate

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "content_ready"
    assert len(calls) == 2
    assert {variant["platform"] for variant in data["platform_variants"]} == {
        "wechat_mp",
        "xiaohongshu",
    }
    assert next(
        check
        for check in data["critic"]["checks"]
        if check["code"] == "social_surface_retry_aggregation"
    )["passed"] is True


def test_compose_uses_the_same_byte_stable_role_prompt(monkeypatch):
    prompts: list[str] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        prompts.append(system)
        return fallback

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    compose_payload = _base_payload()
    compose_payload.update(
        {"approved_proposal": proposal["proposal"], "claims": proposal["claims"]}
    )
    composed = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=compose_payload,
    )

    assert composed.status_code == 200
    assert len(prompts) == 2
    assert prompts[0].encode("utf-8") == prompts[1].encode("utf-8")
    assert composed.json()["system_prompt_sha256"] == hashlib.sha256(
        prompts[1].encode("utf-8")
    ).hexdigest()


def test_blocked_compose_never_calls_the_model(monkeypatch):
    payload = _compose_payload()
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("blocked evidence must not reach the model")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    payload["authorized_context"][0]["source_tier"] = "secondary"

    response = _request("POST", "/api/v1/operators/industry/compose", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "evidence_insufficient"
    assert response.json()["master_content"] is None
    assert called is False


def test_compose_rejects_extra_fields_and_cross_role_context_before_model(monkeypatch):
    extra = _compose_payload()
    cross_role = _compose_payload()
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid compose input must not reach the model")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    extra["direct_database"] = "please"
    extra_response = _request(
        "POST", "/api/v1/operators/industry/compose", json=extra
    )

    cross_role["authorized_context"][0]["space"] = "company_internal"
    cross_role["authorized_context"][0]["record_type"] = "delivery_boundary"
    cross_role_response = _request(
        "POST", "/api/v1/operators/industry/compose", json=cross_role
    )

    assert extra_response.status_code == 422
    assert cross_role_response.status_code == 403
    assert cross_role_response.json()["detail"]["code"] == "knowledge_space_denied"
    assert called is False


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


def test_caller_selected_llm_endpoint_never_inherits_the_service_key(monkeypatch):
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "server-owned-secret")

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("an incomplete endpoint override must not reach the network")

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", ForbiddenClient)
    fallback = {"status": "safe-fallback"}
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            fallback,
            openai_base_url="https://attacker.example/v1",
            openai_api_key=None,
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "openai_override_base_url_and_api_key_must_be_supplied_together"
    assert "server-owned-secret" not in str(result)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://93.184.216.34/v1",
        "http://127.0.0.1:8000/v1",
        "http://169.254.169.254/latest",
        "http://10.0.0.8/v1",
        "http://[::1]/v1",
    ],
)
def test_llm_endpoint_rejects_non_public_ip_targets(monkeypatch, base_url):
    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a private endpoint must not reach the network")

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", ForbiddenClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url=base_url,
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "invalid_openai_base_url"
    assert "caller-owned-secret" not in str(result)


def test_llm_endpoint_rejects_dns_with_any_private_answer(monkeypatch):
    monkeypatch.setattr(
        marketing_api,
        "_resolve_host_addresses",
        lambda _host, _port: {"203.0.113.10", "10.0.0.8"},
    )

    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["_error"] == "invalid_openai_base_url"


def test_llm_endpoint_never_follows_redirects(monkeypatch):
    monkeypatch.setattr(
        marketing_api,
        "_resolve_host_addresses",
        lambda _host, _port: {"93.184.216.34"},
    )

    class RedirectClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **kwargs):
            assert url == "https://93.184.216.34/v1/chat/completions"
            assert kwargs["headers"]["Host"] == "model.example"
            assert kwargs["extensions"] == {"sni_hostname": "model.example"}
            return httpx.Response(
                307,
                headers={"Location": "http://169.254.169.254/latest"},
            )

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", RedirectClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "llm_redirect_forbidden"


def test_llm_endpoint_connects_to_the_validated_ip_without_resolving_hostname_again(monkeypatch):
    resolutions = []

    def resolve_once(host, port):
        resolutions.append((host, port))
        return {"93.184.216.34"}

    monkeypatch.setattr(marketing_api, "_resolve_host_addresses", resolve_once)

    class PinnedClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **kwargs):
            assert "model.example" not in url
            assert url == "https://93.184.216.34/v1/chat/completions"
            assert kwargs["headers"]["Host"] == "model.example"
            assert kwargs["extensions"] == {"sni_hostname": "model.example"}
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"status":"ok"}'}}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", PinnedClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert resolutions == [("model.example", 443)]
    assert result["status"] == "ok"
