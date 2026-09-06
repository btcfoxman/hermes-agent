from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

import ai_marketing_api.main as api
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from tests.ai_marketing_api.test_operator_api import (
    _compose_payload,
    _publishable_model_candidate,
    _request,
    _role_compose_payload,
)


@pytest.fixture(autouse=True)
def service_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def _brief(role="industry"):
    return {
        "role_id": role,
        "objective": "Help animation tool builders decide what to validate",
        "audience": "animation workbench developers",
        "reader_value": "A concrete integration decision",
        "angle": "A small verification task before integration",
        "tone": "practical and specific",
        "cta": "Review the integration documentation",
        "channels": ["wechat_mp"],
        "product_id": "api",
        "landing_url": "https://aiid.edu.kg",
    }


def _review(passed=True):
    return {
        "_model": "independent-editor",
        "relevant_to_brief": passed,
        "useful_to_reader": passed,
        "clear_and_specific": passed,
        "platform_fit": passed,
        "issues": []
        if passed
        else ["Generic filler does not help animation workbench developers."],
        "failed_surfaces": [] if passed else ["master"],
    }


@pytest.mark.parametrize("role", ["commercial", "industry", "personal_ip"])
def test_public_brief_preserves_audience_and_product_intent_for_each_role(
    role, monkeypatch
):
    payload = _role_compose_payload(role)
    payload["channels"] = ["wechat_mp"]
    payload["public_editorial_brief"] = _brief(role)
    payload["objective"] = "private raw objective, do not forward"
    payload["audience"] = "private raw audience, do not forward"
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return {**fallback, "_error": "unavailable"}

    monkeypatch.setattr(api, "_llm_json", fake)
    response = _request("POST", f"/api/v1/operators/{role}/compose", json=payload)
    assert response.status_code == 200
    assert calls[0]["audience"] == payload["public_editorial_brief"]["audience"]
    assert calls[0]["objective"] == payload["public_editorial_brief"]["objective"]
    assert calls[0]["public_editorial_brief"]["tone"] == "practical and specific"
    assert "private raw" not in json.dumps(calls)


def test_cross_role_brief_is_denied_before_any_model_call(monkeypatch):
    payload = _compose_payload()
    payload["public_editorial_brief"] = _brief("commercial")

    async def forbidden(*args, **kwargs):
        pytest.fail("A mismatched brief must not reach a model")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    response = _request("POST", "/api/v1/operators/industry/compose", json=payload)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "editorial_brief_role_denied"


@pytest.mark.parametrize("passed", [True, False])
def test_editorial_review_is_separate_from_safe_structure(passed, monkeypatch):
    payload = _compose_payload()
    payload["public_editorial_brief"] = _brief()
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append((system, body))
        return (
            _review(passed)
            if "editorial_review" in body
            else _publishable_model_candidate(body)
        )

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert len(calls) == (2 if passed else 4)
    assert calls[0][0] == calls[1][0]  # immutable role prompt
    assert data["status"] == ("content_ready" if passed else "quality_insufficient")
    assert data["generation_trace"]["editorial_review"] == (
        "passed" if passed else "failed"
    )
    assert data["generation_trace"]["model_calls"] == len(calls)
    assert data["editorial_assessment"]["relevant_to_brief"] is passed
    assert data["requires_human_review"] is True
    if not passed:
        assert "editorial_review_failed" in data["critic"]["errors"]


def test_model_cannot_self_attest_editorial_pass_in_compose(monkeypatch):
    payload = _compose_payload()
    payload["public_editorial_brief"] = _brief()
    payload["runtime_budget"] = {"max_model_calls": 1}
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return {
            **_publishable_model_candidate(body),
            "generation_trace": {"editorial_review": "passed"},
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert len(calls) == 1
    assert data["status"] == "quality_insufficient"
    assert data["generation_trace"]["budget_exhausted"] is True
    assert data["generation_trace"]["editorial_review"] == "not_completed"


def test_global_budget_limits_partial_surface_repair(monkeypatch):
    payload = _compose_payload()
    payload["runtime_budget"] = {"max_model_calls": 2, "max_surface_revisions": 1}
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return {**fallback, "_model": "real-but-shallow"}

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert len(calls) <= 2
    assert data["status"] != "content_ready"
    assert data["generation_trace"]["repair_calls"] <= 1
    assert all(
        len(attempt["prompt_sha256"]) == 64
        for attempt in data["generation_trace"]["attempts"]
    )


def test_elapsed_deadline_cancels_model_and_reports_budget(monkeypatch):
    run = BoundedModelRun(RuntimeBudget(max_elapsed_seconds=1))
    run.started -= 0.99
    cancelled = []

    async def slow(*args, **kwargs):
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)

    result = asyncio.run(run.call(slow, "system", {}, {}))
    assert result["_error"] == "runtime_deadline_exceeded"
    assert cancelled == [True]
    assert run.exhausted is True
    assert run.attempts[0].outcome == "timeout"


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_model_calls", 9),
        ("max_surface_revisions", 4),
        ("max_elapsed_seconds", 601),
    ],
)
def test_runtime_hard_bounds_are_not_prompt_suggestions(field, value):
    payload = _compose_payload()
    payload["runtime_budget"] = {field: value}
    assert (
        _request("POST", "/api/v1/operators/industry/compose", json=payload).status_code
        == 422
    )


def test_e2e_only_personal_story_is_never_public_even_when_approved(monkeypatch):
    payload = _role_compose_payload("personal_ip")
    payload["authorized_context"][0]["structured_data"] = {
        "e2e_only": True,
        "publishable": False,
    }

    async def forbidden(*args, **kwargs):
        pytest.fail("Test-only private material must be blocked before model")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    data = _request(
        "POST", "/api/v1/operators/personal_ip/compose", json=payload
    ).json()
    assert data["status"] != "content_ready"
    assert data["generation_trace"]["origin"] == "blocked_before_model"


def _planner_payload():
    return {
        "mandate": {
            "objective": "Useful content for media tool builders",
            "products": [
                {
                    "product_id": "api",
                    "name": "API access",
                    "audience": "tool developers",
                    "value_propositions": ["model choice"],
                }
            ],
        },
        "opportunities": [
            {
                "id": "opp-api",
                "role_id": "commercial",
                "title": "Integration workflow",
                "verified": True,
            },
            {
                "id": "opp-news",
                "role_id": "industry",
                "title": "Documented model change",
                "verified": True,
            },
            {
                "id": "opp-private",
                "role_id": "personal_ip",
                "title": "Unconfirmed story",
                "verified": False,
            },
        ],
        "max_packages": 1,
    }


@pytest.mark.parametrize(
    "selection,valid",
    [
        ({"opportunity_id": "opp-api", "role_id": "commercial"}, True),
        ({"opportunity_id": "invented", "role_id": "commercial"}, False),
        ({"opportunity_id": "opp-api", "role_id": "personal_ip"}, False),
        ({"opportunity_id": "opp-private", "role_id": "personal_ip"}, False),
    ],
)
def test_planner_can_select_only_given_verified_role_bound_opportunities(
    selection, valid, monkeypatch
):
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return {
            "_model": "planner-model",
            "selections": [
                {
                    **selection,
                    "topic": "Integration workflow",
                    "angle": "Explain the first integration check",
                    "reason": "Useful next step",
                }
            ],
            "skips": [],
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request(
        "POST", "/api/v1/operators/editorial/plan", json=_planner_payload()
    ).json()
    assert data["status"] == ("planned" if valid else "unavailable")
    assert bool(data["selections"]) is valid
    assert all(item["id"] != "opp-private" for item in calls[0]["opportunities"])


def test_planner_zero_day_is_explicit_and_does_not_invent_work(monkeypatch):
    payload = _planner_payload()
    payload["max_packages"] = 0

    async def forbidden(*args, **kwargs):
        pytest.fail("A configured zero-package day does not consume a model call")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    data = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert data["status"] == "no_opportunities"
    assert data["selections"] == []
    assert data["model"] == "none"


def test_planner_authentication_required():
    assert (
        _request(
            "POST",
            "/api/v1/operators/editorial/plan",
            json=_planner_payload(),
            operator_auth=False,
        ).status_code
        == 401
    )


def test_owner_revision_is_editing_intent_not_evidence(monkeypatch):
    payload = _compose_payload()
    payload["public_editorial_brief"] = _brief()
    payload["owner_revision"] = {
        "instruction": "Start from the developer's first task.",
        "platform_variants": [
            {
                "platform": "wechat_mp",
                "title": "New draft",
                "body": "Unapproved guaranteed discount 99%.",
            }
        ],
    }
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return (
            _review()
            if "editorial_review" in body
            else _publishable_model_candidate(body)
        )

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready"
    assert calls[0]["owner_revision"] == payload["owner_revision"]
    assert all("99%" not in item["text"] for item in calls[0]["claims"])
    assert "99%" not in data["master_content"]
    assert calls[1]["editorial_review"]["owner_revision"] == payload["owner_revision"]
    assert data["requires_human_review"] is True


def test_owner_revision_cannot_add_an_unrequested_channel(monkeypatch):
    payload = _compose_payload()
    payload["owner_revision"] = {
        "instruction": "Change it",
        "platform_variants": [{"platform": "douyin", "body": "New text"}],
    }

    async def forbidden(*args, **kwargs):
        pytest.fail("Cross-channel revision must fail before a model call")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    assert (
        _request("POST", "/api/v1/operators/industry/compose", json=payload).status_code
        == 403
    )


def test_http_compose_verifies_expression_then_reviews_actual_rendered_copy(
    monkeypatch,
):
    payload = _compose_payload()
    fact = "The toolbox supports image and video creation."
    public = "Images and videos can be created in the toolbox."
    payload["authorized_context"][0]["content"] = fact
    payload["claims"] = [
        {
            "text": fact,
            "kind": "fact",
            "evidence_ids": ["source-1"],
            "verification_status": "verified",
        }
    ]
    payload["approved_proposal"]["key_points"] = [fact]
    payload["approved_proposal"]["title"] = "image and video workflow"
    payload["public_editorial_brief"] = _brief()
    payload["fact_expression_mode"] = "grounded_paraphrase"
    stages = []

    async def fake(system, body, fallback, *args, **kwargs):
        if "claim_verification" in body:
            stages.append("verify")
            return {
                "_model": "verifier",
                "reviews": [
                    {
                        "pair_id": item["pair_id"],
                        "verdict": "supported",
                        "key_fields_preserved": True,
                    }
                    for item in body["claim_verification"]["pairs"]
                ],
            }
        if "editorial_review" in body:
            stages.append("review")
            assert public in body["editorial_review"]["draft"]["master_content"]
            return _review()
        stages.append("compose")
        result = _publishable_model_candidate(body)
        result["master_title"] = "image and video workflow"
        for block in result["blocks"]:
            if "block_ref" in block:
                block["public_text"] = public
        for variant in result["platform_variants"]:
            variant["title"] = "image and video creation guide"
            for block in variant["blocks"]:
                if "block_ref" in block:
                    block["public_text"] = public
        return result

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "verify", "review"]
    assert public in data["master_content"]
    assert (
        data["claim_binding_reviews"][0]["source_sha256"]
        == hashlib.sha256(fact.encode()).hexdigest()
    )
    assert data["generation_trace"]["model_calls"] == 3
