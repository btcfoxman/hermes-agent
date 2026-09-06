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


@pytest.mark.parametrize("role", ["commercial", "industry", "personal_ip"])
@pytest.mark.parametrize("fact_mode", ["exact", "grounded_paraphrase"])
def test_initial_public_brief_contract_is_task_based_and_preserves_safety(role, fact_mode):
    raw = _role_compose_payload(role)
    raw["runtime_budget"] = {
        "max_model_calls": 6, "max_elapsed_seconds": 180, "max_surface_revisions": 2,
    }
    raw["channels"] = [
        "wechat_moments", "wechat_mp", "wechat_channels", "douyin",
        "kuaishou", "xiaohongshu", "toutiao", "weitoutiao",
    ]
    raw["fact_expression_mode"] = fact_mode
    original = api.OperatorComposeRequest.model_validate(raw)
    baseline = api.content_request_payload(role, original, original.authorized_context)
    raw["public_editorial_brief"] = {**_brief(role), "channels": raw["channels"]}
    request = api.OperatorComposeRequest.model_validate(raw)
    payload = api.content_request_payload(role, request, request.authorized_context)

    assert payload["public_editorial_brief"]["role_id"] == role
    assert payload["objective"] == raw["public_editorial_brief"]["objective"]
    assert payload["audience"] == raw["public_editorial_brief"]["audience"]
    assert payload["industry_editorial_guard"] == {}
    assert payload["canonical_block_registry"] == baseline["canonical_block_registry"]
    assert payload["claims"] == baseline["claims"]
    assert payload["authorized_context"] == baseline["authorized_context"]
    assert payload["approved_preferences"] == baseline["approved_preferences"]
    assert payload["channels"] == baseline["channels"]
    assert request.runtime_budget == original.runtime_budget
    assert request.runtime_budget.max_model_calls == 6
    assert request.runtime_budget.max_elapsed_seconds == 180
    assert request.runtime_budget.max_surface_revisions == 2

    publication = payload["publication_brief"]
    writing_rules = json.dumps([
        publication, payload["platform_editorial_briefs"],
        payload["response_contract"]["editorial_shape"],
        payload["response_contract"]["safety"],
    ])
    for forced in (
        "Choose one event-specific thesis", "connecting one approved actor",
        "open with the actor and concrete consequence", "subject-action-consequence",
        "actor-action-consequence", "thesis tied to this event", "Explain the changed incentive",
        "name the changed incentive", "explain the event", "compact news commentary",
        "Connect the verified capability and offer", "what decision becomes easier",
    ):
        assert forced not in writing_rules
    assert "selection condition" in publication["role_thesis"] or role == "personal_ip"
    assert "reader task" in " ".join(publication["thesis_contract"])
    assert "verification question" in " ".join(publication["thesis_contract"])
    assert "owner_revision" in " ".join(publication["thesis_contract"])
    assert "unsupported product capabilities and outcomes remain unknown" in publication["reader_outcome"]
    assert len(set(payload["platform_editorial_briefs"].values())) == len(raw["channels"])

    safety = payload["response_contract"]["safety"]
    assert len(safety) == len(baseline["response_contract"]["safety"])
    changed_writing_rules = {5, 11, 15, 17, 19}
    for index, rule in enumerate(safety):
        if index not in changed_writing_rules:
            assert rule == baseline["response_contract"]["safety"][index]
    assert "two to six concise editorial blocks" in safety[5]
    assert "at least two authored blocks" in safety[5]
    assert "one to three authored blocks" in safety[5]
    assert "最值得注意的是" in safety[11]
    assert "meaningfully different in title, rhythm, depth, and reader action" in safety[-1]
    assert "opinion|transition|cta" not in payload["response_contract"]["blocks"]
    assert '{"kind":"opinion"' in payload["response_contract"]["blocks"]
    assert "single legal kind" in payload["response_contract"]["blocks"]
    if fact_mode == "grounded_paraphrase":
        assert "independent verification" in payload["constraints"][0]
        assert '"public_text"' in payload["response_contract"]["blocks"]
    if role == "commercial":
        assert "our company" in publication["authored_block_rule"]
    if role == "personal_ip":
        assert "First-person present-tense judgment is optional" in publication["role_thesis"]
        assert "canonical personal-card block" in publication["authored_block_rule"]


@pytest.mark.parametrize("role,surface", [
    ("industry", "master"), ("industry", "wechat_mp"), ("industry", "wechat_moments"),
    ("commercial", "master"), ("personal_ip", "master"),
])
def test_public_brief_focused_repair_has_consistent_safe_reader_task_contract(
    role, surface, monkeypatch
):
    payload = _role_compose_payload(role)
    payload["channels"] = ["wechat_mp", "wechat_moments"]
    payload["public_editorial_brief"] = {**_brief(role), "channels": payload["channels"]}
    payload["runtime_budget"] = {"max_model_calls": 2}
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        candidate = _publishable_model_candidate(body)
        if len(calls) == 1:
            required = [
                {"block_ref": block["block_ref"]}
                for block in body["canonical_block_registry"] if block["required"]
            ]
            target = candidate if surface == "master" else next(
                variant for variant in candidate["platform_variants"]
                if variant["platform"] == surface
            )
            target["blocks"] = required
        elif surface != "master":
            candidate.pop("blocks")
            candidate.pop("master_title")
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    response = _request("POST", f"/api/v1/operators/{role}/compose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(calls) == 2
    retry_body = calls[1]
    retry = retry_body["quality_retry"]
    assert retry["focus_surface"] == surface
    assert retry_body["public_editorial_brief"] == calls[0]["public_editorial_brief"]
    assert retry_body["canonical_block_registry"] == calls[0]["canonical_block_registry"]
    assert "put no Arabic number or Chinese counted quantity" in retry["instruction"]
    assert "evidence_ids to []" in retry["block_binding_rule"]
    assert "canonical factual ref unchanged" in retry["instruction"]
    assert "owner_revision" in retry["instruction"]
    assert "no new claims, product capabilities" in retry["instruction"]
    assert "causal cost/result comparisons" in retry["instruction"]
    recipe = retry["surface_contracts"][surface]["positive_writing_recipe"]
    positive = json.dumps([
        recipe, retry["positive_pattern"]["hook"],
        retry["positive_pattern"]["analysis"], retry["required_shape"],
    ])
    assert "selection condition" in positive
    assert "verification question" in positive
    assert "quantity-free" in positive
    for legacy in ("cash occupation", "cash-flow", "settlement", "bargaining", "approved remedy", "approved rule/remedy"):
        assert legacy not in positive
    contract = retry_body["response_contract"]
    assert "single legal kind" in contract["authored_kind_rule"]
    blocks = contract["blocks"] if surface == "master" else contract["platform_variants"][0]["blocks"]
    authored = [block for block in blocks if "kind" in block]
    assert authored[0]["kind"] == "transition"
    assert all(block["kind"] in {"transition", "opinion"} for block in authored)
    assert all(block["evidence_ids"] == [] for block in authored)
    assert all("claim_id" not in block and "block_ref" not in block for block in authored)
    expected_refs = [
        {"block_ref": block["block_ref"]}
        for block in retry_body["canonical_block_registry"] if block["required"]
    ]
    assert [block for block in blocks if "block_ref" in block] == expected_refs
    # A successful structural repair cannot self-certify the independent
    # editor when the unchanged global call budget leaves no review call.
    assert data["generation_trace"]["model_calls"] == 2
    assert data["generation_trace"]["editorial_review"] == "not_completed"
    assert data["status"] == "quality_insufficient"
    assert data["master_content"] is None
    assert data["blocks"] == data["platform_variants"] == []


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
        assert data["master_title"] is None
        assert data["master_content"] is None
        assert data["blocks"] == data["platform_variants"] == data["evidence_refs"] == []


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
    assert data["master_content"] is None
    assert data["blocks"] == data["platform_variants"] == data["evidence_refs"] == []


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
    # Leave enough startup time for Windows' event-loop scheduler; the test
    # must exercise cancellation during a call, not pre-call exhaustion.
    run.started -= 0.9
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


def test_independent_editor_repairs_all_failed_surfaces_without_overwriting_accepted(
    monkeypatch,
):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "wechat_moments"]
    payload["public_editorial_brief"] = {**_brief(), "channels": payload["channels"]}
    payload["runtime_budget"] = {"max_model_calls": 4}
    calls = []
    original = {}

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            draft = body["editorial_review"]["draft"]
            if len(calls) == 2:
                original.update(draft)
                return {**_review(False), "failed_surfaces": ["master", "wechat_mp"]}
            assert "revised" in draft["master_content"]
            assert "revised" in draft["platform_variants"][0]["body"]
            assert draft["platform_variants"][1] == original["platform_variants"][1]
            return _review()
        candidate = _publishable_model_candidate(body)
        if "editorial_revision" in body:
            assert body["editorial_revision"]["focus_surfaces"] == [
                "master",
                "wechat_mp",
            ]
            assert body["channels"] == ["wechat_mp"]
            for block in candidate["blocks"]:
                if "text" in block:
                    block["text"] += " revised"
            for variant in candidate["platform_variants"]:
                for block in variant["blocks"]:
                    if "text" in block:
                        block["text"] += " revised"
            extra = _publishable_model_candidate({
                **body,
                "channels": ["wechat_moments"],
            })["platform_variants"][0]
            extra["title"] = "Unrequested change"
            candidate["platform_variants"].append(extra)
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert len(calls) == 4
    assert data["generation_trace"]["repair_calls"] == 1
    assert data["generation_trace"]["attempts"][2]["surface"] == "master,wechat_mp"


def test_editorial_revision_reverifies_new_expression_and_keeps_untouched_expression(
    monkeypatch,
):
    payload = _compose_payload()
    fact = "The toolbox supports image and video creation."
    first_public = "Images and videos can be created in the toolbox."
    revised_public = "The toolbox can create images and videos."
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
    payload["runtime_budget"] = {"max_model_calls": 6}
    stages = []
    first_variant = {}

    async def fake(system, body, fallback, *args, **kwargs):
        if "claim_verification" in body:
            stages.append("verify")
            return {
                "_model": "verifier",
                "reviews": [
                    {
                        "pair_id": row["pair_id"],
                        "verdict": "supported",
                        "key_fields_preserved": True,
                    }
                    for row in body["claim_verification"]["pairs"]
                ],
            }
        if "editorial_review" in body:
            stages.append("review")
            draft = body["editorial_review"]["draft"]
            if len(stages) == 3:
                first_variant.update(draft["platform_variants"][0])
                assert first_public in draft["master_content"]
                return _review(False)
            assert revised_public in draft["master_content"]
            assert draft["platform_variants"][0] == first_variant
            return _review()
        revised = "editorial_revision" in body
        stages.append("repair" if revised else "compose")
        candidate = _publishable_model_candidate(body)
        candidate["master_title"] = "image and video workflow"
        for variant in candidate["platform_variants"]:
            variant["title"] = "image and video creation guide"
        for blocks in [
            candidate["blocks"],
            *[v["blocks"] for v in candidate["platform_variants"]],
        ]:
            for block in blocks:
                if "block_ref" in block:
                    block["public_text"] = revised_public if revised else first_public
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "verify", "review", "repair", "verify", "review"]
    supported = {
        row["public_text_sha256"]
        for row in data["claim_binding_reviews"]
        if row["verdict"] == "supported"
    }
    assert {
        hashlib.sha256(text.encode()).hexdigest()
        for text in (first_public, revised_public)
    } <= supported
    assert data["generation_trace"]["model_calls"] == 6


def test_unsafe_editorial_repair_cannot_hide_original_review_failure(monkeypatch):
    payload = _compose_payload()
    payload["public_editorial_brief"] = _brief()
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            return _review(False)
        candidate = _publishable_model_candidate(body)
        if "editorial_revision" in body:
            candidate["blocks"] = [
                {
                    "kind": "opinion",
                    "text": "Guaranteed benefits and risk-free success.",
                    "evidence_ids": [],
                }
            ]
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "quality_insufficient", data
    assert len(calls) == 3
    assert "editorial_review_failed" in data["critic"]["errors"]
    assert data["editorial_assessment"]["issues"] == _review(False)["issues"]
    assert "Guaranteed" not in (data["master_content"] or "")


def test_sparse_evidence_brief_and_editor_do_not_demand_invented_product_tutorial(
    monkeypatch,
):
    payload = _role_compose_payload("commercial")
    payload["channels"] = ["wechat_mp"]
    payload["public_editorial_brief"] = _brief("commercial")
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            review = body["editorial_review"]
            assert review["canonical_claims"]
            assert "sparse positioning evidence" in review["instruction"]
            return _review()
        assert len(body["evidence_scope_contract"]) == 4
        assert "not additional facts" in body["evidence_scope_contract"][0]
        candidate = _publishable_model_candidate(body)
        candidate["master_title"] = "customer workflow choice"
        candidate["platform_variants"][0]["title"] = "a practical customer choice"
        for blocks in [
            candidate["blocks"],
            candidate["platform_variants"][0]["blocks"],
        ]:
            first_editorial = next(block for block in blocks if "text" in block)
            blocks.remove(first_editorial)
            blocks.insert(0, first_editorial)
        # A closing judgment is not automatically an engagement invitation.
        candidate["blocks"].append({
            "kind": "closing",
            "text": "按眼前的创作任务选择入口，再决定是否需要扩展工作流。",
            "evidence_ids": [],
        })
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/commercial/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert len(calls) == 2
    assert "按眼前的创作任务" in data["master_content"]
    assert all(
        "editorial_intent_required" not in item for item in data["critic"]["warnings"]
    )
