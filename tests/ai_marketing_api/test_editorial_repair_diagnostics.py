"""Bounded editor retries exercise real normalization, replacement and HTTP output."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import ai_marketing_api.main as api
from tests.ai_marketing_api.test_autonomous_editorial import _brief, _review
from tests.ai_marketing_api.test_operator_api import (
    _base_payload,
    _publishable_model_candidate,
    _request,
)


@pytest.fixture(autouse=True)
def service_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def _payload():
    payload = _base_payload()
    fact = payload["authorized_context"][0]["content"]
    payload.update(
        channels=["wechat_mp", "wechat_moments", "weitoutiao"],
        approved_proposal={
            "title": "vendor released version 2", "angle": "Explain the release",
            "audience_value": "A grounded reader judgment", "key_points": [fact],
            "suggested_formats": ["wechat_mp"], "first_person": False,
        },
        claims=[{"text": fact, "kind": "fact", "evidence_ids": ["source-1"],
                 "verification_status": "verified"}],
        runtime_budget={"max_model_calls": 6, "max_surface_revisions": 2,
                        "max_elapsed_seconds": 180},
    )
    payload["public_editorial_brief"] = {**_brief(), "channels": payload["channels"]}
    return payload


def _failed_review():
    return {**_review(False), "failed_surfaces": ["wechat_moments", "weitoutiao"]}


def _defect(candidate, defect):
    variants = candidate["platform_variants"]
    if defect == "invalid":
        variants[0]["blocks"] = [{"kind": "raw-kind-private-marker",
                                  "text": "raw-body-private-marker"}]
    elif defect == "invalid_blocks":
        variants[0]["blocks"] = {"private-key": "raw-body-private-marker"}
    elif defect == "unsafe":
        variants[0]["blocks"] = [
            block if "block_ref" in block else {
                **block, "text": "Guaranteed risk-free private-marker benefits."
            } for block in variants[0]["blocks"]
        ]
    elif defect == "shallow":
        variants[0]["blocks"] = [b for b in variants[0]["blocks"] if "block_ref" in b]
    elif defect == "duplicate_titles":
        variants[0]["title"] = variants[1]["title"]
    elif defect == "normalized_titles":
        for variant in variants:
            variant["title"] = "A calm planning checklist"


@pytest.mark.parametrize(("defect", "code"), [
    ("invalid", "model_block_ref_required"),
    ("invalid_blocks", "invalid_model_blocks"),
    ("unsafe", "social_editorial_hook_missing"),
    ("shallow", "social_editorial_hook_missing"),
    ("duplicate_titles", "platform_titles_not_distinct"),
    ("normalized_titles", "platform_titles_not_distinct"),
])
def test_rejected_batch_can_recover_with_accurate_feedback_and_new_review(monkeypatch, defect, code):
    payload, stages, repairs, original = _payload(), [], [], {}

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            draft = body["editorial_review"]["draft"]
            if not original:
                original.update(copy.deepcopy(draft))
                return _failed_review()
            assert draft["master_title"] == original["master_title"]
            assert draft["master_content"] == original["master_content"]
            assert draft["platform_variants"][0] == original["platform_variants"][0]
            return _review()
        candidate = _publishable_model_candidate(body)
        if "editorial_revision" not in body:
            stages.append("compose")
            return candidate
        stages.append("repair")
        repairs.append(body["editorial_revision"])
        if len(repairs) == 1:
            _defect(candidate, defect)
        else:
            feedback = body["editorial_revision"]["deterministic_validation"]
            assert any(value.endswith(":" + code) for value in feedback["errors"])
            assert "private-marker" not in json.dumps(feedback)
            if defect == "unsafe":
                assert "discarded_unsafe_model_editorial:wechat_moments:assertion_forbidden" in feedback["warnings"]
            if defect == "normalized_titles":
                assert feedback["warnings"] == [
                    "model_title_normalized:wechat_moments:missing_fact_anchor",
                    "model_title_normalized:weitoutiao:missing_fact_anchor",
                ]
            # These unrequested changes must not enter either gate or new review.
            candidate["blocks"] = [{"kind": "raw-kind-private-marker"}]
            candidate["master_title"] = "raw-title-private-marker"
            candidate["platform_variants"].append({
                "platform": "wechat_mp", "title": "raw-title-private-marker",
                "blocks": [{"kind": "raw-kind-private-marker"}],
            })
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "review", "repair", "repair", "review"]
    assert data["generation_trace"]["model_calls"] == 5
    assert data["generation_trace"]["repair_calls"] == 2
    assert data["generation_trace"]["budget_exhausted"] is False
    assert "private-marker" not in json.dumps(data)
    assert not any(error.startswith("editorial_repair_") for error in data["critic"]["errors"])


@pytest.mark.parametrize(("max_calls", "max_revisions", "repair_count"), [(6, 2, 2), (4, 2, 1), (6, 0, 0)])
def test_rejected_batches_stop_at_existing_limits_and_keep_only_bounded_diagnostics(
    monkeypatch, max_calls, max_revisions, repair_count,
):
    payload, stages = _payload(), []
    payload["runtime_budget"].update(max_model_calls=max_calls, max_surface_revisions=max_revisions)

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            return _failed_review()
        candidate = _publishable_model_candidate(body)
        repair = "editorial_revision" in body
        stages.append("repair" if repair else "compose")
        if repair:
            _defect(candidate, "normalized_titles")
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "quality_insufficient"
    assert stages == ["compose", "review", *(["repair"] * repair_count)]
    assert data["generation_trace"]["model_calls"] == 2 + repair_count
    assert data["generation_trace"]["repair_calls"] == repair_count
    assert data["generation_trace"]["budget_exhausted"] is False
    assert data["editorial_assessment"]["issues"] == _failed_review()["issues"]
    assert data["master_title"] is data["master_content"] is None
    assert data["blocks"] == data["platform_variants"] == data["evidence_refs"] == []
    assert "editorial_review_failed" in data["critic"]["errors"]
    if repair_count:
        assert "editorial_repair_rejected:combined" in data["critic"]["errors"]
        assert "editorial_repair_error:package:platform_titles_not_distinct" in data["critic"]["errors"]
        assert "model_title_normalized:wechat_moments:missing_fact_anchor" in data["critic"]["warnings"]
        check = next(c for c in data["critic"]["checks"] if c["code"] == "editorial_repair_rejected")
        assert not check["passed"] and "preceding reviewed draft" in check["message"]


def test_new_editor_failure_is_retried_and_final_assessment_is_not_the_initial_one(monkeypatch):
    payload, stages, reviews = _payload(), [], []

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            reviews.append(body["editorial_review"]["draft"])
            return {**_failed_review(), "issues": [f"Editorial assessment {len(reviews)} failed."]}
        stages.append("repair" if "editorial_revision" in body else "compose")
        return _publishable_model_candidate(body)

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert stages == ["compose", "review", "repair", "review", "repair", "review"]
    assert data["status"] == "quality_insufficient"
    assert data["editorial_assessment"]["issues"] == ["Editorial assessment 3 failed."]
    assert data["generation_trace"]["model_calls"] == 6
    assert data["generation_trace"]["budget_exhausted"] is True
    assert not any(error.startswith("editorial_repair_rejected") for error in data["critic"]["errors"])
    assert data["master_content"] is None and data["platform_variants"] == []


def test_paraphrase_repair_reserves_verification_and_editor_after_rejected_batch(monkeypatch):
    payload, stages, repairs, original = _payload(), [], [], {}
    fact = "The vendor creates image and video tools."
    public_text = "Image and video tools are created by the vendor."
    payload["fact_expression_mode"] = "grounded_paraphrase"
    payload["authorized_context"][0]["content"] = fact
    payload["claims"][0]["text"] = fact
    payload["approved_proposal"].update(title="image and video tools", key_points=[fact])

    async def fake(system, body, fallback, *args, **kwargs):
        if "claim_verification" in body:
            stages.append("verify")
            return {"_model": "test-verifier", "reviews": [
                {"pair_id": pair["pair_id"], "verdict": "supported", "key_fields_preserved": True}
                for pair in body["claim_verification"]["pairs"]
            ]}
        if "editorial_review" in body:
            stages.append("review")
            draft = body["editorial_review"]["draft"]
            if not original:
                original.update(copy.deepcopy(draft))
                return _failed_review()
            assert draft["master_content"] == original["master_content"]
            assert draft["platform_variants"][0] == original["platform_variants"][0]
            assert all(public_text in variant["body"] for variant in draft["platform_variants"][1:])
            return _review()
        candidate = _publishable_model_candidate(body)
        candidate["master_title"] = "image and video tools"
        for variant in candidate["platform_variants"]:
            variant["title"] = "image and video " + variant["platform"]
        repair = "editorial_revision" in body
        stages.append("repair" if repair else "compose")
        if repair:
            repairs.append(body)
            if len(repairs) == 1:
                _defect(candidate, "shallow")
            else:
                for variant in candidate["platform_variants"]:
                    for block in variant["blocks"]:
                        if "block_ref" in block:
                            block["public_text"] = public_text
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "review", "repair", "repair", "verify", "review"]
    assert data["generation_trace"]["model_calls"] == 6
    assert data["claim_binding_reviews"] and all(r["verdict"] == "supported" for r in data["claim_binding_reviews"])


@pytest.mark.parametrize("new_surface", ["master", "wechat_mp"])
def test_new_review_cannot_expand_the_original_repair_scope(monkeypatch, new_surface):
    payload, stages, reviewed, requests = _payload(), [], [], []

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            reviewed.append(body["editorial_review"]["draft"])
            if len(reviewed) == 1:
                return _failed_review()
            return {**_review(False), "issues": ["New assessment rejects another surface."],
                    "failed_surfaces": [new_surface]}
        repair = "editorial_revision" in body
        stages.append("repair" if repair else "compose")
        if repair:
            requests.append(body["editorial_revision"]["focus_surfaces"])
        return _publishable_model_candidate(body)

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert stages == ["compose", "review", "repair", "review"]
    assert requests == [["wechat_moments", "weitoutiao"]]
    assert reviewed[1]["master_content"] == reviewed[0]["master_content"]
    assert reviewed[1]["platform_variants"][0] == reviewed[0]["platform_variants"][0]
    assert data["status"] == "quality_insufficient"
    assert data["editorial_assessment"]["issues"] == ["New assessment rejects another surface."]
    assert f"editorial_repair_scope_expanded:{new_surface}" in data["critic"]["errors"]
    assert not any(c["code"] == "editorial_repair_rejected" for c in data["critic"]["checks"])
    assert data["master_content"] is None and data["platform_variants"] == []


def test_elapsed_limit_prevents_another_repair_even_with_calls_remaining(monkeypatch):
    payload, stages, runs = _payload(), [], []
    original_run = api.BoundedModelRun

    def capture_run(budget):
        run = original_run(budget)
        runs.append(run)
        return run

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            return _failed_review()
        repair = "editorial_revision" in body
        stages.append("repair" if repair else "compose")
        candidate = _publishable_model_candidate(body)
        if repair:
            _defect(candidate, "shallow")
            runs[0].started -= 181
        return candidate

    monkeypatch.setattr(api, "BoundedModelRun", capture_run)
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload).json()
    assert stages == ["compose", "review", "repair"]
    assert data["status"] == "quality_insufficient"
    assert "editorial_repair_rejected:surface" in data["critic"]["errors"]
    assert data["master_content"] is None and data["platform_variants"] == []


def test_repair_diagnostic_projection_drops_raw_suffixes_unknown_codes_and_other_surfaces():
    from ai_marketing_api.operator_review import replace_editorial_surfaces

    repaired = SimpleNamespace(status="evidence_insufficient", critic=SimpleNamespace(
        errors=[
            "model_block_ref_required:variant-wechat_moments-7:private-kind",
            "required_model_blocks_missing:variant-wechat_moments:private-claim-id",
            "model_block_ref_required:variant-wechat_mp-1:private-kind",
            "private-unknown-error:private-body",
        ],
        warnings=[
            "model_title_normalized:wechat_moments:missing_fact_anchor",
            "model_title_normalized:wechat_mp:missing_fact_anchor",
            "model_title_normalized:wechat_moments:private-title",
            "discarded_unsafe_model_editorial:variant-wechat_moments-8:number_ungrounded",
            "discarded_unsafe_model_editorial:variant-wechat_mp-9:number_ungrounded",
            "discarded_unsafe_model_editorial:variant-wechat_moments-8:private-reason",
        ],
    ))
    diagnostics = {}
    assert replace_editorial_surfaces(None, repaired, ["wechat_moments"], diagnostics=diagnostics) is None
    assert diagnostics == {
        "errors": [
            "editorial_repair_rejected:normalization",
            "editorial_repair_error:wechat_moments:model_block_ref_required",
            "editorial_repair_error:wechat_moments:required_model_blocks_missing",
        ],
        "warnings": [
            "model_title_normalized:wechat_moments:missing_fact_anchor",
            "discarded_unsafe_model_editorial:wechat_moments:number_ungrounded",
        ],
    }
    assert "private-" not in json.dumps(diagnostics)


def test_repair_annotation_caps_diagnostics_and_is_independently_fail_closed(monkeypatch):
    from ai_marketing_api.operator_content import OperatorContentOutput
    from ai_marketing_api.operator_review import annotate_editorial_repair_failure

    async def fake(system, body, fallback, *args, **kwargs):
        return _review() if "editorial_review" in body else _publishable_model_candidate(body)

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=_payload()).json()
    assert data["status"] == "content_ready"
    data["critic"].update(
        errors=[f"prior-error-{i}" for i in range(50)],
        warnings=[f"prior-warning-{i}" for i in range(50)],
        checks=[{"code": f"prior-check-{i}", "passed": True, "message": "Prior check"} for i in range(30)],
    )
    data["risk_flags"] = [{"code": f"prior-risk-{i}", "blocking": True, "message": "Prior risk"} for i in range(50)]
    result = annotate_editorial_repair_failure(OperatorContentOutput.model_validate(data), {
        "errors": ["editorial_repair_rejected:surface"],
        "warnings": ["model_title_normalized:wechat_moments:missing_fact_anchor"],
    })
    assert result.status == "quality_insufficient" and result.critic.passed is False
    assert len(result.critic.errors) == len(result.critic.warnings) == len(result.risk_flags) == 50
    assert len(result.critic.checks) == 30
    assert result.critic.errors[0] == "editorial_repair_rejected:surface"
    assert result.critic.warnings[0] == "model_title_normalized:wechat_moments:missing_fact_anchor"
    assert result.critic.checks[-1].code == "editorial_repair_rejected"


def test_safe_normalized_title_alone_does_not_prevent_new_independent_review(monkeypatch):
    stages = []

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            return _failed_review() if len(stages) == 2 else _review()
        repair = "editorial_revision" in body
        stages.append("repair" if repair else "compose")
        candidate = _publishable_model_candidate(body)
        if repair:
            candidate["platform_variants"][0]["title"] = "A calm planning checklist"
        return candidate

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=_payload()).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "review", "repair", "review"]
    assert "model_title_normalized:wechat_moments:missing_fact_anchor" in data["critic"]["warnings"]
