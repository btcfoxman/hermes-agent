from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import ai_marketing_api.main as api
import ai_marketing_api.operator_steps as steps
from ai_marketing_api.operator_editorial import BoundedModelRun, QualityScorecard, RuntimeBudget
from ai_marketing_api.operator_review import (
    MAX_PRIVATE_REVIEW_RECEIPT_BYTES, assess_editorial,
)
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import fact_first, payload, review
from tests.ai_marketing_api.test_operator_api import _request


def draft():
    return SimpleNamespace(blocks=[], platform_variants=[], master_title="Offline editorial case",
        master_content="A source-grounded public draft.")


def reviewer_result():
    return {"_model": "offline-editor", "relevant_to_brief": True, "useful_to_reader": True,
        "clear_and_specific": True, "platform_fit": True, "issues": [], "failed_surfaces": [],
        "quality_scorecard": {"schema_version": "quality_scorecard.v1", "surfaces": [{
            "surface": "master", "reader_relevance": 4, "information_gain": 4,
            "analytical_depth": 4, "natural_expression": 4, "platform_fit": 4,
            "major_issues": [], "issues": []}]}}


def assess(candidate, *, scorecard_required=True):
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    seen = []
    async def fake(system, body, fallback, *args, **kwargs):
        seen.append(body)
        return copy.deepcopy(candidate)
    result, state = asyncio.run(assess_editorial(output=draft(), brief={}, run=run, llm=fake,
        system="offline", model_args=(), scorecard_required=scorecard_required))
    return result, state, run, seen


def test_contract_contains_only_accepted_fields_and_actual_expected_surfaces():
    result, state, run, seen = assess(reviewer_result())
    body = seen[0]
    assert result.passed and state == "passed"
    assert "rubric" not in body["response_contract"]["quality_scorecard"]
    assert body["editorial_review"]["scorecard_rubric"]
    assert body["editorial_review"]["output_rules"]["expected_surfaces"] == ["master"]
    assert set(body["response_contract"]["quality_scorecard"]) <= set(QualityScorecard.model_fields)
    assert run.private_review_receipt["validation"]["code"] == "review_validated"


@pytest.mark.parametrize("failure,stage,code,issue_type", [
    ("rubric", "scorecard", "scorecard_schema_invalid", "extra_forbidden"),
    ("surface", "surface_coverage", "scorecard_surface_mismatch", "scorecard_surface_mismatch"),
    ("boolean", "scorecard", "scorecard_schema_invalid", "int_type"),
    ("string", "scorecard", "scorecard_schema_invalid", "int_type"),
    ("missing", "scorecard", "scorecard_schema_invalid", "missing"),
    ("unknown_private_key", "review_schema", "review_schema_invalid", "extra_forbidden"),
    ("no_card", "scorecard", "scorecard_missing", "scorecard_missing"),
])
def test_invalid_paid_assessment_is_retained_and_diagnosable_without_approving(failure, stage, code, issue_type):
    candidate = reviewer_result()
    row = candidate["quality_scorecard"]["surfaces"][0]
    if failure == "rubric":
        candidate["quality_scorecard"]["rubric"] = "Private retained reviewer wording."
    elif failure == "surface":
        row["surface"] = "unrequested_platform"
    elif failure == "boolean":
        row["analytical_depth"] = True
    elif failure == "string":
        row["analytical_depth"] = "4"
    elif failure == "missing":
        row.pop("major_issues")
    elif failure == "unknown_private_key":
        candidate["private-source-secret-field-name"] = "Private retained reviewer wording."
    else:
        candidate.pop("quality_scorecard")
    result, state, run, _ = assess(candidate)
    assert result is None and state == "not_completed" and len(run.attempts) == 1
    receipt = run.private_review_receipt
    expected = {key: value for key, value in candidate.items() if not key.startswith("_")}
    assert receipt["candidate_retained"] and receipt["candidate"] == expected
    encoded = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert receipt["candidate_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert receipt["candidate_bytes"] == len(encoded)
    diagnostic = receipt["validation"]
    assert (diagnostic["stage"], diagnostic["code"]) == (stage, code)
    assert diagnostic["errors"][0]["type"] == issue_type
    public = run.trace(blocked=False, review=state, stop_reason="quality_insufficient").model_dump(mode="json")
    assert public["editorial_diagnostics"][0] == diagnostic
    assert "Private retained" not in json.dumps(public)
    assert "private-source-secret" not in json.dumps(public)
    assert "candidate" not in public and "private_review_receipt" not in public


def test_oversized_review_retains_hash_and_size_without_partial_replay_or_public_text():
    candidate = reviewer_result()
    candidate["unsupported_extra"] = "private-review-marker" * MAX_PRIVATE_REVIEW_RECEIPT_BYTES
    result, state, run, _ = assess(candidate)
    assert result is None and state == "not_completed"
    receipt = run.private_review_receipt
    assert not receipt["candidate_retained"] and receipt["candidate"] is None
    assert receipt["capture_error"] == "review_candidate_too_large"
    assert receipt["candidate_bytes"] > MAX_PRIVATE_REVIEW_RECEIPT_BYTES
    assert len(receipt["candidate_sha256"]) == 64
    assert len(json.dumps(receipt, ensure_ascii=False).encode()) <= MAX_PRIVATE_REVIEW_RECEIPT_BYTES
    assert "private-review-marker" not in json.dumps(run.trace(blocked=False, review=state, stop_reason="blocked").model_dump(mode="json"))


def test_legacy_review_without_scorecard_still_has_only_safe_public_diagnostics():
    candidate = reviewer_result()
    candidate.pop("quality_scorecard")
    result, state, run, _ = assess(candidate, scorecard_required=False)
    assert state == "passed" and result.passed and result.quality_scorecard is None
    public = run.trace(blocked=False, review=state, stop_reason="completed").model_dump(mode="json")
    assert "private_review_receipt" not in public and "candidate" not in public


def test_staged_http_returns_private_receipt_outside_signed_artifact_and_legacy_hides_it(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": stage})
    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            candidate = review(body)
            candidate["quality_scorecard"]["rubric"] = "private-paid-review-marker"
            return candidate
        if "claim_verification" in body:
            return {"_model": "offline-verifier", "reviews": []}
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", fake)
    value = payload()
    base = "/api/v1/operators/industry/editorial-steps/"
    master = _request("POST", base + "draft_master", json={"compose": value}).json()
    adapted = _request("POST", base + "adapt_platform", json={"compose": value, "surface": "wechat_mp",
        "candidate": master["artifact"], "candidate_signature": master["artifact_signature"]}).json()
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    verified = _request("POST", base + "verify_claims", json={"compose": value, "candidate": merged}).json()
    result = _request("POST", base + "review_editorial", json={"compose": value,
        "candidate": verified["artifact"], "candidate_signature": verified["artifact_signature"]}).json()
    assert result["status"] == "blocked" and not result["publishable"]
    assert result["artifact"]["quality_scorecard"] is None
    assert result["private_review_receipt"]["candidate"]["quality_scorecard"]["rubric"] == "private-paid-review-marker"
    assert steps.artifact_signature(result["artifact"]) == result["artifact_signature"]
    assert "private-paid-review-marker" not in json.dumps(result["artifact"])
    assert "private-paid-review-marker" not in json.dumps(result["generation_trace"])
    legacy = _request("POST", "/api/v1/operators/industry/compose", json=value).json()
    assert "private_review_receipt" not in legacy
    assert "private-paid-review-marker" not in json.dumps(legacy)
