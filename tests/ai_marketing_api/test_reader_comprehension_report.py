"""Reader isolation and durable protocol tests; scripted answers are not model calibration."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from ai_marketing_api import main as api, operator_steps as steps
from ai_marketing_api.operator_content import OperatorContentOutput
from ai_marketing_api.operator_editorial import BoundedModelRun, EditorialReview, RuntimeBudget
from ai_marketing_api.operator_review import (
    _reader_payload, apply_editorial_assessment, assess_editorial, editorial_content_hash,
)
from tests.ai_marketing_api.test_autonomous_editorial import _review
from tests.ai_marketing_api.test_editorial_layout_and_model_version import _layout_normalize
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review
from tests.ai_marketing_api.test_operator_api import _request


def output():
    return _layout_normalize()[2]


def assess(value, response=None, *, required=True, **kwargs):
    calls = []

    async def reader(system, body, fallback, **model_kwargs):
        calls.append((system, deepcopy(body)))
        return deepcopy(response) if response is not None else scripted_review(body)

    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    result, state = asyncio.run(assess_editorial(output=value, brief={"audience": "Tool readers"}, run=run,
        llm=reader, system="legacy-system", model_args=(), scorecard_required=required, **kwargs))
    assert len(calls) == run.trace(blocked=False, review=state, stop_reason=state).model_calls == 1
    return result, state, run, calls


def test_reader_payload_does_not_reveal_author_answers_owner_comments_or_hidden_facts():
    value = output()
    before = value.model_dump(mode="json")
    hidden = {key: "hidden-" + key for key in (
        "objective", "reader_value", "angle", "value_contract", "constraints", "tone", "cta")}
    brief = {"audience": "Tool readers", **hidden}
    first = _reader_payload(value, brief, scorecard_required=True)
    second = _reader_payload(value, {"audience": "Tool readers"}, scorecard_required=True)
    assert first == second
    assert not any(sentinel in json.dumps(first) for sentinel in hidden.values())
    review, state, _, calls = assess(value, owner_revision={"instruction": "hidden-owner-comment"},
        style_context={"thesis": "hidden-style-answer"}, editorial_feedback="hidden-prior-score-5")
    text = json.dumps(calls)
    assert all(token not in text for token in ("hidden-owner-comment", "hidden-style-answer", "hidden-prior-score-5"))
    payload = calls[0][1]["editorial_review"]
    assert not {"public_editorial_brief", "owner_revision", "style_context", "canonical_claims", "editorial_feedback"} & payload.keys()
    assert payload["draft"]["master_content"] == value.master_content
    assert payload["reader_context"] == {"audience": "Tool readers", "genre": value.genre,
        "platforms": ["master", *[v.platform for v in value.platform_variants]]}
    assert review.passed and state == "passed" and value.model_dump(mode="json") == before


@pytest.mark.parametrize("failure", ["coherent", "thesis_focus", "summary", "reader_takeaway", "reasoning"])
def test_unrestatable_or_unfocused_text_blocks_without_rewriting_high_real_grades(failure):
    value = output()
    raw = scripted_review(_reader_payload(value, {}, scorecard_required=True))
    row = raw["narrative_report"]["surfaces"][0]
    row[failure] = False if failure in {"coherent", "thesis_focus"} else ""
    row["issues"] = ["Scripted negative: current text does not connect its point and reasons."]
    raw["narrative_report"].update(passed=True, content_sha256="0" * 64)
    original = deepcopy(raw)
    review, state, _, _ = assess(value, raw)
    assert state == "failed" and not review.passed
    assert review.quality_scorecard.passed is True
    assert all(s.natural_expression == 4 for s in review.quality_scorecard.surfaces)
    assert review.narrative_report.passed is False and review.failed_surfaces == ["master"]
    assert review.narrative_report.content_sha256 == editorial_content_hash(value)
    assert review.narrative_report.surfaces[0].issues == row["issues"]
    assert review.issues == original["issues"] and raw == original
    blocked = apply_editorial_assessment(value, review)
    assert blocked.status == "quality_insufficient"
    assert blocked.master_content == value.master_content and blocked.blocks == value.blocks
    assert blocked.editorial_assessment.narrative_report.passed is False


@pytest.mark.parametrize("fault,code", [
    ("missing", "narrative_report_missing"),
    ("foreign_quote", "narrative_report_span_invalid"),
    ("blank_quote", "narrative_report_span_invalid"),
    ("title_only", "narrative_report_body_span_missing"),
    ("duplicate_surface", "narrative_report_surface_mismatch"),
    ("extra_surface", "narrative_report_surface_mismatch"),
    ("truthy_bool", "narrative_report_schema_invalid"),
])
def test_incomplete_or_unbound_reports_fail_closed_with_safe_private_diagnostics(fault, code):
    value = output()
    raw = scripted_review(_reader_payload(value, {}, scorecard_required=True))
    report = raw["narrative_report"]
    if fault == "missing":
        raw.pop("narrative_report")
    elif fault == "foreign_quote":
        report["surfaces"][0]["supporting_spans"][0]["quote"] = "PRIVATE_UNSEEN_AUTHOR_THESIS"
    elif fault == "blank_quote":
        report["surfaces"][0]["supporting_spans"][0]["quote"] = " "
    elif fault == "title_only":
        report["surfaces"][0]["supporting_spans"] = [{"source": "title", "quote": value.master_title, "role": "thesis"}]
    elif fault == "duplicate_surface":
        report["surfaces"].append(deepcopy(report["surfaces"][0]))
    elif fault == "extra_surface":
        report["surfaces"][0]["surface"] = "unrequested"
    else:
        report["surfaces"][0]["coherent"] = 1
    review, state, run, _ = assess(value, raw)
    assert review is None and state == "not_completed"
    assert run.private_review_receipt["candidate_retained"] is True
    assert run.private_review_receipt["validation"]["code"] == code
    assert "PRIVATE_UNSEEN_AUTHOR_THESIS" not in json.dumps(
        run.trace(blocked=False, review=state, stop_reason=state).model_dump(mode="json"))


def test_valid_report_binds_only_the_actual_text_and_accepts_understanding_without_cta():
    value = output()
    review, state, _, _ = assess(value)
    assert state == "passed" and review.passed and review.narrative_report.passed
    assert review.narrative_report.content_sha256 == editorial_content_hash(value)
    assert all(s.reader_takeaway and s.reasoning for s in review.narrative_report.surfaces)
    changed = value.model_copy(deep=True)
    changed.master_content += " Additional current text."
    changed_review, _, _, _ = assess(changed)
    assert changed_review.narrative_report.content_sha256 != review.narrative_report.content_sha256
    rule = _reader_payload(value, {}, scorecard_required=True)["editorial_review"]["genre_review_rule"]
    assert "Do not force novelty" in rule and "or a CTA" in rule
    assert "Colloquial language is not inherently low quality" in rule


def test_legacy_reviews_without_report_keep_prior_semantics_and_signed_shape():
    raw = _review(True)
    review, state, _, _ = assess(output(), raw, required=False)
    assert state == "passed" and review.passed and review.narrative_report is None
    serialized = review.model_dump(mode="json")
    assert "narrative_report" not in serialized
    assert serialized["quality_scorecard"] is None
    artifact = output().model_dump(mode="json")
    artifact["editorial_assessment"] = serialized
    prior_signature = steps.artifact_signature(artifact)
    parsed = OperatorContentOutput.model_validate(artifact).model_dump(mode="json")
    assert parsed == artifact and steps.artifact_signature(parsed) == prior_signature


def test_actual_signed_rpc_persists_report_and_cached_post_get_do_not_reexecute(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-reader-only-key")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-reader"})
    request, _, _, _, raw = _layout_normalize()
    compose = request.model_dump(mode="json")
    compose["owner_revision"] = {"instruction": "PRIVATE_OWNER_ANSWER", "platform_variants": []}
    compose["editorial_feedback"] = "PRIVATE_OLD_SCORE"

    async def no_verifier(*args, **kwargs):
        pytest.fail("Exact canonical candidate needs no paraphrase model")

    monkeypatch.setattr(api, "_llm_json", no_verifier)
    verified = _request("POST", "/api/v1/operators/industry/editorial-steps/verify_claims", json={"compose": compose, "candidate": raw}).json()
    assert verified["status"] == "artifact"
    calls = []

    async def reader(system, body, fallback, **kwargs):
        calls.append(deepcopy(body))
        assert "PRIVATE_OWNER_ANSWER" not in json.dumps(body) and "PRIVATE_OLD_SCORE" not in json.dumps(body)
        return scripted_review(body)

    monkeypatch.setattr(api, "_llm_json", reader)
    payload = {"compose": compose, "candidate": verified["artifact"], "candidate_signature": verified["artifact_signature"],
               "budget_reservation_id": "synthetic-reader-comprehension"}
    original = deepcopy(payload)
    url = "/api/v1/operators/industry/editorial-steps/review_editorial"
    first = _request("POST", url, json=payload)
    assert first.status_code == 200 and first.json()["status"] == "artifact"
    again = _request("POST", url, json=payload)
    assert again.json() == first.json() and len(calls) == 1
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    receipt = _request("GET", url + "/receipts/synthetic-reader-comprehension", headers={"X-Hermes-Request-SHA256": fingerprint})
    assert receipt.status_code == 200 and receipt.json()["status"] == "completed"
    assert receipt.json()["response"] == first.json() and len(calls) == 1 and payload == original
    report = first.json()["artifact"]["editorial_assessment"]["narrative_report"]
    assert report["passed"] and report["content_sha256"] == first.json()["artifact"]["quality_scorecard"]["content_sha256"]
