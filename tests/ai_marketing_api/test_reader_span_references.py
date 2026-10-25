"""Exact source references; scripted selections are not model quality verdicts."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_marketing_api import main as api, operator_steps as steps
from ai_marketing_api.operator_content import OperatorContentOutput, SUPPORTED_CHANNEL_FORMATS
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_review import (
    _bind_narrative_report, _narrative_span_bank, _reader_payload,
    apply_editorial_assessment, assess_editorial, editorial_content_hash,
)
from tests.ai_marketing_api.test_editorial_layout_and_model_version import _layout_normalize
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review, scripted_analysis
from tests.ai_marketing_api.test_operator_api import _request


def output():
    return _layout_normalize()[2]


def reference_review(body):
    raw = scripted_review(body)
    bank = body["editorial_review"]["source_span_bank"]
    for surface in raw["narrative_report"]["surfaces"]:
        item = next(row for row in bank[surface["surface"]]["body"] if row["text"].strip())
        roles = ("news", "thesis", "reason", "impact", "limit") if "analysis" in surface else ("reason",)
        surface["supporting_spans"] = [{"source": "body", "span_id": item["span_id"], "role": role} for role in roles]
    return raw


def assess(value, raw):
    calls = []
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))

    async def reader(*args, **kwargs):
        calls.append(deepcopy(args[1]))
        return deepcopy(raw)

    review, state = asyncio.run(assess_editorial(output=value, brief={}, run=run,
        llm=reader, system="offline", model_args=(), scorecard_required=True))
    assert len(calls) == 1
    return review, state, run


def test_bank_is_stable_complete_bounded_exact_and_scoped_to_current_text():
    text = "原文，保留中文标点？\n\n" + "连续原文。" * 250 + "\n末段；结束。"
    value = SimpleNamespace(master_title="同文标题，原样？", master_content=text,
        platform_variants=[SimpleNamespace(platform="wechat_mp", title="同文标题，原样？", body=text)])
    bank = _narrative_span_bank(value)
    assert bank == _narrative_span_bank(value)
    ids = []
    for surface, fields in bank.items():
        for source, rows in fields.items():
            expected = value.master_title if source == "title" else text
            assert "".join(row["text"] for row in rows) == expected
            assert all(1 <= len(row["text"]) <= 500 and row["text"] in expected for row in rows)
            ids.extend(row["span_id"] for row in rows)
    assert len(ids) == len(set(ids))  # Same text cannot alias another surface/source.
    value.master_content += "真实当前文本变化"
    changed = _narrative_span_bank(value)
    assert not ({s["span_id"] for s in changed["master"]["body"]}
                & {s["span_id"] for s in bank["master"]["body"]})
    assert changed["wechat_mp"] == bank["wechat_mp"]


def test_payload_bank_uses_only_visible_draft_and_never_supplies_hidden_answers():
    value = output()
    hidden = {key: "PRIVATE_ANSWER_" + key for key in
              ("angle", "value_contract", "reader_value", "objective", "owner_revision", "canonical_claims")}
    body = _reader_payload(value, {"audience": "Readers", **hidden}, scorecard_required=True)
    assert body == _reader_payload(value, {"audience": "Readers"}, scorecard_required=True)
    bank = body["editorial_review"]["source_span_bank"]
    assert "".join(row["text"] for row in bank["master"]["body"]) == value.master_content
    assert all(v not in json.dumps(body) for v in hidden.values())
    fields = body["response_contract"]["narrative_report"]["surfaces"][0]["supporting_spans"][0]
    assert set(fields) == {"source", "span_id", "role"}


def test_ref_selection_binds_exact_quotes_while_private_receipt_keeps_original_ref_form():
    value = output()
    raw = reference_review(_reader_payload(value, {}, scorecard_required=True))
    original = deepcopy(raw)
    review, state, run = assess(value, raw)
    assert state == "passed" and review.passed and raw == original
    assert review.narrative_report.content_sha256 == editorial_content_hash(value)
    receipt = run.private_review_receipt
    expected = {key: val for key, val in raw.items() if not key.startswith("_")}
    assert receipt["candidate"] == expected
    assert receipt["candidate_sha256"] == hashlib.sha256(json.dumps(expected, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert receipt["candidate"]["narrative_report"]["surfaces"][0]["supporting_spans"][0].get("span_id")
    for surface in review.narrative_report.surfaces:
        serialized = surface.supporting_spans[0].model_dump(mode="json")
        assert set(serialized) == {"source", "quote", "role"}
        sources = {"master": value.master_content, **{v.platform: v.body for v in value.platform_variants}}
        assert serialized["quote"] in sources[surface.surface]


@pytest.mark.parametrize("fault", ["other_surface", "other_source", "unknown_id", "stale_id",
                                  "extra_field", "mixed_quote", "invalid_id_type", "no_body", "blank_body"])
def test_refs_cannot_expand_source_or_bypass_strict_report_validation(fault):
    value = output()
    raw = reference_review(_reader_payload(value, {}, scorecard_required=True))
    bank = _narrative_span_bank(value)
    row = raw["narrative_report"]["surfaces"][0]
    span = row["supporting_spans"][0]
    if fault == "other_surface":
        span["span_id"] = bank[value.platform_variants[0].platform]["body"][0]["span_id"]
    elif fault == "other_source":
        span["span_id"] = bank["master"]["title"][0]["span_id"]
    elif fault == "unknown_id":
        span["span_id"] = "span_unknown"
    elif fault == "stale_id":
        value.master_content += "Changed current text."
    elif fault == "extra_field":
        span["extra"] = "not allowed"
    elif fault == "mixed_quote":
        span["quote"] = bank["master"]["body"][0]["text"]
    elif fault == "invalid_id_type":
        span["span_id"] = 1
    elif fault == "no_body":
        row["supporting_spans"] = [{"source": "title", "span_id": bank["master"]["title"][0]["span_id"], "role": "thesis"}]
    else:
        value.master_content = " \n"
        span["span_id"] = _narrative_span_bank(value)["master"]["body"][0]["span_id"]
    original = deepcopy(raw)
    review, state, run = assess(value, raw)
    assert review is None and state == "not_completed" and raw == original
    assert run.private_review_receipt["candidate"] == {k: v for k, v in raw.items() if not k.startswith("_")}


@pytest.mark.parametrize("fault", ["low_grade", "major", "false_report", "false_editor"])
def test_ref_protocol_never_waives_quality_or_understanding_rejections(fault):
    value = output()
    raw = reference_review(_reader_payload(value, {}, scorecard_required=True))
    grade = raw["quality_scorecard"]["surfaces"][0]
    raw["issues"] = ["Preserve the reviewer observation."]
    if fault == "low_grade":
        grade["analytical_depth"] = 3
    elif fault == "major":
        grade["major_issues"] = ["Material current-text problem."]
    elif fault == "false_report":
        raw["narrative_report"]["surfaces"][0]["coherent"] = False
    else:
        raw["useful_to_reader"] = False
    review, state, _ = assess(value, raw)
    assert state == "failed" and not review.passed
    assert review.quality_scorecard.surfaces[0].analytical_depth == grade["analytical_depth"]
    assert review.quality_scorecard.surfaces[0].major_issues == grade["major_issues"]
    assert raw["issues"][0] in review.issues
    blocked = apply_editorial_assessment(value, review)
    assert blocked.status == "quality_insufficient" and blocked.master_content == value.master_content


@pytest.fixture
def synthetic_punctuation_case():
    fixture = json.loads((Path(__file__).parent / "fixtures/editorial_synthetic_quote_punctuation.v1.json").read_text(encoding="utf-8"))
    draft = fixture["draft"]
    value = OperatorContentOutput(role_id="industry", genre="industry_analysis", status="content_ready",
        profile_version="offline-synthetic-quote-regression", system_prompt_sha256="0" * 64,
        master_title=draft["master_title"], master_content=draft["master_content"],
        platform_variants=[{**v, "format": SUPPORTED_CHANNEL_FORMATS[v["platform"]]} for v in draft["platform_variants"]],
        critic={"passed": True, "checks": [], "errors": [], "warnings": []})
    report = {"schema_version": "narrative_report.v2", "surfaces": [{
        "surface": surface, "summary": "Synthetic protocol selection, not a new model summary.",
        "reader_takeaway": "Synthetic structure probe only.", "reasoning": "Test exact evidence binding.",
        "coherent": True, "thesis_focus": True, "issues": [], "supporting_spans": deepcopy(spans),
        "analysis": scripted_analysis()}
        for surface, spans in fixture["quote_spans"].items()]}
    raw = {**deepcopy(fixture["assessment_fields"]), "_model": "offline-synthetic-ref-probe",
           "narrative_report": report}
    return fixture, value, raw


def test_synthetic_same_words_with_changed_punctuation_still_reject(synthetic_punctuation_case):
    fixture, value, raw = synthetic_punctuation_case
    sources = {"master": {"title": value.master_title, "body": value.master_content}, **{
        v.platform: {"title": v.title, "body": v.body} for v in value.platform_variants}}
    spans = [(surface, span) for surface, rows in fixture["quote_spans"].items() for span in rows]
    changed = [(surface, span) for surface, span in spans
               if span["quote"] not in sources[surface][span["source"]]]
    assert changed
    punctuation = str.maketrans({"，": ",", "；": ";", "？": "?"})
    assert all(span["quote"].translate(punctuation)
               in sources[surface][span["source"]].translate(punctuation) for surface, span in changed)
    before = deepcopy(raw)
    review, state, run = assess(value, raw)
    assert review is None and state == "not_completed" and raw == before
    assert run.private_review_receipt["validation"]["code"] == "narrative_report_span_invalid"


def test_synthetic_ref_form_keeps_original_scripted_depth3_failure(synthetic_punctuation_case):
    _, value, raw = synthetic_punctuation_case
    bank = _narrative_span_bank(value)
    original_card = deepcopy(raw["quality_scorecard"])
    for surface in raw["narrative_report"]["surfaces"]:
        surface["supporting_spans"] = [{"source": "body", "span_id": bank[surface["surface"]]["body"][0]["span_id"], "role": role}
            for role in ("news", "thesis", "reason", "impact", "limit")]
    review, state, _ = assess(value, raw)
    assert state == "failed" and not review.passed and review.narrative_report.passed
    assert not review.quality_scorecard.passed and "wechat_moments" in review.failed_surfaces
    assert [s.model_dump(mode="json") for s in review.quality_scorecard.surfaces] == original_card["surfaces"]
    assert next(s for s in review.quality_scorecard.surfaces if s.surface == "wechat_moments").analytical_depth == 3


def test_generic_json_parser_preserves_punctuation_without_normalization():
    expected = {"quote": "先理句意，再调格式；保留原文？"}
    assert api._json_from_response(json.dumps(expected, ensure_ascii=False)) == expected


def test_ref_reader_signed_rpc_and_durable_replay_keep_bound_quote_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-ref-key")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-reader"})
    request, _, _, _, raw = _layout_normalize()
    compose = request.model_dump(mode="json")

    async def no_verifier(*args, **kwargs):
        pytest.fail("Exact canonical candidate needs no verifier model")

    monkeypatch.setattr(api, "_llm_json", no_verifier)
    base = "/api/v1/operators/industry/editorial-steps/"
    verified = _request("POST", base + "verify_claims", json={"compose": compose, "candidate": raw}).json()
    assert verified["status"] == "artifact"
    calls = []

    async def reader(system, body, fallback, **kwargs):
        calls.append(deepcopy(body))
        return reference_review(body)

    monkeypatch.setattr(api, "_llm_json", reader)
    payload = {"compose": compose, "candidate": verified["artifact"],
        "candidate_signature": verified["artifact_signature"], "budget_reservation_id": "synthetic-ref-receipt"}
    original = deepcopy(payload)
    first = _request("POST", base + "review_editorial", json=payload).json()
    assert first["status"] == "artifact" and first["artifact"]["editorial_assessment"]["narrative_report"]["passed"]
    report = first["artifact"]["editorial_assessment"]["narrative_report"]
    assert "quote" in report["surfaces"][0]["supporting_spans"][0]
    assert "span_id" in first["private_review_receipt"]["candidate"]["narrative_report"]["surfaces"][0]["supporting_spans"][0]
    assert steps.artifact_signature(first["artifact"]) == first["artifact_signature"]
    again = _request("POST", base + "review_editorial", json=payload).json()
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    cached = _request("GET", base + "review_editorial/receipts/synthetic-ref-receipt",
        headers={"X-Hermes-Request-SHA256": fingerprint}).json()
    assert first == again == cached["response"] and len(calls) == 1 and payload == original
