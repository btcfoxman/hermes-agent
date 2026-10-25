"""Real edge contracts with scripted readers; these are not model-quality grades."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import main as api, operator_steps as steps
from ai_marketing_api.operator_content import OperatorComposeRequest, content_request_payload
from ai_marketing_api.operator_editorial import BoundedModelRun, EditorialInstruction, RuntimeBudget
from ai_marketing_api.operator_review import _reader_payload, apply_editorial_assessment, assess_editorial
from ai_marketing_api.operator_runtime import ContextAuthorizationError, OperatorRegistry
from tests.ai_marketing_api.test_editorial_layout_and_model_version import _layout_normalize
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose, ANALYSIS
from tests.ai_marketing_api.test_operator_api import _request


def chat_instruction():
    text = "  资讯要交代真实来源，分析要写出自己的结论。不要编造亲测。\n"
    return {"schema_version": "operator.editorial_instruction.v1", "instruction_id": "synthetic-chat-instruction",
        "instruction": text, "source": "user_chat",
        "source_user_message_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "references": [{"url": "https://example.com/style-only", "purpose": "style_only"}],
        "target": {"review_id": "synthetic-review", "review_version": 2, "campaign_id": "synthetic-campaign",
            "package_id": "synthetic-package", "package_version": 7, "role_id": "industry",
            "content_hash": "a" * 64, "evidence_hash": "b" * 64, "policy_hash": "c" * 64},
        "owner_decision": False, "publication_authorized": False}


def writer_request():
    raw = _compose("industry_analysis", "wechat_mp").model_dump(mode="json")
    raw["editorial_instruction"] = chat_instruction()
    return OperatorComposeRequest.model_validate(raw)


def reading_output(*, genre="industry_analysis"):
    # Entirely fictional but naturally phrased; no real product or new source grant.
    value = _layout_normalize()[2]
    body = ("青鸟书柜公告把预约和取书分开，借阅者需要先确认取书时段。"
        "我会把预约成功当成排队资格，而不承诺当天拿到书。时段确认之前，借阅者还无法安排出行，"
        "把这两个状态合成一句成功通知会把等待成本留给借阅者。"
        "需要赶当天使用的人，应先拿到时段再出门；如果现场本来就有可直接取走的书，这条建议就不适用。")
    return value.model_copy(update={"genre": genre, "master_title": "青鸟书柜预约成功后，先确认取书时段",
        "master_content": body, "platform_variants": [v.model_copy(update={
            "title": "青鸟书柜的预约通知还缺什么", "body": body + "这条建议只针对需要预约的书。"})
            for v in value.platform_variants]})


def assessment_for(value):
    body = _reader_payload(value, {}, scorecard_required=True)
    raw = scripted_review(body)
    raw["_model"] = "synthetic-independent-reader"
    for row in raw["narrative_report"]["surfaces"]:
        row["summary"] = "预约资格和确定取书时段需要区分。"
        row["reader_takeaway"] = "赶时间的借阅者先确认时段再安排出行。"
        row["reasoning"] = "时段尚未确定时，预约资格不能支持当天取书的承诺。"
        if "analysis" in row:
            row["analysis"] = {"news_value": "公告将预约与取书分开。",
                "author_conclusion": "预约成功只能当作排队资格，出门前应先确认时段。",
                "causal_explanation": "未确定取书时段就无法安排出行，混淆状态把等待成本转给借阅者。",
                "affected_reader": "需要当天用书的预约借阅者。",
                "decision_consequence": "拿到时段再出门，而不是凭预约成功通知出门。",
                "limits": "可现场直接取走的书不受这条预约建议约束。",
                "source_restatement_only": False, "supported": True}
    return raw


def assess(value, raw, **kwargs):
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    seen = []

    async def reader(system, payload, fallback, **options):
        seen.append(deepcopy(payload))
        return deepcopy(raw)

    review, state = asyncio.run(assess_editorial(output=value, brief=kwargs.pop("brief", {}),
        run=run, llm=reader, system="unused", model_args=(), scorecard_required=True, **kwargs))
    assert len(seen) == 1
    return review, state, run, seen[0]


def test_industry_writer_keeps_exact_chat_intent_separate_from_factual_and_owner_authority():
    request = writer_request()
    before = request.model_dump(mode="json")
    contexts = OperatorRegistry().authorize("industry", request.authorized_context, request.as_of)
    payload = content_request_payload("industry", request, contexts)
    assert payload["editorial_instruction"] == before["editorial_instruction"]
    assert "owner_revision" not in payload
    assert request.model_dump(mode="json") == before
    assert payload["claims"] == before["claims"]
    assert payload["editorial_instruction"]["references"][0]["purpose"] == "style_only"
    assert payload["industry_editorial_method"]["version"] == payload["response_contract"]["editorial_shape"]["method_version"]
    method = " ".join(payload["industry_editorial_method"]["rules"])
    assert "author conclusion" in method and "stop for more research" in method
    assert "date" in method and "attribution" in method and "hands-on test" in method
    generated = json.dumps({key: payload[key] for key in (
        "publication_brief", "platform_editorial_briefs", "response_contract", "industry_narration_contract")})
    assert "with sparse evidence" not in generated.lower()
    assert "when evidence is sparse" not in generated.lower()
    assert "selection condition or verification question" not in generated.lower()


def test_public_brief_scope_cannot_reintroduce_a_sparse_evidence_analysis_fallback():
    raw = writer_request().model_dump(mode="json")
    raw["public_editorial_brief"] = {"role_id": "industry", "objective": "Explain the actual industry significance.",
        "audience": "Operators making this decision", "channels": raw["channels"]}
    request = OperatorComposeRequest.model_validate(raw)
    contexts = OperatorRegistry().authorize("industry", request.authorized_context, request.as_of)
    payload = content_request_payload("industry", request, contexts)
    scope = " ".join(payload["evidence_scope_contract"])
    assert "Insufficient evidence requires research or revision" in scope
    assert "selection condition or verification question" not in scope
    assert "With sparse" not in scope and "When evidence is sparse" not in scope


@pytest.mark.parametrize("fault", ["message_hash", "extra_top", "extra_target", "extra_reference",
    "source", "role", "owner_true", "owner_zero", "publication_true", "publication_zero", "reference_authority"])
def test_chat_instruction_cannot_expand_identity_or_authority(fault):
    raw = chat_instruction()
    if fault == "message_hash": raw["instruction"] += "Changed."
    elif fault == "extra_top": raw["approval"] = True
    elif fault == "extra_target": raw["target"]["owner_approved"] = True
    elif fault == "extra_reference": raw["references"][0]["fact"] = "unsupported"
    elif fault == "source": raw["source"] = "owner_review"
    elif fault == "owner_true": raw["owner_decision"] = True
    elif fault == "owner_zero": raw["owner_decision"] = 0
    elif fault == "publication_true": raw["publication_authorized"] = True
    elif fault == "publication_zero": raw["publication_authorized"] = 0
    elif fault == "reference_authority": raw["references"][0]["purpose"] = "facts"
    else: raw["target"]["role_id"] = "commercial"
    if fault == "role":
        request = writer_request().model_copy(update={"editorial_instruction": EditorialInstruction.model_validate(raw)})
        contexts = OperatorRegistry().authorize("industry", request.authorized_context, request.as_of)
        with pytest.raises(ContextAuthorizationError): content_request_payload("industry", request, contexts)
    else:
        with pytest.raises(ValidationError): EditorialInstruction.model_validate(raw)


def test_real_staged_writer_receives_chat_instruction_without_owner_revision_or_new_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-method-key")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-writer"})
    compose = writer_request().model_dump(mode="json")
    calls = []

    async def writer(system, payload, fallback, **kwargs):
        calls.append(payload)
        assert payload["editorial_instruction"] == compose["editorial_instruction"]
        assert "owner_revision" not in payload
        assert all(c["record_id"] != "synthetic-chat-instruction" for c in payload["authorized_context"])
        return {"_model": "synthetic-writer", "master_title": compose["approved_proposal"]["title"],
            "blocks": [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"] if b["required"]]
                + [{"kind": "opinion", "text": "假设资料需要分类，先约定问题范围，再保留需要人工展开的判断。", "evidence_ids": []}],
            "platform_variants": []}

    monkeypatch.setattr(api, "_llm_json", writer)
    response = _request("POST", "/api/v1/operators/industry/editorial-steps/draft_master", json={"compose": compose}).json()
    assert response["status"] == "artifact" and len(calls) == 1
    assert response["publishable"] is False
    assert response["artifact"]["quality_scorecard"] is None


def test_real_legacy_compose_repair_keeps_current_chat_and_industry_method(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-repair-key")
    compose = writer_request().model_dump(mode="json")
    compose["runtime_budget"] = {"max_model_calls": 4, "max_surface_revisions": 1}
    calls = []

    async def writer_or_reader(system, payload, fallback, *args, **kwargs):
        calls.append(deepcopy(payload))
        if "editorial_review" in payload:
            return {**scripted_review(payload), "_model": "synthetic-reader"}
        refs = [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"] if b["required"]]
        blocks = refs + deepcopy(ANALYSIS)
        candidate = {"_model": "synthetic-writer", "master_title": compose["approved_proposal"]["title"],
            "blocks": blocks, "platform_variants": [{"platform": "wechat_mp", "title": "示例接口的判断任务与创作边界", "blocks": deepcopy(blocks)}]}
        if len(calls) == 1:
            candidate["platform_variants"] = []  # Force the real focused missing-surface repair path.
        return candidate

    monkeypatch.setattr(api, "_llm_json", writer_or_reader)
    response = _request("POST", "/api/v1/operators/industry/compose", json=compose)
    assert response.status_code == 200
    retry = next(payload for payload in calls if "quality_retry" in payload)
    assert retry["editorial_instruction"] == compose["editorial_instruction"]
    assert retry["owner_revision"] is None
    assert retry["industry_editorial_method"]["version"]
    assert "industry_editorial_method" in retry["quality_retry"]["instruction"]
    assert "fallback" not in json.dumps(retry["quality_retry"]).lower()
    assert retry["canonical_block_registry"] == calls[0]["canonical_block_registry"]
    assert response.json()["requires_human_review"] is True


@pytest.mark.parametrize("fault", ["missing_conclusion", "unsupported", "source_only", "missing_impact", "title_limit"])
def test_high_scores_cannot_hide_shallow_or_incomplete_industry_analysis(fault):
    value = reading_output()
    raw = assessment_for(value)
    row = raw["narrative_report"]["surfaces"][0]
    if fault == "missing_conclusion": row["analysis"]["author_conclusion"] = " "
    elif fault == "unsupported": row["analysis"]["supported"] = False
    elif fault == "source_only": row["analysis"]["source_restatement_only"] = True
    elif fault == "missing_impact": row["supporting_spans"] = [s for s in row["supporting_spans"] if s["role"] != "impact"]
    else:
        next(s for s in row["supporting_spans"] if s["role"] == "limit").update(
            source="title", quote=value.master_title)
    before = deepcopy(raw)
    review, state, run, _ = assess(value, raw)
    assert state == "failed" and review is not None and not review.passed
    assert review.quality_scorecard.passed  # Preserve the actual high numeric grades.
    assert not review.narrative_report.passed and "master" in review.failed_surfaces
    assert review.narrative_report.surfaces[0].analysis.model_dump() == row["analysis"]
    assert raw == before and run.private_review_receipt["candidate"]["narrative_report"] == raw["narrative_report"]
    assert apply_editorial_assessment(value, review).status == "quality_insufficient"


@pytest.mark.parametrize("fault", ["analysis_missing", "analysis_extra", "bool_string", "bool_integer", "cross_surface", "legacy_schema"])
def test_industry_reader_schema_and_current_surface_binding_fail_closed(fault):
    value = reading_output()
    raw = assessment_for(value)
    row = raw["narrative_report"]["surfaces"][0]
    if fault == "analysis_missing": row.pop("analysis")
    elif fault == "analysis_extra": row["analysis"]["writer_intention"] = "Do not accept"
    elif fault == "bool_string": row["analysis"]["supported"] = "true"
    elif fault == "bool_integer": row["analysis"]["supported"] = 1
    elif fault == "legacy_schema": raw["narrative_report"]["schema_version"] = "narrative_report.v1"
    else:
        bank = _reader_payload(value, {}, scorecard_required=True)["editorial_review"]["source_span_bank"]
        row["supporting_spans"][0] = {"source": "body", "span_id": bank[value.platform_variants[0].platform]["body"][0]["span_id"], "role": "news"}
    review, state, run, _ = assess(value, raw)
    assert review is None and state == "not_completed"
    assert run.private_review_receipt["candidate"]["narrative_report"] == raw["narrative_report"]


def test_natural_bounded_analysis_can_pass_with_one_body_span_for_multiple_functions():
    value = reading_output()
    raw = assessment_for(value)
    review, state, _, _ = assess(value, raw)
    assert state == "passed" and review.passed
    assert review.narrative_report.schema_version == "narrative_report.v2"
    assert len({s.quote for s in review.narrative_report.surfaces[0].supporting_spans}) == 1
    assert value.master_content == apply_editorial_assessment(value, review).master_content


def test_blind_reader_receives_generic_method_but_no_chat_instruction_or_intended_answer():
    value = reading_output()
    hidden = {key: "PRIVATE_SENTINEL_" + key for key in (
        "editorial_instruction", "value_contract", "owner_revision", "prior_review", "style_references")}
    raw = assessment_for(value)
    review, state, _, payload = assess(value, raw, brief={"audience": "借阅者", **hidden},
        owner_revision={"instruction": hidden["owner_revision"]}, style_context=hidden,
        editorial_feedback=hidden["prior_review"])
    assert state == "passed" and review.passed
    encoded = json.dumps(payload)
    assert all(text not in encoded for text in hidden.values())
    assert payload["editorial_review"]["industry_editorial_method"]["version"]
    assert payload["editorial_review"]["draft"]["master_content"] == value.master_content


def test_non_industry_v1_shape_remains_compatible_and_cannot_accept_v2_report():
    value = reading_output(genre="company_data_brief")
    raw = assessment_for(value)
    review, state, _, payload = assess(value, raw)
    assert state == "passed" and review.passed
    assert "industry_editorial_method" not in payload["editorial_review"]
    assert review.narrative_report.schema_version == "narrative_report.v1"
    assert all("analysis" not in row for row in review.narrative_report.model_dump()["surfaces"])
    raw["narrative_report"]["schema_version"] = "narrative_report.v2"
    assert assess(value, raw)[1] == "not_completed"
