"""Blind input and rejection contracts; scripted verdicts do not test model semantics."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from ai_marketing_api.operator_content import OperatorContentOutput, SUPPORTED_CHANNEL_FORMATS
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_review import (
    EDITOR_SYSTEM_PROMPT, _reader_payload, apply_editorial_assessment, assess_editorial,
)
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import scripted_analysis


# Entirely fictional texts exercise transport without copying a live draft or verdict.
TEXTS = [
    ("先交回书卡，再预约", "青鸟书柜要求持书卡预约；预约成功后再交回书卡。"),
    ("红杉灯熄灭时可以开盖", "红杉灯熄灭时不能开盖；亮起后才能开盖。"),
    ("填写登记表会让纸盒变干", "只有纸盒已经干燥才填写登记表；登记不改变干燥状态。"),
    ("借书只剩两种选择", "青鸟书柜接受现场借阅和邮寄借阅。要么现场借阅，要么放弃借阅。"),
    ("多留一张纸必然更快", "琥珀纸盒允许保留草图。保留草图必然让每次校对更快。"),
    ("没有带尺时先做什么", "假设桌上只有标线纸且只需判断物件是否越线。标线纸能显示越线，"
     "却不提供尺寸读数；在这个条件下可先判断越线，若要知道尺寸仍需尺子。"),
]


def fictional_output(title, body):
    return OperatorContentOutput(
        role_id="industry", genre="industry_analysis", status="content_ready",
        profile_version="synthetic-reasoning-contract", system_prompt_sha256="0" * 64,
        master_title=title, master_content=body,
        platform_variants=[{"platform": "wechat_mp", "format": SUPPORTED_CHANNEL_FORMATS["wechat_mp"],
                            "title": "另一篇虚构文字", "body": "蓝色夹子只用于标记纸页，不改变纸页顺序。"}],
        critic={"passed": True, "checks": [], "errors": [], "warnings": []},
    )


def scripted_assessment(payload):
    """Schema/transport fixture only; no claim to judge the fictional argument."""
    bank = payload["editorial_review"]["source_span_bank"]
    dimensions = ("reader_relevance", "information_gain", "analytical_depth",
                  "natural_expression", "platform_fit")
    return {
        "_model": "offline-scripted-contract",
        "relevant_to_brief": True, "useful_to_reader": True,
        "clear_and_specific": True, "platform_fit": True, "issues": [], "failed_surfaces": [],
        "quality_scorecard": {"schema_version": "quality_scorecard.v1", "surfaces": [
            {"surface": surface, **{key: 4 for key in dimensions}, "major_issues": [], "issues": []}
            for surface in bank]},
        "narrative_report": {"schema_version": "narrative_report.v2", "surfaces": [
            {"surface": surface, "summary": "脚本摘要，仅验证传输。", "reader_takeaway": "脚本理解。",
             "reasoning": "脚本理由，不代表模型已经判断论证。", "coherent": True, "thesis_focus": True,
             "issues": [], "analysis": scripted_analysis(), "supporting_spans": [
                 {"source": source, "span_id": fields[source][0]["span_id"], "role": role}
                 for source, role in (("title", "thesis"), *[("body", role)
                     for role in ("news", "thesis", "reason", "impact", "limit")])]}
            for surface, fields in bank.items()]},
    }


@pytest.mark.parametrize("title,body", TEXTS)
def test_reasoning_examples_reach_blind_reader_and_span_bank_without_repair(title, body):
    output = fictional_output(title, body)
    before = output.model_dump(mode="json")
    visible = {"audience": "虚构书柜与文具使用者"}
    hidden = {key: "PRIVATE_SENTINEL_" + key for key in (
        "author_plan", "angle", "value_contract", "reader_value", "canonical_claims",
        "approved_proposal", "owner_revision", "editorial_feedback", "prior_draft", "quality_scorecard")}
    payload = _reader_payload(output, {**visible, **hidden}, scorecard_required=True)
    assert payload == _reader_payload(output, visible, scorecard_required=True)
    draft = payload["editorial_review"]["draft"]
    assert draft == {key: before[key] for key in ("master_title", "master_content")} | {
        "platform_variants": [{key: row[key] for key in ("platform", "title", "body")}
                              for row in before["platform_variants"]]}
    actual = {"master": {"title": title, "body": body}, **{
        row["platform"]: {"title": row["title"], "body": row["body"]}
        for row in draft["platform_variants"]}}
    for surface, fields in payload["editorial_review"]["source_span_bank"].items():
        for source, spans in fields.items():
            assert "".join(span["text"] for span in spans) == actual[surface][source]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert all(value not in serialized for value in hidden.values())
    assert output.model_dump(mode="json") == before


@pytest.mark.parametrize("fault", ["reader_relevance", "information_gain", "analytical_depth",
                                  "natural_expression", "platform_fit", "major", "narrative_false",
                                  "changed_quote"])
def test_scripted_reasoning_rejections_keep_original_text_evidence_and_verdict(fault):
    output = fictional_output(*TEXTS[0])
    before = output.model_dump(mode="json")
    raw = scripted_assessment(_reader_payload(output, {}, scorecard_required=True))
    issue = "脚本记录：标题要求先交卡，正文要求预约后交卡；不得替作者交换顺序。"
    raw["issues"] = [issue]
    card = raw["quality_scorecard"]["surfaces"][0]
    narrative = raw["narrative_report"]["surfaces"][0]
    if fault == "major":
        card["major_issues"] = [issue]
    elif fault == "narrative_false":
        narrative["coherent"] = False
        narrative["issues"] = [issue]
    elif fault == "changed_quote":
        narrative["supporting_spans"][0] = {
            "source": "title", "quote": "先预约，再交回书卡", "role": "thesis"}
    else:
        card[fault] = 3
    original = deepcopy(raw)
    calls = []

    async def reader(system, payload, fallback, *args, **kwargs):
        calls.append((system, deepcopy(payload)))
        return deepcopy(raw)

    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    review, state = asyncio.run(assess_editorial(
        output=output, brief={}, run=run, llm=reader, system="PRIVATE_SYSTEM_SENTINEL", model_args=(),
        owner_revision={"instruction": "PRIVATE_OWNER_SENTINEL"},
        style_context={"examples": ["PRIVATE_STYLE_SENTINEL"]},
        editorial_feedback="PRIVATE_PREVIOUS_VERDICT_SENTINEL", scorecard_required=True))
    assert len(calls) == 1 and calls[0][0] == EDITOR_SYSTEM_PROMPT
    assert "PRIVATE_" not in json.dumps(calls, ensure_ascii=False)
    assert raw == original
    assert run.private_review_receipt["candidate"] == {key: value for key, value in original.items()
                                                     if not key.startswith("_")}
    if fault == "changed_quote":
        assert review is None and state == "not_completed"
        assert run.private_review_receipt["validation"]["code"] == "narrative_report_span_invalid"
    else:
        assert state == "failed" and not review.passed
        assert issue in review.issues
        assert [row.model_dump(mode="json") for row in review.quality_scorecard.surfaces] == original[
            "quality_scorecard"]["surfaces"]
        row = review.narrative_report.surfaces[0]
        assert row.coherent == narrative["coherent"]
        assert row.issues == narrative["issues"]
        assert [span.quote for span in row.supporting_spans] == [output.master_title, *[output.master_content] * 5]
    blocked = apply_editorial_assessment(output, review)
    assert blocked.status == "quality_insufficient"
    assert blocked.master_title == output.master_title and blocked.master_content == output.master_content
    assert output.model_dump(mode="json") == before
