"""Prompt transport and safety contracts; scripted results are not quality calibration."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_content import OperatorComposeRequest
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_grounding import apply_grounded_paraphrases
from ai_marketing_api.operator_review import _reader_payload, assess_editorial
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_grounded_expression import _block, _output, _raw
from tests.ai_marketing_api.test_operator_api import _base_payload
from tests.ai_marketing_api.test_reader_current_text_reasoning import (
    fictional_output, scripted_assessment,
)


SOURCE = "青鸟标记台仅接受公开文本；材料提交后保留原始栏目与审核备注，修改栏目或备注需要返回材料登记页。"
PROPOSAL = "青鸟标记台只接收公开文本，收到后保留原始栏目和审核备注；需修改栏目或备注时回到材料登记页。"


def compose(mode, surface):
    value = _base_payload()
    value.update(topic="青鸟标记台的材料修改入口", audience="虚构材料台使用者",
        channels=[surface], genre="company_data_brief", fact_expression_mode=mode,
        approved_proposal={"title": "青鸟标记台的材料修改入口", "angle": "解释已公开的材料修改条件。",
            "audience_value": "理解何时需要返回登记页。", "key_points": [SOURCE],
            "suggested_formats": [surface], "first_person": False},
        claims=[{"text": SOURCE, "kind": "fact", "evidence_ids": ["source-1"],
                 "verification_status": "verified"}])
    value["authorized_context"][0].update(title="虚构材料台说明", content=SOURCE)
    return OperatorComposeRequest.model_validate(value)


@pytest.mark.parametrize("surface", ["wechat_mp", "wechat_moments", "weitoutiao"])
@pytest.mark.parametrize("mode", ["exact", "grounded_paraphrase"])
def test_final_master_and_adaptation_payloads_explain_display_without_changing_sources(
    monkeypatch, surface, mode,
):
    monkeypatch.setenv("HERMES_API_KEY", "fictional-surface-prompt-key")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-writer"})
    request = compose(mode, surface)
    before = request.model_dump(mode="json")
    seen = []

    async def writer(system, body, fallback, **kwargs):
        seen.append(deepcopy(body))
        refs = [{"block_ref": block["block_ref"]}
                for block in body["canonical_block_registry"] if block["required"]]
        if body["stage_writer_contract"]["step"] == "draft_master":
            return {"_model": "offline-writer", "master_title": request.approved_proposal.title,
                "blocks": refs, "platform_variants": []}
        return {"_model": "offline-writer", "platform_variants": [{"platform": surface,
            "title": "青鸟标记台收到材料后如何修改", "blocks": refs}]}

    async def run():
        registry = OperatorRegistry()
        master = await steps.run_editorial_step(registry, "industry", "draft_master",
            steps.EditorialStepRequest(compose=request), writer)
        assert master["status"] == "artifact", master
        adapted = await steps.run_editorial_step(registry, "industry", "adapt_platform",
            steps.EditorialStepRequest(compose=request, surface=surface,
                candidate=master["artifact"], candidate_signature=master["artifact_signature"]), writer)
        assert adapted["status"] == "artifact", adapted
        return master, adapted

    master, adapted = asyncio.run(run())
    assert len(seen) == 2
    guidance = [" ".join(body["response_contract"]["safety"]) for body in seen]
    for rules in guidance:
        if mode == "grounded_paraphrase":
            assert "not an automatic natural-language summary" in rules
            assert "public_text proposal beside the original block_ref" in rules
            assert "bounded scope and material conditions" in rules
            assert "wording for display only after independent verification supports it" in rules
            assert "renders long ones as concise" not in rules
        else:
            assert "renders long ones as concise source-exact fact beats" in rules
            assert "never paraphrase it" in rules
            assert "public_text proposal" not in rules
    for body in seen:
        assert [block["text"] for block in body["canonical_block_registry"]] == [SOURCE]
        assert all(block["source_exact"] and block["locked"] for block in body["canonical_block_registry"])
        if mode == "grounded_paraphrase":
            assert body["constraints"][0] == body["response_contract"]["safety"][1]
            assert "Prices, dates, numbers, promises, identity and experience stay verbatim" in body["constraints"][0]
    assert seen[0]["canonical_block_registry"] == seen[1]["canonical_block_registry"]
    for artifact, blocks, rendered in [
        (master["artifact"], master["artifact"]["blocks"], master["artifact"]["master_content"]),
        (adapted["artifact"], adapted["artifact"]["platform_variants"][0]["blocks"],
         adapted["artifact"]["platform_variants"][0]["body"]),
    ]:
        assert rendered == SOURCE  # Guidance alone never synthesizes a shorter expression.
        assert blocks[0]["text"] == SOURCE and not blocks[0]["public_text_verified"]
        assert artifact["quality_scorecard"] is None
    assert request.model_dump(mode="json") == before


@pytest.mark.parametrize("verdict", [None, "uncertain", "supported"])
def test_optional_summary_changes_display_only_after_independent_support(verdict):
    block = _block(SOURCE)
    output = _output(block)
    before = output.model_dump(mode="json")
    raw = _raw(PROPOSAL, public_text_verified=True) if verdict is not None else []
    calls = []

    async def verifier(system, body, fallback, **kwargs):
        calls.append(deepcopy(body))
        return {"_model": "offline-independent-verifier", "reviews": [{
            "pair_id": pair["pair_id"], "verdict": verdict, "key_fields_preserved": True,
            "reason": "Scripted support contract, not a model quality judgment."}
            for pair in body["claim_verification"]["pairs"]]}

    result = asyncio.run(apply_grounded_paraphrases(output=output, raw_candidates=raw,
        run=BoundedModelRun(RuntimeBudget()), llm=verifier, system="stable", model_args=()))
    assert len(calls) == (0 if verdict is None else 1)
    assert result.master_content == (PROPOSAL if verdict == "supported" else SOURCE)
    assert result.platform_variants[0].body == result.master_content
    for actual in (result.blocks[0], result.platform_variants[0].blocks[0]):
        assert actual.text == SOURCE and actual.binding_hash == block.binding_hash
        assert actual.evidence_ids == block.evidence_ids
        assert actual.public_text_verified is (verdict == "supported")
    assert output.model_dump(mode="json") == before


def test_reader_final_payload_scopes_scores_major_issues_and_narrative_to_each_own_text():
    output = fictional_output("纸盒封口前先晾干", "琥珀纸盒必须晾干后封口，否则水汽会留在盒内。")
    output.platform_variants[0].title = "蓝色夹子怎样标记纸页"
    output.platform_variants[0].body = "蓝色夹子用于标记纸页；颜色便于找到标记，不改变纸页顺序。"
    before = output.model_dump(mode="json")
    payload = _reader_payload(output, {"audience": "虚构文具使用者"}, scorecard_required=True)
    rules = payload["editorial_review"]
    instruction = rules["instruction"]
    assert "every score, issue, major issue and narrative report" in instruction
    assert "surface's own title and body" in instruction
    assert "Do not import assertions, conclusions or conditions from another surface" in instruction
    assert "including the master when judging a platform variant" in instruction
    assert "repeat another surface's details" in instruction
    assert "necessary to support a claim actually made in this surface" in instruction
    assert "causal chain, conditions, information gain, analytical depth, platform fit" in instruction
    assert "all five >=4 and zero major issues" in rules["scorecard_rubric"]
    assert "at most 3" in " ".join(rules["comprehension_checks"])
    assert rules["draft"]["master_content"] == before["master_content"]
    assert rules["draft"]["platform_variants"][0]["body"] == before["platform_variants"][0]["body"]
    bank = rules["source_span_bank"]
    assert "".join(span["text"] for span in bank["master"]["body"]) == output.master_content
    assert "".join(span["text"] for span in bank["wechat_mp"]["body"]) == output.platform_variants[0].body
    assert output.model_dump(mode="json") == before


@pytest.mark.parametrize("fault", ["analytical_depth", "major", "narrative_false", "cross_surface_span"])
def test_surface_scope_guidance_preserves_real_low_major_and_exact_span_gates(fault):
    output = fictional_output("纸盒封口前先晾干", "琥珀纸盒需要晾干后封口。")
    before = output.model_dump(mode="json")
    raw = scripted_assessment(_reader_payload(output, {}, scorecard_required=True))
    issue = "虚构测试保留当前面的实际问题，不代表模型已作语义判断。"
    row = raw["quality_scorecard"]["surfaces"][1]
    narrative = raw["narrative_report"]["surfaces"][1]
    if fault == "major":
        row["major_issues"] = [issue]
    elif fault == "narrative_false":
        narrative.update(coherent=False, issues=[issue])
    elif fault == "cross_surface_span":
        narrative["supporting_spans"][1]["span_id"] = raw["narrative_report"]["surfaces"][0][
            "supporting_spans"][1]["span_id"]
    else:
        row[fault] = 3
    original = deepcopy(raw)
    seen = []

    async def reader(system, body, fallback, **kwargs):
        seen.append(deepcopy(body))
        return deepcopy(raw)

    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    review, state = asyncio.run(assess_editorial(output=output, brief={}, run=run, llm=reader,
        system="unused", model_args=(), scorecard_required=True))
    assert len(seen) == 1
    assert "surface's own title and body" in seen[0]["editorial_review"]["instruction"]
    assert raw == original and output.model_dump(mode="json") == before
    assert run.private_review_receipt["candidate"] == {
        key: value for key, value in original.items() if not key.startswith("_")}
    if fault == "cross_surface_span":
        assert review is None and state == "not_completed"
        assert run.private_review_receipt["validation"]["code"] == "narrative_report_span_invalid"
    else:
        assert review is not None and state == "failed" and not review.passed
        assert [row.model_dump(mode="json") for row in review.quality_scorecard.surfaces] == original[
            "quality_scorecard"]["surfaces"]
        assert review.narrative_report.surfaces[1].coherent == narrative["coherent"]
