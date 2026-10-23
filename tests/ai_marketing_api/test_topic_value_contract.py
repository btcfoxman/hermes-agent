"""Value gates and prompt transport, not an offline claim of model judgment quality.

The rejected first article is historical public copy. Scripted assessments below
exercise the real protocol/gates; only a future independent model assessment can
establish whether the revised rubric changes real scoring behavior.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import ai_marketing_api.main as api
from ai_marketing_api.operator_content import OperatorContentOutput, SUPPORTED_CHANNEL_FORMATS
from ai_marketing_api.operator_editorial import (
    BoundedModelRun, PublicEditorialBrief, RuntimeBudget, TopicValueContract,
)
from ai_marketing_api.operator_planning import EditorialPlanRequest
from ai_marketing_api.operator_review import apply_editorial_assessment, assess_editorial
from tests.ai_marketing_api.test_autonomous_editorial import _brief, _planner_payload
from tests.ai_marketing_api.test_operator_api import _request, _role_compose_payload


# Synthetic documentation for a protocol-positive case, not a claim about a live product.
SOURCE = (
    "导出的报告包含结论和引用列表，但不包含原始输入。"
    "审核页允许逐条打开引用，不判断引用是否足以支撑结论。"
)


def contract():
    return {
        "reader_problem": "报告有引用列表时，是否可以直接把结论交给下一位编辑？",
        "thesis": "来源可打开与论证足够是不同的验收动作，引用齐备不代表结论可以直接交接。",
        "evidence_anchor": {"opportunity_id": "opp-api", "quote": SOURCE},
        "decision_delta": "因报告不带原始输入，交接前需另核对原材料与结论的关系；无法核对时先补材料，不把打开引用当作论证已完成。",
        "counterargument": "若另一份明确授权的文档证明可取得原始输入，材料交接安排可以改变；引用是否支撑结论仍需单独判断。",
    }


def selection(**changes):
    return {
        "opportunity_id": "opp-api", "role_id": "commercial",
        "topic": "引用齐备为何还不能直接交接报告",
        "angle": "根据已读报告导出边界，区分来源可查与结论可用。",
        "reason": "帮助编辑避免把引用列表当作论证完成。",
        "value_contract": contract(), **changes,
    }


def planning_request(required=True):
    payload = _planner_payload()
    payload["topic_value_required"] = required
    payload["opportunities"][0].update(title="报告导出的公开说明", summary=SOURCE)
    payload["opportunities"][1]["summary"] = "A verified industry update."
    return payload


@pytest.fixture(autouse=True)
def service_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def run_planner(monkeypatch, proposed, *, required=True, skips=None):
    calls = []

    async def one_model_response(system, body, fallback, *args, **kwargs):
        calls.append((system, body))
        return {"_model": "offline-protocol-stub", "selections": proposed,
                "skips": skips or []}

    monkeypatch.setattr(api, "_llm_json", one_model_response)
    response = _request("POST", "/api/v1/operators/editorial/plan",
                        json=planning_request(required))
    assert response.status_code == 200
    result = response.json()
    assert len(calls) == result["generation_trace"]["model_calls"] == 1
    return result, calls[0]


def test_required_value_contract_has_exact_source_binding_without_an_extra_call(monkeypatch):
    result, (system, body) = run_planner(monkeypatch, [selection()])
    assert result["status"] == "planned"
    assert result["selections"][0]["value_contract"] == contract()
    assert body["topic_value_required"] is True
    assert "value_contract" in body["response_contract"]["selections"][0]
    assert "selections=[]" in system
    assert "Naming a specific task is not" in system
    assert "never new factual authorization" in system
    assert "verbatim contiguous quote from its summary" in system
    assert body["opportunities"][0]["summary"] == SOURCE


@pytest.mark.parametrize("failure", [
    "missing", "empty", "blank_problem", "blank_thesis", "blank_delta",
    "blank_counterargument", "blank_quote", "blank_id", "unknown_id",
    "other_opportunity", "paraphrase", "title_only", "extra_fact_authority",
    "nonstring_thesis",
])
def test_incomplete_or_forged_value_contract_fails_closed(monkeypatch, failure):
    proposed = selection()
    value = proposed["value_contract"]
    if failure == "missing":
        proposed.pop("value_contract")
    elif failure == "empty":
        proposed["value_contract"] = {}
    elif failure.startswith("blank_"):
        field = {"problem": "reader_problem", "thesis": "thesis", "delta": "decision_delta",
                 "counterargument": "counterargument", "quote": "quote", "id": "opportunity_id"}[failure[6:]]
        if field in {"quote", "opportunity_id"}:
            value["evidence_anchor"][field] = " \n "
        else:
            value[field] = " \n "
    elif failure == "unknown_id":
        value["evidence_anchor"]["opportunity_id"] = "invented-source"
    elif failure == "other_opportunity":
        value["evidence_anchor"]["opportunity_id"] = "opp-news"
    elif failure == "paraphrase":
        value["evidence_anchor"]["quote"] = "报告自动确认每条引用均支撑结论。"
    elif failure == "title_only":
        value["evidence_anchor"]["quote"] = "报告导出的公开说明"
    elif failure == "extra_fact_authority":
        value["claims"] = [{"text": " invented capability"}]
    else:
        value["thesis"] = True
    original = copy.deepcopy(proposed)
    result, _ = run_planner(monkeypatch, [proposed])
    assert result["status"] == "unavailable"
    assert result["selections"] == []
    assert proposed == original
    assert result["generation_trace"]["repair_calls"] == 0


def test_other_existing_opportunity_cannot_lend_its_summary(monkeypatch):
    proposed = selection()
    proposed["value_contract"]["evidence_anchor"]["quote"] = "A verified industry update."
    result, _ = run_planner(monkeypatch, [proposed])
    assert result["status"] == "unavailable" and result["selections"] == []


def test_required_lane_without_quotable_material_abstains_before_a_model(monkeypatch):
    payload = planning_request()
    for opportunity in payload["opportunities"]:
        opportunity["summary"] = " \n "

    async def no_material_no_model(*args, **kwargs):
        pytest.fail("A title alone cannot support the required evidence quote")

    monkeypatch.setattr(api, "_llm_json", no_material_no_model)
    result = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert result["status"] == "no_opportunities" and result["selections"] == []
    assert result["generation_trace"]["model_calls"] == 0


def test_original_positioning_topic_without_value_proof_cannot_enter_required_lane(monkeypatch):
    payload = planning_request()
    payload["opportunities"] = [{
        "id": "first-positioning", "role_id": "commercial", "verified": True,
        "title": "AIID API 站面向漫剧创作与视频工具开发者",
        "summary": (
            "AIID API 站的目标用户包括漫剧创作、电商内容与广告团队，也包括视频工具、漫剧工作台和音视频工具箱的开发者。"
            "API 站的产品定位是稳定、性价比高、模型全面；具体模型、价格和服务条件以站内当前说明为准。"),
    }]

    async def old_selection_without_value_proof(*args, **kwargs):
        return {"_model": "offline-protocol-stub", "selections": [{
            "opportunity_id": "first-positioning", "role_id": "commercial",
            "topic": payload["opportunities"][0]["title"],
            "angle": "假设读者要交接脚本，先核对接口是否符合流程。",
            "reason": "帮助目标用户判断是否适合自己的工作流。"}]}

    monkeypatch.setattr(api, "_llm_json", old_selection_without_value_proof)
    result = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert result["status"] == "unavailable" and result["selections"] == []
    assert result["generation_trace"]["model_calls"] == 1
    assert result["generation_trace"]["repair_calls"] == 0


def test_explicit_no_value_abstention_is_successful_and_keeps_material_gap(monkeypatch):
    result, _ = run_planner(monkeypatch, [], skips=[{
        "role_id": "commercial", "reason": "现有材料只有受众与定位，没有支持此判断的产品约束。"}])
    assert result["status"] == "planned" and result["selections"] == []
    assert any("只有受众与定位" in row["reason"] for row in result["skips"])


def test_legacy_request_and_selection_remain_compatible(monkeypatch):
    legacy = selection()
    legacy.pop("value_contract")
    payload = planning_request()
    payload.pop("topic_value_required")
    assert EditorialPlanRequest.model_validate(payload).topic_value_required is False
    result, _ = run_planner(monkeypatch, [legacy], required=False)
    assert result["status"] == "planned"
    assert result["selections"][0]["value_contract"] is None


def test_optional_supplied_contract_cannot_smuggle_a_forged_quote(monkeypatch):
    proposed = selection()
    proposed["value_contract"]["evidence_anchor"]["quote"] = "The product guarantees a result."
    result, _ = run_planner(monkeypatch, [proposed], required=False)
    assert result["status"] == "unavailable" and result["selections"] == []


@pytest.mark.parametrize("value", ["true", 1, None])
def test_topic_value_requirement_is_a_strict_boolean(value):
    payload = planning_request()
    payload["topic_value_required"] = value
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(payload)


def test_value_contract_reaches_real_writer_without_becoming_evidence():
    payload = _role_compose_payload("commercial")
    payload["channels"] = ["wechat_mp"]
    payload["public_editorial_brief"] = {**_brief("commercial"), "value_contract": contract()}
    request = api.OperatorComposeRequest.model_validate(payload)
    without_contract = copy.deepcopy(payload)
    without_contract["public_editorial_brief"].pop("value_contract")
    baseline = api.OperatorComposeRequest.model_validate(without_contract)
    actual = api.content_request_payload("commercial", request, request.authorized_context)
    expected = api.content_request_payload("commercial", baseline, baseline.authorized_context)
    assert actual["public_editorial_brief"]["value_contract"] == contract()
    for key in ("claims", "authorized_context", "canonical_block_registry", "approved_preferences"):
        assert actual[key] == expected[key]
    assert request.runtime_budget == baseline.runtime_budget
    assert PublicEditorialBrief.model_validate(payload["public_editorial_brief"]).value_contract == TopicValueContract.model_validate(contract())


def output_from(draft):
    return OperatorContentOutput(role_id="commercial", status="content_ready",
        profile_version="offline-value-gate", system_prompt_sha256="0" * 64,
        master_title=draft["master_title"], master_content=draft["master_content"],
        platform_variants=[{**v, "format": SUPPORTED_CHANNEL_FORMATS[v["platform"]]}
                           for v in draft["platform_variants"]],
        critic={"passed": True, "checks": [], "errors": [], "warnings": []})


def assess_with_scripted_grade(output, score, brief):
    """A protocol stub, deliberately not a semantic model evaluation."""
    seen = []

    async def scripted_editor(system, body, fallback, *args, **kwargs):
        seen.append(body)
        surfaces = body["editorial_review"]["output_rules"]["expected_surfaces"]
        from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review
        return {"_model": "offline-assessment-stub", "narrative_report": scripted_review(body)["narrative_report"], "relevant_to_brief": True,
            "useful_to_reader": score >= 4, "clear_and_specific": True, "platform_fit": True,
            "issues": [] if score >= 4 else ["受众定位与通用假设未给出有依据的新判断。"],
            "failed_surfaces": [] if score >= 4 else surfaces,
            "quality_scorecard": {"schema_version": "quality_scorecard.v1", "surfaces": [
                {"surface": surface, "reader_relevance": 4, "information_gain": score,
                 "analytical_depth": score, "natural_expression": 4, "platform_fit": 4,
                 "major_issues": [] if score >= 4 else ["缺少来源专属的读者价值。"], "issues": []}
                for surface in surfaces]}}

    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    review, state = asyncio.run(assess_editorial(output=output, brief=brief, run=run,
        llm=scripted_editor, system="unused", model_args=(), scorecard_required=True))
    assert len(seen) == 1 and len(run.attempts) == 1
    return review, state, apply_editorial_assessment(output, review), seen[0]


def test_original_owner_rejected_article_reaches_stricter_review_and_low_grade_blocks():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
        "editorial_scored_minor_issues.v1.json").read_text(encoding="utf-8"))
    original = copy.deepcopy(fixture)
    output = output_from(fixture["draft"])
    review, state, blocked, body = assess_with_scripted_grade(output, 2, {})
    instructions = body["editorial_review"]
    assert instructions["draft"] == fixture["draft"]
    assert set(instructions["dimension_anchors"]) == {
        "reader_relevance", "information_gain", "analytical_depth", "natural_expression", "platform_fit"}
    assert "actual takeaway" in instructions["dimension_anchors"]["analytical_depth"]
    rules = " ".join(instructions["comprehension_checks"])
    assert "at most 3" in rules and "material problem" in rules
    assert "replacing its product name" in rules
    assert "no specific fact in this text" in rules
    assert "canonical_claims" not in instructions
    assert "value_contract" not in body["response_contract"]
    assert state == "failed" and review is not None and not review.passed
    assert blocked.status == "quality_insufficient" and not blocked.critic.passed
    assert blocked.quality_scorecard.passed is False
    assert blocked.master_content == fixture["draft"]["master_content"]
    assert fixture == original  # historical grades/receipt are never rewritten


def test_supported_specific_judgment_can_pass_without_changing_claim_authority():
    draft = {"master_title": "引用可查，为什么还不能直接交接结论", "platform_variants": [],
             "master_content": SOURCE + "\n\n" + contract()["thesis"] + "\n\n" + contract()["decision_delta"]}
    output = output_from(draft)
    brief = {**_brief("commercial"), "value_contract": contract()}
    original = output.model_dump(mode="json")
    review, state, accepted, body = assess_with_scripted_grade(output, 4, brief)
    assert "public_editorial_brief" not in body["editorial_review"]
    assert contract()["thesis"] in body["editorial_review"]["draft"]["master_content"]
    assert state == "passed" and review.passed and accepted.critic.passed
    assert accepted.blocks == output.blocks
    assert accepted.master_content == original["master_content"]
    assert accepted.master_title == original["master_title"]
    assert accepted.quality_scorecard.content_sha256


def test_read_document_constraints_support_a_specific_draft_selection_contract(monkeypatch):
    # Sanitized atomic facts from the actually read official Draft documentation,
    # supplied by the source-audit lane. No API call or service fact import here.
    material = (
        "Draft 样片任务和基于样片的正式视频任务独立计费；生成样片不能视为免费的成片预览。"
        "从 Draft 生成正式视频时，重新传入被规定自动复用的参数会报错，即使值与样片阶段一致也一样。"
    )
    payload = planning_request()
    payload["opportunities"] = [{"id": "draft-official-read", "role_id": "industry",
        "title": "Draft 的计费与输入复用约束", "summary": material, "verified": True}]
    value = {
        "reader_problem": "创意尚未选定时，何时应从样片转入正式视频？",
        "thesis": "把样片当作提前淘汰不满意方案的决策点，而不是免费的成片预览或修改输入的编辑阶段。",
        "evidence_anchor": {"opportunity_id": "draft-official-read", "quote": material},
        "decision_delta": "提示词或参考素材仍需修改时先回到样片阶段；选定方案再进入正式阶段，不重新传入规定复用的参数。",
        "counterargument": "任务已充分确定时，增加样片仍会独立计费；不能据此断言每次都更省钱。",
    }

    async def documented_selection(*args, **kwargs):
        return {"_model": "offline-protocol-stub", "selections": [{
            "opportunity_id": "draft-official-read", "role_id": "industry",
            "topic": "Draft 收费之后，预览真正该替你决定什么",
            "angle": value["thesis"], "reason": value["reader_problem"], "value_contract": value}]}

    monkeypatch.setattr(api, "_llm_json", documented_selection)
    result = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert result["status"] == "planned"
    assert result["selections"][0]["value_contract"] == value
    assert result["generation_trace"]["model_calls"] == 1

