"""R18: conditional creative readiness is not an external event assertion."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_review import apply_editorial_assessment, assess_editorial
from tests.ai_marketing_api.test_industry_conditional_editorial import _canonical, _normalize


ROUND18_CREATIVE_CONDITION = (
    "但如果主体、动作、镜头安排和素材已经确定，只剩正式成片的画质细节需要验收，"
    "低清样片未必能回答关键问题，这时额外的样片费用和审片投入就未必划算。"
)
ROUND18_PAIRED_CONTRAST = "样片不是默认流程，而是用来提前淘汰不满意方案的决策点。"


@pytest.mark.parametrize("text", [
    ROUND18_CREATIVE_CONDITION,
    "如果提示词与参考素材已经确定，先判断样片能否回答尚待决定的问题。",
    "若镜头安排已经确定，是否先看低清样片取决于验收问题。",
    "当素材已经确定时，先明确要讨论的是创意还是成片细节。",
])
def test_creative_condition_survives_four_surfaces_with_exact_facts(text):
    output, fallback = _normalize(text)
    assert text in output.master_content
    assert all(text in variant.body for variant in output.platform_variants)
    assert not output.critic.errors
    assert not any(warning.startswith("discarded_unsafe_model_editorial:") for warning in output.critic.warnings)
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        assert _canonical([block.model_dump(mode="json") for block in blocks]) == _canonical(fallback["blocks"])
    assert output.quality_scorecard is None


@pytest.mark.parametrize("text", [
    "那时已经不能改了。",
    "主体、动作、镜头安排和素材已经确定，只剩成片验收。",
    "如果公司已经确定，后续可以直接进入市场。",
    "如果主体与动作已经确定，可以讨论接下来的安排。",
    "如果素材已经确定上线时间，可以讨论安排。",
    "如果提示词已经确定，平台已经上线新的功能。",
    "如果提示词已经确定，那时已经不能改了。",
    "如果提示词已经确定，公司宣布发布新模型。",
    "如果提示词已经确定，公司获得收益。",
    "如果提示词已经确定，效果已经得到证实。",
    "如果提示词已经确定，平台保证审核通过。",
    "如果提示词已经确定，我用它完成了客户项目。",
    "如果提示词已经确定，准确率达到99%。",
])
def test_conditional_span_does_not_authorize_then_events_capabilities_or_experiences(text):
    output, fallback = _normalize(text)
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert any(warning.startswith("discarded_unsafe_model_editorial:") for warning in output.critic.warnings)
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        assert _canonical([block.model_dump(mode="json") for block in blocks]) == _canonical(fallback["blocks"])


def test_creative_exception_is_only_the_exact_already_token():
    text = "但如果主体、动作、镜头安排和素材已经确定，平台已经宣布发布功能。"
    matches = content._unsupported_editorial_assertion_matches(text, allow_industry_hypothetical=True)
    assert [match.group(0) for match in matches] == ["已经", "宣布", "发布"]
    assert matches[0].start() == text.index("已经", text.index("已经") + 2)
    assert len(content._unsupported_editorial_assertion_matches(text)) == 4


@pytest.mark.parametrize("text", [
    ROUND18_PAIRED_CONTRAST,
    "这并非默认流程，而是需要按实际任务决定的选项。",
    "低清样片未必是最后的验收依据，更是提前筛选创意的材料。",
])
def test_paired_contrast_is_retained_as_a_style_diagnostic_without_issuing_a_grade(text):
    output, fallback = _normalize(text)
    assert text in output.master_content
    assert all(text in variant.body for variant in output.platform_variants)
    warnings = output.critic.warnings
    assert len([warning for warning in warnings if warning.startswith("editorial_style_review_required:")]) == 4
    assert all(text not in warning for warning in warnings)
    assert not any(warning.startswith("discarded_unsafe_model_editorial:") for warning in warnings)
    assert not output.critic.errors and output.quality_scorecard is None
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        assert _canonical([block.model_dump(mode="json") for block in blocks]) == _canonical(fallback["blocks"])


@pytest.mark.parametrize("text", [
    "样片不是默认流程，而是需要考虑的选项。编辑说明：请展开观点部分。",
    "这不是最值得注意的是，而是需要讨论的取舍。",
    "这不是重点不只是事实，而是请讨论的取舍。",
    "这不是默认流程，而是最值得注意的是后续看点。",
    "这不是默认流程，而是平台已经支持自动生成视频。",
    "这不是默认流程，而是工具保证没有风险。",
])
def test_other_meta_nested_wrappers_and_unsupported_claims_still_block(text):
    output, _ = _normalize(text)
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert any(warning.startswith("discarded_unsafe_model_editorial:") for warning in output.critic.warnings)


@pytest.mark.parametrize("genre", ["", "digest", "company_data_brief"])
def test_both_exceptions_remain_outside_legacy_genres(genre):
    for text in (ROUND18_CREATIVE_CONDITION, ROUND18_PAIRED_CONTRAST):
        output, _ = _normalize(text, genre=genre)
        assert text not in output.master_content
        assert all(text not in variant.body for variant in output.platform_variants)


def test_empty_contrast_does_not_bypass_independent_quality_review():
    # Lexical style is diagnostic in analysis, but an empty thesis is still a
    # real quality rejection. The stub verifies protocol, not model judgment.
    text = "这不是一个数字，而是行业规则的真正落点。"
    output, _ = _normalize(text)
    assert text in output.master_content and output.quality_scorecard is None
    original = deepcopy(output.model_dump(mode="json"))

    async def independent_low_grade(_system, body, _fallback, **_kwargs):
        surfaces = body["editorial_review"]["output_rules"]["expected_surfaces"]
        assert text in body["editorial_review"]["draft"]["master_content"]
        return {"_model": "offline-review-protocol", "relevant_to_brief": True,
            "useful_to_reader": False, "clear_and_specific": False, "platform_fit": True,
            "issues": ["空泛对比缺少具体判断。"], "failed_surfaces": surfaces,
            "quality_scorecard": {"surfaces": [{"surface": surface,
                "reader_relevance": 3, "information_gain": 2, "analytical_depth": 2,
                "natural_expression": 3, "platform_fit": 4,
                "major_issues": ["缺少有据判断。"], "issues": []} for surface in surfaces]}}

    review, state = asyncio.run(assess_editorial(output=output, brief={},
        run=BoundedModelRun(RuntimeBudget(max_model_calls=1)), llm=independent_low_grade,
        system="offline", model_args=(), scorecard_required=True))
    assessed = apply_editorial_assessment(output, review)
    assert state == "failed" and not review.passed
    assert not assessed.critic.passed and not assessed.quality_scorecard.passed
    assert output.model_dump(mode="json") == original

