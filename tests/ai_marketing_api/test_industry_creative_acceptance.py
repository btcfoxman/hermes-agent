"""Synthetic creative conditions are local spans, never product assertions."""
import pytest

from ai_marketing_api import operator_content as content
from tests.ai_marketing_api.test_industry_nonnumeric_spans import _assert_facts_unchanged, _normalize


SAFE = (
    "假设镜头已经呈现出期待，只待检查最终画面的细节。",
    "若紧张已经表现出来，先讨论是否沿用既定的创作方案。",
    "如果提示词与参考素材已经定下，先讨论镜头的表达。",
    "当创作输入已经确定时，先区分创意检查与最终验收。",
    "假设预览画面已经拍出不舍，细节仍需另行验收。",
    "若所需的情绪已经呈现出来，先核对是否符合表达目标。",
    "假设镜头已经呈现了预期的情绪，先讨论是否符合目标。",
    "若样片已经表达出了所需的创作意图，细节仍需核验。",
    "这项约束给预览划出了一条明确界线：先讨论表达。",
    "审看更像一次有具体目标的核对。",
    "图板可以帮助读者判断：画面是否表达了期待。",
    "这个假设用来判断：情绪是否合意。",
    "审看时，可以帮你判断：镜头是否清楚。",
)

UNSAFE = (
    "镜头已经呈现出期待，只待检查细节。",
    "假设产品已经支持自动剪辑，先讨论表达。",
    "若情绪已经表达出来，平台已经发布新产品。",
    "若情绪已经表达出来，效果得到证实。",
    "如果输入已经确定上线时间，先讨论安排。",
    "如果输入已经定下，收益增长20%。",
    "并非假设镜头已经呈现出期待，效果仍需核验。",
    "若情绪不是已经表达出来，先讨论安排。",
    "如果样片已经实现自动审查，先讨论表达。",
    "若销量已经表现出来，先讨论安排。",
    "假设画面已经拍出期待，准确率达到99%。",
    "若紧张已经表现出来，工具保证效果。",
    "如果输入已经定下，我亲自完成了客户项目。",
    "审看更像一次有具体目标的核对，之后只需一次就完成验收。",
    "这项约束给预览划出了两条明确界线。",
    "某机构给预览划出了一条业务边界。",
    "一次完成最终验收。",
    "只用一次提升效果。",
    "这项约束给预览划出了一条明确界线，投入下降99%。",
    "若样片已经表达出了所需的创作意图，平台已经支持自动审核。",
    "假设镜头已经呈现了预期的情绪。效果已经得到证实。",
    "帮助读者判断：是否清楚。事实部分：以此为准。",
    "用来判断：是否合意。编辑判断：现在照做。",
    "以下分析用来判断：是否清楚。",
    "工具已经支持帮你判断：可以交付。",
    "帮助你判断：效果提升20%。",
    "我亲自用它帮助你判断：画面是否清楚。",
    "编辑判断：下一段继续。",
)


@pytest.mark.parametrize("prose", SAFE)
def test_synthetic_conditions_and_conceptual_counts_survive_the_real_four_surface_normalizer(prose):
    output, fallback = _normalize(prose)
    assert prose in output.master_content
    assert all(prose in variant.body for variant in output.platform_variants)
    assert not output.critic.errors
    assert not any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    _assert_facts_unchanged(output, fallback)
    assert output.quality_scorecard is None and output.editorial_assessment is None


@pytest.mark.parametrize("prose", UNSAFE)
def test_safe_span_never_authorizes_another_event_quantity_capability_or_experience(prose):
    output, fallback = _normalize(prose)
    assert prose not in output.master_content
    assert all(prose not in variant.body for variant in output.platform_variants)
    assert any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    _assert_facts_unchanged(output, fallback)


@pytest.mark.parametrize("genre,kind", [
    ("", "opinion"), ("digest", "opinion"), ("company_data_brief", "opinion"),
    ("industry_analysis", "cta"), ("industry_analysis", "fact"), ("industry_analysis", "quote"),
])
def test_new_exceptions_cannot_be_requested_by_a_legacy_genre_or_another_block_kind(genre, kind):
    for prose in (SAFE[3], "审看更像一次有具体目标的核对。", SAFE[-1]):
        output, _ = _normalize(prose, genre=genre, kind=kind)
        assert prose not in (output.master_content or "")
        assert all(prose not in (variant.body or "") for variant in output.platform_variants)
        if kind in {"fact", "quote"}:
            assert output.critic.errors


def test_transition_retains_exact_author_text_but_not_an_independent_grade():
    output, fallback = _normalize(SAFE[0], kind="transition")
    assert SAFE[0] in output.master_content
    _assert_facts_unchanged(output, fallback)


def test_only_the_conditional_predicate_and_the_conceptual_measure_are_removed():
    prose = "若期待已经表达出来，平台已经宣布发布新功能。"
    matches = content._unsupported_editorial_assertion_matches(prose, allow_industry_creative_condition=True)
    assert [m.group() for m in matches] == ["已经", "宣布", "发布"]
    assert matches[0].start() == prose.index("已经", prose.index("已经") + 2)
    assert len(content._unsupported_editorial_assertion_matches(prose)) == 4
    prose = "审看更像一次有具体目标的核对，后续少做两次审片。"
    remaining = content._industry_editorial_number_matches(prose, (), allow_industry_analysis=True)
    assert [m.group() for m in remaining] == ["两次"]
    prose = "帮助读者判断：画面是否清楚。判断：请直接照做。"
    matches = content._generic_editorial_meta_matches(prose, allow_industry_reader_task=True)
    assert len(matches) == 1 and matches[0].start() == prose.rindex("判断：")
    assert len(content._generic_editorial_meta_matches(prose)) == 2
