"""Concrete conditional analysis cannot authorize a surrounding event claim."""
from __future__ import annotations

from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose


# Exact model-authored paragraphs from the owner-authorized round-twelve
# diagnosis. No provider envelope, private signatures or third-party article.
ROUND12_DESIGN = (
    "可以假设这样一个工作流：创作工具把稿件正文、来源材料、授权范围与业务规则交给决策链路，"
    "约定的结果可能是「补资料」「进入人工审读」或「暂缓」。"
    "这些标签是业务规则，不是已经实现的功能或测试结果。"
)
ROUND12_TRADEOFF = (
    "标签为什么要先定义？因为机制要求答案范围有界。缺少来源时由谁补材料？"
    "如果创作者还在探索叙事角度，固定答案会过早压缩创作空间；"
    "如果资料已经齐备但论证和读者价值仍需独立审读，那么「资料齐备」只表示满足进入条件，"
    "不等于可以直接放行。"
)
SAFE = [
    ROUND12_DESIGN,
    ROUND12_TRADEOFF,
    "若材料已经齐全，先核对来源与授权，再讨论稿件的读者价值。",
    "当资料已经备齐时，是否进入人工审读仍需按预先约定的边界判断。",
    "这只是流程设计，并非已经完成的测试结果。",
    "如果材料不足，先补来源；如果材料齐备，再由审读者讨论表达取舍。",
]
UNSAFE = [
    "如果资料已经齐备，某公司宣布发布新的决策工具。",
    "如果资料已经齐备，某团队获得收益。",
    "如果资料已经齐备，某公司已经上线新的功能。",
    "如果资料已经齐备，我用它完成了客户项目。",
    "如果资料已经齐备，平台已经支持自动生成视频。",
    "如果资料已经齐备，平台承诺保证通过审核。",
    "如果资料已经齐备，可联系客户张先生的私人手机13912345678。",
    "如果资料已经齐备，审核准确率达到99%。",
    "这些标签是业务规则，不是已经实现的功能或测试结果。某公司宣布正式推出它。",
    "这些标签是业务规则，不是已经实现的功能或测试结果；工具已经实现自动审读。",
    "如果公司已经完成融资，可以考虑后续安排。",
    "如果资料已经发布，读者可以查看相关说明。",
    "资料已经齐备，创作者可以进入人工审读。",
    "并不是不是已经实现的功能，读者可以直接调用它。",
]


def _normalize(text: str, *, genre="industry_analysis"):
    request = _compose(genre, "wechat_mp").model_copy(update={
        "channels": ["wechat_mp", "wechat_moments", "weitoutiao"],
    })
    registry = OperatorRegistry()
    authorized = registry.authorize("industry", request.authorized_context, request.as_of)
    fallback = content.build_content_fallback(registry, "industry", request, authorized)
    body = content.content_request_payload("industry", request, authorized)
    refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"] if block["required"]]
    blocks = [*refs, {"kind": "opinion", "text": text, "evidence_ids": []}]
    raw = {"_model": "offline-industry-writer", "master_title": request.approved_proposal.title,
        "blocks": deepcopy(blocks), "platform_variants": [
            {"platform": channel, "title": request.approved_proposal.title, "blocks": deepcopy(blocks)}
            for channel in request.channels]}
    output = content.normalize_content_output(registry, "industry", request, authorized, fallback, raw)
    return output, fallback


def _canonical(blocks):
    fields = ("text", "claim_id", "evidence_ids", "verification_status", "source_exact", "origin", "locked", "required", "binding_hash")
    return [{field: block[field] for field in fields} for block in blocks if block["source_exact"]]


@pytest.mark.parametrize("text", SAFE)
def test_concrete_conditions_and_explicit_non_results_survive_all_four_surfaces(text):
    output, fallback = _normalize(text)
    assert text in output.master_content
    assert len(output.platform_variants) == 3
    assert all(text in variant.body for variant in output.platform_variants)
    assert not output.critic.errors
    assert not any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    expected = _canonical(fallback["blocks"])
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        assert _canonical([block.model_dump(mode="json") for block in blocks]) == expected
        authored = [block for block in blocks if block.text == text]
        assert len(authored) == 1
        assert authored[0].evidence_ids == []
        assert not authored[0].source_exact
    # Passing the syntax/safety fence does not fabricate an independent card.
    assert output.quality_scorecard is None


@pytest.mark.parametrize("text", UNSAFE)
def test_one_safe_already_span_does_not_exempt_other_assertions_or_safety_gates(text):
    output, fallback = _normalize(text)
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    expected = _canonical(fallback["blocks"])
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        assert _canonical([block.model_dump(mode="json") for block in blocks]) == expected


@pytest.mark.parametrize("text", [
    "这不是一个数字，而是行业规则的真正落点。",
    "这并非表面结果，而是业务模式的核心影响。",
    "这未必是结果，更是规则变化的信号。",
])
def test_paired_generic_contrast_remains_forbidden(text):
    output, _ = _normalize(text)
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert any(w.endswith(":generic_meta_copy_forbidden") for w in output.critic.warnings)


@pytest.mark.parametrize("genre", ["", "digest", "company_data_brief"])
def test_already_exception_does_not_extend_to_legacy_genres(genre):
    output, _ = _normalize(ROUND12_DESIGN, genre=genre)
    assert ROUND12_DESIGN not in output.master_content
    assert all(ROUND12_DESIGN not in variant.body for variant in output.platform_variants)
    assert any(w.endswith(":assertion_forbidden") for w in output.critic.warnings)


def test_exception_tracks_exact_token_spans_and_scans_remaining_matches():
    text = "如果资料已经齐备，团队已经宣布发布功能。"
    remaining = content._unsupported_editorial_assertion_matches(text, allow_industry_hypothetical=True)
    assert [match.group(0) for match in remaining] == ["已经", "宣布", "发布"]
    assert remaining[0].start() == text.index("已经", text.index("已经") + 2)
    assert [match.group(0) for match in content._unsupported_editorial_assertion_matches(text)] == ["已经", "已经", "宣布", "发布"]
