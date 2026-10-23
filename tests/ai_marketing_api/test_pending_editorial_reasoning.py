"""Local uncertainty is not a reported result; other assertion spans remain blocked."""
import pytest

from ai_marketing_api.operator_content import (_build_approved_blocks, _safe_candidate_blocks,
    _unsupported_editorial_assertion_matches, content_request_payload)
from ai_marketing_api.operator_runtime import OperatorRegistry, OperatorRole
from ai_marketing_api.operator_style import editorial_scope
from tests.ai_marketing_api.test_operator_content_runtime import _claim, _context, _request, _run


ACTUAL_ACCEPTANCE = "因此，样片确认通过，只说明方案符合预期；画面最终是否达到制作要求，仍要看生成的成片。若评审目标是构图、场景关系、主体动作或提示词意图，样片有判断价值；若目标是验收成片画质，就不该用低分辨率预览替代成片检查。"
ACTUAL_UNPROVEN_COST = "这也划清了样片的成本边界：它可以帮助团队先确认方案，但不能据此推断总成本会下降；是否值得加做样片，还要结合预期能避免的返工，以及人工评审投入来判断。"


def matches(text, enabled=True):
    return _unsupported_editorial_assertion_matches(text, allow_industry_hypothetical=True,
        allow_industry_direction_condition=True, allow_industry_pending_reasoning=enabled)


@pytest.mark.parametrize("text", [ACTUAL_ACCEPTANCE, ACTUAL_UNPROVEN_COST,
    "输出最终是否达到验收标准，仍需检查成片。", "无法据此证明制作成本会下降。",
    "不足以直接认定产出质量达到目标。", "不能说明收益增长。"])
def test_only_explicit_unproven_or_pending_result_predicate_is_exempt(text):
    assert matches(text) == []
    assert matches(text, enabled=False)


@pytest.mark.parametrize("text", [
    "并非不能据此推断总成本会下降。", "不能不据此推断总成本会下降。",
    "无法证明总成本没有下降。", "不能说明成本不会下降。", "不足以说明并非成本会下降。",
    "不能据此推断总成本会下降但收益增长。", "不能据此推断总成本下降且画质达到目标。",
    "没有更便宜就说明总成本下降。", "画面最终是否已经达到制作要求，仍要看生成的成片。",
    "并非画面最终是否达到制作要求，仍要看生成的成片。", "画面最终是否达到制作要求，但产品达到标准。",
    "是否达到制作要求？产品已经达成合作。", "不能据此认定产品已经发布。",
])
def test_double_negation_comparative_inference_or_positive_result_is_not_exempt(text):
    assert matches(text)


@pytest.mark.parametrize("suffix,remaining", [("产品已经发布。", ["已经", "发布"]),
    ("总成本下降。", ["下降"]), ("结果达到目标并获得收益。", ["达到", "获得"])])
@pytest.mark.parametrize("prefix", [ACTUAL_ACCEPTANCE, ACTUAL_UNPROVEN_COST])
def test_each_other_assertion_span_survives_in_same_paragraph(prefix, suffix, remaining):
    assert [m.group(0) for m in matches(prefix + suffix)] == remaining


def evidence():
    facts = ["Seedance 2.5 官方将低分辨率样片用于验证场景结构、镜头调度、主体动作和提示词意图等要素；确认符合预期后，再基于样片任务 ID 生成高质量成片。",
        "Seedance 2.5 官方文档规定，Draft 样片任务和基于样片生成正式视频的任务独立计费。",
        "基于 Seedance 2.5 的 Draft 生成正式视频时，提示词、参考素材、时长、宽高比、随机种子和音频设置由模型自动复用；这些参数不能重新传入，即使与样片阶段取值相同也会报错。"]
    contexts = [_context(f"source-{i}", "industry", "industry", "source_item", content=t, source_tier="official") for i, t in enumerate(facts, 1)]
    request = _request(contexts, [_claim(t, "fact", [f"source-{i}"]) for i, t in enumerate(facts, 1)], channels=["wechat_mp"])
    request.genre = "industry_analysis"
    return request, contexts


def test_real_normalizer_keeps_actual_two_paragraphs_and_all_immutable_evidence():
    request, contexts = evidence()
    registry = OperatorRegistry()
    registry.authorize(OperatorRole.INDUSTRY, request.authorized_context, request.as_of)
    payload = content_request_payload(OperatorRole.INDUSTRY, request, contexts)
    refs = [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"]]
    blocks = [*refs, {"kind": "opinion", "text": ACTUAL_ACCEPTANCE, "evidence_ids": []},
        {"kind": "transition", "text": ACTUAL_UNPROVEN_COST, "evidence_ids": []}]
    output = _run(OperatorRole.INDUSTRY, request, {"_model": "offline-protocol-fixture", "master_title": "Seedance 2.5 的样片与成片验收",
        "blocks": blocks, "platform_variants": [{"platform": "wechat_mp", "title": "Seedance 2.5 怎样区分方案与画质验收", "blocks": blocks}]})
    assert not output.critic.errors and not output.critic.warnings
    assert ACTUAL_ACCEPTANCE in output.master_content and ACTUAL_UNPROVEN_COST in output.master_content
    assert ACTUAL_ACCEPTANCE in output.platform_variants[0].body and ACTUAL_UNPROVEN_COST in output.platform_variants[0].body
    for rows in (output.blocks, output.platform_variants[0].blocks):
        actual = [(b.claim_id, b.text, b.evidence_ids, b.required, b.locked) for b in rows if b.claim_id]
        assert actual == [(f"claim-{i}", c.text, c.evidence_ids, True, True) for i, c in enumerate(request.claims, 1)]
    assert output.quality_scorecard is None and output.editorial_assessment is None


@pytest.mark.parametrize("role,genre,kind", [(OperatorRole.COMMERCIAL, "industry_analysis", "opinion"),
    (OperatorRole.PERSONAL_IP, "industry_analysis", "opinion"), (OperatorRole.INDUSTRY, "", "opinion"),
    (OperatorRole.INDUSTRY, "news_digest", "transition"), (OperatorRole.INDUSTRY, "industry_analysis", "cta")])
def test_exception_does_not_reach_other_roles_genres_or_cta(role, genre, kind):
    request, contexts = evidence()
    approved, errors = _build_approved_blocks(OperatorRole.INDUSTRY, request, contexts)
    assert not errors
    with editorial_scope(genre):
        selected, _, warnings = _safe_candidate_blocks(role,
            [{"kind": kind, "text": ACTUAL_UNPROVEN_COST, "evidence_ids": []}], approved, prefix="master")
    assert ACTUAL_UNPROVEN_COST not in [b.text for b in selected]
    assert any("assertion_forbidden" in warning for warning in warnings)


@pytest.mark.parametrize("suffix,reason", [("实际节省20%。", "number_ungrounded"),
    ("保证有效。", "assertion_forbidden"), ("预计明天完成交付。", "assertion_forbidden"),
    ("我测试过这款产品。", "first_person_forbidden")])
def test_other_number_promise_delivery_and_personal_guards_remain_active(suffix, reason):
    request, contexts = evidence()
    approved, _ = _build_approved_blocks(OperatorRole.INDUSTRY, request, contexts)
    with editorial_scope(request.genre):
        selected, _, warnings = _safe_candidate_blocks(OperatorRole.INDUSTRY,
            [{"kind": "opinion", "text": ACTUAL_UNPROVEN_COST + suffix, "evidence_ids": []}], approved, prefix="master")
    assert ACTUAL_UNPROVEN_COST + suffix not in [b.text for b in selected]
    assert any(reason in warning for warning in warnings)


def test_bare_cheaper_comparison_is_a_preexisting_limit_not_a_new_exemption():
    # Do not claim that the event-word regex is a semantic comparison verifier.
    assert matches("没有更便宜。") == matches("没有更便宜。", enabled=False) == []
