"""Fictional prose must survive syntax only as independently reviewable analysis."""
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_analysis import DEFER_EDITORIAL_ANALYSIS
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api import test_semantic_editorial_analysis as semantic
from tests.ai_marketing_api.test_semantic_editorial_analysis import service
from tests.ai_marketing_api.test_industry_conditional_editorial import _canonical


STYLE = "团队应核对的不只是呈现，还包括输入是否稳定。"
NEGATED = "这种安排提供判断机会，而不是流程本身保证省钱。"
CONDITIONAL_STYLE = "值得核对的不只是画面能否达到预期，还有输入是否已经稳定。"


def _normalize(text, *, genre="industry_analysis", kind="opinion", defer=True):
    value = semantic._value()
    value["genre"] = genre
    request = content.OperatorComposeRequest.model_validate(value)
    registry = OperatorRegistry()
    contexts = registry.authorize("industry", request.authorized_context, request.as_of)
    fallback = content.build_content_fallback(registry, "industry", request, contexts)
    raw = semantic._candidate(content.content_request_payload("industry", request, contexts), prose=text)
    for blocks in [raw["blocks"], *[variant["blocks"] for variant in raw["platform_variants"]]]:
        for block in blocks:
            if block.get("text") == text:
                block["kind"] = kind
    token = DEFER_EDITORIAL_ANALYSIS.set(defer)
    try:
        output = content.normalize_content_output(registry, "industry", request, contexts, fallback, raw)
    finally:
        DEFER_EDITORIAL_ANALYSIS.reset(token)
    return output, fallback


def _facts(blocks):
    return _canonical([block.model_dump(mode="json") if hasattr(block, "model_dump") else block for block in blocks])


@pytest.mark.parametrize("text", [STYLE, CONDITIONAL_STYLE, NEGATED, "团队不能保证项目成功。"])
def test_newly_retained_expression_is_hidden_pending_not_a_supported_fact(text):
    output, fallback = _normalize(text)
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert len(output.pending_editorial_analysis) == 4
    for blocks in [output.blocks, *[variant.blocks for variant in output.platform_variants]]:
        authored = [block for block in blocks if block.text == text]
        assert len(authored) == 1 and authored[0].analysis_pending is True
        assert authored[0].evidence_ids == [] and not authored[0].source_exact
        assert _facts(blocks) == _facts(fallback["blocks"])
    assert not output.analysis_binding_reviews
    assert output.quality_scorecard is None
    if "不只是" in text:
        assert any(w.endswith(":contrast") and w.startswith("editorial_style_review_required:")
            for w in output.critic.warnings)


@pytest.mark.parametrize("text", [
    "流程本身保证省钱。", "流程必然成功。", "流程承诺效果。",
    "不是不能保证省钱。", "团队不能不保证省钱。", "并非没有保证省钱。",
    "而不是模式本身不保证省钱。", "团队不能保证不失败。",
    "团队不能保证省钱但保证成功。", "不保证这点而必然成功。",
    "听说“团队不能保证省钱”。", "难道团队不能保证省钱吗？",
    "团队不能保证省钱？", "大家说团队不能保证省钱。",
    "团队不能保证省钱；平台宣布发布新工具。",
    "团队不能保证省钱；平台已经支持自动制作。",
    "团队不能保证省钱；收费20元。", "团队不能保证省钱；收益增长20%。",
    "团队应核对的不只是呈现；平台宣布发布新工具。",
    "团队应核对的不只是呈现；平台已经支持自动制作。",
    "团队应核对的不只是呈现；平台支持自动制作。",
    "平台支持自动制作，而不是流程本身保证省钱。",
    "团队应核对的不只是呈现；我亲自完成了客户项目。",
    "以下分析：团队应核对的不只是呈现。",
    "编辑说明：团队不能保证省钱。",
    "团队考虑为正式制作前的一次画面判断支付费用。",
])
def test_pending_expression_cannot_exempt_other_assertions_or_ambiguous_negation(text):
    output, fallback = _normalize(text)
    assert not output.pending_editorial_analysis
    assert text not in output.master_content
    assert all(text not in variant.body for variant in output.platform_variants)
    assert any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    assert _facts(output.blocks) == _facts(fallback["blocks"])


@pytest.mark.parametrize("text", [STYLE, NEGATED])
@pytest.mark.parametrize("kwargs", [{"defer": False}, {"genre": "digest"}, {"kind": "cta"}])
def test_new_exceptions_do_not_extend_to_legacy_or_non_analysis_blocks(text, kwargs):
    output, _ = _normalize(text, **kwargs)
    assert not output.pending_editorial_analysis
    assert text not in output.master_content


@pytest.mark.parametrize("role", [content.OperatorRole.COMMERCIAL, content.OperatorRole.PERSONAL_IP])
def test_non_industry_role_keeps_original_promise_guard_even_under_industry_context(role):
    genre = content.CURRENT_GENRE.set("industry_analysis")
    defer = DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        blocks, errors, warnings = content._safe_candidate_blocks(role,
            [{"kind": "opinion", "text": NEGATED, "evidence_ids": []}], [], prefix="synthetic")
    finally:
        DEFER_EDITORIAL_ANALYSIS.reset(defer)
        content.CURRENT_GENRE.reset(genre)
    assert not blocks
    assert any(w.endswith(":assertion_forbidden") for w in warnings)


@pytest.mark.parametrize("text", [STYLE, NEGATED])
@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_real_staged_pipeline_requires_explicit_independent_result_for_retained_prose(service, monkeypatch, text, verdict):
    original_candidate = semantic._candidate
    monkeypatch.setattr(semantic, "PROSE", text)
    def candidate(body, *, prose=None, public=False):
        return original_candidate(body, prose=text if prose is None else prose, public=public)
    monkeypatch.setattr(semantic, "_candidate", candidate)
    value = semantic._value()
    merged = semantic._written(value)
    before = deepcopy(merged)
    service["verdict"] = verdict
    result = semantic._verify(value, merged)
    assert merged == before
    assert [stage for stage, _ in service["calls"]].count("claim_verification") == 1
    assert service["calls"][-1][1]["analysis_verification"]["candidates"]
    assert all(row["verdict"] == verdict and row["review_model"] == "synthetic-independent-verifier"
        for row in result["artifact"]["analysis_binding_reviews"])
    if verdict == "supported":
        assert result["status"] == "artifact"
        assert text in result["artifact"]["master_content"]
        assert all(text in variant["body"] for variant in result["artifact"]["platform_variants"])
        assert "pending_editorial_analysis" not in result["artifact"]
        assert result["artifact_signature"] == semantic.steps.artifact_signature(result["artifact"])
    else:
        assert result["status"] == "blocked"
        assert result["artifact"]["pending_editorial_analysis"]
        calls = len(service["calls"])
        reviewed = semantic._request("POST", semantic.BASE + "review_editorial", json={"compose": value,
            "candidate": result["artifact"], "candidate_signature": result["artifact_signature"]}).json()
        assert reviewed["status"] == "unavailable" and reviewed["error_code"] == "analysis_verification_required"
        assert len(service["calls"]) == calls
