"""Preserve R52's authored conclusions for verification, never for automatic publication."""
import asyncio
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content, operator_steps as steps
from ai_marketing_api.operator_analysis import DEFER_EDITORIAL_ANALYSIS
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose


ACTUAL_JUDGMENTS = [
    "对负责代码迁移验收的团队，我的判断是：让 Agent 通过测试，不等于已经决定迁移后的软件应该怎样运行。验收既要检查行为有没有偏离，也要确认被保留下来的行为是否值得保留。",
    "我的判断是：这些检查值得保留，但验收目标不能只由旧代码来决定。如果某项旧行为本身存疑，就需要先明确应有行为，再判断迁移结果；否则，检查越严格，也只是更严格地确认它符合原有做法。",
    "我的判断是：这些检查能帮助迁移负责人判断行为是否延续，却不能替他们决定原有行为是否值得保留。要做迁移验收，不能把「与原来一致」直接当成「符合需求」。",
]


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "offline-test-only")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-" + stage})


@pytest.mark.parametrize("text", ACTUAL_JUDGMENTS)
@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_actual_raw_conclusion_survives_ordered_blocks_but_requires_a_real_separate_decision(text, verdict):
    request = _compose("industry_analysis", "wechat_mp")
    registry = OperatorRegistry()
    calls = []

    async def offline(system, body, fallback, *_args, **kwargs):
        stage = kwargs["stage"]
        calls.append((stage, deepcopy(body)))
        if stage == "compose":
            refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"] if block["required"]]
            return {"_model": "offline-writer", "_llm_attempted": True,
                "master_title": "示例接口的判断", "blocks": [{"kind": "opinion", "text": text, "evidence_ids": []}, *refs],
                "platform_variants": []}
        return {"_model": "offline-independent-verifier", "_llm_attempted": True, "reviews": [],
            "analysis_reviews": [{"analysis_id": row["analysis_id"], "verdict": verdict,
                "reason": "Offline protocol fixture only; not evidence of actual editorial quality."}
                for row in body["analysis_verification"]["candidates"]]}

    async def run():
        drafted = await steps.run_editorial_step(registry, "industry", "draft_master",
            steps.EditorialStepRequest(compose=request), offline)
        assert drafted["status"] == "artifact", drafted
        draft = drafted["artifact"]
        assert draft["blocks"][0]["text"] == text and draft["blocks"][0]["analysis_pending"] is True
        assert text not in draft["master_content"]
        assert draft["pending_editorial_analysis"]
        verified = await steps.run_editorial_step(registry, "industry", "verify_claims",
            steps.EditorialStepRequest(compose=request, candidate=draft), offline)
        assert [stage for stage, _ in calls] == ["compose", "claim_verification"]
        assert calls[-1][1]["analysis_verification"]["candidates"][0]["text"] == text
        block = next(block for block in verified["artifact"]["blocks"] if block["text"] == text)
        assert verified["publishable"] is False
        if verdict == "supported":
            assert verified["status"] == "artifact"
            assert not block.get("analysis_pending") and text in verified["artifact"]["master_content"]
        else:
            assert verified["status"] == "blocked"
            assert block["analysis_pending"] is True and text not in verified["artifact"]["master_content"]
        assert verified["artifact"]["analysis_binding_reviews"][0]["verdict"] == verdict
    asyncio.run(run())


@pytest.mark.parametrize("text", [
    "我的判断是：我亲测过接口，所以应该直接采用。",
    "我的判断是：我在微软工作，了解这里的项目效果。",
    "我的判断是：我的团队和客户项目已经证明方案有效。",
    "我曾负责迁移项目，验收方式应当采用这条路径。",
    "我们测试了现有代码，发现它能保证迁移效果。",
])
def test_judgment_prefix_does_not_authorize_experience_identity_or_an_owner_story(text):
    request = _compose("industry_analysis", "wechat_mp")
    registry = OperatorRegistry()
    contexts = registry.authorize("industry", request.authorized_context, request.as_of)
    approved, errors = content._build_approved_blocks(content.OperatorRole.INDUSTRY, request, contexts)
    assert not errors and not content._industry_current_judgment(text)
    token = DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        with content.editorial_scope("industry_analysis"):
            blocks, errors, warnings = content._safe_candidate_blocks(content.OperatorRole.INDUSTRY,
                [{"kind": "opinion", "text": text, "evidence_ids": []}], approved, prefix="unsafe")
    finally:
        DEFER_EDITORIAL_ANALYSIS.reset(token)
    assert not errors and all(block.text != text for block in blocks)
    assert any(warning.startswith("discarded_unsafe_model_editorial:") for warning in warnings)


def test_nonstaged_industry_and_other_genres_keep_the_prior_firstperson_boundary():
    request = _compose("industry_analysis", "wechat_mp")
    registry = OperatorRegistry()
    approved, _ = content._build_approved_blocks(content.OperatorRole.INDUSTRY, request, request.authorized_context)
    text = "我的判断是：验收应先明确需求范围，避免拿行为一致代替需求判断。"
    for genre, deferred in [("industry_analysis", False), ("news", True)]:
        token = DEFER_EDITORIAL_ANALYSIS.set(deferred)
        try:
            with content.editorial_scope(genre):
                blocks, _, warnings = content._safe_candidate_blocks(content.OperatorRole.INDUSTRY,
                    [{"kind": "opinion", "text": text, "evidence_ids": []}], approved, prefix="legacy")
        finally:
            DEFER_EDITORIAL_ANALYSIS.reset(token)
        assert all(block.text != text for block in blocks)
        assert any("first_person_forbidden" in warning for warning in warnings)
