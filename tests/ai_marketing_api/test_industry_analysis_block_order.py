"""Evidence-first industry analysis keeps its model-authored reading order."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_content import OperatorComposeRequest
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_operator_api import _base_payload


FACTS = [
    "示例接口要求先定义判断问题和允许的答案。",
    "示例接口接收文本形式的判断上下文。",
    "示例接口文档列出 400 项请求上限。",
]
ANALYSIS = [
    {"kind": "opinion", "text": "围绕上述问题设计要求，可用稿件资料审核作假设：缺少来源时先补材料，材料齐备后再由人工讨论读者价值。", "evidence_ids": []},
    {"kind": "transition", "text": "这个流程的取舍在于审核责任：答案范围适合预先约定时，先澄清进入条件；探索性的内容创作仍应保留展开空间。", "evidence_ids": []},
]


def _compose(genre: str, surface: str) -> OperatorComposeRequest:
    value = _base_payload()
    value["channels"] = [surface]
    value["genre"] = genre
    value["authorized_context"][0]["content"] = "\n".join(FACTS)
    value["approved_proposal"] = {
        "title": "示例接口的判断任务如何设计",
        "angle": "Explain a hypothetical editorial intake decision.",
        "audience_value": "A concrete boundary between evidence intake and editorial quality.",
        "key_points": FACTS,
        "suggested_formats": [surface],
        "first_person": False,
    }
    value["claims"] = [
        {"text": text, "kind": "fact", "evidence_ids": ["source-1"], "verification_status": "verified"}
        for text in FACTS
    ]
    return OperatorComposeRequest.model_validate(value)


def _canonical(blocks):
    fields = ("text", "claim_id", "evidence_ids", "verification_status", "source_exact", "origin", "locked", "required", "binding_hash")
    return [{key: block[key] for key in fields} for block in blocks if block["source_exact"]]


@pytest.mark.parametrize("genre", ["industry_analysis", "", "digest", "company_data_brief"])
@pytest.mark.parametrize("surface", ["wechat_mp", "wechat_moments", "weitoutiao"])
def test_real_staged_pipeline_preserves_analysis_order_and_legacy_layout(monkeypatch, genre, surface):
    monkeypatch.setenv("HERMES_API_KEY", "offline-order-test-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-writer"})
    request = _compose(genre, surface)
    calls = []
    expected_facts = []

    async def writer(system, body, fallback, **kwargs):
        calls.append(body)
        facts = [block for block in body["canonical_block_registry"] if block["required"]]
        expected_facts[:] = _canonical(facts)
        refs = [{"block_ref": block["block_ref"]} for block in facts]
        # The model deliberately keeps a sourced lead, then interprets that
        # evidence before moving to the remaining facts and closing judgment.
        ordered = [refs[0], deepcopy(ANALYSIS[0]), *refs[1:], deepcopy(ANALYSIS[1])]
        return {
            "_model": "offline-writer",
            "master_title": "示例接口的判断任务如何设计",
            "blocks": ordered,
            "platform_variants": [
                {"platform": channel, "title": "示例接口与稿件资料审核边界", "blocks": deepcopy(ordered)}
                for channel in body["channels"]
            ],
        }

    async def run():
        registry = OperatorRegistry()
        master = await steps.run_editorial_step(registry, "industry", "draft_master", steps.EditorialStepRequest(compose=request), writer)
        assert master["status"] == "artifact", master
        adapted = await steps.run_editorial_step(registry, "industry", "adapt_platform", steps.EditorialStepRequest(
            compose=request, surface=surface, candidate=master["artifact"], candidate_signature=master["artifact_signature"]), writer)
        assert adapted["status"] == "artifact", adapted
        merged = deepcopy(master["artifact"])
        merged["platform_variants"] = adapted["artifact"]["platform_variants"]
        verified = await steps.run_editorial_step(registry, "industry", "verify_claims", steps.EditorialStepRequest(compose=request, candidate=merged), writer)
        assert verified["status"] == "artifact", verified
        return master, adapted, verified

    master, adapted, verified = asyncio.run(run())
    assert len(calls) == 2  # writer + adaptation; exact references need no verifier HTTP
    master_blocks = master["artifact"]["blocks"]
    platform_blocks = adapted["artifact"]["platform_variants"][0]["blocks"]
    expected_order = ([FACTS[0], ANALYSIS[0]["text"], *FACTS[1:], ANALYSIS[1]["text"]]
                      if genre == "industry_analysis"
                      else [ANALYSIS[0]["text"], *FACTS, ANALYSIS[1]["text"]])
    for blocks in (master_blocks, platform_blocks, verified["artifact"]["blocks"], verified["artifact"]["platform_variants"][0]["blocks"]):
        assert [block["text"] for block in blocks] == expected_order
        assert _canonical(blocks) == expected_facts
    for result in (master, adapted, verified):
        assert not result["artifact"]["critic"]["errors"]
        # Revalidating saved server artifacts records harmless canonical echoes;
        # no authored analysis may be discarded or required fact restored.
        assert all(warning.startswith("model_canonical_echo_normalized:") for warning in result["artifact"]["critic"]["warnings"])
        assert result["publishable"] is False
        assert result["generation_trace"]["model_calls"] <= 1
    if genre == "industry_analysis":
        assert master["artifact"]["master_content"].startswith(FACTS[0] + "\n\n" + ANALYSIS[0]["text"])
        assert adapted["artifact"]["platform_variants"][0]["body"].startswith(FACTS[0] + "\n\n" + ANALYSIS[0]["text"])
