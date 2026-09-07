"""Upstream commercial safety follows terminal policy without rewriting facts."""

from __future__ import annotations

from copy import deepcopy

import pytest

import ai_marketing_api.main as api
from ai_marketing_api.operator_content import (
    OperatorComposeRequest,
    _DELIVERY_COMMITMENT_RE,
    build_content_fallback,
    content_request_payload,
    enforce_social_publishability,
    normalize_content_output,
)
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_operator_api import _request as http_request
from tests.ai_marketing_api.test_operator_content_runtime import (
    _claim,
    _commercial_focused_title_fixture,
    _context,
    _request,
)


CAPABILITY_PROSE = [
    "平台可以辅助整理创作素材和展示内容。",
    "系统能够整理创作素材并检查格式。",
    "支持丰富的创作功能，方便处理不同材料。",
    "The platform can provide content creation tools.",
    # Preserve the pre-existing stricter ownership branch as well.
    "公司拥有未经批准的处理能力。",
]
PRICE_PROSE = [
    "免费使用是否值得尝试，仍需核对实际需求。",
    "半价选项是否适合当前任务，仍需核对实际需求。",
    "报价可以面议，具体条件需要核对。",
    "The free option is worth a closer look for this task.",
]
READER_PROSE = [
    "材料整理和表达对象可以作为核对公开说明的起点。",
    "先想清楚要完成的内容，再看目标和边界。",
    "完成任务的前提，是先明确目标和材料范围。",
    "先完成内容整理，三天后的安排另行确认。",
]


def _commercial_case(text, extra_context=None):
    registry, request, authorized, fallback, _, _ = _commercial_focused_title_fixture(
        "AIID创作站适合怎样的创作任务"
    )
    if extra_context is not None:
        extra_text = extra_context.structured_data.get("public_text") or extra_context.content
        request = OperatorComposeRequest.model_validate({
            **request.model_dump(mode="json"),
            "authorized_context": [*request.authorized_context, extra_context],
            "claims": [*request.claims, _claim(extra_text, "fact", [extra_context.record_id])],
        })
        authorized = registry.authorize("commercial", request.authorized_context, request.as_of)
        fallback = build_content_fallback(registry, "commercial", request, authorized)
    payload = content_request_payload("commercial", request, authorized)
    required = [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"] if b["required"]]

    def blocks(label):
        return [
            {"kind": "transition", "text": f"先明确{label}的表达对象，再讨论工具选择的适用边界。", "evidence_ids": []},
            *deepcopy(required),
            {"kind": "opinion", "text": f"围绕{label}的素材准备和表达目标，逐项核对公开说明。", "evidence_ids": []},
        ]

    candidate = {
        "_model": "synthetic-commercial-parity",
        "master_title": "AIID创作站该怎样理解适用边界",
        "blocks": blocks("读者"),
        "platform_variants": [
            {"platform": "wechat_mp", "title": "AIID创作站的选择需要核对什么", "blocks": blocks("长文读者")},
            {"platform": "wechat_moments", "title": "AIID创作站适合怎样的创作任务", "blocks": blocks("熟人交流")},
            {"platform": "weitoutiao", "title": "AIID创作站怎样对应创作任务", "blocks": blocks("短文读者")},
        ],
    }
    candidate["platform_variants"][1]["blocks"].append({"kind": "opinion", "text": text, "evidence_ids": []})
    return registry, request, authorized, fallback, candidate


def _normalize(case):
    registry, request, authorized, fallback, candidate = case
    return normalize_content_output(registry, "commercial", request, authorized, fallback, candidate)


@pytest.mark.parametrize("text", CAPABILITY_PROSE + PRICE_PROSE + ["预计完成部署，具体安排请查看公开说明。"])
def test_unsupported_commercial_editorial_is_removed_before_ready_output(text):
    output = enforce_social_publishability(_normalize(_commercial_case(text)))
    assert output.status == "content_ready", output.critic
    assert text not in output.model_dump_json()
    assert any(w.startswith("discarded_unsafe_model_editorial:variant-wechat_moments-") for w in output.critic.warnings)
    assert all(b.source_exact and b.locked for b in output.blocks if b.origin == "approved_claim")


@pytest.mark.parametrize("text", READER_PROSE[:3])
def test_reader_task_language_is_not_a_delivery_commitment(text):
    output = enforce_social_publishability(_normalize(_commercial_case(text)))
    assert output.status == "content_ready", output.critic
    assert text in output.platform_variants[1].body
    assert output.critic.warnings == []


@pytest.mark.parametrize("text", READER_PROSE + [
    "先完成天然素材的分类，再讨论表达目标。",
    "内容上线后的读者反馈，仍需单独核对。",
])
def test_delivery_guard_does_not_confuse_content_or_preconditions_with_deadlines(text):
    assert _DELIVERY_COMMITMENT_RE.search(text) is None


@pytest.mark.parametrize("text", [
    "我们可以在三天内交付。",
    "项目将在明天上线。",
    "下周三完成部署。",
    "交付周期以约定为准。",
    "预计完成部署，具体安排请查看公开说明。",
    "完成时间为2026-09-15。",
    "The delivery is expected within 3 days.",
])
def test_delivery_guard_recognizes_real_commitments_in_either_order(text):
    assert _DELIVERY_COMMITMENT_RE.search(text) is not None


@pytest.mark.parametrize("role", ["commercial", "industry", "personal_ip"])
def test_non_numeric_delivery_commitment_is_rejected_for_every_editorial_role(role):
    space = {"commercial": "company_public", "industry": "industry", "personal_ip": "personal_approved"}[role]
    record_type = {"commercial": "capability", "industry": "source_item", "personal_ip": "opinion"}[role]
    context = _context("source-delivery-parity", role, space, record_type,
                       content="已批准资料仅介绍内容整理的任务边界。", source_tier="primary")
    kind = "opinion" if role == "personal_ip" else "fact"
    request = _request([context], [_claim(context.content, kind, [context.record_id],
                                        "opinion" if kind == "opinion" else "verified")], channels=["wechat_moments"])
    registry = OperatorRegistry()
    authorized = registry.authorize(role, request.authorized_context, request.as_of)
    fallback = build_content_fallback(registry, role, request, authorized)
    payload = content_request_payload(role, request, authorized)
    required = [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"] if b["required"]]
    text = "预计完成部署，具体安排请查看公开说明。"
    candidate = {"_model": "synthetic-delivery-role", "blocks": [*required, {"kind": "opinion", "text": text, "evidence_ids": []}]}
    output = normalize_content_output(registry, role, request, authorized, fallback, candidate)
    assert output.status == "content_ready", output.critic
    assert text not in output.model_dump_json()
    assert any(w.startswith("discarded_unsafe_model_editorial:master-") and w.endswith(":assertion_forbidden") for w in output.critic.warnings)


@pytest.mark.parametrize("text", [CAPABILITY_PROSE[0], "公开方案价格为 99 元，可在三天内交付。", "公开试用方案免费使用，适用范围以当前活动为准。"])
def test_authorized_capability_and_active_offer_remain_exact_canonical_blocks(text):
    is_offer = text not in CAPABILITY_PROSE
    context = _context(
        "canonical-parity-source", "commercial", "company_public", "business_offer" if is_offer else "capability",
        content="Private offer field must stay absent." if is_offer else text,
        structured_data={"publicly_quoteable": True, "public_text": text} if is_offer else {},
    )
    output = enforce_social_publishability(_normalize(_commercial_case(READER_PROSE[0], context)))
    assert output.status == "content_ready", output.critic
    for blocks in [output.blocks, *[v.blocks for v in output.platform_variants]]:
        matching = [b for b in blocks if b.text == text]
        assert len(matching) == 1
        assert matching[0].source_exact and matching[0].locked and matching[0].origin == "approved_claim"
        assert matching[0].evidence_ids == [context.record_id]
    assert "Private offer field" not in output.model_dump_json()


@pytest.mark.parametrize("text", [PRICE_PROSE[0], PRICE_PROSE[1], PRICE_PROSE[2]])
def test_price_fact_without_an_offer_still_fails_closed(text):
    context = _context("unbound-price", "commercial", "company_public", "capability", content=text)
    output = _normalize(_commercial_case(READER_PROSE[0], context))
    assert output.status == "needs_input"
    assert "unique_public_business_offer_required" in output.critic.errors
    assert output.master_content is None and output.platform_variants == []


def test_industry_editorial_retains_its_existing_non_numeric_price_exception():
    context = _context("industry-price-exception", "industry", "industry", "source_item",
                       content="供应商发布了新的内容整理工具。", source_tier="official")
    request = _request([context], [_claim(context.content, "fact", [context.record_id])], channels=["wechat_moments"])
    registry = OperatorRegistry()
    authorized = registry.authorize("industry", [context], request.as_of)
    fallback = build_content_fallback(registry, "industry", request, authorized)
    payload = content_request_payload("industry", request, authorized)
    text = PRICE_PROSE[0]
    candidate = {"_model": "synthetic-industry-exception", "blocks": [
        *[{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"] if b["required"]],
        {"kind": "opinion", "text": text, "evidence_ids": []},
    ]}
    output = normalize_content_output(registry, "industry", request, authorized, fallback, candidate)
    assert output.status == "content_ready"
    assert text in output.master_content


@pytest.mark.parametrize("text", [CAPABILITY_PROSE[0], CAPABILITY_PROSE[3], PRICE_PROSE[0], "预计完成部署，具体安排请查看公开说明。"])
def test_http_compose_never_returns_the_unsupported_editorial_sentence(monkeypatch, text):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    _, request, _, _, candidate = _commercial_case(text)
    stages = []

    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" in body:
            stages.append("review")
            assert text not in str(body["editorial_review"]["draft"])
            return {"_model": "synthetic-independent-editor", "relevant_to_brief": True,
                    "useful_to_reader": True, "clear_and_specific": True, "platform_fit": True,
                    "issues": [], "failed_surfaces": []}
        stages.append("compose")
        return deepcopy(candidate)

    monkeypatch.setattr(api, "_llm_json", fake)
    data = http_request("POST", "/api/v1/operators/commercial/compose", json=request.model_dump(mode="json")).json()
    assert data["status"] == "content_ready", data
    assert stages == ["compose", "review"]
    assert text not in str(data)
    assert data["requires_human_review"] is True
