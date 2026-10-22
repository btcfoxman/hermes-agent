"""Commercial hypotheses stay reader-side; product assertions remain exact facts."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_content import (
    OperatorComposeRequest, _COMMERCIAL_ASSERTION_RE, build_content_fallback,
    content_request_payload, enforce_social_publishability, normalize_content_output,
)
from tests.ai_marketing_api.test_operator_content_runtime import _commercial_focused_title_fixture
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import payload as industry_payload


# Exact authored paragraph retained from the owner-authorized round-nine
# diagnostic. No provider envelope, signature, credentials or source text.
ROUND9_VERIFICATION = (
    "「稳定、性价比高、模型全面」是产品方的定位表述，不是第三方测评结论。"
    "开发者可以围绕自己的实际需求去验证：需要的模型是否在列、调用延迟和并发是否满足场景要求、"
    "计费方式是否透明可预估。这些问题在站内文档和实际测试中才能得到答案。"
)
SAFE_DESIGNS = [
    ROUND9_VERIFICATION,
    "目标读者若服务于这些团队，受众重合可以作为进一步核对的起点。",
    "目标读者若服务于这些团队，受众重合可以作为第一道筛选。",
    "可先核对模型覆盖、接入方式与你现有架构是否兼容、价格与服务条款是否清楚。",
    "产品定位只是待核对的表述，开发者可以先核对需求，再选择验证方式。",
    "产品定位只是待核对的表述，实际的目标开发者可以先核对需求，再选择验证方式。",
    "The product positioning is a vendor statement, developers can verify their own requirements.",
    "假设客户要整理手头的材料，可先说明输入范围和希望交给人工讨论的输出，再检查缺少哪些依据。",
    "如果目标和材料范围仍不清楚，先澄清需求；进入工具评估后，再核对当前说明是否足以回答这些需求。具体实现和结果仍需验证。",
]
UNAPPROVED_ASSERTIONS = [
    "我们的产品，支持全部模型。",
    "本服务，在企业场景下，能够保证稳定性。",
    "平台，在条件满足时，可以处理视频任务。",
    "系统只要完成配置，就能够整理创作素材。",
    "平台并不支持视频导出。",
    "即使不确定，平台也可能能够生成视频。",
    "如果平台支持视频导出，就能改变内容流程。",
    "The platform can provide a video generation service.",
    "该产品，免费使用，具体费用另议。",
    "系统能够降低延迟，带来更好的稳定性。",
    "开发者可以验证系统的转码能力。",
    "支持图文、视频创作功能。",
    "公司拥有额外处理能力。",
    "我们的产品，通过统一接口，支持多种模型。",
    "Our product, under load, can support all models.",
    "The platform, when configured, might not support video generation.",
    "我们的产品：支持全部模型。",
    "我们的产品，可支持全部模型。",
    "Our product: can support all models.",
    "Our product, does not support video generation.",
    "产品，专为开发者设计，可以处理视频任务。",
    "本平台，低成本地提供视频生成。",
    "产品，面向专业用户，经过开发者测试，可以处理视频任务。",
    "The product, built for creators, can generate video.",
    "The product, designed for developers, under load, can support all models.",
    "产品，开发者可以用它生成视频。",
    "平台，用户可以通过它自动生成视频。",
    "The platform, developers can use it to generate video.",
    "产品，开发者可以生成视频。",
    "产品，开发者可以先生成视频，再验证输出。",
    "The platform, developers can use it to generate video and then verify the output.",
    "平台，用户生成视频后可以核对输出。",
    "The product, users generate art and can check the output.",
    "产品，开发者可以核对材料并用它生成视频。",
    "平台，用户可以先检查材料，再通过它生成视频。",
    "产品，开发者可以核对材料并借助该产品整理材料。",
    "The platform, developers can check material and use it to generate video.",
]


def make_request(genre="use_case"):
    registry, old, _, _, _, _ = _commercial_focused_title_fixture("AIID创作站的任务选择边界")
    request = old.model_copy(update={"genre":genre,
        "editorial_feedback":"按实际读者任务解释假设的输入、期望输出和人工检查条件；不补充产品能力。"})
    authorized = registry.authorize("commercial",request.authorized_context,request.as_of)
    fallback = build_content_fallback(registry,"commercial",request,authorized)
    return registry,request,authorized,fallback


def model_candidate(body, text):
    refs=[{"block_ref":block["block_ref"]} for block in body["canonical_block_registry"] if block["required"]]
    def blocks(label):
        return [{"kind":"transition","text":f"先明确{label}的表达对象，再讨论工具选择的适用边界。","evidence_ids":[]},
            *deepcopy(refs),{"kind":"opinion","text":f"围绕{label}的素材准备和表达目标，逐项核对公开说明。","evidence_ids":[]},
            {"kind":"opinion","text":text,"evidence_ids":[]}]
    titles={"wechat_mp":"AIID创作站的选择需要核对什么","wechat_moments":"AIID创作站适合怎样的创作任务",
            "weitoutiao":"AIID创作站怎样对应创作任务"}
    return {"_model":"offline-commercial-writer","master_title":"AIID创作站该怎样理解适用边界","blocks":blocks("读者"),
        "platform_variants":[{"platform":channel,"title":titles[channel],"blocks":blocks(channel)} for channel in body["channels"]]}


def normalize(text,genre="use_case"):
    registry,request,authorized,fallback=make_request(genre)
    body=content_request_payload("commercial",request,authorized)
    output=normalize_content_output(registry,"commercial",request,authorized,fallback,model_candidate(body,text))
    return enforce_social_publishability(output),fallback


def canonical(blocks):
    fields=("text","claim_id","evidence_ids","verification_status","source_exact","origin","locked","required","binding_hash")
    return [{field:block[field] for field in fields} for block in blocks if block["source_exact"]]


@pytest.mark.parametrize("text",SAFE_DESIGNS)
def test_real_normalizer_keeps_verification_and_hypothetical_customer_design(text):
    output,fallback=normalize(text)
    assert output.status=="content_ready",output.critic
    assert text in output.master_content
    assert all(text in variant.body for variant in output.platform_variants)
    assert not any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    expected=canonical(fallback["blocks"])
    assert len(expected)==2
    for blocks in [output.blocks,*[variant.blocks for variant in output.platform_variants]]:
        assert canonical([block.model_dump(mode="json") for block in blocks])==expected


@pytest.mark.parametrize("text",UNAPPROVED_ASSERTIONS)
def test_hypothesis_contract_does_not_authorize_product_claims_or_negated_capabilities(text):
    output,fallback=normalize(text)
    assert text not in output.model_dump_json()
    assert any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    assert canonical([block.model_dump(mode="json") for block in output.blocks])==canonical(fallback["blocks"])


@pytest.mark.parametrize("text",[
    "我们的产品，支持全部模型。",
    "平台，在条件满足时，可以处理视频任务。",
    "本服务，在企业场景下，能够保证稳定性。",
    "平台未必能够处理复杂材料。",
    "支持图文、视频创作功能。",
    "我们的产品，通过统一接口，支持多种模型。",
    "Our product, under load, can support all models.",
    "产品，专为开发者设计，可以处理视频任务。",
    "本平台，低成本地提供视频生成。",
    "The product, built for creators, can generate video.",
])
def test_explicit_product_subject_continues_through_qualifier_commas_without_negation_exemption(text):
    assert _COMMERCIAL_ASSERTION_RE.search(text) is not None


def test_use_case_prompt_honors_brief_feedback_and_has_guard_compatible_examples():
    registry,request,authorized,fallback=make_request()
    body=content_request_payload("commercial",request,authorized)
    assert body["editorial_feedback"]==request.editorial_feedback
    assert body["public_editorial_brief"]==request.public_editorial_brief.model_dump(mode="json")
    contract=body["commercial_use_case_contract"]
    assert "hypothetical" in contract["permission"] and "public_editorial_brief" in contract["permission"]
    assert "interface steps" in contract["boundary"] and "performance" in contract["boundary"]
    assert "generic selection slogan" in body["evidence_scope_contract"][2]
    for example in contract["examples"]:
        assert example["kind"] in {"opinion","transition"} and example["evidence_ids"]==[]
        assert "claim_id" not in example and "block_ref" not in example
        result=normalize_content_output(registry,"commercial",request,authorized,fallback,model_candidate(body,example["text"]))
        assert example["text"] in result.master_content


@pytest.mark.parametrize("genre",["","digest","company_data_brief"])
def test_other_commercial_genres_keep_existing_prompt_scope(genre):
    registry,request,authorized,_=make_request(genre)
    body=content_request_payload("commercial",request,authorized)
    assert "commercial_use_case_contract" not in body
    assert "a practical choice judgment is enough" in body["evidence_scope_contract"][2]


def test_use_case_override_is_not_applied_to_another_role():
    value=industry_payload()
    value["genre"]="use_case"
    request=OperatorComposeRequest.model_validate(value)
    from ai_marketing_api.operator_runtime import OperatorRegistry
    registry=OperatorRegistry()
    contexts=registry.authorize("industry",request.authorized_context,request.as_of)
    body=content_request_payload("industry",request,contexts)
    assert "commercial_use_case_contract" not in body


def test_real_staged_master_adapt_verify_preserves_canonical_and_hypothesis(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY","offline-commercial-case-secret")
    monkeypatch.setattr(steps,"server_stage_route",lambda stage:{"model":"offline-commercial-writer"})
    registry,request,_,_=make_request()
    calls=[]
    async def writer(system,body,fallback,**kwargs):
        calls.append(body)
        assert body["editorial_feedback"]==request.editorial_feedback
        assert "commercial_use_case_contract" in body
        return model_candidate(body,ROUND9_VERIFICATION)
    async def run():
        master=await steps.run_editorial_step(registry,"commercial","draft_master",steps.EditorialStepRequest(compose=request),writer)
        adapted=await steps.run_editorial_step(registry,"commercial","adapt_platform",steps.EditorialStepRequest(
            compose=request,surface="wechat_mp",candidate=master["artifact"],candidate_signature=master["artifact_signature"]),writer)
        merged=deepcopy(master["artifact"])
        # The complete original scope is preserved for verification.
        merged["platform_variants"]=[]
        for surface in request.channels:
            result=adapted if surface=="wechat_mp" else await steps.run_editorial_step(registry,"commercial","adapt_platform",
                steps.EditorialStepRequest(compose=request,surface=surface,candidate=master["artifact"],candidate_signature=master["artifact_signature"]),writer)
            merged["platform_variants"].extend(result["artifact"]["platform_variants"])
        verified=await steps.run_editorial_step(registry,"commercial","verify_claims",steps.EditorialStepRequest(compose=request,candidate=merged),writer)
        return master,adapted,verified
    master,adapted,verified=asyncio.run(run())
    assert len(calls)==4
    for result in (master,adapted,verified):
        assert result["status"]=="artifact" and result["publishable"] is False
        assert not result["artifact"]["critic"]["errors"]
        assert result["generation_trace"]["model_calls"]<=1
    assert ROUND9_VERIFICATION in master["artifact"]["master_content"]
    assert ROUND9_VERIFICATION in adapted["artifact"]["platform_variants"][0]["body"]
    expected=canonical(master["artifact"]["blocks"])
    assert len(expected)==2
    assert canonical(verified["artifact"]["blocks"])==expected
    assert all(canonical(v["blocks"])==expected for v in verified["artifact"]["platform_variants"])
