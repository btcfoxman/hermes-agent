"""Exact analysis framing spans cannot authorize surrounding quantities or facts."""
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose
from tests.ai_marketing_api.test_industry_conditional_editorial import _canonical


def _normalize(text, *, genre="industry_analysis", kind="opinion", source_brand="Seedance 2.5"):
    # Synthetic source data; not a claim about the named model's real behavior.
    value=_compose(genre,"wechat_mp").model_dump(mode="json")
    facts=[claim["text"] for claim in value["claims"]]
    facts[0]=source_brand+" 示例文档要求先定义判断问题和允许的答案。"
    value["authorized_context"][0]["content"]="\n".join(facts)
    for claim,text_value in zip(value["claims"],facts): claim["text"]=text_value
    value["approved_proposal"]["key_points"]=facts
    value["channels"]=["wechat_mp","wechat_moments","weitoutiao"]
    request=content.OperatorComposeRequest.model_validate(value)
    registry=OperatorRegistry(); contexts=registry.authorize("industry",request.authorized_context,request.as_of)
    fallback=content.build_content_fallback(registry,"industry",request,contexts)
    body=content.content_request_payload("industry",request,contexts)
    refs=[{"block_ref":b["block_ref"]} for b in body["canonical_block_registry"] if b["required"]]
    blocks=[*refs,{"kind":kind,"text":text,"evidence_ids":[]}]
    raw={"_model":"offline-span-protocol","master_title":request.approved_proposal.title,"blocks":deepcopy(blocks),
        "platform_variants":[{"platform":channel,"title":request.approved_proposal.title,"blocks":deepcopy(blocks)}
            for channel in request.channels]}
    return content.normalize_content_output(registry,"industry",request,contexts,fallback,raw),fallback


def _assert_facts_unchanged(output,fallback):
    for rows in [output.blocks,*[v.blocks for v in output.platform_variants]]:
        assert _canonical([b.model_dump(mode="json") for b in rows])==_canonical(fallback["blocks"])
    assert output.quality_scorecard is None


@pytest.mark.parametrize("text",[
    "Seedance 2.5 的这项约束，需要放回具体的创作取舍里理解。",
    "这给样片划出了一条边界：先讨论它能帮助作出什么选择。",
    "样片更像为尚未定案的方向买一次判断机会。",
    "样片更像一次方向闸门：进入正式阶段前，先明确创意。",
    "没有这类待验证问题时，样片增加的是一次审看和一次付费任务；这笔投入未必值得。",
    "提交前可以先写下它要排除的判断：镜头关系是否服务于表达。",
    "提交前先写下它要排除的判断:镜头关系是否服务于表达。",
    "方向已经清楚时，先讨论尚待验收的问题。",
    "若方向已经清楚，先讨论尚待验收的问题。",
])
def test_exact_nonnumeric_framing_survives_all_surfaces_without_changing_facts(text):
    output,fallback=_normalize(text)
    assert text in output.master_content
    assert all(text in v.body for v in output.platform_variants)
    assert not output.critic.errors
    assert not any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    _assert_facts_unchanged(output,fallback)


@pytest.mark.parametrize("text",[
    "Seedance 2.5 的收益是2.5元。",
    "Seedance 2.5 的费用是20元。",
    "Seedance 2.5 可以节省20%的投入。",
    "Seedance 2.5元对应的产品需要讨论。",
    "Seedance 2.5%对应的结果需要讨论。",
    "Seedance 2.50 的这项约束需要讨论。",
    "Seedance 2.5.1 的这项约束需要讨论。",
    "Seedance2.5 的这项约束需要讨论。",
    "某国划出了一条边界，需要讨论其影响。",
    "这给样片划出了两条边界，需要讨论取舍。",
    "样片更像一次方向闸门，之后少做两轮审片。",
    "样片更像为尚未定案的方向买两次判断机会。",
    "样片增加的是一次审看和一次付费任务。",
    "没有这类待验证问题时，样片增加的是一次审看和两次付费任务。",
    "没有这类待验证问题时，样片增加的是一次审看和一次付费任务；平台宣布发布新功能。",
    "没有这类待验证问题时，样片增加的是一次审看和一次付费任务；节省20%的投入。",
    "方向已经清楚，后续可直接安排制作。",
    "镜头关系已经偏离了原先的设想，需要讨论取舍。",
    "方向已经清楚时，平台已经发布新的功能。",
    "方向已经清楚时，流程保证成功。",
    "方向已经清楚时，我用它做过客户项目。",
    "提交前先写下它要排除的判断：镜头关系是否服务于表达。编辑说明：请继续分析。",
    "提交前先写下它要排除的判断：镜头关系是否服务于表达。判断：请继续分析。",
])
def test_exact_spans_do_not_hide_same_value_money_other_counts_events_or_meta(text):
    output,fallback=_normalize(text)
    assert text not in output.master_content
    assert all(text not in v.body for v in output.platform_variants)
    assert any(w.startswith("discarded_unsafe_model_editorial:") for w in output.critic.warnings)
    _assert_facts_unchanged(output,fallback)


def test_matching_version_needs_trusted_full_identifier_and_filters_only_its_own_span():
    text="Seedance 2.5 的收益是2.5元。"
    matches=content._industry_editorial_number_matches(text,["Seedance 2.5 示例资料。"],allow_industry_analysis=True)
    assert [(m.group(),m.span()) for m in matches]==[("2.5",(text.rindex("2.5"),text.rindex("2.5")+3))]
    for source in ("没有品牌版本的资料。","OtherSeedance 2.5 示例资料。","Seedance 2.5元 示例资料。"):
        assert len(content._industry_editorial_number_matches(text,[source],allow_industry_analysis=True))==2
    assert len(content._industry_editorial_number_matches(text,["Seedance 2.5 示例资料。"]))==2
    output,_=_normalize("Seedance 2.5 的这项约束需要具体讨论。",source_brand="其他版本")
    assert "Seedance 2.5" not in "\n".join(b.text for b in output.blocks if not b.source_exact)


def test_only_the_exact_reader_task_and_direction_matches_are_removed():
    text="提交前先写下它要排除的判断：内容是否清楚。判断：继续讨论。"
    matches=content._generic_editorial_meta_matches(text,allow_industry_reader_task=True)
    assert len(matches)==1 and matches[0].start()==text.rindex("判断：")
    assert len(content._generic_editorial_meta_matches(text))==2
    text="方向已经清楚时，产品已经宣布发布。"
    matches=content._unsupported_editorial_assertion_matches(text,allow_industry_direction_condition=True)
    assert [m.group() for m in matches]==["已经","宣布","发布"]
    assert matches[0].start()==text.index("已经",text.index("已经")+2)
    assert len(content._unsupported_editorial_assertion_matches(text,allow_industry_hypothetical=True))==4


@pytest.mark.parametrize("genre",["","digest","company_data_brief"])
def test_legacy_genres_and_cta_do_not_receive_new_exceptions(genre):
    for text in ("Seedance 2.5 的这项约束需要具体讨论。","这给样片划出了一条边界：请讨论选择。",
        "提交前先写下它要排除的判断：请讨论选择。","方向已经清楚时，请讨论选择。"):
        legacy,_=_normalize(text,genre=genre)
        assert text not in legacy.master_content
        cta,_=_normalize(text,kind="cta")
        assert text not in cta.master_content


def test_existing_material_readiness_cta_keeps_its_prior_hypothetical_behavior():
    text="如果资料已经齐备，请核对来源与授权。"
    output,_=_normalize(text,kind="cta")
    assert text in output.master_content
