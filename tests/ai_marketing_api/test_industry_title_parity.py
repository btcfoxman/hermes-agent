"""Synthetic title inputs, including the actual R59 rejected/normalized titles."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_runtime import OperatorRole

LONG = "Claude 经 Amazon Bedrock 在公司既有 AWS 环境及托管区域内运行"
PERMISSIONS = "客户可限定 Custom Agent 获准执行的动作"
RAW_R59 = "Zendesk 的 Claude 案例：智能体选型不能只看模型"

def inputs(*, genre="industry_analysis", requested="PRIVATE_UNGROUNDED_DIRECTION", texts=None):
    request=NS(genre=genre,approved_proposal=NS(title=requested,first_person=False))
    blocks=[NS(text=text,source_exact=True) for text in (texts if texts is not None else [LONG,PERMISSIONS])]
    return request,blocks

def choose(supplied,*,surface="wechat_mp",role=OperatorRole.INDUSTRY,**kwargs):
    request,blocks=inputs(**kwargs); before=deepcopy((request,blocks)); warnings=[]
    value=content._safe_master_title(request,blocks,supplied,role=role,surface=surface,diagnostic_warnings=warnings)
    assert (request,blocks)==before
    return value,warnings,blocks

def test_actual_rejected_r59_wrapper_uses_later_approved_permissions_clause():
    value,warnings,blocks=choose(RAW_R59)
    assert value==PERMISSIONS and value!=LONG and value!=RAW_R59
    assert warnings==["model_title_normalized:wechat_mp:forbidden_wrapper"]
    assert not content._social_title_quality_errors(value,surface="wechat_mp",source_texts=[b.text for b in blocks])

def test_existing_grounded_short_model_title_is_preserved():
    supplied="Custom Agent 动作权限怎么定"
    value,warnings,_=choose(supplied)
    assert value==supplied and warnings==[]

def test_requested_title_must_also_satisfy_platform_quality():
    requested="Custom Agent 动作权限怎么定"
    assert choose(RAW_R59,requested=requested)[0]==requested
    assert choose(RAW_R59,requested=LONG)[0]==PERMISSIONS

@pytest.mark.parametrize("supplied,reason",[
    ("Claude 999 客户权限变动","ungrounded_number"),
    ("陌生平台 Claude 动作权限如何划分","ungrounded_entity"),
    ("Claude guaranteed customer value","forbidden_promise")])
def test_original_grounding_entity_and_promise_fences_remain(supplied,reason):
    value,warnings,_=choose(supplied)
    assert value==PERMISSIONS
    assert warnings==["model_title_normalized:wechat_mp:"+reason]

def test_platform_limits_40_master_and_36_variant():
    supplied="Custom Agent 权限"+"边"*25
    assert len(supplied)==40
    assert choose(supplied,surface="master")[0]==supplied
    assert choose(supplied+"边",surface="master")[0]==PERMISSIONS
    assert choose(supplied,surface="wechat_mp")[0]==PERMISSIONS
    assert choose(supplied[:36],surface="wechat_mp")[0]==supplied[:36]

def test_long_verbatim_source_excerpt_is_rejected_even_below_36():
    excerpt="Claude 对客户服务操作权限范围需要逐项限定并持续核对操作"
    assert 28<len(excerpt)<=36
    assert choose(excerpt,texts=[excerpt,PERMISSIONS])[0]==PERMISSIONS

@pytest.mark.parametrize("genre,role",[(None,OperatorRole.INDUSTRY),("public_tutorial",OperatorRole.INDUSTRY),("industry_analysis",OperatorRole.COMMERCIAL)])
def test_legacy_and_nonindustry_title_behavior_remains_unchanged(genre,role):
    value,warnings,_=choose(LONG,genre=genre,role=role)
    assert value==LONG and warnings==[]

def test_no_complete_eligible_clause_fails_closed_without_truncating():
    request,blocks=inputs(texts=[LONG])
    with pytest.raises(ValueError,match="^social_title_no_eligible_candidate$"):
        content._safe_master_title(request,blocks,RAW_R59,role=OperatorRole.INDUSTRY,surface="wechat_mp")
