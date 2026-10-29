"""Only the explicit human object is exempted from dynamic billing scans."""
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content


@pytest.mark.parametrize("preparation", [False, True])
@pytest.mark.parametrize("text,needs_billing", [
    ("应核对流程能否节省人力。", False), ("先判断能够节省多少人力。", False),
    ("节省", True), ("节省费用", True), ("节省人力成本", True), ("节省多少人力费用", True),
    ("节省人力以后仍需付费。", True), ("节省人力也要比较cost。", True),
    ("收费", True), ("省钱", True), ("price", True), ("节省时间", True),
    ("节省人力费", True), ("节省人力，也要比较cost。", True),
])
def test_dynamic_scanner_keeps_financial_remainder_and_ambiguous_terms(preparation, text, needs_billing):
    plan = {"surfaces": {"wechat_moments": ["claim-1"]},
        "fact_roles": {"anchor": ["claim-1"], "billing": [], "input_constraints": []}}
    if preparation:
        plan["preparation_authorization"] = {}
    original = deepcopy(plan)
    assert content._SURFACE_DEPENDENCIES["billing"].search(text)
    if needs_billing:
        with pytest.raises(content.ContextAuthorizationError) as error:
            content._surface_required_ids(plan, "wechat_moments", [{"text": text}])
        assert error.value.code == "surface_fact_plan_dependency_evidence_missing"
    else:
        assert content._surface_required_ids(plan, "wechat_moments", [{"text": text}]) == {"claim-1"}
    assert plan == original
