"""Fictional title allocation tests; no provider or editorial grade evidence."""
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_runtime import OperatorRole
from tests.ai_marketing_api.test_preparation_surface_selection import signed_preparation_plan
from tests.ai_marketing_api.test_surface_fact_plan import fixture
from tests.ai_marketing_api.test_operator_content_runtime import _run


@pytest.mark.parametrize("raw_title", [None, "", "预览与正式验收"])
def test_automatic_prepared_title_uses_only_the_existing_surface_facts(monkeypatch, raw_title):
    request, _, candidate = fixture(monkeypatch)
    request.approved_proposal.title = "预览与正式验收"
    request.surface_fact_plan = signed_preparation_plan(request)
    if raw_title is None:
        candidate.pop("master_title")
        for variant in candidate["platform_variants"]:
            variant.pop("title")
    else:
        candidate["master_title"] = raw_title
        for variant in candidate["platform_variants"]:
            variant["title"] = raw_title
    before = deepcopy((request.model_dump(mode="json"), candidate))
    output = _run(OperatorRole.INDUSTRY, request, candidate)
    assert output.critic.errors == []
    assert output.surface_fact_plan_sha256 == request.surface_fact_plan["plan_sha256"]
    assert [block.claim_id for block in output.blocks] == ["claim-1", "claim-2", "claim-3"]
    for variant in output.platform_variants:
        expected = request.surface_fact_plan["surfaces"][variant.platform]
        assert [block.claim_id for block in variant.blocks] == expected
        if variant.platform != "wechat_mp":
            assert variant.title in request.claims[0].text
            assert len(variant.blocks) == 1
    assert (request.model_dump(mode="json"), candidate) == before


def test_empty_eligible_surface_title_pool_does_not_use_an_unallocated_fact(monkeypatch):
    request, _, _ = fixture(monkeypatch)
    request.approved_proposal.title = "预览与正式验收"
    request.surface_fact_plan = signed_preparation_plan(request)
    blocks, errors = content._build_approved_blocks(OperatorRole.INDUSTRY,
        request, request.authorized_context)
    assert errors == []
    # An unrelated approved full-pool title remains available, but it cannot
    # repair a surface whose explicitly supplied allocation has no candidate.
    with pytest.raises(ValueError, match="^social_title_no_eligible_candidate$"):
        content._safe_master_title(request, blocks, role=OperatorRole.INDUSTRY,
            surface="wechat_moments", fallback_claim_ids=set())
