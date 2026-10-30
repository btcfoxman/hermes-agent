"""Versioned long-form selection keeps canonical evidence and old signatures."""
from copy import deepcopy

import pytest

from ai_marketing_api import operator_content as content, operator_runtime as runtime, operator_steps as steps
from tests.ai_marketing_api.test_surface_fact_plan import fixture
from tests.ai_marketing_api.test_preparation_surface_selection import signed_preparation_plan, resign_plan
from tests.ai_marketing_api.test_operator_content_runtime import _run

FIELD = "long_surface_selection_version"
VERSION = "thesis_selection.v1"


def request_with_plan(monkeypatch):
    request, contexts, _raw = fixture(monkeypatch)
    body = {key: value for key, value in signed_preparation_plan(request).items()
        if key not in {"signature", "certificate", "plan_sha256"}}
    body[FIELD] = VERSION
    for surface in ("master", "wechat_mp"):
        body["surfaces"][surface] = ["claim-1"]
    request = content.OperatorComposeRequest.model_validate({**request.model_dump(mode="json"),
        FIELD: VERSION, "surface_fact_plan": resign_plan(body)})
    return request, contexts, body


def test_new_long_selection_selects_refs_without_rewriting_or_dropping_required_anchor(monkeypatch):
    request, contexts, _body = request_with_plan(monkeypatch)
    before = deepcopy(request.model_dump(mode="json"))
    writer = content.content_request_payload("industry", request, contexts)
    refs = {row["claim_id"]: {"block_ref": row["block_ref"]} for row in writer["canonical_block_registry"]}
    raw = {"_model": "offline-contract-only", "master_title": "预览怎样确认镜头意图", "blocks": [refs["claim-1"]],
        "platform_variants": [{"platform": surface, "title": "预览怎样确认镜头意图", "blocks": [refs["claim-1"]]}
            for surface in request.channels]}
    output = _run(runtime.OperatorRole.INDUSTRY, request, raw)
    assert output.critic.errors == [] and [block.claim_id for block in output.blocks] == ["claim-1"]
    assert output.master_content == request.claims[0].text
    assert output.surface_fact_plan_sha256 == request.surface_fact_plan["plan_sha256"]
    assert "remaining pool is optional" in writer["surface_fact_plan_rule"]
    assert request.model_dump(mode="json") == before


@pytest.mark.parametrize("change", ["request_absent", "plan_absent", "not_preparation", "unsigned_change"])
def test_unmatched_or_unsigned_selection_policy_is_rejected(monkeypatch, change):
    request, contexts, body = request_with_plan(monkeypatch)
    if change == "request_absent": request.long_surface_selection_version = None
    elif change == "plan_absent": request.surface_fact_plan = None
    elif change == "not_preparation":
        body.pop("preparation_authorization")
        body.pop("preparation_selection_sha256")
        body["editorial_intent"] = {"author": "Codex AI editor", "thesis": "Synthetic non-preparation direction"}
        request.surface_fact_plan = resign_plan(body)
    else: request.surface_fact_plan[FIELD] = "unsigned-change"
    with pytest.raises(runtime.ContextAuthorizationError):
        content.content_request_payload("industry", request, contexts)


def test_old_request_serialization_and_long_full_pool_are_unchanged(monkeypatch):
    request, contexts, _raw = fixture(monkeypatch)
    assert FIELD not in request.model_dump(mode="json")
    request.surface_fact_plan = signed_preparation_plan(request)
    writer = content.content_request_payload("industry", request, contexts)
    assert "All long-form surfaces retain the full pool" in writer["surface_fact_plan_rule"]
    before = deepcopy(request.surface_fact_plan)
    body = {key: value for key, value in before.items() if key not in {"signature", "plan_sha256"}}
    body["surfaces"]["master"] = ["claim-1"]
    request.surface_fact_plan = resign_plan(body)
    with pytest.raises(runtime.ContextAuthorizationError):
        content.content_request_payload("industry", request, contexts)


def test_signed_step_artifact_cannot_be_reinterpreted_under_another_plan(monkeypatch):
    request, contexts, _body = request_with_plan(monkeypatch)
    writer = content.content_request_payload("industry", request, contexts)
    ref = next(row["block_ref"] for row in writer["canonical_block_registry"] if row["claim_id"] == "claim-1")
    output = _run(runtime.OperatorRole.INDUSTRY, request, {"_model": "offline-contract-only",
        "master_title": "预览怎样确认镜头意图", "blocks": [{"block_ref": ref}],
        "platform_variants": [{"platform": surface, "title": "预览怎样确认镜头意图", "blocks": [{"block_ref": ref}]}
            for surface in request.channels]})
    candidate = output.model_dump(mode="json")
    candidate["surface_fact_plan_sha256"] = "0" * 64
    before = deepcopy(candidate)
    with pytest.raises(ValueError, match="artifact_surface_fact_plan_mismatch"):
        steps._normalized_candidate(None, runtime.OperatorRole.INDUSTRY, request, contexts, {}, candidate,
            steps.artifact_signature(candidate))
    assert candidate == before
