"""Offline display and frozen-artifact contracts; no model or owner decisions."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_content as content, operator_runtime as runtime, operator_steps as steps

VERSION = "literal_prose.v1"
FACT = ("报告记述，示例团队先核对公开材料中的输入边界和项目要求，"
        "再组织同一组任务的过程记录以供逐项复核，"
        "并把复核结果交给原来负责交付验收的人员检查，"
        "随后改由该团队维护提示词与评估规则，"
        "但这项记录没有披露完成任务后的净人力变化，不能直接推算岗位数量。")


def request(version=None):
    value = {"objective": "解释虚构报告的适用边界", "genre": "industry_analysis",
        "channels": ["wechat_mp", "wechat_moments", "weitoutiao"],
        "as_of": datetime(2026, 10, 1, tzinfo=timezone.utc),
        "approved_proposal": {"title": "虚构报告", "angle": "核对适用条件",
            "audience_value": "用于离线回归", "key_points": [FACT], "suggested_formats": ["wechat_mp"],
            "first_person": False},
        "claims": [{"text": FACT, "kind": "fact", "evidence_ids": ["source-1"],
            "verification_status": "verified"}]}
    if version is not None:
        value["fact_projection_version"] = version
    return content.OperatorComposeRequest.model_validate(value)


def context():
    return runtime.AuthorizedContext(record_id="source-1", space="industry", record_type="source_fact",
        content=FACT, authorized_roles=["industry"], source_tier="official",
        source_uri="https://publisher.example/fictional-report")


def test_absent_version_keeps_legacy_request_and_block_serialization():
    legacy = request()
    assert "fact_projection_version" not in legacy.model_dump(mode="json")
    explicit_none = content.OperatorComposeRequest.model_validate({**legacy.model_dump(mode="json"),
        "fact_projection_version": None})
    assert legacy.model_dump(mode="json") == explicit_none.model_dump(mode="json")
    blocks, errors = content._build_approved_blocks(runtime.OperatorRole.INDUSTRY, legacy, [context()])
    assert errors == []
    assert "fact_projection_version" not in blocks[0].model_dump(mode="json")
    assert content._render(blocks) == "；".join(FACT.split("，")[:3]) + "。"


def test_new_canonical_projection_keeps_original_punctuation_and_late_conditions():
    blocks, errors = content._build_approved_blocks(runtime.OperatorRole.INDUSTRY, request(VERSION), [context()])
    assert errors == []
    assert blocks[0].fact_projection_version == VERSION
    assert blocks[0].source_exact and blocks[0].locked and blocks[0].required
    assert blocks[0].binding_hash != content._build_approved_blocks(runtime.OperatorRole.INDUSTRY,
        request(), [context()])[0][0].binding_hash
    for surface in ("master", "wechat_mp", "wechat_moments", "weitoutiao"):
        assert content._render(blocks, surface=surface) == FACT


@pytest.mark.parametrize("change", ["role", "genre"])
def test_literal_projection_cannot_expand_to_other_roles_or_genres(change):
    value = request(VERSION)
    role = runtime.OperatorRole.COMMERCIAL if change == "role" else runtime.OperatorRole.INDUSTRY
    if change == "genre":
        value = value.model_copy(update={"genre": "company_data_brief"})
    with pytest.raises(runtime.ContextAuthorizationError) as error:
        content._build_approved_blocks(role, value, [context()])
    assert error.value.code == "fact_projection_scope_invalid"


def test_unknown_projection_is_rejected_by_request_schema():
    with pytest.raises(ValidationError):
        request("unrecognized.v1")


@pytest.mark.parametrize("candidate_version,request_version", [(None, VERSION), (VERSION, None)])
def test_signed_artifact_cannot_be_reinterpreted_under_a_new_projection(monkeypatch, candidate_version, request_version):
    # A fixed offline test key authorizes no application action or real receipt.
    monkeypatch.setenv("HERMES_API_KEY", "offline-display-contract-only")
    blocks, errors = content._build_approved_blocks(runtime.OperatorRole.INDUSTRY,
        request(candidate_version), [context()])
    assert errors == []
    candidate = {"blocks": [blocks[0].model_dump(mode="json")], "platform_variants": []}
    original = deepcopy(candidate)
    with pytest.raises(ValueError, match="artifact_fact_projection_mismatch"):
        steps._normalized_candidate(None, runtime.OperatorRole.INDUSTRY, request(request_version),
            [context()], {}, candidate, steps.artifact_signature(candidate))
    assert candidate == original
