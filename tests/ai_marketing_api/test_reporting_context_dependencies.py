"""Offline evidence propagation contracts; these are not editorial quality scores."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_content as content, operator_runtime as runtime
from ai_marketing_api.operator_grounding import paraphrase_eligible


NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
URI = "https://source.example/report/"
SOURCE = "Synthetic official source text with a publication statement and bounded engineering observations."


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def annotate(rows):
    value = {"schema_version": "operator.reporting_context.v1", "annotation_id": "context-annotation",
        "annotation_version": 1, "source_uri": URI, "source_text_sha256": sha(SOURCE),
        "facts": [{"record_id": rows[-1].record_id, "record_version": 2,
            "body_sha256": sha(rows[-1].content), "role": "report_attribution"}]}
    return sign(value)


def sign(value):
    value = deepcopy(value)
    value.pop("annotation_sha256", None)
    value["annotation_sha256"] = sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False))
    return value


def records(*, long_attribution=False, count=7):
    rows = [runtime.AuthorizedContext(record_id=f"fact-{i}", space="industry", record_type="source_fact",
        content=f"示例机构公开描述了接口行为条件{i}，其适用范围仅为这项记录。",
        authorized_roles=["industry"], source_tier="official", source_uri=URI,
        structured_data={"source_document": {"url": URI, "source_text": SOURCE, "source_text_hash": sha(SOURCE)}})
        for i in range(1, count + 1)]
    rows[-1] = rows[-1].model_copy(update={"content":
        ("该材料中的内容均归属于公开报告的署名作者，不能当作独立实验。" * 20 if long_attribution else "")
        + "示例机构的署名作者于2026年10月26日在官方刊物公开介绍其工程方法。"})
    return rows


def proposed(rows=None, *, binding=True, selected=1):
    rows = rows or records()
    request = runtime.OperatorProposeRequest(objective="分析已授权工程报告的意义。", genre="industry_analysis",
        channels=["wechat_mp", "wechat_moments"], authorized_context=rows, as_of=NOW,
        **({"reporting_context": annotate(rows)} if binding else {}))
    registry = runtime.OperatorRegistry()
    authorized = registry.authorize(runtime.OperatorRole.INDUSTRY, rows, NOW)
    fallback = runtime.build_fallback("industry", runtime.OperatorAction.PROPOSE, request, authorized)
    raw = {runtime.INDUSTRY_FACT_SELECTION_FIELD: (runtime.INDUSTRY_CONTEXT_FACT_SELECTION_VERSION
        if binding else runtime.INDUSTRY_FACT_SELECTION_VERSION), "_model": "offline-writer",
        "claims": [{"kind": "fact", "text": rows[selected - 1].content,
            "evidence_ids": [rows[selected - 1].record_id]}]}
    output = runtime.normalize_output(registry, "industry", runtime.OperatorAction.PROPOSE,
        request, authorized, fallback, raw)
    return request, output, fallback


def test_argument_subset_keeps_only_its_whole_reporting_dependency():
    request, output, original = proposed()
    assert len(original["claims"]) == 7
    assert output.evidence_refs == ["fact-1", "fact-7"]
    assert [claim.text for claim in output.claims] == [records()[0].content, records()[-1].content]
    assert not any(risk.code.startswith("industry_fact_selection") for risk in output.risk_flags)
    body = runtime.request_payload(request, role_id=runtime.OperatorRole.INDUSTRY,
        action=runtime.OperatorAction.PROPOSE, fallback=original)
    assert body["response_contract"][runtime.INDUSTRY_FACT_SELECTION_FIELD] == runtime.INDUSTRY_CONTEXT_FACT_SELECTION_VERSION
    assert body["reporting_context"] == annotate(records())


def test_legacy_request_and_subset_do_not_gain_new_authority_or_wire_fields():
    request, output, _ = proposed(binding=False)
    assert "reporting_context" not in request.model_dump(mode="json")
    assert output.evidence_refs == ["fact-1"]


@pytest.mark.parametrize("change", ["extra", "version_bool", "record_version_bool", "schema", "hash", "duplicate"])
def test_annotation_shape_and_own_binding_fail_closed(change):
    value = annotate(records())
    if change == "extra": value["expected_conclusion"] = "not authorized"
    elif change == "version_bool": value["annotation_version"] = True
    elif change == "record_version_bool": value["facts"][0]["record_version"] = True
    elif change == "schema": value["schema_version"] = "unrecognized"
    elif change == "hash": value["annotation_sha256"] = "0" * 64
    else: value["facts"] *= 2
    if change != "hash": value = sign(value)
    with pytest.raises(ValidationError):
        runtime.ReportingContext.model_validate(value)


@pytest.mark.parametrize("change", ["missing", "body", "source", "document", "tier", "type", "genre", "role", "private"])
def test_dependency_cannot_invent_or_cross_the_current_authorized_source(change):
    rows = records()
    value = annotate(rows)
    role = "industry"
    genre = "industry_analysis"
    if change == "missing": rows = rows[:-1]
    elif change == "body": rows[-1] = rows[-1].model_copy(update={"content": rows[-1].content + "changed"})
    elif change == "source": rows[-1] = rows[-1].model_copy(update={"source_uri": "https://other.example/report"})
    elif change == "document": rows[-1] = rows[-1].model_copy(update={"structured_data": {}})
    elif change == "tier": rows[-1] = rows[-1].model_copy(update={"source_tier": "secondary"})
    elif change == "type": rows[-1] = rows[-1].model_copy(update={"record_type": "source_item"})
    elif change == "genre": genre = "tutorial"
    elif change == "role": role = "commercial"
    elif change == "private": rows[-1] = rows[-1].model_copy(update={"structured_data":
        {**rows[-1].structured_data, "publishable": False}})
    request = runtime.OperatorProposeRequest(objective="Analyze supplied facts.", genre=genre,
        authorized_context=rows, reporting_context=value)
    with pytest.raises(runtime.ContextAuthorizationError):
        runtime.build_fallback(role, runtime.OperatorAction.PROPOSE, request, rows)


def test_long_attribution_outside_the_eight_fact_prefix_survives_whole_and_on_every_surface(monkeypatch):
    rows = records(long_attribution=True, count=10)
    request, output, fallback = proposed(rows)
    assert len(rows[-1].content) > 480
    assert output.evidence_refs == ["fact-1", "fact-10"]
    assert output.claims[-1].text == rows[-1].content
    compose = content.OperatorComposeRequest(**request.model_dump(mode="json"),
        approved_proposal=output.proposal, claims=output.claims)
    monkeypatch.setenv("HERMES_API_KEY", "reporting-test-only")
    pool = [{"claim_id": f"claim-{index}", "kind": "fact", "text_sha256": sha(claim.text),
        "evidence_ids": claim.evidence_ids} for index, claim in enumerate(compose.claims, 1)]
    plan = {"schema_version": "surface_fact_plan.v1", "scope": {"role_id": "industry", "genre": compose.genre,
        "objective_sha256": sha(compose.objective)}, "claim_pool": pool,
        "surfaces": {"master": ["claim-1", "claim-2"], "wechat_mp": ["claim-1", "claim-2"],
            "wechat_moments": ["claim-1"]}, "fact_roles": {"anchor": ["claim-1", "claim-2"],
            "billing": [], "input_constraints": []}, "editorial_intent": {}, "narrative_required": True}
    digest = sha(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    compose.surface_fact_plan = {**plan, "plan_sha256": digest,
        "signature": hmac.new(b"reporting-test-only", ("surface_fact_plan.v1\n" + digest).encode(), hashlib.sha256).hexdigest()}
    registry = runtime.OperatorRegistry()
    base = content.build_content_fallback(registry, "industry", compose, rows)
    writer = content.content_request_payload("industry", compose, rows)
    attribution = next(block for block in writer["canonical_block_registry"]
        if block.get("source_display") == "report_attribution")
    ordinary = next(block for block in writer["canonical_block_registry"] if block is not attribution)
    raw_blocks = [{"block_ref": ordinary["block_ref"]}]
    raw = {"_model": "offline-writer", "master_title": "示例机构的接口行为条件",
        "blocks": raw_blocks, "platform_variants": [{"platform": surface,
            "title": "示例机构的接口行为条件", "blocks": deepcopy(raw_blocks)} for surface in request.channels]}
    rendered = content.normalize_content_output(registry, "industry", compose, rows, base, raw)
    for surface, blocks in [("master", rendered.blocks), *[(v.platform, v.blocks) for v in rendered.platform_variants]]:
        block = next(block for block in blocks if block.source_display == "report_attribution")
        assert block.text == rows[-1].content
        assert content._fact_display_text(block, surface=surface) == rows[-1].content
        assert rows[-1].content in content._render(blocks, surface=surface)
        assert not paraphrase_eligible(block, "删掉署名和日期的改写。")
        tampered = block.model_copy(update={"public_text": "不完整归属", "public_text_verified": True})
        assert content._fact_display_text(tampered, surface=surface) == rows[-1].content
    assert len(fallback["claims"]) == 9  # Eight bounded ordinary facts plus the explicit dependency.


def test_invalid_context_is_rejected_before_a_model_call_on_the_real_route(monkeypatch):
    from tests.ai_marketing_api.test_industry_fact_selection import api_request
    from ai_marketing_api import main as api
    monkeypatch.setenv("HERMES_API_KEY", "isolated-test-key")
    calls = []
    async def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Invalid context must not buy a provider call.")
    monkeypatch.setattr(api, "_llm_json", forbidden)
    request, _, _ = proposed()
    wire = request.model_dump(mode="json")
    wire["authorized_context"][-1]["content"] += " changed"
    response = api_request("/api/v1/operators/industry/propose", wire)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "reporting_context_fact_unavailable"
    assert not calls


def test_corroborating_source_limit_cannot_silently_drop_the_required_record_id():
    rows = records(count=21)
    rows = [row.model_copy(update={"content": rows[-1].content}) for row in rows]
    request = runtime.OperatorProposeRequest(objective="Use approved source evidence.", genre="industry_analysis",
        authorized_context=rows, reporting_context=annotate(rows))
    with pytest.raises(runtime.ContextAuthorizationError, match="complete attribution dependency"):
        runtime.build_fallback("industry", runtime.OperatorAction.PROPOSE, request, rows)


def test_compose_cannot_add_attribution_that_is_absent_from_the_frozen_manifest():
    request, output, _ = proposed()
    compose = content.OperatorComposeRequest(**request.model_dump(mode="json"),
        approved_proposal=output.proposal, claims=output.claims[:1])
    with pytest.raises(runtime.ContextAuthorizationError, match="exact verified canonical claim"):
        content.content_request_payload("industry", compose, records())


def test_only_server_validated_context_assigns_the_display_tag_and_binding():
    request, output, _ = proposed()
    compose = content.OperatorComposeRequest(**request.model_dump(mode="json"),
        approved_proposal=output.proposal, claims=output.claims)
    approved, errors = content._build_approved_blocks(runtime.OperatorRole.INDUSTRY, compose, records())
    assert not errors
    ordinary = approved[0]
    raw = [{"block_ref": ordinary.binding_hash, "source_display": "report_attribution"}]
    normalized, errors, _ = content._safe_candidate_blocks(runtime.OperatorRole.INDUSTRY, raw, approved, prefix="test")
    assert not errors and normalized[0].source_display == "excerpt"
    attribution = approved[-1]
    assert content._binding_hash(attribution.model_dump()) != content._binding_hash(
        {**attribution.model_dump(), "source_display": "excerpt"})
