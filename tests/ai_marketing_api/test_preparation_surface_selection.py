"""Real offline selection/normalizer contracts, never an editorial quality grade."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_content as content, operator_runtime as runtime
from ai_marketing_api.operator_planning import EditorialPlanRequest
from ai_marketing_api.operator_surface_selection import PreparationEvidenceSelection
from tests.ai_marketing_api.test_operator_planning_genre import payload, planned, selection
from tests.ai_marketing_api.test_surface_fact_plan import fixture as surface_fixture
from tests.ai_marketing_api.test_operator_content_runtime import _run


URI = "https://source.example/report/"
NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def allocation(**changes):
    return {"schema_version": "operator.industry_surface_selection.v1", "short_surfaces": {
        "wechat_moments": {"anchor_record_id": "fact-a", "reason": "解释已有报告如何界定交接边界。"}},
        "reported_case_record_ids": ["fact-b"], "case_gap": "", **changes}


def catalog_payload():
    raw = payload(genre="industry_analysis")
    raw["mandate"]["channels"] = ["wechat_mp", "wechat_moments"]
    facts = ["报告允许逐条打开引用，但不会替编辑判断论证是否足够。", "已记录的交接案例中，接收者另行取得原始输入后继续核对。"]
    raw["opportunities"][0]["summary"] += "\n" + "\n".join(facts)
    raw["opportunities"][0]["url"] = URI
    raw["industry_fact_catalog"] = {"source-a": [{"record_id": identity, "version": 2,
        "text": fact, "body_sha256": sha(fact), "excerpt_sha256": sha(fact),
        "source_text_sha256": sha("\n".join(facts)), "source_uri": URI}
        for identity, fact in zip(("fact-a", "fact-b"), facts)]}
    return raw


def test_fresh_independent_planner_selects_existing_case_and_surface_in_same_call(monkeypatch):
    raw = catalog_payload()
    proposed = selection()
    proposed["industry_surface_selection"] = allocation()
    before = deepcopy(raw)
    result, calls = planned(monkeypatch, raw, [proposed])
    assert result["status"] == "planned" and raw == before
    assert len(calls) == result["generation_trace"]["model_calls"] == 1
    assert result["selections"][0]["industry_surface_selection"] == allocation()
    assert calls[0][1]["industry_fact_catalog"] == raw["industry_fact_catalog"]
    assert "industry_surface_selection" in calls[0][1]["response_contract"]["selections"][0]
    assert "owner_review" not in result["selections"][0]
    assert result["generation_trace"]["repair_calls"] == 0


@pytest.mark.parametrize("change", ["missing", "unknown", "case_other_opportunity", "long_subset", "missing_surface", "case_and_gap", "invented_case", "owner_proof"])
def test_planner_cannot_select_unprovided_facts_or_forge_authority(monkeypatch, change):
    proposed = selection()
    choice = allocation()
    if change == "unknown": choice["short_surfaces"]["wechat_moments"]["anchor_record_id"] = "not-displayed"
    elif change == "case_other_opportunity": choice["reported_case_record_ids"] = ["other-source-case"]
    elif change == "long_subset": choice["short_surfaces"]["wechat_mp"] = choice["short_surfaces"]["wechat_moments"]
    elif change == "missing_surface": choice["short_surfaces"] = {}
    elif change == "case_and_gap": choice["case_gap"] = "No case."
    elif change == "invented_case": choice["reported_case_record_ids"] = ["invented-case"]
    elif change == "owner_proof": choice["owner_review"] = {"approved": True}
    if change != "missing": proposed["industry_surface_selection"] = choice
    result, calls = planned(monkeypatch, catalog_payload(), [proposed])
    assert result["status"] == "unavailable" and result["selections"] == []
    assert len(calls) == 1 and result["generation_trace"]["repair_calls"] == 0


@pytest.mark.parametrize("change", ["body", "not_displayed", "source", "document", "duplicate", "version_bool", "genre"])
def test_catalog_binding_is_checked_before_any_planner_call(change):
    raw = catalog_payload()
    fact = raw["industry_fact_catalog"]["source-a"][0]
    if change == "body": fact["text"] += " An unsupported result."
    elif change == "not_displayed": raw["opportunities"][0]["summary"] = "Only a headline."
    elif change == "source": fact["source_uri"] = "https://another.example/report"
    elif change == "document": fact["source_text_sha256"] = "0" * 64
    elif change == "duplicate": raw["industry_fact_catalog"]["source-a"].append(deepcopy(fact))
    elif change == "version_bool": fact["version"] = True
    else: raw["genre"] = "news"
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(raw)


def test_absent_opt_in_keeps_old_planner_fields_and_rejects_unsolicited_suggestion(monkeypatch):
    raw = payload(genre="industry_analysis")
    assert "industry_fact_catalog" not in EditorialPlanRequest.model_validate(raw).model_dump(mode="json")
    result, calls = planned(monkeypatch, raw, [selection()])
    assert result["status"] == "planned"
    assert "industry_surface_selection" not in result["selections"][0]
    assert "industry_fact_catalog" not in calls[0][1]
    assert "industry_surface_selection_contract" not in calls[0][1]
    proposed = selection()
    proposed["industry_surface_selection"] = allocation()
    result, _ = planned(monkeypatch, raw, [proposed])
    assert result["status"] == "unavailable" and result["selections"] == []


def test_long_only_case_gap_is_a_valid_suggestion_without_an_empty_propose_union(monkeypatch):
    raw = catalog_payload()
    raw["mandate"]["channels"] = ["wechat_mp"]
    proposed = selection()
    proposed["industry_surface_selection"] = allocation(short_surfaces={}, reported_case_record_ids=[],
        case_gap="给定事实中没有可用于说明交接过程的具体案例。")
    result, calls = planned(monkeypatch, raw, [proposed])
    assert result["status"] == "planned" and len(calls) == 1
    assert result["selections"][0]["industry_surface_selection"] == proposed["industry_surface_selection"]
    # Main omits the optional native union when there are no selected records.
    request = runtime.OperatorProposeRequest(objective="解释公开报告。", genre="industry_analysis", channels=["wechat_mp"])
    assert "preparation_evidence_selection" not in request.model_dump(mode="json")


def test_catalog_recognizes_the_same_https_default_port_and_final_slash():
    raw = catalog_payload()
    raw["industry_fact_catalog"]["source-a"][0]["source_uri"] = "https://source.example:443/report"
    assert EditorialPlanRequest.model_validate(raw).industry_fact_catalog["source-a"][0].record_id == "fact-a"


def proposal_records():
    bodies = [f"示例报告描述了接口行为条件{i}，仅适用于该公开记录。" for i in range(1, 11)]
    bodies[8] = "该公开案例逐项记录了输入材料、交接安排与验收问题。" * 25
    bodies[9] = "示例机构署名作者于2026年10月26日在官方报告介绍工程方法。"
    source = "\n".join(bodies)
    return [runtime.AuthorizedContext(record_id=f"fact-{i}", space="industry", record_type="source_fact",
        content=body, authorized_roles=["industry"], source_tier="official", source_uri=URI,
        structured_data={"source_document": {"url": URI, "source_text": source, "source_text_hash": sha(source)}})
        for i, body in enumerate(bodies, 1)]


def proposal_request(rows, *, selection_enabled=True):
    annotation = {"schema_version": "operator.reporting_context.v1", "annotation_id": "reported-context",
        "annotation_version": 1, "source_uri": URI,
        "source_text_sha256": rows[-1].structured_data["source_document"]["source_text_hash"],
        "facts": [{"record_id": rows[-1].record_id, "record_version": 2,
            "body_sha256": sha(rows[-1].content), "role": "report_attribution"}]}
    annotation["annotation_sha256"] = digest(annotation)
    return runtime.OperatorProposeRequest(objective="分析公开报告的交接与验收边界。", genre="industry_analysis",
        channels=["wechat_mp", "wechat_moments"], authorized_context=rows, as_of=NOW, reporting_context=annotation,
        **({"preparation_evidence_selection": {"schema_version": "operator.preparation_evidence_selection.v1",
            "planner_selection_sha256": digest(allocation()), "required_evidence_ids": ["fact-9"]}} if selection_enabled else {}))


def test_native_subset_retains_whole_case_beyond_old_cap_and_report_attribution():
    rows = proposal_records()
    request = proposal_request(rows)
    registry = runtime.OperatorRegistry()
    authorized = registry.authorize("industry", rows, NOW)
    fallback = runtime.build_fallback("industry", runtime.OperatorAction.PROPOSE, request, authorized)
    assert fallback["claims"][-2]["text"] == rows[8].content and len(rows[8].content) > 480
    raw = {runtime.INDUSTRY_FACT_SELECTION_FIELD: runtime.INDUSTRY_CONTEXT_FACT_SELECTION_VERSION,
        "_model": "offline-protocol-only", "claims": [{"kind": "fact", "text": rows[1].content, "evidence_ids": ["fact-2"]}]}
    output = runtime.normalize_output(registry, "industry", runtime.OperatorAction.PROPOSE, request, authorized, fallback, raw)
    assert output.evidence_refs == ["fact-2", "fact-9", "fact-10"]
    assert [claim.text for claim in output.claims] == [rows[i].content for i in (1, 8, 9)]
    assert output.requires_human_review is True
    assert "owner_review" not in output.model_dump(mode="json")
    assert not any(risk.code.startswith("industry_fact_selection") for risk in output.risk_flags)
    legacy = proposal_request(rows, selection_enabled=False)
    assert "preparation_evidence_selection" not in legacy.model_dump(mode="json")
    legacy_fallback = runtime.build_fallback("industry", runtime.OperatorAction.PROPOSE, legacy, authorized)
    assert "fact-9" not in {identity for claim in legacy_fallback["claims"] for identity in claim["evidence_ids"]}


@pytest.mark.parametrize("change", ["missing", "type", "document", "source", "tier", "private", "genre", "action"])
def test_unavailable_selected_evidence_fails_before_model_writing(change):
    rows = proposal_records()
    request = proposal_request(rows)
    action = runtime.OperatorAction.PROPOSE
    if change == "missing": rows = rows[:8] + rows[9:]
    elif change == "type": rows[8] = rows[8].model_copy(update={"record_type": "source_item"})
    elif change == "document": rows[8] = rows[8].model_copy(update={"structured_data": {}})
    elif change == "source": rows[8] = rows[8].model_copy(update={"source_uri": "https://other.example/report"})
    elif change == "tier": rows[8] = rows[8].model_copy(update={"source_tier": "secondary"})
    elif change == "private": rows[8] = rows[8].model_copy(update={"structured_data": {**rows[8].structured_data, "publishable": False}})
    elif change == "genre": request.genre = "news"
    else: action = runtime.OperatorAction.REVISE
    with pytest.raises(runtime.ContextAuthorizationError):
        runtime.build_fallback("industry", action, request, rows)


def signed_preparation_plan(request, *, body=None):
    body = deepcopy(body or {key: value for key, value in request.surface_fact_plan.items()
        if key not in {"signature", "certificate", "plan_sha256"}})
    ids = [f"claim-{i + 1}" for i in range(len(request.claims))]
    body["scope"]["proposal_id"] = "proposal-offline"
    body["claim_pool"] = [{"claim_id": identity, "kind": "fact", "text_sha256": sha(claim.text),
        "evidence_ids": list(claim.evidence_ids)} for identity, claim in zip(ids, request.claims)]
    body["fact_roles"] = {"anchor": ids, **{role: [identity for identity, claim in zip(ids, request.claims) if pattern.search(claim.text)]
        for role, pattern in content._SURFACE_DEPENDENCIES.items()}}
    body["preparation_authorization"] = {"schema_version": "operator.surface_preparation_authority.v1",
        "mode": "mandate_preparation_only", "publication_authorized": False, "run_id": "run-offline", "mandate_version": 2,
        "proposal_authorization_id": "proposal-offline", "proposal_authorization_version": 1,
        "planner_request_sha256": "1" * 64, "planner_selection_sha256": "2" * 64, "research_digest_sha256": "3" * 64}
    body["preparation_selection_sha256"] = "4" * 64
    body["editorial_intent"] = {"schema_version": "operator.preparation_editorial_intent.v1",
        "author": "independent workflow editorial planner", "short_surface_reasons": {
            surface: "解释已选事实对读者验收判断的影响。" for surface in body["surfaces"] if surface not in {"master", "wechat_mp", "toutiao"}},
        "reported_case_claim_ids": [], "case_gap": "当前公开材料缺少具体交接案例。"}
    return resign_plan(body)


def resign_plan(body):
    body = deepcopy(body)
    return {**body, "plan_sha256": digest(body), "signature": hmac.new(b"offline-hermes-secret",
        ("surface_fact_plan.v1\n" + digest(body)).encode(), hashlib.sha256).hexdigest()}


def test_auto_plan_overlapping_roles_and_canonical_anchor_conditions_restore_full_groups(monkeypatch):
    request, _, _ = surface_fixture(monkeypatch)
    request.claims[0].text = "预览付费时自动复用提示词和素材，不能重传。"
    request.authorized_context[0].content = request.claims[0].text
    request.surface_fact_plan = signed_preparation_plan(request)
    registry = runtime.OperatorRegistry()
    contexts = registry.authorize("industry", request.authorized_context, request.as_of)
    body = content.content_request_payload("industry", request, contexts)
    refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"]]
    candidate = {"_model": "offline-normalizer-only", "master_title": "预览与素材复用", "blocks": refs,
        "platform_variants": [{"platform": surface, "title": "预览与素材复用",
            "blocks": refs if surface == "wechat_mp" else refs[:1]} for surface in request.channels]}
    result = _run(runtime.OperatorRole.INDUSTRY, request, candidate)
    assert result.critic.errors == []
    assert all({block.claim_id for block in variant.blocks} == {"claim-1", "claim-2", "claim-3"} for variant in result.platform_variants)
    assert body["surface_fact_plan"]["fact_roles"]["anchor"] == ["claim-1", "claim-2", "claim-3"]
    assert "claim-1" in body["surface_fact_plan"]["fact_roles"]["billing"]
    assert "claim-1" in body["surface_fact_plan"]["fact_roles"]["input_constraints"]
    assert "not from an owner review" in body["surface_fact_plan_rule"]


def test_cost_discussion_without_billing_evidence_keeps_explicit_dependency_error(monkeypatch):
    request, _, _ = surface_fixture(monkeypatch)
    request.claims[1].text = "验收按用户功能组织，并单独检查行为差异。"
    request.authorized_context[1].content = request.claims[1].text
    request.surface_fact_plan = signed_preparation_plan(request)
    registry = runtime.OperatorRegistry()
    contexts = registry.authorize("industry", request.authorized_context, request.as_of)
    body = content.content_request_payload("industry", request, contexts)
    refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"]]
    candidate = {"_model": "offline-normalizer-only", "master_title": "预览与验收", "blocks": refs,
        "platform_variants": [{"platform": surface, "title": "预览与验收",
            "blocks": refs if surface == "wechat_mp" else refs[:1]} for surface in request.channels]}
    candidate["platform_variants"][1]["blocks"].append({"kind": "opinion", "text": "审核成本仍需另行判断。", "evidence_ids": []})
    with pytest.raises(runtime.ContextAuthorizationError) as error:
        _run(runtime.OperatorRole.INDUSTRY, request, candidate)
    assert error.value.code == "surface_fact_plan_dependency_evidence_missing"


def test_required_whole_facts_must_fit_the_existing_output_bound_before_a_model():
    rows = proposal_records()
    rows = [rows[0].model_copy(update={"record_id": f"fact-{i}", "content": f"公开记录中的行为条件{i}仅适用于该项验收。"})
        for i in range(1, 52)]
    source = "\n".join(row.content for row in rows)
    for row in rows:
        row.structured_data = {"source_document": {"url": URI, "source_text": source, "source_text_hash": sha(source)}}
    request = runtime.OperatorProposeRequest(objective="解释给定的公开行为条件。", genre="industry_analysis",
        authorized_context=rows, as_of=NOW, preparation_evidence_selection={
            "schema_version": "operator.preparation_evidence_selection.v1", "planner_selection_sha256": "a" * 64,
            "required_evidence_ids": [row.record_id for row in rows[1:]]})
    with pytest.raises(runtime.ContextAuthorizationError) as error:
        runtime.build_fallback("industry", runtime.OperatorAction.PROPOSE, request, rows)
    assert error.value.code == "preparation_evidence_dependency_unrepresentable"


@pytest.mark.parametrize("change", ["owner_proof", "author", "authority", "published", "conditional_role", "case", "selection_hash"])
def test_even_a_signed_new_branch_requires_exact_preparation_metadata(monkeypatch, change):
    request, contexts, _ = surface_fixture(monkeypatch)
    plan = signed_preparation_plan(request)
    body = {key: value for key, value in plan.items() if key not in {"signature", "plan_sha256"}}
    if change == "owner_proof": body["editorial_intent"]["owner_review"] = {"approved": True}
    elif change == "author": body["editorial_intent"]["author"] = "Codex AI editor"
    elif change == "authority": body["preparation_authorization"]["mandate_version"] = True
    elif change == "published": body["preparation_authorization"]["publication_authorized"] = True
    elif change == "conditional_role": body["fact_roles"]["billing"] = []
    elif change == "case": body["editorial_intent"]["reported_case_claim_ids"] = ["claim-does-not-exist"]
    else: body.pop("preparation_selection_sha256")
    request.surface_fact_plan = resign_plan(body)
    with pytest.raises(runtime.ContextAuthorizationError):
        content.content_request_payload("industry", request, contexts)


def test_evidence_selection_ids_cannot_be_empty_duplicated_or_coerced():
    base = {"schema_version": "operator.preparation_evidence_selection.v1",
        "planner_selection_sha256": "a" * 64, "required_evidence_ids": ["fact-a"]}
    for invalid in ([], ["fact-a", "fact-a"], [True]):
        with pytest.raises(ValidationError):
            PreparationEvidenceSelection.model_validate({**base, "required_evidence_ids": invalid})


def test_writer_and_blind_reader_share_specific_progression_guidance_without_changing_facts(monkeypatch):
    from ai_marketing_api.operator_review import _reader_payload
    request, contexts, candidate = surface_fixture(monkeypatch)
    before = request.model_dump(mode="json")
    writer = content.content_request_payload("industry", request, contexts)
    output = _run(runtime.OperatorRole.INDUSTRY, request, candidate)
    reader = _reader_payload(output, {"audience": "公开工程报告的读者"}, scorecard_required=True)
    writer_rules = writer["industry_editorial_method"]["rules"]
    reader_rules = reader["editorial_review"]["industry_editorial_method"]["rules"]
    rule = next(value for value in writer_rules if "Explain the same boundary once" in value)
    assert rule in reader_rules
    assert "new verified detail, mechanism or consequence" in rule
    assert "otherwise remove it" in rule
    assert "an objection that the text has no reason to raise" in rule
    assert "Keep any meaningful limitation" in rule
    assert "end directly" in rule
    assert request.model_dump(mode="json") == before
    assert writer["claims"] == before["claims"] and writer["authorized_context"] == before["authorized_context"]


@pytest.mark.parametrize("condition,proposal_title,required_id", [
    ("billing", "预览与正式任务的计费边界", "claim-2"),
    ("input_constraints", "升级自动复用提示词和参考素材", "claim-3"),
])
@pytest.mark.parametrize("raw_title", [None, "", "未经批准的数值999999"])
def test_preparation_safe_final_title_restores_only_its_complete_condition(monkeypatch, condition, proposal_title, required_id, raw_title):
    request, _, _ = surface_fixture(monkeypatch)
    request.approved_proposal.title = proposal_title
    request.surface_fact_plan = signed_preparation_plan(request)
    registry = runtime.OperatorRegistry()
    contexts = registry.authorize("industry", request.authorized_context, request.as_of)
    writer = content.content_request_payload("industry", request, contexts)
    refs = [{"block_ref": block["block_ref"]} for block in writer["canonical_block_registry"]]
    candidate = {"_model": "offline-final-title-contract", "blocks": refs,
        "platform_variants": [{"platform": surface,
            "blocks": refs if surface == "wechat_mp" else refs[:1]} for surface in request.channels]}
    if raw_title is not None:
        candidate["master_title"] = raw_title
        for variant in candidate["platform_variants"]:
            variant["title"] = raw_title
    result = _run(runtime.OperatorRole.INDUSTRY, request, candidate)
    assert result.critic.errors == []
    assert result.master_title == proposal_title
    for variant in result.platform_variants:
        assert variant.title == proposal_title
        if variant.platform != "wechat_mp":
            assert {block.claim_id for block in variant.blocks} == {"claim-1", required_id}
            # The unrelated condition is still optional on the focused short.
            assert len(variant.blocks) == 2
    assert required_id in request.surface_fact_plan["fact_roles"][condition]
