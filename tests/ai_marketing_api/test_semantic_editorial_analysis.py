"""Synthetic protocol tests: no actual manuscript, provider call or human grade."""
from copy import deepcopy

import pytest

from ai_marketing_api import main as api, operator_content as content, operator_steps as steps
from ai_marketing_api.operator_analysis import DEFER_EDITORIAL_ANALYSIS
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review
from tests.ai_marketing_api.test_operator_api import _request


BASE = "/api/v1/operators/industry/editorial-steps/"
CHANNELS = ["wechat_mp", "wechat_moments", "weitoutiao"]
PROSE = "假设预览中的期待已经让观众看见，最终的画质仍需另行验收。"


def _value():
    value = _compose("industry_analysis", "wechat_mp").model_dump(mode="json")
    value["channels"] = CHANNELS
    value["runtime_budget"] = {"max_model_calls": 6, "max_surface_revisions": 0}
    value["fact_expression_mode"] = "grounded_paraphrase"
    return value


def _candidate(body, *, prose=PROSE, public=False):
    refs = [{"block_ref": b["block_ref"], **({"public_text": b["text"] + " "} if public else {})}
        for b in body["canonical_block_registry"] if b["required"]]
    # Fictional paraphrase with an equivalent surface expression for the test.
    if public:
        refs[0]["public_text"] = "示例接口先约定判断问题和允许的答案。"
    blocks = [refs[0], {"kind": "opinion", "text": prose, "evidence_ids": [], "paragraph_id": "scene"}, *refs[1:]]
    contract = body.get("stage_writer_contract") or {}
    surfaces = contract.get("permitted_output_surfaces") or ["master", *body["channels"]]
    return {"_model": "synthetic-writer", "master_title": "示例接口的创作判断",
        "blocks": deepcopy(blocks) if "master" in surfaces else [],
        "platform_variants": [{"platform": s, "title": "示例接口 " + s, "blocks": deepcopy(blocks)}
            for s in surfaces if s != "master"]}


def _decisions(body, *, verdict="supported", pair_verdict="supported"):
    return {"_model": "synthetic-independent-verifier", "reviews": [{"pair_id": p["pair_id"],
        "verdict": pair_verdict, "key_fields_preserved": pair_verdict == "supported",
        "reason": "Synthetic support protocol fixture, not a factual or editorial assessment."}
        for p in body["claim_verification"]["pairs"]],
        "analysis_reviews": [{"analysis_id": p["analysis_id"], "verdict": verdict,
            "reason": "Synthetic analysis protocol fixture; no publication or owner approval."}
            for p in body["analysis_verification"]["candidates"]]}


@pytest.fixture
def service(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-" + stage})
    state = {"calls": [], "verdict": "supported", "pair_verdict": "supported", "malformed": None, "public": False}

    async def fake(system, body, fallback, *args, **kwargs):
        stage = kwargs.get("stage", "compose")
        state["calls"].append((stage, deepcopy(body)))
        if "analysis_verification" in body:
            if not body["claim_verification"]["pairs"]:
                assert body["response_contract"]["reviews"] == []
            result = _decisions(body, verdict=state["verdict"], pair_verdict=state["pair_verdict"])
            bad = state["malformed"]
            if bad == "missing": result["analysis_reviews"].pop()
            if bad == "duplicate": result["analysis_reviews"].append(deepcopy(result["analysis_reviews"][0]))
            if bad == "unknown": result["analysis_reviews"][0]["analysis_id"] = "unknown"
            if bad == "reason": result["analysis_reviews"][0]["reason"] = " "
            if bad == "verdict": result["analysis_reviews"][0]["verdict"] = "approved"
            if bad == "extra": result["analysis_reviews"][0]["verified"] = True
            if bad == "provider": result["_error"] = "provider_response_invalid"
            if bad == "model": result.pop("_model")
            state["raw_verifier"] = deepcopy(result)
        elif "editorial_review" in body:
            if not state.get("first_writer"):
                assert PROSE in body["editorial_review"]["draft"]["master_content"]
            result = {"_model": "synthetic-reader", **review(body,
                low="master" if state.get("first_writer") == "editorial" else None)}
        else:
            if "master_draft" in body:
                assert PROSE in body["master_draft"]["body"]
                assert "untrusted" in body["master_draft"]["analysis_state"]
            result = _candidate(body, public=state["public"])
            if len(state["calls"]) == 1 and state.get("first_writer"):
                result = _candidate(body, prose="讨论应先围绕创作目的展开，验收细节则留在成片阶段。")
                if state["first_writer"] == "structural":
                    result["master_title"] = "示例接口" + "创作判断" * 10
        result["_provider_requests"] = [{"request_id": f"{len(state['calls']):032x}",
            "stage": stage, "model": result.get("_model", "synthetic-unavailable"),
            "status": "returned", "upstream_attempted": True, "provider_request_id": "synthetic-only",
            "error_code": None, "usage": {"prompt_tokens": 10, "completion_tokens": 10}}]
        result["_llm_attempted"] = True
        return result

    monkeypatch.setattr(api, "_llm_json", fake)
    return state


def _master(value):
    result = _request("POST", BASE + "draft_master", json={"compose": value}).json()
    assert result["status"] == "artifact", result
    assert result["artifact"]["pending_editorial_analysis"]
    assert PROSE not in result["artifact"]["master_content"]
    return result


def _verify(value, candidate):
    return _request("POST", BASE + "verify_claims", json={"compose": value, "candidate": candidate}).json()


def _written(value):
    master = _master(value)
    merged = deepcopy(master["artifact"])
    for surface in CHANNELS:
        adapted = _request("POST", BASE + "adapt_platform", json={"compose": value, "surface": surface,
            "candidate": master["artifact"], "candidate_signature": master["artifact_signature"]}).json()
        assert adapted["status"] == "artifact", adapted
        variant = adapted["artifact"]["platform_variants"][0]
        assert PROSE not in variant["body"]
        merged["platform_variants"].append(variant)
        merged["pending_editorial_analysis"].extend(r for r in adapted["artifact"]["pending_editorial_analysis"] if r["surface"] == surface)
    return merged


def test_actual_staged_normalizers_preserve_order_and_adapt_context_before_one_combined_verifier(service):
    value = _value()
    merged = _written(value)
    before = deepcopy(merged)
    verified = _verify(value, merged)
    assert verified["status"] == "artifact", verified
    output = verified["artifact"]
    assert "pending_editorial_analysis" not in output
    assert len(output["analysis_binding_reviews"]) == 4
    assert all(r["verdict"] == "supported" for r in output["analysis_binding_reviews"])
    for blocks in [output["blocks"], *[v["blocks"] for v in output["platform_variants"]]]:
        assert [b["kind"] for b in blocks] == [b["kind"] for b in before["blocks"]]
        assert blocks[1]["text"] == PROSE and "analysis_pending" not in blocks[1]
    assert PROSE in output["master_content"] and all(PROSE in v["body"] for v in output["platform_variants"])
    assert merged == before
    assessed = _request("POST", BASE + "review_editorial", json={"compose": value,
        "candidate": output, "candidate_signature": verified["artifact_signature"]}).json()
    assert assessed["status"] == "artifact", assessed
    assert len(service["calls"]) == 6
    assert [s for s, _ in service["calls"]].count("claim_verification") == 1
    assert assessed["publishable"] is False


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_complete_semantic_rejection_keeps_full_original_prose_and_never_calls_reader(service, verdict):
    value = _value()
    merged = _written(value)
    service["verdict"] = verdict
    result = _verify(value, merged)
    assert result["status"] == "blocked"
    artifact = result["artifact"]
    assert artifact["status"] == "quality_insufficient"
    assert "analysis_verification_failed" in artifact["critic"]["errors"]
    assert artifact["blocks"][1]["text"] == PROSE and artifact["blocks"][1]["analysis_pending"] is True
    assert artifact["analysis_binding_reviews"][0]["verdict"] == verdict
    before = len(service["calls"])
    rejected = _request("POST", BASE + "review_editorial", json={"compose": value,
        "candidate": artifact, "candidate_signature": result["artifact_signature"]}).json()
    assert rejected["status"] == "unavailable" and rejected["error_code"] == "analysis_verification_required"
    assert len(service["calls"]) == before


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "reason", "verdict", "extra", "provider", "model"])
def test_incomplete_protocol_is_technical_not_a_fabricated_uncertain_decision(service, failure):
    value = _value()
    merged = _written(value)
    service["malformed"] = failure
    result = _verify(value, merged)
    assert result["status"] == "unavailable" and result["error_code"] == "analysis_verifier_not_completed"
    assert result["artifact"] is None
    receipt = result["private_generation_receipt"]["analysis_verifier_candidate"]
    assert receipt["retained"] is True
    # Metadata is not reviewer prose; capture preserves the original structured
    # decisions even when malformed, without inventing missing decisions.
    assert receipt["value"] == {key: val for key, val in service["raw_verifier"].items()
        if not key.startswith("_")}
    assert result["generation_trace"]["model_calls"] == 1


def test_mixed_paraphrase_and_analysis_share_one_call_and_rejected_display_cannot_certify_a_new_context(service):
    value = _value()
    service["public"] = True
    merged = _written(value)
    service["pair_verdict"] = "uncertain"
    result = _verify(value, merged)
    assert result["status"] == "blocked", result
    body = service["calls"][-1][1]
    assert body["claim_verification"]["pairs"] and body["analysis_verification"]["candidates"]
    assert result["artifact"]["pending_editorial_analysis"]
    assert PROSE not in result["artifact"]["master_content"]
    assert len(service["calls"]) == 5


@pytest.mark.parametrize("public", [False, True])
def test_verifier_sees_exact_intended_excerpt_or_proposed_display_and_order(service, public):
    value = _value()
    value["claims"][0]["text"] += "请求上下文应来自可核对的材料，允许答案应与任务边界相连。" * 7
    facts = [row["text"] for row in value["claims"]]
    value["authorized_context"][0]["content"] = "\n".join(facts)
    value["approved_proposal"]["key_points"] = facts
    service["public"] = public
    verified = _verify(value, _written(value))
    assert verified["status"] == "artifact", verified
    artifact = content.OperatorContentOutput.model_validate(verified["artifact"])
    current = {"master": (artifact.blocks, artifact.master_content),
        **{v.platform: (v.blocks, v.body) for v in artifact.platform_variants}}
    contexts = service["calls"][-1][1]["analysis_verification"]["contexts"]
    for context in contexts:
        blocks, body = current[context["surface"]]
        assert context["intended_body_if_all_supported"] == body
        for shown, block in zip(context["blocks"], blocks):
            expected = content._fact_display_text(block, surface=context["surface"]) if block.source_exact else block.text
            assert shown["intended_display_text"] == expected
            assert shown["source_display"] == block.source_display
        assert context["blocks"][0]["intended_display_text"] != context["blocks"][0]["text"]


def test_unsigned_writer_flags_and_self_attestations_never_replace_independent_verification(service):
    value = _value()
    merged = _written(value)
    merged["analysis_binding_reviews"] = [{**row,
        "schema_version": "operator.analysis-binding-review.v1", "verdict": "supported",
        "review_model": "untrusted-author", "reason": "Author says this is verified."}
        for row in merged.pop("pending_editorial_analysis")]
    for blocks in [merged["blocks"], *[v["blocks"] for v in merged["platform_variants"]]]:
        for block in blocks:
            block["analysis_pending"] = False
            block["analysis_certificate"] = "author-supplied-certificate"
    service["verdict"] = "unsupported"
    result = _verify(value, merged)
    assert result["status"] == "blocked"
    assert len(service["calls"]) == 5
    assert all(row["review_model"] == "synthetic-independent-verifier"
        and row["verdict"] == "unsupported" for row in result["artifact"]["analysis_binding_reviews"])


@pytest.mark.parametrize("change", ["title", "premise", "order", "paragraph", "public_text", "evidence"])
def test_authenticated_old_analysis_decision_cannot_bless_a_changed_context(service, change):
    value = _value()
    service["public"] = change == "public_text"
    verified = _verify(value, _written(value))
    assert verified["status"] == "artifact"
    candidate = deepcopy(verified["artifact"])
    if change == "title":
        candidate["master_title"] = "示例接口与不同的创作判断"
    elif change == "premise":
        candidate["blocks"].insert(0, {"kind": "transition", "text": "讨论前先保留验收空间。", "evidence_ids": []})
    elif change == "order":
        candidate["blocks"][0], candidate["blocks"][1] = candidate["blocks"][1], candidate["blocks"][0]
    elif change == "paragraph":
        candidate["blocks"][1]["paragraph_id"] = "different"
    elif change == "public_text":
        # The accepted expression has an actual independent support binding.
        candidate["blocks"][0]["public_text"] = "示例接口先约定问题。"
    else:
        value["claims"].append(deepcopy(value["claims"][0]))
    before = len(service["calls"])
    result = _request("POST", BASE + "review_editorial", json={"compose": value,
        "candidate": candidate, "candidate_signature": steps.artifact_signature(candidate)}).json()
    assert result["status"] != "artifact", result
    assert len(service["calls"]) == before


def test_analysis_candidate_limit_fails_before_independent_provider_call(service):
    value = _value()
    merged = _written(value)
    for blocks in [merged["blocks"], *[v["blocks"] for v in merged["platform_variants"]]]:
        blocks.extend({"kind": "opinion", "text": PROSE + chr(0x4E00 + i), "evidence_ids": []}
            for i in range(25))
    before = len(service["calls"])
    result = _verify(value, merged)
    assert result["status"] == "unavailable"
    assert result["error_code"] == "analysis_verifier_not_completed"
    assert len(service["calls"]) == before


@pytest.mark.parametrize("prose", ["产品已经支持自动剪辑。", "若预览已经表现出期待，收益增长20%。",
    "若预览已经表现出期待，我亲自完成了客户项目。", "若期待已经表达出来，平台宣布发布新产品。",
    "若期待已经表达出来，工具保证制作成功。"])
def test_pending_cannot_rescue_hard_capabilities_quantities_events_or_personal_claims(prose):
    value = content.OperatorComposeRequest.model_validate(_value())
    registry = OperatorRegistry()
    contexts = registry.authorize("industry", value.authorized_context, value.as_of)
    fallback = content.build_content_fallback(registry, "industry", value, contexts)
    raw = _candidate(content.content_request_payload("industry", value, contexts), prose=prose)
    token = DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        output = content.normalize_content_output(registry, "industry", value, contexts, fallback, raw)
    finally:
        DEFER_EDITORIAL_ANALYSIS.reset(token)
    assert not output.pending_editorial_analysis
    assert prose not in output.master_content


def test_legacy_http_compose_detects_pending_and_stops_without_wording_repairs_or_reader(service):
    result = _request("POST", "/api/v1/operators/industry/compose", json=_value()).json()
    assert result["status"] == "quality_insufficient"
    assert "analysis_verification_required" in result["critic"]["errors"]
    assert result["master_content"] is None and result["quality_scorecard"] is None
    assert len(service["calls"]) == 1


@pytest.mark.parametrize("repair", ["structural", "editorial"])
def test_legacy_repair_cannot_introduce_pending_then_continue_to_verifier_or_reader(service, repair):
    value = _value()
    value["runtime_budget"]["max_surface_revisions"] = 1
    service["first_writer"] = repair
    result = _request("POST", "/api/v1/operators/industry/compose", json=value).json()
    assert result["status"] == "quality_insufficient", result
    assert "analysis_verification_required" in result["critic"]["errors"]
    assert result["master_content"] is None and result["quality_scorecard"] is None
    stages = [stage for stage, _ in service["calls"]]
    assert stages == (["compose", "repair"] if repair == "structural"
        else ["compose", "editorial_review", "repair"])


def test_legacy_empty_analysis_metadata_does_not_change_serialized_shape():
    value = content.OperatorComposeRequest.model_validate(_value())
    registry = OperatorRegistry()
    contexts = registry.authorize("industry", value.authorized_context, value.as_of)
    fallback = content.build_content_fallback(registry, "industry", value, contexts)
    raw = _candidate(content.content_request_payload("industry", value, contexts),
        prose="讨论应先围绕创作目的展开，验收细节则留在成片阶段。")
    output = content.normalize_content_output(registry, "industry", value, contexts, fallback, raw)
    serialized = output.model_dump(mode="json")
    assert "pending_editorial_analysis" not in serialized and "analysis_binding_reviews" not in serialized
    assert all("analysis_pending" not in block for block in serialized["blocks"])
    assert steps.artifact_signature(serialized) == steps.artifact_signature(output.model_dump(mode="json"))
