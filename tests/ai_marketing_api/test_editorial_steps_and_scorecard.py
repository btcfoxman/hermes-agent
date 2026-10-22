from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest

import ai_marketing_api.main as api
import ai_marketing_api.operator_steps as steps
from ai_marketing_api.operator_routing import server_input_byte_limit, server_stage_route
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_routing import BUDGET_RESERVATION_ID, PROVIDER_REQUEST_OBSERVER, SINGLE_PROVIDER_REQUEST
from ai_marketing_api.operator_runtime import OperatorProposeRequest, request_payload
from tests.ai_marketing_api.test_operator_api import _base_payload, _compose_payload, _publishable_model_candidate, _request
from tests.ai_marketing_api.test_autonomous_editorial import _brief, _review


@pytest.fixture(autouse=True)
def service_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def payload():
    value = _base_payload()
    value.update(approved_proposal={"title": "Official release", "angle": "Explain the release",
        "audience_value": "A concrete release decision", "key_points": ["The vendor released version 2."],
        "suggested_formats": ["wechat_mp"], "first_person": False},
        claims=[{"text": "The vendor released version 2.", "kind": "fact", "evidence_ids": ["source-1"], "verification_status": "verified"}])
    value.update(genre="company_data_brief", style_references=[{
        "reference_id": "style-only", "genre": "company_data_brief",
        "style_notes": ["Lead with a verified event, then compare the dates."],
        "excerpt": "A fictional product claims 9999 users. This is style data, never evidence.",
    }], public_editorial_brief=_brief(), runtime_budget={"max_model_calls": 2})
    return value


def review(body, *, low=None, missing=False):
    draft = body["editorial_review"]["draft"]
    surfaces = ["master", *[v["platform"] for v in draft["platform_variants"]]]
    if missing:
        surfaces = surfaces[:-1]
    rows = []
    for surface in surfaces:
        rows.append({"surface": surface, "reader_relevance": 4,
            "information_gain": 3 if low == surface else 4, "analytical_depth": 4,
            "natural_expression": 4, "platform_fit": 4, "major_issues": [], "issues": []})
    return {**_review(True), "quality_scorecard": {"schema_version": "quality_scorecard.v1", "surfaces": rows}}


def fact_first(body):
    candidate = _publishable_model_candidate(body)
    refs = [{"block_ref": b["block_ref"]} for b in body["canonical_block_registry"] if b["required"]]
    candidate["blocks"] = refs
    for variant in candidate["platform_variants"]:
        variant["blocks"] = refs
    return candidate


def test_style_references_never_become_evidence_in_propose_or_compose(monkeypatch):
    value = payload()
    proposed = OperatorProposeRequest.model_validate({k: v for k, v in value.items()
        if k not in {"claims", "approved_proposal", "public_editorial_brief", "runtime_budget"}})
    serialized = request_payload(proposed)
    assert serialized["style_context"]["style_references"][0]["reference_id"] == "style-only"
    assert all(c["record_id"] != "style-only" for c in serialized["authorized_context"])
    calls = []
    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            return review(body)
        assert "9999" not in json.dumps(body["claims"])
        assert "9999" not in json.dumps(body["authorized_context"])
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=value).json()
    assert data["status"] == "content_ready", json.dumps({"critic":data["critic"], "trace": data["generation_trace"]})
    assert data["quality_scorecard"]["passed"] is True
    assert len(data["quality_scorecard"]["content_sha256"]) == 64
    assert calls[1]["editorial_review"]["style_context"] == calls[0]["style_context"]
    assert data["blocks"][0]["source_exact"] is True
    assert "9999" not in data["master_content"]


@pytest.mark.parametrize("failure", ["low", "missing", "boolean", "major"])
def test_each_surface_requires_five_integer_scores_and_zero_major_issues(monkeypatch, failure):
    async def fake(system, body, fallback, *args, **kwargs):
        if "editorial_review" not in body:
            return fact_first(body)
        result = review(body, low="wechat_mp" if failure == "low" else None, missing=failure == "missing")
        if failure == "boolean":
            result["quality_scorecard"]["surfaces"][0]["analytical_depth"] = True
        if failure == "major":
            result["quality_scorecard"]["surfaces"][0]["major_issues"] = ["Unsupported comparison"]
        return result
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload()).json()
    assert data["status"] == "quality_insufficient"
    assert data["master_content"] is None
    if failure in {"low", "major"}:
        assert data["quality_scorecard"]["passed"] is False


def test_genre_does_not_allow_a_style_fact_to_enter_the_canonical_graph(monkeypatch):
    async def fake(system, body, fallback, *args, **kwargs):
        candidate = fact_first(body)
        candidate["blocks"].append({"kind": "fact", "text": "The product has 9999 users.", "evidence_ids": ["style-only"]})
        return candidate
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=payload()).json()
    assert data["status"] != "content_ready"
    assert data["master_content"] is None


def test_small_budget_does_not_spend_the_final_editor_call_on_structure(monkeypatch):
    value = _compose_payload()
    value.update(public_editorial_brief=_brief(), runtime_budget={"max_model_calls": 2})
    calls = []
    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=value).json()
    assert len(calls) == 1
    assert data["status"] != "content_ready"
    assert data["generation_trace"]["repair_calls"] == 0


def test_genre_and_references_reach_focused_editorial_repair(monkeypatch):
    value = payload()
    value["runtime_budget"] = {"max_model_calls": 4}
    value["editorial_feedback"] = "Explain the reader's actual next choice."
    calls, reviews = [], []
    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            reviews.append(body)
            return review(body, low="master" if len(reviews) == 1 else None)
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/industry/compose", json=value).json()
    repair = next(body for body in calls if "editorial_revision" in body)
    assert repair["style_context"] == calls[0]["style_context"]
    assert repair["editorial_feedback"] == value["editorial_feedback"]
    assert all(item["editorial_review"]["editorial_feedback"] == value["editorial_feedback"] for item in reviews)
    assert data["quality_scorecard"]["passed"] is True


def test_editorial_feedback_cannot_forward_a_non_public_excerpt(monkeypatch):
    value = payload()
    private = {**value["authorized_context"][0], "record_id": "source-private",
        "content": "Private operating memo containing unreleased customer identifiers.",
        "structured_data": {"publishable": False}}
    value["authorized_context"].append(private)
    value["editorial_feedback"] = "Improve the structure using " + private["content"]
    async def must_not_call(*args, **kwargs):
        raise AssertionError("Private feedback must stop before a provider request")
    monkeypatch.setattr(api, "_llm_json", must_not_call)
    response = _request("POST", "/api/v1/operators/industry/editorial-steps/draft_master", json={"compose": value})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "editorial_feedback_private_excerpt"


def test_real_normalizer_step_chain_never_publishes_and_limits_each_step(monkeypatch):
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": stage})
    calls = []
    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            return review(body)
        if "claim_verification" in body:
            return {"_model": "verifier", "reviews": []}
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", fake)
    value = payload()
    base = "/api/v1/operators/industry/editorial-steps/"
    master = _request("POST", base + "draft_master", json={"compose": value}).json()
    assert master["status"] == "artifact", master
    assert master["artifact"]["platform_variants"] == []
    adapted = _request("POST", base + "adapt_platform", json={"compose": value, "surface": "wechat_mp",
        "candidate": master["artifact"], "candidate_signature": master["artifact_signature"]}).json()
    assert adapted["status"] == "artifact", adapted
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    verified = _request("POST", base + "verify_claims", json={"compose": value, "candidate": merged}).json()
    assert verified["status"] == "artifact", verified
    assessed = _request("POST", base + "review_editorial", json={"compose": value,
        "candidate": verified["artifact"], "candidate_signature": verified["artifact_signature"]}).json()
    assert assessed["status"] == "artifact", assessed
    assert assessed["artifact"]["quality_scorecard"]["passed"] is True
    for result in (master, adapted, verified, assessed):
        assert result["publishable"] is False
        assert result["generation_trace"]["model_calls"] <= 1
    before = len(calls)
    tampered = copy.deepcopy(verified["artifact"])
    tampered["master_title"] += "tampered"
    rejected = _request("POST", base + "review_editorial", json={"compose": value,
        "candidate": tampered, "candidate_signature": verified["artifact_signature"]}).json()
    assert rejected["status"] == "unavailable"
    assert len(calls) == before


def test_step_rejects_provider_overrides_and_unrequested_platform(monkeypatch):
    base = "/api/v1/operators/industry/editorial-steps/"
    response = _request("POST", base + "draft_master", json={"compose": payload()},
        headers={"X-Hermes-Model": "untrusted"})
    assert response.status_code == 400
    response = _request("POST", base + "adapt_platform", json={"compose": payload(), "surface": "douyin"})
    assert response.status_code == 403


def test_real_http_adapter_records_actual_requests_usage_and_safe_errors(monkeypatch):
    monkeypatch.setattr(api, "server_stage_route", lambda stage: None)
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34", "93.184.216.35"})
    seen, observed = [], []
    original = httpx.AsyncClient
    def handle(request):
        seen.append(request)
        return httpx.Response(200, headers={"x-request-id": "provider-123"},
            json={"choices": [{"message": {"content": '{"ok": true}'}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    async def observer(event):
        observed.append(event)
    async def run():
        token = PROVIDER_REQUEST_OBSERVER.set(observer)
        single = SINGLE_PROVIDER_REQUEST.set(True)
        try:
            bounded = BoundedModelRun(RuntimeBudget(max_model_calls=1))
            await bounded.call(api._llm_json, "system", {}, {}, "https://model.example/v1", "private-secret", "writer")
            return bounded.trace(blocked=False, review="not_requested", stop_reason="done")
        finally:
            PROVIDER_REQUEST_OBSERVER.reset(token)
            SINGLE_PROVIDER_REQUEST.reset(single)
    trace = asyncio.run(run()).model_dump(mode="json")
    assert len(seen) == 1
    assert [event["phase"] for event in observed] == ["before_request", "after_request"]
    event = trace["provider_requests"][0]
    assert event["provider_request_id"] == "provider-123"
    assert event["usage"]["total_tokens"] == 15
    assert len(event["request_id"]) == 32
    assert "private-secret" not in json.dumps(trace)
    assert "cny" not in json.dumps(trace).lower()


def test_single_http_mode_keeps_unknown_usage_on_connection_failure(monkeypatch):
    monkeypatch.setattr(api, "server_stage_route", lambda stage: None)
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34", "93.184.216.35"})
    original, seen = httpx.AsyncClient, []
    def handle(request):
        seen.append(request)
        raise httpx.ConnectError("private-secret https://private.endpoint")
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    async def run():
        token = SINGLE_PROVIDER_REQUEST.set(True)
        try:
            return await api._llm_json("system", {}, {}, "https://model.example/v1", "key", "writer")
        finally:
            SINGLE_PROVIDER_REQUEST.reset(token)
    result = asyncio.run(run())
    assert len(seen) == 1
    assert result["_provider_requests"][0]["usage"] is None
    assert result["_provider_requests"][0]["status"] == "unknown"
    assert "private-secret" not in json.dumps(result)


def test_server_stage_route_ignores_caller_model_and_records_reservation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("EDITOR_TEST_KEY", "server-only-secret")
    (tmp_path / "config.yaml").write_text("marketing_api:\n  stage_routes:\n    editorial_review:\n      model: independent-editor\n      base_url: https://editor.example/v1\n      api_key_env: EDITOR_TEST_KEY\n", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34"})
    original, captured = httpx.AsyncClient, []
    def handle(request):
        captured.append(request)
        assert request.headers["Authorization"] == "Bearer server-only-secret"
        assert json.loads(request.content)["model"] == "independent-editor"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    async def run():
        token = BUDGET_RESERVATION_ID.set("reserved-editor-call")
        try:
            return await api._llm_json("system", {}, {}, "https://caller.example/v1", "caller-key", "caller-model", stage="editorial_review")
        finally:
            BUDGET_RESERVATION_ID.reset(token)
    result = asyncio.run(run())
    assert len(captured) == 1
    assert result["_provider_requests"][0]["reservation_id"] == "reserved-editor-call"
    assert result["_provider_requests"][0]["usage"] is None
    assert "server-only-secret" not in json.dumps(result)


def test_server_stage_model_environment_isolated_and_explicit_yaml_still_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MODEL", "default-writer")
    monkeypatch.setenv("HERMES_OPENAI_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("HERMES_EDITORIAL_REVIEW_MODEL", "independent-editor")
    assert server_stage_route("compose")["model"] == "default-writer"
    assert server_stage_route("editorial_review")["model"] == "independent-editor"
    monkeypatch.setenv("HERMES_EDITORIAL_REVIEW_MODEL", " ")
    with pytest.raises(ValueError, match="stage_route_unavailable"):
        server_stage_route("editorial_review")
    (tmp_path / "config.yaml").write_text("marketing_api:\n  stage_routes:\n    editorial_review:\n      model: explicit-editor\n      base_url: https://editor.example/v1\n      api_key_env: HERMES_OPENAI_API_KEY\n", encoding="utf-8")
    assert server_stage_route("editorial_review")["model"] == "explicit-editor"


def test_input_byte_bound_stops_before_http_and_cannot_be_enlarged(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(api, "server_stage_route", lambda stage: None)
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34"})
    def must_not_construct_client(**kwargs):
        raise AssertionError("An oversized prompt must not reach provider dispatch")
    monkeypatch.setattr(api.httpx, "AsyncClient", must_not_construct_client)
    (tmp_path / "config.yaml").write_text("marketing_api:\n  provider_limits:\n    max_input_bytes: 999999\n", encoding="utf-8")
    assert server_input_byte_limit() == 150000
    result = asyncio.run(api._llm_json("system", {"text": "字" * 50000}, {}, "https://model.example/v1", "test-only"))
    assert result["_error"] == "input_budget_exceeded"
    assert result["_provider_requests"] == []
    assert not result.get("_llm_attempted")
    (tmp_path / "config.yaml").write_text("marketing_api:\n  provider_limits:\n    max_input_bytes: 100\n", encoding="utf-8")
    assert server_input_byte_limit() == 100
    result = asyncio.run(api._llm_json("system", {"text": "a" * 100}, {}, "https://model.example/v1", "test-only"))
    assert result["_error"] == "input_budget_exceeded"


@pytest.mark.parametrize("limit,expected", [(None, 12000), (999999, 24000)])
def test_physical_requests_always_have_an_absolute_output_cap(monkeypatch, tmp_path, limit, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(api, "server_stage_route", lambda stage: None)
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34"})
    original = httpx.AsyncClient
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    result = asyncio.run(api._llm_json("system", {}, {}, "https://model.example/v1", "test-only", max_output_tokens=limit))
    assert seen[0]["max_completion_tokens"] == expected
    assert result["_provider_requests"][0]["max_output_tokens"] == expected


def test_authenticated_internal_budget_headers_propagate_to_propose(monkeypatch):
    observed = []
    async def fake(system, body, fallback, *args, **kwargs):
        observed.append((SINGLE_PROVIDER_REQUEST.get(), BUDGET_RESERVATION_ID.get(), kwargs["stage"]))
        return fallback
    monkeypatch.setattr(api, "_llm_json", fake)
    response = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload(),
        headers={"X-Hermes-Single-Provider-Request": "true", "X-Hermes-Budget-Reservation-Id": "reservation-1"})
    assert response.status_code == 200
    assert observed == [(True, "reservation-1", "propose")]
    assert response.json()["generation_trace"]["attempts"][0]["stage"] == "propose"


def test_missing_yaml_uses_existing_server_route_but_invalid_override_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MODEL", "existing-server-model")
    monkeypatch.setenv("HERMES_OPENAI_BASE_URL", "https://existing.example/v1")
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "existing-server-key")
    for stage in ("compose", "claim_verification", "editorial_review", "planner"):
        route = server_stage_route(stage)
        assert route["model"] == "existing-server-model"
        assert route["server_default"] is True
    (tmp_path / "config.yaml").write_text("marketing_api:\n  stage_routes:\n    editorial_review:\n      model: broken-override\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage_route_unavailable"):
        server_stage_route("editorial_review")
    assert server_stage_route("compose")["model"] == "existing-server-model"


def test_step_paraphrase_is_verified_before_the_editor_sees_rendered_copy(monkeypatch):
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": stage})
    value = payload()
    source = "公开工具允许负责人自行选择待审核稿件。"
    public = "负责人可以自行选择需要审核的稿件。"
    value["authorized_context"][0]["content"] = source
    value["claims"][0]["text"] = source
    value["approved_proposal"]["title"] = source
    value["approved_proposal"]["key_points"] = [source]
    value["topic"] = source
    value["fact_expression_mode"] = "grounded_paraphrase"
    reviewed_bodies = []
    async def fake(system, body, fallback, *args, **kwargs):
        if "claim_verification" in body:
            return {"_model": "independent-verifier", "reviews": [
                {"pair_id": pair["pair_id"], "verdict": "supported", "key_fields_preserved": True, "reason": "same meaning"}
                for pair in body["claim_verification"]["pairs"]]}
        if "editorial_review" in body:
            reviewed_bodies.append(body["editorial_review"]["draft"])
            return review(body)
        candidate = fact_first(body)
        for block in candidate["blocks"]:
            block["public_text"] = public
        for variant in candidate["platform_variants"]:
            for block in variant["blocks"]:
                block["public_text"] = public
        return candidate
    monkeypatch.setattr(api, "_llm_json", fake)
    base = "/api/v1/operators/industry/editorial-steps/"
    master = _request("POST", base + "draft_master", json={"compose": value}).json()
    adapted = _request("POST", base + "adapt_platform", json={"compose": value, "surface": "wechat_mp",
        "candidate": master["artifact"], "candidate_signature": master["artifact_signature"]}).json()
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    assert source in merged["master_content"] and public not in merged["master_content"]
    verified = _request("POST", base + "verify_claims", json={"compose": value, "candidate": merged}).json()
    assert verified["artifact"]["master_content"] == public
    assert verified["generation_trace"]["model_calls"] == 1
    assessed = _request("POST", base + "review_editorial", json={"compose": value,
        "candidate": verified["artifact"], "candidate_signature": verified["artifact_signature"]}).json()
    assert reviewed_bodies[0]["master_content"] == public
    assert assessed["artifact"]["generation_trace"]["editorial_review"] == "passed"
    assert assessed["artifact"]["generation_trace"]["evidence_mode"] == "grounded_paraphrase"
    assert assessed["artifact"]["claim_binding_reviews"][0]["verdict"] == "supported"
