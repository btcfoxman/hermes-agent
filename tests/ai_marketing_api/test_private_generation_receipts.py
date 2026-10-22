"""Actual ASGI -> offline provider HTTP -> normalizer -> durable private receipt."""
from __future__ import annotations

import asyncio
import base64
import copy
import gzip
import hashlib
import json

import httpx
import pytest

import ai_marketing_api.main as api
from ai_marketing_api.operator_private_receipts import (
    MAX_PRIVATE_BUNDLE_BYTES, PRIVATE_GENERATION_CAPTURE,
)
from ai_marketing_api.operator_receipts import request_sha256
from ai_marketing_api.operator_steps import artifact_signature
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import fact_first, payload, review
from tests.ai_marketing_api.test_operator_api import _request


BASE = "/api/v1/operators/industry/editorial-steps/"
AUTH = {"Authorization": "Bearer operator-test-secret"}


@pytest.fixture(autouse=True)
def isolated_edge(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "provider-private-secret")
    monkeypatch.setenv("HERMES_OPENAI_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("HERMES_MODEL", "offline-writer")
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda host, port: {"93.184.216.34"})
    assert PRIVATE_GENERATION_CAPTURE.get() is None


def install_provider(monkeypatch, handler):
    original = httpx.AsyncClient
    calls = []
    async def provider(request):
        calls.append(request)
        return await handler(request) if asyncio.iscoroutinefunction(handler) else handler(request)
    def client(*args, **kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(provider)
        return original(*args, **kwargs)
    monkeypatch.setattr(api.httpx, "AsyncClient", client)
    return calls


def model_body(request):
    return json.loads(json.loads(request.content)["messages"][1]["content"])


def provider_reply(content, *, finish="stop", compressed=False):
    raw = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30}}, ensure_ascii=False).encode()
    headers = {"X-Oneapi-Request-Id": "provider-real-id", "Content-Type": "application/json"}
    if compressed:
        headers["Content-Encoding"] = "gzip"
    return httpx.Response(200, content=gzip.compress(raw) if compressed else raw, headers=headers)


def private_bundle(response):
    return {key: value for key, value in response.items() if key.startswith("private_")}


def test_writer_candidate_survives_actual_normalizer_and_durable_cache(monkeypatch, capsys):
    marker = "RAW_WRITER_REMOVED_PROSE 987654321"
    raw_candidates = []
    def provider(request):
        candidate = fact_first(model_body(request))
        candidate["blocks"].append({"kind": "opinion", "text": marker, "evidence_ids": []})
        raw_candidates.append(candidate)
        return provider_reply(json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    body = {"compose": payload(), "budget_reservation_id": "text_private_cache"}
    first = _request("POST", BASE + "draft_master", json=body)
    assert first.status_code == 200
    data = first.json()
    receipt = data["private_generation_receipt"]
    writer = receipt["writer_candidate"]
    assert writer["source"] == "model" and writer["retained"] is True
    assert marker in json.dumps(writer["value"])
    assert marker not in json.dumps({"artifact": data["artifact"], "trace": data["generation_trace"]})
    assert data["artifact_signature"] == artifact_signature(data["artifact"])
    assert data["artifact"]["critic"]["warnings"]
    assert receipt["request_sha256"] == request_sha256(body)
    assert receipt["budget_reservation_id"] == body["budget_reservation_id"]
    provider_row = receipt["provider_responses"][0]
    assert provider_row["provider_request_id"] == "provider-real-id"
    assert provider_row["parse_stage"] == "complete" and provider_row["finish_reason"] == "stop"
    assert provider_row["content"]["type"] == "str"
    assert data == _request("POST", BASE + "draft_master", json=body).json()
    fetched = _request("GET", BASE + "draft_master/receipts/" + body["budget_reservation_id"],
                       headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert fetched.json()["response"] == data and fetched.json()["status"] == "completed"
    denied = _request("GET", BASE + "draft_master/receipts/" + body["budget_reservation_id"],
                      operator_auth=False, headers={"X-Hermes-Request-SHA256":request_sha256(body)})
    assert denied.status_code == 401 and marker not in denied.text
    assert len(calls) == 1 and PRIVATE_GENERATION_CAPTURE.get() is None
    assert calls[0].url.host == "93.184.216.34" and calls[0].headers["Host"] == "model.example"
    assert calls[0].extensions["sni_hostname"] == "model.example"
    assert "provider-private-secret" not in json.dumps(data)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("kind,expected_stage", [
    ("invalid_http_json", "envelope_json"), ("missing_choices", "message_content"),
    ("invalid_model_json", "model_json"), ("empty_content", "model_json"),
    ("non_object_model_json", "model_json"), ("null_content", "model_json"),
])
def test_invalid_provider_response_is_retained_before_parsing(monkeypatch, kind, expected_stage):
    def provider(request):
        if kind == "invalid_http_json":
            return httpx.Response(200, content=b"{PRIVATE_INVALID_HTTP", headers={"X-Oneapi-Request-Id": "receipt-id"})
        if kind == "missing_choices":
            return httpx.Response(200, json={"choices": []})
        content = {"invalid_model_json": "PRIVATE_INVALID_MODEL {", "empty_content": "",
                   "non_object_model_json": "[]", "null_content": None}[kind]
        return provider_reply(content, finish="length")
    calls = install_provider(monkeypatch, provider)
    data = _request("POST", BASE + "draft_master", json={"compose": payload()}).json()
    receipt = data["private_generation_receipt"]
    row = receipt["provider_responses"][0]
    assert row["response_received"] and row["http_status"] == 200
    assert row["parse_stage"] == expected_stage and row["error_code"]
    assert row["body"]["complete"] and row["body"]["retained"]
    raw = base64.b64decode(row["body"]["value"])
    assert hashlib.sha256(raw).hexdigest() == row["body"]["sha256"]
    assert row["body"]["bytes"] == len(raw)
    assert receipt["writer_candidate"]["source"] == "fallback"
    assert "PRIVATE_INVALID" not in json.dumps(data["artifact"])
    assert len(calls) == 1 and PRIVATE_GENERATION_CAPTURE.get() is None


class InterruptedBody(httpx.AsyncByteStream):
    def __init__(self, *, cancel=False):
        self.cancel = cancel
    async def __aiter__(self):
        yield b'{"choices":['
        if self.cancel:
            await asyncio.sleep(10)
        raise httpx.ReadTimeout("PRIVATE upstream details must not appear")


@pytest.mark.parametrize("cancel", [False, True])
def test_received_headers_and_partial_body_diagnostics_survive_timeout(monkeypatch, cancel):
    calls = install_provider(monkeypatch, lambda request: httpx.Response(200,
        stream=InterruptedBody(cancel=cancel), headers={"X-Oneapi-Request-Id": "partial-receipt-id"}))
    compose = payload()
    compose["runtime_budget"] = {"max_elapsed_seconds": 1, "max_model_calls": 1}
    data = _request("POST", BASE + "draft_master", json={"compose": compose}).json()
    row = data["private_generation_receipt"]["provider_responses"][0]
    assert row["http_status"] == 200 and row["provider_request_id"] == "partial-receipt-id"
    assert row["body"]["bytes"] == len(b'{"choices":[')
    assert not row["body"]["complete"] and not row["body"]["retained"]
    assert row["body"]["hash_scope"] == "observed_prefix" and "value" not in row["body"]
    assert row["error_code"] == ("provider_cancelled" if cancel else "provider_timeout")
    assert "PRIVATE upstream" not in json.dumps(data)
    assert len(calls) == 1 and PRIVATE_GENERATION_CAPTURE.get() is None


@pytest.mark.parametrize("size", [21000, 100000])
def test_private_total_limit_omits_complete_values_instead_of_truncating(monkeypatch, size):
    captured = []
    def provider(request):
        candidate = fact_first(model_body(request))
        candidate["private_test_marker"] = "Z" * size
        captured.append(candidate)
        return provider_reply(json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    data = _request("POST", BASE + "draft_master", json={"compose": payload()}).json()
    bundle = private_bundle(data)
    assert len(json.dumps(bundle,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()) <= MAX_PRIVATE_BUNDLE_BYTES
    receipt = data["private_generation_receipt"]
    writer = receipt["writer_candidate"]
    clean = {k:v for k,v in captured[0].items() if not k.startswith("_")}
    raw = json.dumps(clean,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()
    assert writer["sha256"] == hashlib.sha256(raw).hexdigest() and writer["bytes"] == len(raw)
    if size == 100000:
        assert not writer["retained"] and "value" not in writer
        assert writer["omitted_reason"] == "capture_limit_exceeded"
    else:
        assert writer["value"] == clean and writer["retained"]
        assert receipt["provider_responses"][0]["body"]["omitted_reason"] == "private_bundle_limit"
    assert len(calls) == 1


def test_gzip_response_hashes_decoded_body_without_double_decoding(monkeypatch):
    calls = install_provider(monkeypatch, lambda request: provider_reply(json.dumps(fact_first(model_body(request))), compressed=True))
    data = _request("POST", BASE + "draft_master", json={"compose": payload()}).json()
    assert data["status"] == "artifact"
    row = data["private_generation_receipt"]["provider_responses"][0]
    assert row["parse_stage"] == "complete" and row["body"]["byte_source"] == "decoded_http_body"
    assert json.loads(base64.b64decode(row["body"]["value"]))["choices"]
    assert len(calls) == 1


def test_review_protocol_failure_retains_actual_content_and_never_manufactures_card(monkeypatch):
    def provider(request):
        if "editorial_review" in model_body(request):
            return provider_reply("PRIVATE_REVIEW_NON_JSON", finish="length")
        return provider_reply(json.dumps(fact_first(model_body(request))))
    calls = install_provider(monkeypatch, provider)
    compose = payload()
    master = _request("POST", BASE + "draft_master", json={"compose": compose}).json()
    adapted = _request("POST", BASE + "adapt_platform", json={"compose": compose, "surface":"wechat_mp",
        "candidate":master["artifact"],"candidate_signature":master["artifact_signature"]}).json()
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    verified = _request("POST", BASE + "verify_claims", json={"compose":compose,"candidate":merged}).json()
    body = {"compose":compose,"candidate":verified["artifact"],"candidate_signature":verified["artifact_signature"],
            "budget_reservation_id":"text_private_bad_review"}
    data = _request("POST", BASE + "review_editorial", json=body).json()
    assert data["status"] == "blocked" and data["artifact"]["quality_scorecard"] is None
    assert "private_review_receipt" in data, {"calls":len(calls),"critic":data["artifact"]["critic"]}
    assert data["private_review_receipt"]["validation"]["code"] == "reviewer_unavailable"
    assert data["private_generation_receipt"]["provider_responses"][0]["content"]["value"] == "PRIVATE_REVIEW_NON_JSON"
    assert "PRIVATE_REVIEW_NON_JSON" not in json.dumps({"artifact":data["artifact"],"trace":data["generation_trace"]})
    assert len(calls) == 3


def test_request_context_isolation_and_legacy_projection(monkeypatch):
    async def provider(request):
        body = model_body(request)
        await asyncio.sleep(0.01)
        if "editorial_review" in body:
            return provider_reply(json.dumps(review(body)))
        candidate = fact_first(body)
        candidate["private_test_marker"] = body.get("editorial_feedback")
        return provider_reply(json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url="http://offline") as client:
            requests = []
            for marker in ("private_request_a", "private_request_b"):
                compose = payload()
                compose["editorial_feedback"] = marker
                requests.append(client.post(BASE + "draft_master",json={"compose":compose,"budget_reservation_id":"text_"+marker},headers=AUTH))
            return await asyncio.gather(*requests)
    results = asyncio.run(run())
    for response, own, other in zip(results,("private_request_a","private_request_b"),("private_request_b","private_request_a")):
        data = response.json()
        writer = data["private_generation_receipt"]["writer_candidate"]["value"]
        assert writer["private_test_marker"] == own
        assert other not in json.dumps(data["private_generation_receipt"])
    legacy = _request("POST","/api/v1/operators/industry/compose",json=payload()).json()
    assert not any(key.startswith("private_") for key in legacy)
    assert "private_generation_receipt" not in json.dumps(legacy)
    assert PRIVATE_GENERATION_CAPTURE.get() is None


def test_provider_credential_echo_is_hashed_but_never_retained(monkeypatch):
    calls = install_provider(monkeypatch, lambda request: provider_reply('provider-private-secret'))
    data = _request("POST",BASE+"draft_master",json={"compose":payload()}).json()
    row = data["private_generation_receipt"]["provider_responses"][0]
    assert row["body"]["omitted_reason"] == row["content"]["omitted_reason"] == "credential_echo"
    assert "provider-private-secret" not in json.dumps(data)
    assert len(calls) == 1


@pytest.mark.parametrize("size", [20000, 55000])
def test_review_and_generation_receipts_share_one_limit_without_changing_raw_scores(monkeypatch, size):
    raw_reviews = []
    def provider(request):
        body = model_body(request)
        if "editorial_review" in body:
            result = review(body)
            result["diagnostic_extra"] = "Q" * size
            raw_reviews.append(result)
            return provider_reply(json.dumps(result))
        return provider_reply(json.dumps(fact_first(body)))
    calls = install_provider(monkeypatch, provider)
    compose = payload()
    master = _request("POST",BASE+"draft_master",json={"compose":compose}).json()
    adapted = _request("POST",BASE+"adapt_platform",json={"compose":compose,"surface":"wechat_mp",
        "candidate":master["artifact"],"candidate_signature":master["artifact_signature"]}).json()
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    verified = _request("POST",BASE+"verify_claims",json={"compose":compose,"candidate":merged}).json()
    data = _request("POST",BASE+"review_editorial",json={"compose":compose,
        "candidate":verified["artifact"],"candidate_signature":verified["artifact_signature"]}).json()
    bundle = private_bundle(data)
    assert len(json.dumps(bundle,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()) <= MAX_PRIVATE_BUNDLE_BYTES
    assert data["private_review_receipt"]["candidate_retained"]
    assert data["private_review_receipt"]["candidate"] == {key:value for key,value in raw_reviews[0].items() if not key.startswith("_")}
    assert data["artifact"]["quality_scorecard"] is None  # extra fields still fail strict validation
    assert data["private_review_receipt"]["validation"]["code"] == "review_schema_invalid"
    assert len(calls) == 3
