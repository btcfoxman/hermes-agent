"""Real isolated SQLite and ASGI receipt recovery; all model IO is offline."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import stat
import sys

import httpx
import pytest

import ai_marketing_api.main as api
import ai_marketing_api.operator_steps as steps
from ai_marketing_api.operator_receipts import (
    EditorialReceiptStore, execute_with_receipt, request_sha256,
)
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import fact_first, payload, review
from tests.ai_marketing_api.test_operator_api import _request


BASE = "/api/v1/operators/industry/editorial-steps/draft_master"
AUTH = {"Authorization": "Bearer operator-test-secret"}


@pytest.fixture(autouse=True)
def isolated_edge(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "edge-home"))
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline"})


def paid_payload():
    return {"compose": payload(), "budget_reservation_id": "text_offline_receipt"}


def receipt_path(body):
    return BASE + "/receipts/" + body["budget_reservation_id"]


def test_complete_receipt_reuses_exact_private_response_across_real_process_restart(monkeypatch):
    calls = []
    async def model(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", model)
    body = paid_payload()
    first = _request("POST", BASE, json=body)
    assert first.status_code == 200 and first.json()["status"] == "artifact"
    second = _request("POST", BASE, content=json.dumps(body, ensure_ascii=False, indent=2),
                      headers={"Content-Type": "application/json"})
    assert second.json() == first.json() and len(calls) == 1
    fetched = _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert fetched.status_code == 200 and fetched.headers["cache-control"] == "no-store"
    assert fetched.json()["response"] == first.json() and fetched.json()["status"] == "completed"
    # The complete API is imported in another interpreter, with no process cache.
    script = '''import asyncio,json,sys,httpx
import ai_marketing_api.main as api
body=json.load(sys.stdin)
async def forbidden(*args,**kwargs): raise AssertionError("model cannot run on replay")
api._llm_json=forbidden
async def run():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url="http://offline") as client:
        response=await client.post("/api/v1/operators/industry/editorial-steps/draft_master",json=body,headers={"Authorization":"Bearer operator-test-secret"})
        assert response.status_code==200 and response.json()["status"]=="artifact"
        print(json.dumps({"replayed":True}))
asyncio.run(run())'''
    restarted = subprocess.run([sys.executable, "-c", script], input=json.dumps(body),
        text=True, capture_output=True, timeout=15, cwd=Path(__file__).resolve().parents[2])
    assert restarted.returncode == 0, restarted.stderr
    assert json.loads(restarted.stdout) == {"replayed": True}


def test_concurrent_http_requests_run_only_once_and_get_never_invokes_model(monkeypatch):
    calls = []
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def model(system, body, fallback, *args, **kwargs):
            calls.append(body)
            started.set()
            await release.wait()
            return fact_first(body)
        monkeypatch.setattr(api, "_llm_json", model)
        body = paid_payload()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://offline") as client:
            first = asyncio.create_task(client.post(BASE, json=body, headers=AUTH))
            await asyncio.wait_for(started.wait(), 3)
            try:
                duplicate = await client.post(BASE, json=body, headers=AUTH)
                assert duplicate.status_code == 409 and duplicate.json()["detail"]["code"] == "editorial_receipt_running"
                fetched = await client.get(receipt_path(body), headers={**AUTH, "X-Hermes-Request-SHA256": request_sha256(body)})
                assert fetched.json()["status"] == "running" and "response" not in fetched.json()
                assert len(calls) == 1
            finally:
                release.set()
                response = await first
            assert response.status_code == 200
    asyncio.run(run())


def test_separate_sqlite_connections_race_without_duplicate_claim_or_restart_steal():
    body = paid_payload()
    sha = request_sha256(body)
    def claim(_):
        return EditorialReceiptStore().claim("industry", "draft_master", body["budget_reservation_id"], sha)[0]
    with ThreadPoolExecutor(max_workers=6) as workers:
        tokens = list(workers.map(claim, range(6)))
    assert sum(token is not None for token in tokens) == 1
    restarted = EditorialReceiptStore()
    token, receipt = restarted.claim("industry", "draft_master", body["budget_reservation_id"], sha)
    assert token is None and receipt["status"] == "running"
    # No instance-id inference or lease expiry may purchase a replacement call.
    assert restarted.get("industry", "draft_master", body["budget_reservation_id"], sha)["status"] == "running"


def test_wire_hash_binds_exact_payload_and_auth_not_pydantic_defaults(monkeypatch):
    calls = []
    async def model(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", model)
    body = paid_payload()
    assert _request("POST", BASE, json=body).status_code == 200
    changed = {**body, "surface": "master"}  # Equivalent parsed model, different full wire request.
    assert _request("POST", BASE, json=changed).status_code == 409
    assert _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(changed)}).status_code == 409
    assert _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": "0" * 64}).status_code == 409
    assert _request("GET", receipt_path(body), operator_auth=False,
                    headers={"X-Hermes-Request-SHA256": request_sha256(body)}).status_code == 401
    assert _request("GET", receipt_path(body)).status_code == 422
    assert len(calls) == 1


def test_cancellation_is_durable_unknown_and_cannot_be_posted_again(monkeypatch):
    body = paid_payload()
    async def cancelled():
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute_with_receipt(role_id="industry", step="draft_master",
            reservation_id=body["budget_reservation_id"], wire_payload=body, execute=cancelled))
    async def forbidden(*args, **kwargs):
        pytest.fail("unknown receipt dispatched another model request")
    monkeypatch.setattr(api, "_llm_json", forbidden)
    fetched = _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert fetched.json()["status"] == "unknown" and "response" not in fetched.json()
    assert _request("POST", BASE, json=body).json()["detail"]["code"] == "editorial_receipt_unknown"


def test_failed_rpc_stores_unknown_without_exception_or_auth_secrets(monkeypatch, capsys):
    body = paid_payload()
    async def failing(*args, **kwargs):
        raise RuntimeError("private-exception-credential-marker")
    monkeypatch.setattr(api, "run_editorial_step", failing)
    response = _request("POST", BASE, json=body)
    assert response.status_code == 503 and response.json()["detail"]["code"] == "editorial_step_failed"
    fetched = _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert fetched.json()["status"] == "unknown"
    raw = b"".join(path.read_bytes() for path in Path(os.environ["HERMES_HOME"]).rglob("*sqlite3*"))
    assert b"private-exception-credential-marker" not in raw and b"operator-test-secret" not in raw
    assert "private-exception" not in capsys.readouterr().out


def test_legacy_no_id_has_no_receipt_and_missing_get_does_not_initialize_store(monkeypatch):
    calls = []
    async def model(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return fact_first(body)
    monkeypatch.setattr(api, "_llm_json", model)
    body = paid_payload()
    missing = _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert missing.status_code == 404 and not EditorialReceiptStore().path.exists()
    body.pop("budget_reservation_id")
    assert _request("POST", BASE, json=body).status_code == 200
    assert _request("POST", BASE, json=body).status_code == 200 and len(calls) == 2
    assert not EditorialReceiptStore().path.exists()


@pytest.mark.parametrize("timeout", ["0", "601", "NaN", "Infinity", "not-a-number"])
def test_invalid_timeout_header_stops_before_model_or_receipt(monkeypatch, timeout):
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid timeout reached model")
    monkeypatch.setattr(api, "_llm_json", forbidden)
    response = _request("POST", BASE, json=paid_payload(), headers={"X-Hermes-Timeout-Seconds": timeout})
    assert response.status_code == 422 and not EditorialReceiptStore().path.exists()


def test_timeout_header_reaches_real_provider_http_without_adding_calls(monkeypatch):
    monkeypatch.setenv("HERMES_OPENAI_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "provider-fixture-not-recorded")
    monkeypatch.setattr(api, "server_stage_route", lambda stage: None)
    monkeypatch.setattr(api, "_resolve_host_addresses", lambda *_: {"93.184.216.34"})
    observed = []
    original = httpx.AsyncClient
    def provider(request):
        observed.append(request)
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(fact_first(prompt))}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8}})
    def client(**kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(provider)
        return original(**kwargs)
    monkeypatch.setattr(api.httpx, "AsyncClient", client)
    response = _request("POST", BASE, json=paid_payload(), headers={"X-Hermes-Timeout-Seconds": "600"})
    assert response.status_code == 200 and response.json()["status"] == "artifact"
    assert len(observed) == 1 and observed[0].extensions["timeout"]["read"] == 600
    raw = b"".join(path.read_bytes() for path in Path(os.environ["HERMES_HOME"]).rglob("*sqlite3*"))
    assert b"provider-fixture-not-recorded" not in raw


def test_header_cannot_extend_frozen_inner_deadline_or_call_budget(monkeypatch):
    calls = []
    async def slow(*args, **kwargs):
        calls.append(kwargs)
        await asyncio.sleep(10)
        pytest.fail("inner budget deadline did not cancel model")
    monkeypatch.setattr(api, "_llm_json", slow)
    body = paid_payload()
    body["compose"]["runtime_budget"].update(max_elapsed_seconds=1, max_model_calls=1)
    response = _request("POST", BASE, json=body, headers={"X-Hermes-Timeout-Seconds": "600"})
    assert response.status_code == 200 and len(calls) == 1
    assert calls[0]["timeout_seconds"] == "600.0"
    trace = response.json()["generation_trace"]
    assert trace["budget_exhausted"] and trace["attempts"][0]["error_code"] == "runtime_deadline_exceeded"
    assert _request("POST", BASE, json=body).json() == response.json() and len(calls) == 1


def test_real_signed_chain_persists_private_reviewer_candidate_and_exact_completed_reply(monkeypatch):
    calls = []
    async def model(system, body, fallback, *args, **kwargs):
        calls.append(body)
        return review(body) if "editorial_review" in body else fact_first(body)
    monkeypatch.setattr(api, "_llm_json", model)
    compose = payload()
    root = BASE.removesuffix("draft_master")
    master = _request("POST", BASE, json={"compose": compose, "budget_reservation_id": "text_chain_master"}).json()
    adapted = _request("POST", root + "adapt_platform", json={"compose": compose, "budget_reservation_id": "text_chain_adapt",
        "surface": "wechat_mp", "candidate": master["artifact"], "candidate_signature": master["artifact_signature"]}).json()
    merged = copy.deepcopy(master["artifact"])
    merged["platform_variants"] = adapted["artifact"]["platform_variants"]
    verified = _request("POST", root + "verify_claims", json={"compose": compose, "budget_reservation_id": "text_chain_verify", "candidate": merged}).json()
    body = {"compose": compose, "budget_reservation_id": "text_chain_review", "candidate": verified["artifact"],
            "candidate_signature": verified["artifact_signature"]}
    result = _request("POST", root + "review_editorial", json=body)
    assert result.status_code == 200 and result.json()["artifact"]["quality_scorecard"]["passed"]
    private = result.json()["private_review_receipt"]
    assert private["candidate_retained"] and private["validation"]["code"] == "review_validated"
    assert "private_review_receipt" not in result.json()["generation_trace"]
    before = len(calls)
    retrieved = _request("GET", root + "review_editorial/receipts/text_chain_review",
        headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert retrieved.json()["response"] == result.json() and len(calls) == before
    assert _request("POST", root + "review_editorial", json=body).json() == result.json() and len(calls) == before


def test_response_capacity_failure_retains_unknown_tombstone_and_forbids_repurchase(monkeypatch):
    import ai_marketing_api.operator_receipts as receipts
    monkeypatch.setattr(receipts, "MAX_RESPONSE_BYTES", 1024)
    calls = []
    async def oversized(*args, **kwargs):
        calls.append(True)
        return {"schema_version": "operator.editorial_step.v1", "step": "draft_master",
                "status": "artifact", "private_review_receipt": {"candidate": {"text": "x" * 2000}}}
    monkeypatch.setattr(api, "run_editorial_step", oversized)
    body = paid_payload()
    result = _request("POST", BASE, json=body)
    assert result.status_code == 503 and result.json()["detail"]["code"] == "editorial_receipt_response_too_large"
    retrieved = _request("GET", receipt_path(body), headers={"X-Hermes-Request-SHA256": request_sha256(body)})
    assert retrieved.json()["status"] == "unknown" and "response" not in retrieved.json()
    assert _request("POST", BASE, json=body).status_code == 409 and len(calls) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission semantics")
def test_existing_loose_private_volume_and_database_are_tightened_without_changing_home():
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(mode=0o755)
    home.chmod(0o755)
    directory = home / "marketing_api"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    store = EditorialReceiptStore()
    store.path.touch(mode=0o644)
    store.path.chmod(0o644)
    body = paid_payload()
    sha = request_sha256(body)
    store.claim("industry", "draft_master", body["budget_reservation_id"], sha)
    assert stat.S_IMODE(home.stat().st_mode) == 0o755
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    # Hold a live WAL connection so its sidecars actually exist for inspection.
    with store._connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE editorial_receipts SET updated_at=CURRENT_TIMESTAMP")
        for sidecar in (Path(str(store.path) + "-wal"), Path(str(store.path) + "-shm")):
            assert sidecar.exists() and stat.S_IMODE(sidecar.stat().st_mode) & 0o077 == 0
        connection.rollback()
