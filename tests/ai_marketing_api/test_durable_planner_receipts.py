"""GHA-only ASGI and real temporary SQLite coverage for the planner journal."""
from copy import deepcopy

import pytest

import ai_marketing_api.main as api
from ai_marketing_api.operator_receipts import EditorialReceiptStore, request_sha256
from tests.ai_marketing_api.test_operator_api import _request


BASE = "/api/v1/operators/editorial/plan"
IDENTITY = "text_" + "3" * 40


@pytest.fixture(autouse=True)
def isolated_planner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "planner-home"))
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def body():
    return {"mandate": {"objective": "Use documented material"}, "opportunities": [],
            "budget": {"max_model_calls": 1, "max_elapsed_seconds": 300, "max_output_tokens": 6000}}


def headers(value):
    return {"X-Hermes-Budget-Reservation-Id": IDENTITY,
            "X-Hermes-Request-SHA256": request_sha256(value)}


def test_planner_reply_survives_store_restart_and_duplicate_post_never_calls_again(monkeypatch):
    calls = []
    result = {"status": "planned", "selections": [], "generation_trace": {"model_calls": 1}}
    async def planner(request, *args):
        calls.append(request.model_dump(mode="json"))
        return deepcopy(result)
    monkeypatch.setattr(api, "plan_editorial", planner)
    value = body()
    first = _request("POST", BASE, json=value, headers=headers(value))
    assert first.status_code == 200 and first.json() == result
    assert _request("POST", BASE, json=value, headers=headers(value)).json() == result
    fetched = _request("GET", BASE + "/receipts/" + IDENTITY,
                       headers={"X-Hermes-Request-SHA256": request_sha256(value)})
    assert fetched.status_code == 200 and fetched.headers["cache-control"] == "no-store"
    assert fetched.json()["response"] == result and len(calls) == 1
    assert EditorialReceiptStore().get("industry", "editorial_plan", IDENTITY,
                                      request_sha256(value))["response"] == result


def test_completed_unknown_provider_reply_is_cached_without_becoming_success(monkeypatch):
    calls = []
    result = {"status": "unavailable", "selections": [], "generation_trace": {
        "model_calls": 1, "provider_requests": [{"status": "unknown", "error_code": "provider_cancelled"}]}}
    async def planner(request, *args):
        calls.append(True)
        return deepcopy(result)
    monkeypatch.setattr(api, "plan_editorial", planner)
    value = body()
    for _ in range(2):
        assert _request("POST", BASE, json=value, headers=headers(value)).json() == result
    assert len(calls) == 1
    assert _request("GET", BASE + "/receipts/" + IDENTITY,
                    headers={"X-Hermes-Request-SHA256": request_sha256(value)}).json()["response"] == result


def test_unknown_running_and_missing_get_never_start_another_plan(monkeypatch):
    async def forbidden(*args):
        pytest.fail("receipt GET or duplicate POST invoked planner")
    monkeypatch.setattr(api, "plan_editorial", forbidden)
    value = body(); sha = request_sha256(value)
    token, _ = EditorialReceiptStore().claim("industry", "editorial_plan", IDENTITY, sha)
    for status in ("running", "unknown"):
        if status == "unknown":
            EditorialReceiptStore().unknown("industry", "editorial_plan", IDENTITY, token)
        fetched = _request("GET", BASE + "/receipts/" + IDENTITY, headers={"X-Hermes-Request-SHA256": sha})
        assert fetched.status_code == 200 and fetched.json()["status"] == status
        assert "response" not in fetched.json()
        assert _request("POST", BASE, json=value, headers=headers(value)).status_code == 409
    missing = _request("GET", BASE + "/receipts/text_" + "9" * 40, headers={"X-Hermes-Request-SHA256": sha})
    assert missing.status_code == 404


def test_planner_auth_identity_and_original_wire_hash_are_required(monkeypatch):
    calls = []
    async def planner(request, *args):
        calls.append(True)
        return {"status": "no_opportunities", "selections": []}
    monkeypatch.setattr(api, "plan_editorial", planner)
    value = body()
    assert _request("POST", BASE, json=value, headers=headers(value)).status_code == 200
    changed = {**value, "max_packages": 1}
    assert _request("POST", BASE, json=changed, headers=headers(changed)).status_code == 409
    assert _request("GET", BASE + "/receipts/" + IDENTITY,
        headers={"X-Hermes-Request-SHA256": request_sha256(changed)}).status_code == 409
    assert _request("GET", BASE + "/receipts/" + IDENTITY, operator_auth=False,
        headers={"X-Hermes-Request-SHA256": request_sha256(value)}).status_code == 401
    assert _request("POST", BASE, json=value,
        headers={"X-Hermes-Budget-Reservation-Id": IDENTITY}).status_code == 422
    assert len(calls) == 1
