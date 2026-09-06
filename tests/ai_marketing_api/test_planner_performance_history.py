from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

import ai_marketing_api.main as api
from ai_marketing_api.operator_planning import EditorialPlanRequest, PerformanceHistory
from tests.ai_marketing_api.test_autonomous_editorial import _planner_payload
from tests.ai_marketing_api.test_operator_api import _request


@pytest.fixture(autouse=True)
def service_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def _history(**changes):
    return {
        "signal_id": "signal-api-public-1",
        "role_id": "commercial",
        "platform": "wechat_mp",
        "title": "Choosing an integration workflow",
        "published_at": "2026-09-01T08:00:00+08:00",
        "observed_at": "2026-09-02T08:00:00+08:00",
        "metric_window_days": 1,
        "metrics": [
            {"name": "views", "availability": "available", "value": 0},
            {"name": "likes", "availability": "unavailable", "value": None},
            {"name": "comments", "availability": "not_due", "value": None},
            {"name": "shares", "availability": "error", "value": None},
        ],
        **changes,
    }


def _selection(**changes):
    return {
        "opportunity_id": "opp-api",
        "role_id": "commercial",
        "topic": "Choosing an integration workflow",
        "angle": "A concrete check before connecting a creative tool",
        "reason": "Relevant to the approved audience and existing evidence",
        **changes,
    }


def test_history_reaches_the_real_planner_adapter_without_extra_calls_or_zero_coercion(
    monkeypatch,
):
    payload = _planner_payload()
    payload["mandate"]["channels"] = ["wechat_mp"]
    payload["performance_history"] = [_history()]
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append(body)
        history = body["performance_history"]
        assert history[0]["signal_id"] == "signal-api-public-1"
        metrics = {metric["name"]: metric for metric in history[0]["metrics"]}
        assert metrics["views"] == {
            "name": "views",
            "availability": "available",
            "value": 0,
        }
        assert all(
            metrics[name]["value"] is None for name in ("likes", "comments", "shares")
        )
        assert body["mandate"] == EditorialPlanRequest.model_validate(
            payload
        ).mandate.model_dump(mode="json")
        assert "same platform and the same metric window" in system
        assert "stage bucket, not an exact exposure" in system
        assert "published_at-to-observed_at durations" in system
        assert "source makes available" in system
        assert "unknown, never zero" in system
        assert "do not establish causation" in system
        assert "personal-material authorization" in system
        assert "out of reader-facing topic and angle" in system
        return {
            "_model": "test-planner",
            "selections": [
                _selection(
                    history_refs=["signal-api-public-1"],
                    history_reason="A measured zero is not a missing measurement; no causal conclusion.",
                )
            ],
            "skips": [],
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    response = _request("POST", "/api/v1/operators/editorial/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "planned", data
    assert len(calls) == data["generation_trace"]["model_calls"] == 1
    assert data["selections"][0]["history_refs"] == ["signal-api-public-1"]
    assert "signal-api-public-1" not in data["selections"][0]["angle"]


@pytest.mark.parametrize("with_history", [False, True])
def test_unknown_history_reference_fails_closed_without_repair_or_new_call(
    with_history, monkeypatch
):
    payload = _planner_payload()
    if with_history:
        payload["performance_history"] = [_history()]
    calls = []

    async def fake(*args, **kwargs):
        calls.append(True)
        return {
            "_model": "test-planner",
            "selections": [_selection(history_refs=["invented-signal"])],
            "skips": [],
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert data["status"] == "unavailable"
    assert data["selections"] == []
    assert len(calls) == data["generation_trace"]["model_calls"] == 1


def test_legacy_request_and_model_selection_remain_compatible(monkeypatch):
    async def fake(system, body, fallback, *args, **kwargs):
        assert body["performance_history"] == []
        return {"_model": "test-planner", "selections": [_selection()], "skips": []}

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request(
        "POST", "/api/v1/operators/editorial/plan", json=_planner_payload()
    ).json()
    assert data["status"] == "planned"
    assert data["selections"][0]["history_refs"] == []
    assert data["selections"][0]["history_reason"] == ""


@pytest.mark.parametrize(
    "field",
    [
        "account_id",
        "user_id",
        "comments",
        "consultation_text",
        "url",
        "body",
        "source",
        "metadata",
    ],
)
def test_private_or_raw_source_fields_are_rejected_before_model(field, monkeypatch):
    payload = _planner_payload()
    payload["performance_history"] = [_history(**{field: "must-not-reach-model"})]

    async def forbidden(*args, **kwargs):
        pytest.fail("Raw feedback or account metadata must never reach the planner")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    assert (
        _request("POST", "/api/v1/operators/editorial/plan", json=payload).status_code
        == 422
    )


@pytest.mark.parametrize(
    "availability,value,valid",
    [
        ("available", 0, True),
        ("available", 0.5, True),
        ("available", None, False),
        ("available", -1, False),
        ("available", True, False),
        ("available", "0", False),
        ("available", float("nan"), False),
        ("available", float("inf"), False),
        ("unavailable", None, True),
        ("not_due", None, True),
        ("error", None, True),
        ("unavailable", 0, False),
        ("not_due", 0, False),
        ("error", 0, False),
    ],
)
def test_metric_values_preserve_availability_semantics(availability, value, valid):
    history = _history(
        metrics=[{"name": "views", "availability": availability, "value": value}]
    )
    if valid:
        assert PerformanceHistory.model_validate(history).metrics[0].value == value
    else:
        with pytest.raises(ValidationError):
            PerformanceHistory.model_validate(history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_at", "2026-09-01T08:00:00"),
        ("observed_at", "2026-09-02"),
        ("observed_at", 1788307200),
        ("observed_at", "1788307200"),
        ("metric_window_days", 2),
        ("metric_window_days", True),
        ("metric_window_days", "1"),
        ("signal_id", ""),
        ("signal_id", "s" * 161),
        ("platform", "p" * 81),
        ("title", "t" * 501),
        ("metrics", [{"name": "conversions", "availability": "available", "value": 1}]),
        (
            "metrics",
            [
                {
                    "name": "comments",
                    "availability": "available",
                    "value": 1,
                    "comment_text": "private",
                }
            ],
        ),
        ("metrics", [{"name": "views", "availability": "available"}]),
    ],
)
def test_history_strict_bounds_dates_and_metric_allowlist(field, value):
    with pytest.raises(ValidationError):
        PerformanceHistory.model_validate(_history(**{field: value}))


def test_history_and_metrics_have_bounded_unique_identifiers():
    payload = _planner_payload()
    payload["performance_history"] = [
        _history(signal_id=f"signal-{i}") for i in range(31)
    ]
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(payload)
    payload["performance_history"] = [_history(), _history()]
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(payload)
    history = _history()
    history["metrics"].append(deepcopy(history["metrics"][0]))
    with pytest.raises(ValidationError):
        PerformanceHistory.model_validate(history)


def test_personal_engagement_history_cannot_authorize_an_unverified_story(monkeypatch):
    payload = _planner_payload()
    payload["performance_history"] = [
        _history(
            role_id="personal_ip",
            metrics=[{"name": "likes", "availability": "available", "value": 999}],
        )
    ]

    async def fake(system, body, fallback, *args, **kwargs):
        assert all(row["id"] != "opp-private" for row in body["opportunities"])
        return {
            "_model": "test-planner",
            "selections": [
                _selection(
                    opportunity_id="opp-private",
                    role_id="personal_ip",
                    history_refs=["signal-api-public-1"],
                )
            ],
            "skips": [],
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert data["status"] == "unavailable"
    assert data["selections"] == []


@pytest.mark.parametrize(
    "refs,reason",
    [
        (["signal-api-public-1"] * 2, "duplicate"),
        (["signal-api-public-1"] * 11, "too many"),
        (["signal-api-public-1"], "r" * 1001),
    ],
)
def test_selection_history_explanations_are_bounded(refs, reason, monkeypatch):
    payload = _planner_payload()
    payload["performance_history"] = [_history()]

    async def fake(*args, **kwargs):
        return {
            "_model": "test-planner",
            "selections": [_selection(history_refs=refs, history_reason=reason)],
            "skips": [],
        }

    monkeypatch.setattr(api, "_llm_json", fake)
    data = _request("POST", "/api/v1/operators/editorial/plan", json=payload).json()
    assert data["status"] == "unavailable"
    assert data["selections"] == []
