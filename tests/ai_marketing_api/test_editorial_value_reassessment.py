"""Fresh review provenance and rejection guards; scripted grades are not calibration."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from ai_marketing_api import main as api, operator_steps as steps
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_editorial_method import INDUSTRY_METHOD_VERSION, VALUE_REASSESSMENT_MODE
from ai_marketing_api.operator_receipts import request_sha256
from ai_marketing_api.operator_review import (
    _reader_payload, apply_editorial_assessment, assess_editorial, editorial_content_hash,
)
from tests.ai_marketing_api.test_editorial_layout_and_model_version import _layout_normalize
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review
from tests.ai_marketing_api.test_industry_interpretation_method import reading_output, writer_request
from tests.ai_marketing_api.test_operator_api import _request


BASE = "/api/v1/operators/industry/editorial-steps/"


@pytest.fixture
def isolated_edge(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-reassessment-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-reader"})


def reject_restatement(payload):
    """An explicit offline verdict, not a function claiming to evaluate prose."""
    result = {**scripted_review(payload), "_model": "synthetic-value-reader"}
    result["value_resolution"] = "revise"
    for row in result["narrative_report"]["surfaces"]:
        row["analysis"]["author_conclusion"] = ""
        row["analysis"]["decision_consequence"] = ""
        row["analysis"]["source_restatement_only"] = True
        row["analysis"]["supported"] = False
        row["issues"] = ["只复述用途和提醒检查，未说明作者选择及其代价。"]
    return result


@pytest.mark.parametrize("body", [
    "青鸟书柜接受预约并在确认后安排取书。先预约，再决定是否取书；出门前应检查预约状态。",
    "青鸟书柜将预约和取书分开。值得考虑是否预约，不能保证当天拿到。你会怎么选？欢迎分享经验。",
])
def test_reassessment_judges_only_current_bodies_and_preserves_real_failure(body):
    value = reading_output().model_copy(update={"master_title": "青鸟书柜的预约与取书", "master_content": body})
    hidden = {key: "PRIVATE_EXPECTED_" + key for key in (
        "author_conclusion", "source_claims", "owner_revision", "old_grade", "style_reference", "editorial_instruction")}
    seen = []

    async def reader(system, payload, fallback, **kwargs):
        seen.append(deepcopy(payload))
        draft = payload["editorial_review"]["draft"]
        assert draft["master_content"] == body
        assert draft["platform_variants"] == [{"platform": v.platform, "title": v.title, "body": v.body}
            for v in value.platform_variants]
        assert all(secret not in json.dumps(payload) for secret in hidden.values())
        assert "Seedance" not in json.dumps(payload)
        return reject_restatement(payload)

    before = value.model_dump(mode="json")
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    review, state = asyncio.run(assess_editorial(output=value, brief={"audience": "借阅者", **hidden},
        run=run, llm=reader, system="unused", model_args=(), owner_revision=hidden,
        editorial_feedback=hidden["old_grade"], style_context=hidden, scorecard_required=True,
        review_mode=VALUE_REASSESSMENT_MODE))
    assert len(seen) == 1 and state == "failed" and not review.passed
    assert review.quality_scorecard.passed  # No automatic alteration of supplied numeric grades.
    assert not review.narrative_report.passed
    rejected = apply_editorial_assessment(value, review, review_context=run.editorial_review_context)
    assert rejected.status == "quality_insufficient" and rejected.master_content == body
    assert value.model_dump(mode="json") == before
    assert rejected.editorial_review_context.content_sha256 == editorial_content_hash(value)
    assert rejected.editorial_review_context.method_version == INDUSTRY_METHOD_VERSION


def test_method_teaches_a_choice_and_cost_without_a_prescribed_product_stance():
    request = writer_request()
    payload = api.content_request_payload("industry", request, request.authorized_context)
    method = payload["industry_editorial_method"]
    assert method["version"] == INDUSTRY_METHOD_VERSION
    assert "value_checks" not in method and "reader_rule" not in method
    assert payload["claims"] == request.model_dump(mode="json")["claims"]
    assert payload["editorial_instruction"] == request.model_dump(mode="json")["editorial_instruction"]
    actual = reading_output()
    standard = _reader_payload(actual, {}, scorecard_required=True)
    extra = _reader_payload(actual, {}, scorecard_required=True, review_mode=VALUE_REASSESSMENT_MODE)
    assert standard["editorial_review"]["draft"] == extra["editorial_review"]["draft"]
    assert "value_reassessment" not in standard["editorial_review"]
    assert extra["editorial_review"]["value_reassessment"]["mode"] == VALUE_REASSESSMENT_MODE
    assert extra["editorial_review"]["value_reassessment"]["method_version"] == INDUSTRY_METHOD_VERSION
    assert "value_checks" in extra["editorial_review"]["industry_editorial_method"]
    assert "Seedance" not in json.dumps(extra["editorial_review"]["industry_editorial_method"])


def _verified(monkeypatch):
    request, _, _, _, raw = _layout_normalize()
    compose = request.model_dump(mode="json")

    async def never_model(*args, **kwargs):
        pytest.fail("Exact references do not need a new provider request for verification")

    monkeypatch.setattr(api, "_llm_json", never_model)
    verified = _request("POST", BASE + "verify_claims", json={"compose": compose, "candidate": raw}).json()
    assert verified["status"] == "artifact"
    return compose, verified


def test_real_signed_step_new_mode_has_new_receipt_and_cannot_reuse_standard_pass(monkeypatch, isolated_edge):
    compose, verified = _verified(monkeypatch)
    seen = []

    async def reader(system, payload, fallback, **kwargs):
        seen.append(deepcopy(payload))
        if "value_reassessment" in payload["editorial_review"]:
            return reject_restatement(payload)
        return {**scripted_review(payload), "_model": "synthetic-standard-reader"}

    monkeypatch.setattr(api, "_llm_json", reader)
    standard = {"compose": compose, "candidate": verified["artifact"],
        "candidate_signature": verified["artifact_signature"], "budget_reservation_id": "standard-review"}
    first = _request("POST", BASE + "review_editorial", json=standard).json()
    assert first["status"] == "artifact"
    assert "editorial_review_mode" not in first and "editorial_review_context" not in first["artifact"]
    original = deepcopy(first)
    additional = {**deepcopy(standard), "budget_reservation_id": "additional-value-review"}
    additional["candidate"] = deepcopy(first["artifact"])
    additional["candidate_signature"] = first["artifact_signature"]
    additional["compose"]["editorial_review_mode"] = VALUE_REASSESSMENT_MODE
    # A new mode under the old reservation is a different request, never an implicit retry.
    conflict = _request("POST", BASE + "review_editorial", json={**additional, "budget_reservation_id": "standard-review"})
    assert conflict.status_code == 409 and len(seen) == 1
    second = _request("POST", BASE + "review_editorial", json=additional).json()
    assert second["status"] == "blocked" and len(seen) == 2
    assert second["editorial_review_mode"] == VALUE_REASSESSMENT_MODE
    assert second["artifact"]["editorial_assessment"]["value_resolution"] == "revise"
    context = second["artifact"]["editorial_review_context"]
    assert context == {"schema_version": "operator.editorial_review_context.v1", "mode": VALUE_REASSESSMENT_MODE,
        "method_version": INDUSTRY_METHOD_VERSION, "content_sha256": editorial_content_hash(
            api.OperatorContentOutput.model_validate(second["artifact"]))}
    assert second["artifact_signature"] == steps.artifact_signature(second["artifact"])
    assert seen[0]["editorial_review"]["draft"] == seen[1]["editorial_review"]["draft"]
    assert first == original
    def no_current_route(stage):
        pytest.fail("Completed receipt reads must not consult a changed current model route")
    monkeypatch.setattr(steps, "server_stage_route", no_current_route)
    replay = _request("POST", BASE + "review_editorial", json=additional).json()
    assert replay == second and len(seen) == 2
    receipt = _request("GET", BASE + "review_editorial/receipts/additional-value-review",
        headers={"X-Hermes-Request-SHA256": request_sha256(additional)}).json()
    assert receipt["response"] == second and len(seen) == 2
    old_receipt = _request("GET", BASE + "review_editorial/receipts/standard-review",
        headers={"X-Hermes-Request-SHA256": request_sha256(standard)}).json()
    assert old_receipt["response"] == original and "editorial_review_context" not in old_receipt["response"]["artifact"]
    assert "value_resolution" not in old_receipt["response"]["artifact"]["editorial_assessment"]


def test_standard_review_does_not_inherit_a_candidate_context_and_stale_context_is_rejected(monkeypatch, isolated_edge):
    compose, verified = _verified(monkeypatch)
    candidate = deepcopy(verified["artifact"])
    candidate["editorial_review_context"] = {"schema_version": "operator.editorial_review_context.v1",
        "mode": VALUE_REASSESSMENT_MODE, "method_version": INDUSTRY_METHOD_VERSION,
        "content_sha256": editorial_content_hash(api.OperatorContentOutput.model_validate(candidate))}
    seen = []

    async def reader(system, payload, fallback, **kwargs):
        seen.append(payload)
        return {**scripted_review(payload), "_model": "synthetic-reader"}

    monkeypatch.setattr(api, "_llm_json", reader)
    result = _request("POST", BASE + "review_editorial", json={"compose": compose, "candidate": candidate,
        "candidate_signature": steps.artifact_signature(candidate)}).json()
    assert result["status"] == "artifact" and len(seen) == 1
    assert "editorial_review_context" not in result["artifact"]
    value = reading_output()
    with pytest.raises(ValueError, match="editorial_review_context_changed"):
        apply_editorial_assessment(value, None, review_context=candidate["editorial_review_context"])


@pytest.mark.parametrize("resolution,low", [("ready", False), ("ready", True),
    ("revise", False), ("needs_material", False)])
def test_typed_resolution_preserves_scores_and_never_waives_existing_gates(resolution, low):
    value = reading_output()
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    original = None

    async def reader(system, payload, fallback, **kwargs):
        nonlocal original
        original = {**scripted_review(payload, low="master" if low else None),
            "_model": "synthetic-resolution-reader", "value_resolution": resolution}
        return deepcopy(original)

    review, state = asyncio.run(assess_editorial(output=value, brief={}, run=run, llm=reader,
        system="unused", model_args=(), scorecard_required=True, review_mode=VALUE_REASSESSMENT_MODE))
    assert state == ("passed" if resolution == "ready" and not low else "failed")
    assert review.value_resolution == resolution
    assert [row.model_dump() for row in review.quality_scorecard.surfaces] == original["quality_scorecard"]["surfaces"]
    assert run.private_review_receipt["candidate"]["value_resolution"] == resolution
    if resolution == "needs_material":
        assert review.quality_scorecard.passed and review.narrative_report.passed
        assert not review.passed  # The reader's actual material request is independently binding.


@pytest.mark.parametrize("resolution", [None, "unsupported_guess", True, {}])
def test_new_mode_requires_a_real_typed_resolution_and_keeps_invalid_raw_receipt(resolution):
    value = reading_output()
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))

    async def reader(system, payload, fallback, **kwargs):
        result = {**scripted_review(payload), "_model": "synthetic-reader"}
        if resolution is not None:
            result["value_resolution"] = resolution
        return result

    review, state = asyncio.run(assess_editorial(output=value, brief={}, run=run, llm=reader,
        system="unused", model_args=(), scorecard_required=True, review_mode=VALUE_REASSESSMENT_MODE))
    assert review is None and state == "not_completed"
    assert run.private_review_receipt["validation"]["code"] == "value_resolution_required"
    assert run.private_review_receipt["candidate"].get("value_resolution") == resolution


def test_standard_reader_cannot_self_assign_the_additional_value_protocol():
    value = reading_output()
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))

    async def reader(system, payload, fallback, **kwargs):
        return {**scripted_review(payload), "_model": "synthetic-reader", "value_resolution": "ready"}

    review, state = asyncio.run(assess_editorial(output=value, brief={}, run=run, llm=reader,
        system="unused", model_args=(), scorecard_required=True))
    assert review is None and state == "not_completed" and run.editorial_review_context is None
    assert run.private_review_receipt["validation"]["code"] == "value_resolution_unrequested"


@pytest.mark.parametrize("fault", ["genre", "role", "step", "unknown_mode"])
def test_reassessment_mode_cannot_widen_role_genre_or_step(monkeypatch, isolated_edge, fault):
    request = writer_request().model_dump(mode="json")
    request["editorial_review_mode"] = VALUE_REASSESSMENT_MODE
    if fault == "genre": request["genre"] = "company_data_brief"
    if fault == "unknown_mode": request["editorial_review_mode"] = "use-client-model"
    role = "commercial" if fault == "role" else "industry"
    step = "draft_master" if fault == "step" else "review_editorial"

    async def never_model(*args, **kwargs): pytest.fail("Unsupported mode must stop before model execution")

    monkeypatch.setattr(api, "_llm_json", never_model)
    response = _request("POST", f"/api/v1/operators/{role}/editorial-steps/{step}", json={"compose": request})
    assert response.status_code in {403, 422}
