"""Real payload transport and authority invariants, never model-quality verdicts."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from ai_marketing_api.operator_content import content_request_payload
from ai_marketing_api.operator_editorial import EditorialInstruction, OwnerRevision
from ai_marketing_api.operator_editorial_method import industry_reader_method
from ai_marketing_api.operator_review import _reader_payload
from tests.ai_marketing_api.test_industry_interpretation_method import chat_instruction, reading_output
from tests.ai_marketing_api.test_surface_fact_plan import fixture


def _assert_scope_and_choice_guidance(method):
    # This checks instructions delivered to the real consumers, not whether a
    # scripted score or a text-matching substitute can judge an article.
    scope = next(rule for rule in method["rules"] if "documented object, operation and preconditions" in rule)
    assert "later operation does not establish that an earlier choice must be fixed" in scope
    assert "whole process forbids alternatives" in scope
    assert "Distinguish evidence, inference and hypothetical conditions" in scope
    assert "obtain support or narrow the conclusion" in scope
    choice = next(rule for rule in method["rules"] if "observable distinction" in rule)
    assert "not in the conclusion that the choice is worthwhile or useful" in choice
    assert "which observation favors which action and why" in choice
    assert "using only distinctions the evidence supports" in choice
    assert "Do not require the reader to supply the missing value judgment" in choice
    assert "not a prescribed article structure, product stance, numerical threshold or guaranteed result" in choice


@pytest.mark.parametrize("intent", ["none", "owner", "chat", "feedback"])
def test_real_writer_receives_clarifications_without_changing_authority(monkeypatch, intent):
    request, contexts, _ = fixture(monkeypatch)
    original_payload = content_request_payload("industry", request, contexts)
    if intent == "owner":
        request.owner_revision = OwnerRevision(instruction="  Keep the source scope; rebuild the argument.\n")
    elif intent == "chat":
        request.editorial_instruction = EditorialInstruction.model_validate(chat_instruction())
    elif intent == "feedback":
        request.editorial_feedback = '  {"source":"independent_editorial_review","issues":["Explain the decision link"]}\n'
    before = request.model_dump(mode="json")
    payload = content_request_payload("industry", request, contexts)
    _assert_scope_and_choice_guidance(payload["industry_editorial_method"])
    assert payload["industry_editorial_method"]["rules"] == industry_reader_method()["rules"]
    assert "reader_rule" not in payload["industry_editorial_method"]
    assert "value_checks" not in payload["industry_editorial_method"]
    for key in ("claims", "authorized_context", "canonical_block_registry", "audience", "objective", "channels"):
        assert payload[key] == original_payload[key], key
    assert request.model_dump(mode="json") == before
    assert payload["editorial_feedback"] == before["editorial_feedback"]
    for key in ("owner_revision", "editorial_instruction"):
        if before[key] is not None:
            assert payload[key] == before[key]
        else:
            assert key not in payload
    assert payload["surface_fact_plan"]["surfaces"] == before["surface_fact_plan"]["surfaces"]
    assert payload["surface_fact_plan"]["fact_roles"] == before["surface_fact_plan"]["fact_roles"]


@pytest.mark.parametrize("mode", ["standard", "value_reassessment.v1"])
@pytest.mark.parametrize("body", [
    "青鸟书柜在预约确认后安排取书，读者仍需要区分通知对应的状态。",
    "白鹭工坊的说明区分提交与验收；这里仅讨论说明中的两项操作。",
])
def test_real_blind_reader_receives_same_generic_guidance_without_an_intended_answer(mode, body):
    value = reading_output().model_copy(update={"master_content": body})
    before = value.model_dump(mode="json")
    hidden = {key: "PRIVATE_SCOPE_SENTINEL_" + key for key in (
        "owner_revision", "editorial_instruction", "author_conclusion", "old_grade", "value_contract")}
    brief = {"audience": "说明的读者", **hidden}
    brief_before = deepcopy(brief)
    payload = _reader_payload(value, brief, scorecard_required=True, review_mode=mode)
    method = payload["editorial_review"]["industry_editorial_method"]
    _assert_scope_and_choice_guidance(method)
    assert method == industry_reader_method()
    assert payload["editorial_review"]["draft"]["master_content"] == body
    assert all(sentinel not in json.dumps(payload) for sentinel in hidden.values())
    assert body not in json.dumps(method, ensure_ascii=False)
    assert "Seedance" not in json.dumps(method)
    assert value.model_dump(mode="json") == before and brief == brief_before
    assert ("value_reassessment" in payload["editorial_review"]) == (mode != "standard")
    assert ("value_resolution" in payload["response_contract"]) == (mode != "standard")
