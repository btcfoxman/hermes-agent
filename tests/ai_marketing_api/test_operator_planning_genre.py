"""Planner protocol regressions with real schemas and synthetic model responses."""
from copy import deepcopy
import hashlib
import hmac

import pytest

from ai_marketing_api import operator_editorial
from ai_marketing_api.operator_editorial_method import industry_planning_method, industry_reader_method
from ai_marketing_api.operator_planning import EditorialPlanRequest, plan_editorial


SOURCE = "The published handover contains conclusions and citations, but does not contain the original inputs."


def payload(**changes):
    return {"mandate": {"objective": "Explain documented options"}, "opportunities": [{
        "id": "source-a", "role_id": "industry", "title": "Documented handover", "summary": SOURCE,
        "verified": True, "url": "https://example.invalid/primary"}], "max_packages": 1, **changes}


def selection(*, role="industry", identity="source-a", with_value=True):
    result = {"opportunity_id": identity, "role_id": role, "topic": "What citations establish",
        "angle": "Separate availability from sufficient support", "reason": "A documented handover decision"}
    if with_value:
        result["value_contract"] = {
            "reader_problem": "Does a citation list make a conclusion ready for handover?",
            "thesis": "Source access and sufficient support are separate acceptance decisions.",
            "evidence_anchor": {"opportunity_id": identity, "quote": SOURCE},
            "decision_delta": "Review the original inputs before handover; this requires additional review effort.",
            "counterargument": "An authorized package containing the original inputs changes this arrangement."}
    return result


def planned(monkeypatch, raw, selections, *, skips=None):
    """Drive only immediate fake awaits; fail if the planner attempts to suspend."""
    request = EditorialPlanRequest.model_validate(deepcopy(raw))
    before = request.model_dump_json()
    calls = []

    async def fake(system, body, fallback, *args, **kwargs):
        calls.append((system, deepcopy(body), kwargs))
        return {"_model": "offline-planner-protocol", "selections": deepcopy(selections), "skips": skips or []}

    async def immediate(awaitable, timeout):
        return await awaitable

    with monkeypatch.context() as context:
        context.setattr(operator_editorial.asyncio, "wait_for", immediate)
        coroutine = plan_editorial(request, fake)
        try:
            coroutine.send(None)
        except StopIteration as completed:
            result = completed.value
        else:
            pytest.fail("The fake planner adapter must not suspend for I/O")
        finally:
            coroutine.close()
    assert request.model_dump_json() == before
    return result, calls


@pytest.mark.parametrize("genre", [None, ""])
def test_legacy_absent_or_empty_genre_preserves_normalized_json_hmac_and_model_body(monkeypatch, genre):
    raw = payload(**({} if genre is None else {"genre": genre}))
    normalized = EditorialPlanRequest.model_validate(raw).model_dump_json().encode()
    # Captured from this exact public fixture through H647's actual request
    # model and planner adapter. These protect existing normalization/signatures,
    # rather than reimplementing the normalization in the assertion.
    assert hashlib.sha256(normalized).hexdigest() == "6eb147a072ffb02891897d3bcda8179590eb81674adf8a2a26b5fbbf58e5be5b"
    assert hmac.new(b"legacy-planner-test", normalized, hashlib.sha256).hexdigest() == "9d8c9bba9de897c55ca259c92cb0c477d053dd04fff14563f64c6fde13314817"
    result, calls = planned(monkeypatch, raw, [selection(with_value=False)])
    assert result["status"] == "planned" and len(calls) == 1
    assert "genre" not in calls[0][1] and "industry_editorial_method" not in calls[0][1]
    assert result["generation_trace"]["attempts"][0]["prompt_sha256"] == "c5db159ed6cfdc3661b3e9dbc106c490e07de416ae445bd1ba497fae638ace60"
    assert hashlib.sha256(calls[0][0].encode()).hexdigest() == "b8f6b64ced58400c3e43b87825733df6b1a53d6519371ff63567f6784b970f69"


def test_analysis_gets_shared_method_and_industry_only_contract_without_changing_evidence(monkeypatch):
    raw = payload(genre="industry_analysis", topic_value_required=False, max_packages=2)
    raw["opportunities"].append({"id": "commercial-a", "role_id": "commercial", "title": "Documented tutorial", "verified": True})
    before = deepcopy(raw)
    result, calls = planned(monkeypatch, raw, [selection(), selection(role="commercial", identity="commercial-a", with_value=False)])
    assert result["status"] == "planned" and raw == before and len(calls) == 1
    method = calls[0][1]["industry_editorial_method"]
    assert method == industry_planning_method()
    assert method["version"] == industry_reader_method()["version"]
    assert method["rules"][0] == industry_reader_method()["rules"][0]
    assert set(method["rules"]) <= set(industry_reader_method()["rules"])
    assert method["applies_to"] == {"role_id": "industry", "genre": "industry_analysis"}
    assert not set(method) & {"reader_rule", "value_checks", "publication_brief", "platform_editorial_briefs"}
    assert calls[0][1]["topic_value_required"] is False
    assert calls[0][1]["opportunities"][0]["summary"] == SOURCE
    assert result["generation_trace"]["repair_calls"] == 0


@pytest.mark.parametrize("genre,role", [("public_tutorial", "industry"), ("news", "industry"),
    ("industry_analysis", "commercial"), ("industry_analysis", "personal_ip")])
def test_other_genres_and_roles_keep_generic_planning_without_forced_opinion(monkeypatch, genre, role):
    raw = payload(genre=genre)
    raw["opportunities"][0]["role_id"] = role
    result, calls = planned(monkeypatch, raw, [selection(role=role, with_value=False)])
    assert result["status"] == "planned"
    assert "industry_editorial_method" not in calls[0][1]
    assert "value_contract" not in calls[0][1]["response_contract"]["selections"][0]
    assert hashlib.sha256(calls[0][0].encode()).hexdigest() == "b8f6b64ced58400c3e43b87825733df6b1a53d6519371ff63567f6784b970f69"


@pytest.mark.parametrize("failure", ["missing_contract", "unbound_quote", "unknown_opportunity"])
def test_analysis_contract_and_existing_source_anchor_fail_closed_without_repair(monkeypatch, failure):
    item = selection(with_value=failure != "missing_contract")
    if failure == "unbound_quote":
        item["value_contract"]["evidence_anchor"]["quote"] = "A claim that the supplied summary does not contain."
    elif failure == "unknown_opportunity":
        item["opportunity_id"] = "unknown"
    result, calls = planned(monkeypatch, payload(genre="industry_analysis"), [item])
    assert result["status"] == "unavailable" and result["selections"] == []
    assert len(calls) == 1 and result["generation_trace"]["repair_calls"] == 0


def test_analysis_empty_material_abstains_before_adapter_without_changing_other_roles(monkeypatch):
    raw = payload(genre="industry_analysis")
    raw["opportunities"][0]["summary"] = " \n "
    result, calls = planned(monkeypatch, raw, [])
    assert result["status"] == "no_opportunities" and calls == []
    raw["opportunities"].append({"id": "commercial-a", "role_id": "commercial", "title": "Tutorial", "verified": True})
    result, calls = planned(monkeypatch, raw, [selection(role="commercial", identity="commercial-a", with_value=False)])
    assert result["status"] == "planned"
    assert [item["id"] for item in calls[0][1]["opportunities"]] == ["commercial-a"]


def test_analysis_can_return_supported_abstention_without_an_intended_selection(monkeypatch):
    result, calls = planned(monkeypatch, payload(genre="industry_analysis"), [], skips=[{
        "role_id": "industry", "reason": "The supplied facts do not establish a supported tradeoff."}])
    assert result["status"] == "planned" and result["selections"] == [] and len(calls) == 1
    assert "supported tradeoff" in next(item for item in result["skips"] if item["role_id"] == "industry")["reason"]
