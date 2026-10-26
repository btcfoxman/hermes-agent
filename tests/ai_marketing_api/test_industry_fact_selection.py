from copy import deepcopy
from datetime import datetime, timezone
import asyncio

import httpx
import pytest

from ai_marketing_api import operator_runtime as runtime

VERSION = "operator.industry_fact_selection.v1"
MARKER = "_industry_fact_selection"
NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)

@pytest.fixture(autouse=True)
def isolated_operator_auth(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "isolated-test-key")

def contexts():
    return [runtime.AuthorizedContext(record_id=f"fact-{i}", space="industry", record_type="source_fact",
        content=f"Authorized observation {i} applies only to operation {i}.",
        authorized_roles=["industry"], source_uri=f"https://source.example/{i}", source_tier="official")
        for i in range(1, 6)]

def request(rows=None):
    return runtime.OperatorProposeRequest(objective="Assess one justified reader decision without assuming a position.",
        genre="industry_analysis", channels=["wechat_mp"], as_of=NOW, authorized_context=rows or contexts())

def normalize(raw, *, action=runtime.OperatorAction.PROPOSE, rows=None, role=runtime.OperatorRole.INDUSTRY):
    rows = rows or contexts()
    req = request(rows)
    registry = runtime.OperatorRegistry()
    authorized = registry.authorize(role, rows, NOW)
    fallback = runtime.build_fallback(role, action, req, authorized)
    before = deepcopy(fallback)
    result = runtime.normalize_output(registry, role, action, req, authorized, fallback, deepcopy(raw))
    assert fallback == before
    return result, fallback

def selected(indices=(4, 2), *, rows=None):
    rows = rows or contexts()
    return {MARKER: VERSION, "_model": "offline-fixture", "claims": [
        {"kind": "fact", "text": rows[i-1].content, "evidence_ids": [rows[i-1].record_id]} for i in indices]}

def test_marked_subset_copies_canonical_objects_in_original_order():
    output, fallback = normalize(selected())
    assert [claim.model_dump() for claim in output.claims] == [fallback["claims"][1], fallback["claims"][3]]
    assert output.evidence_refs == ["fact-2", "fact-4"]
    assert output.proposal.key_points == [contexts()[1].content, contexts()[3].content]
    assert MARKER not in output.model_dump_json()

def test_legacy_raw_candidate_does_not_acquire_selection_semantics():
    raw = selected((2,))
    del raw[MARKER]
    output, fallback = normalize(raw)
    assert [claim.model_dump() for claim in output.claims] == fallback["claims"]
    assert "industry_fact_selection_missing" in {risk.code for risk in output.risk_flags}

@pytest.mark.parametrize("change", ["empty", "missing", "altered", "unknown", "mixed", "duplicate", "duplicate_ids", "wrong_version", "opinion_only", "mixed_opinion", "bad_ids", "bad_opinion_ids", "non_list"])
def test_invalid_or_empty_selection_retains_entire_legacy_pool(change):
    raw = selected((2,))
    if change == "empty": raw["claims"] = []
    elif change == "missing": del raw["claims"]
    elif change == "altered": raw["claims"][0]["text"] += " Unsupported extra outcome."
    elif change == "unknown": raw["claims"][0]["evidence_ids"] = ["unauthorized"]
    elif change == "mixed": raw["claims"].append({"kind": "fact", "text": "Unknown fact.", "evidence_ids": ["fact-1"]})
    elif change == "duplicate": raw["claims"] *= 2
    elif change == "duplicate_ids": raw["claims"][0]["evidence_ids"] *= 2
    elif change == "wrong_version": raw[MARKER] = "other-version"
    elif change == "opinion_only": raw["claims"] = [{"kind": "opinion", "text": "Unsupported assertion.", "evidence_ids": []}]
    elif change == "mixed_opinion": raw["claims"].append({"kind": "opinion", "text": "Unsupported assertion.", "evidence_ids": []})
    elif change == "bad_ids": raw["claims"][0]["evidence_ids"] = 42
    elif change == "bad_opinion_ids": raw["claims"].append({"kind": "opinion", "text": "Unsupported assertion.", "evidence_ids": 42})
    elif change == "non_list": raw["claims"] = "not a list"
    output, fallback = normalize(raw)
    assert [claim.model_dump() for claim in output.claims] == fallback["claims"]
    assert "industry_fact_selection_invalid" in {risk.code for risk in output.risk_flags}

def test_same_text_multiple_sources_retains_every_original_source_id():
    rows = contexts()
    rows.append(rows[1].model_copy(update={"record_id": "fact-2-also", "source_uri": "https://other.example/2"}))
    output, fallback = normalize(selected((2,), rows=rows), rows=rows)
    expected = next(claim for claim in fallback["claims"] if claim["text"] == rows[1].content)
    assert [claim.model_dump() for claim in output.claims] == [expected]
    assert output.evidence_refs == ["fact-2", "fact-2-also"]

def test_revise_and_other_roles_keep_existing_claim_behavior():
    output, fallback = normalize(selected((2,)), action=runtime.OperatorAction.REVISE)
    assert [claim.model_dump() for claim in output.claims] == fallback["claims"]
    assert not any(risk.code.startswith("industry_fact_selection") for risk in output.risk_flags)
    for role, space, kind in [(runtime.OperatorRole.COMMERCIAL, "company_public", "capability"),
                              (runtime.OperatorRole.PERSONAL_IP, "personal_approved", "experience")]:
        rows = [runtime.AuthorizedContext(record_id="one", space=space, record_type=kind,
            content="An explicitly authorized observation.", authorized_roles=[role])]
        result, legacy = normalize(selected((1,), rows=rows), rows=rows, role=role)
        assert [claim.model_dump() for claim in result.claims] == legacy["claims"]
        assert not any(risk.code.startswith("industry_fact_selection") for risk in result.risk_flags)

def test_selection_cannot_waive_a_nonprimary_source_gate():
    rows = contexts()
    rows[4] = rows[4].model_copy(update={"source_tier": "secondary"})
    output, _ = normalize(selected((2,), rows=rows), rows=rows)
    assert output.status == "evidence_insufficient"
    assert any(risk.code == "single_source_unverified" and risk.blocking for risk in output.risk_flags)

def test_ninth_context_cannot_expand_the_existing_eight_fact_pool():
    rows = contexts()
    rows += [rows[0].model_copy(update={"record_id": f"fact-{i}", "content": f"Distinct authorized observation {i}."})
             for i in range(6, 10)]
    output, fallback = normalize(selected((9,), rows=rows), rows=rows)
    assert len(fallback["claims"]) == 8
    assert [claim.model_dump() for claim in output.claims] == fallback["claims"]
    assert "industry_fact_selection_invalid" in {risk.code for risk in output.risk_flags}

def test_model_verification_flag_cannot_change_a_canonical_fact():
    raw = selected((2,))
    raw["claims"][0]["verification_status"] = "model-invented-approval"
    output, fallback = normalize(raw)
    assert [claim.model_dump() for claim in output.claims] == [fallback["claims"][1]]

def test_previously_normalized_wire_has_no_raw_selection_authority():
    original, fallback = normalize({"_model": "legacy-provider", "claims": selected((2,))["claims"]})
    old_wire = original.model_dump(mode="json")
    before = deepcopy(old_wire)
    replayed, _ = normalize(old_wire)
    assert [claim.model_dump() for claim in replayed.claims] == fallback["claims"]
    assert old_wire == before and MARKER not in old_wire

def test_successful_subset_reanchors_deterministic_outline_to_selected_evidence():
    rows = contexts()
    rows[0] = rows[0].model_copy(update={"title": "Unselected source title"})
    rows[1] = rows[1].model_copy(update={"title": "Selected source title"})
    output, fallback = normalize(selected((2,), rows=rows), rows=rows)
    assert fallback["proposal"]["title"] == "Unselected source title"
    assert output.proposal.title == "Selected source title"
    assert "Selected source title" in output.proposal.angle
    assert "Unselected source title" not in output.proposal.angle

def api_request(path, payload):
    import ai_marketing_api.main as api
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://isolated") as client:
            return await client.post(path, json=payload, headers={"Authorization": "Bearer isolated-test-key"})
    return asyncio.run(send())

def test_real_propose_route_exposes_pool_and_preserves_wire_schema(monkeypatch):
    import ai_marketing_api.main as api
    seen = []
    async def llm(system, payload, fallback, *_args, **_kwargs):
        seen.append(deepcopy(payload))
        assert payload["response_contract"][MARKER] == VERSION
        assert payload["selectable_facts"] == [
            {"text": row.content, "evidence_ids": [row.record_id]} for row in contexts()]
        return selected()
    monkeypatch.setattr(api, "_llm_json", llm)
    response = api_request("/api/v1/operators/industry/propose", request().model_dump(mode="json"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "operator.proposal.v1" and len(body["claims"]) == 2
    assert body["evidence_refs"] == ["fact-2", "fact-4"]
    assert MARKER not in response.text and "selectable_facts" not in response.text
    assert len(seen) == 1

def test_route_old_raw_replay_and_revise_cannot_opt_in(monkeypatch):
    import ai_marketing_api.main as api
    async def old_raw(_system, _payload, _fallback, *_args, **_kwargs):
        raw = selected((2,))
        del raw[MARKER]
        return raw
    monkeypatch.setattr(api, "_llm_json", old_raw)
    proposed = api_request("/api/v1/operators/industry/propose", request().model_dump(mode="json"))
    assert proposed.status_code == 200 and len(proposed.json()["claims"]) == 5
    async def revise(_system, payload, _fallback, *_args, **_kwargs):
        assert "selectable_facts" not in payload and MARKER not in payload["response_contract"]
        return selected((2,))
    monkeypatch.setattr(api, "_llm_json", revise)
    payload = request().model_dump(mode="json")
    payload.update(change_request="Review the current direction.", previous=proposed.json())
    result = api_request("/api/v1/operators/industry/revise", payload)
    assert result.status_code == 200 and len(result.json()["claims"]) == 5
