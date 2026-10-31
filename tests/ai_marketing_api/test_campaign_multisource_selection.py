"""Native planner source-selection contracts; all facts and model output are synthetic."""
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_editorial
from ai_marketing_api.operator_planning import EditorialPlanRequest, EditorialSelection, plan_editorial
from ai_marketing_api.operator_surface_selection import (
    IndustrySourceSelection, validate_source_selection,
)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload(counts=(2, 2, 1)):
    raw = {"mandate": {"objective": "Compare only the supplied documented boundaries.",
            "channels": ["wechat_mp", "wechat_moments"]},
        "genre": "industry_analysis", "max_packages": 1,
        "source_selection_version": "source_selection.v1", "opportunities": [],
        "industry_fact_catalog": {}, "industry_source_bindings": {}}
    for index, count in enumerate(counts):
        identity = "source-" + chr(ord("a") + index)
        uri = "https://official.example/" + identity
        bodies = [f"Document {identity} reports condition {n}: the stated boundary is limited to this supplied record."
                  for n in range(count)]
        document = "\n".join(bodies)
        raw["opportunities"].append({"id": identity, "role_id": "industry", "title": "A documented boundary",
            "summary": document, "verified": True, "url": uri})
        raw["industry_fact_catalog"][identity] = [{"record_id": f"{identity}-fact-{n}", "version": 2,
            "text": body, "body_sha256": sha(body), "excerpt_sha256": sha(body),
            "source_text_sha256": sha(document), "source_uri": uri} for n, body in enumerate(bodies)]
        raw["industry_source_bindings"][identity] = {"candidate_id": identity, "source_id": "native-" + identity,
            "source_kind": "public_document", "source_uri": uri, "source_text_sha256": sha(document),
            "source_document_sha256": sha(json.dumps({"source_text": document}, sort_keys=True)),
            "source_binding_hash": sha("synthetic-native-governance-" + identity)}
    return raw


def selection(*, supports=("source-b",), anchor="source-b-fact-0"):
    return {"opportunity_id": "source-a", "role_id": "industry", "topic": "The supplied documented boundary",
        "angle": "Compare the two records without assuming their meaning.", "reason": "The primary and supporting facts can be checked independently.",
        "value_contract": {"reader_problem": "Which documented boundary applies?", "thesis": "The records distinguish two qualified situations.",
            "evidence_anchor": {"opportunity_id": "source-a", "quote": payload()["industry_fact_catalog"]["source-a"][0]["text"]},
            "decision_delta": "Check the supplied conditions before choosing between them.", "counterargument": "An additional applicable record could change this comparison."},
        "source_selection": {"schema_version": "operator.industry_source_selection.v1", "primary_candidate_id": "source-a",
            "supporting_candidate_ids": list(supports),
            "supporting_record_ids": {identity: [identity + "-fact-0"] for identity in supports}},
        "industry_surface_selection": {"schema_version": "operator.industry_surface_selection.v1", "short_surfaces": {
            "wechat_moments": {"anchor_record_id": anchor, "reason": "Explain the specific selected boundary."}},
            "reported_case_record_ids": [], "case_gap": "No observed customer case is supplied."}}


def planned(monkeypatch, raw, item):
    request = EditorialPlanRequest.model_validate(deepcopy(raw))
    before = request.model_dump_json(); calls = []
    async def fake(system, body, fallback, *args, **kwargs):
        calls.append((system, deepcopy(body), kwargs))
        return {"_model": "synthetic-offline-planner", "selections": [deepcopy(item)]}
    async def immediate(awaitable, timeout):
        return await awaitable
    with monkeypatch.context() as context:
        context.setattr(operator_editorial.asyncio, "wait_for", immediate)
        coroutine = plan_editorial(request, fake)
        try:
            coroutine.send(None)
        except StopIteration as done:
            result = done.value
        else:
            pytest.fail("Synthetic planner must not suspend for I/O")
        finally:
            coroutine.close()
    assert request.model_dump_json() == before
    return result, calls


def test_two_distinct_document_hashes_remain_independent_in_one_bounded_planner_call(monkeypatch):
    raw = payload(); original = deepcopy(raw)
    result, calls = planned(monkeypatch, raw, selection())
    assert raw == original and result["status"] == "planned"
    assert result["selections"][0]["source_selection"] == selection()["source_selection"]
    assert result["selections"][0]["industry_surface_selection"] == selection()["industry_surface_selection"]
    assert len(calls) == result["generation_trace"]["model_calls"] == 1
    assert result["generation_trace"]["repair_calls"] == 0
    assert calls[0][1]["industry_source_bindings"] == raw["industry_source_bindings"]
    assert len({v["source_text_sha256"] for v in raw["industry_source_bindings"].values()}) == 3


def test_primary_with_two_supporting_sources_and_zero_support_legacy_shape_are_explicit(monkeypatch):
    for supports, anchor in [(("source-c", "source-b"), "source-c-fact-0"), ((), "source-a-fact-0")]:
        raw = payload(); item = selection(supports=supports, anchor=anchor)
        result, _ = planned(monkeypatch, raw, item)
        assert result["status"] == "planned" and result["selections"][0]["source_selection"] == item["source_selection"]
        request = EditorialPlanRequest.model_validate(raw)
        union = validate_source_selection(item["source_selection"], primary_candidate_id="source-a",
            opportunities=request.opportunities, catalogs=request.industry_fact_catalog, bindings=request.industry_source_bindings)
        assert [fact.record_id for fact in union] == [fact["record_id"] for candidate in ["source-a", *sorted(supports)]
            for fact in raw["industry_fact_catalog"][candidate]]


@pytest.mark.parametrize("defect", ["no_bindings", "wrong_version", "binding_keys", "opportunity_keys", "duplicate_opportunity",
    "wrong_role", "unverified", "wrong_genre", "source_kind", "blank_source", "fragment", "uri", "doc_hash", "binding_hash", "cross_catalog_record"])
def test_source_authorization_projection_drift_rejects_before_any_planner_call(defect):
    raw = payload(); binding = raw["industry_source_bindings"]["source-b"]
    if defect == "no_bindings": raw.pop("industry_source_bindings")
    elif defect == "wrong_version": raw["source_selection_version"] = "source_selection.v2"
    elif defect == "binding_keys": raw["industry_source_bindings"].pop("source-b")
    elif defect == "opportunity_keys": raw["opportunities"].pop()
    elif defect == "duplicate_opportunity": raw["opportunities"].append(deepcopy(raw["opportunities"][0]))
    elif defect == "wrong_role": raw["opportunities"][1]["role_id"] = "personal_ip"
    elif defect == "unverified": raw["opportunities"][1]["verified"] = False
    elif defect == "wrong_genre": raw["genre"] = "news"
    elif defect == "source_kind": binding["source_kind"] = "rss"
    elif defect == "blank_source": binding["source_id"] = "  "
    elif defect == "fragment": binding["source_uri"] += "#part"
    elif defect == "uri": binding["source_uri"] = "https://another.example/other"
    elif defect == "doc_hash": binding["source_text_sha256"] = "0" * 64
    elif defect == "binding_hash": binding["source_binding_hash"] = "not-a-hash"
    else: raw["industry_fact_catalog"]["source-b"][0]["record_id"] = "source-a-fact-0"
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(raw)


@pytest.mark.parametrize("defect", ["missing", "wrong_primary", "unknown_support", "self_support", "duplicate_support", "too_many_sources",
    "missing_records", "extra_records", "empty_records", "cross_source_record", "duplicate_record", "forged_owner", "unselected_anchor"])
def test_model_cannot_broaden_selected_sources_or_facts_and_never_repairs(monkeypatch, defect):
    raw = payload(); item = selection(); chosen = item["source_selection"]
    if defect == "missing": item.pop("source_selection")
    elif defect == "wrong_primary": chosen["primary_candidate_id"] = "source-b"
    elif defect == "unknown_support": chosen["supporting_candidate_ids"] = ["not-displayed"]; chosen["supporting_record_ids"] = {"not-displayed": ["missing-fact"]}
    elif defect == "self_support": chosen["supporting_candidate_ids"] = ["source-a"]; chosen["supporting_record_ids"] = {"source-a": ["source-a-fact-0"]}
    elif defect == "duplicate_support": chosen["supporting_candidate_ids"] *= 2
    elif defect == "too_many_sources": chosen["supporting_candidate_ids"] = ["source-b", "source-c", "source-d"]
    elif defect == "missing_records": chosen["supporting_record_ids"] = {}
    elif defect == "extra_records": chosen["supporting_record_ids"]["source-c"] = ["source-c-fact-0"]
    elif defect == "empty_records": chosen["supporting_record_ids"]["source-b"] = []
    elif defect == "cross_source_record": chosen["supporting_record_ids"]["source-b"] = ["source-c-fact-0"]
    elif defect == "duplicate_record": chosen["supporting_record_ids"]["source-b"] *= 2
    elif defect == "forged_owner": chosen["owner_approved"] = True
    else: item["industry_surface_selection"]["short_surfaces"]["wechat_moments"]["anchor_record_id"] = "source-c-fact-0"
    result, calls = planned(monkeypatch, raw, item)
    assert result["status"] == "unavailable" and result["selections"] == []
    assert len(calls) == result["generation_trace"]["model_calls"] == 1 and result["generation_trace"]["repair_calls"] == 0


def test_different_candidate_aliases_cannot_represent_one_canonical_uri(monkeypatch):
    raw = payload(); uri = "https://official.example:443/source-a/"
    raw["opportunities"][1]["url"] = uri; raw["industry_source_bindings"]["source-b"]["source_uri"] = uri
    for fact in raw["industry_fact_catalog"]["source-b"]:
        fact["source_uri"] = uri
    result, _ = planned(monkeypatch, raw, selection())
    assert result["status"] == "unavailable" and result["selections"] == []


def test_total_union_over_eight_is_rejected_without_truncation_or_more_calls(monkeypatch):
    raw = payload((5, 4)); before = deepcopy(raw)
    result, calls = planned(monkeypatch, raw, selection())
    assert raw == before and result["status"] == "unavailable" and result["selections"] == []
    assert len(calls) == 1 and result["generation_trace"]["repair_calls"] == 0
    assert sum(map(len, calls[0][1]["industry_fact_catalog"].values())) == 9


def test_old_absent_request_json_and_normalized_hmac_input_are_byte_exact():
    raw = {"mandate": {"objective": "Explain documented options"}, "opportunities": [{"id": "source-a", "role_id": "industry",
        "title": "Documented handover", "summary": "The published handover contains conclusions and citations, but does not contain the original inputs.",
        "verified": True, "url": "https://example.invalid/primary"}], "max_packages": 1}
    normalized = EditorialPlanRequest.model_validate(raw).model_dump_json().encode()
    assert hashlib.sha256(normalized).hexdigest() == "6eb147a072ffb02891897d3bcda8179590eb81674adf8a2a26b5fbbf58e5be5b"
    assert "source_selection_version" not in EditorialPlanRequest.model_validate(raw).model_dump(mode="json")
    assert "industry_source_bindings" not in EditorialPlanRequest.model_validate(raw).model_dump(mode="json")
    old = selection(); old.pop("source_selection")
    assert "source_selection" not in EditorialSelection.model_validate(old).model_dump(mode="json")


@pytest.mark.parametrize("unsolicited", [None, {"schema_version": "operator.industry_source_selection.v1", "primary_candidate_id": "source-a",
    "supporting_candidate_ids": [], "supporting_record_ids": {}}])
def test_unsolicited_source_selection_is_rejected_even_with_null(monkeypatch, unsolicited):
    raw = payload(); raw.pop("source_selection_version"); raw.pop("industry_source_bindings")
    item = selection(); item["source_selection"] = unsolicited
    result, calls = planned(monkeypatch, raw, item)
    assert result["status"] == "unavailable" and result["selections"] == [] and len(calls) == 1
    raw["industry_source_bindings"] = payload()["industry_source_bindings"]
    with pytest.raises(ValidationError):
        EditorialPlanRequest.model_validate(raw)
