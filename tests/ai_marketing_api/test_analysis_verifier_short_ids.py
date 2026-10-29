"""Synthetic verifier protocol regression tests; no live model or human grade."""
from copy import deepcopy
import re

import pytest

from ai_marketing_api.operator_analysis import digest, pending_identity
from tests.ai_marketing_api import test_semantic_editorial_analysis as semantic


service = semantic.service


def test_multiple_short_ids_map_back_to_exact_full_server_bindings(service):
    value = semantic._value()
    candidate = semantic._written(value)
    before = deepcopy(candidate)
    result = semantic._verify(value, candidate)
    assert result["status"] == "artifact", result
    body = service["calls"][-1][1]
    requested = body["analysis_verification"]["candidates"]
    assert len(requested) > 1
    assert len({row["analysis_id"] for row in requested}) == len(requested)
    assert all(re.fullmatch(r"analysis-[1-9][0-9]*", row["analysis_id"]) for row in requested)
    assert all(len(row["analysis_id"]) < 20 for row in requested)
    expected = [pending_identity(row) for row in before["pending_editorial_analysis"]]
    assert [pending_identity(row) for row in requested] == expected
    reviews = result["artifact"]["analysis_binding_reviews"]
    assert [pending_identity(row) for row in reviews] == expected
    assert all(row["verdict"] == "supported" and row["review_model"] == "synthetic-independent-verifier"
               and "analysis_id" not in row for row in reviews)
    assert candidate == before
    assert not result["artifact"].get("pending_editorial_analysis")
    assert result["publishable"] is False and result["requires_human_review"] is True
    repeated = semantic._verify(value, candidate)
    assert repeated["status"] == "artifact", repeated
    assert service["calls"][-1][1]["analysis_verification"]["candidates"] == requested


def test_reordered_short_id_decisions_bind_by_exact_id_not_response_position(service, monkeypatch):
    original = semantic._decisions

    def reordered(body, **kwargs):
        result = original(body, **kwargs)
        for row in result["analysis_reviews"]:
            row["reason"] = "Synthetic support checked for " + row["analysis_id"]
        result["analysis_reviews"].reverse()
        return result

    monkeypatch.setattr(semantic, "_decisions", reordered)
    result = semantic._verify(semantic._value(), semantic._written(semantic._value()))
    assert result["status"] == "artifact", result
    requested = service["calls"][-1][1]["analysis_verification"]["candidates"]
    expected = {digest(pending_identity(row)): "Synthetic support checked for " + row["analysis_id"]
                for row in requested}
    assert {digest(pending_identity(row)): row["reason"]
            for row in result["artifact"]["analysis_binding_reviews"]} == expected


@pytest.mark.parametrize("malformed", ["full_digest", "unknown", "duplicate", "missing", "leading_space",
                                      "trailing_space", "wrong_case", "zero_padded", "boolean"])
def test_old_hash_and_nonexact_aliases_never_create_any_attestation(service, monkeypatch, malformed):
    original = semantic._decisions

    def broken(body, **kwargs):
        result = original(body, **kwargs)
        rows = result["analysis_reviews"]
        if malformed == "full_digest":
            # Reconstruct the OLD wire identity exactly, rather than accepting
            # old paid results under the new protocol or guessing a close ID.
            pending = {"schema_version": "operator.pending-editorial-analysis.v1",
                       **pending_identity(body["analysis_verification"]["candidates"][0])}
            rows[0]["analysis_id"] = digest(pending)
        elif malformed == "unknown":
            rows[0]["analysis_id"] = "analysis-999999"
        elif malformed == "duplicate":
            rows[1]["analysis_id"] = rows[0]["analysis_id"]
        elif malformed == "missing":
            rows.pop()
        elif malformed == "leading_space":
            rows[0]["analysis_id"] = " " + rows[0]["analysis_id"]
        elif malformed == "trailing_space":
            rows[0]["analysis_id"] += " "
        elif malformed == "wrong_case":
            rows[0]["analysis_id"] = rows[0]["analysis_id"].upper()
        elif malformed == "zero_padded":
            rows[0]["analysis_id"] = "analysis-01"
        else:
            rows[0]["analysis_id"] = True
        return result

    monkeypatch.setattr(semantic, "_decisions", broken)
    value = semantic._value()
    candidate = semantic._written(value)
    result = semantic._verify(value, candidate)
    assert result["status"] == "unavailable" and result["error_code"] == "analysis_verifier_not_completed"
    assert result["artifact"] is None
    assert result["publishable"] is False and result["requires_human_review"] is True
    assert service["calls"][-1][0] == "claim_verification"


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_one_negative_short_id_preserves_its_full_identity_and_blocks_the_complete_batch(service, monkeypatch, verdict):
    original = semantic._decisions

    def one_negative(body, **kwargs):
        result = original(body, **kwargs)
        result["analysis_reviews"][1]["verdict"] = verdict
        return result

    monkeypatch.setattr(semantic, "_decisions", one_negative)
    value = semantic._value()
    candidate = semantic._written(value)
    result = semantic._verify(value, candidate)
    assert result["status"] == "blocked", result
    artifact = result["artifact"]
    expected = [pending_identity(row) for row in candidate["pending_editorial_analysis"]]
    assert [pending_identity(row) for row in artifact["analysis_binding_reviews"]] == expected
    assert artifact["analysis_binding_reviews"][1]["verdict"] == verdict
    assert artifact["pending_editorial_analysis"]
    assert artifact["blocks"][1]["analysis_pending"] is True
    assert semantic.PROSE not in artifact["master_content"]
    assert result["publishable"] is False and result["requires_human_review"] is True


def test_short_ids_leave_mixed_claim_review_and_exact_context_guard_intact(service):
    service["public"] = True
    value = semantic._value()
    result = semantic._verify(value, semantic._written(value))
    assert result["status"] == "artifact", result
    body = service["calls"][-1][1]
    assert body["claim_verification"]["pairs"]
    assert all(re.fullmatch(r"[a-f0-9]{64}", row["pair_id"])
               for row in body["claim_verification"]["pairs"])
    candidate = deepcopy(result["artifact"])
    candidate["master_title"] = "示例接口与另一个判断边界"
    before = len(service["calls"])
    assessed = semantic._request("POST", semantic.BASE + "review_editorial", json={"compose": value,
        "candidate": candidate, "candidate_signature": semantic.steps.artifact_signature(candidate)}).json()
    assert assessed["status"] != "artifact", assessed
    assert len(service["calls"]) == before
