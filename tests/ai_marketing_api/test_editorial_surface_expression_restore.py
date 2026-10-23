"""A signed, supported sentence belongs to its surface, not every shared fact."""
from __future__ import annotations

from copy import deepcopy

import pytest

from ai_marketing_api import main as api, operator_steps as steps
from tests.ai_marketing_api.test_editorial_layout_and_model_version import _layout_normalize
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review
from tests.ai_marketing_api.test_industry_analysis_block_order import _canonical
from tests.ai_marketing_api.test_operator_api import _request


EXPRESSIONS = {
    "master": "示例接口先约定判断问题与可选答案。",
    "wechat_mp": "判断问题与允许答案需先为示例接口定义。",
    "wechat_moments": "示例接口的判断问题和答案范围要先写清。",
    "weitoutiao": "先给示例接口定义问题与允许答案。",
}
BASE = "/api/v1/operators/industry/editorial-steps/"


def _surfaces(candidate):
    return {"master": (candidate["master_content"], candidate["blocks"]),
        **{variant["platform"]: (variant["body"], variant["blocks"])
            for variant in candidate["platform_variants"]}}


def _verified(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_API_KEY", "isolated-surface-expression-secret")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "isolated-" + stage})
    request, _, normalized, _, raw = _layout_normalize()
    compose = request.model_dump(mode="json")
    compose["fact_expression_mode"] = "grounded_paraphrase"
    canonical = next(block for block in normalized.blocks if block.source_exact)
    rows = {"master": raw["blocks"], **{v["platform"]: v["blocks"] for v in raw["platform_variants"]}}
    for surface, blocks in rows.items():
        next(block for block in blocks if block.get("block_ref") == canonical.binding_hash)["public_text"] = EXPRESSIONS[surface]
    calls = []

    async def independent_verifier(system, body, fallback, **kwargs):
        calls.append(body)
        return {"_model": "isolated-independent-verifier", "reviews": [{
            "pair_id": pair["pair_id"], "verdict": "supported", "key_fields_preserved": True,
            "reason": "Offline protocol fixture; not an actual content-quality assessment."}
            for pair in body["claim_verification"]["pairs"]]}

    monkeypatch.setattr(api, "_llm_json", independent_verifier)
    response = _request("POST", BASE + "verify_claims", json={"compose": compose, "candidate": raw,
        "budget_reservation_id": "surface-expression-verify"})
    assert response.status_code == 200
    verified = response.json()
    assert verified["status"] == "artifact", verified
    assert verified["generation_trace"]["model_calls"] == len(calls) == 1
    assert len(calls[0]["claim_verification"]["pairs"]) == len(EXPRESSIONS)
    assert len(verified["artifact"]["claim_binding_reviews"]) == len(EXPRESSIONS)
    for surface, (body, blocks) in _surfaces(verified["artifact"]).items():
        block = next(block for block in blocks if block["binding_hash"] == canonical.binding_hash)
        assert block["public_text"] == EXPRESSIONS[surface]
        assert block["public_text_verified"] is True
        assert EXPRESSIONS[surface] in body and canonical.text not in body
    return compose, verified, canonical.binding_hash


@pytest.mark.parametrize("reverse_variants", [False, True])
def test_real_asgi_verify_sign_review_retains_each_surface_expression(monkeypatch, tmp_path, reverse_variants):
    compose, verified, _ = _verified(monkeypatch, tmp_path)
    candidate = deepcopy(verified["artifact"])
    if reverse_variants:
        # Independent expressions must not depend on which platform is last.
        candidate["platform_variants"].reverse()
    before = deepcopy(_surfaces(candidate))
    reviews = []

    async def editor(system, body, fallback, **kwargs):
        reviews.append(body)
        draft = body["editorial_review"]["draft"]
        assert draft["master_content"] == before["master"][0]
        for variant in draft["platform_variants"]:
            assert variant["body"] == before[variant["platform"]][0]
        return review(body)

    monkeypatch.setattr(api, "_llm_json", editor)
    response = _request("POST", BASE + "review_editorial", json={"compose": compose,
        "candidate": candidate, "candidate_signature": steps.artifact_signature(candidate),
        "budget_reservation_id": "surface-expression-review"})
    assert response.status_code == 200
    assessed = response.json()
    assert assessed["status"] == "artifact", assessed
    assert assessed["generation_trace"]["model_calls"] == len(reviews) == 1
    assert assessed["artifact"]["claim_binding_reviews"] == candidate["claim_binding_reviews"]
    for surface, (body, blocks) in _surfaces(assessed["artifact"]).items():
        assert body == before[surface][0]
        assert blocks == before[surface][1]
        assert _canonical(blocks) == _canonical(before[surface][1])
    assert assessed["publishable"] is False


@pytest.mark.parametrize("failure", ["unsigned", "tampered_signature"])
def test_review_never_trusts_unsigned_or_changed_signed_candidate(monkeypatch, tmp_path, failure):
    compose, verified, _ = _verified(monkeypatch, tmp_path)
    candidate = deepcopy(verified["artifact"])
    signature = verified["artifact_signature"]
    if failure == "tampered_signature":
        candidate["blocks"][1]["public_text"] = "Changed after the service signed the artifact."
    else:
        signature = None

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid signed-review input must stop before any editor call")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    result = _request("POST", BASE + "review_editorial", json={"compose": compose,
        "candidate": candidate, "candidate_signature": signature}).json()
    assert result["status"] == "unavailable"
    assert result["error_code"] == ("signed_candidate_required" if failure == "unsigned" else "artifact_signature_invalid")
    assert result["artifact"] is None and result["generation_trace"]["model_calls"] == 0


def test_unsigned_verified_flags_are_rechecked_by_the_independent_verifier(monkeypatch, tmp_path):
    compose, verified, binding_hash = _verified(monkeypatch, tmp_path)
    calls = []

    async def uncertain(system, body, fallback, **kwargs):
        calls.append(body)
        return {"_model": "isolated-independent-verifier", "reviews": [{
            "pair_id": pair["pair_id"], "verdict": "uncertain", "key_fields_preserved": False}
            for pair in body["claim_verification"]["pairs"]]}

    monkeypatch.setattr(api, "_llm_json", uncertain)
    result = _request("POST", BASE + "verify_claims", json={"compose": compose,
        "candidate": verified["artifact"], "budget_reservation_id": "surface-expression-untrusted-recheck"}).json()
    assert result["status"] == "artifact", result
    assert len(calls) == 1
    for _, blocks in _surfaces(result["artifact"]).values():
        block = next(block for block in blocks if block["binding_hash"] == binding_hash)
        assert block["public_text_verified"] is False
    assert all(row["verdict"] != "supported" for row in result["artifact"]["claim_binding_reviews"])


@pytest.mark.parametrize("failure", ["missing", "source_hash", "expression_hash", "claim", "evidence", "verdict", "model"])
def test_signed_expression_requires_matching_real_support_record(monkeypatch, tmp_path, failure):
    compose, verified, _ = _verified(monkeypatch, tmp_path)
    candidate = deepcopy(verified["artifact"])
    row = candidate["claim_binding_reviews"][0]
    if failure == "missing":
        candidate["claim_binding_reviews"].pop(0)
    else:
        field, value = {"source_hash": ("source_sha256", "0" * 64),
            "expression_hash": ("public_text_sha256", "0" * 64),
            "claim": ("claim_id", "other-claim"), "evidence": ("evidence_ids", ["other-source"]),
            "verdict": ("verdict", "uncertain"), "model": ("review_model", "deterministic")}[failure]
        row[field] = value

    async def forbidden(*args, **kwargs):
        pytest.fail("A missing support binding cannot be repaired by buying an editor call")

    monkeypatch.setattr(api, "_llm_json", forbidden)
    result = _request("POST", BASE + "review_editorial", json={"compose": compose,
        "candidate": candidate, "candidate_signature": steps.artifact_signature(candidate)}).json()
    assert result["status"] == "unavailable"
    assert result["error_code"] == "artifact_public_expression_invalid"
    assert result["artifact"] is None and result["generation_trace"]["model_calls"] == 0
