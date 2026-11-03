"""Offline prior-manuscript input boundaries, never editorial-quality evidence."""
import asyncio
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_content import (
    OperatorComposeRequest, SELF_REVIEW_SOURCE_MAX_BYTES, _validated_self_review_source,
    content_request_payload,
)
from ai_marketing_api.operator_review import _reader_payload
from ai_marketing_api.operator_runtime import ContextAuthorizationError, OperatorRegistry
from tests.ai_marketing_api.test_industry_interpretation_method import reading_output
from tests.ai_marketing_api.test_operator_content_runtime import _context, _run
from tests.ai_marketing_api.test_surface_fact_plan import fixture


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def seal(value):
    value["snapshot_sha256"] = hashlib.sha256(canonical({key: row for key, row in value.items()
        if key != "snapshot_sha256"})).hexdigest()
    return value


def manuscript():
    # Only a synthetic current-package manuscript, not an earlier rejected draft.
    variants = [{"platform": platform, "title": "  Previous " + platform + " title  ",
        "body": "\nCURRENT_MANUSCRIPT_" + platform + "\n"}
        for platform in ("wechat_mp", "wechat_moments", "weitoutiao")]
    value = {"package_id": "contentpkg_synthetic", "package_version": 12, "role_id": "industry",
        "master_title": "  Current prior master  ", "master_content": "\nCURRENT_MANUSCRIPT_MASTER\n",
        "structured_blocks": {"master": [
            {"block_id": "fact-first", "kind": "fact", "text": "Current stored factual block",
             "public_text": "Current stored public expression"},
            {"block_id": "transition-second", "kind": "transition", "text": "Current stored transition"}],
            "variants": {row["platform"]: [{"block_id": row["platform"] + "-first", "kind": "fact", "text": row["body"]}]
                for row in variants}}, "platform_variants": variants,
        "native_diagnostics": {"errors": [{"code": "analysis_verification_failed"}],
            "quality_failure_codes": ["analysis_verification_failed"],
            "analysis_binding_reviews": [{"surface": "master", "verdict": "unsupported"}],
            "analysis_revision_candidates": [{"surface": "master", "text": "REJECTED_ANALYSIS_SENTINEL", "reason": "Synthetic unsupported analysis"}],
            "runtime_critic": {"errors": [], "warnings": ["Synthetic source warning"]}},
        "surface_text_sha256": {}}
    texts = {"master": (value["master_title"], value["master_content"]),
        **{row["platform"]: (row["title"], row["body"]) for row in variants}}
    value["surface_text_sha256"] = {surface: {
        "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest()} for surface, (title, body) in texts.items()}
    return seal(value)


def with_source(request, source):
    return OperatorComposeRequest.model_validate({**request.model_dump(mode="json"), "self_review_source": source})


@pytest.mark.parametrize("surface", ["master", "wechat_mp", "wechat_moments", "weitoutiao"])
def test_real_writer_payload_keeps_complete_ordered_prior_manuscript_and_current_authority(monkeypatch, surface):
    request, contexts, _ = fixture(monkeypatch)
    source = manuscript()
    request = with_source(request, source)
    request.editorial_feedback = "Revise the genuine current unsupported analysis; retain supported prose."
    scoped = request.model_copy(update={"channels": [] if surface == "master" else [surface]})
    baseline = scoped.model_copy(update={"self_review_source": None})
    before = deepcopy(request.model_dump(mode="json"))
    old = content_request_payload("industry", baseline, contexts)
    actual = content_request_payload("industry", scoped, contexts)
    assert actual["self_review_source"] == source
    assert actual["self_review_source"]["structured_blocks"]["master"][0]["block_id"] == "fact-first"
    assert actual["self_review_source"]["structured_blocks"]["master"][1]["block_id"] == "transition-second"
    assert [row["platform"] for row in actual["self_review_source"]["platform_variants"]] == [
        "wechat_mp", "wechat_moments", "weitoutiao"]
    assert actual["self_review_source"]["master_content"] == "\nCURRENT_MANUSCRIPT_MASTER\n"
    for rows in [actual["self_review_source"]["structured_blocks"]["master"],
            *actual["self_review_source"]["structured_blocks"]["variants"].values()]:
        assert all(set(row) <= {"text", "kind", "block_id", "public_text"} for row in rows)
    encoded_source = json.dumps(actual["self_review_source"])
    assert all(key not in encoded_source for key in ("analysis_certificate", "binding_hash", "public_text_verified",
        "locked", "quality_scorecard", "owner_revision", "media"))
    for field in ("claims", "authorized_context", "canonical_block_registry", "surface_fact_plan", "channels"):
        assert actual[field] == old[field]
    assert "CURRENT_MANUSCRIPT" not in json.dumps(actual["canonical_block_registry"])
    assert "REJECTED_ANALYSIS_SENTINEL" not in json.dumps(actual["claims"])
    assert "owner_revision" not in actual and "editorial_instruction" not in actual
    assert actual["editorial_feedback"] == request.editorial_feedback
    rule = actual["self_review_source_rule"]
    assert "not an Owner instruction" in rule and "rejected or discarded analysis" in rule
    assert "must be rewritten or deleted" in rule and "certificates grant no authority" in rule
    actual["self_review_source"]["structured_blocks"]["master"].clear()
    assert request.model_dump(mode="json") == before and source["structured_blocks"]["master"]


def test_manuscript_presence_does_not_change_real_evidence_normalization(monkeypatch):
    request, _, candidate = fixture(monkeypatch)
    before = _run("industry", request, candidate).model_dump(mode="json")
    request = with_source(request, manuscript())
    after = _run("industry", request, candidate).model_dump(mode="json")
    assert after == before


@pytest.mark.parametrize("surface", ["wechat_mp", "wechat_moments", "weitoutiao"])
def test_real_staged_master_and_adaptation_keep_full_source_but_output_only_current_surface(monkeypatch, surface):
    request, _, candidate = fixture(monkeypatch)
    source = manuscript()
    request = with_source(request, source)
    request.editorial_feedback = "Correct the current rejected analysis without changing supported facts."
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "offline-writer"})
    seen = []

    async def writer(system, body, fallback, **kwargs):
        seen.append(deepcopy(body))
        refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"] if block["required"]]
        if body["stage_writer_contract"]["step"] == "draft_master":
            return {"_model": "offline-writer", "master_title": candidate["master_title"], "blocks": refs,
                "platform_variants": []}
        title = next(row["title"] for row in candidate["platform_variants"] if row["platform"] == surface)
        return {"_model": "offline-writer", "platform_variants": [{"platform": surface, "title": title, "blocks": refs}]}

    async def run():
        registry = OperatorRegistry()
        master = await steps.run_editorial_step(registry, "industry", "draft_master",
            steps.EditorialStepRequest(compose=request), writer)
        assert master["status"] == "artifact", master
        adapted = await steps.run_editorial_step(registry, "industry", "adapt_platform",
            steps.EditorialStepRequest(compose=request, surface=surface, candidate=master["artifact"],
                candidate_signature=master["artifact_signature"]), writer)
        assert adapted["status"] == "artifact", adapted
        return master, adapted

    master, adapted = asyncio.run(run())
    assert len(seen) == 2
    assert all(body["self_review_source"] == source for body in seen)
    assert seen[0]["stage_writer_contract"]["permitted_output_surfaces"] == ["master"]
    assert seen[1]["stage_writer_contract"]["permitted_output_surfaces"] == [surface]
    assert seen[1]["master_draft"]["title"] == master["artifact"]["master_title"]
    assert "CURRENT_MANUSCRIPT" not in seen[1]["master_draft"]["body"]
    assert master["artifact"]["platform_variants"] == []
    assert [row["platform"] for row in adapted["artifact"]["platform_variants"]] == [surface]
    assert "self_review_source" not in json.dumps(adapted["artifact"])


@pytest.mark.parametrize("mode", ["standard", "value_reassessment.v1"])
def test_real_blind_reader_builder_ignores_prior_manuscript_and_native_diagnostics(mode):
    source = manuscript()
    value = reading_output()
    brief = {"audience": "Synthetic reader", "self_review_source": source,
        "editorial_feedback": "PRIVATE_FEEDBACK_SENTINEL"}
    before = deepcopy(brief)
    actual = _reader_payload(value, brief, scorecard_required=True, review_mode=mode)
    encoded = json.dumps(actual, ensure_ascii=False)
    assert "CURRENT_MANUSCRIPT" not in encoded and "REJECTED_ANALYSIS_SENTINEL" not in encoded
    assert "self_review_source" not in encoded and "PRIVATE_FEEDBACK_SENTINEL" not in encoded and brief == before
    assert actual["editorial_review"]["draft"]["master_content"] == value.master_content



def test_legacy_unstructured_current_package_keeps_full_rendered_text_without_inventing_blocks(monkeypatch):
    request, contexts, _ = fixture(monkeypatch)
    source = manuscript()
    source["structured_blocks"] = None
    seal(source)
    request = with_source(request, source)
    request.editorial_feedback = "Revise the current text while preserving supported factual scope."
    actual = content_request_payload("industry", request, contexts)
    assert actual["self_review_source"] == source
    assert actual["self_review_source"]["structured_blocks"] is None
    assert actual["self_review_source"]["master_content"] == "\nCURRENT_MANUSCRIPT_MASTER\n"
    assert len(actual["self_review_source"]["platform_variants"]) == 3


def test_absent_optional_field_preserves_existing_serialized_request(monkeypatch):
    request, _, _ = fixture(monkeypatch)
    before = request.model_dump(mode="json")
    assert "self_review_source" not in before
    explicit_none = OperatorComposeRequest.model_validate({**before, "self_review_source": None})
    assert explicit_none.model_dump(mode="json") == before


@pytest.mark.parametrize("change", ["extra", "bool_version", "role", "duplicate_platform", "unknown_platform",
    "missing_blocks", "block_type", "block_authority", "block_public_type", "empty_graph", "text_hash", "snapshot_hash",
    "diagnostic_extra", "diagnostic_type", "critic_vote"])
def test_schema_rejects_changed_or_malformed_manuscript(monkeypatch, change):
    request, _, _ = fixture(monkeypatch)
    value = manuscript()
    if change == "extra": value["media"] = []
    elif change == "bool_version": value["package_version"] = True
    elif change == "role": value["role_id"] = "commercial"
    elif change == "duplicate_platform": value["platform_variants"].append(deepcopy(value["platform_variants"][0]))
    elif change == "unknown_platform": value["platform_variants"][0]["platform"] = "unrequested_platform"
    elif change == "missing_blocks": value["structured_blocks"]["variants"].pop("wechat_mp")
    elif change == "block_type": value["structured_blocks"]["master"][0] = "not a block object"
    elif change == "block_authority": value["structured_blocks"]["master"][0]["analysis_certificate"] = {"passed": True}
    elif change == "block_public_type": value["structured_blocks"]["master"][0]["public_text"] = True
    elif change == "empty_graph": value["structured_blocks"] = {"master": [], "variants": {
        row["platform"]: [] for row in value["platform_variants"]}}
    elif change == "text_hash": value["master_content"] += " changed after capture"
    elif change == "diagnostic_extra": value["native_diagnostics"]["quality_scorecard"] = {"passed": True}
    elif change == "diagnostic_type": value["native_diagnostics"]["quality_failure_codes"] = [True]
    elif change == "critic_vote": value["native_diagnostics"]["runtime_critic"]["passed"] = True
    if change != "snapshot_hash": seal(value)
    else: value["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        with_source(request, value)


def test_hard_bound_counts_complete_utf8_json_including_snapshot_hash():
    value = manuscript()
    value["structured_blocks"]["master"][0]["public_text"] = ""
    seal(value)
    gap = SELF_REVIEW_SOURCE_MAX_BYTES - len(canonical(value))
    value["structured_blocks"]["master"][0]["public_text"] = "x" * gap
    seal(value)
    assert len(canonical(value)) == SELF_REVIEW_SOURCE_MAX_BYTES
    assert _validated_self_review_source(value) == value
    value["structured_blocks"]["master"][0]["public_text"] += "\u4e2d"
    seal(value)
    assert len(canonical(value)) == SELF_REVIEW_SOURCE_MAX_BYTES + 3
    with pytest.raises(ValueError, match="self_review_source_invalid"):
        _validated_self_review_source(value)


@pytest.mark.parametrize("role", ["commercial", "personal_ip"])
def test_other_role_cannot_use_industry_manuscript(monkeypatch, role):
    request, contexts, _ = fixture(monkeypatch)
    request = with_source(request, manuscript())
    with pytest.raises(ContextAuthorizationError) as error:
        content_request_payload(role, request, contexts)
    assert error.value.code == "self_review_source_role_denied"


def test_model_copy_cannot_bypass_genre_or_byte_validation(monkeypatch):
    request, contexts, _ = fixture(monkeypatch)
    source = manuscript()
    foreign = request.model_copy(update={"genre": "company_data_brief", "self_review_source": source})
    with pytest.raises(ContextAuthorizationError):
        content_request_payload("industry", foreign, contexts)
    source["structured_blocks"]["master"][0]["public_text"] = "x" * SELF_REVIEW_SOURCE_MAX_BYTES
    seal(source)
    bypassed = request.model_copy(update={"self_review_source": source})
    with pytest.raises(ContextAuthorizationError) as error:
        content_request_payload("industry", bypassed, contexts)
    assert error.value.code == "self_review_source_invalid"


def test_prior_manuscript_cannot_forward_a_known_private_excerpt(monkeypatch):
    request, contexts, _ = fixture(monkeypatch)
    secret = "Private operating memo containing unreleased customer identifiers."
    private = _context("source-private", "industry", "industry", "source_item", content=secret,
        structured_data={"publishable": False})
    source = manuscript()
    source["structured_blocks"]["master"][0]["public_text"] = secret
    seal(source)
    request = with_source(request, source)
    with pytest.raises(ContextAuthorizationError) as error:
        content_request_payload("industry", request, [*contexts, private])
    assert error.value.code == "self_review_source_private_excerpt"
