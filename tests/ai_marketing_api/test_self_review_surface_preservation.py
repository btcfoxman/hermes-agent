"""Synthetic preservation contracts; real verification remains mandatory."""
import asyncio
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_content import OperatorComposeRequest
from ai_marketing_api.operator_review import _reader_payload
from ai_marketing_api.operator_runtime import OperatorRegistry
from tests.ai_marketing_api.test_industry_interpretation_method import reading_output
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import review as scripted_review
from tests.ai_marketing_api.test_operator_content_runtime import _run
from tests.ai_marketing_api.test_self_review_source import seal
from tests.ai_marketing_api.test_surface_fact_plan import fixture


ANALYSIS = "Choose the next test according to the unresolved editorial question."


def snapshot(monkeypatch, targets=("weitoutiao",)):
    request, contexts, candidate = fixture(monkeypatch)
    for blocks in [candidate["blocks"], *[row["blocks"] for row in candidate["platform_variants"]]]:
        blocks.append({"kind": "opinion", "text": ANALYSIS, "evidence_ids": []})
    previous = _run("industry", request, candidate)
    assert previous.critic.errors == []

    def rows(blocks):
        return [{key: raw[key] for key in ("block_id", "kind", "text", "public_text", "paragraph_id")
            if key in raw and raw[key] is not None} for raw in
            [block.model_dump(mode="json") for block in blocks]]

    variants = [{"platform": row.platform, "title": row.title, "body": row.body}
        for row in previous.platform_variants]
    value = {"package_id": "contentpkg_preservation_synthetic", "package_version": 15,
        "role_id": "industry", "master_title": previous.master_title,
        "master_content": previous.master_content,
        "structured_blocks": {"master": rows(previous.blocks), "variants": {
            row.platform: rows(row.blocks) for row in previous.platform_variants}},
        "platform_variants": variants, "native_diagnostics": {"errors": [],
            "quality_failure_codes": [], "analysis_binding_reviews": [],
            "analysis_revision_candidates": [], "runtime_critic": {"errors": [], "warnings": []}},
        "surface_text_sha256": {}}
    for surface, title, body in [("master", value["master_title"], value["master_content"]),
            *[(row["platform"], row["title"], row["body"]) for row in variants]]:
        value["surface_text_sha256"][surface] = {
            "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
            "body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    if targets is not None:
        value["target_surfaces"] = list(targets)
    seal(value)
    request = OperatorComposeRequest.model_validate({**request.model_dump(mode="json"),
        "self_review_source": value})
    return request, contexts, candidate, previous


@pytest.mark.parametrize("bad", [[], ["foreign"], ["wechat_mp", "wechat_mp"],
    ["weitoutiao", "master"], [True], None])
def test_target_scope_is_exact_sorted_and_snapshot_bound(monkeypatch, bad):
    request, _, _, _ = snapshot(monkeypatch)
    raw = request.model_dump(mode="json")
    raw["self_review_source"]["target_surfaces"] = bad
    seal(raw["self_review_source"])
    with pytest.raises(ValidationError):
        OperatorComposeRequest.model_validate(raw)


def test_target_tamper_without_new_snapshot_hash_is_rejected(monkeypatch):
    request, _, _, _ = snapshot(monkeypatch)
    raw = request.model_dump(mode="json")
    raw["self_review_source"]["target_surfaces"] = ["wechat_mp"]
    with pytest.raises(ValidationError):
        OperatorComposeRequest.model_validate(raw)


@pytest.mark.parametrize("authority", ["analysis_certificate", "public_text_verified", "claim_id", "evidence_ids"])
def test_old_authority_fields_cannot_enter_the_preservation_snapshot(monkeypatch, authority):
    request, _, _, _ = snapshot(monkeypatch)
    raw = request.model_dump(mode="json")
    raw["self_review_source"]["structured_blocks"]["master"][0][authority] = "old-authority"
    seal(raw["self_review_source"])
    with pytest.raises(ValidationError):
        OperatorComposeRequest.model_validate(raw)


@pytest.mark.parametrize("paragraph", ["paragraph_a", "A1", "bad paragraph", "1bad", "x" * 49])
def test_paragraph_layout_remains_strict_optional_data(monkeypatch, paragraph):
    request, _, _, _ = snapshot(monkeypatch)
    raw = request.model_dump(mode="json")
    raw["self_review_source"]["structured_blocks"]["master"][0]["paragraph_id"] = paragraph
    seal(raw["self_review_source"])
    if paragraph in {"paragraph_a", "A1"}:
        assert OperatorComposeRequest.model_validate(raw).self_review_source == raw["self_review_source"]
    else:
        with pytest.raises(ValidationError):
            OperatorComposeRequest.model_validate(raw)


@pytest.mark.parametrize("mutation", ["missing", "ambiguous", "unknown_fact"])
def test_current_canonical_identity_is_required_not_the_old_block_id(monkeypatch, mutation):
    request, contexts, _, _ = snapshot(monkeypatch)
    canonical = steps.content_request_payload("industry", request, contexts)["canonical_block_registry"]
    raw = request.self_review_source["structured_blocks"]["master"][0]
    matches = [row for row in canonical if row["kind"] == raw["kind"] and row["text"] == raw["text"]]
    assert len(matches) == 1
    if mutation == "missing":
        canonical = [row for row in canonical if row is not matches[0]]
    elif mutation == "ambiguous":
        canonical.append(deepcopy(matches[0]))
    else:
        raw["text"] = "An unavailable factual statement."
    with pytest.raises(ValueError, match="self_review_preservation_invalid"):
        steps._preserved_writer_candidate({"_model": "synthetic"}, request, "master", canonical)


def test_reconstruction_uses_current_refs_and_keeps_paraphrases_unverified(monkeypatch):
    request, contexts, _, _ = snapshot(monkeypatch)
    source = request.self_review_source
    source["structured_blocks"]["master"][0]["public_text"] = "A proposed public expression."
    source["structured_blocks"]["master"][0]["paragraph_id"] = "intro"
    seal(source)
    canonical = steps.content_request_payload("industry", request, contexts)["canonical_block_registry"]
    original = {"_model": "synthetic", "master_title": "New model wording", "blocks": []}
    before = deepcopy(original)
    actual = steps._preserved_writer_candidate(original, request, "master", canonical)
    first = actual["blocks"][0]
    assert first["block_ref"] in {row["block_ref"] for row in canonical}
    assert first["public_text"] == "A proposed public expression."
    assert first["paragraph_id"] == "intro"
    assert set(first) == {"block_ref", "public_text", "paragraph_id"}
    assert original == before
    assert "certificate" not in json.dumps(actual)


def test_real_writer_call_and_receipt_are_preserved_before_server_reconstruction(monkeypatch):
    request, _, _, previous = snapshot(monkeypatch)
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-writer"})
    returned = []

    async def writer(_system, body, _fallback, **_kwargs):
        refs = [{"block_ref": row["block_ref"]} for row in body["canonical_block_registry"]]
        candidate = {"_model": "synthetic-writer", "master_title": request.approved_proposal.title,
            "blocks": refs, "platform_variants": []}
        returned.append(deepcopy(candidate))
        return candidate

    original_request = deepcopy(request.model_dump(mode="json"))
    result = asyncio.run(steps.run_editorial_step(OperatorRegistry(), "industry", "draft_master",
        steps.EditorialStepRequest(compose=request), writer))
    assert result["status"] == "artifact", result
    assert len(returned) == 1 and result["generation_trace"]["model_calls"] == 1
    receipt = result["private_generation_receipt"]["writer_candidate"]
    assert receipt["source"] == "model" and receipt["retained"] is True
    assert receipt["value"] == {key: value for key, value in returned[0].items() if not key.startswith("_")}
    artifact = result["artifact"]
    assert artifact["master_title"] == previous.master_title
    assert [row["text"] for row in artifact["blocks"]] == [row.text for row in previous.blocks]
    assert ANALYSIS not in artifact["master_content"]
    assert artifact["pending_editorial_analysis"]
    assert all(row["analysis_pending"] for row in artifact["blocks"] if row["kind"] == "opinion")
    assert artifact.get("analysis_binding_reviews", []) == []
    transform = artifact["server_transform"]
    assert transform["source_snapshot_sha256"] == request.self_review_source["snapshot_sha256"]
    assert transform["provider_generated_preserved_text"] is False
    assert transform["preserved_surfaces"][0]["body_sha256"] == hashlib.sha256(previous.master_content.encode()).hexdigest()
    assert result["artifact_signature"] == steps.artifact_signature(artifact)
    assert request.model_dump(mode="json") == original_request


def test_targets_keep_actual_new_candidate_and_absent_targets_keep_legacy_behavior(monkeypatch):
    request, contexts, candidate, _ = snapshot(monkeypatch)
    canonical = steps.content_request_payload("industry", request, contexts)["canonical_block_registry"]
    assert steps._preserved_writer_candidate(candidate, request, "weitoutiao", canonical) is candidate
    legacy, _, _, _ = snapshot(monkeypatch, targets=None)
    assert "target_surfaces" not in legacy.self_review_source
    assert steps._preserved_writer_candidate(candidate, legacy, "master", canonical) is candidate
    output = reading_output()
    assert "server_transform" not in output.model_dump(mode="json")
    steps._mark_preserved_analysis(output, legacy)
    assert steps._preservation_metadata(output, legacy) is None


@pytest.mark.parametrize("surface", ["wechat_mp", "wechat_moments", "weitoutiao"])
def test_native_adaptation_preserves_only_non_targets_and_keeps_real_calls(monkeypatch, surface):
    request, _, _, previous = snapshot(monkeypatch)
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-writer"})
    calls = []

    async def writer(_system, body, _fallback, **_kwargs):
        calls.append(deepcopy(body))
        refs = [{"block_ref": row["block_ref"]} for row in body["canonical_block_registry"]]
        if body["stage_writer_contract"]["step"] == "draft_master":
            return {"_model": "synthetic-writer", "master_title": request.approved_proposal.title,
                "blocks": refs, "platform_variants": []}
        return {"_model": "synthetic-writer", "platform_variants": [{"platform": surface,
            "title": next(row.title for row in previous.platform_variants if row.platform == surface),
            "blocks": refs if surface == "wechat_mp" else refs[:1]}]}

    async def run():
        registry = OperatorRegistry()
        master = await steps.run_editorial_step(registry, "industry", "draft_master",
            steps.EditorialStepRequest(compose=request), writer)
        assert master["status"] == "artifact", master
        return await steps.run_editorial_step(registry, "industry", "adapt_platform",
            steps.EditorialStepRequest(compose=request, surface=surface, candidate=master["artifact"],
                candidate_signature=master["artifact_signature"]), writer)

    result = asyncio.run(run())
    assert result["status"] == "artifact", result
    assert len(calls) == 2 and result["generation_trace"]["model_calls"] == 1
    output = result["artifact"]
    variant = output["platform_variants"][0]
    previous_variant = next(row for row in previous.platform_variants if row.platform == surface)
    assert variant["title"] == previous_variant.title
    if surface == "weitoutiao":
        assert ANALYSIS not in json.dumps(variant["blocks"])
        assert surface not in {row["surface"] for row in output["server_transform"]["preserved_surfaces"]}
    else:
        assert [row["text"] for row in variant["blocks"]] == [row.text for row in previous_variant.blocks]
        assert variant["body"] != previous_variant.body  # Pending prose is hidden until fresh verification.
        assert any(row["surface"] == surface for row in output["pending_editorial_analysis"])


@pytest.mark.parametrize("bad", ["unsafe_title", "discarded_analysis"])
def test_normalization_cannot_silently_change_a_preserved_surface(monkeypatch, bad):
    request, contexts, candidate, _ = snapshot(monkeypatch)
    source = request.self_review_source
    if bad == "unsafe_title":
        source["master_title"] = "https://unapproved.invalid/title"
        source["surface_text_sha256"]["master"]["title_sha256"] = hashlib.sha256(source["master_title"].encode()).hexdigest()
    else:
        source["structured_blocks"]["master"][-1]["text"] = "The system already supports 999 unsupported capabilities."
    seal(source)
    canonical = steps.content_request_payload("industry", request, contexts)["canonical_block_registry"]
    rebuilt = steps._preserved_writer_candidate(candidate, request, "master", canonical)
    scoped = request.model_copy(update={"channels": []})
    token = steps.MASTER_ONLY.set(True)
    analysis = steps.DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        fallback = steps.build_content_fallback(OperatorRegistry(), "industry", scoped, contexts)
        with pytest.raises(ValueError, match="self_review_preservation_invalid"):
            steps._normalized_candidate(OperatorRegistry(), "industry", scoped, contexts, fallback, rebuilt, None)
    finally:
        steps.DEFER_EDITORIAL_ANALYSIS.reset(analysis)
        steps.MASTER_ONLY.reset(token)


def test_signed_prior_step_cannot_reuse_old_analysis_as_fresh_verification(monkeypatch):
    request, contexts, candidate, _ = snapshot(monkeypatch)
    canonical = steps.content_request_payload("industry", request, contexts)["canonical_block_registry"]
    rebuilt = steps._preserved_writer_candidate(candidate, request, "master", canonical)
    scoped = request.model_copy(update={"channels": []})
    token = steps.MASTER_ONLY.set(True)
    analysis = steps.DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        fallback = steps.build_content_fallback(OperatorRegistry(), "industry", scoped, contexts)
        normalized = steps.normalize_content_output(OperatorRegistry(), "industry", scoped, contexts, fallback, rebuilt)
        old_signed = normalized.model_dump(mode="json")
        actual = steps._normalized_candidate(OperatorRegistry(), "industry", scoped, contexts, fallback,
            old_signed, steps.artifact_signature(old_signed))
        assert actual.pending_editorial_analysis
        assert any(row.analysis_pending for row in actual.blocks)
        assert ANALYSIS not in actual.master_content
        assert actual.analysis_binding_reviews == []
    finally:
        steps.DEFER_EDITORIAL_ANALYSIS.reset(analysis)
        steps.MASTER_ONLY.reset(token)


async def verified_current_master(monkeypatch):
    """Use the real staged verifier parser to build supported synthetic reviews."""
    request, _, _, _ = snapshot(monkeypatch)
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-stage"})
    calls = []

    async def model(_system, body, _fallback, **_kwargs):
        if "analysis_verification" in body:
            calls.append("verifier")
            return {"_model": "synthetic-current-verifier", "reviews": [], "analysis_reviews": [
                {"analysis_id": row["analysis_id"], "verdict": "supported",
                    "reason": "Synthetic current evidence-and-context support fixture."}
                for row in body["analysis_verification"]["candidates"]]}
        calls.append("writer")
        refs = [{"block_ref": row["block_ref"]} for row in body["canonical_block_registry"]]
        contract = body["stage_writer_contract"]
        if contract["step"] == "draft_master":
            return {"_model": "synthetic-writer", "master_title": request.approved_proposal.title,
                "blocks": refs, "platform_variants": []}
        surface = contract["target_surface"]
        previous = next(row for row in request.self_review_source["platform_variants"] if row["platform"] == surface)
        return {"_model": "synthetic-writer", "platform_variants": [{"platform": surface,
            "title": previous["title"], "blocks": refs if surface == "wechat_mp" else refs[:1]}]}

    registry = OperatorRegistry()
    master = await steps.run_editorial_step(registry, "industry", "draft_master",
        steps.EditorialStepRequest(compose=request), model)
    assert master["status"] == "artifact", master
    assert master["artifact"]["pending_editorial_analysis"]
    combined = deepcopy(master["artifact"])
    combined["platform_variants"] = []
    for surface in request.channels:
        adapted = await steps.run_editorial_step(registry, "industry", "adapt_platform",
            steps.EditorialStepRequest(compose=request, surface=surface, candidate=master["artifact"],
                candidate_signature=master["artifact_signature"]), model)
        assert adapted["status"] == "artifact", adapted
        assert [row["platform"] for row in adapted["artifact"]["platform_variants"]] == [surface]
        combined["platform_variants"].extend(adapted["artifact"]["platform_variants"])
    verified = await steps.run_editorial_step(registry, "industry", "verify_claims",
        steps.EditorialStepRequest(compose=request, candidate=combined), model)
    assert verified["status"] == "artifact", verified
    assert calls == ["writer", "writer", "writer", "writer", "verifier"]
    assert verified["artifact"]["analysis_binding_reviews"]
    assert all(row["verdict"] == "supported" for row in verified["artifact"]["analysis_binding_reviews"])
    assert not verified["artifact"].get("pending_editorial_analysis")
    return registry, request, verified


@pytest.mark.parametrize("old_transform", ["absent", "different_source"])
def test_valid_old_hmac_and_supported_reviews_cannot_cross_the_source_snapshot(monkeypatch, old_transform):
    async def run():
        registry, request, verified = await verified_current_master(monkeypatch)
        # Simulate an authentic historical artifact with matching prose/review
        # bindings. The difference is its absent or different source lineage.
        historical = deepcopy(verified["artifact"])
        if old_transform == "absent":
            historical.pop("server_transform")
        else:
            historical["server_transform"]["source_snapshot_sha256"] = "0" * 64
        supported = steps.OperatorContentOutput.model_validate(historical)
        for blocks in [supported.blocks, *[row.blocks for row in supported.platform_variants]]:
            for block in blocks:
                if getattr(block.kind, "value", block.kind) == "opinion":
                    block.analysis_pending = True
        assert steps.derive_pending(supported, request.claims)
        restored = steps.restore_supported(supported, request.claims, supported.analysis_binding_reviews)
        assert not restored.pending_editorial_analysis
        assert all(not block.analysis_pending for block in restored.blocks)
        signature = steps.artifact_signature(historical)
        reader_calls = []

        async def reader(*_args, **_kwargs):
            reader_calls.append(True)
            pytest.fail("Historical support must be rejected before a fresh reader call.")

        result = await steps.run_editorial_step(registry, "industry", "review_editorial",
            steps.EditorialStepRequest(compose=request, candidate=historical,
                candidate_signature=signature), reader)
        assert result["status"] == "unavailable", result
        assert result["error_code"] == "self_review_preservation_invalid"
        assert result["generation_trace"]["model_calls"] == 0
        assert reader_calls == []

    asyncio.run(run())


def test_current_fresh_verifier_support_restores_for_the_same_frozen_source_reader(monkeypatch):
    async def run():
        registry, request, verified = await verified_current_master(monkeypatch)
        original = deepcopy(verified)
        calls = []

        async def reader(_system, body, _fallback, **_kwargs):
            calls.append(deepcopy(body))
            assert ANALYSIS in body["editorial_review"]["draft"]["master_content"]
            return {**scripted_review(body), "_model": "synthetic-independent-reader"}

        result = await steps.run_editorial_step(registry, "industry", "review_editorial",
            steps.EditorialStepRequest(compose=request, candidate=verified["artifact"],
                candidate_signature=verified["artifact_signature"]), reader)
        assert result["status"] == "artifact", result
        assert len(calls) == 1 and result["generation_trace"]["model_calls"] == 1
        assert not result["artifact"].get("pending_editorial_analysis")
        assert result["artifact"]["master_content"] == request.self_review_source["master_content"]
        assert result["artifact"]["server_transform"]["source_snapshot_sha256"] == request.self_review_source["snapshot_sha256"]
        assert result["artifact"]["analysis_binding_reviews"] == verified["artifact"]["analysis_binding_reviews"]
        assert verified == original

    asyncio.run(run())


def test_blind_reader_never_receives_targets_source_diagnostics_or_transform(monkeypatch):
    request, _, _, _ = snapshot(monkeypatch)
    output = reading_output()
    output.server_transform = {"source": "server_self_review_preservation", "source_snapshot_sha256": "d" * 64}
    payload = _reader_payload(output, {"self_review_source": request.self_review_source}, scorecard_required=True)
    encoded = json.dumps(payload)
    for private in ("target_surfaces", "self_review_source", "native_diagnostics", "server_transform", "d" * 64):
        assert private not in encoded
