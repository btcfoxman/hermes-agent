"""Single-call internal editorial artifacts; orchestration owns durable loops."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Literal

from pydantic import Field

from ai_marketing_api.operator_content import (
    OperatorComposeRequest, OperatorContentOutput, build_content_fallback,
    content_request_payload, normalize_content_output, SUPPORTED_CHANNEL_FORMATS,
)
from ai_marketing_api.operator_editorial import BoundedModelRun
from ai_marketing_api.operator_grounding import apply_grounded_paraphrases, paraphrase_eligible
from ai_marketing_api.operator_review import assess_editorial, apply_editorial_assessment
from ai_marketing_api.operator_routing import BUDGET_RESERVATION_ID, SINGLE_PROVIDER_REQUEST, server_stage_route
from ai_marketing_api.operator_runtime import ContextAuthorizationError, StrictModel
from ai_marketing_api.operator_style import MASTER_ONLY, editorial_scope, style_context
from ai_marketing_api.operator_private_receipts import PRIVATE_GENERATION_CAPTURE, PrivateGenerationCapture
from ai_marketing_api.operator_analysis import (
    DEFER_EDITORIAL_ANALYSIS, derive_pending, restore_supported, render_internal_context,
)


EditorialStep = Literal["draft_master", "adapt_platform", "verify_claims", "review_editorial"]

_WRITER_JSON_PROTOCOL = (
    "\n\nInternal staged writer output protocol: Return exactly one strict JSON object. "
    "Do not wrap it in Markdown fences and do not add prose before or after the object. "
    "Inside JSON text strings, use Chinese quotation marks such as 「」 or correctly JSON-escape "
    "ASCII quotation marks and control characters. Never put unescaped ASCII double quotes "
    "inside a text string. Follow only the current stage_writer_contract output surfaces; "
    "instructions for later stages in editorial_feedback do not expand this stage's scope. "
    "All canonical evidence and disclosure restrictions remain mandatory."
)


def _writer_trace_completed(candidate):
    trace = candidate.get("generation_trace") if isinstance(candidate, dict) else None
    attempts = trace.get("attempts") if isinstance(trace, dict) else None
    return (isinstance(attempts, list) and len(attempts) == 1
        and trace.get("origin") == "model" and isinstance(attempts[0], dict)
        and attempts[0].get("stage") == "compose" and attempts[0].get("outcome") == "returned"
        and not attempts[0].get("error_code")
        and str(candidate.get("model") or "").strip().lower() not in {"", "fallback", "unavailable", "deterministic"})


def _writer_shape_present(candidate, step, surface):
    def manuscript(value):
        blocks = value.get("blocks")
        title = value.get("master_title" if step == "draft_master" else "title")
        return (isinstance(title, str) and bool(title.strip()) and isinstance(blocks, list) and bool(blocks)
            and all(isinstance(block, dict) for block in blocks)
            and any((isinstance(block.get("block_ref"), str) and block["block_ref"].strip())
                or (isinstance(block.get("kind"), str) and block["kind"] in {"opinion", "transition", "cta"}
                    and isinstance(block.get("text"), str) and block["text"].strip()) for block in blocks))
    variants = candidate.get("platform_variants", [])
    if not isinstance(variants, list):
        return False
    if step == "draft_master":
        return not variants and manuscript(candidate)
    return (len(variants) == 1 and isinstance(variants[0], dict)
        and variants[0].get("platform") == surface and manuscript(variants[0]))


def _stage_writer_contract(body, step, surface):
    contract = body["response_contract"]
    if step == "draft_master":
        contract["platform_variants"] = []
        contract["safety"][2] = "The master must include every canonical registry block marked required. platform_variants must be []."
        contract["safety"][4] = "Write only the master_title and master blocks. Keep the canonical factual title anchor; do not write any platform variant."
        scope = "Write only the master manuscript. Return master_title, blocks and platform_variants=[]. Do not draft any platform yet."
    else:
        blocks_contract = contract.pop("blocks")
        contract.pop("master_title")
        contract["platform_variants"] = [{"platform": surface, "format": SUPPORTED_CHANNEL_FORMATS[surface],
            "title": "a distinct factual-anchor title for this target only", "blocks": blocks_contract}]
        contract["safety"][4] = "Do not rewrite master_draft. Write a distinct factual-anchor title and blocks only for the exact requested platform."
        scope = "Preserve master_draft as immutable context. Return only one platform_variants entry for the exact target_surface; do not emit or rewrite master fields or any other platform."
    body["stage_writer_contract"] = {"step": step, "target_surface": "master" if step == "draft_master" else surface,
        "permitted_output_surfaces": ["master"] if step == "draft_master" else [surface], "rule": scope,
        "feedback_scope": "Keep editorial_feedback unchanged and honor its applicable writing advice. Platform advice belongs only to its own later adaptation step; it never expands this step's outputs."}


class EditorialStepRequest(StrictModel):
    compose: OperatorComposeRequest
    candidate: dict[str, Any] | None = None
    candidate_signature: str | None = Field(default=None, max_length=64)
    surface: str = Field(default="master", max_length=80)
    budget_reservation_id: str = Field(default="", max_length=160)


def artifact_signature(artifact: dict) -> str:
    body = json.dumps(artifact, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hmac.new(os.getenv("HERMES_API_KEY", "").encode(), body.encode(), hashlib.sha256).hexdigest()


def _normalized_candidate(registry, role, request, contexts, fallback, candidate, signature):
    output = normalize_content_output(registry, role, request, contexts, fallback, candidate)
    if output.critic.errors:
        return output
    # An authenticated previous step may carry independently verified public
    # expressions. Never trust incoming verified flags on an unsigned draft.
    if signature and hmac.compare_digest(artifact_signature(candidate), signature):
        signed = OperatorContentOutput.model_validate(candidate)
        old_blocks = {("master", b.binding_hash): b for b in signed.blocks}
        for variant in signed.platform_variants:
            old_blocks.update({(variant.platform, b.binding_hash): b for b in variant.blocks})
        for surface, blocks in [("master", output.blocks), *[(v.platform, v.blocks) for v in output.platform_variants]]:
            for block in blocks:
                old = old_blocks.get((surface, block.binding_hash))
                if old and old.public_text_verified and old.text == block.text:
                    # A claim is shared across surfaces; its accepted public
                    # expression is not. Restore only this surface's sentence
                    # with its actual independent support attestation.
                    supported = old.public_text and any(
                        review.claim_id == block.claim_id
                        and review.evidence_ids == block.evidence_ids
                        and review.source_sha256 == hashlib.sha256(block.text.encode("utf-8")).hexdigest()
                        and review.public_text_sha256 == hashlib.sha256(old.public_text.encode("utf-8")).hexdigest()
                        and review.verdict == "supported"
                        and review.review_model.strip().lower() not in {"", "fallback", "unavailable", "deterministic"}
                        for review in signed.claim_binding_reviews
                    )
                    if not supported:
                        raise ValueError("artifact_public_expression_invalid")
                    block.public_text, block.public_text_verified = old.public_text, True
        from ai_marketing_api.operator_content import _render
        output.master_content = _render(output.blocks, surface="master")
        for variant in output.platform_variants:
            variant.body = _render(variant.blocks, surface=variant.platform)
        output.claim_binding_reviews = signed.claim_binding_reviews
        output = restore_supported(output, request.claims, signed.analysis_binding_reviews)
        output.master_content = _render(output.blocks, surface="master")
        for variant in output.platform_variants:
            variant.body = _render(variant.blocks, surface=variant.platform)
    elif signature:
        raise ValueError("artifact_signature_invalid")
    return output


async def run_editorial_step(registry, role, step: EditorialStep, payload: EditorialStepRequest, llm,
                             *, timeout_seconds: float | None = None, wire_request_sha256: str = ""):
    if timeout_seconds is not None:
        # The provider timeout never extends the independent BoundedModelRun
        # deadline or call budget. This header is limited by the HTTP schema.
        original_llm = llm
        async def timed_llm(*args, **kwargs):
            return await original_llm(*args, timeout_seconds=str(timeout_seconds), **kwargs)
        llm = timed_llm
    request = payload.compose
    contexts = registry.authorize(role, request.authorized_context, request.as_of)
    registry.authorize_preferences(role, request.approved_preferences)
    # Validate the original disclosure, channels, brief and owner-edit scope
    # before narrowing work to one surface.
    content_request_payload(role, request, contexts)
    if request.editorial_review_mode != "standard" and step != "review_editorial":
        raise ContextAuthorizationError("editorial_review_mode_step_denied", "editorial_review_mode",
            "Value reassessment requires the independent review step.")
    if step == "adapt_platform" and payload.surface not in request.channels:
        raise ContextAuthorizationError("step_surface_denied", payload.surface, "The platform must belong to the original scope.")
    stage = {"draft_master": "compose", "adapt_platform": "compose",
             "verify_claims": "claim_verification", "review_editorial": "editorial_review"}[step]
    budget = request.runtime_budget.model_copy(update={"max_model_calls": 1, "max_surface_revisions": 0})
    run = BoundedModelRun(budget)
    review_state = "not_requested"
    single = SINGLE_PROVIDER_REQUEST.set(True)
    reservation = BUDGET_RESERVATION_ID.set(payload.budget_reservation_id or BUDGET_RESERVATION_ID.get())
    master_only = MASTER_ONLY.set(step == "draft_master")
    capture = PrivateGenerationCapture(role_id=getattr(role, "value", role), step=step,
        reservation_id=payload.budget_reservation_id, request_sha256=wire_request_sha256)
    private_token = PRIVATE_GENERATION_CAPTURE.set(capture)
    analysis_token = DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        if server_stage_route(stage) is None:
            raise ValueError("stage_route_unavailable")
        with editorial_scope(request.genre):
            if step in {"draft_master", "adapt_platform"}:
                channels = [] if step == "draft_master" else [payload.surface]
                brief = request.public_editorial_brief.model_copy(update={"channels": channels}) if request.public_editorial_brief else None
                revision = request.owner_revision
                if revision:
                    revision = revision.model_copy(update={"platform_variants": [v for v in revision.platform_variants if v.platform in channels]})
                scoped = request.model_copy(update={"channels": channels, "public_editorial_brief": brief, "owner_revision": revision})
                fallback = build_content_fallback(registry, role, scoped, contexts)
                if fallback.get("_fallback_errors"):
                    output = normalize_content_output(registry, role, scoped, contexts, fallback, fallback)
                else:
                    body = content_request_payload(role, scoped, contexts)
                    if step == "adapt_platform":
                        if not payload.candidate or not payload.candidate_signature:
                            raise ValueError("signed_master_required")
                        prior_request = request.model_copy(update={"channels": [], "public_editorial_brief": request.public_editorial_brief.model_copy(update={"channels": []}) if request.public_editorial_brief else None, "owner_revision": None})
                        token = MASTER_ONLY.set(True)
                        try:
                            prior_fallback = build_content_fallback(registry, role, prior_request, contexts)
                            prior = _normalized_candidate(registry, role, prior_request, contexts, prior_fallback, payload.candidate, payload.candidate_signature)
                        finally:
                            MASTER_ONLY.reset(token)
                        if prior.critic.errors:
                            raise ValueError("master_evidence_invalid")
                        if not _writer_trace_completed(payload.candidate):
                            raise ValueError("master_writer_not_completed")
                        body["master_draft"] = {"title": prior.master_title,
                            "body": render_internal_context(prior.blocks, "master"),
                            "analysis_state": "Editorial analysis in this internal draft is untrusted and not independently verified. It is writing context, never authorized facts.",
                            "pending_editorial_analysis": [row.model_dump(mode="json") for row in prior.pending_editorial_analysis]}
                        fallback = {**fallback, "master_title": prior.master_title, "master_content": prior.master_content,
                            "blocks": [block.model_dump(mode="json") for block in prior.blocks]}
                    _stage_writer_contract(body, step, payload.surface)
                    candidate = await run.call(llm, registry.get(role).system_prompt + _WRITER_JSON_PROTOCOL, body, fallback, stage=stage)
                    capture.writer(candidate)
                    if (len(run.attempts) != 1 or run.attempts[0].outcome != "returned" or candidate.get("_error")
                            or str(candidate.get("_model") or candidate.get("model") or "fallback").strip().lower()
                                in {"fallback", "unavailable", "deterministic"}
                            or not _writer_shape_present(candidate, step, payload.surface)):
                        code = "writer_model_not_completed"
                        trace = run.trace(blocked=not run.attempts, review="not_requested", stop_reason=code)
                        if request.fact_expression_mode == "grounded_paraphrase":
                            trace.evidence_mode = "grounded_paraphrase"
                        return capture.attach({"schema_version": "operator.editorial_step.v1", "step": step,
                            "status": "unavailable", "error_code": code, "artifact": None,
                            "publishable": False, "requires_human_review": True,
                            "generation_trace": trace.model_dump(mode="json")})
                    output = normalize_content_output(registry, role, scoped, contexts, fallback, candidate)
                    if step == "adapt_platform":
                        # This step owns only the requested variant. Its signed
                        # master remains the exact successful previous artifact.
                        output.master_title, output.master_content, output.blocks = prior.master_title, prior.master_content, prior.blocks
                    # Keep proposals as unverified metadata; rendered text remains
                    # canonical until the separately budgeted verifier accepts it.
                    raw_surfaces = {"master": candidate.get("blocks", [])}
                    raw_surfaces.update({v.get("platform"): v.get("blocks", []) for v in (candidate.get("platform_variants") or []) if isinstance(v, dict)})
                    for surface, blocks in [("master", output.blocks), *[(v.platform, v.blocks) for v in output.platform_variants]]:
                        proposed = {raw.get("block_ref"): raw.get("public_text") for raw in (raw_surfaces.get(surface) or []) if isinstance(raw, dict)}
                        for block in blocks:
                            text = proposed.get(block.binding_hash)
                            if isinstance(text, str) and paraphrase_eligible(block, text,
                                    allow_seedance_version=(getattr(role, "value", role) == "industry"
                                        and request.genre == "industry_analysis")):
                                block.public_text, block.public_text_verified = text, False
                    output.pending_editorial_analysis = derive_pending(output, request.claims)
            else:
                if not payload.candidate:
                    raise ValueError("candidate_required")
                fallback = build_content_fallback(registry, role, request, contexts)
                output = _normalized_candidate(registry, role, request, contexts, fallback, payload.candidate, payload.candidate_signature)
                if not output.critic.errors:
                    if step == "verify_claims":
                        def expression_refs(blocks):
                            return [{"block_ref": block.get("block_ref") or block.get("binding_hash"), "public_text": block.get("public_text")}
                                    for block in (blocks or []) if isinstance(block, dict)]
                        proposals = {"blocks": expression_refs(payload.candidate.get("blocks")),
                            "platform_variants": [{"platform": v.get("platform"), "blocks": expression_refs(v.get("blocks"))}
                                for v in (payload.candidate.get("platform_variants") or []) if isinstance(v, dict)]}
                        output = await apply_grounded_paraphrases(output=output, raw_candidates=[proposals],
                            run=run, llm=llm, system=registry.get(role).system_prompt, model_args=(),
                            approved_claims=request.claims)
                    else:
                        if not payload.candidate_signature:
                            raise ValueError("signed_candidate_required")
                        if output.pending_editorial_analysis or any(b.analysis_pending for b in output.blocks) or any(
                                b.analysis_pending for v in output.platform_variants for b in v.blocks):
                            raise ValueError("analysis_verification_required")
                        review, review_state = await assess_editorial(output=output,
                            brief=request.public_editorial_brief.model_dump(mode="json") if request.public_editorial_brief else {},
                            run=run, llm=llm, system=registry.get(role).system_prompt, model_args=(),
                            owner_revision=request.owner_revision.model_dump(mode="json") if request.owner_revision else None,
                            style_context=style_context(request), editorial_feedback=request.editorial_feedback,
                            scorecard_required=True, review_mode=request.editorial_review_mode)
                        output = apply_editorial_assessment(output, review,
                            review_context=run.editorial_review_context)
            trace = run.trace(blocked=bool(output.critic.errors) and not run.attempts,
                review=review_state, stop_reason=output.status)
            if request.fact_expression_mode == "grounded_paraphrase":
                trace.evidence_mode = "grounded_paraphrase"
            output.generation_trace = trace
            artifact = output.model_dump(mode="json")
            result = {"schema_version": "operator.editorial_step.v1", "step": step,
                "status": "artifact" if not output.critic.errors else "blocked",
                "artifact": artifact, "artifact_signature": artifact_signature(artifact),
                "publishable": False, "requires_human_review": True,
                "generation_trace": trace.model_dump(mode="json")}
            if request.editorial_review_mode != "standard":
                result["editorial_review_mode"] = request.editorial_review_mode
            return capture.attach(result, run.private_review_receipt if step == "review_editorial" else None)
    except ValueError as exc:
        code = str(exc) if str(exc) in {"stage_route_unavailable", "artifact_signature_invalid", "artifact_public_expression_invalid", "signed_master_required", "signed_candidate_required", "candidate_required", "master_evidence_invalid", "master_writer_not_completed", "analysis_verification_required", "analysis_verifier_not_completed"} else "step_artifact_invalid"
        result = {"schema_version": "operator.editorial_step.v1", "step": step,
            "status": "unavailable", "error_code": code, "artifact": None,
            "publishable": False, "requires_human_review": True,
            "generation_trace": run.trace(blocked=not run.attempts, review="not_requested", stop_reason=code).model_dump(mode="json")}
        return capture.attach(result, run.private_review_receipt if step == "review_editorial" else None)
    finally:
        DEFER_EDITORIAL_ANALYSIS.reset(analysis_token)
        PRIVATE_GENERATION_CAPTURE.reset(private_token)
        MASTER_ONLY.reset(master_only)
        BUDGET_RESERVATION_ID.reset(reservation)
        SINGLE_PROVIDER_REQUEST.reset(single)
