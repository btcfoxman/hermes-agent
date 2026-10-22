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
    content_request_payload, normalize_content_output,
)
from ai_marketing_api.operator_editorial import BoundedModelRun
from ai_marketing_api.operator_grounding import apply_grounded_paraphrases, paraphrase_eligible
from ai_marketing_api.operator_review import assess_editorial, apply_editorial_assessment
from ai_marketing_api.operator_routing import BUDGET_RESERVATION_ID, SINGLE_PROVIDER_REQUEST, server_stage_route
from ai_marketing_api.operator_runtime import ContextAuthorizationError, StrictModel
from ai_marketing_api.operator_style import MASTER_ONLY, editorial_scope, style_context


EditorialStep = Literal["draft_master", "adapt_platform", "verify_claims", "review_editorial"]


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
        old_blocks = {b.binding_hash: b for b in signed.blocks}
        for variant in signed.platform_variants:
            old_blocks.update({b.binding_hash: b for b in variant.blocks})
        for block in [*output.blocks, *[b for v in output.platform_variants for b in v.blocks]]:
            old = old_blocks.get(block.binding_hash)
            if old and old.public_text_verified and old.text == block.text:
                block.public_text, block.public_text_verified = old.public_text, True
        from ai_marketing_api.operator_content import _render
        output.master_content = _render(output.blocks, surface="master")
        for variant in output.platform_variants:
            variant.body = _render(variant.blocks, surface=variant.platform)
        output.claim_binding_reviews = signed.claim_binding_reviews
    elif signature:
        raise ValueError("artifact_signature_invalid")
    return output


async def run_editorial_step(registry, role, step: EditorialStep, payload: EditorialStepRequest, llm):
    request = payload.compose
    contexts = registry.authorize(role, request.authorized_context, request.as_of)
    registry.authorize_preferences(role, request.approved_preferences)
    # Validate the original disclosure, channels, brief and owner-edit scope
    # before narrowing work to one surface.
    content_request_payload(role, request, contexts)
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
                        body["master_draft"] = {"title": prior.master_title, "body": prior.master_content}
                    candidate = await run.call(llm, registry.get(role).system_prompt, body, fallback, stage=stage)
                    output = normalize_content_output(registry, role, scoped, contexts, fallback, candidate)
                    # Keep proposals as unverified metadata; rendered text remains
                    # canonical until the separately budgeted verifier accepts it.
                    raw_surfaces = {"master": candidate.get("blocks", [])}
                    raw_surfaces.update({v.get("platform"): v.get("blocks", []) for v in (candidate.get("platform_variants") or []) if isinstance(v, dict)})
                    for surface, blocks in [("master", output.blocks), *[(v.platform, v.blocks) for v in output.platform_variants]]:
                        proposed = {raw.get("block_ref"): raw.get("public_text") for raw in (raw_surfaces.get(surface) or []) if isinstance(raw, dict)}
                        for block in blocks:
                            text = proposed.get(block.binding_hash)
                            if isinstance(text, str) and paraphrase_eligible(block, text):
                                block.public_text, block.public_text_verified = text, False
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
                            run=run, llm=llm, system=registry.get(role).system_prompt, model_args=())
                    else:
                        if not payload.candidate_signature:
                            raise ValueError("signed_candidate_required")
                        review, review_state = await assess_editorial(output=output,
                            brief=request.public_editorial_brief.model_dump(mode="json") if request.public_editorial_brief else {},
                            run=run, llm=llm, system=registry.get(role).system_prompt, model_args=(),
                            owner_revision=request.owner_revision.model_dump(mode="json") if request.owner_revision else None,
                            style_context=style_context(request), editorial_feedback=request.editorial_feedback,
                            scorecard_required=True)
                        output = apply_editorial_assessment(output, review)
            trace = run.trace(blocked=bool(output.critic.errors) and not run.attempts,
                review=review_state, stop_reason=output.status)
            if request.fact_expression_mode == "grounded_paraphrase":
                trace.evidence_mode = "grounded_paraphrase"
            output.generation_trace = trace
            artifact = output.model_dump(mode="json")
            return {"schema_version": "operator.editorial_step.v1", "step": step,
                "status": "artifact" if not output.critic.errors else "blocked",
                "artifact": artifact, "artifact_signature": artifact_signature(artifact),
                "publishable": False, "requires_human_review": True,
                "generation_trace": trace.model_dump(mode="json"),
                **({"private_review_receipt": run.private_review_receipt}
                   if step == "review_editorial" and run.private_review_receipt is not None else {})}
    except ValueError as exc:
        code = str(exc) if str(exc) in {"stage_route_unavailable", "artifact_signature_invalid", "signed_master_required", "signed_candidate_required", "candidate_required", "master_evidence_invalid"} else "step_artifact_invalid"
        return {"schema_version": "operator.editorial_step.v1", "step": step,
            "status": "unavailable", "error_code": code, "artifact": None,
            "publishable": False, "requires_human_review": True,
            "generation_trace": run.trace(blocked=True, review="not_requested", stop_reason=code).model_dump(mode="json"),
            **({"private_review_receipt": run.private_review_receipt}
               if step == "review_editorial" and run.private_review_receipt is not None else {})}
    finally:
        MASTER_ONLY.reset(master_only)
        BUDGET_RESERVATION_ID.reset(reservation)
        SINGLE_PROVIDER_REQUEST.reset(single)
