"""Public editorial intent and bounded, observable marketing model execution.

This edge has no knowledge store or publishing tools. Public briefs are explicit
projections supplied by orchestration, never inferred from private raw inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import re
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_serializer

from ai_marketing_api.operator_runtime import OperatorRole, StrictModel
from ai_marketing_api.operator_routing import PROVIDER_EVENT_SINK


class TopicEvidenceAnchor(StrictModel):
    opportunity_id: str = Field(min_length=1, max_length=160, strict=True)
    quote: str = Field(min_length=1, max_length=4000, strict=True)

    @field_validator("opportunity_id", "quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topic evidence anchors must not be blank")
        return value


class TopicValueContract(StrictModel):
    """An editorial hypothesis, never additional factual or publishing authority."""

    reader_problem: str = Field(min_length=1, max_length=1000, strict=True)
    thesis: str = Field(min_length=1, max_length=2000, strict=True)
    evidence_anchor: TopicEvidenceAnchor
    decision_delta: str = Field(min_length=1, max_length=2000, strict=True)
    counterargument: str = Field(min_length=1, max_length=2000, strict=True)

    @field_validator("reader_problem", "thesis", "decision_delta", "counterargument")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topic value fields must not be blank")
        return value


class PublicEditorialBrief(StrictModel):
    schema_version: Literal["operator.editorial_brief.v1"] = (
        "operator.editorial_brief.v1"
    )
    role_id: OperatorRole
    objective: str = Field(min_length=1, max_length=2000)
    audience: str = Field(min_length=1, max_length=1000)
    reader_value: str = Field(default="", max_length=2000)
    angle: str = Field(default="", max_length=2000)
    tone: str = Field(default="", max_length=1000)
    cta: str | None = Field(default=None, max_length=1000)
    channels: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=30)
    product_id: str = Field(default="", max_length=160)
    landing_url: str = Field(default="", max_length=2000)
    value_contract: TopicValueContract | None = None

    @field_validator("landing_url")
    @classmethod
    def public_link(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "landing_url must be an HTTPS public link without credentials"
            )
        return value


class RuntimeBudget(StrictModel):
    max_model_calls: int = Field(default=4, ge=1, le=8)
    max_surface_revisions: int = Field(default=2, ge=0, le=3)
    max_elapsed_seconds: float = Field(default=180, ge=1, le=600)
    max_output_tokens: int = Field(default=12000, ge=512, le=24000)


class OwnerRevisionVariant(StrictModel):
    platform: str = Field(min_length=1, max_length=80)
    title: str = Field(default="", max_length=300)
    body: str = Field(default="", max_length=20000)


class OwnerRevision(StrictModel):
    """Owner-authorized editing intent; not new factual or publishing authority."""

    instruction: str = Field(default="", max_length=4000)
    platform_variants: list[OwnerRevisionVariant] = Field(
        default_factory=list, max_length=20
    )


class GenerationAttempt(StrictModel):
    stage: Literal["planner", "propose", "revise", "compose", "repair", "claim_verification", "editorial_review"]
    surface: str = "all"
    model: str = "fallback"
    outcome: Literal["returned", "unavailable", "failed", "timeout"]
    elapsed_ms: int = 0
    prompt_sha256: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    upstream_attempted: bool = False
    error_code: str | None = None
    provider_requests: list[dict[str, Any]] = Field(default_factory=list, max_length=8)


class ReviewValidationIssue(StrictModel):
    loc: list[str | int] = Field(default_factory=list, max_length=8)
    type: str = Field(max_length=80)


class ReviewValidationDiagnostic(StrictModel):
    stage: str = Field(max_length=40)
    code: str = Field(max_length=80)
    candidate_sha256: str = Field(default="", max_length=64)
    errors: list[ReviewValidationIssue] = Field(default_factory=list, max_length=10)


class GenerationTrace(StrictModel):
    schema_version: Literal["operator.generation_trace.v1"] = (
        "operator.generation_trace.v1"
    )
    origin: Literal["model", "deterministic_fallback", "blocked_before_model"]
    model_calls: int = 0
    repair_calls: int = 0
    elapsed_ms: int = 0
    budget_exhausted: bool = False
    stop_reason: str = "completed"
    evidence_mode: Literal["exact_claim_compatibility", "grounded_paraphrase"] = (
        "exact_claim_compatibility"
    )
    research_mode: Literal["orchestration_authorized_context"] = (
        "orchestration_authorized_context"
    )
    editorial_review: Literal["not_requested", "not_completed", "passed", "failed"] = (
        "not_requested"
    )
    human_review_required: Literal[True] = True
    attempts: list[GenerationAttempt] = Field(default_factory=list, max_length=8)
    provider_requests: list[dict[str, Any]] = Field(default_factory=list, max_length=32)
    editorial_diagnostics: list[ReviewValidationDiagnostic] = Field(default_factory=list, max_length=8)


class NarrativeSupportingSpan(StrictModel):
    source: Literal["title", "body"]
    quote: str = Field(min_length=1, max_length=500, strict=True)
    role: Literal["thesis", "reason", "limit"]


class SurfaceNarrativeReport(StrictModel):
    surface: str = Field(min_length=1, max_length=80, strict=True)
    summary: str = Field(max_length=1000, strict=True)
    reader_takeaway: str = Field(max_length=1000, strict=True)
    reasoning: str = Field(max_length=2000, strict=True)
    supporting_spans: list[NarrativeSupportingSpan] = Field(max_length=12)
    coherent: bool = Field(strict=True)
    thesis_focus: bool = Field(strict=True)
    issues: list[str] = Field(default_factory=list, max_length=20)

    @property
    def passed(self) -> bool:
        return bool(self.summary.strip() and self.reader_takeaway.strip()
                    and self.reasoning.strip() and self.coherent and self.thesis_focus
                    and any(span.source == "body" and span.quote.strip()
                            for span in self.supporting_spans))


class NarrativeReport(StrictModel):
    """A reader's reconstruction of the supplied text, never factual authority."""

    schema_version: Literal["narrative_report.v1"] = "narrative_report.v1"
    content_sha256: str = Field(default="", max_length=64, strict=True)
    surfaces: list[SurfaceNarrativeReport] = Field(min_length=1, max_length=21)
    passed: bool = Field(default=False, strict=True)


class EditorialReview(StrictModel):
    """A separate editorial assessment, never authority to waive evidence gates."""

    relevant_to_brief: bool
    useful_to_reader: bool
    clear_and_specific: bool
    platform_fit: bool
    issues: list[str] = Field(default_factory=list, max_length=8)
    failed_surfaces: list[str] = Field(default_factory=list, max_length=21)
    quality_scorecard: "QualityScorecard | None" = None
    narrative_report: NarrativeReport | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_report_shape(self, handler):
        value = handler(self)
        if self.narrative_report is None:
            value.pop("narrative_report", None)
        return value

    @property
    def passed(self) -> bool:
        if not all((self.relevant_to_brief, self.useful_to_reader,
                    self.clear_and_specific, self.platform_fit)) or self.failed_surfaces:
            return False
        if self.narrative_report is not None and (
            not self.narrative_report.passed
            or not all(surface.passed for surface in self.narrative_report.surfaces)
        ):
            return False
        if self.quality_scorecard is None:
            return not self.issues
        # Scored reviews retain actionable suggestions at publication quality.
        # Recheck the rows as well as the server-computed aggregate, so a stale
        # or constructed pass flag cannot waive a low score or a major issue.
        return (self.quality_scorecard.passed and bool(self.quality_scorecard.surfaces)
                and all(surface.passed for surface in self.quality_scorecard.surfaces))


class SurfaceQuality(StrictModel):
    surface: str = Field(min_length=1, max_length=80)
    reader_relevance: int = Field(ge=1, le=5, strict=True)
    information_gain: int = Field(ge=1, le=5, strict=True)
    analytical_depth: int = Field(ge=1, le=5, strict=True)
    natural_expression: int = Field(ge=1, le=5, strict=True)
    platform_fit: int = Field(ge=1, le=5, strict=True)
    major_issues: list[str] = Field(max_length=20)
    issues: list[str] = Field(default_factory=list, max_length=20)

    @property
    def passed(self) -> bool:
        return not self.major_issues and min(self.reader_relevance, self.information_gain,
            self.analytical_depth, self.natural_expression, self.platform_fit) >= 4


class QualityScorecard(StrictModel):
    schema_version: Literal["quality_scorecard.v1"] = "quality_scorecard.v1"
    content_sha256: str = Field(default="", max_length=64)
    surfaces: list[SurfaceQuality] = Field(min_length=1, max_length=21)
    passed: bool = False


EditorialReview.model_rebuild()


class BoundedModelRun:
    def __init__(self, budget: RuntimeBudget):
        self.budget = budget
        self.started = time.monotonic()
        self.attempts: list[GenerationAttempt] = []
        self.exhausted = False
        # Private structured reviewer output is returned only by the internal
        # staged endpoint, never by legacy compose or the public trace.
        self.private_review_receipt: dict[str, Any] | None = None
        self.review_diagnostics: list[ReviewValidationDiagnostic] = []

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0, self.budget.max_elapsed_seconds - (time.monotonic() - self.started)
        )

    def can_call(self) -> bool:
        allowed = (
            len(self.attempts) < self.budget.max_model_calls
            and self.remaining_seconds > 0
        )
        if not allowed:
            self.exhausted = True
        return allowed

    async def call(
        self,
        llm: Callable[..., Awaitable[dict[str, Any]]],
        system: str,
        payload: dict[str, Any],
        fallback: dict[str, Any],
        *args: Any,
        stage: Literal[
            "planner", "propose", "revise", "compose", "repair", "claim_verification", "editorial_review"
        ] = "compose",
        surface: str = "all",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.can_call():
            return {
                **fallback,
                "_error": "runtime_budget_exhausted",
                "_model": "fallback",
            }
        started = time.monotonic()
        provider_events: list[dict[str, Any]] = []
        event_token = PROVIDER_EVENT_SINK.set(provider_events)
        try:
            result = await asyncio.wait_for(
                llm(
                    system,
                    payload,
                    fallback,
                    *args,
                    max_output_tokens=self.budget.max_output_tokens,
                    stage=stage,
                    **kwargs,
                ),
                timeout=self.remaining_seconds,
            )
        except asyncio.TimeoutError:
            self.exhausted = True
            result = {
                **fallback,
                "_error": "runtime_deadline_exceeded",
                "_llm_attempted": True,
            }
        except Exception:
            # Do not expose raw provider exceptions (which can embed endpoints
            # or credentials) in model-facing content or persisted diagnostics.
            result = {**fallback, "_error": "model_adapter_failure"}
        finally:
            PROVIDER_EVENT_SINK.reset(event_token)
        if not isinstance(result, dict):
            result = {**fallback, "_error": "model_response_not_object"}
        model = str(result.get("_model") or result.get("model") or "fallback")
        error = str(result.get("_error") or "")
        outcome = (
            "timeout"
            if error in {"runtime_deadline_exceeded", "provider_timeout"}
            else "failed"
            if error
            else "unavailable"
            if model == "fallback"
            else "returned"
        )
        usage = result.get("_usage") if isinstance(result.get("_usage"), dict) else {}

        def token_count(name: str) -> int | None:
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else None

        self.attempts.append(
            GenerationAttempt(
                stage=stage,
                surface=surface,
                model=model,
                outcome=outcome,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                prompt_sha256=hashlib.sha256(
                    json.dumps(
                        payload, ensure_ascii=False, sort_keys=True, default=str
                    ).encode()
                ).hexdigest(),
                input_tokens=token_count("prompt_tokens"),
                output_tokens=token_count("completion_tokens"),
                upstream_attempted=bool(result.get("_llm_attempted"))
                or (model != "fallback" and not error),
                error_code=(error if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", error) else "model_call_failed") if error else None,
                provider_requests=list(result.get("_provider_requests") or provider_events)[:8],
            )
        )
        return result

    def trace(self, *, blocked: bool, review: str, stop_reason: str) -> GenerationTrace:
        return GenerationTrace(
            origin="blocked_before_model"
            if blocked
            else "model"
            if any(item.outcome == "returned" for item in self.attempts)
            else "deterministic_fallback",
            model_calls=sum(item.upstream_attempted for item in self.attempts),
            repair_calls=sum(item.stage == "repair" for item in self.attempts),
            elapsed_ms=int((time.monotonic() - self.started) * 1000),
            budget_exhausted=self.exhausted,
            stop_reason=stop_reason,
            editorial_review=review,
            attempts=self.attempts,
            provider_requests=[event for item in self.attempts for event in item.provider_requests][:32],
            editorial_diagnostics=self.review_diagnostics[-8:],
        )
