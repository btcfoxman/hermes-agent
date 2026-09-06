"""Public editorial intent and bounded, observable marketing model execution.

This edge has no knowledge store or publishing tools. Public briefs are explicit
projections supplied by orchestration, never inferred from private raw inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from ai_marketing_api.operator_runtime import OperatorRole, StrictModel


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
    stage: Literal["compose", "repair", "claim_verification", "editorial_review"]
    surface: str = "all"
    model: str = "fallback"
    outcome: Literal["returned", "unavailable", "failed", "timeout"]
    elapsed_ms: int = 0
    prompt_sha256: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    upstream_attempted: bool = False


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


class EditorialReview(StrictModel):
    """A separate editorial assessment, never authority to waive evidence gates."""

    relevant_to_brief: bool
    useful_to_reader: bool
    clear_and_specific: bool
    platform_fit: bool
    issues: list[str] = Field(default_factory=list, max_length=8)
    failed_surfaces: list[str] = Field(default_factory=list, max_length=20)

    @property
    def passed(self) -> bool:
        return (
            all((
                self.relevant_to_brief,
                self.useful_to_reader,
                self.clear_and_specific,
                self.platform_fit,
            ))
            and not self.issues
        )


class BoundedModelRun:
    def __init__(self, budget: RuntimeBudget):
        self.budget = budget
        self.started = time.monotonic()
        self.attempts: list[GenerationAttempt] = []
        self.exhausted = False

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
            "compose", "repair", "claim_verification", "editorial_review"
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
        try:
            result = await asyncio.wait_for(
                llm(
                    system,
                    payload,
                    fallback,
                    *args,
                    max_output_tokens=self.budget.max_output_tokens,
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
        if not isinstance(result, dict):
            result = {**fallback, "_error": "model_response_not_object"}
        model = str(result.get("_model") or result.get("model") or "fallback")
        error = str(result.get("_error") or "")
        outcome = (
            "timeout"
            if error == "runtime_deadline_exceeded"
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
        )
