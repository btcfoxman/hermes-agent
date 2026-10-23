"""Server-derived analysis bindings; an author can never attest its own prose."""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import Field

from ai_marketing_api.operator_runtime import StrictModel


DEFER_EDITORIAL_ANALYSIS: ContextVar[bool] = ContextVar("defer_editorial_analysis", default=False)
MAX_ANALYSIS_BINDINGS = 100
SOFT_ANALYSIS_WORDS = frozenset({"已经", "达到", "增长", "下降", "上涨", "下跌",
    "grew", "declined", "increased", "decreased", "reached"})
ANALYSIS_HARD_CAPABILITY_RE = re.compile(
    r"(?:已经|现已|已)[ \t]*(?:支持|具备|提供|实现|能够|可以|能)|"
    r"(?:产品|平台|模型|工具|系统|服务|接口|API)[ \t]*"
    r"(?:(?:已经|现已|已)[ \t]*)?(?:支持|具备|提供|实现)", re.I)


class PendingEditorialAnalysis(StrictModel):
    schema_version: Literal["operator.pending-editorial-analysis.v1"] = "operator.pending-editorial-analysis.v1"
    role_id: Literal["industry"] = "industry"
    genre: Literal["industry_analysis"] = "industry_analysis"
    surface: str = Field(min_length=1, max_length=80)
    block_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnalysisBindingReview(PendingEditorialAnalysis):
    schema_version: Literal["operator.analysis-binding-review.v1"] = "operator.analysis-binding-review.v1"
    verdict: Literal["supported", "unsupported", "uncertain"]
    review_model: str = Field(min_length=1, max_length=200)
    reason: str = Field(default="", max_length=1000)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def claim_pool(claims):
    return [{"claim_id": f"claim-{i + 1}", "kind": getattr(c.kind, "value", c.kind),
        "text_sha256": text_digest(c.text), "evidence_ids": list(c.evidence_ids)}
        for i, c in enumerate(claims)]


def surfaces(output):
    return {"master": (output.master_title or "", output.blocks),
        **{v.platform: (v.title, v.blocks) for v in output.platform_variants}}


def context_hash(title, blocks):
    from ai_marketing_api.operator_content import _binding_hash
    return digest({"title": title, "blocks": [{"binding_hash": _binding_hash(b.model_dump(mode="json")),
        "paragraph_id": b.paragraph_id, "public_text": b.public_text or None,
        "source_display": b.source_display or "excerpt"} for b in blocks]})


def soft_eligible(text, matches):
    return bool(matches) and all(m.group().lower() in SOFT_ANALYSIS_WORDS for m in matches) \
        and not ANALYSIS_HARD_CAPABILITY_RE.search(text)


def derive_pending(output, claims):
    """Derive from current ordered blocks, never incoming sidecar metadata."""
    if getattr(output.role_id, "value", output.role_id) != "industry" or output.genre != "industry_analysis":
        return []
    evidence = digest(claim_pool(claims))
    found = {}
    for surface, (title, blocks) in surfaces(output).items():
        context = context_hash(title, blocks)
        for block in blocks:
            if block.analysis_pending:
                record = PendingEditorialAnalysis(surface=surface, block_binding_hash=block.binding_hash,
                    text_sha256=text_digest(block.text), context_sha256=context, evidence_sha256=evidence)
                found[digest(record.model_dump(mode="json"))] = record
                if len(found) > MAX_ANALYSIS_BINDINGS:
                    # Match the downstream certificate boundary before buying
                    # an independent verification that cannot be accepted.
                    raise ValueError("analysis_verifier_not_completed")
    return list(found.values())


def pending_identity(row):
    value = row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row)
    return {k: value[k] for k in ("role_id", "genre", "surface", "block_binding_hash", "text_sha256",
        "context_sha256", "evidence_sha256")}


def restore_supported(output, claims, reviews):
    """Only the caller's authenticated artifact can supply these attestations."""
    expected = derive_pending(output, claims)
    if not expected:
        output.analysis_binding_reviews = []
        output.pending_editorial_analysis = []
        return output
    current_surfaces = set(surfaces(output))
    reviews = [row for row in reviews if row.surface in current_surfaces]
    if sorted(digest(pending_identity(row)) for row in reviews) != sorted(
            digest(pending_identity(row)) for row in expected):
        output.analysis_binding_reviews = []
        output.pending_editorial_analysis = expected
        return output
    accepted = []
    for pending in expected:
        matches = [r for r in reviews if pending_identity(r) == pending_identity(pending)]
        if (len(matches) == 1 and matches[0].verdict == "supported"
                and matches[0].review_model.strip().lower() not in {"fallback", "unavailable", "deterministic", ""}):
            accepted.append(pending_identity(pending))
    if len(accepted) != len(expected):
        output.analysis_binding_reviews = list(reviews)
        output.pending_editorial_analysis = expected
        return output
    for surface, (_, blocks) in surfaces(output).items():
        for block in blocks:
            if any(row["surface"] == surface and row["block_binding_hash"] == block.binding_hash
                   and row["text_sha256"] == text_digest(block.text) for row in accepted):
                block.analysis_pending = False
    output.analysis_binding_reviews = list(reviews)
    output.pending_editorial_analysis = derive_pending(output, claims)
    return output


def render_internal_context(blocks, surface):
    from ai_marketing_api.operator_content import _render
    return _render([b.model_copy(update={"analysis_pending": False}) for b in blocks], surface=surface)
