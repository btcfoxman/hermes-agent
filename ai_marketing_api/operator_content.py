from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.parse import urlparse

from pydantic import Field

from ai_marketing_api.operator_runtime import (
    CONTENT_SCHEMA_VERSION,
    AuthorizedContext,
    ClaimKind,
    OperatorClaim,
    OperatorProfile,
    OperatorProposeRequest,
    OperatorQuestion,
    OperatorRegistry,
    OperatorRisk,
    OperatorRole,
    ProposalOutline,
    StrictModel,
    VerificationStatus,
)


class ContentStatus(str, Enum):
    CONTENT_READY = "content_ready"
    NEEDS_INPUT = "needs_input"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


class ContentBlockKind(str, Enum):
    FACT = "fact"
    IDENTITY = "identity"
    EXPERIENCE = "experience"
    OPINION = "opinion"
    TRANSITION = "transition"
    CTA = "cta"


class OperatorComposeRequest(OperatorProposeRequest):
    """A frozen editorial direction plus its currently authorized evidence.

    ai-orchestration remains the only knowledge source.  The runtime rechecks
    every claim against ``authorized_context`` at compose time so a previously
    approved direction cannot keep using an expired, revoked, or cross-role
    record.
    """

    approved_proposal: ProposalOutline
    claims: List[OperatorClaim] = Field(max_length=50)


class ContentBlock(StrictModel):
    block_id: str = Field(min_length=1, max_length=100)
    kind: ContentBlockKind
    text: str = Field(min_length=1, max_length=20_000)
    claim_id: Optional[str] = Field(default=None, max_length=100)
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)
    verification_status: VerificationStatus
    source_exact: bool = False


class PlatformVariant(StrictModel):
    platform: str = Field(min_length=1, max_length=80)
    format: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=100_000)
    blocks: List[ContentBlock] = Field(default_factory=list, max_length=80)


class CriticCheck(StrictModel):
    code: str = Field(min_length=1, max_length=100)
    passed: bool
    message: str = Field(min_length=1, max_length=1_000)


class ContentCritic(StrictModel):
    passed: bool
    errors: List[str] = Field(default_factory=list, max_length=50)
    warnings: List[str] = Field(default_factory=list, max_length=50)
    checks: List[CriticCheck] = Field(default_factory=list, max_length=30)


class OperatorContentOutput(StrictModel):
    schema_version: Literal[CONTENT_SCHEMA_VERSION] = CONTENT_SCHEMA_VERSION
    role_id: OperatorRole
    action: Literal["compose"] = "compose"
    status: ContentStatus
    profile_version: str = Field(min_length=1, max_length=40)
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_title: Optional[str] = Field(default=None, max_length=300)
    master_content: Optional[str] = Field(default=None, max_length=100_000)
    blocks: List[ContentBlock] = Field(default_factory=list, max_length=80)
    platform_variants: List[PlatformVariant] = Field(default_factory=list, max_length=20)
    evidence_refs: List[str] = Field(default_factory=list, max_length=100)
    questions: List[OperatorQuestion] = Field(default_factory=list, max_length=20)
    risk_flags: List[OperatorRisk] = Field(default_factory=list, max_length=50)
    critic: ContentCritic
    requires_human_review: Literal[True] = True
    model: str = Field(default="fallback", max_length=200)


SUPPORTED_CHANNEL_FORMATS: Dict[str, str] = {
    "wechat_moments": "post",
    "wechat_mp": "article",
    "wechat_channels": "video",
    "douyin": "short_video",
    "kuaishou": "short_video",
    "xiaohongshu": "graphic",
    "toutiao": "article",
    "weitoutiao": "weitoutiao",
}

_PRICE_INTENT_RE = re.compile(
    r"(?:报价|价格|售价|费用|折扣|优惠|多少钱|price|pricing|quote|discount|cost)",
    re.IGNORECASE,
)
_PRICE_VALUE_RE = re.compile(
    r"(?:[¥￥$€£]\s*\d|\d+(?:\.\d+)?\s*(?:元|万元|美元|usd|rmb|%\s*(?:off|折)))",
    re.IGNORECASE,
)
_PROMISE_RE = re.compile(
    r"(?:保证|必然|百分之百|无风险|承诺.{0,12}(?:效果|交付)|guaranteed|risk[- ]?free)",
    re.IGNORECASE,
)
_COMMERCIAL_ASSERTION_RE = re.compile(
    r"(?:我们|本公司|公司|产品|服务|团队).{0,16}(?:提供|支持|具备|拥有|实现|帮助|保证|承诺)",
    re.IGNORECASE,
)
_PERSONAL_ATTRIBUTION_RE = re.compile(
    r"(?:我|我们|本人|我司|"
    r"\b(?:i|i['’]m|i['’]ve|me|my|mine|myself|we|us|our|ours|ourselves)\b)",
    re.IGNORECASE,
)
_HTML_RE = re.compile(r"<\s*(?:script|iframe|object|embed|style|link)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d")


def _value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "json"):
        return json.loads(model.json())
    return dict(model)


def _source_text(context: AuthorizedContext) -> str:
    if context.record_type == "business_offer":
        if context.structured_data.get("publicly_quoteable") is not True:
            return ""
        return str(context.structured_data.get("public_text") or "").strip()
    return context.content.strip() or context.title.strip()


def _source_texts(context: AuthorizedContext) -> List[str]:
    if context.record_type == "business_offer":
        value = _source_text(context)
        return [value] if value else []
    values: List[str] = []
    for value in (context.content.strip(), context.title.strip()):
        if value and value not in values:
            values.append(value)
    return values


def _is_disclosable(role: OperatorRole, context: AuthorizedContext) -> bool:
    space = _value(context.space)
    if role is OperatorRole.COMMERCIAL and space == "company_internal":
        return False
    if role is OperatorRole.PERSONAL_IP and (
        space == "personal_private" or context.record_type == "boundary"
    ):
        return False
    if context.record_type == "business_offer" and not _source_text(context):
        return False
    return True


def disclosable_contexts(
    role_id: OperatorRole | str,
    request: OperatorComposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> List[AuthorizedContext]:
    """Return only approved-claim evidence that may enter publishable prose.

    Role read access is deliberately not treated as permission to publish.
    Extra records supplied by orchestration are not sent to the model and
    cannot silently expand the approved editorial direction.
    """

    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    approved_ids = {
        record_id
        for claim in request.claims
        for record_id in claim.evidence_ids
    }
    return [
        context
        for context in contexts
        if context.record_id in approved_ids and _is_disclosable(role, context)
    ]


def _channel_plan(channels: Sequence[str]) -> tuple[List[str], List[str]]:
    cleaned = [str(channel).strip().lower() for channel in channels]
    errors: List[str] = []
    if not cleaned or any(not channel for channel in cleaned):
        errors.append("channels_required")
        return [], errors
    if len(set(cleaned)) != len(cleaned):
        errors.append("duplicate_channels")
    unsupported = sorted({channel for channel in cleaned if channel not in SUPPORTED_CHANNEL_FORMATS})
    if unsupported:
        errors.append(f"unsupported_channels:{','.join(unsupported)}")
    if errors:
        return [], errors
    return cleaned, []


def _claim_kind(value: Any) -> str:
    return _value(value)


def _verification(value: Any) -> str:
    return _value(value)


def _contains_personal_attribution(value: Any) -> bool:
    return bool(_PERSONAL_ATTRIBUTION_RE.search(str(value or "").strip()))


def _build_approved_blocks(
    role: OperatorRole,
    request: OperatorComposeRequest,
    authorized_contexts: Sequence[AuthorizedContext],
) -> tuple[List[ContentBlock], List[str]]:
    all_by_id = {context.record_id: context for context in authorized_contexts}
    public_by_id = {
        context.record_id: context
        for context in authorized_contexts
        if _is_disclosable(role, context)
    }
    blocks: List[ContentBlock] = []
    errors: List[str] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    for index, claim in enumerate(request.claims, start=1):
        kind = _claim_kind(claim.kind)
        evidence_ids = [str(record_id) for record_id in claim.evidence_ids]
        claim_id = f"claim-{index}"

        if kind == ClaimKind.OPINION.value and not evidence_ids:
            if role is not OperatorRole.INDUSTRY:
                errors.append(f"unsupported_opinion:{claim_id}")
                continue
            key = (kind, claim.text, ())
            if key not in seen:
                seen.add(key)
                blocks.append(
                    ContentBlock(
                        block_id=f"block-{len(blocks) + 1}",
                        kind=ContentBlockKind.OPINION,
                        text=claim.text,
                        claim_id=claim_id,
                        evidence_ids=[],
                        verification_status=VerificationStatus.OPINION,
                        source_exact=False,
                    )
                )
            continue

        if not evidence_ids:
            errors.append(f"claim_evidence_missing:{claim_id}")
            continue
        missing = [record_id for record_id in evidence_ids if record_id not in all_by_id]
        if missing:
            errors.append(f"claim_evidence_unavailable:{claim_id}")
            continue
        undisclosable = [record_id for record_id in evidence_ids if record_id not in public_by_id]
        if undisclosable:
            errors.append(f"claim_evidence_not_publishable:{claim_id}")
            continue
        if _verification(claim.verification_status) == VerificationStatus.NEEDS_EVIDENCE.value:
            errors.append(f"claim_not_verified:{claim_id}")
            continue

        matching_ids = [
            record_id
            for record_id in evidence_ids
            if claim.text
            and any(claim.text in source for source in _source_texts(public_by_id[record_id]))
        ]
        if not matching_ids:
            errors.append(f"claim_not_verbatim:{claim_id}")
            continue
        if len(matching_ids) != len(evidence_ids):
            errors.append(f"claim_evidence_mismatch:{claim_id}")
            continue

        if role is OperatorRole.PERSONAL_IP and _contains_personal_attribution(
            claim.text
        ):
            if any(
                _value(public_by_id[record_id].space) != "personal_approved"
                or public_by_id[record_id].record_type == "boundary"
                for record_id in matching_ids
            ):
                errors.append(
                    f"personal_attribution_without_approved_card:{claim_id}"
                )
                continue

        if role is OperatorRole.PERSONAL_IP and kind in {
            ClaimKind.IDENTITY.value,
            ClaimKind.EXPERIENCE.value,
            ClaimKind.OPINION.value,
        }:
            if any(_value(public_by_id[record_id].space) != "personal_approved" for record_id in matching_ids):
                errors.append(f"personal_claim_without_approved_card:{claim_id}")
                continue

        block_kind = ContentBlockKind(kind)
        key = (kind, claim.text, tuple(matching_ids))
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            ContentBlock(
                block_id=f"block-{len(blocks) + 1}",
                kind=block_kind,
                text=claim.text,
                claim_id=claim_id,
                evidence_ids=matching_ids,
                verification_status=(
                    VerificationStatus.OPINION
                    if kind == ClaimKind.OPINION.value and role is OperatorRole.INDUSTRY
                    else VerificationStatus.VERIFIED
                ),
                source_exact=True,
            )
        )

    return blocks, errors


def _source_identity(context: AuthorizedContext) -> str:
    for key in ("publisher_id", "publisher", "source_domain"):
        identity = str(context.structured_data.get(key) or "").strip().lower()
        if identity:
            return identity.removeprefix("www.")
    uri = str(context.source_uri or "").strip()
    if not uri:
        return ""
    host = (urlparse(uri).hostname or "").lower()
    return host.removeprefix("www.")


def _industry_gate(
    blocks: Sequence[ContentBlock],
    contexts: Sequence[AuthorizedContext],
) -> tuple[bool, str]:
    by_id = {context.record_id: context for context in contexts}
    fact_blocks = [
        block
        for block in blocks
        if _value(block.kind) == ContentBlockKind.FACT.value
    ]
    if not fact_blocks:
        return False, "industry_sources_missing"

    # Independence is a property of each factual statement, not of the
    # content package as a whole.  Otherwise an unrelated official statement
    # (or a second domain reporting a different fact) could launder a
    # single-source rumour.  _build_approved_blocks already proves that every
    # evidence record below contains the exact same claim text.
    for block in fact_blocks:
        sources = [
            by_id[record_id]
            for record_id in block.evidence_ids
            if record_id in by_id
            and _value(by_id[record_id].space) == "industry"
        ]
        if not sources:
            return False, f"industry_sources_missing:{block.claim_id or block.block_id}"
        if any(
            context.source_tier in {"official", "primary"}
            for context in sources
        ):
            continue
        identities = {_source_identity(context) for context in sources}
        identities.discard("")
        if len(identities) < 2:
            return False, (
                "industry_single_non_primary_source:"
                f"{block.claim_id or block.block_id}"
            )
    return True, "every_industry_fact_has_primary_or_independent_support"


def _commercial_gate(
    request: OperatorComposeRequest,
    blocks: Sequence[ContentBlock],
    contexts: Sequence[AuthorizedContext],
) -> tuple[bool, str]:
    rendered_direction = "\n".join(
        [
            request.objective,
            request.topic,
            request.approved_proposal.title,
            request.approved_proposal.angle,
            *(claim.text for claim in request.claims),
        ]
    )
    offers = [
        context
        for context in contexts
        if context.record_type == "business_offer" and _source_text(context)
    ]
    price_sensitive = bool(_PRICE_INTENT_RE.search(rendered_direction)) or any(
        _PRICE_VALUE_RE.search(block.text) for block in blocks
    )
    if price_sensitive and len(offers) != 1:
        return False, "unique_public_business_offer_required"
    if price_sensitive:
        public_text = _source_text(offers[0])
        if not any(
            bool(block.text)
            and block.text in public_text
            and block.evidence_ids == [offers[0].record_id]
            for block in blocks
        ):
            return False, "business_offer_claim_missing"
    if any(_PROMISE_RE.search(block.text) for block in blocks):
        return False, "unsafe_business_promise"
    return True, "commercial_terms_verified"


def _personal_gate(
    blocks: Sequence[ContentBlock],
    contexts: Sequence[AuthorizedContext],
) -> tuple[bool, str]:
    by_id = {context.record_id: context for context in contexts}
    for block in blocks:
        if _value(block.kind) not in {
            ContentBlockKind.IDENTITY.value,
            ContentBlockKind.EXPERIENCE.value,
            ContentBlockKind.OPINION.value,
        } and not _contains_personal_attribution(block.text):
            continue
        if block.source_exact and bool(block.evidence_ids) and all(
            record_id in by_id
            and _value(by_id[record_id].space) == "personal_approved"
            and by_id[record_id].record_type != "boundary"
            for record_id in block.evidence_ids
        ):
            return True, "approved_personal_card_present"
    return False, "approved_personal_card_required"


def _safe_freeform(role: OperatorRole, kind: str, text: str) -> bool:
    if kind not in {
        ContentBlockKind.OPINION.value,
        ContentBlockKind.TRANSITION.value,
        ContentBlockKind.CTA.value,
    }:
        return False
    if not text or len(text) > 2_000 or _HTML_RE.search(text) or _NUMBER_RE.search(text):
        return False
    if role is OperatorRole.COMMERCIAL:
        if _PRICE_VALUE_RE.search(text) or _PROMISE_RE.search(text) or _COMMERCIAL_ASSERTION_RE.search(text):
            return False
    if role is OperatorRole.PERSONAL_IP:
        if kind == ContentBlockKind.OPINION.value or _PERSONAL_ATTRIBUTION_RE.search(text):
            return False
    return True


def _fallback_freeform(role: OperatorRole, request: OperatorComposeRequest) -> List[ContentBlock]:
    blocks: List[ContentBlock] = []
    if role is OperatorRole.COMMERCIAL:
        blocks.append(
            ContentBlock(
                block_id="transition-commercial",
                kind=ContentBlockKind.TRANSITION,
                text="沟通建议：结合具体业务场景判断适用性，并由人工确认下一步。",
                evidence_ids=[],
                verification_status=VerificationStatus.OPINION,
                source_exact=False,
            )
        )
        blocks.append(
            ContentBlock(
                block_id="cta-commercial",
                kind=ContentBlockKind.CTA,
                text="如需判断适用性，请提交具体业务场景，由人工商务沟通。",
                evidence_ids=[],
                verification_status=VerificationStatus.OPINION,
                source_exact=False,
            )
        )
    elif role is OperatorRole.INDUSTRY:
        blocks.append(
            ContentBlock(
                block_id="transition-industry",
                kind=ContentBlockKind.TRANSITION,
                text="编辑说明：以下分析严格区分来源事实与编辑观点。",
                evidence_ids=[],
                verification_status=VerificationStatus.OPINION,
                source_exact=False,
            )
        )
    return blocks


def _reindex(blocks: Sequence[ContentBlock], prefix: str = "block") -> List[ContentBlock]:
    result: List[ContentBlock] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for block in blocks:
        key = (_value(block.kind), block.text, tuple(block.evidence_ids))
        if key in seen:
            continue
        seen.add(key)
        data = _dump(block)
        data["block_id"] = f"{prefix}-{len(result) + 1}"
        result.append(ContentBlock(**data))
    return result


def _render(blocks: Sequence[ContentBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text.strip())


def _safe_master_title(
    request: OperatorComposeRequest,
    approved_blocks: Sequence[ContentBlock],
) -> str:
    requested = request.approved_proposal.title.strip()
    exact_texts = [block.text for block in approved_blocks if block.source_exact]
    if requested and any(requested in text for text in exact_texts):
        return requested[:300]
    # A proposal title is an editorial direction, not evidence.  When it is
    # not itself a verbatim approved excerpt, use the first verified excerpt
    # instead of allowing an unreviewed factual headline to bypass blocks.
    return (exact_texts[0] if exact_texts else requested)[:300]


def _platform_blocks(platform: str, blocks: Sequence[ContentBlock]) -> List[ContentBlock]:
    exact = [block for block in blocks if block.source_exact]
    opinions = [
        block
        for block in blocks
        if _value(block.kind) == ContentBlockKind.OPINION.value and not block.source_exact
    ]
    transitions = [
        block
        for block in blocks
        if _value(block.kind) == ContentBlockKind.TRANSITION.value
    ]
    ctas = [block for block in blocks if _value(block.kind) == ContentBlockKind.CTA.value]
    if platform in {
        "wechat_moments",
        "wechat_channels",
        "douyin",
        "kuaishou",
        "xiaohongshu",
        "weitoutiao",
    }:
        ordered = [*transitions, *exact, *opinions, *ctas]
    else:
        ordered = [*exact, *opinions, *transitions, *ctas]
    return _reindex(ordered, prefix=f"{platform}-block")


def _evidence_refs(blocks: Sequence[ContentBlock]) -> List[str]:
    result: List[str] = []
    for block in blocks:
        for record_id in block.evidence_ids:
            if record_id not in result:
                result.append(record_id)
                if len(result) == 100:
                    return result
    return result


def _base_checks(
    role: OperatorRole,
    channels_ok: bool,
    claims_ok: bool,
    role_gate_ok: bool,
    role_gate_message: str,
) -> List[CriticCheck]:
    return [
        CriticCheck(
            code="channels_complete",
            passed=channels_ok,
            message=(
                "Every requested channel has one normalized platform variant."
                if channels_ok
                else "Channels are missing, duplicated, or unsupported."
            ),
        ),
        CriticCheck(
            code="authorized_claims_verbatim",
            passed=claims_ok,
            message=(
                "Every factual or personal block is a verbatim excerpt from current authorized context."
                if claims_ok
                else "At least one approved claim is missing current verbatim evidence."
            ),
        ),
        CriticCheck(
            code=f"{role.value}_role_gate",
            passed=role_gate_ok,
            message=role_gate_message,
        ),
        CriticCheck(
            code="human_review_required",
            passed=True,
            message="The content can only proceed through ai-orchestration human review.",
        ),
    ]


def build_content_fallback(
    registry: OperatorRegistry,
    role_id: OperatorRole | str,
    request: OperatorComposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    """Build a usable deterministic draft without trusting model prose."""

    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    profile = registry.get(role)
    channels, channel_errors = _channel_plan(request.channels)
    approved_blocks, claim_errors = _build_approved_blocks(role, request, contexts)
    blocks = _reindex([*approved_blocks, *_fallback_freeform(role, request)])
    role_gate_ok = True
    role_gate_message = "role gate passed"
    if role is OperatorRole.COMMERCIAL:
        role_gate_ok, role_gate_message = _commercial_gate(request, approved_blocks, contexts)
    elif role is OperatorRole.INDUSTRY:
        role_gate_ok, role_gate_message = _industry_gate(approved_blocks, contexts)
    else:
        role_gate_ok, role_gate_message = _personal_gate(approved_blocks, contexts)

    errors = [*channel_errors, *claim_errors]
    if not role_gate_ok:
        errors.append(role_gate_message)
    if not approved_blocks:
        errors.append("approved_publishable_claims_required")
    errors = list(dict.fromkeys(errors))
    checks = _base_checks(
        role,
        not channel_errors,
        not claim_errors and bool(approved_blocks),
        role_gate_ok,
        role_gate_message,
    )

    if errors:
        return {
            "master_title": None,
            "blocks": [],
            "platform_variants": [],
            "_fallback_errors": errors,
            "_fallback_checks": [_dump(check) for check in checks],
            "_model": "fallback",
        }

    title = _safe_master_title(request, approved_blocks)
    variants = [
        {
            "platform": channel,
            "format": SUPPORTED_CHANNEL_FORMATS[channel],
            "title": title,
            "blocks": [_dump(block) for block in _platform_blocks(channel, blocks)],
        }
        for channel in channels
    ]
    return {
        "master_title": title,
        "blocks": [_dump(block) for block in blocks],
        "platform_variants": variants,
        "_fallback_errors": [],
        "_fallback_checks": [_dump(check) for check in checks],
        "_model": "fallback",
    }


def _safe_candidate_blocks(
    role: OperatorRole,
    raw_blocks: Any,
    approved_blocks: Sequence[ContentBlock],
    *,
    prefix: str,
    template_blocks: Sequence[ContentBlock] = (),
) -> tuple[List[ContentBlock], List[str], List[str]]:
    additions: List[ContentBlock] = []
    errors: List[str] = []
    warnings: List[str] = []
    approved = {
        (_value(block.kind), block.text, tuple(block.evidence_ids))
        for block in approved_blocks
        if block.source_exact
    }
    approved_editorial = {
        (_value(block.kind), block.text, tuple(block.evidence_ids))
        for block in approved_blocks
        if not block.source_exact
    }
    deterministic_templates = {
        (_value(block.kind), block.text, tuple(block.evidence_ids))
        for block in template_blocks
    }
    if raw_blocks is None:
        return additions, errors, warnings
    if not isinstance(raw_blocks, list):
        return additions, [f"invalid_model_blocks:{prefix}"], warnings
    for index, raw in enumerate(raw_blocks[:80], start=1):
        if not isinstance(raw, dict):
            errors.append(f"invalid_model_block:{prefix}-{index}")
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        text = str(raw.get("text") or "").strip()
        evidence_ids = [str(value) for value in raw.get("evidence_ids") or []]
        if kind in {
            ContentBlockKind.FACT.value,
            ContentBlockKind.IDENTITY.value,
            ContentBlockKind.EXPERIENCE.value,
        } or (kind == ContentBlockKind.OPINION.value and evidence_ids):
            if (kind, text, tuple(evidence_ids)) not in approved:
                errors.append(f"unsupported_model_claim:{prefix}-{index}")
            continue
        if (kind, text, tuple(evidence_ids)) in approved_editorial:
            # The approved proposal may already contain a separately labelled
            # industry opinion.  It is part of the frozen direction, not a new
            # model suggestion, and is already present in the base blocks.
            continue
        if (kind, text, tuple(evidence_ids)) in deterministic_templates:
            continue
        if not _safe_freeform(role, kind, text):
            errors.append(f"unsafe_model_freeform:{prefix}-{index}")
            continue
        # A model-selected label is not proof that prose is non-factual.  Only
        # frozen, human-approved editorial claims and deterministic templates
        # may enter the rendered draft; all new free-form prose is discarded.
        warnings.append(f"discarded_unapproved_model_freeform:{prefix}-{index}")
    return additions[:8], errors, warnings


def _blocked_status(role: OperatorRole, errors: Sequence[str]) -> ContentStatus:
    input_codes = (
        "channels_required",
        "duplicate_channels",
        "unsupported_channels",
        "unique_public_business_offer_required",
        "business_offer_claim_missing",
        "approved_personal_card_required",
        "approved_publishable_claims_required",
    )
    if role in {OperatorRole.COMMERCIAL, OperatorRole.PERSONAL_IP} and any(
        error.startswith(input_codes) for error in errors
    ):
        return ContentStatus.NEEDS_INPUT
    return ContentStatus.EVIDENCE_INSUFFICIENT


def _questions(role: OperatorRole, errors: Sequence[str]) -> List[OperatorQuestion]:
    result: List[OperatorQuestion] = []
    if any(error.startswith(("channels_required", "duplicate_channels", "unsupported_channels")) for error in errors):
        result.append(
            OperatorQuestion(
                question="请提供不重复且受支持的目标平台列表。",
                reason="平台变体必须与请求渠道一一对应，不能静默忽略或猜测。",
                blocking=True,
            )
        )
    if role is OperatorRole.COMMERCIAL and any("business_offer" in error for error in errors):
        result.append(
            OperatorQuestion(
                question="请确认本次内容唯一有效且允许公开引用的报价。",
                reason="涉及价格或折扣时，必须由当前唯一 BusinessOffer 逐字提供。",
                blocking=True,
            )
        )
    if role is OperatorRole.INDUSTRY and any("industry_" in error for error in errors):
        result.append(
            OperatorQuestion(
                question="请补充官方一手来源，或第二个独立可信来源。",
                reason="行业成稿不能建立在单一非一手来源上。",
                blocking=True,
            )
        )
    if role is OperatorRole.PERSONAL_IP and any("personal" in error for error in errors):
        result.append(
            OperatorQuestion(
                question="请先提供并批准支持本次表达的个人事实、经历或观点卡片。",
                reason="模型不能补造第一人称经历或替本人表态。",
                blocking=True,
            )
        )
    if errors and not result:
        result.append(
            OperatorQuestion(
                question="请补齐或重新批准当前成稿所引用的证据。",
                reason="已批准提案中的事实无法从当前授权上下文逐字复核。",
                blocking=True,
            )
        )
    return result


def normalize_content_output(
    registry: OperatorRegistry,
    role_id: OperatorRole | str,
    request: OperatorComposeRequest,
    contexts: Sequence[AuthorizedContext],
    fallback: Dict[str, Any],
    candidate: Dict[str, Any],
) -> OperatorContentOutput:
    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    profile: OperatorProfile = registry.get(role)
    approved_blocks, claim_errors = _build_approved_blocks(role, request, contexts)
    fallback_errors = [str(value) for value in fallback.get("_fallback_errors") or []]
    errors = list(dict.fromkeys([*fallback_errors, *claim_errors]))
    warnings: List[str] = []

    base_blocks = [ContentBlock(**raw) for raw in fallback.get("blocks") or []]
    master_additions, master_errors, master_warnings = _safe_candidate_blocks(
        role,
        candidate.get("blocks"),
        approved_blocks,
        prefix="master",
        template_blocks=base_blocks,
    )
    errors.extend(master_errors)
    warnings.extend(master_warnings)
    blocks = _reindex([*base_blocks, *master_additions])

    channels, channel_errors = _channel_plan(request.channels)
    errors.extend(channel_errors)
    raw_variants = candidate.get("platform_variants")
    if raw_variants is not None and not isinstance(raw_variants, list):
        errors.append("invalid_model_platform_variants")
        raw_variants = []
    by_platform: Dict[str, Dict[str, Any]] = {}
    for raw in (raw_variants or [])[:40]:
        if not isinstance(raw, dict):
            errors.append("invalid_model_platform_variant")
            continue
        platform = str(raw.get("platform") or "").strip().lower()
        if platform in by_platform:
            errors.append(f"duplicate_model_platform_variant:{platform}")
            continue
        by_platform[platform] = raw

    variants: List[PlatformVariant] = []
    title = _safe_master_title(request, approved_blocks)
    fallback_variants = {
        str(raw.get("platform") or "").strip().lower(): raw
        for raw in fallback.get("platform_variants") or []
        if isinstance(raw, dict)
    }
    for platform in channels:
        raw = by_platform.get(platform) or {}
        base_variant_blocks = [
            ContentBlock(**block)
            for block in (fallback_variants.get(platform) or {}).get("blocks") or []
        ]
        if not base_variant_blocks:
            base_variant_blocks = _platform_blocks(platform, base_blocks)
        additions, variant_errors, variant_warnings = _safe_candidate_blocks(
            role,
            raw.get("blocks"),
            approved_blocks,
            prefix=f"variant-{platform}",
            template_blocks=base_variant_blocks,
        )
        errors.extend(variant_errors)
        warnings.extend(variant_warnings)
        variant_blocks = _reindex(
            [*base_variant_blocks, *additions],
            prefix=f"{platform}-block",
        )
        body = _render(variant_blocks)
        if body:
            variants.append(
                PlatformVariant(
                    platform=platform,
                    format=SUPPORTED_CHANNEL_FORMATS[platform],
                    title=title,
                    body=body,
                    blocks=variant_blocks,
                )
            )

    if set(by_platform).difference(channels):
        errors.append("model_returned_unrequested_platform")
    if len(variants) != len(channels):
        errors.append("platform_variant_coverage_incomplete")
    if candidate.get("_error"):
        warnings.append("model_unavailable_safe_fallback_used")

    errors = list(dict.fromkeys(errors))[:50]
    warnings = list(dict.fromkeys(warnings))[:50]
    checks = [CriticCheck(**raw) for raw in fallback.get("_fallback_checks") or []]
    checks.append(
        CriticCheck(
            code="model_content_safe",
            passed=not master_errors and not any("model_" in error for error in errors),
            message=(
                "Model suggestions contain no new factual, pricing, promise, or personal-attribution blocks."
                if not master_errors and not any("model_" in error for error in errors)
                else "Unsafe or unsupported model prose was rejected before rendering."
            ),
        )
    )
    critic = ContentCritic(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        checks=checks,
    )
    risks = [
        OperatorRisk(code=error.split(":", 1)[0], message=error, blocking=True)
        for error in errors
    ]
    risks.extend(
        OperatorRisk(code=warning, message=warning, blocking=False)
        for warning in warnings
    )
    model = str(candidate.get("_model") or candidate.get("model") or "fallback")[:200]

    if errors:
        return OperatorContentOutput(
            role_id=role,
            status=_blocked_status(role, errors),
            profile_version=profile.version,
            system_prompt_sha256=profile.prompt_sha256,
            master_title=None,
            master_content=None,
            blocks=[],
            platform_variants=[],
            evidence_refs=[],
            questions=_questions(role, errors),
            risk_flags=risks,
            critic=critic,
            requires_human_review=True,
            model=model,
        )

    master_content = _render(blocks)
    return OperatorContentOutput(
        role_id=role,
        status=ContentStatus.CONTENT_READY,
        profile_version=profile.version,
        system_prompt_sha256=profile.prompt_sha256,
        master_title=title,
        master_content=master_content,
        blocks=blocks,
        platform_variants=variants,
        evidence_refs=_evidence_refs(blocks),
        questions=[],
        risk_flags=risks,
        critic=critic,
        requires_human_review=True,
        model=model,
    )


def _sanitized_context(context: AuthorizedContext) -> Dict[str, Any]:
    data = _dump(context)
    if context.record_type == "business_offer":
        public_text = _source_text(context)
        data["content"] = public_text
        data["structured_data"] = {
            "publicly_quoteable": True,
            "public_text": public_text,
        }
    else:
        # Structured payloads can contain implementation-only metadata.  The
        # model needs the approved excerpt and source identity, not arbitrary
        # fields that have not passed a public-disclosure contract.
        data["structured_data"] = {}
    return data


def content_request_payload(
    role_id: OperatorRole | str,
    request: OperatorComposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    visible = disclosable_contexts(role, request, contexts)
    visible_ids = {context.record_id for context in visible}
    public_claims = [
        _dump(claim)
        for claim in request.claims
        if (
            (not claim.evidence_ids and _claim_kind(claim.kind) == ClaimKind.OPINION.value)
            or (
                bool(claim.evidence_ids)
                and all(record_id in visible_ids for record_id in claim.evidence_ids)
            )
        )
    ]
    approved_blocks, _ = _build_approved_blocks(role, request, contexts)
    public_title = _safe_master_title(request, approved_blocks) if approved_blocks else ""
    public_key_points = [
        str(claim["text"])
        for claim in public_claims
        if str(claim.get("text") or "").strip()
    ][:12]

    # Do not forward the raw objective/topic/audience/constraints or the raw
    # approved proposal to a third-party model.  Those fields can contain a
    # paraphrase or short excerpt of company-internal/private context even
    # after the underlying evidence record has been removed.  The composer
    # needs only the public, currently authorized claims and channel plan.
    payload: Dict[str, Any] = {
        "objective": "Compose platform drafts from the approved public claims only.",
        "topic": public_title,
        "audience": "the approved target audience",
        "channels": list(request.channels),
        "constraints": [
            "Use only the supplied public claims verbatim for factual, pricing, identity, and experience statements.",
        ],
        "as_of": request.as_of.isoformat(),
        "approved_proposal": {
            "title": public_title,
            "angle": "Keep verified source excerpts separate from non-factual editorial framing.",
            "audience_value": "Explain the approved public direction clearly.",
            "key_points": public_key_points,
            "suggested_formats": list(request.channels),
            "cta": None,
            "first_person": role is OperatorRole.PERSONAL_IP,
        },
        "claims": public_claims,
        "authorized_context": [_sanitized_context(context) for context in visible],
    }

    forbidden_texts = [
        source
        for context in contexts
        if context.record_id not in visible_ids
        for source in _source_texts(context)
    ]

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for forbidden in forbidden_texts:
                result = result.replace(forbidden, "[REDACTED_NON_PUBLIC_CONTEXT]")
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    payload = redact(payload)
    payload["response_contract"] = {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "master_title": "string",
        "blocks": [
            {
                "kind": "fact|identity|experience|opinion|transition|cta",
                "text": "string",
                "evidence_ids": ["approved authorized record_id"],
            }
        ],
        "platform_variants": [
            {
                "platform": "one requested channel",
                "format": "server-normalized; model value is advisory only",
                "title": "string",
                "blocks": ["same block contract"],
            }
        ],
        "safety": [
            "Do not write master_content or variant body directly; the server renders normalized blocks.",
            "Facts, identity, and experience must copy an approved claim verbatim with its evidence_ids.",
            "New prose may only be an explicitly non-factual transition, CTA, or labelled editorial opinion.",
        ],
    }
    return payload
