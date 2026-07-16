from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "operator.proposal.v1"
CONTENT_SCHEMA_VERSION = "operator.content.v1"
PROFILE_ROOT = Path(__file__).resolve().parent / "profiles"


class OperatorRole(str, Enum):
    COMMERCIAL = "commercial"
    INDUSTRY = "industry"
    PERSONAL_IP = "personal_ip"


class KnowledgeSpace(str, Enum):
    COMPANY_PUBLIC = "company_public"
    COMPANY_INTERNAL = "company_internal"
    INDUSTRY = "industry"
    PERSONAL_APPROVED = "personal_approved"
    PERSONAL_PRIVATE = "personal_private"
    CUSTOMER_SERVICE = "customer_service"


class OperatorAction(str, Enum):
    PROPOSE = "propose"
    REVISE = "revise"


class OutputStatus(str, Enum):
    PROPOSAL = "proposal"
    NEEDS_INPUT = "needs_input"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


class ClaimKind(str, Enum):
    FACT = "fact"
    OPINION = "opinion"
    IDENTITY = "identity"
    EXPERIENCE = "experience"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    OPINION = "opinion"
    NEEDS_EVIDENCE = "needs_evidence"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class AuthorizedContext(StrictModel):
    """One record authorized and materialized by ai-orchestration.

    The runtime has no database or cache client.  The explicit role grant and
    lifecycle fields make the caller's authorization decision auditable and
    allow this edge service to fail closed if a record crosses a role boundary.
    """

    record_id: str = Field(min_length=1, max_length=160)
    space: KnowledgeSpace
    record_type: str = Field(min_length=1, max_length=80)
    title: str = Field(default="", max_length=300)
    content: str = Field(default="", max_length=20_000)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["approved"] = "approved"
    authorized_roles: List[OperatorRole] = Field(min_length=1)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    source_uri: Optional[str] = Field(default=None, max_length=2_000)
    source_tier: Literal["official", "primary", "trusted", "secondary", "unknown"] = "unknown"


class OperatorProposeRequest(StrictModel):
    objective: str = Field(min_length=1, max_length=2_000)
    topic: str = Field(default="", max_length=2_000)
    audience: str = Field(default="", max_length=500)
    channels: List[str] = Field(default_factory=list, max_length=20)
    constraints: List[str] = Field(default_factory=list, max_length=30)
    authorized_context: List[AuthorizedContext] = Field(default_factory=list, max_length=100)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision_thread_id: Optional[str] = Field(default=None, max_length=160)


class ProposalOutline(StrictModel):
    title: str = Field(min_length=1, max_length=300)
    angle: str = Field(min_length=1, max_length=2_000)
    audience_value: str = Field(min_length=1, max_length=2_000)
    key_points: List[str] = Field(default_factory=list, max_length=12)
    suggested_formats: List[str] = Field(default_factory=list, max_length=12)
    cta: Optional[str] = Field(default=None, max_length=1_000)
    first_person: bool = False


class OperatorClaim(StrictModel):
    text: str = Field(min_length=1, max_length=4_000)
    kind: ClaimKind
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)
    verification_status: VerificationStatus


class OperatorQuestion(StrictModel):
    question: str = Field(min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=1_000)
    blocking: bool = True


class OperatorRisk(StrictModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1_000)
    blocking: bool = False


class OperatorOutput(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    role_id: OperatorRole
    action: OperatorAction
    status: OutputStatus
    profile_version: str = Field(min_length=1, max_length=40)
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: Optional[ProposalOutline] = None
    claims: List[OperatorClaim] = Field(default_factory=list, max_length=50)
    evidence_refs: List[str] = Field(default_factory=list, max_length=100)
    questions: List[OperatorQuestion] = Field(default_factory=list, max_length=20)
    risk_flags: List[OperatorRisk] = Field(default_factory=list, max_length=30)
    requires_human_review: bool = True
    model: str = Field(default="fallback", max_length=200)


class OperatorReviseRequest(OperatorProposeRequest):
    change_request: str = Field(min_length=1, max_length=4_000)
    previous: OperatorOutput


class ContextAuthorizationError(ValueError):
    def __init__(self, code: str, record_id: str, message: str):
        super().__init__(message)
        self.code = code
        self.record_id = record_id


@dataclass(frozen=True)
class OperatorProfile:
    role_id: OperatorRole
    profile_id: str
    version: str
    soul_bytes: bytes
    system_prompt: str
    prompt_sha256: str
    allowed_actions: frozenset[str]
    allowed_spaces: frozenset[str]
    allowed_record_types: Dict[str, frozenset[str]]
    allowed_tools: tuple[str, ...]
    denied_capabilities: tuple[str, ...]


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return json.loads(model.json())


def _load_profile(profile_dir: Path) -> OperatorProfile:
    soul_path = profile_dir / "SOUL.md"
    permission_path = profile_dir / "operator.json"
    distribution_path = profile_dir / "distribution.yaml"
    config_path = profile_dir / "config.yaml"
    missing = [
        str(path.name)
        for path in (soul_path, permission_path, distribution_path, config_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"operator profile {profile_dir.name!r} is incomplete: {', '.join(missing)}")

    soul_bytes = soul_path.read_bytes()
    try:
        system_prompt = soul_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"operator profile {profile_dir.name!r} SOUL.md must be UTF-8") from exc
    if not system_prompt.strip():
        raise RuntimeError(f"operator profile {profile_dir.name!r} has an empty SOUL.md")

    raw = json.loads(permission_path.read_text(encoding="utf-8"))
    role_id = OperatorRole(str(raw["role_id"]))
    if profile_dir.name != role_id.value:
        raise RuntimeError(
            f"operator profile directory {profile_dir.name!r} does not match role {role_id.value!r}"
        )
    allowed_record_types = {
        str(space): frozenset(str(record_type) for record_type in record_types)
        for space, record_types in dict(raw.get("allowed_record_types") or {}).items()
    }
    allowed_spaces = frozenset(str(value) for value in raw.get("allowed_knowledge_spaces") or [])
    if not allowed_spaces or set(allowed_record_types) != set(allowed_spaces):
        raise RuntimeError(
            f"operator profile {role_id.value!r} must declare record types for every allowed space"
        )
    known_spaces = {space.value for space in KnowledgeSpace}
    if not allowed_spaces.issubset(known_spaces) or any(not values for values in allowed_record_types.values()):
        raise RuntimeError(f"operator profile {role_id.value!r} has an invalid knowledge permission")
    profile_id = str(raw["profile_id"])
    version = str(raw["version"])
    allowed_actions = frozenset(str(value) for value in raw.get("allowed_actions") or [])
    denied_capabilities = tuple(str(value) for value in raw.get("denied_capabilities") or [])
    if profile_id != role_id.value or not version:
        raise RuntimeError(f"operator profile {role_id.value!r} has invalid identity metadata")
    expected_actions = {
        OperatorAction.PROPOSE.value,
        OperatorAction.REVISE.value,
        "compose",
    }
    if allowed_actions != expected_actions:
        raise RuntimeError(
            f"operator profile {role_id.value!r} must support propose, revise, and compose only"
        )
    required_denials = {
        "direct_database",
        "direct_cache",
        "filesystem_knowledge",
        "cross_profile_memory",
        "publish",
        "approve",
    }
    if not required_denials.issubset(denied_capabilities):
        raise RuntimeError(f"operator profile {role_id.value!r} is missing a required denial")
    if raw.get("allowed_tools"):
        raise RuntimeError(
            f"operator profile {role_id.value!r} must not have direct tools; knowledge is API-injected"
        )

    return OperatorProfile(
        role_id=role_id,
        profile_id=profile_id,
        version=version,
        soul_bytes=soul_bytes,
        system_prompt=system_prompt,
        prompt_sha256=hashlib.sha256(soul_bytes).hexdigest(),
        allowed_actions=allowed_actions,
        allowed_spaces=allowed_spaces,
        allowed_record_types=allowed_record_types,
        allowed_tools=tuple(str(value) for value in raw.get("allowed_tools") or []),
        denied_capabilities=denied_capabilities,
    )


class OperatorRegistry:
    """Immutable, process-lifetime registry for byte-stable role prompts."""

    def __init__(self, root: Path = PROFILE_ROOT):
        profiles = [_load_profile(root / role.value) for role in OperatorRole]
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise RuntimeError("operator profile_id values must be unique")
        if len({profile.prompt_sha256 for profile in profiles}) != len(profiles):
            raise RuntimeError("operator SOUL prompts must be distinct")
        self._profiles = {profile.role_id: profile for profile in profiles}

    def get(self, role_id: OperatorRole | str) -> OperatorProfile:
        role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
        return self._profiles[role]

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "schema_versions": {
                "proposal": SCHEMA_VERSION,
                "content": CONTENT_SCHEMA_VERSION,
            },
            "roles": {
                role.value: {
                    "ready": True,
                    "profile_id": profile.profile_id,
                    "profile_version": profile.version,
                    "system_prompt_sha256": profile.prompt_sha256,
                    "supported_actions": sorted(profile.allowed_actions),
                }
                for role, profile in sorted(self._profiles.items(), key=lambda item: item[0].value)
            },
        }

    def authorize(
        self,
        role_id: OperatorRole | str,
        contexts: Sequence[AuthorizedContext],
        as_of: datetime,
    ) -> List[AuthorizedContext]:
        profile = self.get(role_id)
        role = profile.role_id.value
        now = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        authorized: List[AuthorizedContext] = []
        seen: Set[str] = set()
        for context in contexts:
            record_id = context.record_id
            if record_id in seen:
                raise ContextAuthorizationError(
                    "duplicate_context", record_id, f"duplicate authorized context record: {record_id}"
                )
            seen.add(record_id)
            granted_roles = {_enum_value(value) for value in context.authorized_roles}
            if role not in granted_roles:
                raise ContextAuthorizationError(
                    "role_not_granted", record_id, f"record {record_id!r} is not authorized for role {role!r}"
                )
            space = _enum_value(context.space)
            if space not in profile.allowed_spaces:
                raise ContextAuthorizationError(
                    "knowledge_space_denied",
                    record_id,
                    f"role {role!r} cannot read knowledge space {space!r}",
                )
            allowed_types = profile.allowed_record_types.get(space, frozenset())
            if context.record_type not in allowed_types:
                raise ContextAuthorizationError(
                    "record_type_denied",
                    record_id,
                    f"role {role!r} cannot read record type {context.record_type!r} in {space!r}",
                )
            public_offer_text = str(context.structured_data.get("public_text") or "").strip()
            if not context.title.strip() and not context.content.strip() and not public_offer_text:
                raise ContextAuthorizationError(
                    "empty_context",
                    record_id,
                    f"record {record_id!r} has no authorized content",
                )
            valid_from = context.valid_from
            if valid_from is not None:
                valid_from = valid_from if valid_from.tzinfo else valid_from.replace(tzinfo=timezone.utc)
                if now < valid_from:
                    raise ContextAuthorizationError(
                        "record_not_active", record_id, f"record {record_id!r} is not active yet"
                    )
            valid_until = context.valid_until
            if valid_until is not None:
                valid_until = valid_until if valid_until.tzinfo else valid_until.replace(tzinfo=timezone.utc)
                if now >= valid_until:
                    raise ContextAuthorizationError(
                        "record_expired", record_id, f"record {record_id!r} has expired"
                    )
            authorized.append(context)
        return authorized


def _context_text(context: AuthorizedContext, limit: int = 480) -> str:
    value = context.content.strip() or context.title.strip()
    # Content claims are later reused by operator.content.v1. Preserve their
    # exact source bytes (apart from surrounding whitespace) so the composer
    # can prove that every factual block is a verbatim authorized excerpt.
    return value[:limit]


def _context_publisher_identity(context: AuthorizedContext) -> str:
    for key in ("publisher_id", "publisher", "source_domain"):
        identity = str(context.structured_data.get(key) or "").strip().lower()
        if identity:
            return identity.removeprefix("www.")
    return (
        str(urlsplit(context.source_uri or "").hostname or "")
        .strip()
        .lower()
        .removeprefix("www.")
    )


def _title(request: OperatorProposeRequest) -> str:
    return (request.topic.strip() or request.objective.strip())[:300]


def _formats(request: OperatorProposeRequest) -> List[str]:
    return [str(channel).strip() for channel in request.channels if str(channel).strip()][:12]


def _risk(code: str, message: str, *, blocking: bool = False) -> OperatorRisk:
    return OperatorRisk(code=code, message=message, blocking=blocking)


def _question(question: str, reason: str, *, blocking: bool = True) -> OperatorQuestion:
    return OperatorQuestion(question=question, reason=reason, blocking=blocking)


def _fact_claim(
    context: AuthorizedContext,
    *,
    kind: ClaimKind = ClaimKind.FACT,
    text: Optional[str] = None,
) -> OperatorClaim:
    return OperatorClaim(
        text=(text or _context_text(context)).strip(),
        kind=kind,
        evidence_ids=[context.record_id],
        verification_status=VerificationStatus.VERIFIED,
    )


_PRICE_INTENT_RE = re.compile(
    r"(?:报价|价格|售价|费用|折扣|优惠|多少钱|price|pricing|quote|discount|cost)", re.IGNORECASE
)
_PRICE_VALUE_RE = re.compile(
    r"(?:[¥￥$€£]\s*\d|\d+(?:\.\d+)?\s*(?:元|万元|美元|usd|rmb|%\s*(?:off|折)))",
    re.IGNORECASE,
)
_PROMISE_RE = re.compile(
    r"(?:保证|必然|百分之百|无风险|承诺.{0,12}(?:效果|交付)|guaranteed|risk[- ]?free)",
    re.IGNORECASE,
)


def _active_offers(contexts: Sequence[AuthorizedContext]) -> List[AuthorizedContext]:
    return [
        context
        for context in contexts
        if context.record_type == "business_offer"
        and context.structured_data.get("publicly_quoteable") is True
        and str(context.structured_data.get("public_text") or "").strip()
    ]


def _commercial_fallback(
    request: OperatorProposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    offers = _active_offers(contexts)
    price_requested = bool(_PRICE_INTENT_RE.search(f"{request.objective}\n{request.topic}"))
    claims = [_fact_claim(context) for context in contexts if context.record_type != "business_offer"][:8]
    for offer in offers[:1] if price_requested and len(offers) == 1 else []:
        public_text = str(offer.structured_data.get("public_text") or "").strip()
        claim = _fact_claim(offer, text=public_text[:4_000])
        claims.append(claim)
    risks: List[OperatorRisk] = []
    questions: List[OperatorQuestion] = []
    status = OutputStatus.PROPOSAL
    if price_requested and len(offers) != 1:
        status = OutputStatus.NEEDS_INPUT
        risks.append(
            _risk(
                "missing_unique_active_offer",
                "A price or discount may be quoted only when ai-orchestration "
                "supplies exactly one active BusinessOffer.",
                blocking=True,
            )
        )
        questions.append(
            _question(
                "请确认本次内容适用的产品、客户范围、渠道，以及唯一有效的公开报价或活动。",
                "当前没有唯一有效的 BusinessOffer，操盘手不能自行生成价格或折扣。",
            )
        )
    if not contexts:
        status = OutputStatus.NEEDS_INPUT
        questions.append(
            _question(
                "今天最希望客户知道哪项已批准的产品能力、案例或活动？",
                "未收到任何经授权的商务事实，不能创建可信的商务提案。",
            )
        )
    return {
        "status": status.value,
        "proposal": {
            "title": _title(request),
            "angle": "把已批准的产品能力或案例证据连接到明确的客户场景和下一步沟通。",
            "audience_value": request.audience.strip() or "帮助潜在客户判断该能力是否适合自己的业务场景。",
            "key_points": [claim.text for claim in claims[:4]],
            "suggested_formats": _formats(request),
            "cta": "邀请读者提供具体场景，进入人工商务沟通。",
            "first_person": False,
        },
        "claims": [_model_dump(claim) for claim in claims],
        "questions": [_model_dump(item) for item in questions],
        "risk_flags": [_model_dump(item) for item in risks],
    }


def _industry_fallback(
    request: OperatorProposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    industry_sources = [context for context in contexts if _enum_value(context.space) == "industry"]
    grouped_sources: Dict[str, List[AuthorizedContext]] = {}
    for context in industry_sources:
        text = _context_text(context)
        if text:
            grouped_sources.setdefault(text, []).append(context)
    claims = [
        OperatorClaim(
            text=text,
            kind=ClaimKind.FACT,
            evidence_ids=[context.record_id for context in rows[:20]],
            verification_status=VerificationStatus.VERIFIED,
        )
        for text, rows in list(grouped_sources.items())[:8]
    ]
    if industry_sources:
        claims.append(
            OperatorClaim(
                text="编辑观点：说明该变化为何重要、影响谁，以及后续应观察什么；这不是来源事实。",
                kind=ClaimKind.OPINION,
                evidence_ids=[],
                verification_status=VerificationStatus.OPINION,
            )
        )
    unsupported_claims = []
    for text, rows in grouped_sources.items():
        source_domains = {
            identity
            for context in rows
            if (identity := _context_publisher_identity(context))
        }
        has_first_party = any(
            context.source_tier in {"official", "primary"} for context in rows
        )
        if not has_first_party and len(source_domains) < 2:
            unsupported_claims.append(text)
    risks: List[OperatorRisk] = []
    questions: List[OperatorQuestion] = []
    status = OutputStatus.PROPOSAL
    if not industry_sources:
        status = OutputStatus.EVIDENCE_INSUFFICIENT
        risks.append(_risk("missing_industry_evidence", "No authorized industry source was supplied.", blocking=True))
        questions.append(
            _question(
                "请提供至少一个官方一手来源，或两个相互独立的可信来源。",
                "行业事实必须可回溯，当前没有可用于提案的行业证据。",
            )
        )
    elif unsupported_claims:
        status = OutputStatus.EVIDENCE_INSUFFICIENT
        risks.append(
            _risk(
                "single_source_unverified",
                "At least one fact has only one non-primary publisher and must remain a candidate until corroborated.",
                blocking=True,
            )
        )
        questions.append(
            _question(
                "能否补充官方一手来源或第二个独立来源？",
                "当前为非一手单源，不能把其陈述直接推进为终稿事实。",
            )
        )
    return {
        "status": status.value,
        "proposal": {
            "title": _title(request),
            "angle": "区分已核验事实与编辑观点，解释变化的业务影响及后续观察项。",
            "audience_value": request.audience.strip() or "帮助读者理解事件影响，而不是重复新闻摘要。",
            "key_points": [claim.text for claim in claims[:4]],
            "suggested_formats": _formats(request),
            "cta": None,
            "first_person": False,
        } if industry_sources else None,
        "claims": [_model_dump(claim) for claim in claims],
        "questions": [_model_dump(item) for item in questions],
        "risk_flags": [_model_dump(item) for item in risks],
    }


_PERSONAL_TYPES = {
    "identity_fact": ClaimKind.IDENTITY,
    "experience": ClaimKind.EXPERIENCE,
    "opinion": ClaimKind.OPINION,
    "story": ClaimKind.EXPERIENCE,
    "signature_expression": ClaimKind.OPINION,
    "boundary": ClaimKind.IDENTITY,
}


def _personal_fallback(
    request: OperatorProposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    cards = [
        context
        for context in contexts
        if _enum_value(context.space) == "personal_approved"
        and context.record_type in _PERSONAL_TYPES
        and context.record_type != "boundary"
    ]
    if not cards:
        return {
            "status": OutputStatus.NEEDS_INPUT.value,
            "proposal": None,
            "claims": [],
            "questions": [
                _model_dump(
                    _question(
                        "这件事发生在什么时间和场景，当时你具体做了什么？",
                        "缺少已批准的个人经历卡，不能生成第一人称经历。",
                    )
                ),
                _model_dump(
                    _question(
                        "你当时最真实的矛盾、判断或意外是什么？",
                        "需要由本人提供叙事冲突和观点，模型不能代替本人推断。",
                    )
                ),
                _model_dump(
                    _question(
                        "哪些细节可以公开，哪些人物、关系或数据必须隐藏？",
                        "个人内容需要先确认公开边界，再创建身份卡。",
                    )
                ),
            ],
            "risk_flags": [
                _model_dump(
                    _risk(
                        "missing_approved_identity_cards",
                        "No approved PersonalIdentityCard supports a first-person proposal.",
                        blocking=True,
                    )
                )
            ],
        }
    claims = [_fact_claim(card, kind=_PERSONAL_TYPES[card.record_type]) for card in cards[:8]]
    return {
        "status": OutputStatus.PROPOSAL.value,
        "proposal": {
            "title": _title(request),
            "angle": "只使用已批准个人卡片中的经历或观点，围绕真实矛盾、判断和变化组织叙事。",
            "audience_value": request.audience.strip() or "以真实经历提供可验证、可共鸣的个人判断。",
            "key_points": [claim.text for claim in claims[:4]],
            "suggested_formats": _formats(request),
            "cta": None,
            "first_person": True,
        },
        "claims": [_model_dump(claim) for claim in claims],
        "questions": [],
        "risk_flags": [],
    }


def build_fallback(
    role_id: OperatorRole | str,
    action: OperatorAction,
    request: OperatorProposeRequest,
    contexts: Sequence[AuthorizedContext],
) -> Dict[str, Any]:
    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    if role is OperatorRole.COMMERCIAL:
        payload = _commercial_fallback(request, contexts)
    elif role is OperatorRole.INDUSTRY:
        payload = _industry_fallback(request, contexts)
    else:
        payload = _personal_fallback(request, contexts)
    if action is OperatorAction.REVISE and isinstance(request, OperatorReviseRequest):
        proposal = payload.get("proposal")
        if isinstance(proposal, dict):
            proposal["angle"] = f"{proposal['angle']} 修订要求：{request.change_request.strip()}"[:2_000]
    return payload


def _safe_model_proposal(
    role: OperatorRole,
    fallback: Optional[Dict[str, Any]],
    candidate: Any,
    contexts: Sequence[AuthorizedContext],
) -> Optional[Dict[str, Any]]:
    if fallback is None or not isinstance(candidate, dict):
        return fallback
    merged = dict(fallback)
    for key in ("title", "angle", "audience_value", "cta"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    for key in ("key_points", "suggested_formats"):
        value = candidate.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                merged[key] = cleaned[:12]
    if isinstance(candidate.get("first_person"), bool):
        merged["first_person"] = candidate["first_person"]

    rendered = json.dumps(merged, ensure_ascii=False)
    if role is OperatorRole.COMMERCIAL:
        if _PROMISE_RE.search(rendered):
            return fallback
        if _PRICE_VALUE_RE.search(rendered):
            # Pricing text is emitted only by the deterministic BusinessOffer
            # claim path, never through free-form model prose.
            return fallback
    if role is OperatorRole.PERSONAL_IP:
        cards = [context for context in contexts if _enum_value(context.space) == "personal_approved"]
        if not cards:
            return None
        # Semantic equivalence is not a sufficient authorization check for an
        # identity claim. Keep all personal prose deterministic and card-backed;
        # the model may help plan internally, but cannot add words attributed to
        # the person at this boundary.
        return fallback
    return merged


def _normalize_model_claims(
    role: OperatorRole,
    candidate: Any,
    fallback_claims: List[Dict[str, Any]],
    contexts: Sequence[AuthorizedContext],
) -> tuple[List[OperatorClaim], List[OperatorRisk]]:
    context_by_id = {context.record_id: context for context in contexts}
    risks: List[OperatorRisk] = []
    if role in {OperatorRole.COMMERCIAL, OperatorRole.PERSONAL_IP}:
        # Business offers and personal identity are high-integrity claims.  Use
        # only exact text copied from the authorized records.
        return [OperatorClaim(**claim) for claim in fallback_claims], risks
    if not isinstance(candidate, list):
        return [OperatorClaim(**claim) for claim in fallback_claims], risks

    fallback_objects = [OperatorClaim(**claim) for claim in fallback_claims]
    fallback_facts = [claim for claim in fallback_objects if claim.kind == ClaimKind.FACT.value]
    fallback_opinions = [
        claim for claim in fallback_objects if claim.kind == ClaimKind.OPINION.value
    ]
    claims: List[OperatorClaim] = list(fallback_facts)
    for raw in candidate[:50]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("kind") or "fact")
        if not text or kind not in {ClaimKind.FACT.value, ClaimKind.OPINION.value}:
            continue
        evidence_ids = [
            str(value) for value in raw.get("evidence_ids") or [] if str(value) in context_by_id
        ][:20]
        if kind == ClaimKind.OPINION.value:
            # A model-selected "opinion" label cannot prove that a sentence is
            # non-factual.  Keep only the deterministic editorial direction
            # already present in the frozen fallback contract.
            if any(
                text == fallback_opinion.text and not evidence_ids
                for fallback_opinion in fallback_opinions
            ):
                claims.append(
                    OperatorClaim(
                        text=text,
                        kind=ClaimKind.OPINION,
                        evidence_ids=[],
                        verification_status=VerificationStatus.OPINION,
                    )
                )
            else:
                risks.append(
                    _risk(
                        "discarded_unapproved_model_opinion",
                        "A model-labelled opinion was not part of the deterministic approved editorial contract and was discarded.",
                    )
                )
            continue
        exact_source_text = any(
            text == _context_text(context_by_id[record_id]) for record_id in evidence_ids
        )
        if not evidence_ids or not exact_source_text:
            risks.append(
                _risk(
                    "unsupported_fact",
                    "A model-proposed industry fact was not an exact authorized source statement and was discarded.",
                    blocking=True,
                )
            )
    if not any(claim.kind == ClaimKind.OPINION.value for claim in claims):
        claims.extend(
            claim for claim in fallback_objects if claim.kind == ClaimKind.OPINION.value
        )
    return claims or fallback_objects, risks


def normalize_output(
    registry: OperatorRegistry,
    role_id: OperatorRole | str,
    action: OperatorAction,
    request: OperatorProposeRequest,
    contexts: Sequence[AuthorizedContext],
    fallback: Dict[str, Any],
    candidate: Dict[str, Any],
) -> OperatorOutput:
    role = role_id if isinstance(role_id, OperatorRole) else OperatorRole(role_id)
    profile = registry.get(role)
    fallback_claims = list(fallback.get("claims") or [])
    claims, claim_risks = _normalize_model_claims(
        role, candidate.get("claims"), fallback_claims, contexts
    )
    proposal = _safe_model_proposal(role, fallback.get("proposal"), candidate.get("proposal"), contexts)

    questions = [OperatorQuestion(**item) for item in fallback.get("questions") or []]
    risks = [OperatorRisk(**item) for item in fallback.get("risk_flags") or []]
    risks.extend(claim_risks)
    risk_codes = {risk.code for risk in risks}
    if role is OperatorRole.INDUSTRY:
        facts = [claim for claim in claims if claim.kind == ClaimKind.FACT.value]
        opinions = [claim for claim in claims if claim.kind == ClaimKind.OPINION.value]
        if facts and not opinions:
            risks.append(
                _risk(
                    "missing_fact_opinion_separation",
                    "Industry output must contain a separately labelled editorial opinion.",
                )
            )
        if any(claim.verification_status == VerificationStatus.NEEDS_EVIDENCE.value for claim in facts):
            if "unsupported_fact" not in risk_codes:
                risks.append(_risk("unsupported_fact", "Industry facts require authorized evidence.", blocking=True))

    evidence_refs: List[str] = []
    valid_ids = {context.record_id for context in contexts}
    for claim in claims:
        for record_id in claim.evidence_ids:
            if record_id in valid_ids and record_id not in evidence_refs:
                evidence_refs.append(record_id)

    status = str(fallback.get("status") or OutputStatus.PROPOSAL.value)
    if any(risk.blocking for risk in risks) and status == OutputStatus.PROPOSAL.value:
        status = OutputStatus.EVIDENCE_INSUFFICIENT.value
    model = str(candidate.get("_model") or candidate.get("model") or "fallback")[:200]
    return OperatorOutput(
        role_id=role,
        action=action,
        status=status,
        profile_version=profile.version,
        system_prompt_sha256=profile.prompt_sha256,
        proposal=ProposalOutline(**proposal) if proposal else None,
        claims=claims,
        evidence_refs=evidence_refs,
        questions=questions,
        risk_flags=risks,
        requires_human_review=True,
        model=model,
    )


def request_payload(request: OperatorProposeRequest) -> Dict[str, Any]:
    """Serialize only user/context data; the SOUL remains a stable system message."""

    payload = _model_dump(request)
    payload["response_contract"] = {
        "schema_version": SCHEMA_VERSION,
        "proposal": {
            "title": "string",
            "angle": "string",
            "audience_value": "string",
            "key_points": ["string"],
            "suggested_formats": ["string"],
            "cta": "string|null",
            "first_person": "boolean",
        },
        "claims": [
            {
                "text": "string",
                "kind": "fact|opinion|identity|experience",
                "evidence_ids": ["authorized record_id"],
            }
        ],
    }
    return payload
