from __future__ import annotations

import hashlib
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
    QUALITY_INSUFFICIENT = "quality_insufficient"


class ContentBlockKind(str, Enum):
    FACT = "fact"
    IDENTITY = "identity"
    EXPERIENCE = "experience"
    OPINION = "opinion"
    TRANSITION = "transition"
    CTA = "cta"


class ContentBlockOrigin(str, Enum):
    LEGACY = "legacy"
    APPROVED_CLAIM = "approved_claim"
    SERVER_TEMPLATE = "server_template"
    MODEL_EDITORIAL = "model_editorial"


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
    origin: ContentBlockOrigin = ContentBlockOrigin.LEGACY
    locked: bool = False
    required: bool = False
    binding_hash: str = Field(default="", max_length=64)


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
_PRICE_INTENT_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[；;。.!！？?\n]+|但(?:是)?|不过|然而|\bbut\b|\bhowever\b)",
    re.IGNORECASE,
)
_NEGATED_PRICE_INTENT_RE = re.compile(
    r"(?:不(?:得|涉及|包含|含|添加|需要|要|提及|写|发布|使用|讨论)?|"
    r"无|无需|禁止|排除|避免|"
    r"\b(?:no|not|without|exclude|omit|avoid|do\s+not|don't)\b)",
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
_PERSONAL_EDITORIAL_ASSERTION_RE = re.compile(
    r"(?:"
    r"(?:我|我们)(?:曾经?|已经|正在|刚刚|亲自|做过|参与|负责|管理|开发|创办|"
    r"服务|拥有|有着|发现|遇到|经历|见过|把|花了|用了|实现|获得|赚|亏)|"
    r"我的(?:公司|团队|客户|产品|项目|业务|经历|收入|员工)|"
    r"\b(?:i|we)\s+(?:have|had|did|built|made|led|managed|served|owned|"
    r"founded|earned|lost|experienced|discovered)\b|"
    r"\b(?:my|our)\s+(?:company|team|client|customer|product|project|"
    r"business|revenue|employee|experience)\b"
    r")",
    re.IGNORECASE,
)
_HTML_RE = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|年|月|日|天|小时|分钟|万|亿)?"
)
_EDITORIAL_TRANSITION_RE = re.compile(
    r"(?:"
    r"先看|再看|接下来|下面|以下|回到|从.+(?:看|出发)|基于|围绕|"
    r"真正值得关注|值得关注|不只是|更重要|"
    r"事实|证据|背景|重点|问题|分析|影响|观察|讨论|视角|"
    r"可能|或许|未必|一旦|如果|信号|边界|规则|成本|选择|机制|合作|"
    r"\b(?:first|next|then|below|context|evidence|fact|analysis|"
    r"impact|question|perspective|worth watching)\b"
    r")",
    re.IGNORECASE,
)
_EDITORIAL_OPINION_RE = re.compile(
    r"(?:"
    r"编辑观点|编辑判断|值得关注|值得思考|需要思考|可以追问|"
    r"更重要|关键在于|这意味着|启示|影响|观察|判断|"
    r"对.+而言|不只是|而是|如何|为什么|"
    r"可能|或许|未必|一旦|如果|信号|边界|规则|成本|选择|"
    r"议价|机制|合作|信心|预期|风险|"
    r"\b(?:editorial|opinion|worth watching|worth asking|"
    r"what matters|why it matters|implication|perspective)\b"
    r")",
    re.IGNORECASE,
)
_EDITORIAL_CTA_RE = re.compile(
    r"(?:"
    r"欢迎|请|可以|不妨|留言|评论|交流|讨论|分享|关注|咨询|"
    r"提供|告诉|联系|查看|阅读|你更|你会|你怎么看|"
    r"\b(?:please|welcome|share|comment|discuss|contact|tell|"
    r"read|follow|learn more|what do you think)\b|[？?]"
    r")",
    re.IGNORECASE,
)
_UNSUPPORTED_EDITORIAL_ASSERTION_RE = re.compile(
    r"(?:"
    r"已经|宣布|发布|推出|收购|融资|签约|达成|增长|下降|"
    r"上涨|下跌|达到|获得|发生|证实|曝光|秘密|据悉|消息称|"
    r"全球最|行业最|最领先|排名第一|唯一一家|"
    r"\b(?:announced|launched|acquired|raised|signed|reached|"
    r"grew|declined|increased|decreased|confirmed|reportedly|"
    r"secretly|world['’]s largest|market leader)\b"
    r")",
    re.IGNORECASE,
)
_GENERIC_EDITORIAL_META_RE = re.compile(
    r"(?:"
    r"先把.{0,16}事实.{0,16}(?:判断|观点).{0,8}分开|"
    r"以下.{0,12}(?:分析|事实|观点)|"
    r"(?:编辑|审稿)(?:说明|观点|判断|分析)[：:]?|"
    r"(?:事实|判断|观点)(?:部分|层面|如下)?[：:]|"
    r"公开信息只是起点|欢迎围绕公开证据|"
    r"对.{0,24}(?:的人|读者)来说[，,]?(?:重点|价值).{0,20}(?:不只是|在于)|"
    r"从(?:行业|商业|平台|用户).{0,8}(?:视角|角度)看[，,]?|"
    r"这类(?:事件|案例|消息).{0,12}(?:价值|重点).{0,12}(?:在于|是)|"
    r"真正值得关注的[，,]?\s*不只是事件本身|"
    r"(?:接下来|后续)可以继续观察|"
    r"后续措施如何落地.{0,24}(?:透明|可预期|到位)"
    r")",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_TITLE_ENTITY_RE = re.compile(
    r"(?:[\u4e00-\u9fffA-Za-z0-9·.\-]{2,28}"
    r"(?:集团|公司|平台|总局|委员会|银行|大学|研究院|实验室))"
)
_TITLE_ACTION_RE = re.compile(
    r"(?:处罚|整改|发布|推出|升级|收购|融资|签约|调查|通报|上线|"
    r"penalty|launch|release|acquisition|funding|investigation)",
    re.IGNORECASE,
)
_EDITORIAL_KIND_ALIASES = {
    "hook": ContentBlockKind.TRANSITION.value,
    "lead": ContentBlockKind.TRANSITION.value,
    "intro": ContentBlockKind.TRANSITION.value,
    "opening": ContentBlockKind.TRANSITION.value,
    "framing": ContentBlockKind.TRANSITION.value,
    "analysis": ContentBlockKind.OPINION.value,
    "editorial": ContentBlockKind.OPINION.value,
    "editorial_analysis": ContentBlockKind.OPINION.value,
    "impact": ContentBlockKind.OPINION.value,
    "implication": ContentBlockKind.OPINION.value,
    "takeaway": ContentBlockKind.OPINION.value,
    "reader_takeaway": ContentBlockKind.OPINION.value,
    "perspective": ContentBlockKind.OPINION.value,
    "reader_value": ContentBlockKind.OPINION.value,
    "significance": ContentBlockKind.OPINION.value,
    "why_it_matters": ContentBlockKind.OPINION.value,
    "call_to_action": ContentBlockKind.CTA.value,
    "closing": ContentBlockKind.CTA.value,
    "question": ContentBlockKind.CTA.value,
    "reader_question": ContentBlockKind.CTA.value,
    "prompt": ContentBlockKind.CTA.value,
}


def _value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "json"):
        return json.loads(model.json())
    return dict(model)


def _binding_hash(data: Dict[str, Any]) -> str:
    payload = {
        "kind": _value(data.get("kind")),
        "text": str(data.get("text") or ""),
        "claim_id": data.get("claim_id") or None,
        "evidence_ids": [str(value) for value in data.get("evidence_ids") or []],
        "origin": _value(data.get("origin") or ContentBlockOrigin.LEGACY),
        "locked": bool(data.get("locked")),
        "required": bool(data.get("required")),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_tokens(value: str) -> set[str]:
    return {match.group(0) for match in _NUMBER_RE.finditer(value)}


def _fact_display_text(block: ContentBlock) -> str:
    """Project a long immutable fact into source-exact, readable fact beats.

    The underlying block text and evidence binding remain byte-for-byte
    unchanged.  The public projection may omit legal boilerplate and add only
    punctuation/bullet markers; every displayed phrase remains a contiguous
    excerpt of the approved source.  Terminal review reproduces this function
    independently, so presentation never weakens evidence binding.
    """

    text = block.text.strip()
    if (
        _value(block.kind) != ContentBlockKind.FACT.value
        or not block.source_exact
        or len(text) < 96
    ):
        return text
    raw_segments = [
        part.strip()
        for part in re.split(r"[，,；;。！？!?]+", text)
        if part.strip()
    ]
    segments: List[str] = []
    for part in raw_segments:
        part = re.sub(r"^(?:同时|此外|其中|并且)(?:还)?", "", part).strip()
        if not part:
            continue
        if (
            len(raw_segments) >= 3
            and re.match(r"^(?:依据|根据).{0,100}(?:规定|法律|办法|条例)$", part)
        ):
            continue
        if (
            segments
            and re.match(r"^(?:并处|罚没)", part)
            and re.search(r"(?:没收|罚款|罚没)", segments[-1])
        ):
            segments[-1] = f"{segments[-1]}；{part}"
            continue
        segments.append(part)
    if len(segments) < 2:
        return text
    return "\n".join(f"- {segment}" for segment in segments)


def _stamp_block(
    block: ContentBlock,
    *,
    origin: ContentBlockOrigin,
    locked: bool,
    required: bool,
) -> ContentBlock:
    data = _dump(block)
    data.update(
        {
            "origin": origin.value,
            "locked": locked,
            "required": required,
        }
    )
    data["binding_hash"] = _binding_hash(data)
    return ContentBlock(**data)


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
                    _stamp_block(
                        ContentBlock(
                            block_id=f"block-{len(blocks) + 1}",
                            kind=ContentBlockKind.OPINION,
                            text=claim.text,
                            claim_id=claim_id,
                            evidence_ids=[],
                            verification_status=VerificationStatus.OPINION,
                            source_exact=False,
                        ),
                        origin=ContentBlockOrigin.APPROVED_CLAIM,
                        locked=True,
                        required=True,
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
            _stamp_block(
                ContentBlock(
                    block_id=f"block-{len(blocks) + 1}",
                    kind=block_kind,
                    text=claim.text,
                    claim_id=claim_id,
                    evidence_ids=matching_ids,
                    verification_status=(
                        VerificationStatus.OPINION
                        if kind == ClaimKind.OPINION.value
                        and role is OperatorRole.INDUSTRY
                        else VerificationStatus.VERIFIED
                    ),
                    source_exact=True,
                ),
                origin=ContentBlockOrigin.APPROVED_CLAIM,
                locked=True,
                required=True,
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
    price_requested = any(
        _PRICE_INTENT_RE.search(clause)
        and not _NEGATED_PRICE_INTENT_RE.search(clause)
        for clause in _PRICE_INTENT_CLAUSE_SPLIT_RE.split(rendered_direction)
    )
    price_sensitive = price_requested or any(
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


def _fallback_freeform(role: OperatorRole, request: OperatorComposeRequest) -> List[ContentBlock]:
    blocks: List[ContentBlock] = []
    if role is OperatorRole.COMMERCIAL:
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="transition-commercial",
                    kind=ContentBlockKind.TRANSITION,
                    text="沟通建议：结合具体业务场景判断适用性，并由人工确认下一步。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
            )
        )
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="cta-commercial",
                    kind=ContentBlockKind.CTA,
                    text="如需判断适用性，请提交具体业务场景，由人工商务沟通。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
            )
        )
    elif role is OperatorRole.INDUSTRY:
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="transition-industry",
                    kind=ContentBlockKind.TRANSITION,
                    text="一个事件是否重要，不能只看热度，还要看它改变了谁的规则、成本与选择。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
            )
        )
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="opinion-industry",
                    kind=ContentBlockKind.OPINION,
                    text="行业判断不能停在结论上：规则是否改变、执行是否持续、相关参与者是否真实感受到变化，才是后续验证重点。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
            )
        )
    elif role is OperatorRole.PERSONAL_IP:
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="transition-personal-ip",
                    kind=ContentBlockKind.TRANSITION,
                    text="先看已确认的经历与观点，再讨论其中的启发。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
            )
        )
        blocks.append(
            _stamp_block(
                ContentBlock(
                    block_id="cta-personal-ip",
                    kind=ContentBlockKind.CTA,
                    text="欢迎分享你的观察与不同视角。",
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.SERVER_TEMPLATE,
                locked=True,
                required=False,
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
        if not data.get("binding_hash"):
            data["binding_hash"] = _binding_hash(data)
        result.append(ContentBlock(**data))
    return result


def _render(blocks: Sequence[ContentBlock]) -> str:
    return "\n\n".join(
        _fact_display_text(block) if block.source_exact else block.text.strip()
        for block in blocks
        if block.text.strip()
    )


def _safe_master_title(
    request: OperatorComposeRequest,
    approved_blocks: Sequence[ContentBlock],
    candidate_title: Any = None,
) -> str:
    requested = request.approved_proposal.title.strip()
    exact_texts = [block.text for block in approved_blocks if block.source_exact]
    supplied = str(candidate_title or "").strip()
    approved_numbers = {
        token for text in exact_texts for token in _number_tokens(text)
    }
    supplied_numbers = _number_tokens(supplied)
    supplied_entities = [
        match.group(0) for match in _TITLE_ENTITY_RE.finditer(supplied)
    ]
    source_text = "\n".join(exact_texts)
    if (
        8 <= len(supplied) <= 72
        and not _HTML_RE.search(supplied)
        and not _URL_RE.search(supplied)
        and not _PROMISE_RE.search(supplied)
        and not _GENERIC_EDITORIAL_META_RE.search(supplied)
        and supplied_numbers.issubset(approved_numbers)
        and all(entity in source_text for entity in supplied_entities)
        and bool(supplied_numbers or supplied_entities)
        and not _UNSUPPORTED_EDITORIAL_ASSERTION_RE.search(supplied)
    ):
        return supplied
    if (
        requested
        and len(requested) <= 72
        and any(requested in text for text in exact_texts)
    ):
        return requested
    # A proposal title is an editorial direction, not evidence.  When it is
    # not itself a verbatim approved excerpt, extract a concise contiguous
    # span from the first verified claim instead of turning a whole source
    # paragraph into the headline.
    if not exact_texts:
        return requested[:300]
    source = exact_texts[0].strip()
    if len(source) <= 80:
        return source[:300]
    spans = [
        match.span()
        for match in re.finditer(r"[^，；。！？!?]+[，；。！？!?]?", source)
        if match.group(0).strip(" \t\r\n，；。！？!?")
    ]
    candidates: List[str] = []
    for start_index in range(len(spans)):
        for end_index in range(start_index, min(len(spans), start_index + 4)):
            start = spans[start_index][0]
            end = spans[end_index][1]
            candidate = source[start:end].strip(" \t\r\n，；。！？!?")
            candidate = re.sub(r"^(?:同时|此外|并且|并|依据|根据|对)", "", candidate).strip()
            if 12 <= len(candidate) <= 68 and candidate in source:
                candidates.append(candidate)
    if not candidates:
        return source[:80]

    def title_score(candidate: str) -> tuple[int, int, int, int, int]:
        has_entity = int(bool(_TITLE_ENTITY_RE.search(candidate)))
        has_number = int(bool(_NUMBER_RE.search(candidate)))
        action = int(bool(_TITLE_ACTION_RE.search(candidate)))
        number_count = len(_NUMBER_RE.findall(candidate))
        return (
            has_entity and has_number,
            has_entity,
            action,
            min(number_count, 3),
            -abs(len(candidate) - 36),
        )

    return max(candidates, key=title_score)[:300]


def _platform_blocks(
    platform: str,
    blocks: Sequence[ContentBlock],
    *,
    lead_with_transition: bool = False,
) -> List[ContentBlock]:
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
    if lead_with_transition or platform in {
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
    blocks = _platform_blocks(
        "master",
        _reindex([*approved_blocks, *_fallback_freeform(role, request)]),
        lead_with_transition=role is OperatorRole.INDUSTRY,
    )
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
            "blocks": [
                _dump(block)
                for block in _platform_blocks(
                    channel,
                    blocks,
                    lead_with_transition=role is OperatorRole.INDUSTRY,
                )
            ],
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
    allow_legacy_canonical: bool = False,
) -> tuple[List[ContentBlock], List[str], List[str]]:
    selected: List[ContentBlock] = []
    errors: List[str] = []
    warnings: List[str] = []

    canonical_blocks = _reindex(
        [*approved_blocks, *template_blocks],
        prefix="canonical",
    )
    canonical_by_ref = {
        block.binding_hash: block
        for block in canonical_blocks
        if block.binding_hash
    }
    canonical_by_legacy_key = {
        (_value(block.kind), block.text, tuple(block.evidence_ids)): block
        for block in canonical_blocks
    }
    required_blocks = [block for block in approved_blocks if block.required]

    if raw_blocks is None:
        return selected, errors, warnings
    if not isinstance(raw_blocks, list):
        return selected, [f"invalid_model_blocks:{prefix}"], warnings
    if len(raw_blocks) > 80:
        errors.append(f"too_many_model_blocks:{prefix}")

    approved_fact_texts = [
        block.text
        for block in approved_blocks
        if block.source_exact
    ]
    approved_number_tokens = {
        token
        for text in approved_fact_texts
        for token in _number_tokens(text)
    }

    def safe_editorial(
        raw: Dict[str, Any],
        *,
        index: int,
        kind: str,
        text: str,
        evidence_ids: List[str],
    ) -> tuple[ContentBlock | None, str | None]:
        if kind not in {
            ContentBlockKind.OPINION.value,
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.CTA.value,
        }:
            return None, "kind_forbidden"
        if not text or len(text) > (160 if kind == ContentBlockKind.CTA.value else 320):
            return None, "length_invalid"
        if evidence_ids or str(raw.get("claim_id") or "").strip():
            return None, "evidence_forbidden"
        if _HTML_RE.search(text) or _URL_RE.search(text):
            return None, "markup_or_url_forbidden"
        # A social hook may repeat a verified amount or date, but cannot invent
        # or recombine a numeric token that is absent from approved evidence.
        if not _number_tokens(text).issubset(approved_number_tokens):
            return None, "number_ungrounded"
        if (
            (
                role is not OperatorRole.INDUSTRY
                and _PRICE_VALUE_RE.search(text)
            )
            or _PROMISE_RE.search(text)
            or _UNSUPPORTED_EDITORIAL_ASSERTION_RE.search(text)
        ):
            return None, "assertion_forbidden"
        if _GENERIC_EDITORIAL_META_RE.search(text):
            return None, "generic_meta_copy_forbidden"
        if _contains_personal_attribution(text):
            # Personal-IP copy needs a recognizable first-person point of view,
            # but the model still may not invent a first-person event, asset,
            # customer, result, or experience.  Exact personal-card statements
            # remain canonical evidence blocks; authored copy may only add a
            # safe present-tense judgment around them.
            if (
                role is not OperatorRole.PERSONAL_IP
                or kind == ContentBlockKind.CTA.value
                or _PERSONAL_EDITORIAL_ASSERTION_RE.search(text)
            ):
                return None, "first_person_forbidden"
        if role is OperatorRole.COMMERCIAL and _COMMERCIAL_ASSERTION_RE.search(text):
            return None, "commercial_assertion_forbidden"
        # A social hook is structural, not a lexical template. Requiring words
        # such as "值得关注", "影响", or "先看" trains the model toward the
        # exact generic AI voice this gate is meant to reject.  Safety is
        # already enforced above; for transitions require only a substantive
        # reader-facing line. Opinion and CTA intent remain explicit.
        compact_text = re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’()（）\-—]", "", text)
        if (
            kind in {
                ContentBlockKind.TRANSITION.value,
                ContentBlockKind.OPINION.value,
            }
            and len(compact_text) < 8
        ):
            return None, "editorial_substance_required"
        if kind == ContentBlockKind.CTA.value and not _EDITORIAL_CTA_RE.search(text):
            return None, "editorial_intent_required"
        # Do not let an editorial block simply echo an exact fact while
        # dropping its evidence binding.
        if text in approved_fact_texts:
            return None, "exact_fact_must_use_ref"
        supplied_origin = str(raw.get("origin") or "").strip().lower()
        if supplied_origin and supplied_origin != ContentBlockOrigin.MODEL_EDITORIAL.value:
            return None, "origin_forbidden"
        for field in ("source_exact", "locked", "required"):
            if raw.get(field) not in (None, False):
                return None, f"{field}_forbidden"
        return (
            _stamp_block(
                ContentBlock(
                    block_id=f"{prefix}-editorial-{index}",
                    kind=ContentBlockKind(kind),
                    text=text,
                    evidence_ids=[],
                    verification_status=VerificationStatus.OPINION,
                    source_exact=False,
                ),
                origin=ContentBlockOrigin.MODEL_EDITORIAL,
                locked=False,
                required=False,
            ),
            None,
        )

    def echo_is_tampered(raw: Dict[str, Any], canonical: ContentBlock) -> bool:
        expected = _dump(canonical)
        scalar_fields = (
            "kind",
            "text",
            "claim_id",
            "verification_status",
            "source_exact",
            "origin",
            "locked",
            "required",
            "binding_hash",
        )
        for field in scalar_fields:
            if field not in raw:
                continue
            actual = raw.get(field)
            wanted = expected.get(field)
            if field in {"kind", "verification_status", "origin"}:
                actual = _value(actual)
                wanted = _value(wanted)
            if actual != wanted:
                return True
        if "evidence_ids" in raw:
            raw_evidence = raw.get("evidence_ids")
            if not isinstance(raw_evidence, list):
                return True
            if [str(value) for value in raw_evidence] != canonical.evidence_ids:
                return True
        return False

    for index, raw in enumerate(raw_blocks[:80], start=1):
        if isinstance(raw, str):
            # Some JSON-only models serialize an opaque selection as the bare
            # block_ref string instead of {"block_ref": "..."}. Accepting an
            # exact server-issued hash is equivalent and cannot introduce
            # model-authored prose; every unknown string still fails closed.
            block_ref = raw.strip()
            canonical = canonical_by_ref.get(block_ref)
            if canonical is None:
                warnings.append(
                    f"discarded_unknown_model_block_ref:{prefix}-{index}"
                )
                continue
            selected.append(canonical)
            continue
        if not isinstance(raw, dict):
            errors.append(f"invalid_model_block:{prefix}-{index}")
            continue

        # JSON-only models do not always reproduce the response-contract field
        # names literally. Normalize harmless structural aliases before the
        # same fail-closed validation below. Some models also copy the
        # ``one_of`` example wrapper from the schema instead of its contents;
        # unwrap it only when it contains one concrete object.
        nested = raw.get("one_of")
        if isinstance(nested, dict):
            raw = nested
        block_ref = str(
            raw.get("block_ref")
            or raw.get("canonical_block_ref")
            or raw.get("ref")
            or ""
        ).strip()
        if block_ref:
            canonical = canonical_by_ref.get(block_ref)
            if canonical is None:
                warnings.append(
                    f"discarded_unknown_model_block_ref:{prefix}-{index}"
                )
                continue
            if echo_is_tampered(raw, canonical):
                errors.append(f"tampered_model_block_ref:{prefix}-{index}")
                continue
            selected.append(canonical)
            continue

        raw_kind = str(
            raw.get("kind")
            or raw.get("type")
            or raw.get("block_type")
            or raw.get("category")
            or ""
        ).strip().lower()
        kind = _EDITORIAL_KIND_ALIASES.get(raw_kind, raw_kind)
        text = str(
            raw.get("text")
            or raw.get("content")
            or raw.get("body")
            or raw.get("value")
            or ""
        ).strip()
        raw_evidence_ids = (
            raw.get("evidence_ids")
            or raw.get("evidence_refs")
            or raw.get("references")
            or []
        )
        if not isinstance(raw_evidence_ids, list):
            errors.append(f"invalid_model_evidence_ids:{prefix}-{index}")
            continue
        evidence_ids = [str(value) for value in raw_evidence_ids]
        canonical = canonical_by_legacy_key.get(
            (kind, text, tuple(evidence_ids))
        )
        if canonical is not None:
            if echo_is_tampered(raw, canonical):
                errors.append(f"tampered_model_block_ref:{prefix}-{index}")
                continue
            # An exact echo is as safe as its binding hash: replace it with the
            # server-owned canonical object before rendering. This keeps model
            # formatting drift from blocking an otherwise valid draft without
            # trusting any model-authored factual prose.
            if not allow_legacy_canonical:
                warnings.append(
                    f"model_canonical_echo_normalized:{prefix}-{index}"
                )
            selected.append(canonical)
            continue

        editorial, reason = safe_editorial(
            raw,
            index=index,
            kind=kind,
            text=text,
            evidence_ids=evidence_ids,
        )
        if editorial is not None:
            selected.append(editorial)
            continue
        if kind in {
            ContentBlockKind.OPINION.value,
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.CTA.value,
        }:
            # Unsafe optional prose is dropped without sacrificing the exact
            # evidence blocks. The warning remains visible to human review.
            warnings.append(
                f"discarded_unsafe_model_editorial:{prefix}-{index}:{reason}"
            )
            continue
        if not raw_kind:
            # A copied response-schema placeholder or untyped optional block is
            # never rendered. Required facts and editorial structure are
            # restored below from immutable server-owned blocks.
            warnings.append(f"discarded_untyped_model_block:{prefix}-{index}")
            continue
        errors.append(
            f"model_block_ref_required:{prefix}-{index}:"
            f"{raw_kind or 'missing_kind'}"
        )

    selected_refs = {
        block.binding_hash for block in selected if block.binding_hash
    }
    missing = [
        block
        for block in required_blocks
        if block.binding_hash not in selected_refs
    ]
    if missing:
        labels = ",".join(
            block.claim_id or block.binding_hash[:12]
            for block in missing[:8]
        )
        if len(missing) > 8:
            labels = f"{labels},+{len(missing) - 8}"
        if errors:
            errors.append(f"required_model_blocks_missing:{prefix}:{labels}")
        else:
            # Required evidence blocks are server-owned and immutable. A model
            # may order or omit them, but cannot veto a required approved
            # claim. Restore omissions deterministically after discarding any
            # unknown bare refs.
            selected.extend(missing)
            warnings.append(
                f"model_required_blocks_restored:{prefix}:{labels}"
            )
    if (
        role is OperatorRole.INDUSTRY
        and not errors
        and not any(
            _value(block.origin) == ContentBlockOrigin.MODEL_EDITORIAL.value
            for block in selected
        )
    ):
        present_kinds = {
            _value(block.kind)
            for block in selected
            if not block.source_exact
        }
        restored_kinds: List[str] = []
        for kind in (
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.OPINION.value,
        ):
            if kind in present_kinds:
                continue
            template = next(
                (
                    block
                    for block in template_blocks
                    if _value(block.kind) == kind
                ),
                None,
            )
            if template is not None:
                selected.append(template)
                restored_kinds.append(kind)
        if restored_kinds:
            warnings.append(
                f"model_editorial_templates_restored:{prefix}:"
                f"{','.join(restored_kinds)}"
            )
    return selected, errors, warnings


def _blocked_status(role: OperatorRole, errors: Sequence[str]) -> ContentStatus:
    quality_codes = (
        "model_generation_required",
        "social_editorial_",
        "platform_editorial_variants_not_distinct",
    )
    if any(error.startswith(quality_codes) for error in errors):
        return ContentStatus.QUALITY_INSUFFICIENT
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
    if any(
        error.startswith(
            (
                "model_generation_required",
                "social_editorial_",
                "platform_editorial_variants_not_distinct",
            )
        )
        for error in errors
    ):
        result.append(
            OperatorQuestion(
                question="请重新生成平台原生成稿，补足具体开场、事件相关分析和自然结尾。",
                reason="当前内容虽未越过事实边界，但仍是内部审稿话术、通用模板或简单事实复述，不适合直接发布。",
                blocking=True,
            )
        )
        return result
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


_LONG_FORM_SURFACES = frozenset({"wechat_mp", "toutiao"})


def _social_surface_quality_errors(
    blocks: Sequence[ContentBlock],
    *,
    surface: str,
) -> List[str]:
    """Require authored social prose, not a safe but generic fact wrapper.

    Evidence and disclosure checks answer whether prose is safe.  They do not
    answer whether it is worth publishing.  Server templates are useful as a
    deterministic diagnostic fallback, but must never satisfy the publishable
    editorial-depth contract.
    """

    editorial = [
        block
        for block in blocks
        if _value(block.origin) == ContentBlockOrigin.MODEL_EDITORIAL.value
        and _value(block.kind) in {
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.OPINION.value,
            ContentBlockKind.CTA.value,
        }
    ]
    # The master is the reusable editorial source and is prompted toward the
    # deepest argument, but two specific authored beats are enough to clear the
    # terminal gate. Requiring a third beat caused models to pad a good hook and
    # analysis with the exact generic observation tail this gate rejects.
    # Long-form platform adaptations may be tighter, while
    # short social/video copy often expresses its hook and its one useful
    # implication in the same authored block.  Requiring every short variant
    # to imitate an article creates padding and rejects platform-native copy.
    if surface == "master":
        minimum_editorial = 2
        minimum_analysis = 1
        hook_counts_as_analysis = False
    elif surface in _LONG_FORM_SURFACES:
        minimum_editorial = 2
        minimum_analysis = 1
        hook_counts_as_analysis = False
    else:
        minimum_editorial = 1
        minimum_analysis = 1
        hook_counts_as_analysis = True
    # Model-authored ``kind`` is useful rendering metadata, but it is not a
    # reliable quality signal: good hooks are often labelled ``opinion`` and
    # analytical steps are often labelled ``transition``.  Judge the actual
    # reading order instead.  The first substantive authored block is the hook;
    # subsequent non-CTA authored blocks must carry the analytical depth.
    hook = editorial[0] if editorial else None
    analysis_candidates = editorial if hook_counts_as_analysis else editorial[1:]
    analysis = [
        block
        for block in analysis_candidates
        if _value(block.kind) in {
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.OPINION.value,
        }
    ]
    errors: List[str] = []
    if any(
        _value(block.origin) == ContentBlockOrigin.SERVER_TEMPLATE.value
        for block in blocks
    ):
        errors.append(f"social_editorial_template_forbidden:{surface}")
    if len(editorial) < minimum_editorial:
        errors.append(
            f"social_editorial_depth_insufficient:{surface}:"
            f"{len(editorial)}/{minimum_editorial}"
        )
    if hook is None:
        errors.append(f"social_editorial_hook_missing:{surface}")
    if len(analysis) < minimum_analysis:
        errors.append(
            f"social_editorial_analysis_insufficient:{surface}:"
            f"{len(analysis)}/{minimum_analysis}"
        )
    return errors


def _editorial_signature(blocks: Sequence[ContentBlock]) -> tuple[str, ...]:
    return tuple(
        block.text.strip()
        for block in blocks
        if _value(block.origin) == ContentBlockOrigin.MODEL_EDITORIAL.value
        and _value(block.kind) in {
            ContentBlockKind.TRANSITION.value,
            ContentBlockKind.OPINION.value,
            ContentBlockKind.CTA.value,
        }
        and block.text.strip()
    )


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
    allow_legacy_canonical = (
        str(candidate.get("_model") or "").strip().lower() == "fallback"
    )

    base_blocks = [ContentBlock(**raw) for raw in fallback.get("blocks") or []]
    raw_master_blocks = candidate.get("blocks")
    master_selection, master_errors, master_warnings = _safe_candidate_blocks(
        role,
        raw_master_blocks,
        approved_blocks,
        prefix="master",
        template_blocks=base_blocks,
        allow_legacy_canonical=allow_legacy_canonical,
    )
    errors.extend(master_errors)
    warnings.extend(master_warnings)
    blocks = _reindex(
        base_blocks if raw_master_blocks is None else master_selection
    )
    if role is OperatorRole.INDUSTRY:
        blocks = _platform_blocks(
            "master",
            blocks,
            lead_with_transition=True,
        )

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
    title = _safe_master_title(
        request,
        approved_blocks,
        candidate.get("master_title"),
    )
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
        raw_variant_blocks = raw.get("blocks")
        variant_selection, variant_errors, variant_warnings = _safe_candidate_blocks(
            role,
            raw_variant_blocks,
            approved_blocks,
            prefix=f"variant-{platform}",
            template_blocks=base_variant_blocks,
            allow_legacy_canonical=allow_legacy_canonical,
        )
        errors.extend(variant_errors)
        warnings.extend(variant_warnings)
        variant_blocks = _reindex(
            (
                base_variant_blocks
                if raw_variant_blocks is None
                else variant_selection
            ),
            prefix=f"{platform}-block",
        )
        if role is OperatorRole.INDUSTRY:
            variant_blocks = _platform_blocks(
                platform,
                variant_blocks,
                lead_with_transition=True,
            )
        body = _render(variant_blocks)
        if body:
            variant_title = _safe_master_title(
                request,
                approved_blocks,
                raw.get("title"),
            )
            variants.append(
                PlatformVariant(
                    platform=platform,
                    format=SUPPORTED_CHANNEL_FORMATS[platform],
                    title=variant_title,
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


def enforce_social_publishability(
    output: OperatorContentOutput,
) -> OperatorContentOutput:
    """Fail closed when safe content is still not real social copy.

    ``normalize_content_output`` owns evidence and disclosure safety.  This
    separate terminal gate owns editorial usefulness so unit callers can
    inspect safely-normalized blocks while the HTTP compose boundary never
    exposes a generic fallback as ``content_ready``.
    """

    if _value(output.status) != ContentStatus.CONTENT_READY.value:
        return output

    errors: List[str] = []
    if output.model.strip().lower() == "fallback":
        errors.append("model_generation_required")
    errors.extend(
        _social_surface_quality_errors(output.blocks, surface="master")
    )
    for variant in output.platform_variants:
        errors.extend(
            _social_surface_quality_errors(
                variant.blocks,
                surface=variant.platform,
            )
        )
    if len(output.platform_variants) > 1:
        signatures = [
            _editorial_signature(variant.blocks)
            for variant in output.platform_variants
        ]
        if len(set(signatures)) != len(signatures):
            errors.append("platform_editorial_variants_not_distinct")
    errors = list(dict.fromkeys(errors))[:50]
    checks = list(_dump(output.critic).get("checks", []))
    checks.append(
        _dump(
            CriticCheck(
                code="social_copy_publishable",
                passed=not errors,
                message=(
                    "Every surface contains model-authored, platform-native editorial depth."
                    if not errors
                    else "Safe fallback, generic templates, or shallow prose cannot be published."
                ),
            )
        )
    )
    if not errors:
        data = _dump(output)
        data["critic"] = {
            **data["critic"],
            "checks": checks,
        }
        return OperatorContentOutput(**data)

    data = _dump(output)
    blocking_quality_risks = [
        {
            "code": error.split(":", 1)[0],
            "message": error,
            "blocking": True,
        }
        for error in errors
    ]
    # A model can emit one discarded optional block per surface, saturating
    # the 50-item warning budget before this terminal quality gate runs.  The
    # API must still return a valid fail-closed response instead of raising a
    # Pydantic validation error and leaking an empty HTTP 500 body.  Keep the
    # terminal blockers first, then retain as many earlier diagnostics as the
    # public response contract permits.
    capped_risks = [
        *blocking_quality_risks,
        *list(data.get("risk_flags", [])),
    ][:50]
    data.update(
        {
            "status": ContentStatus.QUALITY_INSUFFICIENT.value,
            "master_title": None,
            "master_content": None,
            "blocks": [],
            "platform_variants": [],
            "evidence_refs": [],
            "questions": [
                _dump(question)
                for question in _questions(output.role_id, errors)
            ],
            "risk_flags": capped_risks,
            "critic": {
                **data["critic"],
                "passed": False,
                "errors": errors,
                "checks": checks,
            },
        }
    )
    return OperatorContentOutput(**data)


def combine_publishable_surfaces(
    outputs: Sequence[OperatorContentOutput],
) -> Optional[OperatorContentOutput]:
    """Combine whole, independently safe surfaces from multiple model attempts.

    A single compose call can contain a strong master and six strong platform
    variants while one or two variants are shallow.  Replacing that entire
    response with a second all-or-nothing attempt makes quality nondeterministic:
    the retry may fix those two surfaces but regress a different one.  This
    helper keeps only *whole* surfaces that already pass the same terminal
    editorial gate.  It never splices paragraphs, changes immutable evidence
    blocks, or promotes a server template.
    """

    ready = [
        output
        for output in outputs
        if _value(output.status) == ContentStatus.CONTENT_READY.value
        and output.master_content
        and output.platform_variants
        and not output.critic.errors
    ]
    if not ready:
        return None

    master = next(
        (
            output
            for output in ready
            if not _social_surface_quality_errors(output.blocks, surface="master")
        ),
        None,
    )
    if master is None:
        return None

    platform_order = [variant.platform for variant in ready[0].platform_variants]
    selected_variants: List[PlatformVariant] = []
    used_signatures: set[tuple[str, ...]] = set()
    for platform in platform_order:
        options = [
            variant
            for output in ready
            for variant in output.platform_variants
            if variant.platform == platform
            and not _social_surface_quality_errors(
                variant.blocks,
                surface=platform,
            )
        ]
        selected = next(
            (
                variant
                for variant in options
                if _editorial_signature(variant.blocks) not in used_signatures
            ),
            None,
        )
        if selected is None:
            return None
        selected_variants.append(selected)
        used_signatures.add(_editorial_signature(selected.blocks))

    warnings = list(
        dict.fromkeys(
            warning
            for output in ready
            for warning in output.critic.warnings
        )
    )[:50]
    checks = list(_dump(master.critic).get("checks", []))
    checks.append(
        _dump(
            CriticCheck(
                code="social_surface_retry_aggregation",
                passed=True,
                message=(
                    "Each retained master or platform variant independently passed "
                    "the terminal social-copy gate without paragraph splicing."
                ),
            )
        )
    )
    selected_blocks = [
        *master.blocks,
        *(
            block
            for variant in selected_variants
            for block in variant.blocks
        ),
    ]
    data = _dump(master)
    data.update(
        {
            "master_content": _render(master.blocks),
            "blocks": [_dump(block) for block in master.blocks],
            "platform_variants": [
                _dump(variant) for variant in selected_variants
            ],
            "evidence_refs": _evidence_refs(selected_blocks),
            "questions": [],
            "risk_flags": [
                {
                    "code": warning.split(":", 1)[0],
                    "message": warning,
                    "blocking": False,
                }
                for warning in warnings
            ],
            "critic": {
                "passed": True,
                "errors": [],
                "warnings": warnings,
                "checks": checks,
            },
        }
    )
    return OperatorContentOutput(**data)


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
    canonical_blocks = _reindex(
        [*approved_blocks, *_fallback_freeform(role, request)],
        prefix="canonical",
    )
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
    approved_direction = request.approved_proposal
    editorial_brief = (
        {
            "angle": approved_direction.angle,
            "audience_value": approved_direction.audience_value,
            "cta": approved_direction.cta,
        }
        if role is OperatorRole.INDUSTRY
        else {
            "angle": "Explain why the approved public facts matter to the intended reader.",
            "audience_value": "Give the reader a clear implication or question to consider.",
            "cta": None,
        }
    )
    publication_brief = {
        "commercial": {
            "voice": "A precise business operator who connects one verified capability or offer to one concrete customer situation.",
            "reader_outcome": "The reader understands whether the verified offer fits their situation and what useful next step is available.",
        },
        "industry": {
            "voice": "An informed industry operator with a clear thesis, not a news copier or compliance reviewer.",
            "reader_outcome": "The reader understands one concrete change in incentives, costs, bargaining position, or operating rules and knows what observable consequence matters next.",
        },
        "personal_ip": {
            "voice": "A recognizable personal-IP operator who uses only approved personal cards and never invents a first-person experience.",
            "reader_outcome": "The reader gets one specific, useful point of view grounded in the owner's approved experience or viewpoint.",
        },
    }[role.value]
    platform_briefs = {
        "wechat_moments": (
            "A concise personal-feed post: lead with a concrete judgment, "
            "surface the verified facts quickly, add one useful implication, "
            "and do not force a CTA."
        ),
        "wechat_mp": (
            "A readable analysis article: concrete headline logic, short "
            "paragraphs, two or three distinct analytical steps, and a natural "
            "ending rather than an audit disclaimer."
        ),
        "wechat_channels": (
            "A spoken script with a sharp opening, short sentences, clear fact "
            "beats, one concrete interpretation, and a memorable final line."
        ),
        "douyin": (
            "A fast spoken script: state the tension in the first sentence, "
            "use compact fact beats, explain why it matters, and avoid formal "
            "announcement language."
        ),
        "kuaishou": (
            "A plainspoken short-video script: direct opening, accessible fact "
            "breakdown, practical industry implication, no bureaucratic tone."
        ),
        "xiaohongshu": (
            "A scan-friendly note: specific hook, compact information-card "
            "structure, clear takeaway, and no fake personal experience."
        ),
        "toutiao": (
            "A fact-led analysis article: explain the event, separate two or "
            "three implications, and end with concrete follow-up questions."
        ),
        "weitoutiao": (
            "A compact news commentary: one strong judgment, the essential "
            "verified facts, and one non-generic implication."
        ),
    }
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
            "angle": editorial_brief["angle"],
            "audience_value": editorial_brief["audience_value"],
            "key_points": public_key_points,
            "suggested_formats": list(request.channels),
            "cta": editorial_brief["cta"],
            "first_person": role is OperatorRole.PERSONAL_IP,
        },
        "approved_editorial_brief": editorial_brief,
        "publication_brief": {
            **publication_brief,
            "thesis_contract": [
                "Choose one event-specific thesis before writing. Do not merely summarize the source or announce that facts and opinions are separate.",
                "Build the thesis around a concrete relationship already visible in the approved claims: actor versus affected party, penalty versus remedy, capability versus customer constraint, or action versus observable consequence.",
                "Make every editorial block earn its place: hook with the tension, explain the mechanism or reader impact, then land on a useful judgment. Do not add an automatic observation list or engagement question.",
                "Use factual nouns, actors, rules, and exact grounded numbers from the approved claims as anchors. Avoid abstract paragraphs that could be pasted under an unrelated news item.",
            ],
            "role_thesis": {
                "commercial": (
                    "Connect the verified capability and offer to one concrete customer "
                    "constraint: what decision becomes easier, what risk remains under "
                    "human control, and what specific next conversation is useful."
                ),
                "industry": (
                    "Explain the changed incentive, bargaining position, operating cost, "
                    "or enforceable rule and name the affected participant."
                ),
                "personal_ip": (
                    "Let the exact approved personal-card block carry the experience. "
                    "Around it, write neutral present-tense analysis of the tradeoff and "
                    "a practical principle the reader can use. Do not use I or we in "
                    "authored blocks and do not invent another event or credential."
                ),
            }[role.value],
            "reader_facing_rule": (
                "Never expose editorial workflow labels such as fact section, "
                "opinion section, editor's note, review note, evidence note, "
                "or 'first separate fact from judgment'."
            ),
            "authored_block_rule": {
                "commercial": (
                    "Keep capability, service, price, discount, and company claims in "
                    "canonical block_ref objects only. In authored blocks, do not say "
                    "we, our company, our product, or our team provides or supports "
                    "anything. Write only the customer decision, constraint, tradeoff, "
                    "risk, or useful next conversation implied by the canonical facts."
                ),
                "industry": (
                    "Write analysis rather than another event assertion: name the changed "
                    "incentive, cost, bargaining position, rule, or observable consequence."
                ),
                "personal_ip": (
                    "Use canonical block_ref objects for every action, project, customer, "
                    "result, credential, and past experience. Do not use first-person "
                    "attribution in any authored block. Use neutral present-tense analysis "
                    "and let the canonical personal-card block carry the owner's voice "
                    "and every first-person experience."
                ),
            }[role.value],
        },
        "platform_editorial_briefs": {
            channel: platform_briefs[channel]
            for channel in request.channels
            if channel in platform_briefs
        },
        "claims": public_claims,
        "authorized_context": [_sanitized_context(context) for context in visible],
        "canonical_block_registry": [
            {
                "block_ref": block.binding_hash,
                "kind": _value(block.kind),
                "text": block.text,
                "claim_id": block.claim_id,
                "evidence_ids": list(block.evidence_ids),
                "verification_status": _value(block.verification_status),
                "source_exact": block.source_exact,
                "origin": _value(block.origin),
                "locked": block.locked,
                "required": block.required,
                "binding_hash": block.binding_hash,
            }
            for block in canonical_blocks
        ],
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
        "blocks": (
            "array of actual block objects; each object must be either "
            '{"block_ref":"<one exact block_ref from canonical_block_registry>"} '
            "or "
            '{"kind":"opinion|transition|cta","text":"<original editorial '
            'framing>","evidence_ids":[]}; do not copy this description or '
            "return a one_of/schema wrapper"
        ),
        "platform_variants": [
            {
                "platform": "one requested channel",
                "format": "server-normalized; model value is advisory only",
                "title": "string",
                "blocks": "array of actual block objects using the same blocks contract",
            }
        ],
        "editorial_shape": {
            "master": (
                "include at least one authored hook followed by at least one "
                "distinct authored analysis block"
            ),
            "long_form": (
                "for wechat_mp and toutiao, include one authored hook followed "
                "by at least one distinct authored analysis block"
            ),
            "short_form": (
                "include at least one authored block whose first sentence is a "
                "platform-native hook and whose full text carries one concrete "
                "implication; the same block may satisfy both jobs"
            ),
            "classification_rule": (
                "kind=transition and kind=opinion are rendering hints. Reading order "
                "is mandatory: hook first, then mechanism, tradeoff, implication, "
                "reader impact, or concluding judgment."
            ),
        },
        "safety": [
            "Do not write master_content or variant body directly; the server renders normalized blocks.",
            "Every factual, pricing, identity, or experience statement must use its exact block_ref from canonical_block_registry; never paraphrase it.",
            "Every platform variant must include every registry block marked required, but may choose its own safe order.",
            "Write a native social master_title and a distinct title for each platform. Keep each title between 12 and 36 Chinese characters when possible; it may use grounded entity names and numbers from public claims plus a clearly editorial judgment, but no new event assertion.",
            "For the master, use two to six concise editorial blocks: a concrete hook, at least one distinct analytical step, and an optional natural close. For wechat_mp and toutiao use at least two authored blocks: a hook plus one analytical step. A short-feed or video variant may use one to three authored blocks; its first block must itself contain a concrete event-specific implication, not merely announce the topic.",
            "Editorial blocks may only be opinion, transition, or CTA. They must not add unsupported facts, named-entity claims, quotations, prices, promises, or unapproved first-person attribution.",
            "For personal_ip, do not use first-person attribution in authored opinion, transition, or CTA blocks. The exact canonical personal-card block carries the owner's voice and every action, project, customer, result, credential, or experience; authored blocks add only neutral present-tense analysis.",
            "Every authored opinion, transition, or CTA object must set evidence_ids to [] and omit claim_id and block_ref. Evidence binding belongs only to canonical registry blocks selected by block_ref.",
            "A verified number or date may appear in editorial framing only when copied exactly from the supplied public claims; never calculate, round, compare, or combine numbers.",
            "Do not use audit/meta copy such as '先把事实和判断分开', '以下分析', '编辑说明', '编辑观点', '事实部分', '公开信息只是起点', or '接下来可以继续观察'. The draft must read as publishable copy, not an internal review note.",
            "Avoid interchangeable filler such as '对关注某领域的人来说', '从行业视角看', '这类案例的价值在于', or '重点不只是……更是……'. Every analytical block must advance a concrete thesis tied to this event.",
            "Build the thesis from a concrete contrast or relationship already present in the approved facts (for example penalty versus restitution, announcement versus enforceable action, or platform versus affected participant). Name the affected actor, changed rule/incentive, or observable consequence instead of merely saying the event is important.",
            "Keep fact/opinion separation in block metadata, never as reader-facing wording. The published copy should not explain its own editorial process.",
            "Do not force a question, invitation, disclaimer, or CTA when a firm closing judgment is more natural.",
            "The server keeps required factual blocks verbatim for audit and renders long ones as concise source-exact fact beats. Do not repeat the full announcement in editorial prose.",
            "Make each requested platform variant meaningfully different in title, rhythm, depth, and reader action while preserving every required factual block reference.",
        ],
    }
    return payload
