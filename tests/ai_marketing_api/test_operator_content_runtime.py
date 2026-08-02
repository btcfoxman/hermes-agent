from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from ai_marketing_api.operator_content import (
    ContentBlockKind,
    ContentStatus,
    OperatorComposeRequest,
    build_content_fallback,
    content_request_payload,
    curated_industry_surface_candidate,
    disclosable_contexts,
    enforce_social_publishability,
    missing_publishable_surfaces,
    normalize_content_output,
)
from ai_marketing_api.operator_runtime import (
    ApprovedPreference,
    AuthorizedContext,
    OperatorClaim,
    OperatorRegistry,
    OperatorRisk,
    OperatorRole,
    ProposalOutline,
)


AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)


def _context(
    record_id: str,
    role: str,
    space: str,
    record_type: str,
    *,
    content: str,
    source_uri: str | None = None,
    source_tier: str = "trusted",
    structured_data: dict | None = None,
) -> AuthorizedContext:
    return AuthorizedContext(
        record_id=record_id,
        space=space,
        record_type=record_type,
        content=content,
        structured_data=structured_data or {},
        status="approved",
        authorized_roles=[role],
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2027, 1, 1, tzinfo=timezone.utc),
        source_uri=source_uri,
        source_tier=source_tier,
    )


def _claim(
    text: str,
    kind: str,
    evidence_ids: list[str],
    verification_status: str = "verified",
) -> OperatorClaim:
    return OperatorClaim(
        text=text,
        kind=kind,
        evidence_ids=evidence_ids,
        verification_status=verification_status,
    )


def _request(
    contexts: list[AuthorizedContext],
    claims: list[OperatorClaim],
    *,
    objective: str = "Create an approved daily content package",
    channels: list[str] | None = None,
) -> OperatorComposeRequest:
    return OperatorComposeRequest(
        objective=objective,
        topic="Approved daily topic",
        audience="business owners",
        channels=channels if channels is not None else ["wechat_mp", "xiaohongshu"],
        constraints=["Keep facts traceable."],
        authorized_context=contexts,
        as_of=AS_OF,
        approved_proposal=ProposalOutline(
            title="Approved editorial title",
            angle="Use the approved evidence and keep opinions separate.",
            audience_value="Explain the approved direction clearly.",
            key_points=[claim.text for claim in claims],
            suggested_formats=channels or ["wechat_mp", "xiaohongshu"],
            cta="邀请读者提供具体场景，进入人工沟通。",
            first_person=any(claim.kind in {"identity", "experience"} for claim in claims),
        ),
        claims=claims,
    )


def _run(
    role: OperatorRole,
    request: OperatorComposeRequest,
    candidate: dict | None = None,
):
    registry = OperatorRegistry()
    authorized = registry.authorize(role, request.authorized_context, request.as_of)
    fallback = build_content_fallback(registry, role, request, authorized)
    return normalize_content_output(
        registry,
        role,
        request,
        authorized,
        fallback,
        candidate if candidate is not None else fallback,
    )


def test_compose_forwards_only_matching_non_account_preferences_as_style_guidance():
    fact = "The vendor released version 2."
    request = _request(
        [
            _context(
                "source-1",
                "industry",
                "industry",
                "source_item",
                content=fact,
                source_tier="official",
            )
        ],
        [_claim(fact, "fact", ["source-1"])],
        channels=["wechat_mp"],
    )
    request.approved_preferences = [
        ApprovedPreference(
            preference_version_id="operatorpref-industry-v2",
            role_id="industry",
            preference_key="industry.article.structure",
            platform="wechat_mp",
            content_type="article",
            guidance="先说明规则变化，再落到产品团队的可观察影响。",
            status="approved",
            version=2,
        ),
        ApprovedPreference(
            preference_version_id="operatorpref-industry-account-v1",
            role_id="industry",
            preference_key="industry.account.voice",
            account_id="private-account",
            guidance="仅用于指定账号。",
            status="approved",
            version=1,
        ),
    ]
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.INDUSTRY, request.authorized_context, request.as_of
    )
    registry.authorize_preferences(
        OperatorRole.INDUSTRY, request.approved_preferences
    )

    payload = content_request_payload(
        OperatorRole.INDUSTRY, request, authorized
    )

    assert [
        item["preference_version_id"] for item in payload["approved_preferences"]
    ] == ["operatorpref-industry-v2"]
    assert "not evidence" in " ".join(payload["preference_usage_contract"])


def test_industry_compose_replaces_internal_review_angle_with_reader_facing_thesis():
    fact = "监管要求平台退还被扣留的经营资金并公开整改措施。"
    source = _context(
        "source-review-angle",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(fact, "fact", [source.record_id])],
        channels=["wechat_mp"],
    )
    request.approved_proposal.angle = (
        "区分已核验事实与编辑观点，解释变化的业务影响及后续观察项。"
    )
    request.approved_proposal.audience_value = (
        "帮助读者理解事件影响，而不是重复新闻摘要。"
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.INDUSTRY, request.authorized_context, request.as_of
    )

    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    angle = payload["approved_editorial_brief"]["angle"]

    assert "区分" not in angle
    assert "事实" not in angle
    assert "观点" not in angle
    assert "cash flow" in angle
    assert "clear industry thesis" in angle
    audience_value = payload["approved_editorial_brief"]["audience_value"]
    assert "重复新闻摘要" not in audience_value
    assert "operating decision" in audience_value
    guard = payload["industry_editorial_guard"]
    assert guard["event_type"] == "regulatory_return_of_withheld_business_funds"
    assert guard["fund_or_rule"] == "经营资金"
    assert "discounts or promotions" in guard["forbidden_reframes"]
    assert "经营资金不应继续被平台内部规则强制占用" in guard[
        "positive_thesis_seed"
    ][0]


def test_curated_funds_remedy_repairs_are_publishable_on_every_social_surface():
    fact = (
        "监管部门要求平台全额退还强制扣除酒店经营者的订单储备金，"
        "并要求企业全面整改及公开整改措施。"
    )
    channels = [
        "wechat_moments",
        "wechat_mp",
        "wechat_channels",
        "douyin",
        "kuaishou",
        "xiaohongshu",
        "toutiao",
        "weitoutiao",
    ]
    context = _context(
        "source-funds-remedy",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_tier="official",
    )
    request = _request(
        [context],
        [_claim(fact, "fact", [context.record_id])],
        channels=channels,
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.INDUSTRY,
        request.authorized_context,
        request.as_of,
    )
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    compose_payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )

    for surface in ["master", *channels]:
        candidate = curated_industry_surface_candidate(compose_payload, surface)
        assert candidate is not None
        normalized = normalize_content_output(
            registry,
            OperatorRole.INDUSTRY,
            request,
            authorized,
            fallback,
            candidate,
        )
        missing = missing_publishable_surfaces(
            [normalized],
            [] if surface == "master" else [surface],
        )
        assert surface not in missing, normalized.critic.warnings
        candidate_text = "\n".join(
            str(block.get("text") or "")
            for block in [
                *(candidate.get("blocks") or []),
                *(
                    block
                    for variant in candidate.get("platform_variants") or []
                    for block in variant.get("blocks") or []
                ),
            ]
            if isinstance(block, dict) and not block.get("block_ref")
        )
        assert "先把" not in candidate_text
        assert "不是" not in candidate_text
        assert "值得关注" not in candidate_text
        assert not any(char.isdigit() for char in candidate_text)


def _candidate_from_required_refs(
    payload: dict,
    channels: list[str],
) -> dict:
    blocks = [
        {"block_ref": block["block_ref"]}
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]
    return {
        "_model": "structured-ref-test-model",
        "blocks": deepcopy(blocks),
        "platform_variants": [
            {
                "platform": channel,
                "blocks": deepcopy(blocks),
            }
            for channel in channels
        ],
    }


@pytest.mark.parametrize("role", list(OperatorRole))
def test_three_roles_compose_safe_fallback_with_exact_evidence_and_all_channels(role):
    expected_templates = {
        OperatorRole.COMMERCIAL: {
            "沟通建议：结合具体业务场景判断适用性，并由人工确认下一步。",
            "如需判断适用性，请提交具体业务场景，由人工商务沟通。",
        },
        OperatorRole.INDUSTRY: {
            "一个事件是否重要，不能只看热度，还要看它改变了谁的规则、成本与选择。",
            "行业判断不能停在结论上：规则是否改变、执行是否持续、相关参与者是否真实感受到变化，才是后续验证重点。",
        },
        OperatorRole.PERSONAL_IP: {
            "先看已确认的经历与观点，再讨论其中的启发。",
            "欢迎分享你的观察与不同视角。",
        },
    }
    if role is OperatorRole.COMMERCIAL:
        context = _context(
            "cap-1",
            role.value,
            "company_public",
            "capability",
            content="已批准能力：把内容任务交接到人工终审。",
        )
        claims = [_claim(context.content, "fact", [context.record_id])]
    elif role is OperatorRole.INDUSTRY:
        context = _context(
            "source-1",
            role.value,
            "industry",
            "source_item",
            content="官方发布了新的模型版本。",
            source_uri="https://vendor.example/releases/model",
            source_tier="official",
        )
        claims = [_claim(context.content, "fact", [context.record_id])]
    else:
        context = _context(
            "card-1",
            role.value,
            "personal_approved",
            "experience",
            content="第一次带团队出海时，我暂停了未经验证的投放。",
        )
        claims = [_claim(context.content, "experience", [context.record_id])]

    output = _run(role, _request([context], claims))

    assert output.schema_version == "operator.content.v1"
    assert output.action == "compose"
    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.requires_human_review is True
    assert output.critic.passed is True
    assert output.master_title == context.content[:300]
    assert {variant.platform for variant in output.platform_variants} == {
        "wechat_mp",
        "xiaohongshu",
    }
    assert len(output.platform_variants) == 2
    exact_blocks = [block for block in output.blocks if block.source_exact]
    assert exact_blocks
    assert all(block.text in context.content for block in exact_blocks)
    assert all(block.evidence_ids == [context.record_id] for block in exact_blocks)
    assert context.content in (output.master_content or "")
    assert all(context.content in variant.body for variant in output.platform_variants)
    templates = [block for block in output.blocks if block.origin == "server_template"]
    assert {block.text for block in templates} == expected_templates[role]
    assert all(block.locked is True for block in templates)
    assert all(block.required is False for block in templates)
    assert all(len(block.binding_hash) == 64 for block in templates)


def test_claim_text_drift_blocks_every_publishable_field():
    source = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        content="The official source states one exact fact.",
        source_uri="https://official.example/fact",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim("The source probably means a larger claim.", "fact", [source.record_id])],
    )

    output = _run(OperatorRole.INDUSTRY, request)

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_title is None
    assert output.master_content is None
    assert output.blocks == []
    assert output.platform_variants == []
    assert "claim_not_verbatim:claim-1" in output.critic.errors


def test_long_industry_claim_uses_a_concise_verbatim_title_excerpt():
    source_text = (
        "中国国家市场监督管理总局通报相关决定，"
        "对携程集团有限公司滥用市场支配地位实施垄断行为作出行政处罚，"
        "没收违法所得16.58亿元，并处以罚款35.21亿元，"
        "罚没款合计51.79亿元，同时要求企业全面整改并公开整改措施。"
    )
    source = _context(
        "source-long-title",
        "industry",
        "industry",
        "source_item",
        content=source_text,
        source_uri="https://official.example/decision",
        source_tier="official",
    )
    output = _run(
        OperatorRole.INDUSTRY,
        _request(
            [source],
            [_claim(source_text, "fact", [source.record_id])],
        ),
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.master_title
    assert output.master_title in source_text
    assert "携程集团有限公司" in output.master_title
    assert len(output.master_title) <= 96
    assert output.master_title != source_text


def test_only_claim_references_are_disclosed_to_the_model():
    approved = _context(
        "approved",
        "commercial",
        "company_public",
        "capability",
        content="Approved public capability.",
    )
    unrelated = _context(
        "unrelated",
        "commercial",
        "company_public",
        "case_study",
        content="Extra context that was not in the approved proposal.",
    )
    request = _request(
        [approved, unrelated],
        [_claim(approved.content, "fact", [approved.record_id])],
    )

    visible = disclosable_contexts(OperatorRole.COMMERCIAL, request, request.authorized_context)
    payload = content_request_payload(
        OperatorRole.COMMERCIAL,
        request,
        request.authorized_context,
    )

    assert [context.record_id for context in visible] == [approved.record_id]
    assert [context["record_id"] for context in payload["authorized_context"]] == [approved.record_id]
    assert unrelated.content not in str(payload)


def test_content_request_exposes_server_owned_canonical_block_refs():
    approved = _context(
        "approved-ref",
        "commercial",
        "company_public",
        "capability",
        content="Approved public capability.",
    )
    request = _request(
        [approved],
        [_claim(approved.content, "fact", [approved.record_id])],
    )

    payload = content_request_payload(
        OperatorRole.COMMERCIAL,
        request,
        request.authorized_context,
    )

    registry = payload["canonical_block_registry"]
    required = [block for block in registry if block["required"]]
    assert len(required) == 1
    assert required[0]["text"] == approved.content
    assert required[0]["evidence_ids"] == [approved.record_id]
    assert required[0]["origin"] == "approved_claim"
    assert required[0]["locked"] is True
    assert required[0]["block_ref"] == required[0]["binding_hash"]
    assert len(required[0]["block_ref"]) == 64
    canonical = {
        "kind": required[0]["kind"],
        "text": required[0]["text"],
        "claim_id": required[0]["claim_id"] or None,
        "evidence_ids": required[0]["evidence_ids"],
        "origin": required[0]["origin"],
        "locked": required[0]["locked"],
        "required": required[0]["required"],
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert required[0]["binding_hash"] == expected_hash
    block_contract = payload["response_contract"]["blocks"]
    assert "one exact block_ref from canonical_block_registry" in block_contract
    assert "opinion|transition|cta" in block_contract
    assert "do not copy this description" in block_contract
    editorial_shape = payload["response_contract"]["editorial_shape"]
    assert "at least one distinct authored analysis block" in editorial_shape["master"]
    assert "at least one distinct authored analysis block" in editorial_shape["long_form"]
    assert "same block may satisfy both jobs" in editorial_shape["short_form"]
    assert "Reading order is mandatory" in editorial_shape["classification_rule"]
    assert "our company" in payload["publication_brief"]["authored_block_rule"]
    thesis_contract = " ".join(payload["publication_brief"]["thesis_contract"])
    assert "connecting one approved actor" in thesis_contract
    assert "not as a correction" in thesis_contract


def test_commercial_internal_context_is_readable_but_never_publishable_or_model_visible():
    internal = _context(
        "internal-1",
        "commercial",
        "company_internal",
        "delivery_boundary",
        content="Internal floor price and delivery boundary must stay private.",
    )
    request = _request(
        [internal],
        [_claim(internal.content, "fact", [internal.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.COMMERCIAL, [internal], AS_OF)
    payload = content_request_payload(OperatorRole.COMMERCIAL, request, authorized)
    output = _run(OperatorRole.COMMERCIAL, request)

    assert payload["authorized_context"] == []
    assert internal.content not in str(payload)
    assert output.status != ContentStatus.CONTENT_READY.value
    assert internal.content not in output.model_dump_json()


def test_private_context_paraphrases_in_direction_fields_are_not_sent_to_model():
    internal = _context(
        "internal-direction-1",
        "commercial",
        "company_internal",
        "delivery_boundary",
        content="Internal floor price and delivery boundary must stay private.",
    )
    request = _request(
        [internal],
        [_claim(internal.content, "fact", [internal.record_id])],
    )
    leaked_fragments = {
        "objective": "Use the secret floor price in today's draft.",
        "topic": "Secret floor price launch",
        "audience": "Unannounced enterprise customer",
        "constraint": "Mention the private delivery date.",
        "title": "Private floor price",
        "angle": "Reveal part of the internal delivery boundary.",
        "audience_value": "Tell the customer the unannounced condition.",
        "cta": "Ask for the secret discount.",
    }
    request.objective = leaked_fragments["objective"]
    request.topic = leaked_fragments["topic"]
    request.audience = leaked_fragments["audience"]
    request.constraints = [leaked_fragments["constraint"]]
    request.approved_proposal.title = leaked_fragments["title"]
    request.approved_proposal.angle = leaked_fragments["angle"]
    request.approved_proposal.audience_value = leaked_fragments["audience_value"]
    request.approved_proposal.cta = leaked_fragments["cta"]

    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.COMMERCIAL, [internal], AS_OF)
    payload = content_request_payload(OperatorRole.COMMERCIAL, request, authorized)
    serialized = str(payload)

    assert payload["authorized_context"] == []
    assert payload["claims"] == []
    assert internal.record_id not in serialized
    assert internal.content not in serialized
    for fragment in leaked_fragments.values():
        assert fragment not in serialized


def test_commercial_unique_public_offer_is_copied_verbatim():
    public_text = "标准公开方案：每月 3000 元，适用于已批准渠道。"
    offer = _context(
        "offer-1",
        "commercial",
        "company_public",
        "business_offer",
        content="Internal field must not be used.",
        structured_data={"publicly_quoteable": True, "public_text": public_text},
    )
    request = _request(
        [offer],
        [_claim(public_text, "fact", [offer.record_id])],
        objective="生成当前公开报价说明",
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.blocks[0].text == public_text
    assert output.blocks[0].evidence_ids == [offer.record_id]
    assert "Internal field" not in output.model_dump_json()


def test_commercial_business_offer_contiguous_excerpt_is_allowed():
    public_text = "标准公开方案：每月 3000 元，适用于已批准渠道，有效期以当前活动为准。"
    excerpt = "每月 3000 元，适用于已批准渠道"
    offer = _context(
        "offer-excerpt",
        "commercial",
        "company_public",
        "business_offer",
        content="Internal field must not be used.",
        structured_data={"publicly_quoteable": True, "public_text": public_text},
    )
    request = _request(
        [offer],
        [_claim(excerpt, "fact", [offer.record_id])],
        objective="生成当前公开报价说明",
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.blocks[0].text == excerpt
    assert output.blocks[0].evidence_ids == [offer.record_id]
    assert public_text not in (output.master_content or "")


@pytest.mark.parametrize(
    "objective",
    [
        "不涉及报价、活动、折扣或交付承诺，只说明已批准能力。",
        "Do not include price, quote, discount, or cost; use approved capability only.",
    ],
)
def test_commercial_explicit_no_offer_direction_does_not_require_business_offer(
    objective,
):
    capability = _context(
        "cap-no-offer",
        "commercial",
        "company_public",
        "capability",
        content="已批准能力：内容任务保留人工终审与排期确认。",
    )
    request = _request(
        [capability],
        [_claim(capability.content, "fact", [capability.record_id])],
        objective=objective,
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert "unique_public_business_offer_required" not in output.critic.errors


def test_commercial_offer_excerpt_cannot_mix_non_offer_evidence() -> None:
    public_text = "标准公开方案：每月 3000 元，适用于已批准渠道。"
    excerpt = "每月 3000 元"
    offer = _context(
        "offer-exclusive",
        "commercial",
        "company_public",
        "business_offer",
        content="not public",
        structured_data={"publicly_quoteable": True, "public_text": public_text},
    )
    knowledge = _context(
        "knowledge-price-copy",
        "commercial",
        "company_public",
        "capability",
        content=f"重复文本：{excerpt}",
    )
    request = _request(
        [offer, knowledge],
        [_claim(excerpt, "fact", [offer.record_id, knowledge.record_id])],
        objective="生成当前公开报价说明",
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status == ContentStatus.NEEDS_INPUT.value
    assert "business_offer_claim_missing" in output.critic.errors


def test_commercial_multiple_current_offers_block_price_content_even_if_one_was_claimed():
    offers = [
        _context(
            f"offer-{index}",
            "commercial",
            "company_public",
            "business_offer",
            content="not public",
            structured_data={
                "publicly_quoteable": True,
                "public_text": f"公开方案 {index}：每月 {index}000 元。",
            },
        )
        for index in (1, 2)
    ]
    first_text = offers[0].structured_data["public_text"]
    request = _request(
        offers,
        [_claim(first_text, "fact", [offers[0].record_id])],
        objective="生成当前报价和折扣说明",
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status == ContentStatus.NEEDS_INPUT.value
    assert output.master_content is None
    assert "unique_public_business_offer_required" in output.critic.errors


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "沟通建议：只需 99 元即可购买。",
        "沟通建议：我们保证效果并且没有风险。",
        "沟通建议：我们的产品能够实现未经批准的能力。",
    ],
)
def test_commercial_model_cannot_add_price_promise_or_capability(unsafe_text):
    capability = _context(
        "cap-1",
        "commercial",
        "company_public",
        "capability",
        content="Approved capability only.",
    )
    request = _request(
        [capability],
        [_claim(capability.content, "fact", [capability.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.COMMERCIAL, [capability], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.COMMERCIAL, request, authorized)
    payload = content_request_payload(
        OperatorRole.COMMERCIAL,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["_model"] = "unsafe-test-model"
    candidate["blocks"].append(
        {"kind": "transition", "text": unsafe_text, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.COMMERCIAL,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert unsafe_text not in output.model_dump_json()
    assert any(
        warning.startswith("discarded_unsafe_model_editorial")
        for warning in output.critic.warnings
    )


def test_approved_commercial_effect_guarantee_is_still_blocked():
    capability = _context(
        "cap-unsafe",
        "commercial",
        "company_public",
        "capability",
        content="该服务保证效果并承诺交付结果。",
    )
    request = _request(
        [capability],
        [_claim(capability.content, "fact", [capability.record_id])],
    )

    output = _run(OperatorRole.COMMERCIAL, request)

    assert output.status != ContentStatus.CONTENT_READY.value
    assert "unsafe_business_promise" in output.critic.errors
    assert output.master_content is None


def test_industry_single_secondary_and_same_domain_sources_do_not_pass_double_source_gate():
    single = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        content="Secondary report one.",
        source_uri="https://news.example/story-one",
        source_tier="secondary",
    )
    same_domain = _context(
        "source-2",
        "industry",
        "industry",
        "report",
        content="Secondary report two.",
        source_uri="https://news.example/story-two",
        source_tier="trusted",
    )

    single_output = _run(
        OperatorRole.INDUSTRY,
        _request([single], [_claim(single.content, "fact", [single.record_id])]),
    )
    same_domain_output = _run(
        OperatorRole.INDUSTRY,
        _request(
            [single, same_domain],
            [
                _claim(single.content, "fact", [single.record_id]),
                _claim(same_domain.content, "fact", [same_domain.record_id]),
            ],
        ),
    )

    assert single_output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert same_domain_output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert any(
        error.startswith("industry_single_non_primary_source:")
        for error in single_output.critic.errors
    )
    assert any(
        error.startswith("industry_single_non_primary_source:")
        for error in same_domain_output.critic.errors
    )


def test_industry_different_facts_from_two_domains_do_not_cross_corroborate():
    sources = [
        _context(
            "source-1",
            "industry",
            "industry",
            "source_item",
            content="First source exact statement.",
            source_uri="https://one.example/story",
            source_tier="trusted",
        ),
        _context(
            "source-2",
            "industry",
            "industry",
            "report",
            content="Second source exact statement.",
            source_uri="https://two.example/report",
            source_tier="secondary",
        ),
    ]
    request = _request(
        sources,
        [_claim(source.content, "fact", [source.record_id]) for source in sources],
    )

    output = _run(OperatorRole.INDUSTRY, request)

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert any(
        error.startswith("industry_single_non_primary_source:")
        for error in output.critic.errors
    )


def test_industry_official_fact_cannot_launder_unrelated_secondary_fact():
    official = _context(
        "source-official",
        "industry",
        "industry",
        "source_item",
        content="The regulator published the final rule.",
        source_uri="https://regulator.example/final-rule",
        source_tier="official",
    )
    secondary = _context(
        "source-rumour",
        "industry",
        "industry",
        "source_item",
        content="An unnamed source predicts a market shutdown.",
        source_uri="https://rumour.example/story",
        source_tier="secondary",
    )

    output = _run(
        OperatorRole.INDUSTRY,
        _request(
            [official, secondary],
            [
                _claim(official.content, "fact", [official.record_id]),
                _claim(secondary.content, "fact", [secondary.record_id]),
            ],
        ),
    )

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert any(
        error.startswith("industry_single_non_primary_source:")
        for error in output.critic.errors
    )


def test_model_prompt_exposes_only_approved_evidence_refs():
    exact_fact = "Two independent publishers report the same exact statement."
    sources = [
        _context(
            "source-1",
            "industry",
            "industry",
            "source_item",
            content=exact_fact,
            source_uri="https://one.example/story",
            source_tier="trusted",
        ),
        _context(
            "source-2",
            "industry",
            "industry",
            "report",
            content=exact_fact,
            source_uri="https://two.example/report",
            source_tier="secondary",
        ),
    ]
    request = _request(
        sources,
        [_claim(exact_fact, "fact", [source.record_id for source in sources])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, sources, AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    assert payload["canonical_block_registry"]
    assert all(
        block["origin"] == "approved_claim"
        for block in payload["canonical_block_registry"]
    )
    assert all(
        block["required"] is True
        for block in payload["canonical_block_registry"]
    )
    assert not any(
        block.get("origin") == "server_template"
        for block in payload["canonical_block_registry"]
    )
    assert any(
        block.get("origin") == "server_template"
        for block in fallback["blocks"]
    )


def test_model_may_select_canonical_blocks_with_bare_hash_refs():
    source = _context(
        "source-official",
        "industry",
        "industry",
        "source_item",
        content="The regulator published the official decision.",
        source_uri="https://official.example/decision",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    required_refs = [
        block["block_ref"]
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]
    candidate = {
        "_model": "bare-ref-json-model",
        "blocks": list(required_refs),
        "platform_variants": [
            {
                "platform": channel,
                "blocks": list(required_refs),
            }
            for channel in request.channels
        ],
    }

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert source.content in (output.master_content or "")
    assert all(
        {
            block.binding_hash
            for block in variant.blocks
            if block.required
        }
        == set(required_refs)
        for variant in output.platform_variants
    )


@pytest.mark.parametrize(
    "unknown_block",
    [
        "invented prose must not render",
        {"block_ref": "invented prose must not render"},
    ],
)
def test_unknown_block_ref_is_discarded_and_required_claim_is_restored(
    unknown_block,
):
    source = _context(
        "source-official",
        "industry",
        "industry",
        "source_item",
        content="The regulator published the official decision.",
        source_uri="https://official.example/decision",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = {
        "_model": "unsafe-bare-ref-model",
        "blocks": [deepcopy(unknown_block)],
        "platform_variants": [
            {
                "platform": channel,
                "blocks": [deepcopy(unknown_block)],
            }
            for channel in request.channels
        ],
    }

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert source.content in (output.master_content or "")
    assert "invented prose must not render" not in output.model_dump_json()
    assert any(
        warning.startswith("discarded_unknown_model_block_ref")
        for warning in output.critic.warnings
    )
    assert any(
        warning.startswith("model_required_blocks_restored")
        for warning in output.critic.warnings
    )


def test_industry_same_publisher_on_two_domains_is_not_independent():
    exact_fact = "The same publisher repeated one exact statement."
    sources = [
        _context(
            "source-1",
            "industry",
            "industry",
            "source_item",
            content=exact_fact,
            source_uri="https://news-one.example/story",
            source_tier="trusted",
            structured_data={"publisher_id": "publisher-one"},
        ),
        _context(
            "source-2",
            "industry",
            "industry",
            "report",
            content=exact_fact,
            source_uri="https://news-two.example/report",
            source_tier="secondary",
            structured_data={"publisher_id": "publisher-one"},
        ),
    ]

    output = _run(
        OperatorRole.INDUSTRY,
        _request(
            sources,
            [_claim(exact_fact, "fact", [source.record_id for source in sources])],
        ),
    )

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert any(
        error.startswith("industry_single_non_primary_source:")
        for error in output.critic.errors
    )


@pytest.mark.parametrize("kind", ["opinion", "transition", "cta"])
def test_model_cannot_smuggle_an_unsupported_fact_through_a_non_fact_label(kind):
    source = _context(
        "official-safe",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    candidate = deepcopy(fallback)
    unsupported = "某公司已经秘密收购竞争对手"
    candidate["blocks"].append(
        {"kind": kind, "text": unsupported, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert unsupported not in output.model_dump_json()
    assert any(
        warning.startswith("discarded_unsafe_model_editorial")
        for warning in output.critic.warnings
    )


def test_plausible_transition_label_cannot_smuggle_a_model_authored_claim():
    source = _context(
        "official-transition-attack",
        "industry",
        "industry",
        "source_item",
        content="OpenAI published an official product update.",
        source_uri="https://openai.com/news/",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    attack = "以下值得关注：OpenAI 是全球最领先的公司。"
    candidate["blocks"].append(
        {"kind": "transition", "text": attack, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert attack not in output.model_dump_json()
    assert any(
        warning.startswith("discarded_unsafe_model_editorial:master")
        for warning in output.critic.warnings
    )


def test_industry_model_fact_outside_approved_claims_blocks_the_entire_draft():
    source = _context(
        "official-1",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    candidate = deepcopy(fallback)
    candidate["_model"] = "unsafe-test-model"
    candidate["blocks"].append(
        {"kind": "fact", "text": "Invented market result.", "evidence_ids": [source.record_id]}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_content is None
    assert "Invented market result" not in output.model_dump_json()
    assert any(error.startswith("model_block_ref_required") for error in output.critic.errors)


def test_master_and_variant_bodies_are_deterministically_rendered_from_safe_blocks():
    source = _context(
        "official-1",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["_model"] = "body-injection-test-model"
    candidate["master_content"] = "Invented text outside structured blocks."
    candidate["platform_variants"][0]["body"] = "Invented variant body."

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert "Invented" not in (output.master_content or "")
    assert all("Invented" not in variant.body for variant in output.platform_variants)
    assert output.master_content == "\n\n".join(block.text for block in output.blocks)
    assert all(
        variant.body == "\n\n".join(block.text for block in variant.blocks)
        for variant in output.platform_variants
    )


@pytest.mark.parametrize(
    ("tamper", "error_prefix"),
    [
        ("text", "tampered_model_block_ref"),
        ("evidence", "tampered_model_block_ref"),
    ],
)
def test_canonical_block_ref_tampering_fails_closed(tamper, error_prefix):
    source = _context(
        "official-ref",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    if tamper == "text":
        candidate["blocks"][0]["text"] = "Forged model text."
    else:
        candidate["blocks"][0]["evidence_ids"] = ["forged-evidence"]

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_content is None
    assert "Forged model text" not in output.model_dump_json()
    assert "forged-evidence" not in output.model_dump_json()
    assert any(
        error.startswith(error_prefix) for error in output.critic.errors
    )


def test_model_ref_order_and_server_templates_create_distinct_channel_drafts():
    source = _context(
        "cap-structured",
        "commercial",
        "company_public",
        "capability",
        content="已批准能力：把内容任务交接到人工终审。",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.COMMERCIAL, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.COMMERCIAL,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.COMMERCIAL,
        request,
        authorized,
    )
    required_ref = next(
        block["block_ref"]
        for block in payload["canonical_block_registry"]
        if block["required"]
    )
    transition = next(
        block
        for block in fallback["blocks"]
        if block["kind"] == "transition" and block["origin"] == "server_template"
    )
    cta = next(
        block
        for block in fallback["platform_variants"][0]["blocks"]
        if block["kind"] == "cta" and block["origin"] == "server_template"
    )
    candidate = {
        "_model": "structured-editorial-model",
        "blocks": [
            {"block_ref": required_ref},
            {"block_ref": transition["binding_hash"]},
        ],
        "platform_variants": [
            {
                "platform": "wechat_mp",
                "blocks": [
                    {"block_ref": required_ref},
                    {"block_ref": cta["binding_hash"]},
                ],
            },
            {
                "platform": "xiaohongshu",
                "blocks": [
                    {"block_ref": transition["binding_hash"]},
                    {"block_ref": required_ref},
                ],
            },
        ],
    }

    output = normalize_content_output(
        registry,
        OperatorRole.COMMERCIAL,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert transition["text"] in (output.master_content or "")
    assert output.platform_variants[0].body != output.platform_variants[1].body
    assert output.platform_variants[0].blocks[0].binding_hash == required_ref
    assert output.platform_variants[1].blocks[-1].binding_hash == required_ref
    assert all(
        any(block.required for block in variant.blocks)
        for variant in output.platform_variants
    )
    server_templates = [
        block
        for block in [
            *output.blocks,
            *(block for variant in output.platform_variants for block in variant.blocks),
        ]
        if block.origin == "server_template"
    ]
    assert server_templates
    assert all(block.locked is True for block in server_templates)
    assert all(block.required is False for block in server_templates)
    assert all(len(block.binding_hash) == 64 for block in server_templates)
    approved_claim = next(block for block in output.blocks if block.required)
    assert approved_claim.origin == "approved_claim"
    assert approved_claim.locked is True
    assert approved_claim.binding_hash == required_ref


def test_server_restores_required_claim_when_model_only_returns_safe_editorial():
    source = _context(
        "cap-required",
        "commercial",
        "company_public",
        "capability",
        content="Approved public capability.",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.COMMERCIAL, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.COMMERCIAL,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.COMMERCIAL,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["platform_variants"][0]["blocks"] = [
        {
            "kind": "transition",
            "text": "先看已核验事实，再讨论具体场景。",
            "evidence_ids": [],
        }
    ]

    output = normalize_content_output(
        registry,
        OperatorRole.COMMERCIAL,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.platform_variants
    assert source.content in output.platform_variants[0].body
    assert any(
        warning.startswith(
            "model_required_blocks_restored:variant-wechat_mp"
        )
        for warning in output.critic.warnings
    )


def test_reader_facing_editorial_label_is_rejected_as_meta_copy():
    source = _context(
        "official-opinion-gate",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["blocks"].append(
        {
            "kind": "opinion",
            "text": "编辑观点：这项变化值得持续观察。",
            "evidence_ids": [],
        }
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert "编辑观点：这项变化值得持续观察。" not in (output.master_content or "")
    assert any(
        warning.endswith(":generic_meta_copy_forbidden")
        for warning in output.critic.warnings
    )
    publishable = enforce_social_publishability(output)
    assert publishable.status == ContentStatus.QUALITY_INSUFFICIENT.value


@pytest.mark.parametrize(
    ("alias", "text", "expected_kind"),
    [
        ("hook", "A penalty puts both sides of the platform on the same ledger.", "transition"),
        ("analysis", "后续规则透明度将决定这次通报能否真正改变行业约束。", "opinion"),
        ("reader_question", "你更关注规则透明度，还是后续执行？", "cta"),
    ],
)
def test_common_model_editorial_kind_aliases_are_safely_normalized(
    alias,
    text,
    expected_kind,
):
    source = _context(
        "official-editorial-alias",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["blocks"].append(
        {"kind": alias, "text": text, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    block = next(block for block in output.blocks if block.text == text)
    assert block.kind == expected_kind
    assert block.origin == "model_editorial"


def test_exact_canonical_echoes_and_schema_placeholders_are_safely_normalized():
    source = _context(
        "official-canonical-echo",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
        channels=["wechat_mp"],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = deepcopy(fallback)
    candidate["_model"] = "canonical-echo-test-model"
    placeholder = {
        "one_of": [
            {"block_ref": "one exact block_ref from canonical_block_registry"},
            {"kind": "opinion | transition | cta", "text": "editorial"},
        ]
    }
    candidate["blocks"].insert(0, placeholder)
    candidate["platform_variants"][0]["blocks"].insert(0, deepcopy(placeholder))

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert source.content in output.master_content
    assert any(
        warning.startswith("discarded_untyped_model_block:master-1")
        for warning in output.critic.warnings
    )
    assert any(
        warning.startswith("model_canonical_echo_normalized:master-")
        for warning in output.critic.warnings
    )


def test_editorial_type_and_content_field_aliases_are_safely_normalized():
    source = _context(
        "official-editorial-fields",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    editorial = "一张罚单，把平台和经营者拉回了同一张账本；扣款落到谁身上，话语权也落到谁身上。"
    candidate["blocks"].append(
        {"type": "analysis", "content": editorial, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    block = next(block for block in output.blocks if block.text == editorial)
    assert block.kind == ContentBlockKind.OPINION.value
    assert block.origin == "model_editorial"


def test_long_verified_fact_is_rendered_as_platform_native_excerpt_without_rewriting_binding():
    fact = (
        "中国国家市场监督管理总局通报，依据相关规定，对某平台作出行政处罚，"
        "没收违法所得16.58亿元，并处以罚款35.21亿元，罚没款合计51.79亿元。"
        "同时，责令其退还相关款项，要求企业全面整改并公开整改措施。"
    )
    source = _context(
        "official-long-fact",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/release",
        source_tier="official",
    )

    output = _run(
        OperatorRole.INDUSTRY,
        _request([source], [_claim(fact, "fact", [source.record_id])]),
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    exact = next(block for block in output.blocks if block.source_exact)
    assert exact.text == fact
    assert exact.evidence_ids == [source.record_id]
    assert "中国国家市场监督管理总局通报；" in output.master_content
    assert "依据相关规定" not in output.master_content
    assert "；没收违法所得16.58亿元；并处以罚款35.21亿元；罚没款合计51.79亿元；" in output.master_content
    assert "；责令其退还相关款项；" in output.master_content
    short = next(
        variant for variant in output.platform_variants
        if variant.platform == "xiaohongshu"
    )
    assert "对某平台作出行政处罚；" in short.body
    assert "中国国家市场监督管理总局通报" not in short.body
    assert "要求企业全面整改并公开整改措施" not in short.body


def test_industry_model_can_supply_grounded_social_titles_per_platform():
    fact = (
        "监管部门对某平台作出行政处罚，罚没款合计51.79亿元，"
        "并要求企业全面整改。"
    )
    source = _context(
        "official-social-title",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(fact, "fact", [source.record_id])],
        channels=["wechat_mp", "xiaohongshu"],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["master_title"] = "51.79亿元之后，整改才是真正看点"
    candidate["platform_variants"][0]["title"] = "51.79亿元之后，整改如何落地"
    candidate["platform_variants"][1]["title"] = "51.79亿元罚没后，整改要看执行动作"

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.master_title == "51.79亿元之后，整改才是真正看点"
    assert {
        item.platform: item.title for item in output.platform_variants
    } == {
        "wechat_mp": "51.79亿元之后，整改如何落地",
        "xiaohongshu": "51.79亿元罚没后，整改要看执行动作",
    }


def test_safe_approved_proposal_title_beats_a_legalistic_source_excerpt():
    fact = (
        "监管部门对某平台作出行政处罚，罚没款合计51.79亿元，"
        "并责令其退还酒店经营者订单储备金1.22亿元。"
    )
    source = _context(
        "official-approved-title",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/approved-title",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    request = request.model_copy(
        update={
            "approved_proposal": request.approved_proposal.model_copy(
                update={
                    "title": "51.79亿元罚没之外，酒店经营者拿回了什么"
                }
            )
        }
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.master_title == "51.79亿元罚没之外，酒店经营者拿回了什么"


def test_industry_authored_hook_leads_even_when_model_labels_it_opinion():
    fact = "监管部门责令某平台退还酒店经营者订单储备金1.22亿元。"
    source = _context(
        "official-hook-order",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/hook-order",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    hook = "监管处罚如何改变平台与酒店经营者之间的资金边界？"
    analysis = "退款要求把酒店经营者承受的资金约束，拉回到可核验的监管框架内。"
    candidate["blocks"].extend(
        [
            {"kind": "opinion", "text": hook, "evidence_ids": []},
            {"kind": "opinion", "text": analysis, "evidence_ids": []},
        ]
    )
    for variant in candidate["platform_variants"]:
        variant["blocks"].extend(
            [{"kind": "opinion", "text": hook, "evidence_ids": []}]
        )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.blocks[0].text == hook
    assert all(variant.blocks[0].text == hook for variant in output.platform_variants)


def test_personal_model_can_supply_grounded_titles_without_entities_or_numbers():
    experience = (
        "在这次端到端验收中，我把自动发布改成了人工终审。"
        "真正费时间的不是多点一次确认，而是在错误内容发出去之后补救。"
    )
    card = _context(
        "personal-grounded-title",
        "personal_ip",
        "personal_approved",
        "experience",
        content=experience,
    )
    request = _request(
        [card],
        [_claim(experience, "experience", [card.record_id])],
        channels=["wechat_mp", "xiaohongshu"],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.PERSONAL_IP, [card], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["master_title"] = "从自动发布到人工终审：我更看重补救成本"
    candidate["platform_variants"][0]["title"] = "我把自动发布改成了人工终审"
    candidate["platform_variants"][1]["title"] = "错误内容发出去，补救才最费时间"

    output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.master_title == "从自动发布到人工终审：我更看重补救成本"
    assert {
        item.platform: item.title for item in output.platform_variants
    } == {
        "wechat_mp": "我把自动发布改成了人工终审",
        "xiaohongshu": "错误内容发出去，补救才最费时间",
    }

    unsafe = deepcopy(candidate)
    unsafe["master_title"] = "为什么我把自动发布改成了人工终审"
    unsafe_output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        unsafe,
    )
    assert unsafe_output.master_title != unsafe["master_title"]


def test_industry_editorial_keeps_all_numbers_in_canonical_blocks_and_drops_meta_copy():
    fact = "监管部门作出行政处罚，罚没款合计51.79亿元，并要求企业全面整改。"
    source = _context(
        "official-grounded-number",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    grounded = "51.79亿元罚没款之后，更关键的问题是整改能否真正改变相关规则。"
    meta = "先把这条消息里的事实和判断分开。"
    filler = "对关注平台经济和反垄断的人来说，重点不只是金额，更是监管定性与整改要求。"
    unpaired_contrast = "51.79亿元罚没之后，不只是金额。"
    contrast = "这次处罚真正改变的，不只是一个数字，而是平台与经营者之间的规则。"
    not_but = "这次处罚真正改变的，不是一个数字，而是平台与经营者之间的规则。"
    vague = "这次最该被看见的，不只是51.79亿元，而是后续执行。"
    pressure = "压力不只在账面金额，更在平台与经营者之间的谈判位置。"
    not_in_but_in = "最有分量的变化不在表态，而在后续规则如何执行。"
    cannot_only_watch = "判断整改效果不能只看公开表态，还要看执行结果。"
    dont_only_watch = "以后看这类治理，别只看处罚金额，先看钱有没有回到经营者手里。"
    paired_if = "如果整改停在纸面，约束有限；如果进入日常执行，议价关系不同。"
    more_only = "平台把资金边界拿得越紧，后面的治理就越难只靠口头整改过去。"
    worth_noticing = "最值得注意的是平台与经营者之间的议价关系。"
    next_watch = "接下来最值得关注的是整改措施如何落地。"
    real_landing = "真正落点是平台与经营者之间的规则。"
    core_impact = "核心影响是在平台与经营者之间重新分配谈判空间。"
    most_direct_change = "最直接的变化，是酒店经营者不必再承担这笔资金占用。"
    directly_felt = "酒店经营者最直接感受到的变化，是现金不再被平台长期锁住。"
    next_focus = "接下来盯住一件事：平台如何公开整改措施。"
    later_watch = "后面要看的是平台会不会调整合同条款。"
    followup_point = "后续看点是平台规则能否变得透明。"
    business_landing = "这类处罚真正落到业务上，会改变平台与酒店的议价关系。"
    governance_landing = "这类治理真正落到经营端，看的就是资金能不能回到商家手里。"
    truly_changes = "携程这次被罚没，真正改变的是平台与酒店经营者之间的议价边界。"
    truly_must_change = "对酒店经营者而言，真正要变的是资金占用和结算方式。"
    pushed_to_front = "平台和酒店之间的规则边界被推到台前。"
    placed_on_stage = "平台内部规则能否继续占用商家资金的问题被摆到台前。"
    reexamined = "平台对酒店经营者的规则定价权会被重新审视。"
    mixed_language = "这会迫使相关参与者重新评估 bargaining position。"
    loaded_label = "平台不能再把灰色扣费当成默认规则。"
    invented = "52亿元罚没款之后，更关键的问题是整改能否真正改变相关规则。"
    invented_chinese_quantity = "企业将落实十九项整改措施。"
    candidate["blocks"].extend(
        [
            {"kind": "transition", "text": grounded, "evidence_ids": []},
            {"kind": "transition", "text": meta, "evidence_ids": []},
            {"kind": "opinion", "text": filler, "evidence_ids": []},
            {"kind": "opinion", "text": unpaired_contrast, "evidence_ids": []},
            {"kind": "opinion", "text": contrast, "evidence_ids": []},
            {"kind": "opinion", "text": not_but, "evidence_ids": []},
            {"kind": "opinion", "text": vague, "evidence_ids": []},
            {"kind": "opinion", "text": pressure, "evidence_ids": []},
            {"kind": "opinion", "text": not_in_but_in, "evidence_ids": []},
            {"kind": "opinion", "text": cannot_only_watch, "evidence_ids": []},
            {"kind": "opinion", "text": dont_only_watch, "evidence_ids": []},
            {"kind": "opinion", "text": paired_if, "evidence_ids": []},
            {"kind": "opinion", "text": more_only, "evidence_ids": []},
            {"kind": "opinion", "text": worth_noticing, "evidence_ids": []},
            {"kind": "opinion", "text": next_watch, "evidence_ids": []},
            {"kind": "opinion", "text": real_landing, "evidence_ids": []},
            {"kind": "opinion", "text": core_impact, "evidence_ids": []},
            {"kind": "opinion", "text": most_direct_change, "evidence_ids": []},
            {"kind": "opinion", "text": directly_felt, "evidence_ids": []},
            {"kind": "opinion", "text": next_focus, "evidence_ids": []},
            {"kind": "opinion", "text": later_watch, "evidence_ids": []},
            {"kind": "opinion", "text": followup_point, "evidence_ids": []},
            {"kind": "opinion", "text": business_landing, "evidence_ids": []},
            {"kind": "opinion", "text": governance_landing, "evidence_ids": []},
            {"kind": "opinion", "text": truly_changes, "evidence_ids": []},
            {"kind": "opinion", "text": truly_must_change, "evidence_ids": []},
            {"kind": "opinion", "text": pushed_to_front, "evidence_ids": []},
            {"kind": "opinion", "text": placed_on_stage, "evidence_ids": []},
            {"kind": "opinion", "text": reexamined, "evidence_ids": []},
            {"kind": "opinion", "text": mixed_language, "evidence_ids": []},
            {"kind": "opinion", "text": loaded_label, "evidence_ids": []},
            {"kind": "transition", "text": invented, "evidence_ids": []},
            {
                "kind": "opinion",
                "text": invented_chinese_quantity,
                "evidence_ids": [],
            },
        ]
    )
    candidate["master_title"] = "51.79亿元罚没之后，不只是金额"

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert grounded not in output.master_content
    assert meta not in output.master_content
    assert filler not in output.master_content
    assert unpaired_contrast not in output.master_content
    assert contrast not in output.master_content
    assert not_but not in output.master_content
    assert vague not in output.master_content
    assert pressure not in output.master_content
    assert not_in_but_in not in output.master_content
    assert cannot_only_watch not in output.master_content
    assert dont_only_watch not in output.master_content
    assert paired_if not in output.master_content
    assert more_only not in output.master_content
    assert worth_noticing not in output.master_content
    assert next_watch not in output.master_content
    assert real_landing not in output.master_content
    assert core_impact not in output.master_content
    assert most_direct_change not in output.master_content
    assert directly_felt not in output.master_content
    assert next_focus not in output.master_content
    assert later_watch not in output.master_content
    assert followup_point not in output.master_content
    assert business_landing not in output.master_content
    assert governance_landing not in output.master_content
    assert truly_changes not in output.master_content
    assert truly_must_change not in output.master_content
    assert pushed_to_front not in output.master_content
    assert placed_on_stage not in output.master_content
    assert reexamined not in output.master_content
    assert mixed_language not in output.master_content
    assert loaded_label not in output.master_content
    assert invented not in output.master_content
    assert invented_chinese_quantity not in output.master_content
    assert sum(
        warning.endswith(":generic_meta_copy_forbidden")
        for warning in output.critic.warnings
    ) == 28
    assert output.master_title != candidate["master_title"]
    assert any(
        warning.endswith(":mixed_language_phrase_forbidden")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":loaded_editorial_label_forbidden")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":number_ungrounded")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":number_context_forbidden")
        for warning in output.critic.warnings
    )
    publishable = enforce_social_publishability(output)
    assert publishable.status == ContentStatus.QUALITY_INSUFFICIENT.value
    assert publishable.master_content is None
    assert publishable.platform_variants == []
    assert any(
        error.startswith("social_editorial_")
        for error in publishable.critic.errors
    )


def test_industry_editorial_cannot_repurpose_a_grounded_date_as_a_count():
    fact = "监管部门于25日通报处罚决定，并要求平台退还经营者资金。"
    source = _context(
        "official-number-context",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/number-context",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry, OperatorRole.INDUSTRY, request, authorized
    )
    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    candidate = _candidate_from_required_refs(payload, request.channels)
    repurposed = "25个相关项目被放进同一张比较表里，经营者的选择成本变了。"
    candidate["blocks"].insert(
        0,
        {"kind": "opinion", "text": repurposed, "evidence_ids": []},
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert repurposed not in (output.master_content or "")
    assert any(
        warning.endswith((":number_context_forbidden", ":number_ungrounded"))
        for warning in output.critic.warnings
    )


def test_industry_funds_remedy_cannot_be_reframed_as_promotion_or_reversed_cash_flow():
    fact = "监管要求平台全额退还强制扣除酒店经营者的订单储备金，并公开整改措施。"
    source = _context(
        "official-funds-remedy",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/funds-remedy",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry, OperatorRole.INDUSTRY, request, authorized
    )
    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    candidate = _candidate_from_required_refs(payload, request.channels)
    promotion = "免单会把流量提前释放出来，商家要想办法沉淀复购。"
    reversed_cash_flow = "酒店经营者先要面对回款节奏变紧和现金流压力。"
    event_restatement = (
        "平台把酒店经营者的订单储备金拿去内部占用，资金边界随之改变。"
    )
    unsupported_result = (
        "对酒店经营者而言，现金更充足，后续谈结算时也更有底气。"
    )
    candidate["blocks"] = [
        {"kind": "opinion", "text": promotion, "evidence_ids": []},
        {
            "kind": "transition",
            "text": event_restatement,
            "evidence_ids": [],
        },
        *candidate["blocks"],
        {
            "kind": "opinion",
            "text": reversed_cash_flow,
            "evidence_ids": [],
        },
        {
            "kind": "opinion",
            "text": unsupported_result,
            "evidence_ids": [],
        },
    ]

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert promotion not in (output.master_content or "")
    assert reversed_cash_flow not in (output.master_content or "")
    assert event_restatement not in (output.master_content or "")
    assert unsupported_result not in (output.master_content or "")
    assert any(
        warning.endswith(":industry_funds_remedy_reframing_forbidden")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":industry_funds_remedy_direction_forbidden")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":industry_funds_remedy_event_restatement_forbidden")
        for warning in output.critic.warnings
    )
    assert any(
        warning.endswith(":industry_funds_remedy_speculation_forbidden")
        for warning in output.critic.warnings
    )

    forced = output.model_copy(
        update={
            "master_title": (
                "携程集团有限公司滥用市场支配地位实施垄断行为作出行政处罚，"
                "没收违法所得并处以罚款，罚没款合计"
            )
        }
    )
    blocked = enforce_social_publishability(forced)
    assert blocked.status == ContentStatus.QUALITY_INSUFFICIENT.value
    assert any(
        error.startswith(("social_title_length_invalid", "social_title_legalese_forbidden"))
        for error in blocked.critic.errors
    )


@pytest.mark.parametrize(
    "cliche",
    [
        "这次处理最直接的一点，是平台的内部规则需要调整。",
        "最直观的变化，是经营者可以重新判断结算条件。",
        "这类规则不再只是商业安排，而是经营边界。",
        "对经营者来说，资金边界更清楚；对平台来说，规则成本更高。",
        "这类处理对行业的信号很明确，平台规则需要调整。",
        "结果很清楚，经营者需要重新判断谈判位置。",
    ],
)
def test_industry_current_social_cliches_are_discarded(cliche: str):
    fact = "监管部门公布平台整改要求，经营者可以核对相关规则。"
    source = _context(
        "official-current-cliche",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/current-cliche",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry, OperatorRole.INDUSTRY, request, authorized
    )
    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["blocks"].insert(
        0,
        {"kind": "transition", "text": cliche, "evidence_ids": []},
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert cliche not in (output.master_content or "")
    assert any(
        warning.endswith(":generic_meta_copy_forbidden")
        for warning in output.critic.warnings
    )


def test_model_shape_errors_are_quality_failures_that_can_be_retried():
    fact = "监管部门发布了经批准的行业规则。"
    source = _context(
        "official-model-shape",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/model-shape",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry, OperatorRole.INDUSTRY, request, authorized
    )
    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["platform_variants"].append(
        {"platform": None, "blocks": candidate["blocks"]}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.QUALITY_INSUFFICIENT.value
    assert "model_returned_unrequested_platform" in output.critic.errors


def test_industry_editorial_cannot_restate_the_approved_source_paragraph():
    fact = (
        "中国国家市场监督管理总局周日（25日）通报，依据《中华人民共和国反垄断法》相关规定，"
        "对携程集团有限公司滥用市场支配地位实施垄断行为作出行政处罚，没收违法所得16.58亿元，"
        "并处以罚款35.21亿元，罚没款合计51.79亿元。同时，责令其全额退还强制扣除酒店经营者的"
        "订单储备金1.22亿元，要求企业全面整改并公开整改措施。"
    )
    source = _context(
        "official-source-restatement",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request([source], [_claim(fact, "fact", [source.record_id])])
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    restatement = (
        "携程集团有限公司因滥用市场支配地位实施垄断行为，被作出行政处罚，"
        "没收违法所得16.58亿元，并处以罚款35.21亿元，罚没款合计51.79亿元。"
    )
    candidate["blocks"].append(
        {"kind": "transition", "text": restatement, "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert restatement not in output.master_content
    assert any(
        warning.endswith(":source_restatement_forbidden")
        for warning in output.critic.warnings
    )


def test_terminal_social_gate_caps_saturated_diagnostics_and_keeps_blockers_first():
    fact = "监管部门作出行政处罚，并要求企业全面整改。"
    source = _context(
        "official-risk-budget",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/risk-budget",
        source_tier="official",
    )
    output = _run(
        OperatorRole.INDUSTRY,
        _request([source], [_claim(fact, "fact", [source.record_id])]),
    )
    saturated = output.model_copy(
        update={
            "risk_flags": [
                OperatorRisk(
                    code=f"warning_{index}",
                    message=f"warning {index}",
                    blocking=False,
                )
                for index in range(50)
            ]
        }
    )

    publishable = enforce_social_publishability(saturated)

    assert publishable.status == ContentStatus.QUALITY_INSUFFICIENT.value
    assert len(publishable.risk_flags) == 50
    assert publishable.risk_flags[0].blocking is True
    assert publishable.risk_flags[0].code in {
        "model_generation_required",
        "social_editorial_template_forbidden",
        "social_editorial_depth_insufficient",
        "social_editorial_hook_missing",
        "social_editorial_analysis_insufficient",
    }


def test_terminal_social_gate_accepts_deep_distinct_model_editorial():
    fact = "The regulator published an approved platform rule update."
    source = _context(
        "official-social-depth",
        "industry",
        "industry",
        "source_item",
        content=fact,
        source_uri="https://official.example/rule-update",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(fact, "fact", [source.record_id])],
        channels=["wechat_mp", "xiaohongshu"],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.INDUSTRY,
        request,
        authorized,
    )
    required = [
        {"block_ref": block["block_ref"]}
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]

    def editorial(label: str, count: int = 3) -> list[dict]:
        return [
            {
                "kind": "transition",
                "text": (
                    f"First, {label} should focus on the concrete tension "
                    "already present in the approved facts."
                ),
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": (
                    f"What matters for {label} is how that tension changes "
                    "the reader's available choice."
                ),
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": (
                    f"The implication for {label} is to compare the rule, "
                    "the cost, and the observable outcome."
                ),
                "evidence_ids": [],
            },
        ][:count]

    candidate = {
        "_model": "publishable-test-model",
        "blocks": [*deepcopy(required), *editorial("the master story", 2)],
        "platform_variants": [
            {
                "platform": channel,
                "blocks": [
                    *deepcopy(required),
                    *editorial(
                        channel,
                        2 if channel == "wechat_mp" else 1,
                    ),
                ],
            }
            for channel in request.channels
        ],
    }
    normalized = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    publishable = enforce_social_publishability(normalized)

    assert publishable.status == ContentStatus.CONTENT_READY.value
    assert publishable.critic.passed is True
    assert publishable.master_content
    assert len(publishable.platform_variants) == 2
    assert next(
        check
        for check in publishable.critic.checks
        if check.code == "social_copy_publishable"
    ).passed is True

    # The reader-facing structure stays valid even when the model labels every
    # authored block as a transition.  Safety is still enforced block by block;
    # quality must not depend on model-authored metadata being semantically exact.
    swapped = deepcopy(candidate)
    for block in swapped["blocks"]:
        if "block_ref" not in block:
            block["kind"] = "transition"
    for variant in swapped["platform_variants"]:
        for block in variant["blocks"]:
            if "block_ref" not in block:
                block["kind"] = "transition"
    relabelled = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        swapped,
    )
    relabelled_publishable = enforce_social_publishability(relabelled)
    assert relabelled_publishable.status == ContentStatus.CONTENT_READY.value, (
        relabelled_publishable.critic.errors,
        relabelled_publishable.critic.warnings,
    )


def test_personal_background_fact_cannot_replace_an_approved_personal_card():
    industry = _context(
        "industry-1",
        "personal_ip",
        "industry",
        "source_fact",
        content="An industry event happened.",
        source_uri="https://official.example/event",
        source_tier="official",
    )
    request = _request(
        [industry],
        [_claim(industry.content, "fact", [industry.record_id])],
    )

    output = _run(OperatorRole.PERSONAL_IP, request)

    assert output.status == ContentStatus.NEEDS_INPUT.value
    assert output.master_content is None
    assert "approved_personal_card_required" in output.critic.errors


def test_unrelated_personal_card_cannot_launder_external_first_person_fact():
    external = _context(
        "external-biography",
        "personal_ip",
        "industry",
        "source_fact",
        content="我曾带领该项目完成出海。",
        source_uri="https://publisher.example.test/profile",
        source_tier="official",
    )
    unrelated_card = _context(
        "unrelated-card",
        "personal_ip",
        "personal_approved",
        "identity_fact",
        content="本人姓名为测试用户。",
    )
    request = _request(
        [external, unrelated_card],
        [
            _claim(external.content, "fact", [external.record_id]),
            _claim(unrelated_card.content, "identity", [unrelated_card.record_id]),
        ],
    )

    output = _run(OperatorRole.PERSONAL_IP, request)

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_content is None
    assert "personal_attribution_without_approved_card:claim-1" in output.critic.errors


def test_personal_first_person_fact_label_is_safe_when_exclusively_card_backed():
    card = _context(
        "experience-card-as-fact",
        "personal_ip",
        "personal_approved",
        "experience",
        content="我曾带领该项目完成出海。",
    )
    request = _request(
        [card],
        [_claim(card.content, "fact", [card.record_id])],
    )

    output = _run(OperatorRole.PERSONAL_IP, request)

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.blocks[0].kind == ContentBlockKind.FACT.value
    assert output.blocks[0].evidence_ids == [card.record_id]


def test_personal_approved_viewpoint_stays_verbatim_and_keeps_card_evidence():
    viewpoint = _context(
        "opinion-card-1",
        "personal_ip",
        "personal_approved",
        "opinion",
        content="我更看重可持续的客户价值，而不是短期流量。",
    )
    request = _request(
        [viewpoint],
        [_claim(viewpoint.content, "opinion", [viewpoint.record_id])],
    )

    output = _run(OperatorRole.PERSONAL_IP, request)

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    block = next(block for block in output.blocks if block.required)
    assert block.kind == ContentBlockKind.OPINION.value
    assert block.text == viewpoint.content
    assert block.evidence_ids == [viewpoint.record_id]
    assert block.verification_status == "verified"
    assert block.source_exact is True
    assert all(viewpoint.content in variant.body for variant in output.platform_variants)


def test_personal_model_editorial_can_add_safe_first_person_judgment():
    experience = _context(
        "experience-with-safe-editorial",
        "personal_ip",
        "personal_approved",
        "experience",
        content=(
            "在这次验收中，我把自动发布改成了人工终审。"
            "真正费时间的，是错误内容发出后的补救。"
        ),
    )
    request = _request(
        [experience],
        [_claim(experience.content, "experience", [experience.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.PERSONAL_IP,
        [experience],
        AS_OF,
    )
    fallback = build_content_fallback(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    assert "First-person present-tense judgment is allowed" in (
        payload["publication_brief"]["authored_block_rule"]
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["blocks"].extend(
        [
            {
                "kind": "transition",
                "text": "我越来越确定，内容运营最怕的不是慢一步，而是错一步。",
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": "我的判断是，人工确认不是流程负担，而是把补救成本提前变成一道选择题。",
                "evidence_ids": [],
            },
        ]
    )

    output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert any(block.text.startswith("我越来越确定") for block in output.blocks)
    assert any(block.text.startswith("我的判断是") for block in output.blocks)


def test_personal_model_editorial_cannot_invent_first_person_experience():
    experience = _context(
        "experience-with-invented-editorial",
        "personal_ip",
        "personal_approved",
        "experience",
        content="我选择让最终发布保留人工确认。",
    )
    request = _request(
        [experience],
        [_claim(experience.content, "experience", [experience.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.PERSONAL_IP,
        [experience],
        AS_OF,
    )
    fallback = build_content_fallback(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["blocks"].append(
        {
            "kind": "opinion",
            "text": "我服务过很多客户，所以这套方法对所有人都有效。",
            "evidence_ids": [],
        }
    )

    output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert all("我服务过很多客户" not in block.text for block in output.blocks)
    assert any(
        warning.endswith(":first_person_forbidden")
        for warning in output.critic.warnings
    )


def test_personal_server_template_refs_are_selectable_without_dropping_required_claims():
    experience = _context(
        "experience-template-ref",
        "personal_ip",
        "personal_approved",
        "experience",
        content="第一次带团队出海时，我暂停了未经验证的投放。",
    )
    request = _request(
        [experience],
        [_claim(experience.content, "experience", [experience.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(
        OperatorRole.PERSONAL_IP,
        [experience],
        AS_OF,
    )
    fallback = build_content_fallback(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    payload = content_request_payload(
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    required_refs = [
        block["block_ref"]
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]
    template_refs = [
        block["binding_hash"]
        for block in fallback["blocks"]
        if block["origin"] == "server_template"
    ]
    selected = [
        {"block_ref": block_ref}
        for block_ref in [*required_refs, *template_refs]
    ]
    candidate = {
        "_model": "structured-ref-test-model",
        "blocks": deepcopy(selected),
        "platform_variants": [
            {"platform": channel, "blocks": deepcopy(selected)}
            for channel in request.channels
        ],
    }

    output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert {
        block.text
        for block in output.blocks
        if block.origin == "server_template"
    } == {
        "先看已确认的经历与观点，再讨论其中的启发。",
        "欢迎分享你的观察与不同视角。",
    }
    assert all(
        {
            block.binding_hash
            for block in variant.blocks
            if block.required
        }
        == set(required_refs)
        for variant in output.platform_variants
    )


def test_personal_model_cannot_add_first_person_experience_or_stance():
    card = _context(
        "card-1",
        "personal_ip",
        "personal_approved",
        "experience",
        content="第一次带团队出海时，我暂停了未经验证的投放。",
    )
    request = _request(
        [card],
        [_claim(card.content, "experience", [card.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.PERSONAL_IP, [card], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.PERSONAL_IP, request, authorized)
    payload = content_request_payload(
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
    )
    candidate = _candidate_from_required_refs(payload, request.channels)
    candidate["_model"] = "unsafe-test-model"
    candidate["blocks"].append(
        {"kind": "transition", "text": "我赚到了从未记录的收入。", "evidence_ids": []}
    )

    output = normalize_content_output(
        registry,
        OperatorRole.PERSONAL_IP,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert "从未记录的收入" not in output.model_dump_json()
    assert any(
        warning.startswith("discarded_unsafe_model_editorial")
        for warning in output.critic.warnings
    )


def test_personal_boundary_card_is_never_disclosed_or_rendered():
    boundary = _context(
        "boundary-1",
        "personal_ip",
        "personal_approved",
        "boundary",
        content="绝不能公开的私人家庭细节。",
    )
    request = _request(
        [boundary],
        [_claim(boundary.content, "identity", [boundary.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.PERSONAL_IP, [boundary], AS_OF)
    payload = content_request_payload(OperatorRole.PERSONAL_IP, request, authorized)
    output = _run(OperatorRole.PERSONAL_IP, request)

    assert payload["authorized_context"] == []
    assert boundary.content not in str(payload)
    assert boundary.content not in output.model_dump_json()
    assert output.status == ContentStatus.NEEDS_INPUT.value


@pytest.mark.parametrize(
    "channels,error_prefix",
    [
        ([], "channels_required"),
        (["wechat_mp", "wechat_mp"], "duplicate_channels"),
        (["unknown_platform"], "unsupported_channels"),
    ],
)
def test_channel_contract_never_silently_drops_or_duplicates_variants(channels, error_prefix):
    source = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
        channels=channels,
    )

    output = _run(OperatorRole.INDUSTRY, request)

    assert output.status != ContentStatus.CONTENT_READY.value
    assert output.platform_variants == []
    assert any(error.startswith(error_prefix) for error in output.critic.errors)


def test_model_failure_returns_a_safe_deterministic_fallback_with_warning():
    source = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        content="Official exact statement.",
        source_uri="https://official.example/release",
        source_tier="official",
    )
    request = _request(
        [source],
        [_claim(source.content, "fact", [source.record_id])],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    candidate = {**deepcopy(fallback), "_error": "upstream timeout", "_model": "fallback"}

    output = normalize_content_output(
        registry,
        OperatorRole.INDUSTRY,
        request,
        authorized,
        fallback,
        candidate,
    )

    assert output.status == ContentStatus.CONTENT_READY.value
    assert output.critic.passed is True
    assert output.critic.warnings == ["model_unavailable_safe_fallback_used"]
    assert output.master_content and source.content in output.master_content
