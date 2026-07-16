from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from ai_marketing_api.operator_content import (
    ContentBlockKind,
    ContentStatus,
    OperatorComposeRequest,
    build_content_fallback,
    content_request_payload,
    disclosable_contexts,
    normalize_content_output,
)
from ai_marketing_api.operator_runtime import (
    AuthorizedContext,
    OperatorClaim,
    OperatorRegistry,
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


@pytest.mark.parametrize("role", list(OperatorRole))
def test_three_roles_compose_safe_fallback_with_exact_evidence_and_all_channels(role):
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
        claims = [
            _claim(context.content, "fact", [context.record_id]),
            _claim("编辑观点：这项变化值得持续观察。", "opinion", [], "opinion"),
        ]
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
    candidate = deepcopy(fallback)
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

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_content is None
    assert unsafe_text not in output.model_dump_json()
    assert any(error.startswith("unsafe_model_freeform") for error in output.critic.errors)


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


def test_industry_two_independent_sources_for_same_fact_pass_and_model_opinion_is_labelled():
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
    candidate = deepcopy(fallback)
    candidate["_model"] = "safe-test-model"
    candidate["blocks"].append(
        {"kind": "opinion", "text": "变化的长期影响仍需观察。", "evidence_ids": []}
    )
    candidate["platform_variants"][0]["blocks"].append(
        {"kind": "transition", "text": "先看已核验事实，再看后续影响。", "evidence_ids": []}
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
    assert output.critic.passed is True
    assert "变化的长期影响仍需观察" not in (output.master_content or "")
    assert "先看已核验事实，再看后续影响" not in output.platform_variants[0].body
    assert any(
        warning.startswith("discarded_unapproved_model_freeform")
        for warning in output.critic.warnings
    )
    exact = [block for block in output.blocks if block.source_exact]
    assert {block.text for block in exact} == {source.content for source in sources}


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
        warning.startswith("discarded_unapproved_model_freeform")
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
    assert any(error.startswith("unsupported_model_claim") for error in output.critic.errors)


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
    candidate = deepcopy(fallback)
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
    assert len(output.blocks) == 1
    block = output.blocks[0]
    assert block.kind == ContentBlockKind.OPINION.value
    assert block.text == viewpoint.content
    assert block.evidence_ids == [viewpoint.record_id]
    assert block.verification_status == "verified"
    assert block.source_exact is True
    assert all(viewpoint.content in variant.body for variant in output.platform_variants)


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
    candidate = deepcopy(fallback)
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

    assert output.status == ContentStatus.EVIDENCE_INSUFFICIENT.value
    assert output.master_content is None
    assert "从未记录的收入" not in output.model_dump_json()


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
