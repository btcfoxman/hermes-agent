from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_marketing_api.operator_runtime import (
    AuthorizedContext,
    ClaimKind,
    ContextAuthorizationError,
    OperatorAction,
    OperatorProposeRequest,
    OperatorRegistry,
    OperatorRole,
    OutputStatus,
    VerificationStatus,
    build_fallback,
    normalize_output,
)


AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)


def _context(
    record_id: str,
    role: str,
    space: str,
    record_type: str,
    *,
    content: str = "Approved source fact.",
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


def _request(objective: str = "Create today's proposal") -> OperatorProposeRequest:
    return OperatorProposeRequest(
        objective=objective,
        topic="AI operations",
        audience="business owners",
        channels=["wechat_mp", "xiaohongshu"],
        as_of=AS_OF,
    )


def _run(
    registry: OperatorRegistry,
    role: OperatorRole,
    request: OperatorProposeRequest,
    contexts: list[AuthorizedContext],
    candidate: dict | None = None,
):
    authorized = registry.authorize(role, contexts, request.as_of)
    fallback = build_fallback(role, OperatorAction.PROPOSE, request, authorized)
    return normalize_output(
        registry,
        role,
        OperatorAction.PROPOSE,
        request,
        authorized,
        fallback,
        candidate or fallback,
    )


def test_profiles_are_distinct_and_system_prompts_are_byte_stable():
    registry = OperatorRegistry()
    profiles = [registry.get(role) for role in OperatorRole]

    assert len({profile.profile_id for profile in profiles}) == len(profiles)
    assert len({profile.prompt_sha256 for profile in profiles}) == len(profiles)
    for profile in profiles:
        assert profile.allowed_tools == ()
        assert registry.get(profile.role_id).system_prompt is profile.system_prompt
        assert registry.get(profile.role_id).soul_bytes == profile.soul_bytes
        assert profile.prompt_sha256 == __import__("hashlib").sha256(profile.soul_bytes).hexdigest()


@pytest.mark.parametrize(
    "role,space,record_type",
    [
        (OperatorRole.COMMERCIAL, "personal_approved", "experience"),
        (OperatorRole.INDUSTRY, "company_internal", "business_offer"),
        (OperatorRole.PERSONAL_IP, "company_internal", "case_study"),
        (OperatorRole.PERSONAL_IP, "personal_private", "experience"),
    ],
)
def test_cross_role_knowledge_is_rejected(role, space, record_type):
    registry = OperatorRegistry()
    context = _context("forbidden", role.value, space, record_type)

    with pytest.raises(ContextAuthorizationError, match="cannot read knowledge space") as exc:
        registry.authorize(role, [context], AS_OF)

    assert exc.value.code == "knowledge_space_denied"


def test_explicit_role_grant_is_required_even_in_an_allowed_space():
    registry = OperatorRegistry()
    context = _context("industry-1", "personal_ip", "industry", "source_fact")

    with pytest.raises(ContextAuthorizationError) as exc:
        registry.authorize(OperatorRole.INDUSTRY, [context], AS_OF)

    assert exc.value.code == "role_not_granted"


@pytest.mark.parametrize("role", [OperatorRole.INDUSTRY, OperatorRole.PERSONAL_IP])
def test_non_commercial_profiles_cannot_read_public_offer_records(role):
    registry = OperatorRegistry()
    context = _context("offer", role.value, "company_public", "business_offer")

    with pytest.raises(ContextAuthorizationError) as exc:
        registry.authorize(role, [context], AS_OF)

    assert exc.value.code == "record_type_denied"


def test_expired_context_fails_closed():
    registry = OperatorRegistry()
    context = _context("old", "commercial", "company_public", "product")
    context.valid_until = datetime(2026, 7, 1, tzinfo=timezone.utc)

    with pytest.raises(ContextAuthorizationError) as exc:
        registry.authorize(OperatorRole.COMMERCIAL, [context], AS_OF)

    assert exc.value.code == "record_expired"


def test_empty_authorized_context_fails_closed_before_model_execution():
    registry = OperatorRegistry()
    context = _context(
        "empty",
        "industry",
        "industry",
        "source_item",
        content="",
        source_uri="https://source.test/empty",
    )

    with pytest.raises(ContextAuthorizationError) as exc:
        registry.authorize(OperatorRole.INDUSTRY, [context], AS_OF)

    assert exc.value.code == "empty_context"


def test_commercial_price_intent_without_unique_offer_requests_input():
    output = _run(OperatorRegistry(), OperatorRole.COMMERCIAL, _request("发布新报价和折扣"), [])

    assert output.status == OutputStatus.NEEDS_INPUT.value
    assert "missing_unique_active_offer" in {risk.code for risk in output.risk_flags}
    assert output.questions
    assert all("¥99" not in claim.text for claim in output.claims)


def test_commercial_offer_text_is_copied_from_the_authorized_record():
    offer = _context(
        "offer-1",
        "commercial",
        "company_public",
        "business_offer",
        content="Do not use this internal-looking fallback text.",
        structured_data={
            "public_text": "标准公开方案：每月 3000 元，适用于已批准渠道。",
            "publicly_quoteable": True,
        },
    )
    output = _run(
        OperatorRegistry(),
        OperatorRole.COMMERCIAL,
        _request("说明当前标准报价"),
        [offer],
    )

    assert output.status == OutputStatus.PROPOSAL.value
    assert output.claims[0].text == "标准公开方案：每月 3000 元，适用于已批准渠道。"
    assert output.claims[0].evidence_ids == ["offer-1"]


def test_commercial_non_public_offer_is_never_quoted():
    offer = _context(
        "internal-offer",
        "commercial",
        "company_public",
        "business_offer",
        content="Internal floor price: ¥1.",
        structured_data={"public_text": "¥1", "publicly_quoteable": False},
    )
    output = _run(
        OperatorRegistry(),
        OperatorRole.COMMERCIAL,
        _request("说明当前报价"),
        [offer],
    )

    assert output.status == OutputStatus.NEEDS_INPUT.value
    assert output.claims == []
    assert "¥1" not in (output.model_dump_json() if hasattr(output, "model_dump_json") else output.json())


def test_commercial_model_cannot_invent_price_or_guarantee():
    capability = _context(
        "cap-1", "commercial", "company_public", "capability", content="Approved automation capability."
    )
    candidate = {
        "_model": "test-model",
        "proposal": {
            "title": "Guaranteed growth for only ¥99",
            "angle": "Risk-free results are guaranteed.",
            "audience_value": "Buy now.",
            "key_points": ["Guaranteed"],
            "suggested_formats": ["wechat_mp"],
            "cta": "Pay ¥99",
            "first_person": False,
        },
        "claims": [
            {
                "text": "The service costs ¥99.",
                "kind": "fact",
                "evidence_ids": ["cap-1"],
            }
        ],
    }
    output = _run(
        OperatorRegistry(), OperatorRole.COMMERCIAL, _request(), [capability], candidate
    )

    rendered = output.json() if hasattr(output, "json") else str(output)
    assert "¥99" not in rendered
    assert "Guaranteed growth" not in rendered
    assert output.claims[0].text == "Approved automation capability."


def test_industry_single_non_primary_source_is_blocked_and_opinion_is_separate():
    source = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        content="A vendor announced a new model.",
        source_uri="https://example.test/story",
        source_tier="secondary",
    )
    output = _run(OperatorRegistry(), OperatorRole.INDUSTRY, _request(), [source])

    assert output.status == OutputStatus.EVIDENCE_INSUFFICIENT.value
    assert "single_source_unverified" in {risk.code for risk in output.risk_flags}
    assert {claim.kind for claim in output.claims} == {ClaimKind.FACT.value, ClaimKind.OPINION.value}
    fact = next(claim for claim in output.claims if claim.kind == ClaimKind.FACT.value)
    assert fact.evidence_ids == ["source-1"]


def test_industry_two_independent_sources_can_form_a_proposal():
    contexts = [
        _context(
            "source-1",
            "industry",
            "industry",
            "source_item",
            source_uri="https://one.test/story",
        ),
        _context(
            "source-2",
            "industry",
            "industry",
            "report",
            source_uri="https://two.test/report",
        ),
    ]
    output = _run(OperatorRegistry(), OperatorRole.INDUSTRY, _request(), contexts)

    assert output.status == OutputStatus.PROPOSAL.value
    assert "single_source_unverified" not in {risk.code for risk in output.risk_flags}


def test_industry_model_fact_with_unknown_evidence_is_flagged():
    source = _context(
        "source-1",
        "industry",
        "industry",
        "source_item",
        source_uri="https://official.test/news",
        source_tier="official",
    )
    candidate = {
        "_model": "test-model",
        "claims": [
            {"text": "Unsupported market-size claim.", "kind": "fact", "evidence_ids": ["other-role"]},
            {"text": "Editorial interpretation.", "kind": "opinion", "evidence_ids": []},
        ],
    }
    output = _run(OperatorRegistry(), OperatorRole.INDUSTRY, _request(), [source], candidate)

    fact = next(claim for claim in output.claims if claim.kind == ClaimKind.FACT.value)
    assert fact.text == source.content
    assert fact.evidence_ids == ["source-1"]
    assert fact.verification_status == VerificationStatus.VERIFIED.value
    assert all(claim.text != "Unsupported market-size claim." for claim in output.claims)
    assert "unsupported_fact" in {risk.code for risk in output.risk_flags}


def test_personal_ip_without_approved_cards_returns_interview_questions():
    public_source = _context(
        "industry-1",
        "personal_ip",
        "industry",
        "source_fact",
        source_uri="https://official.test/news",
        source_tier="official",
    )
    output = _run(OperatorRegistry(), OperatorRole.PERSONAL_IP, _request(), [public_source])

    assert output.status == OutputStatus.NEEDS_INPUT.value
    assert output.proposal is None
    assert len(output.questions) >= 3
    assert output.claims == []
    assert "missing_approved_identity_cards" in {risk.code for risk in output.risk_flags}


def test_personal_ip_claims_are_exact_approved_card_text():
    card = _context(
        "card-1",
        "personal_ip",
        "personal_approved",
        "experience",
        content="2025 年第一次带团队出海时，我主动暂停了未经验证的投放。",
    )
    candidate = {
        "_model": "test-model",
        "proposal": {
            "title": "我靠一次投放赚了百万",
            "angle": "我取得了从未记录的巨大成功。",
            "audience_value": "我的经验保证成功。",
            "key_points": ["虚构经历"],
            "suggested_formats": ["wechat_moments"],
            "cta": None,
            "first_person": True,
        },
        "claims": [
            {"text": "我赚了百万。", "kind": "experience", "evidence_ids": ["card-1"]}
        ],
    }
    output = _run(OperatorRegistry(), OperatorRole.PERSONAL_IP, _request(), [card], candidate)

    assert [claim.text for claim in output.claims] == [card.content]
    assert output.claims[0].evidence_ids == ["card-1"]
    assert output.proposal is not None
    assert output.proposal.key_points == [card.content]
    assert output.proposal.title != "我靠一次投放赚了百万"


def test_personal_boundary_card_is_not_repeated_as_publishable_content():
    boundary = _context(
        "boundary-1",
        "personal_ip",
        "personal_approved",
        "boundary",
        content="Never publish this private family detail.",
    )
    output = _run(OperatorRegistry(), OperatorRole.PERSONAL_IP, _request(), [boundary])

    assert output.status == OutputStatus.NEEDS_INPUT.value
    assert output.claims == []
    assert "private family detail" not in (
        output.model_dump_json() if hasattr(output, "model_dump_json") else output.json()
    )
