from __future__ import annotations

import asyncio
import hashlib

import pytest

from ai_marketing_api.operator_content import (
    ContentBlock,
    ContentCritic,
    OperatorContentOutput,
    PlatformVariant,
)
from ai_marketing_api.operator_editorial import BoundedModelRun, RuntimeBudget
from ai_marketing_api.operator_grounding import (
    apply_grounded_paraphrases,
    paraphrase_eligible,
)


def _block(text="The toolbox supports image and video creation.", kind="fact"):
    return ContentBlock(
        block_id="block-1",
        kind=kind,
        text=text,
        claim_id="claim-1",
        evidence_ids=["approved-capability"],
        verification_status="verified",
        source_exact=True,
        origin="approved_claim",
        locked=True,
        required=True,
        binding_hash="b" * 64,
    )


def _output(block):
    return OperatorContentOutput(
        role_id="commercial",
        status="content_ready",
        profile_version="test",
        system_prompt_sha256="a" * 64,
        master_title="A practical creative workflow",
        master_content=block.text,
        blocks=[block],
        platform_variants=[
            PlatformVariant(
                platform="wechat_mp",
                format="article",
                title="Toolbox workflow",
                body=block.text,
                blocks=[block],
            )
        ],
        evidence_refs=block.evidence_ids,
        critic=ContentCritic(passed=True),
        model="writer",
    )


def _raw(text, **extra):
    ref = {"block_ref": "b" * 64, "public_text": text, **extra}
    return [
        {
            "blocks": [ref],
            "platform_variants": [{"platform": "wechat_mp", "blocks": [ref]}],
        }
    ]


@pytest.mark.parametrize(
    "source,paraphrase",
    [
        (
            "The toolbox supports image and video creation.",
            "Images and videos can be created in the toolbox.",
        ),
        (
            "The service exposes an asynchronous task status endpoint.",
            "An endpoint lets clients check asynchronous task status.",
        ),
        ("创作工具箱整合图片与视频创作入口。", "图片和视频创作入口集中在工具箱中。"),
    ],
)
def test_grounded_expression_keeps_evidence_immutable_and_rebuilds_real_body(
    source, paraphrase
):
    block = _block(source)
    output = _output(block)
    calls = []

    async def reviewer(system, payload, fallback, *args, **kwargs):
        calls.append(payload)
        return {
            "_model": "independent-verifier",
            "reviews": [
                {
                    "pair_id": pair["pair_id"],
                    "verdict": "supported",
                    "key_fields_preserved": True,
                    "reason": "Equivalent scope and meaning.",
                }
                for pair in payload["claim_verification"]["pairs"]
            ],
        }

    result = asyncio.run(
        apply_grounded_paraphrases(
            output=output,
            raw_candidates=_raw(paraphrase),
            run=BoundedModelRun(RuntimeBudget()),
            llm=reviewer,
            system="stable",
            model_args=(),
        )
    )
    assert len(calls) == 1  # shared expression verified once across surfaces
    assert len(calls[0]["claim_verification"]["pairs"]) == 1
    assert result.master_content == paraphrase
    assert result.platform_variants[0].body == paraphrase
    new = result.blocks[0]
    assert new.text == source
    assert new.evidence_ids == block.evidence_ids
    assert new.binding_hash == block.binding_hash
    assert new.public_text_verified is True
    review = result.claim_binding_reviews[0]
    assert review.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert review.public_text_sha256 == hashlib.sha256(paraphrase.encode()).hexdigest()
    assert review.review_model == "independent-verifier"


@pytest.mark.parametrize(
    "verdict,fields",
    [
        ("contradicted", True),
        ("uncertain", True),
        ("supported", False),
        ("supported", "true"),
    ],
)
def test_uncertain_or_changed_meaning_never_reaches_rendered_body(verdict, fields):
    block = _block()

    async def reviewer(system, payload, fallback, *args, **kwargs):
        return {
            "_model": "verifier",
            "reviews": [
                {
                    "pair_id": payload["claim_verification"]["pairs"][0]["pair_id"],
                    "verdict": verdict,
                    "key_fields_preserved": fields,
                }
            ],
        }

    result = asyncio.run(
        apply_grounded_paraphrases(
            output=_output(block),
            raw_candidates=_raw(
                "The toolbox creates professional films without human work.",
                public_text_verified=True,
            ),
            run=BoundedModelRun(RuntimeBudget()),
            llm=reviewer,
            system="stable",
            model_args=(),
        )
    )
    assert result.master_content == block.text
    assert result.blocks[0].public_text_verified is False
    assert result.claim_binding_reviews[0].verdict in {"uncertain", "contradicted"}


@pytest.mark.parametrize(
    "source,kind,proposed",
    [
        ("The price is $10.", "fact", "The price is lower."),
        (
            "The model supports 8 input images.",
            "fact",
            "The model supports multiple images.",
        ),
        ("The release date is 2026-09-07.", "fact", "The model is ready now."),
        ("I built a video tool.", "experience", "I built a useful video tool."),
        ("I am a software developer.", "identity", "I develop software."),
        ("价格是九十九元。", "fact", "价格很划算。"),
        ("支持图片创作。", "fact", "支持图片创作并保证交付。"),
    ],
)
def test_sensitive_claims_retain_exact_expression_without_model_spend(
    source, kind, proposed
):
    block = _block(source, kind)
    assert not paraphrase_eligible(block, proposed)

    async def forbidden(*args, **kwargs):
        pytest.fail(
            "Sensitive claim cannot be converted through semantic-only verification"
        )

    result = asyncio.run(
        apply_grounded_paraphrases(
            output=_output(block),
            raw_candidates=_raw(proposed),
            run=BoundedModelRun(RuntimeBudget()),
            llm=forbidden,
            system="stable",
            model_args=(),
        )
    )
    assert result.master_content == source
    assert result.claim_binding_reviews[0].verdict == "exact_required"


def test_no_verifier_means_no_unchecked_paraphrase_even_if_model_self_attests():
    block = _block()

    async def unavailable(*args, **kwargs):
        return {"_error": "unavailable"}

    result = asyncio.run(
        apply_grounded_paraphrases(
            output=_output(block),
            raw_candidates=_raw(
                "Image and video production are supported.", public_text_verified=True
            ),
            run=BoundedModelRun(RuntimeBudget()),
            llm=unavailable,
            system="stable",
            model_args=(),
        )
    )
    assert result.master_content == block.text
    assert result.blocks[0].public_text is None
    assert result.claim_binding_reviews[0].verdict == "uncertain"


def test_duplicate_verifier_attestations_are_rejected():
    block = _block()

    async def malformed(system, payload, fallback, *args, **kwargs):
        item = {
            "pair_id": payload["claim_verification"]["pairs"][0]["pair_id"],
            "verdict": "supported",
            "key_fields_preserved": True,
        }
        return {"_model": "verifier", "reviews": [item, item]}

    result = asyncio.run(
        apply_grounded_paraphrases(
            output=_output(block),
            raw_candidates=_raw("Image and video production are supported."),
            run=BoundedModelRun(RuntimeBudget()),
            llm=malformed,
            system="stable",
            model_args=(),
        )
    )
    assert result.blocks[0].public_text_verified is False
