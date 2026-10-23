"""Scored suggestions are preserved without weakening material rejection gates."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from ai_marketing_api.operator_content import OperatorContentOutput, SUPPORTED_CHANNEL_FORMATS
from ai_marketing_api.operator_editorial import BoundedModelRun, EditorialReview, RuntimeBudget
from ai_marketing_api.operator_review import apply_editorial_assessment, assess_editorial


@pytest.fixture
def actual_review():
    return json.loads((Path(__file__).parent / "fixtures" /
        "editorial_scored_minor_issues.v1.json").read_text(encoding="utf-8"))


def replay(fixture, assessment):
    draft = fixture["draft"]
    output = OperatorContentOutput(role_id="commercial", status="content_ready",
        profile_version="offline-regression", system_prompt_sha256="0" * 64,
        master_title=draft["master_title"], master_content=draft["master_content"],
        platform_variants=[{**variant, "format": SUPPORTED_CHANNEL_FORMATS[variant["platform"]]}
                           for variant in draft["platform_variants"]],
        critic={"passed": True, "checks": [], "errors": [], "warnings": []})
    run = BoundedModelRun(RuntimeBudget(max_model_calls=1))
    calls = []

    async def recorded_assessment(*args, **kwargs):
        calls.append(kwargs.get("stage"))
        return {**copy.deepcopy(assessment), "_model": "offline-recorded-review"}

    review, state = asyncio.run(assess_editorial(output=output, brief={}, run=run,
        # Historical assessment replay retains its original optional-report
        # contract; it must not synthesize a new reader report for old grades.
        llm=recorded_assessment, system="offline", model_args=(), scorecard_required=False))
    return review, state, apply_editorial_assessment(output, review), calls


def test_actual_848e6_review_keeps_suggestions_and_scores_without_a_new_provider(actual_review):
    original = copy.deepcopy(actual_review["editorial_assessment"])
    review, state, output, calls = replay(actual_review, original)
    assert state == "passed" and review.passed
    assert calls == ["editorial_review"]  # the sole invocation is the local replay stub
    assert review.model_dump(mode="json") == original
    assert output.editorial_assessment.model_dump(mode="json") == original
    assert output.quality_scorecard.model_dump(mode="json") == original["quality_scorecard"]
    assert output.critic.passed and output.critic.errors == []
    assert output.master_content == actual_review["draft"]["master_content"]
    assert [v.body for v in output.platform_variants] == [v["body"] for v in actual_review["draft"]["platform_variants"]]
    assert review.issues and all(surface.issues for surface in review.quality_scorecard.surfaces)
    assert actual_review["source_response_status"] == "blocked"  # historical receipt is not rewritten


@pytest.mark.parametrize("failure", ["low", "major", "false_boolean", "failed_surface"])
def test_scored_advice_never_waives_a_material_rejection(actual_review, failure):
    assessment = copy.deepcopy(actual_review["editorial_assessment"])
    if failure == "low":
        assessment["quality_scorecard"]["surfaces"][0]["analytical_depth"] = 3
    elif failure == "major":
        assessment["quality_scorecard"]["surfaces"][0]["major_issues"] = ["Unsupported claim"]
    elif failure == "false_boolean":
        assessment["useful_to_reader"] = False
    else:
        assessment["failed_surfaces"] = ["wechat_mp"]
    review, state, output, _ = replay(actual_review, assessment)
    assert review is not None and not review.passed and state == "failed"
    assert not output.critic.passed and "editorial_review_failed" in output.critic.errors
    assert output.editorial_assessment.issues


@pytest.mark.parametrize("failure", ["false_card", "stale_card", "empty_surfaces"])
def test_direct_gate_does_not_trust_a_constructed_or_stale_aggregate(actual_review, failure):
    review = EditorialReview.model_validate(actual_review["editorial_assessment"])
    if failure == "false_card":
        review.quality_scorecard.passed = False
    elif failure == "stale_card":
        review.quality_scorecard.surfaces[0].reader_relevance = 3
    else:
        review.quality_scorecard.surfaces = []
    assert not review.passed


def test_legacy_review_still_requires_no_issues_or_failed_surfaces(actual_review):
    assessment = copy.deepcopy(actual_review["editorial_assessment"])
    assessment.pop("quality_scorecard")
    assert not EditorialReview.model_validate(assessment).passed
    assessment["issues"] = []
    assert EditorialReview.model_validate(assessment).passed
    assessment["failed_surfaces"] = ["master"]
    assert not EditorialReview.model_validate(assessment).passed
