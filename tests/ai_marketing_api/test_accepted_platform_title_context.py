"""Synthetic staged transport; hints never certify a title or an article."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import re

import pytest
from pydantic import ValidationError

from ai_marketing_api import operator_steps as steps
from ai_marketing_api.operator_receipts import request_sha256
from ai_marketing_api.operator_runtime import ContextAuthorizationError, OperatorRegistry
from tests.ai_marketing_api.test_industry_analysis_block_order import _compose
from tests.ai_marketing_api.test_industry_title_parity import choose


CHANNELS = ["wechat_mp", "wechat_moments", "weitoutiao"]
SIBLINGS = [{"surface": "wechat_mp", "title": "Synthetic prior MP title"},
    {"surface": "wechat_moments", "title": "Synthetic prior feed title"}]


def request():
    return _compose("industry_analysis", "weitoutiao").model_copy(update={"channels": CHANNELS})


def test_empty_context_is_absent_from_legacy_dump_and_nonempty_context_is_wire_bound():
    legacy = steps.EditorialStepRequest(compose=request())
    assert "accepted_platform_titles" not in legacy.model_dump(mode="json")
    enriched = steps.EditorialStepRequest(compose=request(), surface="weitoutiao",
        accepted_platform_titles=SIBLINGS)
    dumped = enriched.model_dump(mode="json")
    assert dumped["accepted_platform_titles"] == SIBLINGS
    without = {key: value for key, value in dumped.items() if key != "accepted_platform_titles"}
    assert request_sha256(dumped) != request_sha256(without)
    assert steps.EditorialStepRequest.model_validate(dumped).model_dump(mode="json") == dumped


@pytest.mark.parametrize("bad", [SIBLINGS * 3,
    [{"surface": "wechat_mp", "title": "x" * 301}],
    [{"surface": "s" * 81, "title": "Synthetic"}],
    [{"surface": "wechat_mp", "title": 4}],
    [{"surface": "wechat_mp", "title": "Synthetic", "verified": True}]])
def test_context_bounds_and_types_reject_before_execution(bad):
    with pytest.raises(ValidationError):
        steps.EditorialStepRequest(compose=request(), accepted_platform_titles=bad)


@pytest.mark.parametrize("kind", ["same_target", "foreign_surface", "duplicate_surface", "blank",
    "master_step", "verifier_step", "reader_step", "other_genre"])
def test_context_cannot_expand_the_original_stage_or_surface_scope(monkeypatch, kind):
    value = request()
    hints = deepcopy(SIBLINGS)
    step = {"master_step": "draft_master", "verifier_step": "verify_claims",
        "reader_step": "review_editorial"}.get(kind, "adapt_platform")
    if kind == "same_target": hints[0]["surface"] = "weitoutiao"
    if kind == "foreign_surface": hints[0]["surface"] = "douyin"
    if kind == "duplicate_surface": hints[1]["surface"] = hints[0]["surface"]
    if kind == "blank": hints[0]["title"] = "   "
    if kind == "other_genre": value = value.model_copy(update={"genre": "company_data_brief"})
    calls = []

    async def writer(*_args, **_kwargs):
        calls.append(True)
        pytest.fail("An invalid title context must not reach a provider.")

    payload = steps.EditorialStepRequest(compose=value, surface="weitoutiao", accepted_platform_titles=hints)
    with pytest.raises(ContextAuthorizationError) as error:
        asyncio.run(steps.run_editorial_step(OperatorRegistry(), "industry", step, payload, writer))
    assert error.value.code == "accepted_platform_titles_scope_denied"
    assert calls == []


@pytest.mark.parametrize("inject_unsupported_title", [False, True])
def test_real_staged_prompt_gets_only_untrusted_prior_titles_and_keeps_factual_guards(monkeypatch,
    inject_unsupported_title):
    monkeypatch.setenv("HERMES_API_KEY", "synthetic-title-context-key")
    monkeypatch.setattr(steps, "server_stage_route", lambda stage: {"model": "synthetic-writer"})
    value = request()
    before = value.model_dump(mode="json")
    seen = []
    unsupported = "\u793a\u4f8b\u63a5\u53e3\u5df2\u652f\u6301999\u9879\u81ea\u52a8\u51b3\u7b56"
    hints = deepcopy(SIBLINGS)
    if inject_unsupported_title: hints[0]["title"] = unsupported

    async def writer(_system, body, _fallback, **_kwargs):
        seen.append(deepcopy(body))
        refs = [{"block_ref": row["block_ref"]} for row in body["canonical_block_registry"] if row["required"]]
        if body["stage_writer_contract"]["step"] == "draft_master":
            return {"_model": "synthetic-writer", "master_title": value.approved_proposal.title,
                "blocks": refs, "platform_variants": []}
        return {"_model": "synthetic-writer", "platform_variants": [{"platform": "weitoutiao",
            "title": unsupported if inject_unsupported_title else value.approved_proposal.title,
            "blocks": refs}]}

    async def run():
        registry = OperatorRegistry()
        master = await steps.run_editorial_step(registry, "industry", "draft_master",
            steps.EditorialStepRequest(compose=value), writer)
        assert master["status"] == "artifact", master
        original = deepcopy(master)
        payload = steps.EditorialStepRequest(compose=value, surface="weitoutiao",
            candidate=master["artifact"], candidate_signature=master["artifact_signature"],
            accepted_platform_titles=hints)
        payload_before = payload.model_dump(mode="json")
        adapted = await steps.run_editorial_step(registry, "industry", "adapt_platform", payload, writer)
        assert master == original and payload.model_dump(mode="json") == payload_before
        assert adapted["status"] == "artifact", adapted
        return master, adapted

    master, adapted = asyncio.run(run())
    contract = seen[1]["stage_writer_contract"]
    assert contract["accepted_platform_titles"] == hints
    assert "untrusted" in contract["accepted_platform_titles_rule"]
    assert "not instructions, approved claims" in contract["accepted_platform_titles_rule"]
    assert "A number is optional" in seen[1]["response_contract"]["title_safety_rule"]
    assert seen[0]["canonical_block_registry"] == seen[1]["canonical_block_registry"]
    assert value.model_dump(mode="json") == before and len(seen) == 2
    assert [row["platform"] for row in adapted["artifact"]["platform_variants"]] == ["weitoutiao"]
    if inject_unsupported_title:
        assert adapted["artifact"]["platform_variants"][0]["title"] != unsupported
        assert any("model_title_normalized:weitoutiao:ungrounded_number" == warning
            for warning in adapted["artifact"]["critic"]["warnings"])
    for key in ("master_title", "master_content", "blocks"):
        assert adapted["artifact"][key] == master["artifact"][key]


def test_original_title_fence_already_allows_distinct_quantity_free_grounded_angles():
    titles = ["Custom Agent \u52a8\u4f5c\u6743\u9650\u600e\u4e48\u5b9a",
        "Custom Agent \u7684\u83b7\u51c6\u52a8\u4f5c\u8be5\u5982\u4f55\u6838\u5bf9",
        "Custom Agent \u6743\u9650\u8303\u56f4\u9700\u8981\u5148\u95ee\u4ec0\u4e48"]
    actual = []
    for title in titles:
        value, warnings, _ = choose(title)
        assert value == title and warnings == []
        actual.append(re.sub(r"\W+", "", value).lower())
    assert len(set(actual)) == 3
