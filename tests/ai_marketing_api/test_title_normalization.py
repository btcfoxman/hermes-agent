from __future__ import annotations

from copy import deepcopy

import pytest

from ai_marketing_api.operator_content import (
    ContentBlock,
    _safe_master_title,
    build_content_fallback,
    content_request_payload,
    normalize_content_output,
)
from ai_marketing_api.operator_runtime import OperatorRegistry, OperatorRole
from tests.ai_marketing_api.test_operator_api import _publishable_model_candidate
from tests.ai_marketing_api.test_operator_content_runtime import (
    AS_OF,
    _claim,
    _context,
    _request,
)


def _inputs():
    fact = "The vendor released version 2."
    source = _context(
        "synthetic-title-source", "industry", "industry", "source_item",
        content=fact, source_tier="official",
    )
    request = _request(
        [source], [_claim(fact, "fact", [source.record_id])],
        channels=["wechat_mp", "wechat_moments", "weitoutiao"],
    )
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, [source], AS_OF)
    fallback = build_content_fallback(registry, OperatorRole.INDUSTRY, request, authorized)
    payload = content_request_payload(OperatorRole.INDUSTRY, request, authorized)
    return registry, request, authorized, fallback, payload


def _title_warnings(output):
    return [warning for warning in output.critic.warnings if warning.startswith("model_title_normalized:")]


@pytest.mark.parametrize("title, reason", [
    (None, "missing"),
    ("  ", "missing"),
    ("short", "invalid_length"),
    ("v" * 49, "invalid_length"),
    ("<b>version</b>", "forbidden_markup"),
    ("version www.invalid", "forbidden_url"),
    ("version guaranteed value", "forbidden_promise"),
    ("编辑说明：version 的选择", "forbidden_meta"),
    ("不是 version 的全部变化", "forbidden_wrapper"),
    ("version 3 reader note", "ungrounded_number"),
    ("陌生平台 version 判断", "ungrounded_entity"),
    ("Generic poetic framing", "missing_fact_anchor"),
    ("version announced new tools", "ungrounded_assertion"),
    ("I built version tools", "unsafe_personal_attribution"),
])
def test_rejected_title_reports_only_a_fixed_reason_without_changing_fallback(title, reason):
    _, request, _, fallback, _ = _inputs()
    request.approved_proposal.first_person = True
    blocks = [ContentBlock(**block) for block in fallback["blocks"]]
    warnings = []

    normalized = _safe_master_title(
        request, blocks, title, role=OperatorRole.INDUSTRY,
        diagnostic_warnings=warnings, surface="wechat_moments",
    )

    assert normalized == "The vendor released version 2."
    assert normalized == _safe_master_title(request, blocks, title, role=OperatorRole.INDUSTRY)
    assert warnings == [f"model_title_normalized:wechat_moments:{reason}"]
    assert "synthetic-title-source" not in str(warnings)
    assert normalized not in str(warnings)


@pytest.mark.parametrize("title", [
    "version 2 reader choices",
    "vendor workflow choices",
    "我认为 version 更适合先看选择",
])
def test_safe_titles_remain_identical_without_normalization_warnings(title):
    _, request, _, fallback, _ = _inputs()
    request.approved_proposal.first_person = True
    blocks = [ContentBlock(**block) for block in fallback["blocks"]]
    warnings = []

    assert _safe_master_title(
        request, blocks, title, role=OperatorRole.INDUSTRY,
        diagnostic_warnings=warnings,
    ) == title
    assert warnings == []


def test_valid_approved_title_remains_the_first_fallback_choice():
    _, request, _, fallback, _ = _inputs()
    request.approved_proposal.title = "vendor workflow choices"
    warnings = []
    assert _safe_master_title(
        request, [ContentBlock(**block) for block in fallback["blocks"]],
        "PRIVATE_REJECTED_TITLE", role=OperatorRole.INDUSTRY,
        diagnostic_warnings=warnings,
    ) == request.approved_proposal.title
    assert warnings == ["model_title_normalized:master:missing_fact_anchor"]


@pytest.mark.parametrize("existing", [49, 50])
def test_title_diagnostics_respect_the_existing_warning_limit(existing):
    _, request, _, fallback, _ = _inputs()
    warnings = [f"prior_warning:{index}" for index in range(existing)]
    _safe_master_title(
        request, [ContentBlock(**block) for block in fallback["blocks"]],
        None, role=OperatorRole.INDUSTRY, diagnostic_warnings=warnings,
    )
    assert len(warnings) == 50
    assert warnings[:existing] == [f"prior_warning:{index}" for index in range(existing)]
    if existing < 50:
        assert warnings[-1] == "model_title_normalized:master:missing"


def test_focused_variant_warns_for_its_missing_title_but_not_omitted_surfaces():
    registry, request, authorized, fallback, payload = _inputs()
    candidate = _publishable_model_candidate({**payload, "channels": ["wechat_moments"]})
    candidate.pop("blocks")
    candidate.pop("master_title")
    candidate["platform_variants"][0].pop("title")

    output = normalize_content_output(
        registry, OperatorRole.INDUSTRY, request, authorized, fallback, candidate,
    )

    assert _title_warnings(output) == ["model_title_normalized:wechat_moments:missing"]
    assert len(output.critic.warnings) <= 50
    assert not output.critic.errors


@pytest.mark.parametrize("provided", ["blocks", "master_title"])
def test_a_submitted_master_warns_even_when_only_one_master_field_is_present(provided):
    registry, request, authorized, fallback, payload = _inputs()
    candidate = _publishable_model_candidate({**payload, "channels": []})
    if provided == "blocks":
        candidate.pop("master_title")
        reason = "missing"
    else:
        candidate.pop("blocks")
        candidate["master_title"] = "PRIVATE_REJECTED_TITLE"
        reason = "missing_fact_anchor"

    output = normalize_content_output(
        registry, OperatorRole.INDUSTRY, request, authorized, fallback, candidate,
    )

    assert _title_warnings(output) == [f"model_title_normalized:master:{reason}"]
    assert "PRIVATE_REJECTED_TITLE" not in str(output.critic.warnings)


def test_a_submitted_platform_object_warns_even_when_its_blocks_are_absent():
    registry, request, authorized, fallback, _ = _inputs()
    output = normalize_content_output(
        registry, OperatorRole.INDUSTRY, request, authorized, fallback,
        {"_model": "synthetic-model", "platform_variants": [{"platform": "wechat_moments"}]},
    )
    assert _title_warnings(output) == ["model_title_normalized:wechat_moments:missing"]


def test_valid_model_titles_and_deterministic_fallback_do_not_generate_title_warnings():
    registry, request, authorized, fallback, payload = _inputs()
    for candidate in (_publishable_model_candidate(payload), deepcopy(fallback)):
        output = normalize_content_output(
            registry, OperatorRole.INDUSTRY, request, authorized, fallback, candidate,
        )
        assert _title_warnings(output) == []


def test_title_warning_cannot_overflow_risks_when_blocking_errors_fill_the_budget():
    registry, request, authorized, fallback, payload = _inputs()
    candidate = _publishable_model_candidate(payload)
    candidate["blocks"] = [None] * 50
    candidate.pop("master_title")

    output = normalize_content_output(
        registry, OperatorRole.INDUSTRY, request, authorized, fallback, candidate,
    )

    assert output.status == "evidence_insufficient"
    assert len(output.critic.errors) == 50
    assert _title_warnings(output) == ["model_title_normalized:master:missing"]
    assert len(output.risk_flags) == 50
    assert all(risk.blocking for risk in output.risk_flags)
    assert [risk.message for risk in output.risk_flags] == output.critic.errors
    assert output.master_content is None
    assert output.blocks == []
    assert output.platform_variants == []
