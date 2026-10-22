import copy
import hashlib
import json

import pytest

from ai_marketing_api.evaluation_runner import DEFAULT_FIXTURE, load_scenarios, main, score_response
from ai_marketing_api.operator_content import OperatorComposeRequest, build_content_fallback, content_request_payload, normalize_content_output
from ai_marketing_api.operator_runtime import OperatorRegistry


def test_fixture_has_three_roles_and_sealed_balanced_splits():
    rows = load_scenarios(DEFAULT_FIXTURE)
    assert len(rows) == 30
    assert len({row["scenario_id"] for row in rows}) == 30
    for role in ("commercial", "industry", "personal_ip"):
        assert sum(row["role_id"] == role and row["split"] == "tune" for row in rows) == 8
        assert sum(row["role_id"] == role and row["split"] == "heldout" for row in rows) == 2
    assert all(row["material_type"] == "synthetic_evaluation_only" for row in rows)


def test_positive_scenario_evidence_is_usable_by_the_real_role_gates():
    registry = OperatorRegistry()
    for row in load_scenarios(DEFAULT_FIXTURE):
        request = OperatorComposeRequest.model_validate(row["request"])
        contexts = registry.authorize(row["role_id"], request.authorized_context, request.as_of)
        fallback = build_content_fallback(registry, row["role_id"], request, contexts)
        if row["expected"] == "scorecard_pass":
            assert not fallback.get("_fallback_errors"), (row["scenario_id"], fallback.get("_fallback_errors"))


def test_default_runner_is_offline_and_does_not_invent_success(capsys):
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["live"] is False
    assert report["passed"] == 0
    assert report["not_run"] == 24
    assert report["qualified"] is False
    assert report["human_approval_credit"] == 0


def test_live_requires_explicit_paid_switch_and_reservations():
    with pytest.raises(SystemExit) as exc:
        main(["--live"])
    assert exc.value.code == 2


def test_replay_does_not_accept_a_stale_or_partial_scorecard():
    scenario = next(row for row in load_scenarios(DEFAULT_FIXTURE) if row["expected"] == "scorecard_pass")
    request = OperatorComposeRequest.model_validate(scenario["request"])
    registry = OperatorRegistry()
    contexts = registry.authorize(scenario["role_id"], request.authorized_context, request.as_of)
    fallback = build_content_fallback(registry, scenario["role_id"], request, contexts)
    body = content_request_payload(scenario["role_id"], request, contexts)
    refs = [{"block_ref": block["block_ref"]} for block in body["canonical_block_registry"] if block["required"]]
    response = normalize_content_output(registry, scenario["role_id"], request, contexts, fallback,
        {"_model": "editor-reviewed", "master_title": request.approved_proposal.title, "blocks": refs,
         "platform_variants": [{"platform": "wechat_mp", "title": request.approved_proposal.title, "blocks": refs}]}).model_dump(mode="json")
    draft = {"master_title": response["master_title"], "master_content": response["master_content"],
        "platform_variants": [{key: variant[key] for key in ("platform", "title", "body")} for variant in response["platform_variants"]]}
    response["quality_scorecard"] = {"schema_version": "quality_scorecard.v1",
        "content_sha256": hashlib.sha256(json.dumps(draft, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "surfaces": [{"surface": surface, "reader_relevance": 4, "information_gain": 4,
            "analytical_depth": 4, "natural_expression": 4, "platform_fit": 4, "major_issues": []}
            for surface in ("master", "wechat_mp")]}
    assert score_response(scenario, response)["passed"] is True, json.dumps({"result":score_response(scenario, response), "critic":response["critic"]})
    changed = copy.deepcopy(response)
    changed["platform_variants"][0]["body"] = "Edited after review"
    assert "scorecard_content_changed" in score_response(scenario, changed)["reasons"]
    changed = copy.deepcopy(response)
    changed["quality_scorecard"]["surfaces"].pop()
    assert "scorecard_surfaces_incomplete" in score_response(scenario, changed)["reasons"]
    changed = copy.deepcopy(response)
    changed["quality_scorecard"]["surfaces"][1]["major_issues"] = ["Invented factual outcome"]
    assert score_response(scenario, changed)["passed"] is False
