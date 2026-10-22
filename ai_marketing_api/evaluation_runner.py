"""Replay editorial scorecards offline. Network and paid calls are opt-in.

These synthetic fixtures test contracts, never establish factual authority or
the owner's acceptance. The live switch requires existing budget reservations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from ai_marketing_api.operator_content import OperatorComposeRequest, _render, build_content_fallback, enforce_social_publishability, normalize_content_output
from ai_marketing_api.operator_editorial import QualityScorecard
from ai_marketing_api.operator_runtime import OperatorRegistry
from ai_marketing_api.operator_grounding import paraphrase_eligible


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/ai_marketing_api/fixtures/editorial_scenarios.v1.json"


def recheck_graph(scenario: dict, response: dict) -> bool:
    """Run the actual evidence normalizer, not the returned critic's assertion."""
    try:
        request = OperatorComposeRequest.model_validate(scenario["request"])
        registry = OperatorRegistry()
        contexts = registry.authorize(scenario["role_id"], request.authorized_context, request.as_of)
        fallback = build_content_fallback(registry, scenario["role_id"], request, contexts)
        output = enforce_social_publishability(normalize_content_output(registry, scenario["role_id"], request, contexts, fallback, response))
        if output.critic.errors:
            return False
        originals = {"master": response.get("blocks", []), **{v["platform"]: v.get("blocks", []) for v in response.get("platform_variants", [])}}
        certificates = response.get("claim_binding_reviews", [])
        surfaces = {"master": output.blocks, **{v.platform: v.blocks for v in output.platform_variants}}
        for surface, blocks in surfaces.items():
            for block in blocks:
                old = next((b for b in originals.get(surface, []) if b.get("binding_hash") == block.binding_hash), {})
                if old.get("public_text_verified") and old.get("public_text"):
                    public = old["public_text"]
                    if not paraphrase_eligible(block, public):
                        return False
                    matches = [c for c in certificates if c.get("claim_id") == block.claim_id
                        and c.get("evidence_ids") == block.evidence_ids and c.get("verdict") == "supported"
                        and c.get("source_sha256") == hashlib.sha256(block.text.encode()).hexdigest()
                        and c.get("public_text_sha256") == hashlib.sha256(public.encode()).hexdigest()
                        and c.get("review_model") not in {None, "", "fallback", "unavailable", "deterministic"}]
                    if len(matches) != 1:
                        return False
                    block.public_text, block.public_text_verified = public, True
            rendered = _render(blocks, surface=surface)
            original_body = response.get("master_content") if surface == "master" else next(v["body"] for v in response["platform_variants"] if v["platform"] == surface)
            if rendered != original_body:
                return False
        return True
    except (ValueError, TypeError, KeyError, StopIteration):
        return False


def load_scenarios(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "editorial.evaluation_scenarios.v1":
        raise ValueError("unsupported_scenario_schema")
    scenarios = data["scenarios"]
    ids = [row["scenario_id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_scenario")
    for row in scenarios:
        OperatorComposeRequest.model_validate(row["request"])
        if row["role_id"] not in {"commercial", "industry", "personal_ip"} or row["split"] not in {"tune", "heldout"}:
            raise ValueError("invalid_scenario")
    return scenarios


def score_response(scenario: dict, response: dict | None) -> dict:
    result = {"scenario_id": scenario["scenario_id"], "role_id": scenario["role_id"],
        "split": scenario["split"], "status": "not_run", "passed": False, "reasons": [],
        "human_approval_credit": 0}
    if response is None:
        return result
    if isinstance(response.get("artifact"), dict):
        response = response["artifact"]
    if scenario["expected"] == "abstain":
        passed = response.get("status") in {"needs_input", "evidence_insufficient", "quality_insufficient"} and not response.get("master_content")
        result.update(status="passed" if passed else "failed", passed=passed,
            reasons=[] if passed else ["material_gap_not_respected"])
        return result
    reasons = []
    if response.get("status") != "content_ready" or response.get("model") in {None, "", "fallback"}:
        reasons.append("draft_not_ready")
    if response.get("critic", {}).get("errors") or any(risk.get("blocking") for risk in response.get("risk_flags", [])):
        reasons.append("hard_failure")
    if response.get("requires_human_review") is not True:
        reasons.append("human_review_missing")
    if not recheck_graph(scenario, response):
        reasons.append("evidence_graph_failed")
    try:
        card = QualityScorecard.model_validate(response.get("quality_scorecard"))
        expected = {"master", *scenario["request"]["channels"]}
        surfaces = [row.surface for row in card.surfaces]
        if set(surfaces) != expected or len(surfaces) != len(expected):
            reasons.append("scorecard_surfaces_incomplete")
        if not all(row.passed for row in card.surfaces):
            reasons.append("quality_threshold_failed")
        draft = {"master_title": response.get("master_title"), "master_content": response.get("master_content"),
            "platform_variants": [{"platform": v["platform"], "title": v["title"], "body": v["body"]}
                                  for v in response.get("platform_variants", [])]}
        digest = hashlib.sha256(json.dumps(draft, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if card.content_sha256 != digest:
            reasons.append("scorecard_content_changed")
        if {v["platform"] for v in draft["platform_variants"]} != set(scenario["request"]["channels"]):
            reasons.append("draft_surfaces_incomplete")
    except (ValueError, TypeError, KeyError):
        reasons.append("scorecard_invalid")
    result.update(status="failed" if reasons else "passed", passed=not reasons, reasons=reasons)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--results", type=Path, help="JSON list of {scenario_id,response}; no network")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("tune", "heldout", "all"), default="tune")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    parser.add_argument("--api-url")
    parser.add_argument("--service-key-env", default="HERMES_API_KEY")
    parser.add_argument("--reservations", type=Path, help="Existing scenario_id -> reservation ID JSON mapping")
    parser.add_argument("--save-results", type=Path)
    args = parser.parse_args(argv)
    scenarios = [row for row in load_scenarios(args.fixture) if args.split == "all" or row["split"] == args.split]
    results = {}
    if args.results:
        rows = json.loads(args.results.read_text(encoding="utf-8"))
        for row in rows:
            if row["scenario_id"] in results:
                parser.error("duplicate result")
            results[row["scenario_id"]] = row["response"]
    if args.live:
        if not args.allow_paid or not args.reservations or not args.api_url or args.results:
            parser.error("live requires --allow-paid --api-url --reservations and no --results")
        if args.split != "tune" and not args.unlock_heldout:
            parser.error("heldout live evaluation requires --unlock-heldout")
        parsed = urlsplit(args.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            parser.error("invalid service URL")
        reservations = json.loads(args.reservations.read_text(encoding="utf-8"))
        if any(not isinstance(reservations.get(row["scenario_id"]), str) or not reservations[row["scenario_id"]] for row in scenarios):
            parser.error("every selected scenario requires its own existing reservation")
        if len({reservations[row["scenario_id"]] for row in scenarios}) != len(scenarios):
            parser.error("reservations cannot be reused across scenarios")
        key = os.getenv(args.service_key_env, "")
        if not key:
            parser.error("service credential is unavailable")
        import httpx
        with httpx.Client(timeout=600, trust_env=False, follow_redirects=False) as client:
            for row in scenarios:
                response = client.post(f"{args.api_url.rstrip('/')}/api/v1/operators/{row['role_id']}/compose",
                    json=row["request"], headers={"Authorization": f"Bearer {key}",
                        "X-Hermes-Single-Provider-Request": "true",
                        "X-Hermes-Budget-Reservation-Id": reservations[row["scenario_id"]]})
                response.raise_for_status()
                results[row["scenario_id"]] = response.json()
    rows = [score_response(row, results.get(row["scenario_id"])) for row in scenarios]
    report = {"schema_version": "editorial.evaluation_report.v1", "live": args.live,
        "synthetic_contract_exercise": any(row.get("material_type") == "synthetic_evaluation_only" for row in scenarios), "human_approval_credit": 0,
        "qualified": bool(rows) and all(row["passed"] for row in rows),
        "total": len(rows), "passed": sum(row["passed"] for row in rows),
        "not_run": sum(row["status"] == "not_run" for row in rows), "results": rows}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.save_results:
        args.save_results.write_text(json.dumps([{"scenario_id": key, "response": value}
            for key, value in results.items()], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
