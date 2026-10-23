"""Signed server display plans use the real evidence normalizer, without HTTP."""
import hashlib
import hmac
import json
from copy import deepcopy

import pytest

from ai_marketing_api.operator_content import content_request_payload, _validated_surface_fact_plan
from ai_marketing_api.operator_runtime import ContextAuthorizationError, OperatorRegistry, OperatorRole
from tests.ai_marketing_api.test_operator_content_runtime import _claim, _context, _request, _run


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "offline-hermes-secret")
    facts = ["预览用于确认镜头意图。", "预览与正式任务独立计费。", "升级自动复用提示词和参考素材，不能重传。"]
    contexts = [_context(f"source-{i}", "industry", "industry", "source_item", content=fact, source_tier="official") for i, fact in enumerate(facts, 1)]
    request = _request(contexts, [_claim(fact, "fact", [f"source-{i}"]) for i, fact in enumerate(facts, 1)],
        channels=["wechat_mp", "wechat_moments", "weitoutiao"])
    request.genre = "industry_analysis"
    ids = [f"claim-{i}" for i in range(1, 4)]
    body = {"schema_version": "surface_fact_plan.v1", "scope": {"role_id": "industry", "genre": request.genre,
        "objective_sha256": hashlib.sha256(request.objective.encode()).hexdigest()},
        "claim_pool": [{"claim_id": f"claim-{i}", "kind": "fact", "text_sha256": hashlib.sha256(f.encode()).hexdigest(),
            "evidence_ids": [f"source-{i}"]} for i, f in enumerate(facts, 1)],
        "surfaces": {"master": ids, "wechat_mp": ids, "wechat_moments": [ids[0]], "weitoutiao": [ids[0]]},
        "fact_roles": {"anchor": [ids[0]], "billing": [ids[1]], "input_constraints": [ids[2]]},
        "editorial_intent": {"author": "Codex AI editor", "thesis": "按待确认的问题选预览"}, "narrative_required": True}
    request.surface_fact_plan = {**body, "plan_sha256": digest(body), "signature": hmac.new(b"offline-hermes-secret",
        ("surface_fact_plan.v1\n" + digest(body)).encode(), hashlib.sha256).hexdigest()}
    reg = OperatorRegistry()
    contexts = reg.authorize(OperatorRole.INDUSTRY, request.authorized_context, request.as_of)
    payload = content_request_payload(OperatorRole.INDUSTRY, request, contexts)
    refs = [{"block_ref": b["block_ref"]} for b in payload["canonical_block_registry"]]
    candidate = {"_model": "offline-writer", "master_title": "镜头意图的确认", "blocks": refs,
        "platform_variants": [{"platform": s, "title": "预览怎样确认镜头", "blocks": refs if s == "wechat_mp" else refs[:1]}
            for s in request.channels]}
    return request, contexts, candidate


def test_real_normalizer_selects_per_surface_without_changing_approved_claims(monkeypatch):
    request, _, candidate = fixture(monkeypatch)
    output = _run(OperatorRole.INDUSTRY, request, candidate)
    assert output.critic.errors == []
    assert [b.claim_id for b in output.blocks] == ["claim-1", "claim-2", "claim-3"]
    assert {v.platform: [b.claim_id for b in v.blocks] for v in output.platform_variants} == {
        "wechat_mp": ["claim-1", "claim-2", "claim-3"], "wechat_moments": ["claim-1"], "weitoutiao": ["claim-1"]}
    assert all(b.required and b.locked for b in output.blocks)
    assert output.surface_fact_plan_sha256 == request.surface_fact_plan["plan_sha256"]
    request.surface_fact_plan = None
    legacy = _run(OperatorRole.INDUSTRY, request, candidate)
    assert all({b.claim_id for b in v.blocks} == {"claim-1", "claim-2", "claim-3"} for v in legacy.platform_variants)
    assert "surface_fact_plan_sha256" not in legacy.model_dump(mode="json")


@pytest.mark.parametrize("text,expected", [("仍需比较新增成本。", "claim-2"), ("提示词不能改，需要先确认方向。", "claim-3")])
def test_authored_condition_restores_its_complete_required_evidence(monkeypatch, text, expected):
    request, _, candidate = fixture(monkeypatch)
    candidate["platform_variants"][1]["blocks"].append({"kind": "opinion", "text": text, "evidence_ids": []})
    output = _run(OperatorRole.INDUSTRY, request, candidate)
    assert output.critic.errors == []
    assert expected in {b.claim_id for b in output.platform_variants[1].blocks}


@pytest.mark.parametrize("field", ["signature", "claim", "objective", "genre", "required"])
def test_untrusted_candidate_cannot_change_signed_plan_or_claim_scope(monkeypatch, field):
    request, contexts, candidate = fixture(monkeypatch)
    if field == "signature": request.surface_fact_plan["signature"] = "0" * 64
    if field == "claim": request.claims[0].text += " changed"
    if field == "objective": request.objective += " changed"
    if field == "genre": request.genre = "public_tutorial"
    if field == "required": request.surface_fact_plan["surfaces"]["master"] = ["claim-1"]
    with pytest.raises(ContextAuthorizationError):
        content_request_payload(OperatorRole.INDUSTRY, request, contexts)


def test_writer_projection_removes_old_ai_thesis_but_preserves_real_owner_instruction(monkeypatch):
    request, contexts, _ = fixture(monkeypatch)
    from ai_marketing_api.operator_editorial import OwnerRevision, PublicEditorialBrief
    request.public_editorial_brief = PublicEditorialBrief(role_id="industry", objective="解释预览的问题范围", audience="创作团队",
        angle="旧的否决错误才有价值", reader_value="旧的否决错误才有价值", channels=request.channels)
    request.owner_revision = OwnerRevision(instruction="仍然AI味太重，整篇不知道到底在表达什么？")
    before = request.model_dump(mode="json")
    payload = content_request_payload(OperatorRole.INDUSTRY, request, contexts)
    assert "旧的否决错误才有价值" not in json.dumps(payload, ensure_ascii=False)
    assert payload["owner_revision"]["instruction"] == request.owner_revision.instruction
    assert payload["public_editorial_brief"]["audience"] == "创作团队"
    assert payload["public_editorial_brief"]["objective"] == "解释预览的问题范围"
    assert "approved_editorial_brief" not in payload
    assert request.model_dump(mode="json") == before


def test_surface_scoped_payload_projects_required_display_but_keeps_full_pool(monkeypatch):
    request, contexts, _ = fixture(monkeypatch)
    request.channels = ["wechat_moments"]
    payload = content_request_payload(OperatorRole.INDUSTRY, request, contexts)
    assert len(payload["canonical_block_registry"]) == 3
    assert [r["claim_id"] for r in payload["canonical_block_registry"] if r["required"]] == ["claim-1"]


def test_title_cost_discussion_also_requires_billing_evidence(monkeypatch):
    request, _, candidate = fixture(monkeypatch)
    candidate["platform_variants"][1]["title"] = "预览付费前先确认什么"
    result = _run(OperatorRole.INDUSTRY, request, candidate)
    assert "claim-2" in {b.claim_id for b in result.platform_variants[1].blocks}


def test_model_required_false_cannot_downgrade_a_server_reference(monkeypatch):
    request, _, candidate = fixture(monkeypatch)
    candidate["blocks"][0]["required"] = False
    output = _run(OperatorRole.INDUSTRY, request, candidate)
    assert output.critic.errors or any("tampered" in w for w in output.critic.warnings)
    assert all(b.required for b in output.blocks if b.claim_id)
