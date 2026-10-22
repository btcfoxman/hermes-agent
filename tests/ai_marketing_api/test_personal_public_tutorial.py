"""Public tutorials are public facts, never implicit personal testimony."""
from __future__ import annotations

import pytest

import ai_marketing_api.main as api
from tests.ai_marketing_api.test_operator_api import _request


FACT = "公开工具允许负责人自行选择待审核稿件。"


def payload(space="industry"):
    return {"objective": "解释有依据的工具操作", "topic": "审核工具如何选择稿件", "audience": "内容编辑",
        "channels": ["wechat_mp"], "as_of": "2026-07-16T08:00:00Z", "genre": "public_tutorial",
        "authorized_context": [{"record_id": "public-source", "space": space,
            "record_type": "source_item" if space == "industry" else "source_fact",
            "title": "审核工具如何选择稿件", "content": FACT, "structured_data": {},
            "status": "approved", "authorized_roles": ["personal_ip"],
            "source_uri": "https://evaluation.invalid/official-documentation", "source_tier": "official"}]}


def candidate(body, *, invented_testimony=False):
    refs = [{"block_ref": row["block_ref"]} for row in body["canonical_block_registry"] if row["required"]]
    extra = [{"kind": "opinion", "text": "我亲测审核工具后发现自己的客户效果明显更好。", "evidence_ids": []}] if invented_testimony else []
    return {"_model": "offline-author", "master_title": "我亲测审核工具如何选择稿件" if invented_testimony else "审核工具如何选择稿件",
        "blocks": [*refs, *extra], "platform_variants": [{"platform": platform,
            "title": "负责人如何选择审核稿件", "blocks": [*refs, *extra]} for platform in body["channels"]]}


def install_model(monkeypatch, *, invented_testimony=False):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")
    calls = []
    async def model(system, body, fallback, *args, **kwargs):
        calls.append(body)
        if "editorial_review" in body:
            draft = body["editorial_review"]["draft"]
            assert "我亲测" not in str(draft)
            return {"_model": "offline-independent-editor", "quality_scorecard": {
                "schema_version": "quality_scorecard.v1", "surfaces": [{"surface": surface,
                    "reader_relevance": 4, "information_gain": 4, "analytical_depth": 4,
                    "natural_expression": 4, "platform_fit": 4, "major_issues": [], "issues": []}
                    for surface in ["master", *[v["platform"] for v in draft["platform_variants"]]]]}}
        if "canonical_block_registry" in body:
            assert body["approved_proposal"]["first_person"] is False
            assert "public tool tutorial" in body["publication_brief"]["thesis_contract"][0]
            return candidate(body, invented_testimony=invented_testimony)
        # A model cannot convert an authorized public tutorial into a claimed
        # first-person experiment or replace the exact canonical source fact.
        return {"_model": "offline-author", "proposal": {"title": "我亲测得到了显著结果", "first_person": True},
            "claims": [{"text": "我亲测得到了显著结果", "kind": "experience", "evidence_ids": ["public-source"]}]}
    monkeypatch.setattr(api, "_llm_json", model)
    return calls


@pytest.mark.parametrize("space", ["industry", "company_public"])
def test_explicit_public_tutorial_proposes_and_composes_without_owner_cards(monkeypatch, space):
    install_model(monkeypatch)
    value = payload(space)
    proposed = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    assert proposed["status"] == "proposal", proposed
    assert proposed["proposal"]["first_person"] is False
    assert [claim["text"] for claim in proposed["claims"]] == [FACT]
    assert all(claim["kind"] == "fact" for claim in proposed["claims"])
    composed = _request("POST", "/api/v1/operators/personal_ip/compose", json={**value,
        "approved_proposal": proposed["proposal"], "claims": proposed["claims"]}).json()
    assert composed["status"] == "content_ready", composed["critic"]
    assert composed["master_content"] == FACT
    assert composed["quality_scorecard"]["passed"] is True
    assert composed["requires_human_review"] is True


@pytest.mark.parametrize("missing", ["legacy", "empty", "secondary", "synthetic", "testimony"])
def test_public_tutorial_requires_explicit_mode_and_publishable_primary_facts(monkeypatch, missing):
    install_model(monkeypatch)
    value = payload()
    if missing == "legacy":
        value.pop("genre")
    elif missing == "empty":
        value["authorized_context"] = []
    elif missing == "secondary":
        value["authorized_context"][0]["source_tier"] = "secondary"
    elif missing == "synthetic":
        value["authorized_context"][0]["structured_data"] = {"e2e_only": True}
    else:
        value["authorized_context"][0]["content"] = "亲测审核工具后，客户效果明显更好。"
    response = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    assert response["status"] == "needs_input", response
    assert response["proposal"] is None
    assert response["claims"] == []


@pytest.mark.parametrize("kind", ["identity", "experience", "opinion"])
def test_public_tutorial_does_not_turn_public_sources_into_owner_cards(monkeypatch, kind):
    calls = install_model(monkeypatch)
    value = payload()
    proposed = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    before = len(calls)
    composed = _request("POST", "/api/v1/operators/personal_ip/compose", json={**value,
        "approved_proposal": proposed["proposal"], "claims": [{**proposed["claims"][0], "kind": kind}]}).json()
    assert composed["status"] != "content_ready"
    assert "personal_claim_without_approved_card:claim-1" in composed["critic"]["errors"]
    assert composed["master_content"] is None
    assert len(calls) == before


def test_public_tutorial_rejects_first_person_outline_before_dispatch(monkeypatch):
    calls = install_model(monkeypatch)
    value = payload()
    proposed = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    before = len(calls)
    proposed["proposal"]["first_person"] = True
    composed = _request("POST", "/api/v1/operators/personal_ip/compose", json={**value,
        "approved_proposal": proposed["proposal"], "claims": proposed["claims"]}).json()
    assert composed["status"] == "needs_input"
    assert "public_tutorial_attribution_forbidden" in composed["critic"]["errors"]
    assert len(calls) == before


def test_public_tutorial_cannot_mix_even_approved_owner_claims_into_public_mode(monkeypatch):
    calls = install_model(monkeypatch)
    value = payload()
    proposed = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    opinion = "工具设计应当保留清楚的审核边界。"
    value["authorized_context"].append({**value["authorized_context"][0], "record_id": "owner-opinion",
        "space": "personal_approved", "record_type": "opinion", "content": opinion})
    before = len(calls)
    composed = _request("POST", "/api/v1/operators/personal_ip/compose", json={**value,
        "approved_proposal": proposed["proposal"], "claims": [*proposed["claims"], {"text": opinion,
            "kind": "opinion", "evidence_ids": ["owner-opinion"], "verification_status": "opinion"}]}).json()
    assert composed["status"] == "needs_input"
    assert "public_tutorial_fact_only" in composed["critic"]["errors"]
    assert len(calls) == before


def test_public_tutorial_discards_invented_testimony_from_title_and_prose(monkeypatch):
    install_model(monkeypatch, invented_testimony=True)
    value = payload()
    proposed = _request("POST", "/api/v1/operators/personal_ip/propose", json=value).json()
    composed = _request("POST", "/api/v1/operators/personal_ip/compose", json={**value,
        "approved_proposal": proposed["proposal"], "claims": proposed["claims"]}).json()
    assert composed["status"] == "content_ready", composed["critic"]
    assert "亲测" not in str({"title": composed["master_title"], "body": composed["master_content"],
        "variants": composed["platform_variants"]})
    assert any("first_person_forbidden" in warning for warning in composed["critic"]["warnings"])
