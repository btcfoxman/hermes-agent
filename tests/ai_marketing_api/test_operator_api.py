from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest

os.environ["HERMES_API_KEY"] = "operator-test-secret"

import ai_marketing_api.main as marketing_api


app = marketing_api.app


@pytest.fixture(autouse=True)
def _operator_api_key(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    operator_auth = kwargs.pop("operator_auth", True)
    if operator_auth and path.startswith("/api/v1/"):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Authorization", f"Bearer {os.environ.get('HERMES_API_KEY', '')}")
        kwargs["headers"] = headers

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def _base_payload() -> dict:
    return {
        "objective": "Propose today's AI industry topic",
        "topic": "New model release",
        "audience": "AI product teams",
        "channels": ["wechat_mp"],
        "as_of": "2026-07-16T08:00:00Z",
        "authorized_context": [
            {
                "record_id": "source-1",
                "space": "industry",
                "record_type": "source_item",
                "title": "Official release",
                "content": "The vendor released version 2.",
                "structured_data": {},
                "status": "approved",
                "authorized_roles": ["industry"],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": "https://vendor.test/news",
                "source_tier": "official"
            }
        ],
    }


def test_proposal_rejects_cross_role_preference_before_model(monkeypatch):
    payload = _base_payload()
    payload["approved_preferences"] = [
        {
            "preference_version_id": "operatorpref-commercial-v1",
            "role_id": "commercial",
            "preference_key": "commercial.case.structure",
            "guidance": "Use a commercial case structure.",
            "status": "approved",
            "version": 1,
        }
    ]

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("cross-role preference must fail before model execution")

    monkeypatch.setattr(marketing_api, "_llm_json", should_not_run)
    response = _request(
        "POST", "/api/v1/operators/industry/propose", json=payload
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "preference_role_denied"


def _revision_payload() -> dict:
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Focus on product teams.", "previous": proposal})
    return payload


def _compose_payload() -> dict:
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update(
        {
            "approved_proposal": proposal["proposal"],
            "claims": proposal["claims"],
        }
    )
    return payload


def _role_compose_payload(role: str) -> dict:
    if role == "commercial":
        content = "已批准能力：内容交接后必须经过人工终审。"
        space, record_type, kind = "company_public", "capability", "fact"
    elif role == "personal_ip":
        content = "第一次带团队出海时，我暂停了未经验证的投放。"
        space, record_type, kind = "personal_approved", "experience", "experience"
    else:
        return _compose_payload()
    return {
        "objective": "Compose the approved proposal",
        "topic": "Approved topic",
        "audience": "business owners",
        "channels": ["wechat_mp", "xiaohongshu"],
        "as_of": "2026-07-16T08:00:00Z",
        "approved_proposal": {
            "title": content,
            "angle": "Use only approved evidence.",
            "audience_value": "Provide a traceable explanation.",
            "key_points": [content],
            "suggested_formats": ["wechat_mp", "xiaohongshu"],
            "cta": None,
            "first_person": role == "personal_ip",
        },
        "claims": [
            {
                "text": content,
                "kind": kind,
                "evidence_ids": ["record-1"],
                "verification_status": "verified",
            }
        ],
        "authorized_context": [
            {
                "record_id": "record-1",
                "space": space,
                "record_type": record_type,
                "title": "Approved record",
                "content": content,
                "structured_data": {},
                "status": "approved",
                "authorized_roles": [role],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": None,
                "source_tier": "primary",
            }
        ],
    }


def _publishable_model_candidate(payload: dict) -> dict:
    required = [
        {"block_ref": block["block_ref"]}
        for block in payload["canonical_block_registry"]
        if block["required"]
    ]

    def editorial(label: str) -> list[dict]:
        return [
            {
                "kind": "transition",
                "text": f"First, {label} opens with the concrete approved tension.",
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": f"What matters for {label} is the reader's changed choice.",
                "evidence_ids": [],
            },
            {
                "kind": "opinion",
                "text": f"The implication for {label} is a clearer decision boundary.",
                "evidence_ids": [],
            },
        ]

    platform_titles = {
        "wechat_moments": "version 2 moments note",
        "wechat_mp": "version 2 wechat article",
        "wechat_channels": "version 2 channel script",
        "douyin": "version 2 douyin script",
        "kuaishou": "version 2 kuaishou script",
        "xiaohongshu": "version 2 xiaohongshu note",
        "toutiao": "version 2 toutiao article",
        "weitoutiao": "version 2 short headline",
    }
    return {
        "_model": "publishable-test-model",
        "master_title": "vendor released version 2",
        "blocks": [*required, *editorial("the master story")],
        "platform_variants": [
            {
                "platform": channel,
                "title": platform_titles.get(channel, f"version 2 {channel} note"),
                "blocks": [*required, *editorial(channel)],
            }
            for channel in payload["channels"]
        ],
    }


def _protected_endpoints() -> list[tuple[str, str, dict | None]]:
    return [
        ("GET", "/api/v1/operators/health", None),
        ("GET", "/api/v1/operators/industry/probe", None),
        ("POST", "/api/v1/operators/industry/propose", _base_payload()),
        ("POST", "/api/v1/operators/industry/revise", _revision_payload()),
        ("POST", "/api/v1/operators/industry/compose", _compose_payload()),
        ("POST", "/api/v1/topic/analyze", {}),
        ("POST", "/api/v1/brief/generate", {}),
        ("POST", "/api/v1/draft/generate", {}),
        ("POST", "/api/v1/draft/humanize", {}),
        ("POST", "/api/v1/risk/review", {}),
    ]


def test_operator_health_and_role_probes_cover_all_profiles():
    response = _request("GET", "/api/v1/operators/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["schema_version"] == "operator.proposal.v1"
    assert data["schema_versions"] == {
        "proposal": "operator.proposal.v1",
        "content": "operator.content.v1",
    }
    assert set(data["roles"]) == {"commercial", "industry", "personal_ip"}
    for role in data["roles"]:
        probe = _request("GET", f"/api/v1/operators/{role}/probe")
        assert probe.status_code == 200
        assert probe.json()["direct_tools"] == []
        assert probe.json()["knowledge_source"] == "request_authorized_context_only"
        assert probe.json()["supported_actions"] == ["compose", "propose", "revise"]
        assert probe.json()["schema_versions"]["content"] == "operator.content.v1"


def test_api_directory_lists_the_compose_contract():
    response = _request("GET", "/", operator_auth=False)

    assert response.status_code == 200
    assert response.json()["endpoints"]["operatorCompose"] == "/api/v1/operators/{role_id}/compose"


def test_propose_returns_the_strict_versioned_contract(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    response = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "operator.proposal.v1"
    assert data["role_id"] == "industry"
    assert data["action"] == "propose"
    assert data["requires_human_review"] is True
    assert data["evidence_refs"] == ["source-1"]
    assert data["system_prompt_sha256"] == _request(
        "GET", "/api/v1/operators/industry/probe"
    ).json()["system_prompt_sha256"]


def test_revise_stays_in_the_same_profile(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Focus on product teams.", "previous": proposal})

    response = _request("POST", "/api/v1/operators/industry/revise", json=payload)

    assert response.status_code == 200
    assert response.json()["role_id"] == "industry"
    assert response.json()["action"] == "revise"
    assert "Focus on product teams" in response.json()["proposal"]["angle"]


def test_compose_without_model_never_promotes_safe_fallback(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]

    response = _request("POST", "/api/v1/operators/industry/compose", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "operator.content.v1"
    assert data["role_id"] == "industry"
    assert data["action"] == "compose"
    assert data["status"] == "quality_insufficient"
    assert data["requires_human_review"] is True
    assert data["critic"]["passed"] is False
    assert data["master_content"] is None
    assert data["platform_variants"] == []
    assert "model_generation_required" in data["critic"]["errors"]


@pytest.mark.parametrize("role", ["commercial", "personal_ip"])
def test_compose_endpoint_blocks_fallback_for_each_role_profile(role, monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    payload = _role_compose_payload(role)

    response = _request("POST", f"/api/v1/operators/{role}/compose", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["role_id"] == role
    assert data["status"] == "quality_insufficient"
    assert data["critic"]["passed"] is False
    assert "model_generation_required" in data["critic"]["errors"]
    assert data["blocks"] == []


def test_compose_repairs_each_shallow_surface_with_a_focused_request(monkeypatch):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]
    calls: list[dict] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            required = [
                {"block_ref": block["block_ref"]}
                for block in payload["canonical_block_registry"]
                if block["required"]
            ]
            evidence_id = next(
                block["evidence_ids"][0]
                for block in payload["canonical_block_registry"]
                if block["required"] and block["evidence_ids"]
            )
            return {
                "_model": "shallow-test-model",
                "blocks": [
                    *required,
                    {
                        "kind": "opinion",
                        "text": "What matters is how this rule changes the reader's available choice.",
                        "evidence_ids": [evidence_id],
                    },
                ],
                "platform_variants": [
                    {"platform": channel, "blocks": required}
                    for channel in payload["channels"]
                ],
            }
        return _publishable_model_candidate(payload)

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "content_ready"
    assert len(calls) == 4
    assert "quality_retry" not in calls[0]
    assert calls[1]["quality_retry"]["critic_errors"]
    assert calls[1]["quality_retry"]["failed_surfaces"]
    assert calls[1]["quality_retry"]["requested_surfaces"] == ["master"]
    assert calls[1]["quality_retry"]["focus_surface"] == "master"
    assert calls[1]["channels"] == []
    assert "claims" not in calls[1]
    assert "authorized_context" not in calls[1]
    assert calls[1]["canonical_block_registry"]
    assert calls[2]["quality_retry"]["requested_surfaces"] == ["wechat_mp"]
    assert calls[2]["channels"] == ["wechat_mp"]
    assert calls[3]["quality_retry"]["requested_surfaces"] == ["xiaohongshu"]
    assert calls[3]["channels"] == ["xiaohongshu"]
    assert calls[1]["quality_retry"]["requested_channels"] == calls[1]["channels"]
    assert "master" in calls[1]["quality_retry"]["surface_contracts"]
    assert calls[1]["quality_retry"]["surface_contracts"]["master"][
        "authored_blocks_required"
    ] == 2
    assert calls[1]["quality_retry"]["required_shape"]
    assert any(
        "analytical block" in item
        for item in calls[1]["quality_retry"]["required_shape"]
    )
    assert "Do not describe the review process" in calls[1]["quality_retry"]["instruction"]
    assert "focus_surface is master" in calls[1]["quality_retry"]["instruction"]
    assert "server-template block_ref" in calls[1]["quality_retry"]["instruction"]
    assert "put no Arabic number or Chinese counted quantity" in calls[1]["quality_retry"]["instruction"]
    assert "真正改变的是" in calls[1]["quality_retry"]["instruction"]
    assert calls[1]["quality_retry"]["positive_pattern"]["hook"]
    assert "evidence_ids to []" in calls[1]["quality_retry"]["block_binding_rule"]
    assert any(
        diagnostic.endswith(":evidence_forbidden")
        for diagnostic in calls[1]["quality_retry"]["discarded_block_diagnostics"]
    )


def test_compose_keeps_good_surfaces_while_retry_fills_other_platforms(monkeypatch):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "xiaohongshu"]
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        candidate = _publishable_model_candidate(model_payload)
        required = [
            {"block_ref": block["block_ref"]}
            for block in model_payload["canonical_block_registry"]
            if block["required"]
        ]
        if len(calls) == 1:
            # The first attempt has a good master and WeChat article, but its
            # Xiaohongshu body is only a fact wrapper.
            next(
                variant
                for variant in candidate["platform_variants"]
                if variant["platform"] == "xiaohongshu"
            )["blocks"] = required
        else:
            # The retry is deliberately narrowed to Xiaohongshu. The
            # normalizer will restore deterministic placeholders for omitted
            # channels, and whole-surface aggregation must retain the earlier
            # passing WeChat article without asking the model to rewrite it.
            assert model_payload["channels"] == ["xiaohongshu"]
            assert model_payload["quality_retry"]["requested_channels"] == [
                "xiaohongshu"
            ]
            references = model_payload["quality_retry"]["validated_reference_copy"]
            assert references[0]["surface"] == "master"
            assert references[0]["authored_blocks"]
            assert all("block_ref" not in item for item in references)
            contract = model_payload["response_contract"]
            assert contract["exact_platform_variant_count"] == 1
            assert [item["platform"] for item in contract["platform_variants"]] == [
                "xiaohongshu"
            ]
            assert contract["platform_variants"][0]["blocks"][0]["evidence_ids"] == []
            assert any(
                "block_ref" in item
                for item in contract["platform_variants"][0]["blocks"]
            )
        return candidate

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "content_ready"
    assert len(calls) == 2
    assert {variant["platform"] for variant in data["platform_variants"]} == {
        "wechat_mp",
        "xiaohongshu",
    }
    assert next(
        check
        for check in data["critic"]["checks"]
        if check["code"] == "social_surface_retry_aggregation"
    )["passed"] is True


def test_compose_targeted_retries_only_request_the_remaining_surface(monkeypatch):
    payload = _compose_payload()
    payload["channels"] = ["wechat_mp", "wechat_channels", "xiaohongshu"]
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        candidate = _publishable_model_candidate(model_payload)
        required = [
            {"block_ref": block["block_ref"]}
            for block in model_payload["canonical_block_registry"]
            if block["required"]
        ]
        if len(calls) == 1:
            for platform in ("wechat_mp", "wechat_channels"):
                next(
                    variant
                    for variant in candidate["platform_variants"]
                    if variant["platform"] == platform
                )["blocks"] = required
        elif len(calls) == 2:
            assert model_payload["channels"] == ["wechat_mp"]
            assert model_payload["quality_retry"]["focus_surface"] == "wechat_mp"
        else:
            assert model_payload["channels"] == ["wechat_channels"]
            assert model_payload["quality_retry"]["focus_surface"] == "wechat_channels"
        return candidate

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "content_ready"
    assert len(calls) == 3


def test_compose_uses_curated_funds_policy_without_model_dependency(monkeypatch):
    fact = (
        "监管部门要求平台全额退还强制扣除酒店经营者的订单储备金，"
        "并要求企业全面整改及公开整改措施。"
    )
    channels = ["wechat_moments", "wechat_mp", "toutiao"]
    payload = {
        "objective": "形成可发布的平台经济行业分析",
        "topic": "平台经营资金边界",
        "audience": "平台经营者与行业从业者",
        "channels": channels,
        "as_of": "2026-08-02T00:00:00Z",
        "approved_proposal": {
            "title": "平台经营资金边界",
            "angle": "解释监管措施对经营资金和结算条款的影响。",
            "audience_value": "给经营者提供具体的资金与合同判断。",
            "key_points": [fact],
            "suggested_formats": channels,
            "cta": None,
            "first_person": False,
        },
        "claims": [
            {
                "text": fact,
                "kind": "fact",
                "evidence_ids": ["source-funds-remedy"],
                "verification_status": "verified",
            }
        ],
        "authorized_context": [
            {
                "record_id": "source-funds-remedy",
                "space": "industry",
                "record_type": "source_item",
                "title": "监管整改要求",
                "content": fact,
                "structured_data": {"source_text": fact},
                "status": "approved",
                "authorized_roles": ["industry"],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": "https://example.com/official-source",
                "source_tier": "official",
            }
        ],
    }
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        raise AssertionError("known funds-remedy policy must not wait for the model")

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "content_ready"
    assert calls == []
    assert any(
        check["code"] == "curated_industry_policy_repair"
        and check["passed"] is True
        for check in data["critic"]["checks"]
    )
    assert any(
        warning.startswith("curated_industry_policy_repair:master,")
        and "toutiao" in warning
        for warning in data["critic"]["warnings"]
    )
    toutiao = next(
        variant
        for variant in data["platform_variants"]
        if variant["platform"] == "toutiao"
    )
    assert "先把" not in toutiao["body"]
    assert "不是" not in toutiao["body"]
    assert "资金" in toutiao["body"]
    assert "结算" in toutiao["body"]


def test_compose_uses_commercial_human_review_policy_only_after_model_attempt(
    monkeypatch,
):
    capability = (
        "本测试系统不会把模型生成结果直接发布；负责人可以查看证据、修改方向、"
        "退回内容并完成终审，批准、驳回和恢复动作都会记录到同一条工作流中。"
    )
    channels = ["wechat_moments", "wechat_mp", "douyin"]
    payload = {
        "objective": "说明已批准的人工终审能力",
        "topic": "AI 内容流程的人工终审",
        "audience": "需要控制内容风险的团队负责人",
        "channels": channels,
        "as_of": "2026-08-02T00:00:00Z",
        "approved_proposal": {
            "title": "AI 内容流程的人工终审",
            "angle": "说明模型生成、人工确认和退回修改之间的责任边界。",
            "audience_value": "帮助团队判断自动化内容流程是否保留最终控制权。",
            "key_points": [capability],
            "suggested_formats": channels,
            "cta": None,
            "first_person": False,
        },
        "claims": [
            {
                "text": capability,
                "kind": "fact",
                "evidence_ids": ["cap-human-review"],
                "verification_status": "verified",
            }
        ],
        "authorized_context": [
            {
                "record_id": "cap-human-review",
                "space": "company_public",
                "record_type": "capability",
                "title": "人工终审能力",
                "content": capability,
                "structured_data": {},
                "status": "approved",
                "authorized_roles": ["commercial"],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": None,
                "source_tier": "trusted",
            }
        ],
    }
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        if len(calls) == 1:
            candidate = _publishable_model_candidate(model_payload)
            # Reproduce a realistic partial model result: two surfaces pass,
            # while the master repeats the source as its title and Moments is
            # only an evidence wrapper.  The final repair must replace the
            # entire bundle instead of retaining these unrelated live drafts.
            candidate["master_title"] = capability
            moments = next(
                variant
                for variant in candidate["platform_variants"]
                if variant["platform"] == "wechat_moments"
            )
            moments["blocks"] = [
                {"block_ref": block["block_ref"]}
                for block in model_payload["canonical_block_registry"]
                if block["required"]
            ]
            return candidate
        return {
            **fallback,
            "_error": "upstream timeout after request started",
            "_llm_attempted": True,
        }

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/commercial/compose",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert calls
    assert data["status"] == "content_ready", (
        data["critic"]["errors"],
        data["critic"]["warnings"],
        data["model"],
        len(calls),
    )
    assert any(
        check["code"] == "curated_commercial_policy_repair"
        and check["passed"] is True
        for check in data["critic"]["checks"]
    )
    assert any(
        warning.startswith("curated_commercial_policy_repair:master,")
        and "douyin" in warning
        for warning in data["critic"]["warnings"]
    )
    assert len(calls) == 2
    assert all(
        "version 2" not in variant["title"]
        for variant in data["platform_variants"]
    )
    assert "负责人" in data["master_content"]
    assert "先把事实和判断分开" not in data["master_content"]


def test_compose_uses_personal_confirm_and_replay_policy_after_model_attempt(
    monkeypatch,
):
    experience = (
        "在这次端到端验收中，我把原来追求自动发布的流程改成了AI先生成、"
        "负责人再确认、失败可以回放。真正费时间的不是多点一次确认，而是在"
        "错误内容已经发出去之后补救。"
    )
    channels = ["wechat_moments", "wechat_mp", "douyin"]
    payload = {
        "objective": "形成一篇关于自动发布与补救成本的个人内容",
        "topic": "从自动发布改成人工确认",
        "audience": "关注 AI 内容工作流的团队负责人",
        "channels": channels,
        "as_of": "2026-08-02T00:00:00Z",
        "approved_proposal": {
            "title": "从自动发布改成人工确认",
            "angle": "结合已批准经历说明为什么保留人工确认和失败回放。",
            "audience_value": "帮助团队理解自动化内容的补救成本。",
            "key_points": [experience],
            "suggested_formats": channels,
            "cta": None,
            "first_person": True,
        },
        "claims": [
            {
                "text": experience,
                "kind": "experience",
                "evidence_ids": ["personal-confirm-replay"],
                "verification_status": "verified",
            }
        ],
        "authorized_context": [
            {
                "record_id": "personal-confirm-replay",
                "space": "personal_approved",
                "record_type": "experience",
                "title": "人工确认与失败回放",
                "content": experience,
                "structured_data": {},
                "status": "approved",
                "authorized_roles": ["personal_ip"],
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
                "source_uri": None,
                "source_tier": "primary",
            }
        ],
    }
    calls: list[dict] = []

    async def fake_llm(system, model_payload, fallback, *args, **kwargs):
        calls.append(model_payload)
        return {
            **fallback,
            "_error": "upstream timeout after request started",
            "_llm_attempted": True,
        }

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)

    response = _request(
        "POST",
        "/api/v1/operators/personal_ip/compose",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert calls
    assert data["status"] == "content_ready", data["critic"]["errors"]
    assert data["master_title"] == "从自动发布到人工确认：补救成本的取舍"
    assert all("我" not in variant["title"] for variant in data["platform_variants"])
    assert any(
        check["code"] == "curated_personal_ip_policy_repair"
        and check["passed"] is True
        for check in data["critic"]["checks"]
    )


def test_compose_uses_the_same_byte_stable_role_prompt(monkeypatch):
    prompts: list[str] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        prompts.append(system)
        return fallback

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    compose_payload = _base_payload()
    compose_payload.update(
        {"approved_proposal": proposal["proposal"], "claims": proposal["claims"]}
    )
    composed = _request(
        "POST",
        "/api/v1/operators/industry/compose",
        json=compose_payload,
    )

    assert composed.status_code == 200
    assert len(prompts) == 2
    assert prompts[0].encode("utf-8") == prompts[1].encode("utf-8")
    assert composed.json()["system_prompt_sha256"] == hashlib.sha256(
        prompts[1].encode("utf-8")
    ).hexdigest()


def test_blocked_compose_never_calls_the_model(monkeypatch):
    payload = _compose_payload()
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("blocked evidence must not reach the model")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    payload["authorized_context"][0]["source_tier"] = "secondary"

    response = _request("POST", "/api/v1/operators/industry/compose", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "evidence_insufficient"
    assert response.json()["master_content"] is None
    assert called is False


def test_compose_rejects_extra_fields_and_cross_role_context_before_model(monkeypatch):
    extra = _compose_payload()
    cross_role = _compose_payload()
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid compose input must not reach the model")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    extra["direct_database"] = "please"
    extra_response = _request(
        "POST", "/api/v1/operators/industry/compose", json=extra
    )

    cross_role["authorized_context"][0]["space"] = "company_internal"
    cross_role["authorized_context"][0]["record_type"] = "delivery_boundary"
    cross_role_response = _request(
        "POST", "/api/v1/operators/industry/compose", json=cross_role
    )

    assert extra_response.status_code == 422
    assert cross_role_response.status_code == 403
    assert cross_role_response.json()["detail"]["code"] == "knowledge_space_denied"
    assert called is False


def test_system_prompt_bytes_do_not_change_between_propose_and_revise(monkeypatch):
    prompts: list[str] = []

    async def fake_llm(system, payload, fallback, *args, **kwargs):
        prompts.append(system)
        return fallback

    monkeypatch.setattr(marketing_api, "_llm_json", fake_llm)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = _base_payload()
    payload.update({"change_request": "Change the audience value only.", "previous": proposal})
    revised = _request("POST", "/api/v1/operators/industry/revise", json=payload)

    assert revised.status_code == 200
    assert len(prompts) == 2
    assert prompts[0].encode("utf-8") == prompts[1].encode("utf-8")
    assert hashlib.sha256(prompts[0].encode("utf-8")).hexdigest() == proposal["system_prompt_sha256"]


def test_revise_rejects_cross_profile_previous_output(monkeypatch):
    monkeypatch.delenv("HERMES_OPENAI_API_KEY", raising=False)
    proposal = _request("POST", "/api/v1/operators/industry/propose", json=_base_payload()).json()
    payload = {
        "objective": "Revise commercial proposal",
        "change_request": "Add a CTA.",
        "previous": proposal,
        "authorized_context": [],
        "as_of": "2026-07-16T08:00:00Z",
    }

    response = _request("POST", "/api/v1/operators/commercial/revise", json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "operator_role_mismatch"


def test_cross_role_context_is_rejected_before_model_execution(monkeypatch):
    called = False

    async def forbidden_llm(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not receive cross-role context")

    monkeypatch.setattr(marketing_api, "_llm_json", forbidden_llm)
    payload = _base_payload()
    payload["authorized_context"][0]["space"] = "company_internal"
    payload["authorized_context"][0]["record_type"] = "business_offer"

    response = _request("POST", "/api/v1/operators/industry/propose", json=payload)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "knowledge_space_denied"
    assert called is False


def test_bearer_auth_uses_the_existing_marketing_api_secret(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "operator-test-secret")

    denied = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        operator_auth=False,
    )
    allowed = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        headers={"Authorization": "Bearer operator-test-secret"},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "needs_input"


def test_only_public_health_is_available_without_a_service_secret(monkeypatch):
    assert _request("GET", "/health", operator_auth=False).status_code == 200

    protected_endpoints = _protected_endpoints()
    monkeypatch.delenv("HERMES_API_KEY", raising=False)

    for method, path, payload in protected_endpoints:
        kwargs = {"operator_auth": False}
        if payload is not None:
            kwargs["json"] = payload
        response = _request(method, path, **kwargs)
        assert response.status_code == 503, path


def test_every_protected_endpoint_rejects_the_wrong_bearer_key():
    for method, path, payload in _protected_endpoints():
        kwargs = {
            "operator_auth": False,
            "headers": {"Authorization": "Bearer definitely-wrong"},
        }
        if payload is not None:
            kwargs["json"] = payload
        response = _request(method, path, **kwargs)
        assert response.status_code == 401, path


def test_operator_authentication_fails_closed_when_secret_is_missing(monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)

    response = _request(
        "POST",
        "/api/v1/operators/personal_ip/propose",
        json={"objective": "Interview me"},
        operator_auth=False,
    )

    assert response.status_code == 503


def test_unknown_role_is_rejected_by_the_router():
    response = _request("POST", "/api/v1/operators/chief/propose", json={"objective": "Do everything"})

    assert response.status_code == 422


def test_caller_selected_llm_endpoint_never_inherits_the_service_key(monkeypatch):
    monkeypatch.setenv("HERMES_OPENAI_API_KEY", "server-owned-secret")

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("an incomplete endpoint override must not reach the network")

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", ForbiddenClient)
    fallback = {"status": "safe-fallback"}
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            fallback,
            openai_base_url="https://attacker.example/v1",
            openai_api_key=None,
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "openai_override_base_url_and_api_key_must_be_supplied_together"
    assert "server-owned-secret" not in str(result)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://93.184.216.34/v1",
        "http://127.0.0.1:8000/v1",
        "http://169.254.169.254/latest",
        "http://10.0.0.8/v1",
        "http://[::1]/v1",
    ],
)
def test_llm_endpoint_rejects_non_public_ip_targets(monkeypatch, base_url):
    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a private endpoint must not reach the network")

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", ForbiddenClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url=base_url,
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "invalid_openai_base_url"
    assert result.get("_llm_attempted") is not True
    assert "caller-owned-secret" not in str(result)


def test_llm_endpoint_rejects_dns_with_any_private_answer(monkeypatch):
    monkeypatch.setattr(
        marketing_api,
        "_resolve_host_addresses",
        lambda _host, _port: {"203.0.113.10", "10.0.0.8"},
    )

    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["_error"] == "invalid_openai_base_url"


def test_llm_endpoint_never_follows_redirects(monkeypatch):
    monkeypatch.setattr(
        marketing_api,
        "_resolve_host_addresses",
        lambda _host, _port: {"93.184.216.34"},
    )

    class RedirectClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **kwargs):
            assert url == "https://93.184.216.34/v1/chat/completions"
            assert kwargs["headers"]["Host"] == "model.example"
            assert kwargs["extensions"] == {"sni_hostname": "model.example"}
            return httpx.Response(
                307,
                headers={"Location": "http://169.254.169.254/latest"},
            )

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", RedirectClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert result["status"] == "safe-fallback"
    assert result["_error"] == "llm_redirect_forbidden"
    assert result["_llm_attempted"] is True


def test_llm_endpoint_connects_to_the_validated_ip_without_resolving_hostname_again(monkeypatch):
    resolutions = []

    def resolve_once(host, port):
        resolutions.append((host, port))
        return {"93.184.216.34"}

    monkeypatch.setattr(marketing_api, "_resolve_host_addresses", resolve_once)

    class PinnedClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **kwargs):
            assert "model.example" not in url
            assert url == "https://93.184.216.34/v1/chat/completions"
            assert kwargs["headers"]["Host"] == "model.example"
            assert kwargs["extensions"] == {"sni_hostname": "model.example"}
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"status":"ok"}'}}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(marketing_api.httpx, "AsyncClient", PinnedClient)
    result = asyncio.run(
        marketing_api._llm_json(
            "system",
            {"request": True},
            {"status": "safe-fallback"},
            openai_base_url="https://model.example/v1",
            openai_api_key="caller-owned-secret",
        )
    )

    assert resolutions == [("model.example", 443)]
    assert result["status"] == "ok"
    assert result["_llm_attempted"] is True
