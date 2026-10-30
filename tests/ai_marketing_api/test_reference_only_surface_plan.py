"""Focused native compose contracts. No app entry, model, DB or network."""
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace

import pytest
from ai_marketing_api import operator_content as content
from ai_marketing_api.operator_runtime import AuthorizedContext, ContextAuthorizationError, OperatorClaim, OperatorRegistry, OperatorRole
from tests.ai_marketing_api.test_operator_content_runtime import _request, _run

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
URI = "https://source.example/article/"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def resign(request):
    body = {key: value for key, value in request.surface_fact_plan.items()
            if key not in {"signature", "certificate", "plan_sha256"}}
    request.surface_fact_plan = {**body, "plan_sha256": content._plan_digest(body),
        "signature": hmac.new(b"offline-hermes-secret", ("surface_fact_plan.v1\n" + content._plan_digest(body)).encode(), hashlib.sha256).hexdigest()}


def fixture(monkeypatch, *, same_text=False, mixed=False):
    monkeypatch.setenv("HERMES_API_KEY", "offline-hermes-secret")
    facts = ["官方教程介绍样片到正式视频的流程。", "新建样片任务从创建时间起有效14天。",
             "文件链接保存24小时，最多下载100次。", "女孩抱着狐狸的样片用于确认镜头方向。",
             "正式视频自动复用提示词和参考素材，不能重传；水印可以另设。"]
    source = "完整教程的表格费用信息999999和提示词复用说明。\n" + "\n".join(facts) + "\n样片模式\n#\n" + "输入参数表格说明。" * 80
    document = {"status": "ready", "url": URI, "source_text": source, "source_text_hash": sha(source),
                "retrieved_at": "2026-10-29T10:00:00Z", "coverage": {"text": "read"}}
    contexts = [AuthorizedContext(record_id=f"fact-{index}", space="industry", record_type="source_fact",
        content=fact, source_uri=URI, source_tier="official", authorized_roles=["industry"],
        structured_data={"source_document": deepcopy(document)}) for index, fact in enumerate(facts, 1)]
    contexts.append(AuthorizedContext(record_id="raw-document", space="industry", record_type="source_item",
        content=source, source_uri=URI, source_tier="official", authorized_roles=["industry"],
        structured_data={"operator_input": True, "input_kind": "public_document", "verification": {"verified": True},
                         "source_document": deepcopy(document)}))
    claims = [OperatorClaim(text=fact, kind="fact", evidence_ids=[f"fact-{index}"], verification_status="verified")
              for index, fact in enumerate(facts, 1)]
    if mixed:
        claims[2].evidence_ids.append("raw-document")
    claims.append(OperatorClaim(text=facts[2] if same_text else source[:480], kind="fact", evidence_ids=["raw-document"], verification_status="verified"))
    request = _request(contexts, claims, channels=["wechat_mp", "wechat_moments"])
    request.as_of = NOW
    request.genre = "industry_analysis"
    request.approved_proposal.title = "样片确认与文件保存"
    request.long_surface_selection_version = "thesis_selection.v2"
    ids = [f"claim-{index}" for index in range(1, 6)]
    body = {"schema_version": "surface_fact_plan.v1", "scope": {"role_id": "industry", "genre": request.genre,
        "proposal_id": "proposal-native", "objective_sha256": sha(request.objective)},
        "claim_pool": [{"claim_id": f"claim-{index}", "kind": "fact", "text_sha256": sha(claim.text), "evidence_ids": claim.evidence_ids[:]}
                       for index, claim in enumerate(claims, 1)],
        "surfaces": {"master": ["claim-1"], "wechat_mp": ["claim-1"], "wechat_moments": ["claim-1"]},
        "fact_roles": {"anchor": ids, "billing": [], "input_constraints": ["claim-5"]},
        "preparation_authorization": {"schema_version": "operator.surface_preparation_authority.v1", "mode": "mandate_preparation_only",
            "publication_authorized": False, "run_id": "run-native", "mandate_version": 1,
            "proposal_authorization_id": "proposal-native", "proposal_authorization_version": 1,
            "planner_request_sha256": "1" * 64, "planner_selection_sha256": "2" * 64, "research_digest_sha256": "3" * 64},
        "preparation_selection_sha256": "4" * 64, "narrative_required": True,
        "editorial_intent": {"schema_version": "operator.preparation_editorial_intent.v1", "author": "independent workflow editorial planner",
            "short_surface_reasons": {"wechat_moments": "解释创作确认与文件保存的安排。"}, "reported_case_claim_ids": [], "case_gap": "材料没有具体交接案例。"},
        "long_surface_selection_version": "thesis_selection.v2", "reference_only_claim_ids": ["claim-6"],
        "source_evidence_bindings": [{"record_id": "raw-document", "source_uri": URI, "source_text_sha256": sha(source),
            "curated_record_ids": [f"fact-{index}" for index in range(1, 6)]}]}
    request.surface_fact_plan = body
    resign(request)
    registry = OperatorRegistry()
    authorized = registry.authorize(OperatorRole.INDUSTRY, request.authorized_context, NOW)
    return request, authorized


def candidate(request, contexts):
    payload = content.content_request_payload("industry", request, contexts)
    first = payload["canonical_block_registry"][0]["block_ref"]
    return {"_model": "offline-native", "master_title": "样片确认与文件保存", "blocks": [{"block_ref": first}],
        "platform_variants": [{"platform": surface, "title": "样片确认与文件保存", "blocks": [{"block_ref": first}]} for surface in request.channels]}


def test_complete_pool_and_document_survive_but_writer_and_fallback_are_atomic(monkeypatch):
    request, contexts = fixture(monkeypatch)
    before = request.model_dump(mode="json")
    payload = content.content_request_payload("industry", request, contexts)
    assert [row["claim_id"] for row in payload["canonical_block_registry"]] == [f"claim-{i}" for i in range(1, 6)]
    assert len(payload["claims"]) == len(payload["approved_proposal"]["key_points"]) == 5
    assert payload["reference_only_evidence"][0]["text"] == request.claims[5].text
    assert payload["authorized_context"][-1]["content"] == contexts[-1].content
    assert payload["authorized_context"][-1]["evidence_role"] == "source_reference_only"
    fallback = content.build_content_fallback(OperatorRegistry(), "industry", request, contexts)
    assert all(row.get("claim_id") != "claim-6" for row in fallback["blocks"])
    assert request.model_dump(mode="json") == before
    assert len(request.surface_fact_plan["claim_pool"]) == 6
    assert "select additional canonical references" in payload["surface_fact_plan_rule"]


@pytest.mark.parametrize("mode", ["ref", "bare", "claim", "echo", "fulltext", "public_text", "tampered", "alias", "nested"])
def test_reference_publication_is_a_hard_failure(monkeypatch, mode):
    request, contexts = fixture(monkeypatch)
    raw = content._build_all_approved_blocks(OperatorRole.INDUSTRY, request, contexts)[0][-1]
    payload = candidate(request, contexts)
    obj = {"ref": {"block_ref": raw.binding_hash}, "bare": raw.binding_hash,
           "claim": {"claim_id": "claim-6", "kind": "opinion", "text": "仍需按实际工作安排确认。"},
           "echo": {"kind": "fact", "text": raw.text, "evidence_ids": raw.evidence_ids},
           "fulltext": {"kind": "opinion", "text": contexts[-1].content},
           "public_text": {"block_ref": payload["blocks"][0]["block_ref"], "public_text": raw.text},
           "tampered": {"block_ref": raw.binding_hash, "text": "无关文字"},
           "alias": {"ref": raw.binding_hash}, "nested": {"one_of": {"claim_id": "claim-6", "text": "其他文字"}}}[mode]
    payload["blocks"].append(obj)
    try:
        output = _run(OperatorRole.INDUSTRY, request, payload)
    except ContextAuthorizationError as error:
        assert error.code == "surface_fact_plan_reference_only_claim_published"
    else:
        assert "surface_fact_plan_reference_only_claim_published" in output.critic.errors
        assert not output.critic.passed


def test_same_text_atomic_claim_and_mixed_group_remain_publishable(monkeypatch):
    for mixed in (False, True):
        request, contexts = fixture(monkeypatch, same_text=True, mixed=mixed)
        payload = content.content_request_payload("industry", request, contexts)
        atomic = next(row for row in payload["canonical_block_registry"] if row["claim_id"] == "claim-3")
        raw = candidate(request, contexts)
        raw["blocks"].append({"block_ref": atomic["block_ref"], "public_text": request.claims[2].text})
        result = _run(OperatorRole.INDUSTRY, request, raw)
        assert "surface_fact_plan_reference_only_claim_published" not in result.critic.errors
        block = next(row for row in result.blocks if row.claim_id == "claim-3")
        assert block.evidence_ids == (["fact-3", "raw-document"] if mixed else ["fact-3"])


@pytest.mark.parametrize("change", ["metadata", "hash", "uri", "curated_body", "raw_kind", "read", "retrieved", "scope", "version_conflict", "private"])
def test_source_drift_cannot_be_hidden_by_reference_classification(monkeypatch, change):
    request, contexts = fixture(monkeypatch)
    if change == "metadata": request.surface_fact_plan["reference_only_claim_ids"] = ["claim-5", "claim-6"]; resign(request)
    elif change == "hash": request.authorized_context[-1].structured_data["source_document"]["source_text_hash"] = "0" * 64
    elif change == "uri": request.authorized_context[-1].source_uri = "https://other.example/article"
    elif change == "curated_body": request.authorized_context[0].content += "changed"
    elif change == "raw_kind": request.authorized_context[-1].structured_data["input_kind"] = "rss"
    elif change == "read": request.authorized_context[-1].structured_data["source_document"]["coverage"]["text"] = "unread"
    elif change == "retrieved": request.authorized_context[-1].structured_data["source_document"]["retrieved_at"] = "2027-01-01T00:00:00Z"
    elif change == "scope": request.authorized_context[-1].structured_data["source_document"]["scope"] = {"kind": "section"}
    elif change == "private": request.authorized_context[-1].structured_data["publishable"] = False
    else:
        document = request.authorized_context[0].structured_data["source_document"]
        document["source_text"] += "new source version"
        document["source_text_hash"] = sha(document["source_text"])
    with pytest.raises(ContextAuthorizationError):
        content.content_request_payload("industry", request, request.authorized_context)


def test_invalid_raw_original_claim_is_validated_before_filtering(monkeypatch):
    request, contexts = fixture(monkeypatch)
    request.claims[-1].text = "This text never appeared in the raw source."
    request.surface_fact_plan["claim_pool"][-1]["text_sha256"] = sha(request.claims[-1].text)
    resign(request)
    blocks, errors = content._build_approved_blocks(OperatorRole.INDUSTRY, request, contexts)
    assert "claim_not_verbatim:claim-6" in errors
    assert all(block.claim_id != "claim-6" for block in blocks)
    with pytest.raises(ContextAuthorizationError): content.content_request_payload("industry", request, contexts)


def test_raw_number_does_not_ground_a_title(monkeypatch):
    request, contexts = fixture(monkeypatch)
    blocks, _ = content._build_approved_blocks(OperatorRole.INDUSTRY, request, contexts)
    assert "999999" not in content._safe_master_title(request, blocks, "样片确认999999", role=OperatorRole.INDUSTRY)


@pytest.mark.parametrize("version", [None, "thesis_selection.v1"])
def test_v2_fields_cannot_reinterpret_an_old_signed_plan(monkeypatch, version):
    request, contexts = fixture(monkeypatch)
    request.long_surface_selection_version = version
    if version is None: request.surface_fact_plan.pop("long_surface_selection_version")
    else: request.surface_fact_plan["long_surface_selection_version"] = version
    resign(request)
    with pytest.raises(ContextAuthorizationError): content.content_request_payload("industry", request, contexts)


def test_conditional_closure_uses_atomic_roles_not_raw_words(monkeypatch):
    request, contexts = fixture(monkeypatch)
    blocks, _ = content._build_approved_blocks(OperatorRole.INDUSTRY, request, contexts)
    plan = content._validated_surface_fact_plan(request)
    assert content._surface_required_ids(plan, "master", [], "样片确认与文件保存", canonical_blocks=blocks) == {"claim-1"}
    assert content._surface_required_ids(plan, "master", [], "提示词复用之前", canonical_blocks=blocks) == {"claim-1", "claim-5"}
    with pytest.raises(ContextAuthorizationError) as error:
        content._surface_required_ids(plan, "master", [], "需要比较成本", canonical_blocks=blocks)
    assert error.value.code == "surface_fact_plan_dependency_evidence_missing"


def pending_output(request, contexts):
    from ai_marketing_api.operator_analysis import derive_pending
    output = _run(OperatorRole.INDUSTRY, request, candidate(request, contexts))
    block = content._stamp_block(content.ContentBlock(block_id="analysis-example", kind="opinion",
        text="团队可以分别安排方向确认与文件保存。", verification_status="opinion", analysis_pending=True),
        origin=content.ContentBlockOrigin.MODEL_EDITORIAL, locked=False, required=False)
    output.blocks.append(block)
    output.pending_editorial_analysis = derive_pending(output, request.claims)
    return output


class FakeVerifier:
    def __init__(self):
        self.bodies = []

    async def call(self, llm, system, body, *args, **kwargs):
        self.bodies.append(deepcopy(body))
        return {"_model": "offline-independent-verifier", "reviews": [],
            "analysis_reviews": [{"analysis_id": row["analysis_id"], "verdict": "supported",
                "reason": "The synthetic bounded reasoning follows the supplied complete atomic facts."}
                for row in body["analysis_verification"]["candidates"]]}


def run_native(coroutine):
    # The fake async boundary returns immediately. Driving it directly keeps
    # Windows asyncio from creating a socketpair under the strict I/O guard.
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    finally:
        coroutine.close()
    raise AssertionError("Unexpected asynchronous I/O in a native focused test")


def test_verifier_atomic_model_view_keeps_original_ids_and_full_pool_pending_digest(monkeypatch):
    from ai_marketing_api.operator_grounding import apply_grounded_paraphrases
    from ai_marketing_api.operator_analysis import claim_pool, digest
    request, contexts = fixture(monkeypatch)
    output = pending_output(request, contexts)
    expected = digest(claim_pool(request.claims))
    run = FakeVerifier()
    result = run_native(apply_grounded_paraphrases(output=output, raw_candidates=[], run=run,
        llm=None, system="offline", model_args=(), approved_claims=request.claims, compose_request=request))
    view = run.bodies[0]["analysis_verification"]
    assert [row["claim_id"] for row in view["approved_claims"]] == [f"claim-{i}" for i in range(1, 6)]
    assert all(row["text"] != request.claims[-1].text for row in view["approved_claims"])
    assert {row["evidence_sha256"] for row in view["candidates"]} == {expected}
    assert all(row.evidence_sha256 == expected for row in result.analysis_binding_reviews)
    assert request.surface_fact_plan["claim_pool"] == claim_pool(request.claims)


@pytest.mark.parametrize("change", ["missing_request", "signature", "output_plan", "output_version", "partial_pool", "invalid_hidden_claim"])
def test_verifier_rejects_invalid_v2_binding_before_model_call(monkeypatch, change):
    from ai_marketing_api.operator_grounding import apply_grounded_paraphrases
    request, contexts = fixture(monkeypatch)
    output = pending_output(request, contexts)
    approved = request.claims
    compose = request
    if change == "missing_request": compose = None
    elif change == "signature": request.surface_fact_plan["signature"] = "0" * 64
    elif change == "output_plan": output.surface_fact_plan_sha256 = "0" * 64
    elif change == "output_version": output.long_surface_selection_version = None
    elif change == "partial_pool": approved = request.claims[:-1]
    else:
        request.claims[-1].text = "An unsupported hidden raw claim."
        request.surface_fact_plan["claim_pool"][-1]["text_sha256"] = sha(request.claims[-1].text)
        resign(request)
        output.surface_fact_plan_sha256 = request.surface_fact_plan["plan_sha256"]
    run = FakeVerifier()
    with pytest.raises(ContextAuthorizationError):
        run_native(apply_grounded_paraphrases(output=output, raw_candidates=[], run=run,
            llm=None, system="offline", model_args=(), approved_claims=approved, compose_request=compose))
    assert run.bodies == []


@pytest.mark.parametrize("version", [None, "thesis_selection.v1"])
def test_old_verifier_view_and_full_pool_identity_are_unchanged(monkeypatch, version):
    from ai_marketing_api.operator_grounding import apply_grounded_paraphrases
    from ai_marketing_api.operator_analysis import claim_pool, digest, derive_pending
    request, contexts = fixture(monkeypatch)
    output = pending_output(request, contexts)
    request.long_surface_selection_version = version
    output.long_surface_selection_version = version
    output.pending_editorial_analysis = derive_pending(output, request.claims)
    run = FakeVerifier()
    run_native(apply_grounded_paraphrases(output=output, raw_candidates=[], run=run,
        llm=None, system="offline", model_args=(), approved_claims=request.claims))
    view = run.bodies[0]["analysis_verification"]
    assert view["approved_claims"] == [{**row, "text": claim.text}
        for row, claim in zip(claim_pool(request.claims), request.claims)]
    assert {row["evidence_sha256"] for row in view["candidates"]} == {digest(claim_pool(request.claims))}


def test_internal_verifier_callers_keep_legacy_kwargs_and_pass_v2_request_without_app_import():
    seen = 0
    for filename, variable in (("main.py", "payload"), ("operator_steps.py", "request")):
        source = Path(content.__file__).with_name(filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "apply_grounded_paraphrases":
                continue
            seen += 1
            expanded = [keyword.value for keyword in node.keywords if keyword.arg is None]
            assert len(expanded) == 1
            expression = compile(ast.Expression(expanded[0]), filename, "eval")
            for version in (None, "thesis_selection.v1", "thesis_selection.v2"):
                request = SimpleNamespace(long_surface_selection_version=version)
                kwargs = eval(expression, {"__builtins__": {}}, {variable: request})
                assert kwargs == ({"compose_request": request} if version == "thesis_selection.v2" else {})
    assert seen == 3
