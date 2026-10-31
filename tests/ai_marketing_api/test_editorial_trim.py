"""Pure edit/provenance contracts; actual private fixture assertions expose hashes only."""
import ast
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import pytest

from ai_marketing_api.operator_editorial_trim import candidate_digest, prepare_editorial_trim, text_digest

HERE = Path(__file__).resolve().parent
HERMES_ROOT = Path(__file__).resolve().parents[2]
SENTENCE = "至于这种安排是否已经带来更广泛的成本、效率或行业习惯变化，当前材料没有给出足够依据，不能替它下结论。"
RESIDUAL = "实际使用者需要先核对输入素材与输出要求，再判断流程是否符合当前任务的约束。"


def value(text=None, kind="opinion"):
    return {"master_title": "核对流程与输入要求", "blocks": [{"block_ref": "canonical-anchor"},
        {"kind": kind, "text": text if text is not None else RESIDUAL+SENTENCE, "evidence_ids": []}], "platform_variants": []}


def canonical(text="输入素材须按指定条件处理。", kind="fact", evidence_ids=("approved-source",)):
    return SimpleNamespace(text=text, kind=kind, evidence_ids=list(evidence_ids), binding_hash="canonical-anchor")


@pytest.mark.parametrize("kind", ["opinion", "transition"])
@pytest.mark.parametrize("sentence", [SENTENCE, "当前资料不足，无法判断成本变化。", "现有证据尚不足以判断成本影响。",
    "该安排是否带来成本变化，现有资料未提供充分证据，无法作出判断。"])
def test_exact_one_complete_nonassertive_sentence_removed_with_explicit_provenance(kind, sentence):
    original = value(RESIDUAL+sentence, kind); snapshot = deepcopy(original)
    edited, edits = prepare_editorial_trim(original, canonical_blocks=[canonical()], source_texts=["已核验的源材料。"])
    assert original == snapshot
    assert edited["blocks"][1]["text"] == RESIDUAL
    assert edited["blocks"][0] == original["blocks"][0]
    assert edited["master_title"] == original["master_title"] and edited["platform_variants"] == []
    assert len(edits) == 1
    edit = edits[0]
    assert edit["start"] == len(RESIDUAL) and edit["end"] == len(RESIDUAL+sentence)
    assert edit["removed_text_sha256"] == text_digest(sentence)
    assert edit["old_text_sha256"] == text_digest(RESIDUAL+sentence) and edit["new_text_sha256"] == text_digest(RESIDUAL)
    assert edit["source_candidate_sha256"] == candidate_digest(original) and edit["edited_candidate_sha256"] == candidate_digest(edited)
    assert edit["policy_version"] == "billing_boundary_editorial_trim.v1"
    assert edit["offset_unit"] == "unicode_codepoints" and edit["field"] == "text" and edit["surface"] == "master"


def test_unicode_offsets_and_all_nonremoved_whitespace_are_preserved():
    prefix = "😀"+RESIDUAL+"  "
    original = value(prefix+SENTENCE)
    edited, edits = prepare_editorial_trim(original)
    assert edited["blocks"][1]["text"] == prefix
    assert edits[0]["start"] == len(prefix) and edits[0]["end"]-edits[0]["start"] == len(SENTENCE)
    assert len(prefix.encode()) > edits[0]["start"]


@pytest.mark.parametrize("field,new", [("kind", "cta"), ("kind", "fact"), ("kind", "identity"), ("kind", "experience"),
    ("block_ref", "canonical-anchor"), ("canonical_block_ref", "canonical-anchor"), ("ref", "canonical-anchor"),
    ("one_of", {"block_ref": "canonical-anchor"}), ("canonical", True), ("canonical_block", {"claim_id": "claim-1"}),
    ("binding_hash", "a"*64), ("claim_id", "claim-1"), ("evidence_ids", ["source"]), ("evidence_refs", ["source"]),
    ("references", ["source"]), ("source_ids", ["source"]), ("origin", "approved_claim"), ("source_exact", True), ("locked", True), ("required", True)])
def test_bound_editorial_cta_and_factual_blocks_cannot_be_edited(field, new):
    original = value(); original["blocks"][1][field] = new
    before = candidate_digest(original)
    with pytest.raises(ValueError): prepare_editorial_trim(original)
    assert candidate_digest(original) == before


@pytest.mark.parametrize("text", ["成本确实下降了。", "成本并没有下降。", "价格并不更低。", "收费规则没有变化。", "这是免费使用。",
    "这肯定省钱。", "这已经更便宜。", "这已经更划算。", "这更加实惠。", "这不是更廉价。", "无需再花钱。",
    "5元。", "壹元。", "١元。", "$5。", "20%。", "每次开支。", "单价未知。"])
@pytest.mark.parametrize("location", ["same_block", "other_block", "title", "public_text"])
def test_deletion_cannot_hide_any_other_money_comparison_number_or_price_alias(text, location):
    original = value()
    if location == "same_block": original["blocks"][1]["text"] += text
    elif location == "other_block": original["blocks"].append({"kind": "transition", "text": text})
    elif location == "title": original["master_title"] += text
    else: original["blocks"][1]["public_text"] = text
    before = candidate_digest(original)
    with pytest.raises(ValueError): prepare_editorial_trim(original)
    assert candidate_digest(original) == before


@pytest.mark.parametrize("text", [SENTENCE, "值得观察。"+SENTENCE, "需要判断。"+SENTENCE, " " + SENTENCE,
    RESIDUAL+SENTENCE+SENTENCE, RESIDUAL+SENTENCE[:-1], RESIDUAL+SENTENCE[:-1]+"？", RESIDUAL+"“"+SENTENCE+"”"])
def test_empty_boilerplate_partial_quoted_or_multiple_sentence_removals_are_forbidden(text):
    with pytest.raises(ValueError): prepare_editorial_trim(value(text))


def test_multiple_matching_blocks_cannot_authorize_multiple_edits():
    original = value(); original["blocks"].append({"kind": "transition", "text": RESIDUAL+SENTENCE})
    with pytest.raises(ValueError, match="sentence_count"): prepare_editorial_trim(original)


@pytest.mark.parametrize("evidence_ids", [[], ["approved-source"]])
def test_exact_canonical_opinion_echo_is_never_unbound_regardless_of_evidence(evidence_ids):
    original = value()
    with pytest.raises(ValueError):
        prepare_editorial_trim(original, canonical_blocks=[canonical(original["blocks"][1]["text"], "opinion", evidence_ids)])


@pytest.mark.parametrize("source_kind", ["canonical", "source"])
def test_approved_source_sentence_cannot_be_removed(source_kind):
    kwargs = {"canonical_blocks": [canonical("源材料保留："+SENTENCE)]} if source_kind == "canonical" else {"source_texts": ["源材料保留："+SENTENCE]}
    with pytest.raises(ValueError, match="protected_source"): prepare_editorial_trim(value(), **kwargs)


def test_canonical_quantities_and_conditions_remain_exact():
    blocks = [canonical("指定模型支持 16:9，提示词不能改。")]
    original = value(); frozen = deepcopy(blocks)
    edited, _ = prepare_editorial_trim(original, canonical_blocks=blocks)
    assert blocks == frozen and edited["blocks"][0] == original["blocks"][0]


@pytest.mark.parametrize("text", ["费用按请求计费。", "这比原来更便宜。", "单价尚未确认。"])
def test_canonical_money_or_comparison_is_not_hidden_by_an_editorial_deletion(text):
    with pytest.raises(ValueError): prepare_editorial_trim(value(), canonical_blocks=[canonical(text)])


def native_definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_dependency_definitions", "exec"), scope)
    return SimpleNamespace(**scope)


class DependencyError(ValueError):
    def __init__(self, code, record_id, message):
        super().__init__(message)
        self.code, self.record_id = code, record_id


def h_definitions():
    return native_definitions(HERMES_ROOT/"ai_marketing_api/operator_content.py",
        {"ContentBlockKind", "ContentBlockOrigin", "_SURFACE_DEPENDENCIES", "_EDITORIAL_KIND_ALIASES", "_value", "_binding_hash", "_dependency_scan_text", "_surface_required_ids"},
        dict(re=re, Enum=Enum, Any=Any, Dict=Dict, Sequence=Sequence, hashlib=hashlib, json=json, ContextAuthorizationError=DependencyError))


def test_unchanged_H_native_gate_keeps_full_input_condition_closure():
    required = h_definitions()._surface_required_ids
    plan = {"surfaces": {"master": ["claim-1"]}, "fact_roles": {"billing": [], "input_constraints": ["claim-3"]},
        "long_surface_selection_version": "thesis_selection.v2", "reference_only_claim_ids": [], "claim_pool": [], "source_evidence_bindings": [], "preparation_authorization": {}}
    texts = {"claim-1": "主事实完整保留。", "claim-2": "提示词不能改。", "claim-3": "输入条件须完整保留。"}
    blocks = [SimpleNamespace(claim_id=claim_id, text=text, kind="fact", evidence_ids=["source"], binding_hash="ref-"+claim_id)
        for claim_id, text in texts.items()]
    raw = [{"block_ref": "ref-claim-2"}, {"kind": "opinion", "text": RESIDUAL+SENTENCE}]
    with pytest.raises(DependencyError) as failure:
        required(plan, "master", raw, canonical_blocks=blocks)
    assert failure.value.code == "surface_fact_plan_dependency_evidence_missing"
    edited, _ = prepare_editorial_trim(value())
    raw[1]["text"] = edited["blocks"][1]["text"]
    assert required(plan, "master", raw, canonical_blocks=blocks) == {"claim-1", "claim-3"}


def private_inputs():
    # Public GHA has neither this private capture bundle nor the immutable Main
    # snapshot. Only this one integration case skips; synthetic tests use H alone.
    for ancestor in (HERE, *HERE.parents):
        for tmp in (ancestor, ancestor/".data/tmp"):
            fixture = tmp/"editorial_r64_child_replay_inputs_root02_actual_private_20261031.json"
            artifacts = tmp/"editorial_billing_boundary_editorial_trim_candidate_20261031_v1"
            paths = (fixture, artifacts/"actual_edit_metadata_private.json", artifacts/"immutable_main0f4_operator_surface_fact_plan.py")
            if all(path.is_file() for path in paths): return paths
    return None


def test_actual_private_fixture_reproduces_metadata_without_output_or_modifying_source():
    inputs = private_inputs()
    if inputs is None:
        pytest.skip("Private actual fixture/metadata/immutable Main source are not distributed in H GHA")
    fixture, metadata, main_source = inputs
    raw = fixture.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "9ec969caf85da67833469f6cd254b9460db807162b3cbd6cb9f9d742adf676e7"
    data = json.loads(raw)["PRIVATE_REPLAY_INPUTS_NEVER_PRINT_TO_CHAT"]
    candidate = data["writer_candidate"]["value"]; compose = data["logical_payload"]["compose"]
    reference_only = set(compose["surface_fact_plan"]["reference_only_claim_ids"])
    blocks = [SimpleNamespace(**claim) for index, claim in enumerate(compose["claims"], 1) if "claim-"+str(index) not in reference_only]
    sources = [record["content"] for record in compose["authorized_context"]]
    before_compose = candidate_digest(compose)
    edited, edits = prepare_editorial_trim(candidate, canonical_blocks=blocks, source_texts=sources)
    saved = json.loads(metadata.read_text(encoding="utf-8"))["editorial_edit"]
    assert candidate_digest(edits[0]) == candidate_digest(saved)
    assert candidate_digest(edited) == saved["edited_candidate_sha256"]
    assert candidate_digest(candidate) == saved["source_candidate_sha256"]
    assert candidate_digest(compose) == before_compose
    restored = deepcopy(edited); index = saved["block_index"]
    restored["blocks"][index]["text"] = candidate["blocks"][index]["text"]
    assert candidate_digest(restored) == candidate_digest(candidate)
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest()
    # Optional local Main late-gate regression remains inside this private case.
    assert hashlib.sha256(main_source.read_bytes()).hexdigest() == "1a64cf1461a660d1f730ad49fbd6a8f5b17ebe9e3cd554a7f423ac14d1ff503f"
    main = native_definitions(main_source, {"DEPENDENCIES", "_dependency_scan_text", "required_claim_ids"}, dict(re=re, Mapping=Mapping))
    h = h_definitions(); canonical_by_ref = {}; canonical_texts = {}
    report_ids = {fact["record_id"] for fact in (compose.get("reporting_context") or {}).get("facts", [])}
    for index, claim in enumerate(compose["claims"], 1):
        claim_id = "claim-"+str(index)
        if claim_id in reference_only: continue
        item = dict(kind=claim["kind"], text=claim["text"], claim_id=claim_id, evidence_ids=claim["evidence_ids"], origin="approved_claim", locked=True, required=True)
        if set(claim["evidence_ids"]) & report_ids: item["source_display"] = "report_attribution"
        elif compose.get("fact_projection_version") is not None: item["fact_projection_version"] = compose["fact_projection_version"]
        canonical_by_ref[h._binding_hash(item)] = claim_id
        canonical_texts[claim_id] = claim["text"]
    referenced = set()
    def expanded_body(value):
        parts = [value["master_title"]]
        for block in value["blocks"]:
            ref = block if isinstance(block, str) else block.get("block_ref") or block.get("canonical_block_ref") or block.get("ref")
            if ref:
                claim_id = canonical_by_ref[ref]; referenced.add(claim_id); parts.append(canonical_texts[claim_id])
            else: parts.append(block.get("text") or "")
        return "\n\n".join(parts)
    original_body, edited_body = expanded_body(candidate), expanded_body(edited)
    with pytest.raises(ValueError, match="dependency_evidence_missing"):
        main.required_claim_ids(compose["surface_fact_plan"], "master", original_body, canonical_texts=canonical_texts, referenced_claim_ids=referenced)
    assert main.required_claim_ids(compose["surface_fact_plan"], "master", edited_body,
        canonical_texts=canonical_texts, referenced_claim_ids=referenced) == set(canonical_texts)
