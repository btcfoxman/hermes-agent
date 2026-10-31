"""Synthetic wire/certificate/store contracts; no provider or content-quality claim."""
import base64, hashlib, json
from copy import deepcopy
import pytest
from ai_marketing_api import operator_artifact_derivation as d
from ai_marketing_api.operator_receipts import EditorialReceiptStore

def fixture(*,payload=None,candidate=None):
    child,parent = "text_"+"a"*40,"text_"+"b"*40
    candidate = candidate or {"master_title":"Synthetic sourced title","blocks":[{"kind":"opinion","text":"所以，现有官方教程支持的结论是流程上的，而不是行业趋势上的：它提醒创作者把创作判断、正式生成和文件留存分开管理。至于这种安排是否已经带来更广泛的成本、效率或行业习惯变化，当前材料没有给出足够依据，不能替它下结论。","evidence_ids":[]}],"platform_variants":[]}
    usage = {"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"completion_tokens_details":{"reasoning_tokens":2}}
    payload = payload or {"compose":{"genre":"industry_analysis","long_surface_selection_version":"thesis_selection.v2","surface_fact_plan":{"preparation_authorization":{"fixture":True}}}}
    wire = d.digest({**payload,"budget_reservation_id":child})
    envelope = {"model":"gpt-5.6-terra","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":json.dumps(candidate,ensure_ascii=False)}}],"usage":usage}
    raw = d.canonical(envelope); candidate_raw = d.canonical(candidate)
    event = {"request_id":"c"*32,"provider_request_id":"synthetic-provider","model":"gpt-5.6-terra","stage":"compose",
        "reservation_id":child,"status":"returned","http_status":200,"error_code":None,"usage":usage}
    trace = {"origin":"model","model_calls":1,"repair_calls":0,"stop_reason":"step_artifact_invalid","editorial_review":"not_requested",
        "provider_requests":[event],"attempts":[{"stage":"compose","outcome":"returned","upstream_attempted":True,"model":event["model"],
            "error_code":None,"provider_requests":[deepcopy(event)],"input_tokens":10,"output_tokens":20}]}
    response = {"schema_version":"operator.editorial_step.v1","step":"draft_master","status":"unavailable","error_code":"step_artifact_invalid",
        "artifact":None,"publishable":False,"requires_human_review":True,"generation_trace":trace,"private_generation_receipt":{
            "schema_version":"operator.private_generation_receipt.v1","role_id":"industry","step":"draft_master","budget_reservation_id":child,
            "request_sha256":wire,"writer_candidate":{"type":"dict","source":"model","retained":True,"byte_source":"canonical_json",
                "bytes":len(candidate_raw),"sha256":hashlib.sha256(candidate_raw).hexdigest(),"value":candidate},"provider_responses":[{
                    "request_id":event["request_id"],"provider_request_id":event["provider_request_id"],"http_status":200,"response_received":True,
                    "parse_stage":"complete","error_code":None,"body":{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"complete":True,
                        "retained":True,"encoding":"base64","hash_scope":"complete_body","byte_source":"decoded_http_body","value":base64.b64encode(raw).decode()}}]}}
    spec = {"schema_version":d.REQUEST_SCHEMA,"policy_version":d.POLICY,"parent_request_id":parent,"source_request_id":child,
        "source_wire_sha256":wire,"source_response_sha256":d.digest(response),"provider_body_sha256":hashlib.sha256(raw).hexdigest(),
        "writer_candidate_sha256":hashlib.sha256(candidate_raw).hexdigest(),"settlement_sha256":"d"*64,"retry_binding_sha256":"e"*64,"logical_payload":payload}
    return spec,response

def test_exact_retained_source_is_copied_without_changing_source():
    spec,response = fixture(); before = deepcopy(response)
    value = d.source_candidate(spec,response)
    assert value["_model"] == "gpt-5.6-terra" and value["blocks"] == response["private_generation_receipt"]["writer_candidate"]["value"]["blocks"]
    assert response == before and spec["source_response_sha256"] == d.digest(response)

@pytest.mark.parametrize("defect",["policy","role","step","wire","responsehash","bodyhash","candidatehash","fee_binding","http","usage","partial","fallback","bound","nonv2","refusal","length"])
def test_ineligible_or_mismatched_source_is_rejected(defect):
    spec,response = fixture(); private = response["private_generation_receipt"]; received = private["provider_responses"][0]
    if defect == "policy":
        with pytest.raises(Exception): d.DerivationRequest.model_validate({**spec,"policy_version":"other.v1"})
        with pytest.raises(d.DerivationError): d.source_candidate({**spec,"policy_version":"other.v1"},response)
        return
    if defect == "role": private["role_id"] = "commercial"
    elif defect == "step": response["step"] = "adapt_platform"
    elif defect == "wire": spec["source_wire_sha256"] = "0"*64
    elif defect == "responsehash": spec["source_response_sha256"] = "0"*64
    elif defect == "bodyhash": spec["provider_body_sha256"] = "0"*64
    elif defect == "candidatehash": spec["writer_candidate_sha256"] = "0"*64
    elif defect == "fee_binding":
        with pytest.raises(Exception): d.DerivationRequest.model_validate({**spec,"settlement_sha256":"not-a-hash"})
        return
    elif defect == "http": received["http_status"] = 503
    elif defect == "usage": response["generation_trace"]["provider_requests"][0]["usage"] = None
    elif defect == "partial": received["body"]["complete"] = False
    elif defect == "fallback": private["writer_candidate"]["source"] = "fallback"
    elif defect == "bound":
        block = private["writer_candidate"]["value"]["blocks"][0]; block["block_ref"] = "f"*64
    elif defect == "nonv2": spec["logical_payload"]["compose"]["long_surface_selection_version"] = None
    elif defect in {"refusal","length"}:
        raw = base64.b64decode(received["body"]["value"]); envelope = json.loads(raw)
        if defect == "refusal": envelope["choices"][0]["message"]["refusal"] = "No"
        else: envelope["choices"][0]["finish_reason"] = "length"
        raw = d.canonical(envelope); received["body"].update(value=base64.b64encode(raw).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()); spec["provider_body_sha256"] = received["body"]["sha256"]
    if defect != "responsehash": spec["source_response_sha256"] = d.digest(response)
    with pytest.raises(Exception): d.source_candidate(spec,response)

def test_exact_retained_source_real_strict_normalizer_after_single_editorial_trim(monkeypatch):
    from ai_marketing_api import operator_content as content
    from ai_marketing_api.operator_runtime import OperatorRegistry
    from ai_marketing_api.operator_style import editorial_scope
    from tests.ai_marketing_api.test_reference_only_surface_plan import fixture as native_fixture
    request,contexts = native_fixture(monkeypatch)
    registry = OperatorRegistry()
    with editorial_scope(request.genre):
        contract = content.content_request_payload("industry",request,contexts)
    _spec,original = fixture()
    opinion = original["private_generation_receipt"]["writer_candidate"]["value"]["blocks"][0]
    candidate = {"master_title":request.approved_proposal.title,
        "blocks":[*({"block_ref":row["block_ref"]} for row in contract["canonical_block_registry"]),deepcopy(opinion)],"platform_variants":[]}
    spec,response = fixture(payload={"compose":request.model_dump(mode="json"),"surface":"master"},candidate=candidate)
    before = deepcopy(response)
    artifact,provenance = d.normalize_saved(registry,spec,response)
    assert artifact["status"] == "content_ready" and artifact["critic"]["errors"] == [] and artifact["platform_variants"] == []
    assert artifact["generation_trace"] == before["generation_trace"] and response == before
    assert {block.get("claim_id") for block in artifact["blocks"] if block.get("claim_id")} == {"claim-"+str(index) for index in range(1,6)}
    assert provenance["actor_kind"] == "ai_editor" and provenance["editorial_intervention"] is True
    assert provenance["source_writer_candidate_sha256"] == spec["writer_candidate_sha256"] and len(provenance["edits"]) == 1
    edit = provenance["edits"][0]
    assert edit["block_index"] == 5 and edit["offset_start"] == 56 and edit["offset_end"] == 106

def test_native_journal_and_separate_derivation_store_exactly_once(tmp_path,monkeypatch):
    spec,response = fixture(); old = EditorialReceiptStore(tmp_path/"original.sqlite3")
    wire = {**spec["logical_payload"],"budget_reservation_id":spec["source_request_id"]}
    token,_ = old.claim("industry","draft_master",spec["source_request_id"],d.digest(wire)); old.finish("industry","draft_master",spec["source_request_id"],token,response)
    before = old.path.read_bytes(); calls = []
    monkeypatch.setenv("HERMES_API_KEY","synthetic-key")
    monkeypatch.setattr(d,"EditorialReceiptStore",lambda:old)
    def normalized(_registry,_spec,saved):
        calls.append(deepcopy(saved)); return ({"role_id":"industry","genre":"industry_analysis","status":"content_ready","generation_trace":deepcopy(saved["generation_trace"])},
            {"actor_kind":"ai_editor","editorial_intervention":True,"source_writer_candidate_sha256":spec["writer_candidate_sha256"],"edited_candidate_sha256":"f"*64,"edits":[{"store_fixture":True}]})
    monkeypatch.setattr(d,"normalize_saved",normalized)
    store = d.DerivationStore(tmp_path/"derived.sqlite3")
    first = store.execute(None,spec); second = store.execute(None,spec); looked_up = store.get(d.derivation_id(spec),d.digest(spec))
    assert first == second == looked_up and first["status"] == "completed" and len(calls) == 1
    assert old.path.read_bytes() == before and old.get("industry","draft_master",spec["source_request_id"],spec["source_wire_sha256"])["response"] == response
    result = first["response"]
    assert result["provenance"]["new_model_calls"] == 0 and result["step_response"]["generation_trace"] == response["generation_trace"]
    assert result["step_response"]["publishable"] is False and result["provenance"]["quality_review_performed"] is False
    with pytest.raises(d.DerivationError): store.get(d.derivation_id(spec),"0"*64)
    changed = {**spec,"settlement_sha256":"0"*64}
    with pytest.raises(d.DerivationError): store.execute(None,changed)
    assert len(calls) == 1 and store.get(d.derivation_id(spec),d.digest(spec)) == first

def test_missing_derivation_get_does_not_initialize_database(tmp_path):
    store = d.DerivationStore(tmp_path/"missing.sqlite3")
    with pytest.raises(d.DerivationError) as error: store.get("wderive_"+"a"*48,"b"*64)
    assert error.value.status_code == 404 and not store.path.exists()
