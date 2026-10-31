"""One explicit AI-editor sentence removal from a retained writer reply; no model call."""
from __future__ import annotations
import base64, hashlib, hmac, json, os, re, secrets, sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from pydantic import Field
from ai_marketing_api.operator_runtime import StrictModel
from ai_marketing_api.operator_receipts import EditorialReceiptStore, _state_path
from ai_marketing_api.operator_content import (OperatorComposeRequest, build_content_fallback,
    content_request_payload, normalize_content_output)
from ai_marketing_api.operator_editorial_trim import prepare_editorial_trim
from ai_marketing_api.operator_style import MASTER_ONLY, editorial_scope
from ai_marketing_api.operator_analysis import DEFER_EDITORIAL_ANALYSIS

POLICY = "billing_boundary_editorial_trim.v1"
REQUEST_SCHEMA = "operator.writer_derivation_request.v1"
RECEIPT_SCHEMA = "operator.writer_derivation_receipt.v1"
RESULT_SCHEMA = "operator.writer_derivation_result.v1"
BINDINGS = ("policy_version","parent_request_id","source_request_id","source_wire_sha256","source_response_sha256",
    "provider_body_sha256","writer_candidate_sha256","settlement_sha256","retry_binding_sha256")
MAX_BYTES = 2 * 1024 * 1024

class DerivationError(ValueError):
    def __init__(self,code,status_code=409): super().__init__(code); self.code,self.status_code = code,status_code
def require(ok,code):
    if not ok: raise DerivationError(code)
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def derivation_id(value): return "wderive_"+digest({key:value[key] for key in BINDINGS})[:48]
def strict_json(raw):
    def pairs(items):
        result = {}
        for key,value in items: require(key not in result,"derivation_duplicate_json_key"); result[key] = value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: require(False,"derivation_nonfinite_json"))

class DerivationRequest(StrictModel):
    schema_version: Literal[REQUEST_SCHEMA]
    policy_version: Literal[POLICY]
    parent_request_id: str = Field(pattern=r"^text_[a-f0-9]{40}$")
    source_request_id: str = Field(pattern=r"^text_[a-f0-9]{40}$")
    source_wire_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_response_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_body_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    writer_candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    settlement_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    retry_binding_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    logical_payload: dict[str,Any]

def source_candidate(spec,response):
    require(spec.get("schema_version") == REQUEST_SCHEMA and spec.get("policy_version") == POLICY,"derivation_policy_scope_invalid")
    payload,identity = spec["logical_payload"],spec["source_request_id"]
    require(len(canonical(spec)) <= 512 * 1024 and spec["parent_request_id"] != identity
        and digest({**payload,"budget_reservation_id":identity}) == spec["source_wire_sha256"]
        and payload.get("surface","master") == "master" and payload.get("candidate") is None and payload.get("candidate_signature") is None
        and (payload.get("compose") or {}).get("genre") == "industry_analysis"
        and payload["compose"].get("long_surface_selection_version") == "thesis_selection.v2","derivation_original_scope_invalid")
    require(digest(response) == spec["source_response_sha256"] and response.get("schema_version") == "operator.editorial_step.v1"
        and response.get("step") == "draft_master" and response.get("status") == "unavailable"
        and response.get("error_code") == "step_artifact_invalid" and response.get("artifact") is None
        and response.get("publishable") is False and response.get("requires_human_review") is True
        and not any(response.get(k) is not None for k in ("artifact_signature","quality_scorecard","editorial_assessment","private_review_receipt")),"derivation_original_response_invalid")
    trace,private = response.get("generation_trace") or {},response.get("private_generation_receipt") or {}
    events,attempts = trace.get("provider_requests"),trace.get("attempts")
    require(trace.get("origin") == "model" and trace.get("model_calls") == 1 and trace.get("repair_calls") == 0
        and trace.get("stop_reason") == "step_artifact_invalid" and trace.get("editorial_review") == "not_requested"
        and isinstance(events,list) and len(events) == 1 and isinstance(attempts,list) and len(attempts) == 1,"derivation_original_trace_invalid")
    event,attempt = events[0],attempts[0]; usage = event.get("usage") or {}
    require(event.get("status") == "returned" and event.get("http_status") == 200 and event.get("stage") == "compose"
        and event.get("model") == "gpt-5.6-terra" and event.get("reservation_id") == identity and not event.get("error_code")
        and re.fullmatch(r"[a-f0-9]{32}",str(event.get("request_id")))
        and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] >= 0
        and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > 0
        and attempt.get("stage") == "compose" and attempt.get("outcome") == "returned" and attempt.get("upstream_attempted") is True
        and attempt.get("model") == event["model"] and not attempt.get("error_code") and attempt.get("provider_requests") == events
        and attempt.get("input_tokens") == usage["prompt_tokens"] and attempt.get("output_tokens") == usage["completion_tokens"],"derivation_original_model_invalid")
    require(private.get("schema_version") == "operator.private_generation_receipt.v1" and private.get("role_id") == "industry"
        and private.get("step") == "draft_master" and private.get("budget_reservation_id") == identity
        and private.get("request_sha256") == spec["source_wire_sha256"] and isinstance(private.get("provider_responses"),list)
        and len(private["provider_responses"]) == 1,"derivation_private_receipt_invalid")
    received = private["provider_responses"][0]; body = received.get("body") or {}; writer = private.get("writer_candidate") or {}
    require(received.get("request_id") == event["request_id"] and received.get("provider_request_id") == event.get("provider_request_id")
        and received.get("http_status") == 200 and received.get("response_received") is True and received.get("parse_stage") == "complete"
        and not received.get("error_code") and body.get("complete") is True and body.get("retained") is True
        and body.get("encoding") == "base64" and body.get("hash_scope") == "complete_body"
        and body.get("byte_source") == "decoded_http_body" and type(body.get("bytes")) is int and 0 < body["bytes"] <= 24576,"derivation_complete_body_required")
    raw = base64.b64decode(body.get("value",""),validate=True)
    require(len(raw) == body["bytes"] and hashlib.sha256(raw).hexdigest() == body.get("sha256") == spec["provider_body_sha256"],"derivation_body_hash_changed")
    envelope = strict_json(raw); choices = envelope.get("choices") or []
    require(envelope.get("usage") == usage and envelope.get("model") == event["model"] and isinstance(choices,list)
        and len(choices) == 1 and choices[0].get("finish_reason") == "stop","derivation_complete_choice_required")
    message = choices[0].get("message") or {}; content = message.get("content")
    require(type(content) is str and not message.get("refusal") and not message.get("tool_calls") and not message.get("function_call"),"derivation_refusal_or_tools_denied")
    candidate = strict_json(content)
    require(type(candidate) is dict and writer.get("type") == "dict" and writer.get("source") == "model"
        and writer.get("retained") is True and writer.get("byte_source") == "canonical_json" and type(writer.get("value")) is dict
        and {k:v for k,v in candidate.items() if not str(k).startswith("_")} == writer["value"]
        and digest(writer["value"]) == writer.get("sha256") == spec["writer_candidate_sha256"]
        and len(canonical(writer["value"])) == writer.get("bytes"),"derivation_writer_hash_changed")
    require(isinstance(candidate.get("master_title"),str) and candidate["master_title"].strip() and candidate.get("platform_variants") == []
        and isinstance(candidate.get("blocks"),list) and candidate["blocks"]
        and (payload["compose"].get("surface_fact_plan") or {}).get("preparation_authorization"),"derivation_policy_scope_invalid")
    return {**deepcopy(writer["value"]),"_model":event["model"]}

def normalize_saved(registry,spec,response):
    candidate = source_candidate(spec,response); request = OperatorComposeRequest.model_validate(spec["logical_payload"]["compose"])
    require(request.editorial_review_mode == "standard","derivation_review_mode_denied")
    contexts = registry.authorize("industry",request.authorized_context,request.as_of); registry.authorize_preferences("industry",request.approved_preferences)
    brief = request.public_editorial_brief.model_copy(update={"channels":[]}) if request.public_editorial_brief else None
    revision = request.owner_revision.model_copy(update={"platform_variants":[]}) if request.owner_revision else None
    scoped = request.model_copy(update={"channels":[],"public_editorial_brief":brief,"owner_revision":revision})
    master,defer = MASTER_ONLY.set(True),DEFER_EDITORIAL_ANALYSIS.set(True)
    try:
        with editorial_scope(request.genre):
            content_request_payload("industry",request,contexts)
            fallback = build_content_fallback(registry,"industry",scoped,contexts)
            require(not fallback.get("_fallback_errors"),"derivation_fallback_invalid"); content_request_payload("industry",scoped,contexts)
            def strings(value):
                if isinstance(value,str): return [value]
                if isinstance(value,dict): return [text for item in value.values() for text in strings(item)]
                if isinstance(value,list): return [text for item in value for text in strings(item)]
                return []
            source_texts = [text for context in contexts for text in [context.content,*strings(context.structured_data)]]
            templates = fallback.get("blocks") or []
            canonical_blocks = [block for block in templates if block.get("origin") == "approved_claim"
                or block.get("required") is True or block.get("source_exact") is True]
            source_texts.extend(block["text"] for block in templates if isinstance(block.get("text"),str))
            original = {key:value for key,value in candidate.items() if not key.startswith("_")}
            edited,edits = prepare_editorial_trim(original,canonical_blocks=canonical_blocks,source_texts=source_texts)
            require(len(edits) == 1 and digest(original) == spec["writer_candidate_sha256"],"derivation_single_editorial_edit_required")
            raw_edit = edits[0]
            require(raw_edit.get("policy_version") == POLICY and raw_edit.get("source_candidate_sha256") == digest(original)
                and raw_edit.get("edited_candidate_sha256") == digest(edited),"derivation_editorial_edit_hash_invalid")
            edit = {"surface":"master","block_index":raw_edit["block_index"],"field":"text","offset_unit":"unicode_codepoints",
                "offset_start":raw_edit["start"],"offset_end":raw_edit["end"],"old_text_sha":raw_edit["old_text_sha256"],
                "new_text_sha":raw_edit["new_text_sha256"],"removed_sentence_sha":raw_edit["removed_text_sha256"],
                "reason":"remove_irrelevant_nonassertive_cost_boundary"}
            output = normalize_content_output(registry,"industry",scoped,contexts,fallback,{**edited,"_model":candidate["_model"]})
            require(not output.critic.errors and output.status == "content_ready" and output.master_content and not output.platform_variants,"derivation_normalization_blocked")
            artifact = output.model_dump(mode="json"); artifact["generation_trace"] = deepcopy(response["generation_trace"])
            return artifact,{"actor_kind":"ai_editor","editorial_intervention":True,
                "source_writer_candidate_sha256":digest(original),"edited_candidate_sha256":digest(edited),"edits":[edit]}
    finally: DEFER_EDITORIAL_ANALYSIS.reset(defer); MASTER_ONLY.reset(master)

def signature(value,domain=""):
    key = os.getenv("HERMES_API_KEY","").strip(); require(bool(key),"derivation_signing_key_missing")
    return hmac.new(key.encode(),domain.encode()+canonical(value),hashlib.sha256).hexdigest()

def derive(registry,spec,original_store=None):
    old = (original_store or EditorialReceiptStore()).get("industry","draft_master",spec["source_request_id"],spec["source_wire_sha256"])
    require(old.get("status") == "completed", "derivation_original_not_completed")
    response = old["response"]
    try:
        artifact,editorial = normalize_saved(registry,spec,response)
        step = {"schema_version":"operator.editorial_step.v1","step":"draft_master","status":"artifact","artifact":artifact,
            "artifact_signature":signature(artifact),"publishable":False,"requires_human_review":True,
            "generation_trace":deepcopy(response["generation_trace"]),"private_generation_receipt":deepcopy(response["private_generation_receipt"])}
    except (ValueError,TypeError,KeyError,AttributeError) as error:
        code = getattr(error,"code",None); code = code if isinstance(code,str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}",code) else "derivation_normalization_rejected"
        step = {"schema_version":"operator.editorial_step.v1","step":"draft_master","status":"unavailable","artifact":None,
            "error_code":code,"publishable":False,"requires_human_review":True}
        editorial = {"actor_kind":"ai_editor","editorial_intervention":False,"source_writer_candidate_sha256":spec["writer_candidate_sha256"],
            "edited_candidate_sha256":None,"edits":[]}
    provenance = {**{key:spec[key] for key in BINDINGS},"logical_payload_sha256":digest(spec["logical_payload"]),
        "source_trace_sha256":digest(response.get("generation_trace")),"derived_artifact_sha256":digest(step.get("artifact")),
        "provider_posts":0,"new_model_calls":0,"quality_review_performed":False,"financial_claim":False,"owner_decision":False,**editorial}
    result = {"schema_version":RESULT_SCHEMA,"derivation_id":derivation_id(spec),"request_sha256":digest(spec),
        "provenance":provenance,"step_response":step}
    return {**result,"signature":signature(result,"writer_artifact_derivation.v1\n")}

class DerivationStore:
    """Separate file in the existing receipt volume; original table is never opened for writing."""
    def __init__(self,path=None): self.path = path or _state_path().with_name("editorial_artifact_derivations.sqlite3")
    def connection(self):
        require(not any(path.is_symlink() for path in (self.path,*self.path.parents)),"derivation_store_symlink_denied")
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        descriptor = os.open(self.path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try: os.fchmod(descriptor,0o600)
        finally: os.close(descriptor)
        db = sqlite3.connect(self.path,timeout=10,isolation_level=None); db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS artifact_derivations(id TEXT PRIMARY KEY,policy TEXT NOT NULL,source_child TEXT NOT NULL,request_sha256 TEXT NOT NULL,status TEXT NOT NULL,claim_token TEXT NOT NULL,response_json TEXT,response_sha256 TEXT,UNIQUE(policy,source_child))")
        return db
    def envelope(self,row,identity,request_sha):
        require(row is not None,"derivation_not_found"); require(row["id"] == identity and row["request_sha256"] == request_sha,"derivation_request_mismatch")
        value = {"schema_version":RECEIPT_SCHEMA,"derivation_id":identity,"request_sha256":request_sha,"status":row["status"]}
        if row["status"] == "completed":
            raw = row["response_json"].encode(); require(len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == row["response_sha256"],"derivation_saved_response_invalid")
            value["response"] = strict_json(raw)
        return value
    def get(self,identity,request_sha):
        if not self.path.exists(): raise DerivationError("derivation_not_found",404)
        require(not any(path.is_symlink() for path in (self.path,*self.path.parents)),"derivation_store_symlink_denied")
        db = sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro",uri=True,timeout=10)
        try:
            db.row_factory = sqlite3.Row; row = db.execute("SELECT * FROM artifact_derivations WHERE id=?",(identity,)).fetchone()
        finally: db.close()
        if row is None: raise DerivationError("derivation_not_found",404)
        return self.envelope(row,identity,request_sha)
    def execute(self,registry,spec):
        require(spec.get("schema_version") == REQUEST_SCHEMA and spec.get("policy_version") == POLICY,"derivation_policy_scope_invalid")
        identity,request_sha,token = derivation_id(spec),digest(spec),secrets.token_hex(32)
        db = self.connection()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM artifact_derivations WHERE policy=? AND source_child=?",(POLICY,spec["source_request_id"])).fetchone()
            if row is None:
                db.execute("INSERT INTO artifact_derivations(id,policy,source_child,request_sha256,status,claim_token) VALUES(?,?,?,?,'running',?)",(identity,POLICY,spec["source_request_id"],request_sha,token))
                row = db.execute("SELECT * FROM artifact_derivations WHERE id=?",(identity,)).fetchone()
            saved = self.envelope(row,identity,request_sha); db.commit()  # Same child/policy with changed hashes is a conflict.
            if not secrets.compare_digest(row["claim_token"],token): return saved
            result = derive(registry,spec); raw = canonical(result); require(len(raw) <= MAX_BYTES,"derivation_response_too_large")
            changed = db.execute("UPDATE artifact_derivations SET status='completed',response_json=?,response_sha256=? WHERE id=? AND request_sha256=? AND claim_token=? AND status='running'",
                (raw.decode(),hashlib.sha256(raw).hexdigest(),identity,request_sha,token)).rowcount
            require(changed == 1,"derivation_claim_changed")
            return self.get(identity,request_sha)
        finally: db.close()
