"""Derive reference-only display roles from complete frozen source evidence.

This is a compose-only, explicitly versioned display boundary. Original claims,
evidence groups, source documents and paid receipts are never rewritten.
"""
from datetime import datetime, timezone
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit


REFERENCE_SELECTION_VERSION = "thesis_selection.v2"


def _moment(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value:
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def _source_key(value):
    try:
        parsed = urlsplit(str(value or ""))
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.port not in {None, 443}):
            return ""
        return urlunsplit(("https", parsed.hostname.lower(), parsed.path.rstrip("/") or "/", parsed.query, ""))
    except ValueError:
        return ""


def _scope_valid(document):
    scope = document.get("scope")
    if scope is None:
        return True
    if not isinstance(scope, dict):
        return False
    start, end, total = (scope.get(key) for key in ("start_offset", "end_offset", "document_text_chars"))
    text = document.get("source_text")
    return bool(scope.get("kind") == "section" and scope.get("partial_document") is True
        and scope.get("offset_unit") == "unicode_codepoints"
        and scope.get("text_normalization") == "html_get_text_newlines_v1"
        and scope.get("fully_read") is True and isinstance(text, str)
        and all(type(value) is int for value in (start, end, total))
        and 0 <= start < end <= total and end - start == len(text)
        and scope.get("passage_chars") == len(text)
        and scope.get("passage_sha256") == document.get("source_text_hash")
        and re.fullmatch(r"[0-9a-f]{64}", str(scope.get("document_text_sha256") or ""))
        and isinstance(scope.get("start_heading"), str) and scope["start_heading"].strip()
        and (scope.get("end_heading") is None or isinstance(scope.get("end_heading"), str)))


def _read_document(document, uri, now):
    if not isinstance(document, dict) or document.get("status") not in {"ready", "partial"}:
        return False
    text = document.get("source_text")
    retrieved = _moment(document.get("retrieved_at"))
    return bool(_source_key(uri) and _source_key(uri) == _source_key(document.get("url"))
        and isinstance(text, str) and 80 <= len(text) <= 24000
        and isinstance(document.get("coverage"), dict) and document["coverage"].get("text") == "read"
        and _scope_valid(document) and retrieved is not None and retrieved <= now
        and hashlib.sha256(text.encode("utf-8")).hexdigest() == document.get("source_text_hash"))


def reference_evidence_metadata(claims, records, *, as_of):
    """Mirror Main's whole-group classification; all used sources are checked."""
    def invalid():
        raise ValueError("surface_preparation_reference_evidence_invalid")

    now = _moment(as_of)
    if now is None:
        invalid()
    by_id = {}
    for value in records:
        row = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        if not isinstance(row, dict) or row.get("record_id") in by_id:
            invalid()
        by_id[row.get("record_id")] = row
    claims = [claim.model_dump(mode="json") if hasattr(claim, "model_dump") else claim for claim in claims]
    used = {identity for claim in claims for identity in claim.get("evidence_ids") or []}
    raw, curated, versions = {}, {}, {}
    for identity in used:
        row = by_id.get(identity) or {}
        structured = row.get("structured_data") or {}
        if not isinstance(structured, dict):
            invalid()
        document = structured.get("source_document") or {}
        start, end = _moment(row.get("valid_from")), _moment(row.get("valid_until"))
        if not (row.get("space") in {"industry", "company_public"}
                and row.get("status") == "approved" and "industry" in (row.get("authorized_roles") or [])
                and row.get("source_tier") in {"official", "primary"}
                and not (row.get("valid_from") and start is None)
                and not (row.get("valid_until") and end is None)
                and (start is None or start <= now) and (end is None or end > now)
                and _read_document(document, row.get("source_uri"), now)):
            invalid()
        uri = _source_key(row.get("source_uri"))
        if versions.setdefault(uri, document["source_text_hash"]) != document["source_text_hash"]:
            invalid()
        verification = structured.get("verification") or {}
        if (row.get("record_type") == "source_item" and row.get("space") == "industry"
                and structured.get("operator_input") is True and structured.get("input_kind") == "public_document"
                and isinstance(verification, dict) and verification.get("verified") is True
                and row.get("content") == document["source_text"]):
            raw[identity] = row
        elif row.get("record_type") == "source_fact" and str(row.get("content") or "").strip():
            curated[identity] = row
        else:
            invalid()
    bindings = [{"record_id": identity, "source_uri": row["source_uri"],
                 "source_text_sha256": row["structured_data"]["source_document"]["source_text_hash"],
                 "curated_record_ids": sorted(key for key, value in curated.items()
                     if _source_key(value.get("source_uri")) == _source_key(row.get("source_uri")))}
                for identity, row in sorted(raw.items())]
    references = []
    for index, claim in enumerate(claims, 1):
        evidence = set(claim.get("evidence_ids") or [])
        if (claim.get("kind") != "fact" or claim.get("verification_status") != "verified" or not evidence
                or not evidence <= raw.keys() | curated.keys()):
            invalid()
        if evidence <= raw.keys():
            references.append(f"claim-{index}")
        elif any(curated[value].get("content") != claim.get("text") for value in evidence & curated.keys()):
            invalid()
    return {"reference_only_claim_ids": references, "source_evidence_bindings": bindings}
