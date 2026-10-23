"""Bounded private diagnostics for authenticated, isolated editorial steps.

No capture is active on legacy/public routes. Raw values never become model
metadata, artifacts or traces; only the internal step's top-level receipt owns
them. Limits discard complete values explicitly, never masquerade a prefix as
a complete provider reply or model candidate.
"""
from __future__ import annotations

import base64
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
import re

import httpx


MAX_PRIVATE_BUNDLE_BYTES = 64 * 1024
MAX_RAW_VALUE_BYTES = 24 * 1024
MAX_PROVIDER_PARSE_BYTES = 8 * 1024 * 1024
PRIVATE_GENERATION_CAPTURE: ContextVar["PrivateGenerationCapture | None"] = ContextVar(
    "marketing_private_generation_capture", default=None)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class PrivateGenerationCapture:
    def __init__(self, *, role_id, step, reservation_id, request_sha256=""):
        self.data = {"schema_version": "operator.private_generation_receipt.v1",
            "role_id": str(role_id), "step": step, "budget_reservation_id": reservation_id,
            "request_sha256": request_sha256, "provider_responses": []}
        self.secrets = []
        self.protect(os.getenv("HERMES_API_KEY", ""))

    def protect(self, *values):
        self.secrets.extend(value.encode("utf-8") for value in values if isinstance(value, str) and value)

    def _sensitive(self, raw):
        return any(value in raw for value in self.secrets)

    def _value(self, value):
        try:
            raw = encoded(value)
            result = {"type": type(value).__name__, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(), "retained": False,
                      "byte_source": "canonical_json"}
            if isinstance(value, str):
                result["characters"] = len(value)
            if len(raw) > MAX_RAW_VALUE_BYTES:
                result["omitted_reason"] = "capture_limit_exceeded"
            elif self._sensitive(raw):
                result["omitted_reason"] = "credential_echo"
            else:
                result.update(retained=True, value=deepcopy(value))
            return result
        except (ValueError, TypeError, OverflowError):
            return {"type": type(value).__name__, "retained": False,
                    "omitted_reason": "value_not_json"}

    def begin(self, event):
        # A staged step owns at most one physical request. Do not let a future
        # caller grow the private journal into a second request/retry facility.
        if self.data["provider_responses"]:
            raise ValueError("private_receipt_single_request_required")
        row = {"request_id": event["request_id"], "parse_stage": "request_started",
               "response_received": False}
        self.data["provider_responses"].append(row)
        return row

    @property
    def current(self):
        return self.data["provider_responses"][-1] if self.data["provider_responses"] else None

    def stage(self, value):
        if self.current is not None:
            self.current["parse_stage"] = value

    def failure(self, code, exception):
        if self.current is not None:
            self.current["error_code"] = code
            kind = type(exception).__name__
            self.current["exception_type"] = kind if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", kind) else "Exception"

    async def post(self, client, url, **kwargs):
        """Capture headers/partial-byte metadata even if reading then times out."""
        row = self.current
        digest, kept, chunks, observed, complete = hashlib.sha256(), bytearray(), [], 0, False
        try:
            async with client.stream("POST", url, **kwargs) as response:
                row.update(response_received=True, http_status=response.status_code, parse_stage="response_body")
                rid = (response.headers.get("x-oneapi-request-id") or response.headers.get("x-request-id")
                       or response.headers.get("request-id"))
                if rid and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", rid) and not self._sensitive(rid.encode()):
                    row["provider_request_id"] = rid
                media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                row["media_type"] = media_type if media_type in {"application/json", "text/plain", "text/html", "text/event-stream"} else "other"
                async for chunk in response.aiter_bytes():
                    digest.update(chunk)
                    observed += len(chunk)
                    if observed <= MAX_RAW_VALUE_BYTES:
                        kept.extend(chunk)
                    else:
                        kept.clear()
                    if observed > MAX_PROVIDER_PARSE_BYTES:
                        raise ValueError("provider_response_too_large")
                    chunks.append(chunk)
                complete = True
                # The adapter retains its existing HTTP/JSON semantics. Only
                # staged calls use this reader; legacy transports stay intact.
                headers = {key: value for key, value in response.headers.items()
                           if key.lower() not in {"content-encoding", "content-length"}}
                return httpx.Response(response.status_code, headers=headers,
                                      content=b"".join(chunks), request=response.request)
        finally:
            body = {"bytes": observed, "sha256": digest.hexdigest(), "complete": complete,
                    "byte_source": "decoded_http_body",
                    "retained": False, "hash_scope": "complete_body" if complete else "observed_prefix"}
            if not complete:
                body["omitted_reason"] = "response_incomplete"
            elif observed > MAX_RAW_VALUE_BYTES:
                body["omitted_reason"] = "capture_limit_exceeded"
            elif self._sensitive(bytes(kept)):
                body["omitted_reason"] = "credential_echo"
            else:
                body.update(retained=True, encoding="base64", value=base64.b64encode(kept).decode("ascii"))
            row["body"] = body

    def envelope(self, value):
        if self.current is None:
            return
        self.current["envelope_type"] = type(value).__name__
        if not isinstance(value, dict):
            return
        choices = value.get("choices")
        self.current["choices_type"] = type(choices).__name__
        self.current["choices_count"] = len(choices) if isinstance(choices, list) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            self.current["finish_reason"] = (reason if reason in {"stop", "length", "content_filter", "tool_calls", "function_call", None} else "other") if isinstance(reason, (str, type(None))) else "invalid_type"
            message = choices[0].get("message")
            self.current["message_type"] = type(message).__name__
            if isinstance(message, dict):
                self.current["content"] = self._value(message.get("content"))

    def writer(self, candidate):
        clean = {key: value for key, value in candidate.items() if not str(key).startswith("_")} if isinstance(candidate, dict) else candidate
        self.data["writer_candidate"] = {**self._value(clean),
            "source": "model" if isinstance(candidate, dict)
                and candidate.get("_model") not in {None, "", "fallback", "unavailable", "deterministic"}
                and not candidate.get("_error") else "fallback"}

    def analysis_verifier(self, candidate):
        """Keep the real structured verifier reply before binding decisions."""
        clean = {key: value for key, value in candidate.items() if not str(key).startswith("_")} if isinstance(candidate, dict) else candidate
        self.data["analysis_verifier_candidate"] = {**self._value(clean),
            "source": "model" if isinstance(candidate, dict)
                and candidate.get("_model") not in {None, "", "fallback", "unavailable", "deterministic"}
                and not candidate.get("_error") else "fallback"}

    def attach(self, response, review_receipt=None):
        private = {"private_generation_receipt": deepcopy(self.data)}
        if review_receipt is not None:
            private["private_review_receipt"] = deepcopy(review_receipt)
            old = private["private_review_receipt"]
            if old.get("candidate_retained") and self._sensitive(encoded(old.get("candidate"))):
                old.update(candidate=None, candidate_retained=False, capture_error="credential_echo")
        # Prefer the structured candidate; provider body/content are redundant
        # for valid JSON. Keep length/hash diagnostics for every omitted value.
        removable = []
        for row in private["private_generation_receipt"]["provider_responses"]:
            removable.extend([row.get("body"), row.get("content")])
        removable.append(private["private_generation_receipt"].get("writer_candidate"))
        removable.append(private["private_generation_receipt"].get("analysis_verifier_candidate"))
        for item in removable:
            if len(encoded(private)) <= MAX_PRIVATE_BUNDLE_BYTES:
                break
            if isinstance(item, dict) and item.get("retained"):
                item.pop("value", None)
                item.update(retained=False, omitted_reason="private_bundle_limit")
        if len(encoded(private)) > MAX_PRIVATE_BUNDLE_BYTES and "private_review_receipt" in private:
            old = private["private_review_receipt"]
            old.update(candidate=None, candidate_retained=False, capture_error="private_bundle_limit")
        return {**response, **private}
