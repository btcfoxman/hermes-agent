"""Private verifier decisions remain original, bounded and outside artifacts."""
from copy import deepcopy
import hashlib
import json

import pytest

from ai_marketing_api.operator_private_receipts import (
    MAX_PRIVATE_BUNDLE_BYTES, MAX_RAW_VALUE_BYTES, PrivateGenerationCapture,
)


@pytest.mark.parametrize("decisions", [
    [{"candidate_id": "fictional-a", "verdict": "supported", "reason": "A bounded hypothetical."}],
    [{"candidate_id": "fictional-a", "verdict": "unsupported", "reason": "The result is not established."}],
    [{"candidate_id": "fictional-a", "verdict": "uncertain", "reason": "The premise is ambiguous."}],
    [{"candidate_id": "fictional-a"}, {"candidate_id": "fictional-a", "verdict": "supported"}],
])
def test_verifier_candidate_is_retained_before_protocol_binding(decisions):
    capture = PrivateGenerationCapture(role_id="industry", step="verify_claims", reservation_id="synthetic-reservation")
    candidate = {"analysis_reviews": decisions, "_model": "offline-verifier", "_usage": {"total_tokens": 17}}
    original = deepcopy(candidate)
    capture.analysis_verifier(candidate)
    candidate["analysis_reviews"].clear()
    public = {"artifact": {"status": "quality_insufficient"}, "generation_trace": {"model_calls": 1}}
    result = capture.attach(public)
    retained = result["private_generation_receipt"]["analysis_verifier_candidate"]
    assert retained["source"] == "model" and retained["retained"] is True
    assert retained["value"] == {"analysis_reviews": original["analysis_reviews"]}
    assert "writer_candidate" not in result["private_generation_receipt"]
    assert public == {"artifact": {"status": "quality_insufficient"}, "generation_trace": {"model_calls": 1}}
    assert "analysis_reviews" not in json.dumps(public)
    retained["value"]["analysis_reviews"].clear()
    assert capture.data["analysis_verifier_candidate"]["value"]["analysis_reviews"] == original["analysis_reviews"]


def test_verifier_echoed_secret_is_not_retained_in_private_output():
    capture = PrivateGenerationCapture(role_id="industry", step="verify_claims", reservation_id="synthetic-reservation")
    secret = "fictional-credential-for-redaction"
    capture.protect(secret)
    capture.analysis_verifier({"analysis_reviews": [{"reason": secret}], "_model": "offline-verifier"})
    result = capture.attach({"artifact": None})
    retained = result["private_generation_receipt"]["analysis_verifier_candidate"]
    assert retained["retained"] is False and retained["omitted_reason"] == "credential_echo"
    assert "value" not in retained and secret not in json.dumps(result)


def test_oversized_verifier_reply_has_complete_hash_without_a_false_partial_value():
    capture = PrivateGenerationCapture(role_id="industry", step="verify_claims", reservation_id="synthetic-reservation")
    candidate = {"analysis_reviews": [{"reason": "x" * (MAX_RAW_VALUE_BYTES + 10)}]}
    raw = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    capture.analysis_verifier({**candidate, "_model": "offline-verifier"})
    result = capture.attach({"artifact": None})
    retained = result["private_generation_receipt"]["analysis_verifier_candidate"]
    assert retained["retained"] is False and retained["omitted_reason"] == "capture_limit_exceeded"
    assert retained["sha256"] == hashlib.sha256(raw).hexdigest() and retained["bytes"] == len(raw)
    assert "value" not in retained
    bundle = json.dumps({"private_generation_receipt": result["private_generation_receipt"]},
                        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert len(bundle) <= MAX_PRIVATE_BUNDLE_BYTES
