"""Typed suggestions from the independent planner, never factual authority.

These contracts are optional on fresh research only.  Orchestration owns their
current-source and mandate checks; the runtime still verifies canonical facts.
"""
from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


LONG_SURFACES = frozenset({"master", "wechat_mp", "toutiao"})
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
RecordId = Annotated[str, Field(min_length=1, max_length=160, strict=True)]


class SelectionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def public_source_key(value: str) -> tuple[str, str, str, str]:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment or parsed.port not in {None, 443}):
        raise ValueError("industry_surface_source_invalid")
    return parsed.scheme, parsed.hostname.lower(), parsed.path.rstrip("/"), parsed.query


class PlannerAtomicFact(SelectionContract):
    record_id: RecordId
    version: int = Field(ge=1, strict=True)
    text: str = Field(min_length=1, max_length=4000, strict=True)
    body_sha256: SHA256
    excerpt_sha256: SHA256
    source_text_sha256: SHA256
    source_uri: str = Field(min_length=1, max_length=2000, strict=True)

    @field_validator("record_id", "text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("industry_surface_fact_blank")
        return value

    @field_validator("source_uri")
    @classmethod
    def valid_source(cls, value):
        public_source_key(value)
        return value

    @model_validator(mode="after")
    def exact_body(self):
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.body_sha256:
            raise ValueError("industry_surface_fact_body_changed")
        return self


class ShortSurfaceAnchor(SelectionContract):
    anchor_record_id: RecordId
    reason: str = Field(min_length=1, max_length=1000, strict=True)

    @field_validator("anchor_record_id", "reason")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("industry_surface_anchor_blank")
        return value


class IndustrySurfaceSelection(SelectionContract):
    schema_version: Literal["operator.industry_surface_selection.v1"]
    short_surfaces: dict[str, ShortSurfaceAnchor] = Field(max_length=20)
    reported_case_record_ids: list[RecordId] = Field(max_length=3)
    case_gap: str = Field(max_length=1000, strict=True)

    @model_validator(mode="after")
    def case_or_gap(self):
        if (len(set(self.reported_case_record_ids)) != len(self.reported_case_record_ids)
                or any(not identity.strip() for identity in self.reported_case_record_ids)
                or bool(self.reported_case_record_ids) == bool(self.case_gap.strip())):
            raise ValueError("industry_surface_case_invalid")
        return self


class PreparationEvidenceSelection(SelectionContract):
    schema_version: Literal["operator.preparation_evidence_selection.v1"]
    planner_selection_sha256: SHA256
    required_evidence_ids: list[RecordId] = Field(min_length=1, max_length=50)

    @field_validator("required_evidence_ids")
    @classmethod
    def unique_ids(cls, values):
        if len(set(values)) != len(values) or any(not identity.strip() for identity in values):
            raise ValueError("preparation_evidence_selection_invalid")
        return values


def validate_surface_selection(selection, *, catalog, channels):
    """Bind a model suggestion to the caller's exact displayed atomic catalog."""
    if not isinstance(selection, IndustrySurfaceSelection):
        selection = IndustrySurfaceSelection.model_validate(selection)
    expected = set(channels) - LONG_SURFACES
    if set(selection.short_surfaces) != expected:
        raise ValueError("industry_surface_channels_changed")
    identities = {fact.record_id for fact in catalog}
    selected = {anchor.anchor_record_id for anchor in selection.short_surfaces.values()}
    selected.update(selection.reported_case_record_ids)
    if not selected <= identities:
        raise ValueError("industry_surface_fact_unbound")
    return selection


def validate_preparation_plan_metadata(body, *, claim_texts, dependencies, reporting_claim_ids=()):
    """Validate only the signed new preparation branch; legacy plans are separate."""
    authority = body.get("preparation_authorization")
    intent = body.get("editorial_intent")
    authority_fields = {"schema_version", "mode", "publication_authorized", "run_id", "mandate_version",
        "proposal_authorization_id", "proposal_authorization_version", "planner_request_sha256",
        "planner_selection_sha256", "research_digest_sha256"}
    intent_fields = {"schema_version", "author", "short_surface_reasons", "reported_case_claim_ids", "case_gap"}
    is_hash = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if (not isinstance(authority, dict) or set(authority) != authority_fields
            or authority.get("schema_version") != "operator.surface_preparation_authority.v1"
            or authority.get("mode") != "mandate_preparation_only" or authority.get("publication_authorized") is not False
            or any(type(authority.get(key)) is not int or authority[key] < 1
                for key in ("mandate_version", "proposal_authorization_version"))
            or any(not isinstance(authority.get(key), str) or not authority[key].strip()
                for key in ("run_id", "proposal_authorization_id"))
            or authority.get("proposal_authorization_id") != (body.get("scope") or {}).get("proposal_id")
            or any(not is_hash(authority.get(key)) for key in
                ("planner_request_sha256", "planner_selection_sha256", "research_digest_sha256"))
            or not is_hash(body.get("preparation_selection_sha256"))
            or not isinstance(intent, dict) or set(intent) != intent_fields
            or intent.get("schema_version") != "operator.preparation_editorial_intent.v1"
            or intent.get("author") != "independent workflow editorial planner"):
        raise ValueError("surface_preparation_authority_invalid")
    ids = list(claim_texts)
    expected_roles = {"anchor": ids, **{role: [identity for identity, text in claim_texts.items() if pattern.search(text)]
        for role, pattern in dependencies.items()}}
    surfaces = body.get("surfaces") or {}
    reasons = intent.get("short_surface_reasons")
    cases, gap = intent.get("reported_case_claim_ids"), intent.get("case_gap")
    if (body.get("fact_roles") != expected_roles
            or not isinstance(reasons, dict) or set(reasons) != set(surfaces) - LONG_SURFACES
            or any(not isinstance(value, str) or not value.strip() or len(value) > 1000 for value in reasons.values())
            or not isinstance(cases, list) or len(cases) > 3 or any(not isinstance(value, str) or value not in claim_texts for value in cases)
            or len(cases) != len(set(cases)) or not isinstance(gap, str) or len(gap) > 1000
            or bool(cases) == bool(gap.strip())
            or any(not set(selected) - set(reporting_claim_ids) for surface, selected in surfaces.items() if surface not in LONG_SURFACES)):
        raise ValueError("surface_preparation_plan_invalid")
    return body
