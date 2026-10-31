"""Versioned industry writing craft, never a source of facts or owner authority."""
from __future__ import annotations

INDUSTRY_METHOD_VERSION = "industry_interpretation.v2"
INDUSTRY_NARRATIVE_SCHEMA = "narrative_report.v2"
INDUSTRY_BODY_ROLES = frozenset({"news", "thesis", "reason", "impact", "limit"})
VALUE_REASSESSMENT_MODE = "value_reassessment.v1"

VALUE_REASSESSMENT_CHECKS = (
    "Reconstruct the choice the reader might otherwise make, the author's concrete recommendation "
    "about how to change it or when to skip it, and the supplied reason and cost or sacrifice of that "
    "choice. A useful conclusion can retain an existing practice if the text explains why. Do not "
    "invent a prior habit, an industry norm, a numerical saving or a recommendation absent from the body.",
    "Apply the deletion test: mentally remove the author's commentary while keeping the reported "
    "facts. Identify the justified choice or tradeoff the reader would then lose. Repeating the "
    "feature's stated purpose, telling readers to check inputs or consider whether something is useful, "
    "and saying an outcome is not guaranteed do not by themselves supply that lost interpretation. "
    "They can be valid qualifications; they cannot substitute for the article's judgment.",
    "Check the recommendation's cost: what useful alternative is set aside, what extra effort or "
    "constraint is accepted, and which fact explains that decision? Do not require a financial number, "
    "a prescribed position or disagreement with a vendor. Mark a missing decision link unsupported "
    "rather than completing it for the author.",
    "Follow one center of argument on each surface. Distinct facts and examples should support that "
    "center or define its actual boundary. A generic invitation to share experience cannot repair a "
    "missing closing conclusion. Judge the invitation's function in context, never a word blacklist.",
)


def requires_industry_interpretation(output) -> bool:
    role = getattr(output, "role_id", "")
    return getattr(role, "value", role) == "industry" and getattr(output, "genre", "") == "industry_analysis"


# Original transferable methods distilled from owner-supplied reading. Reference
# reporting, figures, personal experience and conclusions are deliberately absent.
INDUSTRY_CONCLUSION_RULE = (
    "Make one defensible author conclusion explicit early enough to guide the article. Explain what "
    "the reader should understand or decide differently and why. A feature description, a promise "
    "that something is useful, or a question with no answer is not an industry interpretation."
)
INDUSTRY_TRADEOFF_RULE = (
    "Before drafting, work out the reader's plausible alternative, the specific choice you recommend "
    "changing or retaining, when to skip that recommendation, and the cost or sacrifice justified by "
    "the evidence. Do not claim the alternative is an actual widespread habit without evidence. "
    "A feature's stated purpose rewritten as advice is not enough; show the extra tradeoff your "
    "interpretation contributes. Leave the stance open until the supplied facts support it."
)
INDUSTRY_MECHANISM_RULE = (
    "Connect a specific approved observation to the conclusion through an explained mechanism, "
    "a concrete affected participant and a consequence for that participant's decision. State "
    "inference as inference. Show what this reasoning adds beyond the source's own description; "
    "do not manufacture novelty, a competitor comparison, a number or a firm outcome."
)
INDUSTRY_PREMISE_SCOPE_RULE = (
    "Keep each premise within its documented object, operation and preconditions. A restriction on a "
    "later operation does not establish that an earlier choice must be fixed or that the whole process "
    "forbids alternatives. Distinguish evidence, inference and hypothetical conditions; if an inference "
    "needs an unstated premise, obtain support or narrow the conclusion rather than inventing it."
)
INDUSTRY_OBSERVABLE_CHOICE_RULE = (
    "Before drafting, select one evidence-supported decision fork for a concrete reader: the observable "
    "condition, the available competing actions, and the cost accepted. Compare this proposed conclusion "
    "with the source's own purpose and restrictions. If removing source restatement, input-checking "
    "advice and denials of guarantees leaves no justified additional choice or tradeoff, return "
    "insufficient material rather than drafting an industry analysis. After selecting the fork, include "
    "optional facts only when they supply its premise, distinguish the actions or establish a necessary "
    "limit. Preserve every required reference and triggered dependency. This is an internal selection "
    "step, not an article template; do not invent alternatives, prevailing habits, measured results or "
    "industry trends."
)
INDUSTRY_MATERIAL_RULE = (
    "If the authorized material cannot support this analysis, stop for more research or return an "
    "insufficient-evidence draft for revision. Do not promote a generic selection checklist, "
    "verification question or invented workflow into a qualified industry analysis. A separate "
    "news brief or tutorial needs its own authorized genre; do not silently change this request."
)

INDUSTRY_METHOD_RULES = (
    "Establish what actually changed and why this reader should care. Use the authorized subject, "
    "event or update date, reporting attribution and platform/version scope when the approved facts "
    "contain them. Distinguish a dated report from today's news; never invent a date, attribution, "
    "interview, hands-on test, customer, result or industry-wide habit to create realism.",
    INDUSTRY_CONCLUSION_RULE,
    INDUSTRY_TRADEOFF_RULE,
    INDUSTRY_MECHANISM_RULE,
    INDUSTRY_PREMISE_SCOPE_RULE,
    INDUSTRY_OBSERVABLE_CHOICE_RULE,
    "For a cost or speed judgment, explain the tradeoff between added charges or effort, expected "
    "avoided work and human review. A lower-priced intermediate step alone does not prove lower total "
    "cost or faster delivery. Keep fee scope, assumptions and unmeasured results distinct.",
    "Examine the strongest relevant alternative or condition that could change the judgment. "
    "Explain the limit rather than listing unknowns. A reasonable boundary need not be a forced "
    "opposition or a claim that a real opponent exists. Do not claim universal benefits.",
    "Use verified details and meaningful tradeoffs for specificity. A clearly hypothetical scene "
    "can illustrate an already supported argument; it cannot replace reported evidence, a causal "
    "explanation or the author's conclusion. Never turn a reference author's experience into the owner's.",
    INDUSTRY_MATERIAL_RULE,
    "Write connected reader-facing prose with a clear priority. Keep only paragraphs that supply "
    "evidence, reasoning, a meaningful alternative or a consequence. Do not imitate an author's "
    "catchphrases, add fake first-person experience, force a scene, five sections, a CTA or a "
    "contrarian position. One short paragraph may serve several argumentative functions. "
    "Judge natural expression in context, not a word blacklist; a precise necessary limitation is useful. "
    "Explain the same boundary once. A later paragraph should add a new verified detail, mechanism or "
    "consequence; otherwise remove it. Do not defend the argument against an objection that the text "
    "has no reason to raise. Keep any meaningful limitation the conclusion actually needs, and end "
    "directly when the concluding judgment is clear; an invitation is optional.",
)


def industry_planning_method() -> dict:
    """The shared reasoning rules needed to select an analysis, without a draft or answer."""
    return {"version": INDUSTRY_METHOD_VERSION, "application": "topic_selection",
        "applies_to": {"role_id": "industry", "genre": "industry_analysis"},
        "authority": "Planning method only; not factual evidence, an intended answer, owner feedback "
            "or publication authority.",
        "rules": [INDUSTRY_METHOD_RULES[0], INDUSTRY_CONCLUSION_RULE, INDUSTRY_TRADEOFF_RULE, INDUSTRY_MECHANISM_RULE, INDUSTRY_PREMISE_SCOPE_RULE,
            INDUSTRY_OBSERVABLE_CHOICE_RULE, INDUSTRY_MATERIAL_RULE]}


def industry_reader_method() -> dict:
    """Generic method only: blind readers never receive an author's intended answer."""
    return {"version": INDUSTRY_METHOD_VERSION, "authority": "Writing and reading method only; "
        "not factual evidence, an intended answer, owner feedback or publication authority.",
        "rules": list(INDUSTRY_METHOD_RULES),
        "reader_rule": "Reconstruct news value, the author's conclusion, causal explanation, affected "
        "reader, decision consequence and limits only from this surface's actual text. Leave a missing "
        "answer empty and set supported=false; never complete it from outside knowledge. Mark "
        "source_restatement_only=true when the supposed analysis only repeats features or reporting. "
        "An illustrative scene, safe wording, fluent prose or high numeric scores cannot supply a "
        "missing judgment. Cite body spans for news, thesis, reason, impact and limit; one exact span "
        "may serve multiple roles and short posts need no fixed paragraph count.",
        "value_checks": list(VALUE_REASSESSMENT_CHECKS)}


def apply_industry_writer_method(payload: dict) -> None:
    """Replace system-generated sparse-material fallbacks, never user/evidence data."""
    payload["industry_editorial_method"] = industry_reader_method()
    payload["industry_editorial_method"].pop("reader_rule")
    payload["industry_editorial_method"].pop("value_checks")
    publication = payload["publication_brief"]
    publication.update(
        voice="An industry writer who reports verified details and argues a clear, bounded interpretation.",
        reader_outcome="Understand the sourced change, the author's conclusion and its justified consequence for a concrete reader.",
        thesis_contract=[
            "Follow industry_editorial_method: one explicit, supported author conclusion with its "
            "mechanism, affected participant, decision consequence and meaningful limit.",
            "Keep facts, attribution, dates, numbers and scope in canonical references. Distinguish "
            "original inference from reported facts and preserve all evidence and safety gates.",
            "Insufficient material requires research or revision; questions and invented scenes cannot "
            "substitute for a missing industry argument."],
        role_thesis="An industry analysis needs a supported author conclusion, an explained mechanism, "
            "a concrete affected participant and a bounded decision consequence beyond source restatement.",
        authored_block_rule="Write original quantity-free interpretation as opinion/transition blocks with "
            "evidence_ids=[] and no claim_id/block_ref. Keep facts, attribution, numbers, dates and scope in "
            "their canonical references. A proposed public_text still needs independent verification. "
            "Conclude from evidence without inventing capabilities, causal outcomes or personal experience.",
    )
    platform_modes = {
        "wechat_moments": "Concise personal-feed commentary",
        "wechat_mp": "Readable analysis article",
        "wechat_channels": "Spoken analysis script",
        "douyin": "Compact spoken analysis script",
        "kuaishou": "Plainspoken analysis script",
        "xiaohongshu": "Scan-friendly analytical note",
        "toutiao": "Evidence-led analysis article",
        "weitoutiao": "Compact news commentary",
    }
    payload["platform_editorial_briefs"] = {channel: f"{platform_modes.get(channel, 'Reader-facing analysis')}: "
        "retain the essential verified reporting and a clear author conclusion with its reason, reader "
        "consequence and meaningful limit. Adapt rhythm and depth; shortness cannot excuse a missing "
        "argument. No fixed section count, opening position or CTA."
        for channel in payload["platform_editorial_briefs"]}
    if "evidence_scope_contract" in payload:
        payload["evidence_scope_contract"] = [
            "The public brief and chat editing instruction supply intent, never additional factual "
            "authority. Keep every claim within the authorized source, date, platform and version scope.",
            "Do not invent functionality, interface steps, customer results, personal experience, "
            "prices, comparisons or measured benefits. Specificity must come from supplied verified "
            "details and original bounded reasoning rather than fabricated reporting.",
            "Insufficient evidence requires research or revision. An industry analysis cannot become "
            "qualified merely by ending with questions, a generic selection criterion or a vivid scenario. "
            "Keep the text concise and honest; do not pad it to a platform word-count target."]
    contract = payload["response_contract"]
    contract["editorial_shape"] = {"genre": "industry_analysis", "method_version": INDUSTRY_METHOD_VERSION,
        "rule": "Follow industry_editorial_method. Several functions may share one paragraph. Preserve "
        "all required factual references and safety gates. Insufficient material requires research or "
        "revision, never a qualified article made only of questions or hypothetical scenes."}
    # Only this program's own generated instructions are replaced. The brief,
    # owner revisions, canonical registry and authorized contexts stay byte-exact.
    contract["safety"] = [rule for rule in contract.get("safety", []) if not any(
        phrase in rule.lower() for phrase in ("sparse evidence", "evidence is sparse", "selection condition or verification question"))]
    method_gate = ("Follow industry_editorial_method without changing factual, role, personal or publication "
        "authority. Missing evidence or reasoning requires research or revision, not invented reporting.")
    if method_gate not in contract["safety"]:
        contract["safety"].append(method_gate)
    payload["industry_narration_contract"] = [
        "Apply industry_editorial_method in natural connected prose, without narrating the method to the reader.",
        "Use optional paragraph_id to join adjacent canonical references and original reasoning naturally; "
        "keep every canonical text, required reference and evidence binding. Do not repeat source passages "
        "in surrounding commentary or expose the internal review process. A proposed public_text needs "
        "the independent verifier; never set verified flags yourself."]
    if "quality_retry" in payload:
        retry = payload["quality_retry"]
        retry["instruction"] = (
            "Revise exactly focus_surface under industry_editorial_method and the original canonical "
            "block contract. Address actual current-text diagnostics, including editorial_instruction "
            "when supplied, without copying diagnostic text into the article. Preserve all other surfaces. "
            "Return top-level blocks/master_title only for master, otherwise exactly the requested variant. "
            "Do not fill missing reporting or a missing conclusion with a hypothetical or a question. "
            "Facts, dates, attribution and quantities stay in canonical references. No new factual, "
            "personal, capability, price, result or publication authority is granted by a repair. "
            "If the argument cannot be supported, leave the draft insufficient for research or revision.")
        retry["required_shape"] = [payload["response_contract"]["editorial_shape"]["rule"]]
        retry["positive_pattern"] = {"do_not_copy_literal_placeholders": True,
            "argument": "A specific authorized observation -> explicit author conclusion -> explained "
                "mechanism -> affected reader's consequence -> meaningful applicability limit.",
            "bad_shapes": ["source-only feature restatement", "an unanswered verification question",
                "an invented scene replacing evidence", "a conclusion with no explained support"]}
        for contract in retry.get("surface_contracts", {}).values():
            contract["positive_writing_recipe"] = [
                "Follow industry_editorial_method for the requested surface, keeping canonical facts "
                "unchanged and developing an explicit supported conclusion with its consequence and limit."]
    if "editorial_revision" in payload:
        payload["editorial_revision"]["instruction"] = (
            "Revise exactly focus_surfaces under industry_editorial_method, preserving already accepted "
            "surfaces and every canonical reference and safety gate. Address the independent reader's "
            "actual observations using a clear supported author conclusion and explained consequence. "
            "Return top-level blocks/master_title only when master is listed and exactly the requested "
            "channels otherwise. Missing evidence requires research or revision, never an invented fact, "
            "hypothetical scene or verification question substituted for the missing argument.")
