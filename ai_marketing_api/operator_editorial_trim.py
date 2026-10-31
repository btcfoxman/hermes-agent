"""Pure, explicit AI sentence deletion; callers retain every native authorization gate."""
from copy import deepcopy
import hashlib
import json
import re

POLICY_VERSION = "billing_boundary_editorial_trim.v1"
_BILLING_DECLINATION_COMPARISON_RE = re.compile(
    r"便宜|昂贵|更贵|更低|更高|低于|高于|划算|实惠|廉价|低廉|价钱|单价|"
    r"开支|支出|花钱|花费|省下|节约|节减|涨价|降价")
_BILLING_DECLINATION_BLOCKER_RE = re.compile(
    r"[\d零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億%％$￥¥€£]|"
    r"免费|付费|收费|计费|费用|价格|售价|报价|折扣|优惠|省钱|节省|"
    r"降低|下降|下跌|上涨|上升|升高|减少|增加|更低|更高|低于|高于|便宜|昂贵|更贵|"
    r"人民币|美元|欧元|\b(?:USD|CNY|RMB|pay|paid|billing|cost|price|saving)\b|"
    + _BILLING_DECLINATION_COMPARISON_RE.pattern, re.I)
_BILLING_DECLINATION_RE = re.compile(
    r"(?:(?:至于)?(?:这种安排|该安排|这种做法|该做法)是否(?:已经)?带来(?:更广泛的)?"
    r"成本(?:、效率(?:或|和|与)行业习惯)?(?:变化|影响)[，,]"
    r"(?:当前|现有)(?:材料|资料|证据)(?:没有给出|未给出|未提供)(?:足够|充分)(?:依据|证据)[，,]"
    r"(?:因此)?(?:不能|无法)(?:替它)?(?:下结论|作出判断)|"
    r"(?:当前|现有|目前)(?:材料|资料|证据)(?:不足|尚不充分|不充分)[，,]"
    r"(?:因此)?(?:不能|无法)(?:判断|确定)成本(?:影响|变化)|"
    r"(?:当前|现有)(?:材料|资料|证据)(?:尚)?不足以(?:判断|确定)成本(?:影响|变化))"
    r"[。.!！]")


def candidate_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nonassertive_cost_sentence(sentence):
    value = sentence.strip()
    return bool(not _BILLING_DECLINATION_BLOCKER_RE.search(value)
        and _BILLING_DECLINATION_RE.fullmatch(value))


def _editorial_billing_scan_text(text):
    """Classification evidence only; never replaces a native billing scan."""
    if _BILLING_DECLINATION_BLOCKER_RE.search(text):
        return text
    return re.sub(r"[^。！？.!?\r\n]+[。！？.!?]",
        lambda match: " " if _nonassertive_cost_sentence(match.group()) else match.group(), text)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _canonical_texts(canonical_blocks):
    return [text for block in canonical_blocks for key in ("text", "public_text")
        if isinstance(text := _get(block, key), str) and text]


def _unbound_billing_editorial(raw, *, canonical_blocks=()):
    if not isinstance(raw, dict) or raw.get("kind") not in {"opinion", "transition"}:
        return False
    return (str(raw.get("origin") or "").strip().lower() in {"", "model_editorial"}
        and not any(raw.get(key) for key in ("canonical", "canonical_block", "block_ref", "canonical_block_ref", "ref", "binding_hash", "one_of",
            "claim_id", "evidence_ids", "evidence_refs", "references", "source_ids", "source_exact", "locked", "required"))
        and str(raw.get("text") or "").strip() not in {text.strip() for text in _canonical_texts(canonical_blocks)})


def _substantive_residual(text):
    # Structural minimum only. Native normalization and later quality/Owner gates still apply.
    return (len(re.findall(r"[\u3400-\u9fffA-Za-z]", text)) >= 24
        and bool(re.search(r"[。.!！?？]\s*$", text))
        and bool(re.search(r"判断|核对|检查|选择|比较|使用|确认|需要|考虑|取舍|评估|是否|能否|先看|先问|厘清|分清", text)))


def _money_assertion(text):
    return bool(_BILLING_DECLINATION_COMPARISON_RE.search(text)
        or re.search(r"付费|计费|费用|价格|省钱|节省|成本|收费|免费|\b(?:pay|paid|billing|cost|price|saving)\b", text, re.I))


def prepare_editorial_trim(candidate, *, canonical_blocks=(), source_texts=()):
    """Copy one unbound editorial sentence away; return edited value and explicit provenance.

    This pure transform grants no retry, publication, fact verification or quality approval.
    The recovery caller must check native industry/v2/once-child scope before calling it,
    and run the unchanged strict normalizer on the returned candidate afterwards.
    """
    if (not isinstance(candidate, dict) or not isinstance(candidate.get("blocks"), list)
            or not 1 <= len(candidate["blocks"]) <= 80
            or not isinstance(candidate.get("master_title"), str)
            or candidate.get("platform_variants") != []):
        raise ValueError("editorial_trim_candidate_shape")
    canonical_blocks = tuple(canonical_blocks)
    sources = (*_canonical_texts(canonical_blocks), *tuple(source_texts))
    if any(not isinstance(value, str) for value in sources):
        raise ValueError("editorial_trim_source_shape")
    found = []
    for index, raw in enumerate(candidate["blocks"]):
        if not _unbound_billing_editorial(raw, canonical_blocks=canonical_blocks):
            continue
        text = raw.get("text")
        if not isinstance(text, str) or _editorial_billing_scan_text(text) == text:
            continue
        for match in re.finditer(r"[^。！？.!?\r\n]+[。！？.!?]", text):
            sentence = match.group().strip()
            if _nonassertive_cost_sentence(sentence):
                start = match.start() + len(match.group()) - len(match.group().lstrip())
                found.append((index, start, start + len(sentence), sentence))
    if len(found) != 1:
        raise ValueError("editorial_trim_sentence_count")
    index, start, end, removed = found[0]
    if any(removed in value for value in sources):
        raise ValueError("editorial_trim_protected_source")
    old = candidate["blocks"][index]["text"]
    new = old[:start] + old[end:]
    if not _substantive_residual(new):
        raise ValueError("editorial_trim_residual_not_substantive")
    edited = deepcopy(candidate)
    edited["blocks"][index]["text"] = new
    # A monetary/numeric/comparison assertion elsewhere cannot disappear with the
    # removed boilerplate, even when a model split it across separate raw blocks.
    raw_texts = [edited["master_title"], *(str(raw.get(key) or "")
        for raw in edited["blocks"] if isinstance(raw, dict) for key in ("text", "public_text"))]
    if any(_BILLING_DECLINATION_BLOCKER_RE.search(text) or _money_assertion(text) for text in raw_texts):
        raise ValueError("editorial_trim_other_assertion")
    # Canonical quantities remain exact; their money/conditions still face native closure.
    if any(_money_assertion(text) for text in _canonical_texts(canonical_blocks)):
        raise ValueError("editorial_trim_canonical_money")
    metadata = {
        "schema_version": "ai_editorial_sentence_trim.v1", "policy_version": POLICY_VERSION,
        "surface": "master", "block_index": index, "field": "text", "offset_unit": "unicode_codepoints",
        "start": start, "end": end, "old_text_sha256": text_digest(old), "removed_text_sha256": text_digest(removed),
        "new_text_sha256": text_digest(new), "source_candidate_sha256": candidate_digest(candidate),
        "edited_candidate_sha256": candidate_digest(edited), "removed_chars": len(removed), "remaining_chars": len(new),
        "reason": "AI editor removed one nonassertive material-insufficiency boilerplate sentence; no fact or source condition removed",
    }
    return edited, [metadata]
