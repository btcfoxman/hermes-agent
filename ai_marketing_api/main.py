from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import secrets
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException

from ai_marketing_api.operator_content import (
    ContentStatus,
    OperatorComposeRequest,
    OperatorContentOutput,
    build_content_fallback,
    combine_publishable_surfaces,
    content_request_payload,
    enforce_social_publishability,
    missing_publishable_surfaces,
    normalize_content_output,
    publishable_editorial_references,
)
from ai_marketing_api.operator_runtime import (
    CONTENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ContextAuthorizationError,
    OperatorAction,
    OperatorOutput,
    OperatorProposeRequest,
    OperatorRegistry,
    OperatorReviseRequest,
    OperatorRole,
    build_fallback,
    normalize_output,
    request_payload,
)

app = FastAPI(title="Hermes AI Marketing API", version="0.2.0")

DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
OPERATOR_REGISTRY = OperatorRegistry()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Hermes AI Marketing API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "endpoints": {
            "topicAnalyze": "/api/v1/topic/analyze",
            "briefGenerate": "/api/v1/brief/generate",
            "draftGenerate": "/api/v1/draft/generate",
            "draftHumanize": "/api/v1/draft/humanize",
            "riskReview": "/api/v1/risk/review",
            "operatorHealth": "/api/v1/operators/health",
            "operatorPropose": "/api/v1/operators/{role_id}/propose",
            "operatorRevise": "/api/v1/operators/{role_id}/revise",
            "operatorCompose": "/api/v1/operators/{role_id}/compose",
        },
    }


def _require_auth(authorization: Optional[str]) -> None:
    expected = os.getenv("HERMES_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="hermes authentication is not configured")
    prefix = "bearer "
    token = ""
    if authorization and authorization.lower().startswith(prefix):
        token = authorization[len(prefix):].strip()
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid hermes token")


def _require_operator_auth(authorization: Optional[str]) -> None:
    """Operator profiles are never exposed when their service key is absent."""

    _require_auth(authorization)


def _text(payload: Dict[str, Any]) -> str:
    values = [
        payload.get("text"),
        payload.get("body"),
        payload.get("summary"),
        payload.get("title"),
    ]
    context = payload.get("context")
    if isinstance(context, dict):
        values.extend([
            context.get("summary"),
            context.get("fact_card"),
            context.get("angle_card"),
        ])
    return " ".join(str(value or "") for value in values).strip()


def _title(payload: Dict[str, Any]) -> str:
    title = str(payload.get("title") or "").strip()
    if title:
        return title[:160]
    text = _text(payload)
    return (text[:80] or "AI marketing topic").strip()


def _keywords(value: str) -> List[str]:
    words = re.split(r"[\s,.;:!?，。；：、]+", value)
    seen = set()
    result = []
    for word in words:
        cleaned = word.strip("#()[]{}\"'").strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned[:32])
        if len(result) >= 8:
            break
    return result


def _sources(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    sources = payload.get("sources")
    if isinstance(sources, list):
        return [
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
            }
            for item in sources
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ][:8]
    return []


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def _infer_language(value: str) -> str:
    text = str(value or "")
    if not text.strip():
        return ""
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text.lower() if "a" <= char <= "z")
    if cjk_count >= 4 and cjk_count >= max(4, latin_count // 4):
        return "zh-CN"
    if latin_count >= 20:
        return "en"
    return ""


def _source_language(payload: Dict[str, Any]) -> str:
    for key in ("source_language", "language"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    snapshot = payload.get("source_snapshot")
    if isinstance(snapshot, dict):
        value = str(snapshot.get("language") or "").strip()
        if value:
            return value
    return _infer_language(f"{_title(payload)}\n{_text(payload)}")


def _is_chinese_language(language: str) -> bool:
    return str(language or "").lower().startswith(("zh", "cn"))


def _split_facts(text: str, limit: int = 6) -> List[str]:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])|(?<=\.)\s+", raw)
    facts: List[str] = []
    seen = set()
    for part in parts:
        item = part.strip(" \n\t-•")
        if len(item) < 8:
            continue
        key = item[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(item[:320])
        if len(facts) >= limit:
            break
    if not facts and raw:
        facts.append(raw[:360])
    return facts


def _fact_card_from_source(payload: Dict[str, Any]) -> str:
    language = _source_language(payload)
    facts = _split_facts(_text(payload))
    if facts:
        return "\n".join(f"- {item}" for item in facts)
    return "原文未提供可抽取事实。" if _is_chinese_language(language) else "No extractable facts were provided in the source."


def _fallback_topic(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _text(payload)
    return {
        "title": _title(payload),
        "summary": (text[:260] or _title(payload)).strip(),
        "keywords": _keywords(f"{_title(payload)} {text}"),
        "entities": [],
        "topic_score": 0.62,
        "risk_score": 0.12,
        "relevance_score": 0.68,
        "duplicate_score": 0.05,
        "risk_flags": [],
        "recommendation": "accept",
        "reason": "Fallback analysis generated because no LLM key is configured.",
        "preference_suggestions": [],
        "_model": "fallback",
    }


def _fallback_brief(payload: Dict[str, Any]) -> Dict[str, Any]:
    language = _source_language(payload)
    is_zh = _is_chinese_language(language)
    title = _title(payload)
    return {
        "title": title,
        "fact_card": _fact_card_from_source(payload),
        "angle_card": "运营解读：可围绕事件的变化、影响对象和后续动作设计选题角度；该字段不是原文事实。" if is_zh else "Editorial interpretation: frame the angle around what changed, who is affected, and what may happen next. This field is not a source fact.",
        "audience_value": "受众价值：帮助读者快速确认原文事实、判断是否值得继续关注，并识别可能的行动线索。" if is_zh else "Audience value: helps readers verify the source facts, decide whether to keep following the topic, and identify possible next actions.",
        "source_links": _sources(payload),
        "risk_notes": [],
        "recommended_personas": ["operator", "founder", "content strategist"],
        "recommended_platforms": ["wechat_mp", "toutiao"],
        "confidence": 0.55,
        "source_language": language,
        "fact_card_policy": "extractive_source_facts_only",
        "preference_suggestions": [],
        "_model": "fallback",
    }


def _normalize_brief_result(payload: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    fallback = _fallback_brief(payload)
    data = {**fallback, **(result or {})}
    language = _source_language(payload)
    source_text = _text(payload)
    data["source_language"] = data.get("source_language") or language
    data["fact_card_policy"] = "extractive_source_facts_only"
    if not data.get("source_links"):
        data["source_links"] = _sources(payload)
    if not data.get("fact_card"):
        data["fact_card"] = fallback["fact_card"]
    if _is_chinese_language(language) and _contains_cjk(source_text):
        for key in ("fact_card", "angle_card", "audience_value"):
            value = str(data.get(key) or "")
            if value and not _contains_cjk(value):
                data[key] = fallback[key]
        if str(result.get("fact_card") or "") and not _contains_cjk(str(result.get("fact_card") or "")):
            notes = data.get("risk_notes") if isinstance(data.get("risk_notes"), list) else []
            data["risk_notes"] = [*notes, "模型输出语言与原文不一致，事实卡已回退为原文摘录。"]
    return data


def _fallback_draft(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = _title(payload)
    text = (_text(payload) or title).strip()
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    platform = str(payload.get("platform") or context.get("platform") or "wechat_mp")
    content_format = str(context.get("format") or "article")
    body = (
        f"{title}\n\n"
        f"{text[:800]}\n\n"
        "Why it matters\n"
        "This topic can influence positioning, content planning, and distribution decisions.\n\n"
        "Suggested action\n"
        "Add your own case, data, and point of view before publishing."
    )
    html = (
        f"<h2>{_escape_html(title)}</h2>"
        f"<p>{_escape_html(text[:800])}</p>"
        "<h3>Why it matters</h3>"
        "<p>This topic can influence positioning, content planning, and distribution decisions.</p>"
        "<h3>Suggested action</h3>"
        "<p>Add your own case, data, and point of view before publishing.</p>"
    )
    return {
        "title": title,
        "description": text[:180],
        "body": body,
        "html": html,
        "blocks": [
            {"type": "heading", "level": 2, "text": title},
            {"type": "paragraph", "text": text[:800]},
            {"type": "heading", "level": 3, "text": "Why it matters"},
            {
                "type": "paragraph",
                "text": "This topic can influence positioning, content planning, and distribution decisions.",
            },
        ],
        "components": [],
        "topics": [],
        "media": [],
        "cover_url": None,
        "platform": platform,
        "format": content_format,
        "risk_flags": [],
        "preference_suggestions": [
            {
                "type": "draft_pattern",
                "title": "草稿结构样本",
                "rationale": "记录本次草稿结构，待人工审核后判断是否沉淀为偏好。",
                "preferences": {
                    "draft_structures": [
                        {
                            "platform": platform,
                            "content_type": content_format,
                            "sections": ["opening", "why_it_matters", "suggested_action"],
                        }
                    ]
                },
            }
        ],
        "_model": "fallback",
    }


def _fallback_humanize(payload: Dict[str, Any]) -> Dict[str, Any]:
    draft = payload.get("draft") if isinstance(payload.get("draft"), dict) else payload
    title = str(draft.get("title") or _title(payload)).strip()
    description = str(draft.get("description") or "").strip()
    body = str(draft.get("body") or _text(draft) or _text(payload)).strip()
    html = str(draft.get("html") or "").strip()
    if body:
        body = (
            body.replace("Additionally,", "")
            .replace("In conclusion,", "")
            .replace("It is important to note that", "值得注意的是")
            .strip()
        )
    if not html and body:
        html = "<p>" + _escape_html(body).replace("\n\n", "</p><p>").replace("\n", "<br/>") + "</p>"
    return {
        "title": title,
        "description": description,
        "body": body,
        "html": html,
        "blocks": draft.get("blocks") or [],
        "components": draft.get("components") or [],
        "topics": draft.get("hashtags") or [],
        "media": draft.get("media") or [],
        "cover_url": draft.get("cover_url"),
        "platform": draft.get("platform") or "wechat_mp",
        "format": draft.get("format") or "article",
        "risk_flags": draft.get("risk_flags") or [],
        "preference_suggestions": [
            {
                "type": "humanize_pattern",
                "title": "润色处理样本",
                "rationale": "记录本次润色处理方式，待人工确认后可作为账号表达偏好。",
                "preferences": {
                    "humanize_rules": [
                        "减少套话，保留事实、图片、视频和组件。",
                    ]
                },
            }
        ],
        "_model": "fallback",
    }


def _escape_html(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_from_response(content: str) -> Dict[str, Any]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model returned non-object JSON")
    return data


def _controlled_system_prompt(payload: Dict[str, Any], fallback: str) -> str:
    control = payload.get("prompt_control")
    if isinstance(control, dict):
        system_prompt = str(control.get("system_prompt") or "").strip()
        if system_prompt:
            contract = control.get("response_contract")
            if contract:
                return f"{system_prompt}\n\nResponse contract:\n{json.dumps(contract, ensure_ascii=False, default=str)}"
            return system_prompt
    return fallback


def _timeout_seconds(value: Optional[str] = None) -> float:
    raw = (value or os.getenv("HERMES_LLM_TIMEOUT_SECONDS") or os.getenv("HERMES_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return timeout if timeout > 0 else DEFAULT_LLM_TIMEOUT_SECONDS


def _resolve_host_addresses(hostname: str, port: int) -> set[str]:
    addresses: set[str] = set()
    for family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
    ):
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        addresses.add(str(sockaddr[0]).split("%", 1)[0])
    return addresses


@dataclass(frozen=True)
class _PinnedOpenAITarget:
    request_base_urls: tuple[str, ...]
    host_header: str
    sni_hostname: str


async def _resolve_public_openai_target(base_url: str) -> Optional[_PinnedOpenAITarget]:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port or 443
        hostname = str(parsed.hostname or "").rstrip(".").lower()
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None
    if (
        parsed.scheme != "https"
        or not ascii_hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or ascii_hostname in {"localhost", "localhost.localdomain"}
    ):
        return None
    try:
        addresses = await asyncio.to_thread(_resolve_host_addresses, ascii_hostname, port)
    except (OSError, UnicodeError, ValueError):
        return None
    if not addresses:
        return None
    try:
        public_addresses = [ipaddress.ip_address(address) for address in addresses]
    except ValueError:
        return None
    if any(not address.is_global for address in public_addresses):
        return None

    public_addresses.sort(key=lambda address: (address.version, int(address)))
    base_path = parsed.path.rstrip("/")
    request_base_urls = tuple(
        f"https://{'[' + str(address) + ']' if address.version == 6 else address}"
        f"{':' + str(port) if port != 443 else ''}{base_path}"
        for address in public_addresses
    )
    try:
        source_ip = ipaddress.ip_address(ascii_hostname)
    except ValueError:
        source_ip = None
    host = f"[{ascii_hostname}]" if source_ip and source_ip.version == 6 else ascii_hostname
    host_header = f"{host}:{port}" if port != 443 else host
    return _PinnedOpenAITarget(
        request_base_urls=request_base_urls,
        host_header=host_header,
        sni_hostname=ascii_hostname,
    )


async def _llm_json(
    system: str,
    payload: Dict[str, Any],
    fallback: Dict[str, Any],
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_seconds: Optional[str] = None,
    *,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    requested_base_url = str(openai_base_url or "").strip()
    requested_api_key = str(openai_api_key or "").strip()
    if bool(requested_base_url) != bool(requested_api_key):
        blocked = dict(fallback)
        blocked["_error"] = "openai_override_base_url_and_api_key_must_be_supplied_together"
        return blocked
    if requested_base_url:
        # A caller-selected endpoint must never inherit the service's own key.
        base_url = requested_base_url.rstrip("/")
        api_key = requested_api_key
    else:
        api_key = os.getenv("HERMES_OPENAI_API_KEY", "").strip()
        base_url = os.getenv(
            "HERMES_OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ).strip().rstrip("/")
    if not api_key:
        return fallback
    target = await _resolve_public_openai_target(base_url)
    if target is None:
        blocked = dict(fallback)
        blocked["_error"] = "invalid_openai_base_url"
        return blocked
    model = (model_name or os.getenv("HERMES_MODEL", "gpt-4.1-mini")).strip() or "gpt-4.1-mini"
    request = {
        "model": model,
        "temperature": max(0.0, min(float(temperature), 1.0)),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
    }
    resolved_timeout = _timeout_seconds(timeout_seconds)
    try:
        async with httpx.AsyncClient(
            timeout=resolved_timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            last_error: Optional[Exception] = None
            for request_base_url in target.request_base_urls:
                try:
                    resp = await client.post(
                        f"{request_base_url}/chat/completions",
                        json=request,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "Host": target.host_header,
                        },
                        extensions={"sni_hostname": target.sni_hostname},
                    )
                    if resp.is_redirect:
                        raise RuntimeError("llm_redirect_forbidden")
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_error = exc
            else:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("no_public_llm_address")
        content = data["choices"][0]["message"]["content"]
        result = _json_from_response(content)
        result["_model"] = model
        return result
    except Exception as exc:
        fallback = dict(fallback)
        fallback["_error"] = str(exc)
        fallback["_timeout_seconds"] = resolved_timeout
        return fallback


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/operators/health")
def operator_health(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Deployment probe for the three bundled, immutable operator profiles."""

    _require_operator_auth(authorization)
    return OPERATOR_REGISTRY.health()


@app.get("/api/v1/operators/{role_id}/probe")
def operator_probe(
    role_id: OperatorRole,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_operator_auth(authorization)
    profile = OPERATOR_REGISTRY.get(role_id)
    return {
        "status": "ok",
        "role_id": role_id.value,
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "system_prompt_sha256": profile.prompt_sha256,
        "knowledge_source": "request_authorized_context_only",
        "direct_tools": list(profile.allowed_tools),
        "supported_actions": sorted(profile.allowed_actions),
        "schema_versions": {
            "proposal": SCHEMA_VERSION,
            "content": CONTENT_SCHEMA_VERSION,
        },
    }


async def _run_operator(
    role_id: OperatorRole,
    action: OperatorAction,
    payload: OperatorProposeRequest,
    authorization: Optional[str],
    openai_base_url: Optional[str],
    openai_api_key: Optional[str],
    model_name: Optional[str],
    timeout_seconds: Optional[str],
) -> OperatorOutput:
    _require_operator_auth(authorization)
    profile = OPERATOR_REGISTRY.get(role_id)
    if action.value not in profile.allowed_actions:
        raise HTTPException(status_code=403, detail={"code": "operator_action_denied"})
    if isinstance(payload, OperatorReviseRequest) and payload.previous.role_id != role_id.value:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "operator_role_mismatch",
                "message": "A revision must stay inside the profile that produced the previous output.",
            },
        )
    try:
        contexts = OPERATOR_REGISTRY.authorize(role_id, payload.authorized_context, payload.as_of)
        OPERATOR_REGISTRY.authorize_preferences(
            role_id, payload.approved_preferences
        )
    except ContextAuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail={"code": exc.code, "record_id": exc.record_id, "message": str(exc)},
        ) from exc

    fallback = build_fallback(role_id, action, payload, contexts)
    candidate = await _llm_json(
        profile.system_prompt,
        request_payload(payload),
        fallback,
        openai_base_url,
        openai_api_key,
        model_name,
        timeout_seconds,
    )
    return normalize_output(
        OPERATOR_REGISTRY,
        role_id,
        action,
        payload,
        contexts,
        fallback,
        candidate,
    )


async def _run_compose(
    role_id: OperatorRole,
    payload: OperatorComposeRequest,
    authorization: Optional[str],
    openai_base_url: Optional[str],
    openai_api_key: Optional[str],
    model_name: Optional[str],
    timeout_seconds: Optional[str],
) -> OperatorContentOutput:
    """Compose publishable drafts from an orchestration-approved direction.

    The model never receives a database/cache/tool handle.  The current
    request context is re-authorized here and the content normalizer rebuilds
    every factual block before any prose is returned.
    """

    _require_operator_auth(authorization)
    profile = OPERATOR_REGISTRY.get(role_id)
    if "compose" not in profile.allowed_actions:
        raise HTTPException(status_code=403, detail={"code": "operator_action_denied"})
    try:
        contexts = OPERATOR_REGISTRY.authorize(role_id, payload.authorized_context, payload.as_of)
        OPERATOR_REGISTRY.authorize_preferences(
            role_id, payload.approved_preferences
        )
    except ContextAuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail={"code": exc.code, "record_id": exc.record_id, "message": str(exc)},
        ) from exc

    fallback = build_content_fallback(OPERATOR_REGISTRY, role_id, payload, contexts)
    compose_payload: Dict[str, Any] = {}
    if fallback.get("_fallback_errors"):
        # A blocked evidence/disclosure gate must not be sent to the model.
        # The deterministic result already contains the questions and critic
        # state orchestration needs to recover safely.
        candidate = fallback
    else:
        compose_payload = content_request_payload(role_id, payload, contexts)
        candidate = await _llm_json(
            profile.system_prompt,
            compose_payload,
            fallback,
            openai_base_url,
            openai_api_key,
            model_name,
            timeout_seconds,
        )
    normalized = normalize_content_output(
        OPERATOR_REGISTRY,
        role_id,
        payload,
        contexts,
        fallback,
        candidate,
    )
    normalized_attempts = [normalized]
    output = enforce_social_publishability(normalized)
    candidate_model = str(
        candidate.get("_model") or candidate.get("model") or "fallback"
    ).strip().lower()

    def candidate_authored_text(
        raw_candidate: Dict[str, Any],
        surface: str,
    ) -> List[str]:
        """Return bounded public model prose for one surface."""

        raw_blocks: Any = raw_candidate.get("blocks")
        if surface != "master":
            variants = raw_candidate.get("platform_variants")
            raw_blocks = None
            if isinstance(variants, list):
                raw_blocks = next(
                    (
                        item.get("blocks")
                        for item in variants
                        if isinstance(item, dict)
                        and str(item.get("platform") or "") == surface
                    ),
                    None,
                )
        if not isinstance(raw_blocks, list):
            return []
        return [
            str(item.get("text") or "").strip()[:320]
            for item in raw_blocks
            if isinstance(item, dict)
            and not item.get("block_ref")
            and str(item.get("text") or "").strip()
        ][:6]

    rejected_text_by_surface = {
        surface: candidate_authored_text(candidate, surface)
        for surface in ["master", *compose_payload.get("channels", [])]
    }
    # A multi-platform response often has five good surfaces and one bad
    # short-feed variant. Repair one surface at a time so a small model can
    # concentrate on a real hook and mechanism instead of reproducing the
    # whole eight-channel JSON contract on every retry. Whole-surface
    # aggregation keeps every independently valid draft. The repair budget is
    # capped globally and per surface, and remains fail closed: no partial or
    # template surface leaves this endpoint.
    surface_attempts: Dict[str, int] = {}
    for retry_number in range(1, 19):
        if (
            output.status != ContentStatus.QUALITY_INSUFFICIENT.value
            or candidate_model == "fallback"
            or candidate.get("_error")
        ):
            break
        failed_surfaces = missing_publishable_surfaces(
            normalized_attempts,
            compose_payload.get("channels", []),
        )
        if not failed_surfaces:
            break
        eligible_surfaces = [
            surface
            for surface in failed_surfaces
            if surface_attempts.get(surface, 0) < 3
        ]
        if not eligible_surfaces:
            break
        # Rotate across remaining failures before giving any one surface a
        # second attempt. This prevents a stubborn master draft from starving
        # otherwise repairable platform variants.
        retry_surface = min(
            eligible_surfaces,
            key=lambda surface: (
                surface_attempts.get(surface, 0),
                eligible_surfaces.index(surface),
            ),
        )
        surface_attempts[retry_surface] = surface_attempts.get(retry_surface, 0) + 1
        requested_surfaces = [retry_surface]
        retry_channels = [] if retry_surface == "master" else [retry_surface]
        retry_platform_briefs = compose_payload.get("platform_editorial_briefs")
        if isinstance(retry_platform_briefs, dict):
            retry_platform_briefs = {
                channel: brief
                for channel, brief in retry_platform_briefs.items()
                if channel in retry_channels
            }
        approved_proposal = compose_payload.get("approved_proposal")
        if isinstance(approved_proposal, dict):
            approved_proposal = {
                **approved_proposal,
                "suggested_formats": list(retry_channels),
            }
        surface_contracts: Dict[str, Any] = {}
        for surface in requested_surfaces:
            is_long = surface == "master" or surface in {"wechat_mp", "toutiao"}
            surface_contracts[surface] = {
                "authored_blocks_required": 2 if is_long else 1,
                "required_reading_order": (
                    [
                        "original authored hook first",
                        "all required canonical evidence block_ref objects",
                        "a distinct original authored analysis after the evidence",
                    ]
                    if is_long
                    else [
                        "one original authored hook/implication block first",
                        "all required canonical evidence block_ref objects",
                    ]
                ),
                "positive_writing_recipe": [
                    "Open from the directly affected party's concrete constraint, choice, cost, or bargaining position; the canonical evidence block will report the event and numbers.",
                    "Explain one operating mechanism: cash occupation, settlement, contract boundary, workflow, cost, bargaining position, or available choice.",
                    "Use short declarative sentences in the source language. Mark uncertain future effects with may, depends on, or an equivalent cautious phrase.",
                    "End on one observable operating consequence, not on the importance of the story or a list of things to watch.",
                    "Do not restate the source paragraph in an authored block; the canonical evidence block already renders the approved facts.",
                ],
            }
        validated_reference_copy = publishable_editorial_references(
            normalized_attempts,
        )
        required_block_refs = [
            {"block_ref": block.get("block_ref")}
            for block in compose_payload.get("canonical_block_registry", [])
            if isinstance(block, dict)
            and block.get("required")
            and block.get("block_ref")
        ]

        def repair_blocks(surface: str) -> List[Any]:
            hook = {
                "kind": "transition|opinion",
                "text": (
                    "original platform-native hook; no number, source restatement, "
                    "event-reporting verb, or banned wrapper"
                ),
                "evidence_ids": [],
            }
            blocks: List[Any] = [hook, *required_block_refs]
            if surface == "master" or surface in {"wechat_mp", "toutiao"}:
                blocks.append(
                    {
                        "kind": "opinion|transition",
                        "text": (
                            "distinct mechanism or reader-impact analysis; no number, "
                            "source restatement, future-watch ending, or banned wrapper"
                        ),
                        "evidence_ids": [],
                    }
                )
            return blocks

        repair_response_contract = {
            "return_only_these_top_level_keys": [
                "master_title",
                "blocks",
                "platform_variants",
            ],
            "master_title": (
                "required distinct string"
                if retry_surface == "master"
                else "omit this key; the server retains the validated master title"
            ),
            "blocks": (
                repair_blocks("master")
                if retry_surface == "master"
                else "omit this key; the server retains the validated master blocks"
            ),
            "exact_platform_variant_count": len(retry_channels),
            "platform_variants": [
                {
                    "platform": channel,
                    "title": f"required distinct title for {channel}",
                    "blocks": repair_blocks(channel),
                }
                for channel in retry_channels
            ],
            "hard_rule": (
                "Return every platform listed above exactly once. Do not replace "
                "platform_variants with variants, channels, a schema, prose, or one example."
            ),
        }
        retry_payload = {
            "objective": "Repair exactly one failed social-content surface.",
            "channels": list(retry_channels),
            "approved_proposal": (
                {
                    key: approved_proposal.get(key)
                    for key in (
                        "title",
                        "angle",
                        "audience_value",
                        "suggested_formats",
                        "cta",
                        "first_person",
                    )
                    if approved_proposal.get(key) not in (None, "", [])
                }
                if isinstance(approved_proposal, dict)
                else approved_proposal
            ),
            "approved_editorial_brief": compose_payload.get(
                "approved_editorial_brief"
            ),
            "publication_brief": compose_payload.get("publication_brief"),
            "platform_editorial_briefs": retry_platform_briefs,
            "approved_preferences": compose_payload.get("approved_preferences", []),
            # The canonical registry is the complete factual universe for a
            # repair. Omitting duplicated claims and authorized_context keeps
            # the focused request small without weakening the normalizer,
            # which still validates against the original frozen request.
            "canonical_block_registry": compose_payload.get(
                "canonical_block_registry", []
            ),
            "response_contract": repair_response_contract,
            "quality_retry": {
                "attempt": retry_number,
                "failed_surfaces": failed_surfaces,
                "requested_surfaces": requested_surfaces,
                "requested_channels": list(retry_channels),
                "focus_surface": retry_surface,
                "surface_attempt": surface_attempts[retry_surface],
                "previous_rejected_authored_text": rejected_text_by_surface.get(
                    retry_surface, []
                ),
                "validated_reference_copy": validated_reference_copy,
                "instruction": (
                    "Return a compact repair only for focus_surface. If focus_surface is "
                    "master, repair the top-level blocks; otherwise return exactly one "
                    "platform_variants item for that channel and omit top-level blocks. The "
                    "previous normalized draft still failed the social-copy gate. First "
                    "choose one concrete reader-facing thesis from the approved facts. Do "
                    "not write another news lead: let the canonical evidence block report "
                    "the event, action, and amount. Start the authored hook from the affected "
                    "party's operating constraint, choice, cost, cash flow, contract term, or "
                    "bargaining position, then explain why the approved remedy changes that "
                    "mechanism. Use a direct subject-consequence declaration and return "
                    "original, event-specific editorial blocks for the requested surface. "
                    "Do not describe the review process, label facts/opinions, "
                    "repeat a generic 'worth watching', 'not ... but', or "
                    "'not only ... but also' "
                    "wrapper. For Chinese industry copy, never use 最直接的变化, "
                    "最直接感受到, 接下来要看/盯, 后面要看, 后续看点, "
                    "这类事件/处罚落到业务上, 真正改变的是, 真正要变的是, "
                    "规则被推到台前, or 规则会被重新审视. Do not use event-reporting verbs "
                    "such as 已经, 宣布, 发布, 推出, 发生, 据悉, or 消息称 in an "
                    "authored block; the canonical evidence block owns those facts. "
                    "During repair, put no Arabic number or Chinese counted quantity "
                    "in authored blocks; approved numbers remain visible in the canonical "
                    "evidence. Do not mix untranslated English into "
                    "Chinese prose, or end "
                    "with an automatic observation list. Do not invent loaded labels "
                    "such as grey fees, black-box practices, rip-offs, or scandals. "
                    "Never select a server-template block_ref; the registry contains "
                    "approved evidence refs only. Do not reuse any phrase in "
                    "previous_rejected_authored_text. Write only focus_surface in this "
                    "repair request; "
                    "other valid surfaces are retained independently by the server. "
                    "When validated_reference_copy is present, preserve its concrete "
                    "thesis and mechanism while changing the wording and rhythm for "
                    "the requested platform; never copy it verbatim or repeat the source."
                ),
                "surface_contracts": surface_contracts,
                "positive_pattern": {
                    "do_not_copy_literal_placeholders": True,
                    "hook": (
                        "[directly affected party] + [specific operating constraint or choice] "
                        "+ [clear consequence of the approved remedy]"
                    ),
                    "analysis": (
                        "[name one cash-flow, settlement, contract, workflow, cost, incentive, "
                        "or bargaining mechanism] + [explain how the approved rule/remedy "
                        "changes that mechanism] + [firm useful judgment]"
                    ),
                    "bad_shapes": [
                        "retelling who announced what and how much",
                        "saying the important point is not X but Y",
                        "calling the event worth watching without explaining a mechanism",
                        "ending with a generic list of future observations",
                    ],
                },
                "required_shape": [
                    "one platform-native authored hook anchored to an affected party, rule, constraint, choice, cost, or consequence from approved claims; leave event reporting and numbers to the canonical block",
                    "for master: at least one distinct authored analytical block after the hook that explains mechanism or reader impact without inventing facts",
                    "for wechat_mp and toutiao: at least one authored analytical block after the hook",
                    "for every shorter platform: one authored block may combine the hook and one concrete implication; do not pad it with a generic second paragraph",
                    "a firm useful landing; CTA only when it is genuinely natural for that platform",
                ],
                "block_binding_rule": (
                    "For every authored transition, opinion, or CTA block, "
                    "set evidence_ids to [] and omit claim_id and block_ref. "
                    "Only a canonical fact/identity/experience block may use "
                    "block_ref or retain evidence bindings."
                ),
                "critic_errors": list(output.critic.errors),
                "discarded_block_diagnostics": [
                    warning
                    for warning in output.critic.warnings
                    if warning.startswith(
                        (
                            "discarded_unsafe_model_editorial:",
                            "model_platform_variant_missing:",
                            "model_platform_blocks_missing:",
                        )
                    )
                ],
            },
        }
        candidate = await _llm_json(
            profile.system_prompt,
            retry_payload,
            fallback,
            openai_base_url,
            openai_api_key,
            model_name,
            timeout_seconds,
            # Keep repair output steady while allowing a rejected stock phrase
            # to escape the exact same low-temperature completion.
            temperature=0.3,
        )
        latest_rejected_text = candidate_authored_text(candidate, retry_surface)
        if latest_rejected_text:
            rejected_text_by_surface[retry_surface] = latest_rejected_text
        normalized = normalize_content_output(
            OPERATOR_REGISTRY,
            role_id,
            payload,
            contexts,
            fallback,
            candidate,
        )
        normalized_attempts.append(normalized)
        combined = combine_publishable_surfaces(normalized_attempts)
        output = enforce_social_publishability(combined or normalized)
        candidate_model = str(
            candidate.get("_model") or candidate.get("model") or "fallback"
        ).strip().lower()
    return output


@app.post("/api/v1/operators/{role_id}/propose", response_model=OperatorOutput)
async def operator_propose(
    role_id: OperatorRole,
    payload: OperatorProposeRequest,
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> OperatorOutput:
    return await _run_operator(
        role_id,
        OperatorAction.PROPOSE,
        payload,
        authorization,
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/operators/{role_id}/revise", response_model=OperatorOutput)
async def operator_revise(
    role_id: OperatorRole,
    payload: OperatorReviseRequest,
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> OperatorOutput:
    return await _run_operator(
        role_id,
        OperatorAction.REVISE,
        payload,
        authorization,
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/operators/{role_id}/compose", response_model=OperatorContentOutput)
async def operator_compose(
    role_id: OperatorRole,
    payload: OperatorComposeRequest,
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> OperatorContentOutput:
    return await _run_compose(
        role_id,
        payload,
        authorization,
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/topic/analyze")
async def analyze_topic(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        _controlled_system_prompt(
            payload,
            "Return strict JSON for topic analysis with title, summary, keywords, topic_score, risk_score, relevance_score, duplicate_score, risk_flags, recommendation, reason, and optional preference_suggestions. If preference_context is provided, use it as historical guidance, not as source fact.",
        ),
        payload,
        _fallback_topic(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/brief/generate")
async def generate_brief(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    result = await _llm_json(
        _controlled_system_prompt(
            payload,
            (
                "Return strict JSON for a source-grounded editorial brief. "
                "If preference_context is provided, use it only for angle/tone guidance and keep fact_card source-grounded. "
                "Use the same language as source_language or the source text. Do not translate. "
                "fact_card must contain only facts explicitly present in the source text; do not add assumptions, advice, background, or interpretation. "
                "If a fact is not stated, say it is not stated in the source language. "
                "angle_card and audience_value are editorial interpretation fields; keep them separate from fact_card and use the same language as the source. "
                "Return title, fact_card, angle_card, audience_value, source_links, risk_notes, recommended_personas, recommended_platforms, confidence, and optional preference_suggestions."
            ),
        ),
        payload,
        _fallback_brief(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )
    return _normalize_brief_result(payload, result)


@app.post("/api/v1/draft/generate")
async def generate_draft(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        _controlled_system_prompt(
            payload,
            "Return strict JSON for a publishable article draft with title, description, body, html, blocks, components, topics, media, cover_url, platform, format, risk_flags, and optional preference_suggestions. If preference_context is provided, follow accepted style, structure, avoid_patterns, and successful examples while preserving source facts.",
        ),
        payload,
        _fallback_draft(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/draft/humanize")
async def humanize_draft(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
    x_hermes_timeout_seconds: Optional[str] = Header(default=None, alias="X-Hermes-Timeout-Seconds"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        _controlled_system_prompt(
            payload,
            "Return strict JSON for a humanized article draft. Preserve facts, links, media, platform, and format. If preference_context is provided, apply accepted human editing preferences and avoid rejected patterns. Rewrite only title, description, body, html, and blocks to remove AI writing patterns and add a more natural editorial voice. Return title, description, body, html, blocks, components, topics, media, cover_url, platform, format, risk_flags, and optional preference_suggestions.",
        ),
        payload,
        _fallback_humanize(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
        x_hermes_timeout_seconds,
    )


@app.post("/api/v1/risk/review")
async def review_risk(payload: Dict[str, Any], authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_auth(authorization)
    text = _text(payload)
    flags = []
    lowered = text.lower()
    for keyword in ("guaranteed", "risk-free", "medical cure", "investment advice"):
        if keyword in lowered:
            flags.append(keyword)
    return {
        "status": "pass" if not flags else "needs_review",
        "risk_flags": flags,
        "risk_score": min(0.9, 0.15 + len(flags) * 0.2),
        "notes": [],
        "_model": "rules",
    }
