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
    content_request_payload,
    enforce_social_publishability,
    normalize_content_output,
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
        "temperature": 0.3,
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
    output = enforce_social_publishability(
        normalize_content_output(
            OPERATOR_REGISTRY,
            role_id,
            payload,
            contexts,
            fallback,
            candidate,
        )
    )
    candidate_model = str(
        candidate.get("_model") or candidate.get("model") or "fallback"
    ).strip().lower()
    if (
        output.status == ContentStatus.QUALITY_INSUFFICIENT.value
        and candidate_model != "fallback"
        and not candidate.get("_error")
    ):
        retry_payload = {
            **compose_payload,
            "quality_retry": {
                "instruction": (
                    "Rewrite once from scratch. The previous draft was safe "
                    "but not publishable social copy. First choose one concrete "
                    "reader-facing thesis from the approved facts, then return "
                    "original, event-specific editorial blocks for every surface. "
                    "Do not describe the review process, label facts/opinions, "
                    "repeat a generic 'worth watching' wrapper, or end with an "
                    "automatic observation list."
                ),
                "required_shape": [
                    "platform-native hook anchored to a concrete actor, amount, rule, constraint, or consequence from approved claims",
                    "at least two distinct analytical steps that explain mechanism and reader impact without inventing facts",
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
                    if warning.startswith("discarded_unsafe_model_editorial:")
                ],
            },
        }
        retry_candidate = await _llm_json(
            profile.system_prompt,
            retry_payload,
            fallback,
            openai_base_url,
            openai_api_key,
            model_name,
            timeout_seconds,
        )
        return enforce_social_publishability(
            normalize_content_output(
                OPERATOR_REGISTRY,
                role_id,
                payload,
                contexts,
                fallback,
                retry_candidate,
            )
        )
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
