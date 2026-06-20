from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="Hermes AI Marketing API", version="0.1.0")


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
        },
    }


def _require_auth(authorization: Optional[str]) -> None:
    expected = os.getenv("HERMES_API_KEY", "").strip()
    if not expected:
        return
    prefix = "bearer "
    token = ""
    if authorization and authorization.lower().startswith(prefix):
        token = authorization[len(prefix):].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid hermes token")


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
        "_model": "fallback",
    }


def _fallback_brief(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = _title(payload)
    summary = (_text(payload)[:320] or title).strip()
    return {
        "title": title,
        "fact_card": summary,
        "angle_card": "Explain why this matters now and connect it to a concrete operator workflow.",
        "audience_value": "Helps the audience evaluate timing, risks, and practical next steps.",
        "source_links": _sources(payload),
        "risk_notes": [],
        "recommended_personas": ["operator", "founder", "content strategist"],
        "recommended_platforms": ["wechat_mp", "toutiao"],
        "confidence": 0.55,
        "_model": "fallback",
    }


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


async def _llm_json(
    system: str,
    payload: Dict[str, Any],
    fallback: Dict[str, Any],
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = (openai_api_key or os.getenv("HERMES_OPENAI_API_KEY", "")).strip()
    if not api_key:
        return fallback
    base_url = (openai_base_url or os.getenv("HERMES_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
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
    try:
        async with httpx.AsyncClient(timeout=80.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=request,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        result = _json_from_response(content)
        result["_model"] = model
        return result
    except Exception as exc:
        fallback = dict(fallback)
        fallback["_error"] = str(exc)
        return fallback


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/topic/analyze")
async def analyze_topic(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        "Return strict JSON for topic analysis with title, summary, keywords, topic_score, risk_score, relevance_score, duplicate_score, risk_flags, recommendation, reason.",
        payload,
        _fallback_topic(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
    )


@app.post("/api/v1/brief/generate")
async def generate_brief(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        "Return strict JSON for an editorial brief with title, fact_card, angle_card, audience_value, source_links, risk_notes, recommended_personas, recommended_platforms, confidence.",
        payload,
        _fallback_brief(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
    )


@app.post("/api/v1/draft/generate")
async def generate_draft(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        "Return strict JSON for a publishable article draft with title, description, body, html, blocks, components, topics, media, cover_url, platform, format, risk_flags.",
        payload,
        _fallback_draft(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
    )


@app.post("/api/v1/draft/humanize")
async def humanize_draft(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_hermes_openai_base_url: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-Base-URL"),
    x_hermes_openai_api_key: Optional[str] = Header(default=None, alias="X-Hermes-OpenAI-API-Key"),
    x_hermes_model: Optional[str] = Header(default=None, alias="X-Hermes-Model"),
) -> Dict[str, Any]:
    _require_auth(authorization)
    return await _llm_json(
        "Return strict JSON for a humanized article draft. Preserve facts, links, media, platform, and format. Rewrite only title, description, body, html, and blocks to remove AI writing patterns and add a more natural editorial voice. Return title, description, body, html, blocks, components, topics, media, cover_url, platform, format, risk_flags.",
        payload,
        _fallback_humanize(payload),
        x_hermes_openai_base_url,
        x_hermes_openai_api_key,
        x_hermes_model,
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
