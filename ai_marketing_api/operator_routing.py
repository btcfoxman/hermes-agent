"""Server-owned stage routes and a request-bound cost authorization hook.

The orchestrator may install a ProviderRequestObserver to reserve before each
actual HTTP request and reconcile its result. No monetary amount is inferred
from invocation counts or missing usage.
"""

from contextvars import ContextVar
from typing import Any, Awaitable, Callable
import os
from pathlib import Path

import yaml
import httpx


ProviderRequestObserver = Callable[[dict[str, Any]], Awaitable[None]]
PROVIDER_REQUEST_OBSERVER: ContextVar[ProviderRequestObserver | None] = ContextVar(
    "marketing_provider_request_observer", default=None)
SINGLE_PROVIDER_REQUEST: ContextVar[bool] = ContextVar("marketing_single_provider_request", default=False)
BUDGET_RESERVATION_ID: ContextVar[str] = ContextVar("marketing_budget_reservation_id", default="")
PROVIDER_EVENT_SINK: ContextVar[list | None] = ContextVar("marketing_provider_events", default=None)
MAX_PROVIDER_INPUT_BYTES = 150000
MAX_PROVIDER_OUTPUT_TOKENS = 24000
STAGE_MODEL_ENV = {stage: "HERMES_" + stage.upper() + "_MODEL" for stage in
    ("compose", "repair", "planner", "propose", "revise", "claim_verification", "editorial_review")}


def _server_config() -> dict:
    # This standalone edge image intentionally does not import Hermes core.
    path = Path(os.getenv("HERMES_HOME", "/data/hermes")) / "config.yaml"
    if not path.exists():
        return {}
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("server_config_invalid")
    return config.get("marketing_api", {})


def server_stage_route(stage: str) -> dict[str, Any] | None:
    try:
        routes = _server_config().get("stage_routes", {})
        if not isinstance(routes, dict):
            raise ValueError
        if stage not in routes:
            # Existing standalone deployments configure these three server-owned
            # values. An absent optional YAML file must not disable their route.
            stage_model = os.getenv(STAGE_MODEL_ENV.get(stage, ""))
            model = (stage_model if stage_model is not None else os.getenv("HERMES_MODEL", "")).strip()
            if stage_model is not None and (not model or len(model) > 200 or any(ord(c) < 32 for c in model)):
                raise ValueError
            base = os.getenv("HERMES_OPENAI_BASE_URL", "").strip()
            key = os.getenv("HERMES_OPENAI_API_KEY", "").strip()
            return {"model": model, "base_url": base, "api_key": key, "server_default": True} if model and base and key else None
        route = routes[stage]
        if not isinstance(route, dict) or set(route) - {"model", "base_url", "api_key_env"}:
            raise ValueError
        if not all(isinstance(route.get(key), str) and route[key].strip()
                   for key in ("model", "base_url", "api_key_env")):
            raise ValueError
        key = os.getenv(route["api_key_env"], "").strip()
        if not key:
            raise ValueError
        return {"model": route["model"], "base_url": route["base_url"], "api_key": key}
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise ValueError("stage_route_unavailable") from exc


def server_input_byte_limit() -> int:
    """The server may tighten the absolute request bound, never enlarge it."""
    try:
        limits = _server_config().get("provider_limits", {})
        value = limits.get("max_input_bytes", MAX_PROVIDER_INPUT_BYTES)
        if type(value) is not int or value <= 0:
            raise ValueError
        return min(value, MAX_PROVIDER_INPUT_BYTES)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise ValueError("provider_limits_unavailable") from exc


async def observe_provider(event: dict[str, Any]) -> None:
    observer = PROVIDER_REQUEST_OBSERVER.get()
    if observer is not None:
        await observer(event)
        return
    observer_config = _server_config().get("cost_observer")
    if not observer_config:
        return
    key = os.getenv(observer_config.get("api_key_env", ""), "")
    if not key or not observer_config.get("url"):
        raise ValueError("cost_observer_unavailable")
    async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
        response = await client.post(observer_config["url"], json=event,
            headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        if event["phase"] == "before_request" and response.json().get("authorized") is not True:
            raise ValueError("cost_reservation_denied")
