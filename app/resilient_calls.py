from __future__ import annotations

import os
import time
from dataclasses import dataclass

from api_clients import (
    PROVIDERS,
    ProviderError,
    agent_cli_available,
    call_provider,
    disabled_providers,
    resolve_agent_cli_model_label,
)
from nvidia_catalog import all_available_models, available_models_for_role, smartest_chat_models


@dataclass(frozen=True)
class Attempt:
    provider: str
    model: str
    status: str
    detail: str


@dataclass(frozen=True)
class ResilientResult:
    content: str
    provider: str
    model: str
    attempts: list[Attempt]


def call_nvidia_with_fallback(
    role: str,
    prompt: str,
    *,
    candidates: list[dict] | None = None,
    limit: int = 6,
    fallback_providers: tuple[str, ...] = (),
) -> ResilientResult:
    attempts: list[Attempt] = []
    models = candidates or smartest_chat_models(role, limit=limit)
    if len(models) < limit:
        seen = {model["id"] for model in models}
        for model in available_models_for_role(role):
            if model["id"] not in seen and model.get("endpoint_type") == "chat":
                models.append(model)
                seen.add(model["id"])
            if len(models) >= limit:
                break

    for model in models[:limit]:
        model_id = model["id"]
        if model.get("endpoint_type") != "chat":
            attempts.append(Attempt("NVIDIA NIM", model_id, "skipped", f"Endpoint type is {model.get('endpoint_type')}"))
            continue
        try:
            content = call_provider("NVIDIA NIM", role, prompt, model_override=model_id)
            attempts.append(Attempt("NVIDIA NIM", model_id, "ok", ""))
            return ResilientResult(content, "NVIDIA NIM", model_id, attempts)
        except ProviderError as exc:
            attempts.append(Attempt("NVIDIA NIM", model_id, _error_status(exc), _safe_error(exc)))
            if exc.is_rate_limit or "provider-level rate limit" in str(exc).lower():
                break
            continue
        except Exception as exc:  # noqa: BLE001
            attempts.append(Attempt("NVIDIA NIM", model_id, "error", str(exc)[:300]))
            continue

    for provider in fallback_providers:
        try:
            content = call_provider(provider, role, prompt)
            attempts.append(Attempt(provider, "", "ok", ""))
            return ResilientResult(content, provider, "", attempts)
        except Exception as exc:  # noqa: BLE001
            attempts.append(Attempt(provider, "", "failed", str(exc)[:300]))

    summary = "\n".join(f"- {item.provider} {item.model}: {item.status} {item.detail}".strip() for item in attempts)
    raise ProviderError(f"No configured model route succeeded.\n{summary}", retryable=True)


def best_available_routes(role: str, limit: int = 14) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    routes.extend(_preferred_frontier_routes(role))
    for model in smartest_chat_models(role, limit=min(10, limit)):
        routes.append(("NVIDIA NIM", model["id"]))

    disabled = disabled_providers()
    for provider_name in ["Gemini", "OpenAI API", "Kimi / Moonshot"]:
        provider = PROVIDERS[provider_name]
        if provider.name in disabled:
            continue
        if provider.env_key and not os.environ.get(provider.env_key):
            continue
        routes.append((provider.name, provider.model))

    seen = set()
    deduped = []
    for route in routes:
        if route in seen:
            continue
        seen.add(route)
        deduped.append(route)
        if len(deduped) >= limit:
            break
    return deduped


def _preferred_frontier_routes(role: str) -> list[tuple[str, str]]:
    available = {model["id"] for model in all_available_models()}
    routes: list[tuple[str, str]] = []
    disabled = disabled_providers()
    # Subscription-backed CLI lead routes first, when the CLI is on PATH. These
    # authenticate through the local CLI login (no API key) and are preferred
    # over paid API credits.
    for provider_name in ("Cursor Agent CLI", "Claude Code CLI", "Codex CLI"):
        provider = PROVIDERS[provider_name]
        if provider.name in disabled:
            continue
        if not agent_cli_available(provider_name):
            continue
        routes.append((provider.name, provider.model))
    # API-key lead frontier models next, only when their API key is present.
    for provider_name in ("Anthropic API", "OpenAI API"):
        provider = PROVIDERS[provider_name]
        if provider.name in disabled:
            continue
        if provider.env_key and not os.environ.get(provider.env_key):
            continue
        routes.append((provider.name, provider.model))
    preferred_nvidia = [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "openai/gpt-oss-120b",
        "deepseek-ai/deepseek-v4-pro",
        "minimaxai/minimax-m3",
    ]
    routes.extend(("NVIDIA NIM", model_id) for model_id in preferred_nvidia if model_id in available)
    provider = PROVIDERS["Gemini"]
    if provider.name not in disabled and (not provider.env_key or os.environ.get(provider.env_key)):
        routes.append(("Gemini", provider.model))
    return routes


def call_best_available_with_fallback(
    role: str,
    prompt: str,
    *,
    limit: int = 14,
    local_timeout_seconds: int = 120,
    max_cooldown_wait_seconds: int | None = None,
) -> ResilientResult:
    attempts: list[Attempt] = []
    routes = best_available_routes(role, limit=limit)
    pinned_provider = os.environ.get("COMPANYBRAIN_PIN_PROVIDER", "").strip()
    if pinned_provider:
        routes = [route for route in routes if route[0] == pinned_provider]
        if not routes:
            raise ProviderError(
                f"COMPANYBRAIN_PIN_PROVIDER={pinned_provider!r} yielded no usable Best Available route; "
                "refusing to route to a different provider.",
                retryable=True,
            )
    wait_budget = _cooldown_wait_budget(max_cooldown_wait_seconds)
    waited = 0.0

    while True:
        pass_waits: list[float] = []

        for provider_name, model in routes:
            is_agent_cli = PROVIDERS.get(provider_name) is not None and PROVIDERS[provider_name].kind == "agent_cli"
            # agent_cli providers pick their own model, so never send a model_override -
            # but the registry's static model label ("auto", "codex-cli", ...) is not
            # what actually runs, so log the resolved argv model instead (see
            # resolve_agent_cli_model_label).
            logged_model = resolve_agent_cli_model_label(provider_name) if is_agent_cli else model
            try:
                model_override = (
                    model
                    if not is_agent_cli and provider_name in {"NVIDIA NIM", "OpenAI API", "Kimi / Moonshot", "Anthropic API"}
                    else None
                )
                content = call_provider(provider_name, role, prompt, model_override=model_override)
                attempts.append(Attempt(provider_name, logged_model, "ok", ""))
                return ResilientResult(content, provider_name, logged_model, attempts)
            except ProviderError as exc:
                attempts.append(Attempt(provider_name, logged_model, _error_status(exc), _safe_error(exc)))
                if _is_waitable_cooldown(exc):
                    pass_waits.append(float(exc.retry_after or 0))
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(Attempt(provider_name, logged_model, "error", str(exc)[:300]))
                continue

        wait_seconds = _next_cooldown_wait(pass_waits, wait_budget - waited)
        if wait_seconds > 0:
            attempts.append(Attempt("Router", "cooldown", "waiting", f"Sleeping {int(wait_seconds)}s until a cooled-down route can be retried."))
            time.sleep(wait_seconds)
            waited += wait_seconds
            continue

        summary = "\n".join(f"- {item.provider} {item.model}: {item.status} {item.detail}".strip() for item in attempts)
        raise ProviderError(f"No best-available route succeeded.\n{summary}", retryable=True)


def call_tier_with_fallback(
    role: str,
    prompt: str,
    *,
    tier: str,
    local_timeout_seconds: int = 120,
    cloud_request_timeout_seconds: int | None = None,
    cloud_max_retries: int | None = None,
    max_tokens: int | None = None,
    max_cooldown_wait_seconds: int | None = None,
    fallback_to_best: bool = False,
) -> ResilientResult:
    """Route a call through registry routes for one tier, in priority order.

    Tier calls stay inside the registry by default. Best Available is only
    used when the caller asks for it and the registry explicitly marks an
    enabled route in that tier as fallback_allowed.
    """
    from model_registry import best_available_fallback_allowed, tier_routes  # local import to avoid module cycles

    attempts: list[Attempt] = []
    routes = list(tier_routes(tier))
    pinned_provider = os.environ.get("COMPANYBRAIN_PIN_PROVIDER", "").strip()

    for provider_name, model in routes:
        is_agent_cli = PROVIDERS.get(provider_name) is not None and PROVIDERS[provider_name].kind == "agent_cli"
        # The registry's static model_id ("auto", "codex-cli", ...) is a label, not
        # a confirmed value, for agent_cli providers - resolve what actually runs.
        logged_model = resolve_agent_cli_model_label(provider_name) if is_agent_cli else model
        try:
            model_override = model if provider_name in {"NVIDIA NIM", "OpenAI API", "Kimi / Moonshot", "Anthropic API"} else None
            content = call_provider(
                provider_name,
                role,
                prompt,
                model_override=model_override,
                request_timeout=cloud_request_timeout_seconds,
                max_retries_override=cloud_max_retries,
                max_tokens=max_tokens,
            )
            attempts.append(Attempt(provider_name, logged_model, "ok", f"registry tier: {tier}"))
            return ResilientResult(content, provider_name, logged_model, attempts)
        except ProviderError as exc:
            attempts.append(Attempt(provider_name, logged_model, _error_status(exc), _safe_error(exc)))
            continue
        except Exception as exc:  # noqa: BLE001
            attempts.append(Attempt(provider_name, logged_model, "error", str(exc)[:300]))
            continue

    if pinned_provider:
        summary = "\n".join(f"- {item.provider} {item.model}: {item.status} {item.detail}".strip() for item in attempts)
        raise ProviderError(
            f"COMPANYBRAIN_PIN_PROVIDER={pinned_provider!r} yielded no successful {tier} route; "
            "refusing Best Available fallback or any different provider."
            + (f"\n{summary}" if summary else ""),
            retryable=True,
        )

    if fallback_to_best and best_available_fallback_allowed(tier):
        routed = call_best_available_with_fallback(
            role,
            prompt,
            max_cooldown_wait_seconds=max_cooldown_wait_seconds,
        )
        return ResilientResult(routed.content, routed.provider, routed.model, [*attempts, *routed.attempts])

    summary = "\n".join(f"- {item.provider} {item.model}: {item.status} {item.detail}".strip() for item in attempts)
    fallback_note = "Best Available fallback was not allowed for this tier."
    if fallback_to_best:
        fallback_note = "Best Available fallback was requested but no enabled registry entry has fallback_allowed=true."
    raise ProviderError(f"No {tier} registry route succeeded. {fallback_note}\n{summary}", retryable=True)


def format_attempts(attempts: list[Attempt]) -> str:
    if not attempts:
        return "No attempts recorded."
    lines = ["Route attempts:"]
    for attempt in attempts:
        label = f"{attempt.provider} {attempt.model}".strip()
        detail = f" - {attempt.detail}" if attempt.detail else ""
        lines.append(f"- {label}: {attempt.status}{detail}")
    return "\n".join(lines)


def _error_status(error: ProviderError) -> str:
    if error.is_rate_limit:
        return "rate_limited"
    if error.is_unavailable:
        return "unavailable"
    if error.retryable:
        return "retryable_error"
    return "failed"


def _safe_error(error: ProviderError) -> str:
    if error.is_rate_limit:
        wait = f" retry after {int(error.retry_after)}s" if error.retry_after else ""
        return f"HTTP 429{wait}"
    if error.status_code:
        return f"HTTP {error.status_code}"
    return str(error)[:300]


def _is_waitable_cooldown(error: ProviderError) -> bool:
    if error.quota_exhausted or error.is_unavailable:
        return False
    if not error.retryable:
        return False
    return bool(error.retry_after and error.retry_after > 0)


def _cooldown_wait_budget(value: int | None) -> float:
    if value is not None:
        return max(0.0, float(value))
    raw = os.environ.get("BEST_AVAILABLE_MAX_COOLDOWN_WAIT_SECONDS", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


def _next_cooldown_wait(wait_candidates: list[float], remaining_budget: float) -> float:
    waits = [wait for wait in wait_candidates if wait > 0]
    if not waits or remaining_budget <= 0:
        return 0.0
    return max(0.0, min(min(waits) + 1.0, remaining_budget))
