"""Model routing and per-run cost accounting.

`call_tier_with_fallback` / `call_best_available_with_fallback` / `_engineer_primary_tier`
are read through this module by every caller (`models.call_tier_with_fallback(...)`)
so a caller that rebinds them for a dry test is actually observed.

A pinned provider with no usable route raises. It never silently falls through to a
different tier: an explicit pin is a safety statement, not a preference.
"""

from __future__ import annotations

import os
from datetime import datetime

import api_clients
from resilient_calls import (
    call_best_available_with_fallback as _raw_call_best_available_with_fallback,
    call_tier_with_fallback as _raw_call_tier_with_fallback,
)


# Token counts are ESTIMATED from character length because the provider clients
# return content only, not usage headers. Character counts are exact; treat the
# token and dollar columns as consistent-but-approximate.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "Anthropic API": (3.0, 15.0),
    "OpenAI API": (2.5, 10.0),
    "Gemini": (0.0, 0.0),
    "NVIDIA NIM": (0.0, 0.0),
    "Kimi / Moonshot": (0.0, 0.0),
}

# Subscription-backed routes have no marginal per-token cost, so dollars are the
# wrong unit. Count their calls instead - that is what burns the rate limit.
SUBSCRIPTION_PROVIDERS = {"Claude Code CLI", "Codex CLI", "Cursor Agent CLI"}
LEAD_PROVIDERS = {"Anthropic API", "OpenAI API", "Claude Code CLI", "Codex CLI", "Cursor Agent CLI"}

_MODEL_CALL_LEDGER: list[dict] = []


def _engineer_primary_tier() -> str:
    """Tier that carries Engineer reasoning: plan, replan, repair, patch.

    Under the current thesis the harness wraps a frontier LEAD model and uses free
    worker routes only for cheap subtasks. So lead is the primary path and the free
    coding_elite pool is a fallback, not the default. Hardcoding coding_elite was a
    leftover from the earlier cheap-fleet thesis and silently fails closed whenever
    the free routes are unavailable.
    """
    from model_registry import tier_routes  # local import to avoid a module cycle

    pinned_provider = os.environ.get("COMPANYBRAIN_PIN_PROVIDER", "").strip()
    try:
        if tier_routes("lead"):
            return "lead"
    except Exception as exc:  # noqa: BLE001 - an explicit pin must fail closed
        if pinned_provider:
            raise RuntimeError(_pin_failure_message(pinned_provider)) from exc
    if pinned_provider:
        raise RuntimeError(_pin_failure_message(pinned_provider))
    return "coding_elite"


def _pin_failure_message(pinned_provider: str) -> str:
    """A pin failure must fail closed either way; this only makes the closed
    failure diagnosable in one read instead of requiring a manual registry trace.
    """
    from model_registry import stale_unroutable_entries  # local import to avoid a module cycle

    base = (
        f"COMPANYBRAIN_PIN_PROVIDER={pinned_provider!r} yielded no usable lead routes; "
        "refusing to fall back to a different tier or provider."
    )
    stale = stale_unroutable_entries(pinned_provider)
    if not stale:
        return base
    probe_model_id = (stale[0].get('model_id') or '<model_id>') if len(stale) == 1 else '<model_id>'
    rows = "; ".join(
        f"{item.get('model_id')} (last checked {item.get('last_checked') or 'never'})" for item in stale
    )
    return (
        base
        + f" Likely stale, not a live failure: {rows}. Re-probe before assuming the route is broken: "
        + f'python scripts/company_brain_action.py model-probe --provider "{pinned_provider}" '
        + f"--model-id {probe_model_id} --enable-if-ok"
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _estimate_cost_usd(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    if provider in SUBSCRIPTION_PROVIDERS:
        return 0.0
    prompt_rate, completion_rate = MODEL_PRICING_USD_PER_MTOK.get(provider, (0.0, 0.0))
    return round((prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000, 6)


def _ledger_record(role: str, prompt: str, routed: object, seconds: float, tier: str = "") -> None:
    provider = str(getattr(routed, "provider", "") or "")
    content = str(getattr(routed, "content", "") or "")

    # Prefer exact usage when the provider reported it (agent CLIs do); fall back to
    # the character estimate otherwise. Recording which one was used keeps the cost
    # column honest instead of silently mixing measured and guessed numbers.
    usage = dict(getattr(api_clients, "LAST_CALL_USAGE", {}) or {})
    if usage.get("exact"):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        token_source = "provider_reported"
    else:
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        token_source = "estimated_from_chars"

    reported_cost = usage.get("reported_cost_usd")
    cost = (
        float(reported_cost)
        if isinstance(reported_cost, (int, float)) and provider not in SUBSCRIPTION_PROVIDERS
        else _estimate_cost_usd(provider, prompt_tokens, completion_tokens)
    )

    _MODEL_CALL_LEDGER.append(
        {
            "role": role,
            "tier": tier,
            "provider": provider,
            "model": str(getattr(routed, "model", "") or ""),
            "prompt_chars": len(prompt or ""),
            "completion_chars": len(content),
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "token_source": token_source,
            "cached_tokens": int(usage.get("cached_tokens") or 0),
            "cost_usd_est": round(cost, 6),
            "seconds": round(seconds, 2),
            "attempts": len(getattr(routed, "attempts", []) or []),
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def ledger_mark() -> int:
    """Return a cursor so a caller can attribute only its own model calls."""
    return len(_MODEL_CALL_LEDGER)


def ledger_since(mark: int) -> dict:
    entries = _MODEL_CALL_LEDGER[mark:]
    return {
        "estimated": True,
        "basis": "characters//4; providers return content only, not usage headers",
        "model_calls": len(entries),
        "prompt_tokens_est": sum(e["prompt_tokens_est"] for e in entries),
        "completion_tokens_est": sum(e["completion_tokens_est"] for e in entries),
        "cost_usd_est": round(sum(e["cost_usd_est"] for e in entries), 6),
        "subscription_calls": sum(1 for e in entries if e["provider"] in SUBSCRIPTION_PROVIDERS),
        "seconds": round(sum(e["seconds"] for e in entries), 2),
        "lead_calls": sum(1 for e in entries if e["provider"] in LEAD_PROVIDERS),
        "worker_calls": sum(1 for e in entries if e["provider"] not in LEAD_PROVIDERS),
        "calls": entries,
    }


def call_tier_with_fallback(role: str, prompt: str, **kwargs: object):
    began = datetime.now()
    routed = _raw_call_tier_with_fallback(role, prompt, **kwargs)  # type: ignore[arg-type]
    _ledger_record(role, prompt, routed, (datetime.now() - began).total_seconds(), str(kwargs.get("tier") or ""))
    return routed


def call_best_available_with_fallback(role: str, prompt: str, **kwargs: object):
    began = datetime.now()
    routed = _raw_call_best_available_with_fallback(role, prompt, **kwargs)  # type: ignore[arg-type]
    _ledger_record(role, prompt, routed, (datetime.now() - began).total_seconds(), "best_available")
    return routed


def attempts_of(routed: object) -> list[dict]:
    return [item.__dict__ for item in getattr(routed, "attempts", []) or []]


def route_of(routed: object) -> str:
    return f"{getattr(routed, 'provider', '')} {getattr(routed, 'model', '')}".strip()
