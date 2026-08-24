from __future__ import annotations

import json
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "model_rate_limits.json"
STATE_PATH = ROOT / "brain" / "model_health" / "provider_rate_limit_state.json"
LEGACY_STATE_PATH = ROOT / "brain" / "model_health" / "nvidia_model_health.json"

# If a slot is granted but record_success/record_failure never fires within this
# window, treat the previous acquisition as an unresolved failure on next acquire.
PENDING_SLOT_TIMEOUT_SECONDS = 120


DEFAULT_CONFIG: dict[str, Any] = {
    "providers": {
        "NVIDIA NIM": {
            "requests_per_minute": 12,
            "min_seconds_between_provider_calls": 2.5,
            "min_seconds_between_model_calls": 15,
            "provider_rate_limit_cooldown_seconds": 120,
            "rate_limit_cooldown_seconds": 120,
            "unavailable_cooldown_seconds": 86400,
            "transient_error_cooldown_seconds": 300,
        },
        "Gemini": {
            "requests_per_minute": 5,
            "min_seconds_between_provider_calls": 12,
            "min_seconds_between_model_calls": 12,
            "provider_rate_limit_cooldown_seconds": 60,
            "rate_limit_cooldown_seconds": 60,
            "quota_exhausted_cooldown_seconds": 86400,
            "unavailable_cooldown_seconds": 3600,
            "transient_error_cooldown_seconds": 300,
        },
        "Kimi / Moonshot": {
            "requests_per_minute": 3,
            "min_seconds_between_provider_calls": 20,
            "min_seconds_between_model_calls": 20,
            "provider_rate_limit_cooldown_seconds": 60,
            "rate_limit_cooldown_seconds": 60,
            "quota_exhausted_cooldown_seconds": 86400,
            "unavailable_cooldown_seconds": 3600,
            "transient_error_cooldown_seconds": 300,
        }
    },
    "models": {},
}


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_seconds(headers: dict[str, str]) -> float | None:
    value = headers.get("retry-after") or headers.get("x-ratelimit-reset") or headers.get("x-rate-limit-reset")
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        parsed = parsedate_to_datetime(value)
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def observed_rate_limits(headers: dict[str, str]) -> dict[str, Any]:
    rate_headers = _safe_rate_limit_headers(headers)
    observed: dict[str, Any] = {"headers": rate_headers}
    retry_after = retry_after_seconds(rate_headers)
    if retry_after is not None:
        observed["retry_after_seconds"] = retry_after

    for key, value in rate_headers.items():
        parsed = _parse_number(value)
        if parsed is None:
            continue
        lower = key.lower()
        if "remaining" in lower:
            observed.setdefault("remaining", {})[key] = parsed
        elif "limit" in lower:
            observed.setdefault("limits", {})[key] = parsed
        elif "reset" in lower:
            observed.setdefault("resets", {})[key] = parsed
    return observed


def is_model_usable(provider: str, model: str) -> bool:
    state = _load_state()
    provider_entry = _provider_state(state, provider)
    provider_cooldown_until = float(provider_entry.get("cooldown_until", 0) or 0)
    if provider_cooldown_until > time.time():
        return False
    entry = _model_state(state, provider, model)
    cooldown_until = float(entry.get("cooldown_until", 0) or 0)
    return cooldown_until <= time.time()


def provider_status(provider: str) -> dict[str, Any]:
    state = _load_state()
    entry = dict(_provider_state(state, provider))
    cooldown_until = float(entry.get("cooldown_until", 0) or 0)
    entry["cooldown_remaining_seconds"] = max(0, int(cooldown_until - time.time()))
    return entry


def model_status(provider: str, model: str) -> dict[str, Any]:
    state = _load_state()
    entry = dict(_model_state(state, provider, model))
    cooldown_until = float(entry.get("cooldown_until", 0) or 0)
    entry["cooldown_remaining_seconds"] = max(0, int(cooldown_until - time.time()))
    return entry


def acquire_slot(provider: str, model: str) -> None:
    config = _provider_config(provider, model)
    state = _load_state()
    now = time.time()

    provider_entry = _provider_state(state, provider)
    provider_cooldown_until = float(provider_entry.get("cooldown_until", 0) or 0)
    if provider_cooldown_until > now:
        raise RateLimitError(
            f"{provider} is cooling down for {int(provider_cooldown_until - now)} more seconds after a prior provider-level rate limit.",
            retry_after=provider_cooldown_until - now,
        )

    entry = _model_state(state, provider, model)
    cooldown_until = float(entry.get("cooldown_until", 0) or 0)
    if cooldown_until > now:
        raise RateLimitError(
            f"{model} is cooling down for {int(cooldown_until - now)} more seconds after a prior rate-limit/unavailable response.",
            retry_after=cooldown_until - now,
        )

    pending_since = float(entry.get("pending_since", 0) or 0)
    if pending_since and (now - pending_since) > PENDING_SLOT_TIMEOUT_SECONDS:
        cooldown = float(config.get("transient_error_cooldown_seconds", 300))
        entry["status"] = "transient_error"
        entry["cooldown_until"] = now + cooldown
        entry["last_error_at"] = datetime.now().isoformat(timespec="seconds")
        entry["last_error"] = (
            "Unresolved outcome: previous acquire_slot call was never followed "
            "by record_success/record_failure."
        )
        entry.pop("pending_since", None)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _save_state(state)
        raise RateLimitError(
            f"{model} had an unresolved prior request outcome; cooling down for {int(cooldown)} seconds before retrying.",
            retry_after=cooldown,
        )

    provider_last = float(provider_entry.get("last_call_at", 0) or 0)
    model_last = float(entry.get("last_call_at", 0) or 0)

    requests_per_minute = float(config.get("requests_per_minute", 0) or 0)
    rpm_interval = 60 / requests_per_minute if requests_per_minute > 0 else 0

    wait_seconds = max(
        max(float(config.get("min_seconds_between_provider_calls", 0)), rpm_interval) - (now - provider_last),
        float(config.get("min_seconds_between_model_calls", 0)) - (now - model_last),
        0,
    )
    if wait_seconds > 0:
        time.sleep(min(wait_seconds, 30))

    now = time.time()
    provider_entry["last_call_at"] = now
    entry["last_call_at"] = now
    entry["pending_since"] = now
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state)


def record_success(provider: str, model: str, headers: dict[str, str] | None = None) -> None:
    state = _load_state()
    safe_headers = _safe_rate_limit_headers(headers or {})
    observed = observed_rate_limits(headers or {})
    provider_entry = _provider_state(state, provider)
    provider_entry["status"] = "healthy"
    provider_entry["last_success_at"] = datetime.now().isoformat(timespec="seconds")
    provider_entry["last_headers"] = safe_headers
    provider_entry["observed_rate_limits"] = observed
    entry = _model_state(state, provider, model)
    entry["status"] = "healthy"
    entry["last_success_at"] = datetime.now().isoformat(timespec="seconds")
    entry["cooldown_until"] = 0
    entry["successes"] = int(entry.get("successes", 0)) + 1
    entry["last_headers"] = safe_headers
    entry["observed_rate_limits"] = observed
    entry.pop("pending_since", None)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state)


def record_failure(provider: str, model: str, error: Any) -> None:
    config = _provider_config(provider, model)
    state = _load_state()
    entry = _model_state(state, provider, model)

    status_code = getattr(error, "status_code", None)
    retry_after = getattr(error, "retry_after", None)
    error_headers = getattr(error, "headers", {}) or {}
    safe_headers = _safe_rate_limit_headers(error_headers)
    observed = observed_rate_limits(error_headers)
    now = time.time()

    if getattr(error, "quota_exhausted", False):
        status = "quota_exhausted"
        cooldown = float(config.get("quota_exhausted_cooldown_seconds", 86400))
        provider_entry = _provider_state(state, provider)
        provider_entry["status"] = "quota_exhausted"
        provider_entry["cooldown_until"] = now + cooldown
        provider_entry["last_quota_error_at"] = datetime.now().isoformat(timespec="seconds")
        provider_entry["quota_exhausted_events"] = int(provider_entry.get("quota_exhausted_events", 0)) + 1
        provider_entry["last_headers"] = safe_headers
        provider_entry["observed_rate_limits"] = observed
    elif status_code == 429:
        status = "rate_limited"
        cooldown = retry_after or float(config.get("rate_limit_cooldown_seconds", 120))
        provider_cooldown = retry_after or float(config.get("provider_rate_limit_cooldown_seconds", 120))
        provider_entry = _provider_state(state, provider)
        provider_entry["status"] = "rate_limited"
        provider_entry["cooldown_until"] = now + provider_cooldown
        provider_entry["last_rate_limit_at"] = datetime.now().isoformat(timespec="seconds")
        provider_entry["rate_limit_events"] = int(provider_entry.get("rate_limit_events", 0)) + 1
        provider_entry["last_headers"] = safe_headers
        provider_entry["observed_rate_limits"] = observed
        entry["rate_limit_events"] = int(entry.get("rate_limit_events", 0)) + 1
    elif status_code in {400, 404, 410, 422}:
        status = "unavailable"
        cooldown = float(config.get("unavailable_cooldown_seconds", 86400))
        entry["unavailable_events"] = int(entry.get("unavailable_events", 0)) + 1
    elif status_code in {408, 409, 500, 502, 503, 504} or getattr(error, "retryable", False):
        status = "transient_error"
        cooldown = retry_after or float(config.get("transient_error_cooldown_seconds", 300))
    else:
        status = "error"
        cooldown = float(config.get("transient_error_cooldown_seconds", 300))

    entry["status"] = status
    entry["cooldown_until"] = now + cooldown
    entry["last_error_at"] = datetime.now().isoformat(timespec="seconds")
    entry["last_error"] = str(error)[:600]
    entry["last_status_code"] = status_code
    entry["failures"] = int(entry.get("failures", 0)) + 1
    entry["last_headers"] = safe_headers
    entry["observed_rate_limits"] = observed
    entry.pop("pending_since", None)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state)


def _provider_config(provider: str, model: str) -> dict[str, Any]:
    config = _load_config()
    provider_config = dict(DEFAULT_CONFIG["providers"].get(provider, {}))
    provider_config.update(config.get("providers", {}).get(provider, {}))
    provider_config.update(config.get("models", {}).get(model, {}))
    return provider_config


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return DEFAULT_CONFIG
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_CONFIG


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        if LEGACY_STATE_PATH.exists():
            try:
                return json.loads(LEGACY_STATE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"updated_at": None, "providers": {}, "models": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"updated_at": None, "providers": {}, "models": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _model_state(state: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    key = f"{provider}:{model}"
    return state.setdefault("models", {}).setdefault(key, {"status": "unknown", "cooldown_until": 0})


def _provider_state(state: dict[str, Any], provider: str) -> dict[str, Any]:
    return state.setdefault("providers", {}).setdefault(provider, {"status": "unknown", "cooldown_until": 0})


def _safe_rate_limit_headers(headers: dict[str, str]) -> dict[str, str]:
    allowed = {}
    for key, value in headers.items():
        lower = key.lower()
        if "limit" in lower or "remaining" in lower or "retry-after" in lower or "reset" in lower:
            allowed[lower] = value
    return allowed


def _parse_number(value: str) -> int | float | None:
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number
