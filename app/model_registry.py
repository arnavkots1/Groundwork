"""Tiered model registry for the Engineer Employee harness.

The registry is a human-editable JSON file (config/model_registry.json) that
assigns every known route to a tier:

- lead: the frontier model that plans, synthesizes, and writes patches.
- utility_light: summaries, extraction, formatting. Never routed to elite models.
- coding_elite: engineer-plan, engineer-patch, patch repair (NVIDIA free workers).
- reasoning_elite: architecture decisions, multi-file reasoning, hard debugging.
- disabled_candidate: tracked but never routed (e.g. unverified GLM-5.2).

Refresh is deterministic and offline: it checks synced provider catalogs and
env-key presence, never makes network calls, and never reads secret values.
Live model calls are handled by the explicit model-probe CLI action.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from api_clients import PROVIDERS, agent_cli_available, disabled_providers

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "model_registry.json"
NVIDIA_AVAILABLE_PATH = ROOT / "config" / "nvidia_available_models.json"
PROVIDER_HEALTH_PATH = ROOT / "brain" / "model_health" / "provider_rate_limit_state.json"

VALID_TIERS = {"lead", "utility_light", "coding_elite", "reasoning_elite", "disabled_candidate"}
VALID_TRI_STATE = {True, False, "unknown"}
VALID_HEALTH = {"unchecked", "ok", "unavailable", "error"}
UNROUTABLE_HEALTH = {"unavailable", "error"}
TIER_CAPABILITY_HINTS = {
    "lead": {"reasoning", "architecture", "coding", "agentic"},
    "utility_light": {"summarization", "extraction", "formatting"},
    "coding_elite": {"coding", "reasoning", "agentic"},
    "reasoning_elite": {"reasoning", "architecture", "coding", "agentic"},
}


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"updated_at": None, "models": []}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"updated_at": None, "models": [], "load_error": "model_registry.json is not valid JSON"}
    if not isinstance(data.get("models"), list):
        data["models"] = []
    for item in data["models"]:
        if isinstance(item, dict):
            _normalize_entry(item)
    return data


def registry_entries(tier: str | None = None, include_disabled: bool = False) -> list[dict[str, Any]]:
    entries = []
    for item in load_registry()["models"]:
        if not isinstance(item, dict):
            continue
        entry_tiers = {str(item.get("tier", "")), *[str(value) for value in item.get("additional_tiers", [])]}
        if tier and tier not in entry_tiers:
            continue
        if not include_disabled and not item.get("enabled", False):
            continue
        entries.append(item)
    entries.sort(key=lambda item: (item.get("priority", 999), str(item.get("model_id", ""))))
    return entries


def tier_routes(tier: str) -> list[tuple[str, str]]:
    """Return (provider, model_id) routes for a tier, healthiest-first.

    Entries whose deterministic health check failed are excluded; entries that
    have never been checked ("unverified") are kept so the registry works
    before the first refresh.
    """
    # Route pinning for reproducible experiments. A benchmark that lets one arm drift
    # to a different provider than the other is measuring the model, not the harness.
    # When this is set, only routes from the named provider are returned; an empty
    # result is correct and must fail visibly rather than silently fall back.
    pinned = os.environ.get("COMPANYBRAIN_PIN_PROVIDER", "").strip()

    routes = []
    for item in registry_entries(tier):
        if item.get("tier") == "disabled_candidate":
            continue
        if pinned and str(item.get("provider", "")) != pinned:
            continue
        if item.get("health_status") in UNROUTABLE_HEALTH:
            continue
        if item.get("local_catalog_present") is False and item.get("health_status") != "ok":
            continue
        if item.get("account_available") is False:
            continue
        provider = str(item.get("provider", ""))
        model_id = str(item.get("model_id", ""))
        if not (provider and model_id and provider in PROVIDERS):
            continue
        # agent_cli routes are only usable when their CLI is on PATH. Gate live so
        # routing is correct even before a registry refresh writes account_available.
        if PROVIDERS[provider].kind == "agent_cli" and not agent_cli_available(provider):
            continue
        routes.append((provider, model_id))
    return routes


def stale_unroutable_entries(provider: str, stale_after_hours: int = 24) -> list[dict[str, Any]]:
    """Entries for `provider` that block routing but whose health check is old
    enough to no longer be trustworthy.

    Discovered 2026-08-18: a pin can fail closed on a route that is actually
    fine right now, purely because nothing re-checked it in days. That failure
    is correct (never silently reroute) but was undiagnosable without manually
    reading model_registry.json - this makes the stale-vs-live distinction part
    of the error message itself instead of something a human has to trace by hand.
    """
    now = datetime.now()
    stale: list[dict[str, Any]] = []
    for item in registry_entries(None, include_disabled=True):
        if str(item.get("provider", "")) != provider:
            continue
        if item.get("health_status") not in UNROUTABLE_HEALTH:
            continue
        last_checked = item.get("last_checked")
        is_stale = True
        if last_checked:
            try:
                checked_at = datetime.fromisoformat(str(last_checked))
                is_stale = (now - checked_at).total_seconds() > stale_after_hours * 3600
            except ValueError:
                is_stale = True
        if is_stale:
            stale.append(item)
    return stale


def best_available_fallback_allowed(tier: str) -> bool:
    """Best Available is opt-in per registry entry, not an implicit escape hatch."""
    return any(
        item.get("fallback_allowed") is True
        for item in registry_entries(tier, include_disabled=True)
        if item.get("enabled", False)
    )


def _nvidia_catalog_ids() -> set[str]:
    if not NVIDIA_AVAILABLE_PATH.exists():
        return set()
    try:
        data = json.loads(NVIDIA_AVAILABLE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    models = data.get("models", [])
    ids = set()
    for item in models:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and item.get("id"):
            ids.add(str(item["id"]))
    return ids


def _terminal_provider_observation(provider: str, model_id: str) -> dict[str, Any] | None:
    if not PROVIDER_HEALTH_PATH.exists():
        return None
    try:
        data = json.loads(PROVIDER_HEALTH_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    observed = (data.get("models") or {}).get(f"{provider}:{model_id}")
    if not isinstance(observed, dict):
        return None
    status_code = observed.get("last_status_code")
    if status_code not in {400, 404, 410, 422}:
        return None
    if str(observed.get("last_success_at") or "") > str(observed.get("last_error_at") or ""):
        return None
    return observed


def _provider_configured(provider_name: str) -> bool:
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        return False
    if provider.name in disabled_providers():
        return False
    if provider.kind == "agent_cli":
        # Subscription-backed CLIs authenticate through their own login, not an
        # env key. Availability is whether the CLI binary resolves on PATH.
        return agent_cli_available(provider_name)
    if provider.env_key:
        return bool(os.environ.get(provider.env_key))
    return True


def _normalize_entry(item: dict[str, Any]) -> None:
    item.setdefault("official_listed", "unknown")
    item.setdefault("local_catalog_present", "unknown")
    item.setdefault("account_available", "unknown")
    item.setdefault("health_status", "unchecked")
    item.setdefault("source_notes", item.get("notes", ""))
    item.setdefault("last_checked", None)
    if item.get("health_status") not in VALID_HEALTH:
        previous = str(item.get("health_status") or "")
        if previous in {"catalog_listed", "provider_configured"}:
            item["health_status"] = "unchecked"
        elif previous in {"missing_from_catalog", "provider_unconfigured"}:
            item["health_status"] = "unavailable"
        else:
            item["health_status"] = "unchecked"
    for key in ("official_listed", "local_catalog_present", "account_available"):
        if item.get(key) not in VALID_TRI_STATE:
            item[key] = "unknown"


def _catalog_state(provider: str, model_id: str, nvidia_ids: set[str]) -> bool | str:
    if provider == "NVIDIA NIM":
        return model_id in nvidia_ids if nvidia_ids else "unknown"
    return "unknown"


def _account_state(provider: str) -> bool | str:
    if provider not in PROVIDERS:
        return False
    if not _provider_configured(provider):
        return False
    # Configured credentials only prove that a probe can be attempted. Cloud
    # account/model access remains unknown until a direct health probe succeeds.
    return "unknown"


def refresh_registry() -> dict[str, Any]:
    """Deterministically re-check every registry entry against local catalogs.

    Offline by design: no network calls, no secret values read beyond
    env-key presence. Disabled candidates get health updates but stay disabled.
    """
    data = load_registry()
    nvidia_ids = _nvidia_catalog_ids()
    now = datetime.now().isoformat(timespec="seconds")
    changes = []

    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider", ""))
        model_id = str(item.get("model_id", ""))
        _normalize_entry(item)
        previous = {
            "local_catalog_present": item.get("local_catalog_present"),
            "account_available": item.get("account_available"),
            "health_status": item.get("health_status"),
        }
        item["local_catalog_present"] = _catalog_state(provider, model_id, nvidia_ids)
        configured_account_state = _account_state(provider)
        if configured_account_state is False:
            item["account_available"] = False
        elif not item.get("last_probe_at"):
            item["account_available"] = configured_account_state
        terminal_observation = _terminal_provider_observation(provider, model_id)
        if terminal_observation:
            item["health_status"] = "unavailable"
            item["account_available"] = False
            item["last_health_error"] = str(terminal_observation.get("last_error") or "")[:900]
            item["last_health_error_at"] = terminal_observation.get("last_error_at")
        elif item["account_available"] is False:
            item["health_status"] = "unavailable"
        elif item["local_catalog_present"] is False and item.get("enabled", False):
            item["health_status"] = "unavailable"
        elif item.get("health_status") not in VALID_HEALTH:
            item["health_status"] = "unchecked"
        item["last_checked"] = now
        current = {
            "local_catalog_present": item.get("local_catalog_present"),
            "account_available": item.get("account_available"),
            "health_status": item.get("health_status"),
        }
        if previous != current:
            changes.append({"provider": provider, "model_id": model_id, "from": previous, "to": current})

    data["updated_at"] = now
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    active = [
        item
        for item in data["models"]
        if isinstance(item, dict)
        and item.get("enabled")
        and item.get("health_status") not in UNROUTABLE_HEALTH
        and (item.get("local_catalog_present") is not False or item.get("health_status") == "ok")
        and item.get("account_available") is not False
    ]
    candidates = [
        item
        for item in data["models"]
        if isinstance(item, dict)
        and not item.get("enabled", False)
        and item.get("health_status") in {"unchecked", "unavailable"}
        and item.get("official_listed") is True
    ]
    unavailable = [
        item
        for item in data["models"]
        if isinstance(item, dict) and item.get("health_status") in {"unavailable", "error"}
    ]
    stale_or_missing = [
        item
        for item in data["models"]
        if isinstance(item, dict) and item.get("local_catalog_present") in {False, "unknown"}
    ]
    return {
        "updated_at": now,
        "checked": len(data["models"]),
        "changes": changes,
        "verified_active_models": _compact_entries(active),
        "candidates_needing_probe": _compact_entries(candidates),
        "unavailable_models": _compact_entries(unavailable),
        "stale_or_missing_catalog_models": _compact_entries(stale_or_missing),
        "nvidia_catalog_models": len(nvidia_ids),
        "path": str(REGISTRY_PATH),
    }


def record_probe_result(
    provider: str,
    model_id: str,
    *,
    ok: bool,
    health_status: str,
    error_type: str,
    error_reason: str = "",
    enable_if_ok: bool = False,
) -> dict[str, Any]:
    data = load_registry()
    now = datetime.now().isoformat(timespec="seconds")
    target = None
    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        if item.get("provider") == provider and item.get("model_id") == model_id:
            target = item
            break
    if target is None:
        target = {
            "provider": provider,
            "model_id": model_id,
            "tier": "disabled_candidate",
            "capabilities": [],
            "enabled": False,
            "priority": 999,
        }
        data["models"].append(target)
    _normalize_entry(target)
    was_enabled = bool(target.get("enabled", False))
    previous_tier = str(target.get("tier") or "disabled_candidate")
    previous_additional_tiers = list(target.get("additional_tiers", []))
    target["health_status"] = health_status if health_status in VALID_HEALTH else "error"
    unavailable_errors = {"missing_api_key", "unavailable", "quota_exhausted", "unknown_provider"}
    target["account_available"] = (
        True
        if ok
        else False
        if error_type in unavailable_errors or (error_type.startswith("http_4") and error_type != "http_429")
        else "unknown"
    )
    target["last_checked"] = now
    target["last_probe_at"] = now
    if error_type:
        target["last_probe_error_type"] = error_type
    else:
        target.pop("last_probe_error_type", None)
    if error_reason:
        target["last_probe_error"] = error_reason
    else:
        target.pop("last_probe_error", None)
    if ok and was_enabled:
        target["enabled"] = True
        target["tier"] = previous_tier
        target["additional_tiers"] = previous_additional_tiers
        target["notes"] = "Direct account probe succeeded; existing routing policy remains enabled."
    elif ok and enable_if_ok:
        candidate_tiers = [str(value) for value in target.get("candidate_tiers", []) if str(value)]
        target["tier"] = candidate_tiers[0] if candidate_tiers else "coding_elite"
        target["additional_tiers"] = candidate_tiers[1:] if len(candidate_tiers) > 1 else []
        target["enabled"] = True
        target["notes"] = "Direct account probe succeeded; model explicitly enabled for elite Engineer routing."
    elif ok:
        target["enabled"] = False
        target["tier"] = "disabled_candidate"
        target["notes"] = "Direct account probe succeeded; model remains disabled until --enable-if-ok is passed."
    else:
        detail = f" ({error_reason})" if error_reason else ""
        if was_enabled:
            target["enabled"] = True
            target["tier"] = previous_tier
            target["additional_tiers"] = previous_additional_tiers
            target["notes"] = f"Direct account probe failed: {error_type}{detail}. Policy remains enabled, but health makes this route unroutable."
        else:
            target["enabled"] = False
            target["tier"] = "disabled_candidate"
            target["notes"] = f"Direct account probe failed: {error_type}{detail}. Model remains disabled_candidate."
    data["updated_at"] = now
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def registry_summary() -> dict[str, Any]:
    data = load_registry()
    tiers: dict[str, list[dict[str, Any]]] = {}
    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        tiers.setdefault(str(item.get("tier", "unknown")), []).append(
            {
                "provider": item.get("provider"),
                "model_id": item.get("model_id"),
                "tier": item.get("tier"),
                "additional_tiers": item.get("additional_tiers", []),
                "candidate_tiers": item.get("candidate_tiers", []),
                "enabled": item.get("enabled", False),
                "priority": item.get("priority"),
                "health_status": item.get("health_status"),
                "official_listed": item.get("official_listed", "unknown"),
                "local_catalog_present": item.get("local_catalog_present", "unknown"),
                "account_available": item.get("account_available", "unknown"),
                "last_checked": item.get("last_checked"),
                "last_probe_at": item.get("last_probe_at"),
                "last_probe_error_type": item.get("last_probe_error_type", ""),
                "last_probe_error": item.get("last_probe_error", ""),
                "notes": item.get("notes", ""),
                "source_notes": item.get("source_notes", ""),
                "source_urls": item.get("source_urls", []),
            }
        )
    return {
        "updated_at": data.get("updated_at"),
        "path": str(REGISTRY_PATH),
        "tiers": tiers,
        "routable": {tier: tier_routes(tier) for tier in sorted(VALID_TIERS - {"disabled_candidate"})},
    }


def _compact_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "provider": item.get("provider"),
            "model_id": item.get("model_id"),
            "tier": item.get("tier"),
            "additional_tiers": item.get("additional_tiers", []),
            "candidate_tiers": item.get("candidate_tiers", []),
            "enabled": item.get("enabled", False),
            "health_status": item.get("health_status", "unchecked"),
            "official_listed": item.get("official_listed", "unknown"),
            "local_catalog_present": item.get("local_catalog_present", "unknown"),
            "account_available": item.get("account_available", "unknown"),
        }
        for item in entries
    ]
