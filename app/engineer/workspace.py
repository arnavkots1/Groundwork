"""Workspace-level operations that are not part of the Engineer loop.

Status, artifact summaries, model-registry inspection, provider probes, and the
legacy free-form agent-run helpers. Kept separate from the loop so `api.py` reads as
one product surface without these becoming loop concepts.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from api_clients import PROVIDERS, ProviderError, call_provider, list_openai_compatible_models, provider_status
from model_registry import record_probe_result, refresh_registry, registry_summary
from nvidia_catalog import all_available_models
from resilient_calls import format_attempts

from . import artifacts, config, jspace, models, prompts
from .util import read_text, truncate


def run_summary(target: str, use_model: bool = False) -> dict:
    """Summarize a saved artifact. Deterministic extraction always works;
    a utility_light model route is optional and never escalates to elite routes."""
    path = artifacts.resolve_summary_target(target)
    text = read_text(path)
    payload = None
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
    lines = artifacts.deterministic_artifact_summary(path, payload, text)
    summary = "\n".join(lines)
    result = {
        "ok": True,
        "path": str(path),
        "summarySource": "deterministic",
        "summaryLines": lines,
        "summary": summary,
        "output": summary,
    }
    if use_model:
        try:
            routed = models.call_tier_with_fallback(
                "Run Summarizer",
                prompts.render("run_summary", {"artifact_text": truncate(text, 6000)}),
                tier="utility_light",
                fallback_to_best=False,
            )
            model_summary = routed.content.strip()
            if model_summary:
                result.update(
                    {
                        "summarySource": f"utility_light: {routed.provider} {routed.model}".strip(),
                        "summary": model_summary,
                        "output": model_summary,
                        "deterministicFallback": summary,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            result["modelSummaryError"] = str(exc)[:400]
    return result


def model_registry_show() -> dict:
    summary = registry_summary()
    lines = [f"Model registry ({summary.get('updated_at') or 'never refreshed'}):"]
    for tier, entries in sorted((summary.get("tiers") or {}).items()):
        lines.append(f"- {tier}:")
        for entry in entries:
            state = "enabled" if entry.get("enabled") else "disabled"
            source_state = (
                f"official={entry.get('official_listed', 'unknown')}, "
                f"catalog={entry.get('local_catalog_present', 'unknown')}, "
                f"account={entry.get('account_available', 'unknown')}"
            )
            notes = entry.get("notes") or entry.get("source_notes") or ""
            lines.append(
                f"  - {entry.get('provider')} {entry.get('model_id')} "
                f"[tier={entry.get('tier')}, {state}, health={entry.get('health_status') or 'unchecked'}, {source_state}]"
            )
            if notes:
                lines.append(f"    notes: {str(notes)[:220]}")
    return {**summary, "output": "\n".join(lines)}


def model_registry_refresh() -> dict:
    report = refresh_registry()
    changes = report.get("changes") or []
    lines = [
        f"Registry refreshed at {report.get('updated_at')}: {report.get('checked')} entries checked.",
        f"NVIDIA catalog models: {report.get('nvidia_catalog_models')}.",
        f"Verified active models: {len(report.get('verified_active_models') or [])}.",
        f"Candidates needing probe: {len(report.get('candidates_needing_probe') or [])}.",
        f"Unavailable models: {len(report.get('unavailable_models') or [])}.",
        f"Stale/missing catalog models: {len(report.get('stale_or_missing_catalog_models') or [])}.",
    ]
    if changes:
        lines.append("Registry field changes:")
        lines.extend(f"- {item['provider']} {item['model_id']}: {item['from']} -> {item['to']}" for item in changes)
    else:
        lines.append("No registry field changes.")
    return {**report, "ok": True, "output": "\n".join(lines)}


def _canonical_provider_name(provider: str) -> str:
    normalized = provider.strip().lower()
    aliases = {
        "nvidia": "NVIDIA NIM",
        "nvidia nim": "NVIDIA NIM",
        "nim": "NVIDIA NIM",
        "gemini": "Gemini",
        "openai": "OpenAI API",
        "openai api": "OpenAI API",
        "kimi": "Kimi / Moonshot",
        "moonshot": "Kimi / Moonshot",
        "cursor": "Cursor Agent CLI",
        "cursor agent": "Cursor Agent CLI",
        "cursor agent cli": "Cursor Agent CLI",
    }
    return aliases.get(normalized, provider)


def _probe_error_type(exc: Exception) -> str:
    text = str(exc).lower()
    if isinstance(exc, ProviderError):
        if exc.is_rate_limit:
            return "rate_limited"
        if exc.is_unavailable:
            return "unavailable"
        if exc.quota_exhausted:
            return "quota_exhausted"
        if exc.status_code:
            return f"http_{exc.status_code}"
    if "api key" in text or "env var" in text:
        return "missing_api_key"
    if "network" in text:
        return "network_error"
    if "timeout" in text:
        return "timeout"
    if isinstance(exc, ProviderError) and exc.retryable:
        return "retryable_error"
    return "error"


def _safe_probe_error(exc: Exception) -> str:
    """Flatten and redact a provider error before it reaches an artifact."""
    text = str(exc).replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(api[_-]?key[=:\s]+)[^\s,;]+", r"\1[REDACTED]", text)
    return text[:900] or "Provider call failed without an error message."


def _latest_probe_route_error(provider: str, model_id: str) -> str:
    state_path = config.ROOT / "brain" / "model_health" / "provider_rate_limit_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    entry = (state.get("models") or {}).get(f"{provider}:{model_id}")
    if not isinstance(entry, dict):
        return ""
    raw_error = str(entry.get("last_error") or "").strip()
    return _safe_probe_error(RuntimeError(raw_error)) if raw_error else ""


def model_probe(provider: str, model_id: str, enable_if_ok: bool = False) -> dict:
    provider_name = _canonical_provider_name(provider)
    if not model_id.strip():
        return {
            "ok": False,
            "provider": provider_name,
            "model_id": "",
            "health_status": "error",
            "error_type": "missing_model_id",
            "latency_ms": 0,
            "enabled_recommendation": "do_not_enable",
            "output": "Missing --model-id.",
        }
    model_id = model_id.strip()
    if provider_name not in PROVIDERS:
        return {
            "ok": False,
            "provider": provider_name,
            "model_id": model_id,
            "health_status": "error",
            "error_type": "unknown_provider",
            "latency_ms": 0,
            "enabled_recommendation": "do_not_enable",
            "output": "Unknown provider.",
        }
    start = datetime.now()
    latency_ms = 0
    error_type = ""
    health_status = "error"
    ok = False
    error_reason = ""
    try:
        content = call_provider(
            provider_name,
            "Model Registry Probe",
            "Reply with exactly OK.",
            model_override=model_id,
            request_timeout=20,
            max_retries_override=0,
            max_tokens=4,
        )
        latency_ms = int((datetime.now() - start).total_seconds() * 1000)
        ok = bool(str(content).strip())
        health_status = "ok" if ok else "error"
        error_type = "" if ok else "empty_response"
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((datetime.now() - start).total_seconds() * 1000)
        error_type = _probe_error_type(exc)
        error_reason = _safe_probe_error(exc)
        if "cooling down" in error_reason.lower():
            latest_route_error = _latest_probe_route_error(provider_name, model_id)
            if latest_route_error:
                error_reason = latest_route_error
                error_type = _probe_error_type(RuntimeError(latest_route_error))
        health_status = "unavailable" if error_type in {"missing_api_key", "unavailable", "quota_exhausted"} or error_type.startswith("http_4") else "error"

    record_probe_result(
        provider_name,
        model_id,
        ok=ok,
        health_status=health_status,
        error_type=error_type,
        error_reason=error_reason,
        enable_if_ok=enable_if_ok,
    )
    recommendation = "enable" if ok else "do_not_enable"
    if ok and not enable_if_ok:
        recommendation = "enable_with_--enable-if-ok"
    payload = {
        "ok": ok,
        "provider": provider_name,
        "model_id": model_id,
        "health_status": health_status,
        "error_type": error_type,
        "error_reason": error_reason,
        "latency_ms": latency_ms,
        "enabled_recommendation": recommendation,
    }
    payload["output"] = json.dumps(payload, ensure_ascii=False)
    return payload


def save_run(role: str, provider: str, prompt: str, result: str) -> str:
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_role = "".join(ch.lower() if ch.isalnum() else "-" for ch in role).strip("-")
    path = config.RUNS_DIR / f"{stamp}-{safe_role}.md"
    path.write_text(
        f"# Agent Run: {role}\n\n"
        f"- Provider: {provider}\n"
        f"- Date: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"## Prompt\n\n{prompt}\n\n"
        f"## Output\n\n{result}\n",
        encoding="utf-8",
    )
    return str(path)


def status() -> dict:
    projects = config.ROOT / "brain" / "projects" / "index.md"
    project_rows = 0
    if projects.exists():
        project_rows = sum(
            1 for line in projects.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("| [")
        )
    return {
        "providers": [{"name": name, "model": model, "status": state} for name, model, state in provider_status()],
        "projects": project_rows,
        "markdownFiles": sum(1 for _path in (config.ROOT / "brain").rglob("*.md")),
        "brainV2Exists": config.BRAIN_V2_DIR.exists(),
        "engineerRuns": len(list(config.ENGINEER_RUNS_DIR.glob("*.md"))) if config.ENGINEER_RUNS_DIR.exists() else 0,
        "engineerJSpaceTasks": (
            len([path for path in config.ENGINEER_J_SPACE_TASKS_DIR.iterdir() if path.is_dir()])
            if config.ENGINEER_J_SPACE_TASKS_DIR.exists()
            else 0
        ),
        "nvidiaModels": len(all_available_models()),
        "modelRegistry": {
            "updatedAt": registry_summary().get("updated_at"),
            "models": sum(len(items) for items in registry_summary().get("tiers", {}).values()),
        },
        "files": [{"label": label, "path": str(path), "exists": path.exists()} for label, path in config.FILES.items()],
    }


def call_agent(provider: str, role: str, prompt: str) -> dict:
    if provider == "Best Available":
        routed = models.call_best_available_with_fallback(role, prompt)
        result = f"{routed.content}\n\n{format_attempts(routed.attempts)}"
        path = save_run(role, models.route_of(routed), prompt, result)
        return {"output": result, "path": path}
    result = call_provider(provider, role, prompt)
    path = save_run(role, provider, prompt, result)
    return {"output": result, "path": path}


def read_file(label: str) -> dict:
    path = config.FILES.get(label)
    if path is None:
        raise RuntimeError(f"Unknown Company Brain file: {label}")
    if not path.exists():
        return {"label": label, "path": str(path), "content": f"Missing file: {path}"}
    return {"label": label, "path": str(path), "content": path.read_text(encoding="utf-8", errors="replace")}


def read_run(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_absolute():
        path = config.ROOT / path
    resolved = path.resolve()
    root_resolved = config.ROOT.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise RuntimeError(f"Refusing to read run outside repo: {resolved}")
    if not resolved.exists():
        raise RuntimeError(f"Run file does not exist: {resolved}")
    return {"path": str(resolved), "content": resolved.read_text(encoding="utf-8", errors="replace")}


def list_runs() -> dict:
    runs = []
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for path in config.RUNS_DIR.glob("*.md"):
        runs.append({"name": path.name, "path": str(path), "modified": path.stat().st_mtime})
    config.ENGINEER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for path in config.ENGINEER_RUNS_DIR.glob("*_engineer_plan.md"):
        run = {"name": path.name, "path": str(path), "modified": path.stat().st_mtime}
        j_space_fields = jspace.response_fields(path.stem)
        if j_space_fields:
            run.update(
                {
                    "jSpacePath": j_space_fields["jSpacePath"],
                    "jSpaceJsonPath": j_space_fields["jSpaceJsonPath"],
                }
            )
        runs.append(run)
    for directory, pattern in [
        (config.ENGINEER_REVIEWS_DIR, "*_engineer_review.md"),
        (config.ENGINEER_PATCHES_DIR, "*_engineer_patch.md"),
        (config.ENGINEER_APPLIED_PATCHES_DIR, "*_engineer_applied_patch.md"),
        (config.ENGINEER_VERIFICATIONS_DIR, "*_engineer_verification.md"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob(pattern):
            runs.append({"name": path.name, "path": str(path), "modified": path.stat().st_mtime})
    runs.sort(key=lambda item: item["modified"], reverse=True)
    return {"runs": runs[:30]}


def save_outbox(prompt: str) -> dict:
    content = prompt.strip()
    if not content:
        raise RuntimeError("Outbox prompt is empty.")
    config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = config.OUTBOX_DIR / f"{stamp}-codex-request.md"
    path.write_text(f"# Codex Bridge Request\n\n{content}\n", encoding="utf-8")
    return {"output": "Saved Codex bridge request.", "path": str(path)}


def list_nvidia_models() -> dict:
    model_ids = list_openai_compatible_models("NVIDIA NIM")
    out = config.ROOT / "config" / "nvidia_available_models.json"
    out.write_text(
        json.dumps({"count": len(model_ids), "models": model_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output": f"NVIDIA /models returned {len(model_ids)} models.", "path": str(out), "models": model_ids}
