"""Trajectory: one consolidated, chronological, human-readable view of a single
run's plan -> retrieve -> replan -> patch chain.

Assembled entirely from data the harness already writes (the manifest's
`events` and `artifact_refs`, jspace.py) - no new logging is added anywhere.
Every real bug found and fixed in this harness in one long session required
manually cross-referencing three or four separate JSON files under
`j_space/tasks/<run_id>/context/{requests,retrieval,replans}/` by hand to
reconstruct what actually happened. That data was always there; it was just
never surfaced as one readable narrative. This is a pure read-only report -
it never writes into the manifest itself, so building or re-running it can
never affect a run's own provenance record.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, jspace
from .util import now_iso, portable_repo_path


def _read_json(path: str) -> dict:
    if not path:
        return {}
    candidate = config.ROOT / path
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _artifact_json_by_id(manifest: dict, artifact_type: str, artifact_id: str) -> dict:
    for ref in manifest.get("artifact_refs") or []:
        if ref.get("type") == artifact_type and ref.get("artifact_id") == artifact_id:
            return _read_json(str(ref.get("json_path") or ""))
    return {}


def _latest_artifact_json(manifest: dict, artifact_type: str) -> dict:
    matches = [ref for ref in manifest.get("artifact_refs") or [] if ref.get("type") == artifact_type]
    if not matches:
        return {}
    matches.sort(key=lambda ref: str(ref.get("recorded_at") or ""))
    return _read_json(str(matches[-1].get("json_path") or ""))


def _summarize_event(event: dict, manifest: dict) -> str:
    name = str(event.get("event") or "")
    if name == "workspace_created":
        return "Workspace created."
    if name == "selected_plan_excerpts_snapshotted":
        count = event.get("entry_count", 0)
        return f"Snapshotted {count} selected file excerpt(s) for planning."
    if name == "checkpoint_updated":
        stage = event.get("stage")
        status = event.get("checkpoint_status")
        detail = str(event.get("detail") or "").strip()
        meta = event.get("metadata") or {}
        route = meta.get("final_route") or meta.get("model_route") or ""
        parts = [f"[{stage}] {status}"]
        if detail:
            parts.append(detail)
        if route:
            parts.append(f"(model: {route})")
        return " - ".join(parts)
    if name == "context_request_recorded":
        return f"Context request classified: status={event.get('status')}."
    if name == "repository_retrieval_recorded":
        return f"Retrieval executed: status={event.get('status', '')}."
    if name == "context_request_invalidated":
        return "Context request invalidated (stale context)."
    if name == "repository_map_created":
        return "Repository map created."
    if name == "grounded_replan_recorded":
        revision_id = str(event.get("revision_id") or "")
        replan_json = _artifact_json_by_id(manifest, "grounded_replan", revision_id)
        checker = replan_json.get("checker_report") or {}
        failed_rules = checker.get("failed_rules") or []
        route = replan_json.get("model_route") or ""
        parts = [
            f"Grounded replan {revision_id}: checker={event.get('checker_status')}, "
            f"context={event.get('context_sufficiency')}"
        ]
        if route:
            parts.append(f"(model: {route})")
        if failed_rules:
            parts.append("failed_rules=" + ", ".join(str(item) for item in failed_rules))
        return " - ".join(parts)
    return name or "(unrecognized event)"


def build_trajectory(run_id: str) -> dict:
    """Assemble the chronological step list for one run_id from its manifest."""
    manifest = jspace.load_manifest(run_id)
    if not manifest:
        raise RuntimeError(f"No J-space manifest found for run_id: {run_id}")
    events = sorted(manifest.get("events") or [], key=lambda item: str(item.get("created_at") or ""))
    steps = []
    for event in events:
        steps.append(
            {
                "event": event.get("event"),
                "stage": event.get("stage"),
                "created_at": event.get("created_at"),
                "actor": event.get("actor"),
                "summary": _summarize_event(event, manifest),
            }
        )
    patch_json = _latest_artifact_json(manifest, "engineer_patch")
    return {
        "run_id": run_id,
        "task": str((manifest.get("task") or {}).get("request") or ""),
        "status": manifest.get("status"),
        "current_stage": manifest.get("current_stage"),
        "step_count": len(steps),
        "steps": steps,
        "final_patch_status": patch_json.get("patch_status"),
        "final_patch_change_summary": patch_json.get("change_summary"),
        "generated_at": now_iso(),
    }


def trajectory_markdown(trajectory: dict) -> str:
    lines = [
        f"# Trajectory: {trajectory.get('run_id', '')}",
        "",
        f"- Status: `{trajectory.get('status', '')}` (stage: `{trajectory.get('current_stage', '')}`)",
        f"- Steps: {trajectory.get('step_count', 0)}",
        f"- Final patch status: `{trajectory.get('final_patch_status') or '(no patch attempted)'}`",
        "",
        "## Task",
        "",
        str(trajectory.get("task") or ""),
        "",
        "## Steps",
        "",
    ]
    for index, step in enumerate(trajectory.get("steps") or [], start=1):
        lines.append(f"{index}. **[{step.get('created_at', '')}]** {step.get('summary', '')}")
    if trajectory.get("final_patch_change_summary"):
        lines.extend(["", "## Final patch", "", str(trajectory["final_patch_change_summary"])])
    lines.append("")
    return "\n".join(lines)


def engineer_trajectory(run_id: str, save: bool = False) -> dict:
    """Build a trajectory report for run_id. Saves to disk under the run's own
    j_space workspace only when save=True (this is a reporting tool, not a
    provenance-writing one, so saving is opt-in rather than automatic).
    """
    trajectory = build_trajectory(run_id)
    result = dict(trajectory)
    result["ok"] = True
    result["output"] = trajectory_markdown(trajectory)
    if save:
        workspace, _, _ = jspace.paths(run_id)
        workspace.mkdir(parents=True, exist_ok=True)
        json_path = workspace / "trajectory.json"
        md_path = workspace / "trajectory.md"
        json_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(trajectory_markdown(trajectory), encoding="utf-8")
        result["path"] = str(md_path)
        result["jsonPath"] = str(json_path)
        result["portable_path"] = portable_repo_path(md_path)
    return result
