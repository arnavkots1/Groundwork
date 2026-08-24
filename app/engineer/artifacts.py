"""Human-readable renderings of every artifact, plus artifact lookup.

Artifacts are always written, including for blocked and failed runs: a refusal that
leaves no trace is indistinguishable from a step that never ran. The Markdown here
is a view over the JSON; the JSON is the record.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .util import repo_path


def _sections(title: str, fields: list[tuple[str, object]], diff_field: str = "") -> str:
    sections = [title, ""]
    for name, value in fields:
        sections.extend([f"## {name}", ""])
        if diff_field and name == diff_field:
            sections.extend(["```diff", str(value), "```"])
        elif isinstance(value, str):
            sections.append(value)
        else:
            sections.append(json.dumps(value, ensure_ascii=False, indent=2))
        sections.append("")
    return "\n".join(sections)


def plan_markdown(payload: dict) -> str:
    fields = [
        ("Task", payload.get("task", "")),
        ("Blocked Plan", payload.get("blocked_plan", False)),
        ("Context Request", payload.get("context_request", "")),
        ("J-space", payload.get("j_space", {})),
        ("Context Manifest", json.dumps(payload.get("context_manifest", {}), ensure_ascii=False, indent=2)),
        ("Selected File Inspections", json.dumps(payload.get("selected_file_inspections", []), ensure_ascii=False, indent=2)),
        ("Model Routes Attempted", json.dumps(payload.get("model_routes_attempted", []), ensure_ascii=False, indent=2)),
        ("Final Route", payload.get("final_route", "")),
        ("Route Status", payload.get("route_status", "")),
        ("Structured Parse Status", payload.get("structured_parse_status", "")),
        ("Checker Status", payload.get("checker_status", "")),
        ("Checker Report", payload.get("checker_report", {})),
        ("Checker Warnings", payload.get("checker_warnings", [])),
        ("Checker Candidate Lessons", payload.get("candidate_lessons_from_checker", [])),
        ("Verified Facts", payload.get("verified_facts", [])),
        ("Unverified Assumptions", payload.get("unverified_assumptions", [])),
        ("Repo Grounding Score", payload.get("repo_grounding_score", "")),
        ("Original Plan", payload.get("original_plan", {})),
        ("Checker-Adjusted Plan", payload.get("checker_adjusted_plan", {})),
        ("Task Understanding", payload.get("task_understanding", "")),
        ("Assumptions", payload.get("assumptions", [])),
        ("Files Likely Involved", payload.get("files_likely_involved", [])),
        ("Implementation Plan", payload.get("implementation_plan", [])),
        ("Risks", payload.get("risks", [])),
        ("Forbidden Changes", payload.get("forbidden_changes", [])),
        ("Acceptance Tests", payload.get("acceptance_tests", [])),
        ("Codex Prompt", payload.get("codex_prompt", "")),
        ("Revised Codex Prompt", payload.get("revised_codex_prompt", "")),
        ("Self Review Checklist", payload.get("self_review_checklist", [])),
        ("Post Run Review Template", payload.get("post_run_review_template", "")),
        ("Rubric Self Score", payload.get("rubric_self_score", {})),
        ("Limitations", payload.get("limitations", [])),
    ]
    text = _sections(f"# Engineer Plan: {payload.get('run_id', '')}", fields)
    if payload.get("baseline_output"):
        text += "\n" + "\n".join(["## Baseline Output", "", str(payload["baseline_output"]), ""])
    return text


def review_markdown(review: dict) -> str:
    fields = [
        "source_run_id",
        "j_space",
        "execution_summary",
        "planned_files",
        "selected_files",
        "changed_files",
        "files_match",
        "compile_passed",
        "tests_passed",
        "files_touched_correct",
        "manual_correction",
        "plan_followed_score",
        "prompt_quality_score",
        "missed_context",
        "what_worked",
        "what_failed",
        "checker_missed",
        "proposed_candidate_lessons",
        "next_prompt_improvement",
        "decision",
        "review_rules_triggered",
    ]
    return _sections(
        f"# Engineer Review: {review.get('review_id', '')}",
        [(field.replace("_", " ").title(), review.get(field)) for field in fields],
    )


def patch_markdown(payload: dict) -> str:
    fields = [
        ("Source Run", payload.get("source_run_id", "")),
        ("J-space", payload.get("j_space", {})),
        ("Task", payload.get("task", "")),
        ("Risk Level", payload.get("risk_level", "")),
        ("Risk Reasons", payload.get("risk_reasons", [])),
        ("Target Files", payload.get("target_files", [])),
        ("Change Summary", payload.get("change_summary", "")),
        ("Apply Allowed", payload.get("apply_allowed", False)),
        ("Patch Status", payload.get("patch_status", "")),
        ("Generation Mode", payload.get("generation_mode", "")),
        ("Grounding Required", payload.get("grounding_required", False)),
        ("Grounding Gate", payload.get("grounding_gate", {})),
        ("Available Claim IDs", payload.get("available_claim_ids", [])),
        ("Available Plan Action IDs", payload.get("available_plan_action_ids", [])),
        ("Claim Links", payload.get("claim_links", [])),
        ("Patch Claim Trace", payload.get("patch_claim_trace", [])),
        ("Patch Checker", payload.get("patch_checker", {})),
        ("Unified Diff", payload.get("unified_diff", "")),
        ("Verification Commands", payload.get("verification_commands", [])),
        ("Verification Commands Quality", payload.get("verification_commands_quality", "")),
        ("Verification Command Warnings", payload.get("verification_command_warnings", [])),
        ("Verification Commands Suggested Manual", payload.get("verification_commands_suggested_manual", [])),
        ("Verification Commands Original", payload.get("verification_commands_original", [])),
        ("Rollback Plan", payload.get("rollback_plan", "")),
        ("Forbidden Changes Checked", payload.get("forbidden_changes_checked", [])),
        ("Limitations", payload.get("limitations", [])),
    ]
    return _sections(f"# Engineer Patch: {payload.get('patch_id', '')}", fields, diff_field="Unified Diff")


def applied_patch_markdown(payload: dict) -> str:
    fields = [
        ("Source Run", payload.get("source_run_id", "")),
        ("Patch ID", payload.get("patch_id", "")),
        ("Applied Patch ID", payload.get("applied_patch_id", "")),
        ("J-space", payload.get("j_space", {})),
        ("Created At", payload.get("created_at", "")),
        ("Status", payload.get("status", "")),
        ("Changed Files", payload.get("changed_files", [])),
        ("Backup Directory", payload.get("backup_dir", "")),
        ("Safety Checks", payload.get("safety_checks", {})),
        ("Warnings", payload.get("warnings", [])),
        ("Verification Commands", payload.get("verification_commands", [])),
        ("Verification Commands Quality", payload.get("verification_commands_quality", "")),
        ("Verification Commands Suggested Manual", payload.get("verification_commands_suggested_manual", [])),
    ]
    return _sections(f"# Engineer Applied Patch: {payload.get('applied_patch_id', '')}", fields)


def verification_markdown(payload: dict) -> str:
    fields = [
        ("Source Run", payload.get("source_run_id", "")),
        ("Applied Patch ID", payload.get("applied_patch_id", "")),
        ("Verification ID", payload.get("verification_id", "")),
        ("J-space", payload.get("j_space", {})),
        ("Created At", payload.get("created_at", "")),
        ("Run Commands", payload.get("run_commands", False)),
        ("Commands Prepared", payload.get("commands", [])),
        ("Commands Run", payload.get("commands_run", [])),
        ("Commands Skipped", payload.get("commands_skipped", [])),
        ("Commands Suggested Manual", payload.get("commands_suggested_manual", [])),
        ("Verification Passed", payload.get("verification_passed", False)),
        ("Verification Incomplete", payload.get("verification_incomplete", True)),
        ("Results", payload.get("results", [])),
        ("Warnings", payload.get("warnings", [])),
    ]
    return _sections(f"# Engineer Verification: {payload.get('verification_id', '')}", fields)


def resolve_artifact_json(path_id: str, directory: Path) -> Path:
    """Resolve an artifact id to a JSON file that must live under `directory`."""
    candidate = path_id.strip()
    if not candidate:
        raise RuntimeError("Missing artifact id.")
    path = Path(candidate)
    if path.suffix != ".json":
        path = Path(f"{candidate}.json")
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (directory / path.name).resolve()
        if not resolved.exists():
            resolved = (directory / path).resolve()
    if not resolved.exists():
        raise RuntimeError(f"Artifact not found: {candidate}")
    if resolved.suffix.lower() != ".json":
        raise RuntimeError(f"Expected JSON artifact: {resolved}")
    checked = repo_path(str(resolved))
    if directory.resolve() not in checked.parents:
        raise RuntimeError(f"Artifact must live under {directory}: {checked}")
    return checked


def load_run_json(run_id: str) -> tuple[Path, dict]:
    if not run_id.strip():
        raise RuntimeError("engineer-check-run requires --run-id.")
    candidate = Path(run_id)
    if not candidate.is_absolute():
        candidate = config.ENGINEER_RUNS_DIR / candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    resolved = candidate.resolve()
    root_resolved = config.ROOT.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise RuntimeError(f"Refusing to read Engineer run outside repo: {resolved}")
    if not resolved.exists():
        raise RuntimeError(f"Engineer run JSON not found: {resolved}")
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def artifact_dirs() -> list[Path]:
    return [
        config.ENGINEER_RUNS_DIR,
        config.ENGINEER_PATCHES_DIR,
        config.ENGINEER_APPLIED_PATCHES_DIR,
        config.ENGINEER_VERIFICATIONS_DIR,
        config.ENGINEER_REVIEWS_DIR,
    ]


def resolve_summary_target(target: str) -> Path:
    text = target.strip()
    if not text:
        raise RuntimeError("run-summary requires --target <artifact id or path>.")
    direct = Path(text)
    candidates = [direct] if direct.suffix else [direct.with_suffix(".json"), direct.with_suffix(".md")]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else config.ROOT / candidate
        if path.exists() and path.is_file():
            return repo_path(str(path))
    stem = Path(text).stem if Path(text).suffix else Path(text).name
    for directory in artifact_dirs():
        for suffix in (".json", ".md"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    raise RuntimeError(f"run-summary target not found: {text}")


def _count(value) -> int:
    return len(value) if isinstance(value, list) else 0


def deterministic_artifact_summary(path: Path, payload: dict | None, text: str) -> list[str]:
    """Extractive summary. Never calls a model, so it is always available."""
    if isinstance(payload, dict):
        if payload.get("review_id") or payload.get("decision"):
            return [
                f"Engineer review {payload.get('review_id', path.stem)} for run {payload.get('source_run_id', '?')}.",
                f"Decision: {payload.get('decision', 'unknown')}; plan-followed {payload.get('plan_followed_score', '?')}/5, prompt-quality {payload.get('prompt_quality_score', '?')}/5.",
                f"Compile: {payload.get('compile_passed')}; tests: {payload.get('tests_passed')}; files correct: {payload.get('files_touched_correct')}; manual correction: {payload.get('manual_correction') or 'unset'}.",
                f"What failed: {_count(payload.get('what_failed'))} item(s); candidate lessons proposed: {_count(payload.get('proposed_candidate_lessons'))}.",
            ]
        if payload.get("verification_id"):
            results = payload.get("results") or []
            passed = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "pass")
            failed = sum(1 for item in results if isinstance(item, dict) and item.get("status") in {"fail", "error"})
            return [
                f"Engineer verification {payload.get('verification_id', path.stem)} for {payload.get('applied_patch_id', '?')}.",
                f"Commands: {_count(payload.get('commands'))}; executed: {'yes' if payload.get('run_commands') else 'no (prepared only)'}.",
                f"Results: {passed} passed, {failed} failed/errored." if payload.get("run_commands") else "Run with --run to execute allowlisted commands.",
            ]
        if payload.get("applied_patch_id") and payload.get("changed_files") is not None:
            return [
                f"Engineer applied patch {payload.get('applied_patch_id', path.stem)} (from {payload.get('patch_id', '?')}).",
                f"Status: {payload.get('status', 'unknown')}; {_count(payload.get('changed_files'))} file(s) changed.",
                f"Backup: {Path(str(payload.get('backup_dir', ''))).name or 'none'}.",
                f"Next: engineer-verify --applied-patch-id {payload.get('applied_patch_id', path.stem)}.",
            ]
        if payload.get("patch_id") and payload.get("patch_status"):
            lines = [
                f"Engineer patch {payload.get('patch_id', path.stem)} from run {payload.get('source_run_id', '?')}.",
                f"Status: {payload.get('patch_status')}; risk: {payload.get('risk_level', 'unknown')}; apply_allowed: {payload.get('apply_allowed', False)}.",
                f"Targets: {', '.join(payload.get('target_files') or []) or 'none'}.",
            ]
            if payload.get("patch_status") == "blocked":
                limitations = payload.get("limitations") or []
                lines.append(f"Blocked: {str(limitations[0])[:200] if limitations else 'see artifact limitations'}.")
            return lines
        if payload.get("run_id"):
            lines = [
                f"Engineer plan {payload.get('run_id', path.stem)}.",
                f"Task: {str(payload.get('task', ''))[:180]}",
                f"Checker: {payload.get('checker_status', 'unknown')}; route: {payload.get('route_status', 'unknown')} ({str(payload.get('final_route', ''))[:80]}).",
                f"Selected files: {_count((payload.get('context_manifest') or {}).get('selected_files'))}; warnings: {_count(payload.get('checker_warnings'))}; acceptance tests: {_count(payload.get('acceptance_tests'))}.",
            ]
            if payload.get("blocked_plan"):
                lines.append("BLOCKED: code task had no selected files; rerun with --selected-file.")
            return lines
        keys = ", ".join(list(payload.keys())[:8])
        return [f"JSON artifact {path.name}.", f"Top-level fields: {keys}."]
    heading = next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith("#")), "")
    body = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    lines = [f"Artifact {path.name}." if not heading else f"{heading} ({path.name})."]
    lines.extend(body[:3])
    return lines
