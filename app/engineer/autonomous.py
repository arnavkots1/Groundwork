"""Envelope-bound Engineer execution in a disposable code checkout.

Only the code checkout is disposable. Plans, retrievals, J-space provenance,
patches, sandbox-Apply artifacts, verification results, reviews, refusals, and the
final report are written to the real Engineer state tree.

The real source surface is hashed before and after. Equality proves an identical
end state, not the absence of a transient write by an out-of-process adversary.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from . import artifacts, config, jspace
from .apply import engineer_apply_patch
from .envelope import (
    load_task_envelope,
    record_envelope_event,
    verification_command_specs,
    verification_commands,
    write_targets,
)
from .loop import engineer_check_run, engineer_plan
from .patch import engineer_patch
from .replan import engineer_replan
from .retrieval import (
    engineer_context_execute,
    engineer_context_request,
)
from .review import engineer_review
from .util import append_jsonl, now_tz_iso, portable_repo_path, sha256_file
from .verify import engineer_verify


AUTONOMOUS_REPORT_SCHEMA = "companybrain.engineer.autonomous_run.v1"
PROTECTED_SOURCE_HASH_SCHEMA = "companybrain.protected_source_tree.v1"
_EVIDENCE_PREFIXES = (
    "brain_v2/employees/engineer/runs",
    "brain_v2/employees/engineer/patches",
    "brain_v2/employees/engineer/applied_patches",
    "brain_v2/employees/engineer/verifications",
    "brain_v2/employees/engineer/reviews",
    "brain_v2/employees/engineer/j_space",
    "brain_v2/employees/engineer/envelopes",
    "brain_v2/employees/engineer/autonomous_runs",
    "brain_v2/evals/engineer",
    "brain_v2/runs/engineer",
    "brain_v2/research/evidence_notes",
    "brain_v2/artifacts",
)
_PROTECTED_BEHAVIOR_PREFIXES = (
    "brain_v2/employees/engineer/behavior_promotions",
    "brain_v2/evals/engineer/behavior_evaluations",
)
_EVIDENCE_FILES = frozenset(
    {
        "brain_v2/employees/engineer/candidate_lessons.jsonl",
        "brain_v2/employees/engineer/rejected_lessons.jsonl",
        "brain_v2/employees/engineer/failures.jsonl",
        "brain_v2/employees/engineer/eval_history.jsonl",
        "brain_v2/employees/engineer/version_history.jsonl",
        "brain_v2/employees/engineer/consent_events.jsonl",
        "brain/model_health/provider_rate_limit_state.json",
    }
)
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".tmp",
        ".permcheck_tmp2",
        ".claude",
        ".next",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "coverage",
        "dist",
    }
)


def _is_evidence_path(relative: str) -> bool:
    # Promotion locks/snapshots and their held-out evaluator bundles control future
    # behavior. They remain part of the protected source/behavior hash even though
    # neighboring eval artifacts are ordinary durable evidence.
    if any(
        relative == prefix or relative.startswith(prefix + "/")
        for prefix in _PROTECTED_BEHAVIOR_PREFIXES
    ):
        return False
    return relative in _EVIDENCE_FILES or any(
        relative == prefix or relative.startswith(prefix + "/")
        for prefix in _EVIDENCE_PREFIXES
    )


def _is_ignored_path(relative: Path) -> bool:
    parts = set(relative.parts)
    if parts & _IGNORED_PARTS:
        return True
    return any(part.startswith(".venv") for part in relative.parts) or any(
        part.endswith((".pyc", ".pyo")) for part in relative.parts
    )


def _tree_snapshot(root: Path, *, evidence_only: bool = False) -> dict[str, Any]:
    """Hash sorted relative names, node types, symlink targets, and file bytes."""

    resolved_root = root.resolve()
    entries: list[dict[str, Any]] = []
    if not resolved_root.exists():
        return {
            "schema": PROTECTED_SOURCE_HASH_SCHEMA,
            "root": str(resolved_root),
            "sha256": hashlib.sha256(b"<missing>").hexdigest(),
            "entries": [],
            "file_count": 0,
        }
    for path in sorted(resolved_root.rglob("*"), key=lambda item: item.relative_to(resolved_root).as_posix()):
        relative_path = path.relative_to(resolved_root)
        relative = relative_path.as_posix()
        if _is_ignored_path(relative_path):
            continue
        evidence = _is_evidence_path(relative)
        if evidence_only != evidence:
            continue
        if path.is_symlink():
            entries.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    digest_value = _entries_digest(entries)
    return {
        "schema": PROTECTED_SOURCE_HASH_SCHEMA,
        "root": str(resolved_root),
        "sha256": digest_value,
        "entries": entries,
        "file_count": sum(item.get("type") == "file" for item in entries),
        "exclusions": {
            "durable_evidence_prefixes": list(_EVIDENCE_PREFIXES),
            "durable_evidence_files": sorted(_EVIDENCE_FILES),
            "protected_behavior_prefixes": list(
                _PROTECTED_BEHAVIOR_PREFIXES
            ),
            "ignored_runtime_parts": sorted(_IGNORED_PARTS),
        },
    }


def _entries_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(PROTECTED_SOURCE_HASH_SCHEMA.encode("utf-8") + b"\0")
    for entry in entries:
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _entry_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("path") or ""): item for item in snapshot.get("entries") or []}


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left = _entry_map(before)
    right = _entry_map(after)
    return {
        "created": [right[key] for key in sorted(set(right) - set(left))],
        "modified": [
            {"path": key, "before": left[key], "after": right[key]}
            for key in sorted(set(left) & set(right))
            if left[key] != right[key]
        ],
        "removed": [left[key] for key in sorted(set(left) - set(right))],
    }


def _prefix_sha256(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = max(0, int(size))
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    if remaining:
        return ""
    return digest.hexdigest()


def _evidence_accumulation(
    before: dict[str, Any],
    after: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Require durable artifacts to be created or append-only, never rebuilt."""

    delta = _state_delta(before, after)
    violations: list[dict[str, Any]] = []
    for removed in delta["removed"]:
        violations.append(
            {"path": removed.get("path"), "reason": "durable_evidence_removed"}
        )
    for item in delta["modified"]:
        relative = str(item.get("path") or "")
        previous = item.get("before") or {}
        current = item.get("after") or {}
        # Shared provider cooldown is mutable operational state, not run evidence.
        if relative == "brain/model_health/provider_rate_limit_state.json":
            continue
        if (
            relative.endswith(".jsonl")
            and current.get("type") == "file"
            and int(current.get("size_bytes") or 0)
            >= int(previous.get("size_bytes") or 0)
        ):
            path = root / relative
            if (
                path.is_file()
                and _prefix_sha256(path, int(previous.get("size_bytes") or 0))
                == str(previous.get("sha256") or "")
            ):
                continue
        violations.append(
            {
                "path": relative,
                "reason": "preexisting_durable_evidence_rebuilt_not_appended",
            }
        )
    written = bool(delta["created"] or delta["modified"])
    return {
        "written": written,
        "additive": written and not violations,
        "violations": violations,
        "delta": delta,
    }


def _copy_protected_source(snapshot: dict[str, Any], source_root: Path, checkout: Path) -> None:
    for entry in snapshot.get("entries") or []:
        relative = Path(str(entry.get("path") or ""))
        source = source_root / relative
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("type") == "symlink":
            os.symlink(str(entry.get("target") or ""), target)
        elif entry.get("type") == "directory":
            target.mkdir(parents=True, exist_ok=True)
        elif entry.get("type") == "file":
            shutil.copy2(source, target)


def _load_json(path_text: str) -> dict[str, Any]:
    path = Path(str(path_text or ""))
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    return payload if isinstance(payload, dict) else {}


def _artifact_record(path_text: str, artifact_type: str) -> dict[str, Any]:
    path = Path(str(path_text or ""))
    if not path.is_absolute():
        path = config.ROOT / path
    if not path.is_file():
        return {"type": artifact_type, "path": str(path), "exists": False, "sha256": ""}
    return {
        "type": artifact_type,
        "path": portable_repo_path(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _refresh_artifact_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in records:
        artifact_type = str(item.get("type") or "artifact")
        record = _artifact_record(str(item.get("path") or ""), artifact_type)
        refreshed.append(record)
        if not record.get("exists") or not record.get("sha256"):
            failures.append(
                {
                    "type": artifact_type,
                    "path": str(record.get("path") or ""),
                    "reason": "artifact_missing_at_report_finalization",
                }
            )
    return refreshed, {
        "verified_at_report_finalization": not failures,
        "failures": failures,
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    integrity = payload.get("protected_source") or {}
    verification = payload.get("verification") or {}
    return "\n".join(
        [
            f"# Autonomous Engineer Run: {payload.get('report_id', '')}",
            "",
            f"- Task: `{payload.get('task_id', '')}`",
            f"- Status: `{payload.get('status', '')}`",
            f"- Promotion: `{payload.get('promotion_status', '')}`",
            f"- Protected source byte-identical: `{str(integrity.get('byte_identical', False)).lower()}`",
            f"- Durable evidence changed: `{str((payload.get('durable_state') or {}).get('changed', False)).lower()}`",
            f"- Scratch disposed: `{str((payload.get('scratch_checkout') or {}).get('disposed', False)).lower()}`",
            f"- Verification passed: `{str(verification.get('verification_passed', False)).lower()}`",
            f"- Standalone reconstruction check: `{str((payload.get('report_validation') or {}).get('ok', False)).lower()}`",
            "",
            "The JSON report embeds the envelope, evidence excerpts, claims/actions, diff, "
            "sandbox Apply record, verification argv/output/exit codes, refusals, and hash scopes.",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def validate_autonomous_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate reconstruction using only the supplied report object."""

    failures: list[str] = []
    if payload.get("schema") != AUTONOMOUS_REPORT_SCHEMA:
        failures.append("invalid_report_schema")
    envelope = payload.get("envelope_snapshot")
    if not isinstance(envelope, dict) or not envelope.get("task_id"):
        failures.append("missing_envelope_snapshot")
    else:
        integrity = envelope.get("integrity") or {}
        attribution = (envelope.get("declaration") or {}).get(
            "consent_attribution"
        ) or {}
        if (
            integrity.get("algorithm") != "hmac-sha256"
            or not integrity.get("hmac_sha256")
            or not integrity.get("canonical_payload_sha256")
        ):
            failures.append("missing_envelope_integrity")
        if attribution.get("actor") not in {
            "human",
            "human_via_agent",
            "unknown",
        }:
            failures.append("missing_consent_attribution")
        execution = envelope.get("execution") or {}
        if (
            execution.get("real_working_tree_write") is not False
            or execution.get("in_run_expansion") is not False
            or execution.get("behavior_promotion") is not False
        ):
            failures.append("unsafe_envelope_execution_authority")
    protected = payload.get("protected_source") or {}
    inventory = protected.get("hash_inventory")
    if not isinstance(inventory, list) or _entries_digest(inventory) != protected.get(
        "before_sha256"
    ):
        failures.append("protected_source_inventory_invalid")
    if (
        protected.get("before_sha256") != protected.get("after_sha256")
        or protected.get("byte_identical") is not True
    ):
        failures.append("protected_source_hash_mismatch")
    source_delta = protected.get("end_state_delta") or {}
    if any(source_delta.get(key) for key in ("created", "modified", "removed")):
        failures.append("protected_source_delta_not_empty")
    if not (protected.get("exclusions") or {}).get(
        "protected_behavior_prefixes"
    ):
        failures.append("protected_behavior_scope_missing")
    durable = payload.get("durable_state") or {}
    delta = durable.get("delta") or {}
    accumulation = durable.get("accumulation") or {}
    if (
        not durable.get("changed")
        or not any(delta.get(key) for key in ("created", "modified"))
        or delta.get("removed")
        or accumulation.get("written") is not True
        or accumulation.get("additive") is not True
        or accumulation.get("violations")
    ):
        failures.append("missing_durable_state_delta")
    child_integrity = payload.get("child_artifact_integrity") or {}
    if child_integrity.get("verified_at_report_finalization") is not True or child_integrity.get(
        "failures"
    ):
        failures.append("child_artifact_hashes_not_final")
    patch = payload.get("patch") or {}
    if payload.get("status") == "verified_pending_human_acceptance":
        unified_diff = str(patch.get("unified_diff") or "")
        diff_evidence = payload.get("diff_evidence") or {}
        encoded_diff = unified_diff.encode("utf-8")
        if not unified_diff.strip():
            failures.append("missing_diff")
        if (
            diff_evidence.get("sha256")
            != hashlib.sha256(encoded_diff).hexdigest()
            or diff_evidence.get("size_bytes") != len(encoded_diff)
            or diff_evidence.get("unified_diff") != unified_diff
        ):
            failures.append("diff_evidence_invalid")
        plan = payload.get("plan") or {}
        claims_list = [
            item
            for item in plan.get("material_claims") or []
            if isinstance(item, dict)
        ]
        actions_list = [
            item
            for item in plan.get("plan_actions") or []
            if isinstance(item, dict)
        ]
        claims = {
            str(item.get("claim_id") or "")
            for item in claims_list
            if item.get("claim_id")
        }
        evidence_ids = {
            str(item.get("evidence_id") or "")
            for item in (payload.get("evidence_catalog") or [])
            if isinstance(item, dict)
        }
        claim_evidence = {
            str(evidence_id)
            for item in claims_list
            for evidence_id in item.get("evidence_ids") or []
        }
        if (
            not claims
            or not claim_evidence
            or not evidence_ids
            or not claim_evidence <= evidence_ids
        ):
            failures.append("claim_references_missing_evidence")
        if not actions_list or any(
            not set(map(str, item.get("claim_ids") or [])) <= claims
            or not item.get("claim_ids")
            for item in actions_list
        ):
            failures.append("plan_actions_not_grounded")
        traces = patch.get("patch_claim_trace") or []
        if not traces or any(
            not item.get("claim_ids")
            or not set(map(str, item.get("claim_ids") or [])) <= claims
            for item in traces
            if isinstance(item, dict)
        ):
            failures.append("incomplete_claim_hunk_trace")
        retrievals = [
            item
            for item in payload.get("retrievals") or []
            if isinstance(item, dict)
        ]
        if not retrievals:
            failures.append("missing_retrieval_evidence")
        for retrieval in retrievals:
            if retrieval.get("status") not in {"complete", "partial"}:
                failures.append("retrieval_not_complete")
            results = [
                item
                for item in retrieval.get("results") or []
                if isinstance(item, dict)
            ]
            if not results:
                failures.append("retrieval_results_missing")
            for result in results:
                if len(str(result.get("sha256") or "")) != 64:
                    failures.append("retrieval_file_hash_missing")
                excerpts = [
                    item
                    for item in result.get("excerpts") or []
                    if isinstance(item, dict)
                ]
                if not excerpts or any(
                    len(str(item.get("sha256") or "")) != 64
                    for item in excerpts
                ):
                    failures.append("retrieval_excerpt_hash_missing")
        verification = payload.get("verification") or {}
        results = verification.get("results") or []
        if (
            not verification.get("verification_passed")
            or verification.get("verification_incomplete")
            or verification.get("allowed_commands_enforced") is not True
            or verification.get("sealed_command_specs_enforced") is not True
            or verification.get("required_commands")
            != verification.get("commands_run")
            or not results
        ):
            failures.append("verification_not_proven")
        if any(
            item.get("status") != "pass"
            or "exit_code" not in item
            or not item.get("argv")
            or not item.get("cwd")
            or len(str(item.get("stdout_sha256") or "")) != 64
            or len(str(item.get("stderr_sha256") or "")) != 64
            or item.get("secret_output_blocked") is True
            for item in results
            if isinstance(item, dict)
        ):
            failures.append("verification_result_incomplete")
        review = payload.get("review") or {}
        if (
            review.get("compile_passed") is not True
            or review.get("tests_passed") is not None
            or "Tests passed." in (review.get("what_worked") or [])
        ):
            failures.append("review_overstates_verification")
        events = payload.get("events")
        refusals = payload.get("refusals")
        if not isinstance(events, list) or not events:
            failures.append("event_journal_missing")
        if not isinstance(refusals, list) or any(
            not isinstance(item, dict) or not str(item.get("detail") or "").strip()
            for item in refusals or []
        ):
            failures.append("refusal_reason_missing")
    scratch = payload.get("scratch_checkout") or {}
    if not scratch.get("disposed"):
        failures.append("scratch_not_disposed")
    if scratch.get("inside_real_repository") is not False:
        failures.append("scratch_inside_real_repository")
    return {"ok": not failures, "failed_rules": sorted(set(failures))}


def engineer_autonomous_run(task_id: str) -> dict[str, Any]:
    """Run Plan -> Context -> Retrieve -> Replan -> Patch -> scratch Apply -> Verify."""

    task_key = str(task_id or "").strip()
    report_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + "_engineer_autonomous"
    # Snapshot behavior/source before any initializer is allowed to create state.
    source_before = _tree_snapshot(config.ROOT, evidence_only=False)
    state_before = _tree_snapshot(config.ROOT, evidence_only=True)
    config.ensure_harness()
    report_json = config.ENGINEER_AUTONOMOUS_RUNS_DIR / f"{report_id}.json"
    report_md = config.ENGINEER_AUTONOMOUS_RUNS_DIR / f"{report_id}.md"
    journal_path = config.ENGINEER_AUTONOMOUS_RUNS_DIR / f"{report_id}_events.jsonl"
    events: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    child_artifacts: list[dict[str, Any]] = []
    scratch_path: Path | None = None
    scratch_namespace = hashlib.sha256(
        str(config.ROOT.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    scratch_parent = (
        Path(tempfile.gettempdir())
        / "companybrain_engineer_autonomous"
        / scratch_namespace
    ).resolve()
    scratch_snapshot = {}
    scratch_after = {}
    # "Disposed" is vacuously true until a scratch checkout is actually created.
    scratch_disposed = True
    cleanup_error = ""
    envelope_snapshot: dict[str, Any] = {}
    plan_payload: dict[str, Any] = {}
    replan_payload: dict[str, Any] = {}
    patch_payload: dict[str, Any] = {}
    apply_payload: dict[str, Any] = {}
    verification_payload: dict[str, Any] = {}
    review_payload: dict[str, Any] = {}
    retrieval_payloads: list[dict[str, Any]] = []
    status = "refused"

    def event(stage: str, event_status: str, detail: str, **metadata: Any) -> None:
        row = {
            "sequence": len(events) + 1,
            "stage": stage,
            "status": event_status,
            "detail": str(detail)[:1800],
            "metadata": metadata,
            "recorded_at": now_tz_iso(),
        }
        events.append(row)
        append_jsonl(journal_path, row)
        if event_status in {"refused", "error", "blocked"}:
            refusals.append(row)

    try:
        # Load once without the time gate so an expired refusal can still embed the
        # authentic envelope, then enforce active validity immediately.
        envelope_snapshot = load_task_envelope(task_key, require_unexpired=False)
        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        attempt_dir = config.ENGINEER_ENVELOPES_DIR / "attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_path = attempt_dir / f"{task_key}.json"
        legacy_run_paths = [
            config.ENGINEER_RUNS_DIR / f"{task_key}.json",
            config.ENGINEER_J_SPACE_TASKS_DIR / task_key / "manifest.json",
        ]
        if attempt_path.exists() or any(path.exists() for path in legacy_run_paths):
            raise RuntimeError(
                "This immutable envelope already has an autonomous attempt; "
                "declare a new task envelope for another run."
            )
        try:
            with attempt_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "companybrain.engineer.autonomous_attempt.v1",
                            "task_id": task_key,
                            "report_id": report_id,
                            "started_at": now_tz_iso(),
                            "one_envelope_one_attempt": True,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
        except FileExistsError as exc:
            raise RuntimeError(
                "This immutable envelope already has an autonomous attempt; "
                "declare a new task envelope for another run."
            ) from exc
        event("envelope", "ok", "immutable envelope HMAC, task binding, and expiry validated")

        targets = write_targets(active_envelope)
        commands = verification_commands(active_envelope)
        sealed_specs = verification_command_specs(active_envelope)
        task_text = str((active_envelope.get("task") or {}).get("request") or "")
        envelope_budgets = dict(active_envelope.get("budgets") or {})
        plan_payload = engineer_plan(
            task_text,
            selected_files=targets,
            project="Envelope-bound autonomous Engineer run",
            use_approved_lessons=True,
            allow_web=False,
            retrieval_roots=list((active_envelope.get("scope") or {}).get("allowed_roots") or []),
            retrieval_intents=list((active_envelope.get("scope") or {}).get("allowed_intents") or []),
            task_id=task_key,
            task_envelope=active_envelope,
            max_chars_per_call=envelope_budgets.get("max_chars_per_call"),
        )
        child_artifacts.append(_artifact_record(str(plan_payload.get("jsonPath") or ""), "engineer_plan"))
        event("plan", "ok", str(plan_payload.get("output") or "plan saved"), run_id=task_key)

        checked = engineer_check_run(task_key, write=True)
        if checked.get("checkerStatus") == "fail":
            raise RuntimeError(
                "Autonomous run blocked by deterministic checker: "
                + ", ".join(checked.get("failedRules") or [])
            )
        event("check", "ok", f"checker={checked.get('checkerStatus')}")

        event(
            "context",
            "ok",
            "Repository map skipped; exact write targets are retrieved only through "
            "the envelope-budgeted broker.",
        )

        if "read_exact_file" not in set((active_envelope.get("scope") or {}).get("allowed_intents") or []):
            raise RuntimeError(
                "Autonomous grounding requires read_exact_file inside the declared envelope."
            )
        budget = active_envelope.get("budgets") or {}
        for batch_index in range(0, len(targets), 12):
            batch = targets[batch_index : batch_index + 12]
            request_id = f"autonomous_context_{(batch_index // 12) + 1:02d}"
            items = []
            allowed_roots = list((active_envelope.get("scope") or {}).get("allowed_roots") or [])
            for item_index, target in enumerate(batch, start=1):
                target_path = (config.ROOT / target).resolve()
                matching_roots = [
                    root
                    for root in allowed_roots
                    if (config.ROOT / root).resolve() == target_path
                    or (config.ROOT / root).resolve() in target_path.parents
                ]
                if not matching_roots:
                    raise RuntimeError(f"Write target is outside declared retrieval roots: {target}")
                items.append(
                    {
                        "item_id": f"target_{item_index:02d}",
                        "intent": "read_exact_file",
                        "reason": "Ground the declared write target before sandbox Apply.",
                        "requested_roots": matching_roots,
                        "path": target,
                        "estimated_budget": {
                            "max_files": min(1, int(budget.get("max_files_per_call") or 1)),
                            "max_excerpts": min(1, int(budget.get("max_excerpts_per_call") or 1)),
                            "max_chars": int(budget.get("max_chars_per_call") or 8000),
                            "max_file_bytes": int(budget.get("max_file_bytes") or 256000),
                        },
                        "budget_reason": "Declared once in the immutable task envelope.",
                    }
                )
            requested = engineer_context_request(
                task_key,
                json.dumps(
                    {
                        "request_id": request_id,
                        "task_id": task_key,
                        "requested_items": items,
                        "external_retrieval": False,
                    }
                ),
            )
            child_artifacts.append(_artifact_record(str(requested.get("jsonPath") or ""), "context_request"))
            if requested.get("status") != "ready":
                raise RuntimeError(
                    f"Envelope context request failed closed: status={requested.get('status')}"
                )
            executed = engineer_context_execute(task_key, request_id)
            retrievals = [
                item for item in executed.get("retrieval_results") or [] if isinstance(item, dict)
            ]
            retrieval_payloads.extend(retrievals)
            for item in retrievals:
                child_artifacts.append(_artifact_record(str(item.get("jsonPath") or ""), "repository_retrieval"))
            if not retrievals or any(not item.get("ok") for item in retrievals):
                raise RuntimeError("Envelope retrieval did not produce complete/partial evidence.")
            event("retrieve", "ok", f"retrieval artifacts={len(retrievals)}", request_id=request_id)

        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        event("envelope", "ok", "expiry and integrity revalidated before grounded replan")
        replan_payload = engineer_replan(task_key)
        child_artifacts.append(_artifact_record(str(replan_payload.get("jsonPath") or ""), "grounded_replan"))
        if not replan_payload.get("ok") or replan_payload.get("contextSufficiency") != "sufficient":
            raise RuntimeError(
                f"Grounded replan failed closed: checker={replan_payload.get('checkerStatus')}; "
                f"context={replan_payload.get('contextSufficiency')}"
            )
        event("replan", "ok", str(replan_payload.get("output") or "grounded replan saved"))

        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        event("envelope", "ok", "expiry and integrity revalidated before patch")
        patch_payload = engineer_patch(task_key, selected_files=targets)
        child_artifacts.append(_artifact_record(str(patch_payload.get("jsonPath") or ""), "engineer_patch"))
        if patch_payload.get("patch_status") != "proposed":
            raise RuntimeError(
                "Patch proposal failed closed: " + "; ".join(patch_payload.get("limitations") or [])
            )
        prepared_commands = [str(item) for item in patch_payload.get("verification_commands") or []]
        outside_commands = [item for item in prepared_commands if item not in set(commands)]
        if outside_commands:
            raise RuntimeError(
                "Patch requested verification outside the envelope: " + ", ".join(outside_commands)
            )
        for refused_command in patch_payload.get(
            "verification_commands_outside_envelope"
        ) or []:
            if isinstance(refused_command, dict):
                event(
                    "verification_scope",
                    "refused",
                    str(refused_command.get("reason") or "command outside envelope"),
                    command=str(refused_command.get("command") or ""),
                )
        event("patch", "ok", f"patch={patch_payload.get('patch_id')}")

        if (
            scratch_parent == config.ROOT.resolve()
            or config.ROOT.resolve() in scratch_parent.parents
            or scratch_parent in config.ROOT.resolve().parents
        ):
            raise RuntimeError(
                f"Refusing an autonomous scratch parent that overlaps the real "
                f"repository: {scratch_parent}"
            )
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_path = (
            scratch_parent
            / f"run_{secrets.token_hex(6)}"
        ).resolve()
        if scratch_path.parent != scratch_parent:
            raise RuntimeError(
                f"Autonomous scratch path escaped its parent: {scratch_path}"
            )
        scratch_path.mkdir(parents=False, exist_ok=False)
        scratch_disposed = False
        _copy_protected_source(source_before, config.ROOT, scratch_path)
        source_after_copy = _tree_snapshot(config.ROOT, evidence_only=False)
        scratch_snapshot = _tree_snapshot(scratch_path, evidence_only=False)
        if source_after_copy["sha256"] != source_before["sha256"]:
            raise RuntimeError("Real protected source changed while creating the disposable checkout.")
        if scratch_snapshot["sha256"] != source_before["sha256"]:
            raise RuntimeError("Disposable checkout bytes do not match the protected source snapshot.")
        event("scratch_checkout", "ok", "disposable code checkout copied and hash-matched")

        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        event("envelope", "ok", "expiry and integrity revalidated before sandbox Apply")
        authorization = {
            "kind": "task_envelope_sandbox",
            "task_id": task_key,
            "envelope_id": active_envelope.get("envelope_id"),
            "canonical_payload_sha256": (active_envelope.get("integrity") or {}).get(
                "canonical_payload_sha256"
            ),
            "real_working_tree_write": False,
        }
        apply_payload = engineer_apply_patch(
            str(patch_payload.get("patch_id") or ""),
            yes=False,
            allow_package_json=False,
            allow_delete=False,
            source_root=scratch_path,
            authorization=authorization,
            allowed_write_targets=targets,
        )
        child_artifacts.append(_artifact_record(str(apply_payload.get("jsonPath") or ""), "sandbox_apply"))
        if apply_payload.get("status") != "sandbox_applied":
            raise RuntimeError("Shared applier did not record a sandbox-only application.")
        scratch_after = _tree_snapshot(scratch_path, evidence_only=False)
        event("sandbox_apply", "ok", f"changed={len(apply_payload.get('changed_files') or [])}")

        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        event("envelope", "ok", "expiry and integrity revalidated before Verify")
        verification_payload = engineer_verify(
            str(apply_payload.get("applied_patch_id") or ""),
            run=True,
            save=True,
            source_root=scratch_path,
            allowed_commands=commands,
            required_commands=commands,
            sealed_command_specs=sealed_specs,
            authorization=authorization,
        )
        child_artifacts.append(_artifact_record(str(verification_payload.get("jsonPath") or ""), "verification"))
        if not verification_payload.get("verification_passed") or verification_payload.get(
            "verification_incomplete"
        ):
            raise RuntimeError(
                f"Sandbox verification did not pass completely: "
                f"passed={verification_payload.get('verification_passed')}; "
                f"incomplete={verification_payload.get('verification_incomplete')}"
            )
        event("verify", "ok", f"commands={len(verification_payload.get('commands_run') or [])}")
        active_envelope = load_task_envelope(task_key, require_unexpired=True)
        event(
            "envelope",
            "ok",
            "expiry, integrity, nonce locator, and sealed verification specs "
            "revalidated after Verify",
        )

        changed = [str(item) for item in apply_payload.get("changed_files") or []]
        files_correct = set(changed) <= set(targets) and bool(changed)
        verification_results = [
            item
            for item in verification_payload.get("results") or []
            if isinstance(item, dict)
        ]
        compile_passed = bool(verification_results) and all(
            str(item.get("command") or "").startswith("python -m py_compile ")
            and item.get("status") == "pass"
            for item in verification_results
        )
        review_payload = engineer_review(
            task_key,
            changed_files=changed,
            codex_summary="Patch was applied and verified only in the disposable checkout.",
            compile_passed=compile_passed,
            tests_passed=None,
            files_touched_correct=files_correct,
            manual_correction="",
            notes=(
                "Isolated py_compile passed; no test-class command ran. Real source "
                "promotion remains pending explicit human acceptance."
            ),
        )
        child_artifacts.append(_artifact_record(str(review_payload.get("jsonPath") or ""), "engineer_review"))
        event("review", "ok", str(review_payload.get("output") or "review saved"))
        load_task_envelope(task_key, require_unexpired=True)
        event("envelope", "ok", "final envelope revalidation passed before report status")
        status = "verified_pending_human_acceptance"
    except Exception as exc:  # noqa: BLE001
        event("run", "refused", f"{type(exc).__name__}: {exc}")
        if task_key:
            try:
                record_envelope_event(
                    task_key,
                    "autonomous_run",
                    "refused",
                    f"{type(exc).__name__}: {exc}",
                    details={"report_id": report_id},
                )
            except Exception:  # noqa: BLE001
                pass
    finally:
        if scratch_path is not None:
            try:
                resolved_scratch = scratch_path.resolve()
                if scratch_parent not in resolved_scratch.parents:
                    raise RuntimeError(
                        f"Refusing scratch cleanup outside {scratch_parent}: {resolved_scratch}"
                    )
                shutil.rmtree(resolved_scratch)
            except Exception as exc:  # noqa: BLE001
                cleanup_error = f"{type(exc).__name__}: {exc}"
            scratch_disposed = not scratch_path.exists()
            event(
                "scratch_cleanup",
                "ok" if scratch_disposed and not cleanup_error else "error",
                cleanup_error or "disposable checkout removed",
            )

    source_after = _tree_snapshot(config.ROOT, evidence_only=False)
    source_delta = _state_delta(source_before, source_after)
    source_identical = source_before["sha256"] == source_after["sha256"]
    if not source_identical:
        status = "unsafe_real_source_changed"
        event("protected_source", "error", "protected real source end-state hash changed")
    if not scratch_disposed:
        status = "scratch_cleanup_failed"
    state_after = _tree_snapshot(config.ROOT, evidence_only=True)
    accumulation = _evidence_accumulation(state_before, state_after, config.ROOT)
    if not accumulation["additive"]:
        status = "unsafe_durable_evidence_not_additive"
        event(
            "durable_state",
            "error",
            "Durable evidence was removed, rebuilt, or no new evidence was written.",
            violations=accumulation["violations"],
        )
        state_after = _tree_snapshot(config.ROOT, evidence_only=True)
        accumulation = _evidence_accumulation(state_before, state_after, config.ROOT)
    delta = accumulation["delta"]
    durable_changed = state_before["sha256"] != state_after["sha256"]

    manifest = {}
    evidence_catalog: list[dict[str, Any]] = []
    grounded_replan = _load_json(str(replan_payload.get("jsonPath") or ""))
    try:
        manifest = jspace.load_manifest(task_key)
        workspace, _, _ = jspace.paths(task_key)
        from engineer_grounding import build_evidence_catalog

        evidence_catalog = build_evidence_catalog(
            manifest, workspace, config.ROOT, task_text=task_text
        ).get("entries") or []
    except Exception:  # noqa: BLE001
        pass
    plan_embedded = _load_json(str(plan_payload.get("jsonPath") or ""))
    embedded_patch = _load_json(str(patch_payload.get("jsonPath") or "")) or patch_payload
    child_artifacts, child_artifact_integrity = _refresh_artifact_records(
        child_artifacts
    )
    unified_diff = str(embedded_patch.get("unified_diff") or "")
    unified_diff_bytes = unified_diff.encode("utf-8")
    report: dict[str, Any] = {
        "schema": AUTONOMOUS_REPORT_SCHEMA,
        "report_id": report_id,
        "task_id": task_key,
        "created_at": now_tz_iso(),
        "status": status,
        "promotion_status": "pending_human_acceptance" if status == "verified_pending_human_acceptance" else "not_eligible",
        "envelope_snapshot": {
            key: value
            for key, value in envelope_snapshot.items()
            if key not in {"path", "jsonPath", "output"}
        },
        "events": events,
        "refusals": refusals,
        "protected_source": {
            "hash_schema": PROTECTED_SOURCE_HASH_SCHEMA,
            "before_sha256": source_before["sha256"],
            "after_sha256": source_after["sha256"],
            "byte_identical": source_identical,
            "before_file_count": source_before["file_count"],
            "after_file_count": source_after["file_count"],
            "hash_inventory": source_before.get("entries") or [],
            "end_state_delta": source_delta,
            "exclusions": source_before.get("exclusions") or {},
            "limitation": "Equal end-state hashes do not prove the absence of a transient write.",
        },
        "durable_state": {
            "before_sha256": state_before["sha256"],
            "after_sha256_before_report": state_after["sha256"],
            "changed": durable_changed,
            "delta": delta,
            "accumulation": accumulation,
            "report_files_excluded_from_after_digest": [report_json.name, report_md.name],
        },
        "scratch_checkout": {
            "created": scratch_path is not None,
            "path_disclosed_for_local_audit": str(scratch_path) if scratch_path else "",
            "before_sha256": scratch_snapshot.get("sha256", ""),
            "after_sha256": scratch_after.get("sha256", ""),
            "disposed": scratch_disposed,
            "cleanup_error": cleanup_error,
            "inside_real_repository": bool(
                scratch_path
                and (
                    scratch_path == config.ROOT.resolve()
                    or config.ROOT.resolve() in scratch_path.parents
                )
            ),
            "dependencies_excluded": ["node_modules", ".venv*", ".next"],
        },
        "plan": plan_embedded,
        "j_space_manifest": manifest,
        "retrievals": retrieval_payloads,
        "evidence_catalog": evidence_catalog,
        "grounded_replan": grounded_replan,
        "patch": embedded_patch,
        "diff_evidence": {
            "unified_diff": unified_diff,
            "size_bytes": len(unified_diff_bytes),
            "sha256": hashlib.sha256(unified_diff_bytes).hexdigest(),
        },
        "sandbox_apply": _load_json(str(apply_payload.get("jsonPath") or "")) or apply_payload,
        "verification": _load_json(str(verification_payload.get("jsonPath") or "")) or verification_payload,
        "review": _load_json(str(review_payload.get("jsonPath") or "")) or review_payload,
        "child_artifacts": child_artifacts,
        "child_artifact_integrity": child_artifact_integrity,
        "limitations": [
            "Consent attribution is observable provenance, not identity authentication.",
            "In-repository evidence is not tamper-proof against an arbitrary same-user shell.",
            "Secret detection is heuristic rather than comprehensive DLP.",
            "Passing an isolated non-executing verifier is not proof of complete semantic correctness.",
            "The disposable path is not an OS sandbox. Autonomous envelopes therefore "
            "permit only sealed non-executing verification specs; repository tests/builds "
            "remain human or require a future external sandbox backend.",
        ],
    }
    report["report_validation"] = validate_autonomous_report(report)
    if status == "verified_pending_human_acceptance" and not report["report_validation"]["ok"]:
        report["status"] = "report_incomplete"
        report["promotion_status"] = "not_eligible"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_report_markdown(report), encoding="utf-8")
    record_envelope_event(
        task_key,
        "autonomous_run",
        report["status"],
        "autonomous report saved",
        details={
            "report_id": report_id,
            "source_byte_identical": source_identical,
            "durable_state_changed": durable_changed,
            "report_validation": report["report_validation"],
        },
    )
    return {
        **report,
        "ok": report["status"] == "verified_pending_human_acceptance",
        "path": str(report_md),
        "jsonPath": str(report_json),
        "output": (
            f"Autonomous run {report_id}: status={report['status']}; "
            f"source_unchanged={str(source_identical).lower()}; "
            f"evidence_written={str(durable_changed).lower()}; "
            f"promotion={report['promotion_status']}."
        ),
    }
