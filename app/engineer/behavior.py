"""Human-gated Engineer behavior promotion with exact rollback.

Behavior promotion is deliberately separate from ordinary patch Apply.  An
autonomous run may produce candidate lessons and evaluation evidence, but it must
never call the functions in this module.  Only a CLI surface that requires an
explicit ``--yes`` should expose them; MCP and autonomous dispatch must not.

The gate has three phases:

1. prove one candidate against held-out task ids disjoint from its source task;
2. append exactly that lesson and leave one exclusive active promotion pointer;
3. either validate the installed behavior with new held-out evidence or restore
   the exact pre-promotion bytes.

The rollback snapshot covers every file below ``prompts/``,
``canonical_behavior.md``, and ``approved_lessons.jsonl``.  Consent attribution is
recorded exactly as observed by :mod:`engineer.consent`; ``human_via_agent`` and
``unknown`` remain valid when ``--yes`` was explicit because attribution is
evidence, not authentication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

from . import config, consent
from .util import now_tz_iso, read_jsonl


HELD_OUT_EVALUATION_SCHEMA = "companybrain.engineer.held_out_behavior_evaluation.v1"
BEHAVIOR_PROMOTION_SCHEMA = "companybrain.engineer.behavior_promotion.v1"
BEHAVIOR_SNAPSHOT_SCHEMA = "companybrain.engineer.behavior_snapshot.v1"
ACTIVE_PROMOTION_SCHEMA = "companybrain.engineer.active_behavior_promotion.v1"
BEHAVIOR_TRANSITION_LOCK_SCHEMA = (
    "companybrain.engineer.behavior_transition_lock.v1"
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE_STATUSES = frozenset({"complete", "completed", "pass", "passed", "scored", "success", "validated"})


def _resolved(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = config.ROOT / candidate
    return candidate.resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def behavior_write_block_reason(path: Path | str, *, role: "config.RoleConfig" = None) -> str | None:
    """Return the behavior-state rule that forbids an autonomous write.

    The explicit Engineer paths come from ``RoleConfig``/``ENGINEER_FILES`` so the
    policy follows configured behavior inputs rather than a task-specific list.
    Generic employee patterns cover future roles as well.
    """

    role = role or config.ENGINEER
    target = _resolved(path)
    canonical = _resolved(role.harness_files["canonical_behavior"])
    approved = _resolved(role.harness_files["approved_lessons"])
    prompts_root = _resolved(role.prompts_dir)
    promotions_root = _resolved(config.ENGINEER_BEHAVIOR_PROMOTIONS_DIR)
    evaluations_root = _resolved(config.ENGINEER_BEHAVIOR_EVALUATIONS_DIR)

    if target == canonical:
        return "canonical_behavior_target_blocked"
    if target == approved:
        return "approved_lessons_target_blocked"
    if _inside(target, prompts_root):
        return "behavior_prompt_target_blocked"
    if _inside(target, promotions_root):
        return "behavior_promotion_state_target_blocked"
    if _inside(target, evaluations_root):
        return "behavior_evaluation_evidence_target_blocked"

    for key in sorted(config.ENGINEER_PROMPT_FILE_KEYS):
        configured = config.ENGINEER_FILES.get(key)
        if configured is not None and target == _resolved(configured):
            return "engineer_behavior_policy_target_blocked"

    try:
        relative_parts = [part.lower() for part in target.relative_to(config.ROOT.resolve()).parts]
    except ValueError:
        return None
    if len(relative_parts) >= 4 and relative_parts[:2] == ["brain_v2", "employees"]:
        employee_relative = relative_parts[3:]
        if employee_relative and employee_relative[0] == "prompts":
            return "behavior_prompt_target_blocked"
        if employee_relative and employee_relative[0] == "behavior_promotions":
            return "behavior_promotion_state_target_blocked"
        if employee_relative and employee_relative[-1] == "canonical_behavior.md":
            return "canonical_behavior_target_blocked"
        if employee_relative and employee_relative[-1] == "approved_lessons.jsonl":
            return "approved_lessons_target_blocked"
    return None


def _safe_id(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_ID_RE.fullmatch(text):
        raise RuntimeError(f"{label} must contain only letters, numbers, dot, underscore, or hyphen.")
    return text


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 8 hex chars (32 bits), matching the collision-avoidance suffix convention used
    # everywhere else in this codebase (jspace.py, util.py) - this is momentary,
    # single-process, single-write disambiguation, not a durable identifier, so it
    # doesn't need full uuid4().hex's 128 bits. Discovered 2026-08-18: the full
    # 32-char form pushed a real behavior-promotion snapshot path (nested under
    # behavior_promotions/<id>/snapshot/files/prompts/) past Windows' 260-char
    # MAX_PATH on a longer checkout path, causing a silent-looking FileNotFoundError
    # from path.open() despite the parent directory having just been created.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        _fsync_write(temporary, data, exclusive=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _exclusive_write_json(path: Path, payload: dict[str, Any]) -> None:
    _fsync_write(path, _json_bytes(payload), exclusive=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} does not exist as a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object.")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _candidate_identity_sha256(candidate: dict[str, Any]) -> str:
    """Hash candidate identity while excluding promotion-only bookkeeping."""

    ignored = {
        "status",
        "approved_at",
        "behavior_promotion_id",
        "held_out_evidence_sha256",
    }
    identity = {key: value for key, value in candidate.items() if key not in ignored}
    return _sha256_bytes(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validated_evaluation_artifact(
    value: object,
    *,
    label: str,
    run_dir: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Validate one evaluator-produced task/output artifact and its declared hash."""

    if not isinstance(value, dict):
        raise RuntimeError(f"Held-out evidence {label} must be an artifact object.")
    raw_path = str(value.get("path") or "").strip()
    declared_sha = str(value.get("sha256") or "").strip().lower()
    if not raw_path or not _SHA256_RE.fullmatch(declared_sha):
        raise RuntimeError(
            f"Held-out evidence {label} requires path and lowercase SHA-256."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = config.ROOT / path
    resolved = path.resolve(strict=False)
    if (
        not _inside(resolved, run_dir)
        or resolved == evidence_path
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise RuntimeError(
            f"Held-out evidence {label} must be a regular evaluator artifact "
            f"inside {run_dir}."
        )
    observed = _sha256_path(resolved)
    if observed != declared_sha:
        raise RuntimeError(
            f"Held-out evidence {label} hash mismatch: {observed} != {declared_sha}."
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Held-out evidence {label} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Held-out evidence {label} must be finite.")
    return number


def _first_value(mapping: dict[str, Any], names: tuple[str, ...]) -> object:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _score_from_mapping(mapping: dict[str, Any], arm: str) -> object:
    if arm == "baseline":
        names = ("baseline_score", "baseline", "control_score", "before_score", "score_baseline")
    else:
        names = ("candidate_score", "candidate", "treatment_score", "after_score", "score_candidate")
    value = _first_value(mapping, names)
    if value is not None:
        return value
    nested = mapping.get("scores")
    if isinstance(nested, dict):
        return _first_value(nested, names)
    return None


def _candidate_source_task_ids(candidate: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("source_run_id", "source_task_id", "task_id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            ids.add(value)
    for key in ("source_run_ids", "source_task_ids"):
        values = candidate.get(key)
        if isinstance(values, list):
            ids.update(str(value).strip() for value in values if str(value).strip())
    return ids


def _resolve_candidate_lesson(lesson_id: str) -> dict[str, Any]:
    lesson_key = _safe_id(lesson_id, "lesson_id")
    candidates = read_jsonl(config.ENGINEER_FILES["candidate_lessons"])
    selected = next(
        (
            item
            for item in candidates
            if isinstance(item, dict)
            and not item.get("parse_error")
            and str(item.get("lesson_id") or "") == lesson_key
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(f"Candidate lesson not found: {lesson_key}")
    if any(
        isinstance(item, dict) and str(item.get("lesson_id") or "") == lesson_key
        for item in read_jsonl(config.ENGINEER_FILES["approved_lessons"])
    ):
        raise RuntimeError(f"Lesson already approved: {lesson_key}")
    if any(
        isinstance(item, dict) and str(item.get("lesson_id") or "") == lesson_key
        for item in read_jsonl(config.ENGINEER_FILES["rejected_lessons"])
    ):
        raise RuntimeError(f"Lesson already rejected: {lesson_key}")
    return dict(selected)


def _evidence_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("task_results", "tasks", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("task_results", "tasks", "results"):
            value = evaluation.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _declared_task_ids(payload: dict[str, Any]) -> list[str]:
    for key in ("held_out_task_ids", "task_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    return []


def _declared_aggregate(payload: dict[str, Any]) -> tuple[object, object]:
    containers = [payload]
    for key in ("aggregate", "aggregate_scores", "scores", "summary"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append(value)
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        containers.append(evaluation)
        for key in ("aggregate", "aggregate_scores", "scores", "summary"):
            value = evaluation.get(key)
            if isinstance(value, dict):
                containers.append(value)
    for container in containers:
        baseline = _first_value(
            container,
            ("baseline_aggregate", "baseline_score", "baseline", "control_score", "score_baseline"),
        )
        candidate = _first_value(
            container,
            ("candidate_aggregate", "candidate_score", "candidate", "treatment_score", "score_candidate"),
        )
        if baseline is not None or candidate is not None:
            return baseline, candidate
    return None, None


def _phase_value(payload: dict[str, Any]) -> str:
    return str(
        _first_value(payload, ("phase", "evaluation_phase", "evaluation_type", "kind")) or ""
    ).strip().lower().replace("-", "_")


def validate_held_out_evaluation(
    candidate: dict[str, Any] | str,
    evidence_file: Path | str,
    *,
    expected_phase: str = "pre_promotion",
    promotion_id: str = "",
) -> dict[str, Any]:
    """Validate one durable, candidate-bound evaluator evidence bundle.

    Each paired task row must point to hash-matched task, baseline-output, and
    candidate-output artifacts in the evidence run directory. Scores remain an
    evaluator judgment; the gate makes that judgment attributable and replayable
    from exact bytes rather than accepting a bare score-only JSON assertion.
    """

    lesson = _resolve_candidate_lesson(candidate) if isinstance(candidate, str) else dict(candidate)
    lesson_id = _safe_id(lesson.get("lesson_id"), "candidate lesson_id")
    source_task_ids = _candidate_source_task_ids(lesson)
    if not source_task_ids:
        raise RuntimeError("Candidate lesson has no source task id; held-out disjointness cannot be proven.")

    evidence_path = Path(evidence_file)
    if not evidence_path.is_absolute():
        evidence_path = config.ROOT / evidence_path
    evidence_path = evidence_path.resolve(strict=False)
    evaluations_root = Path(config.ENGINEER_BEHAVIOR_EVALUATIONS_DIR).resolve(
        strict=False
    )
    if (
        not _inside(evidence_path, evaluations_root)
        or not evidence_path.is_file()
        or evidence_path.is_symlink()
    ):
        raise RuntimeError(f"Held-out evidence does not exist as a regular file: {evidence_path}")
    evidence_bytes = evidence_path.read_bytes()
    try:
        payload = json.loads(evidence_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Held-out evidence is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Held-out evidence must contain a JSON object.")
    if payload.get("schema") != HELD_OUT_EVALUATION_SCHEMA:
        raise RuntimeError(
            f"Held-out evidence schema must be {HELD_OUT_EVALUATION_SCHEMA}."
        )
    if payload.get("held_out") is not True:
        raise RuntimeError("Evidence must explicitly declare held_out=true.")
    if payload.get("complete") is not True:
        raise RuntimeError("Held-out evidence is incomplete.")
    status = str(payload.get("status") or "").strip().lower()
    if status not in _COMPLETE_STATUSES:
        raise RuntimeError(f"Held-out evidence status is not complete: {status}")

    producer = payload.get("producer")
    if not isinstance(producer, dict) or any(
        not str(producer.get(key) or "").strip()
        for key in ("name", "version", "evaluation_run_id", "generated_at")
    ):
        raise RuntimeError(
            "Held-out evidence requires producer name, version, "
            "evaluation_run_id, and generated_at."
        )
    run_id = _safe_id(producer.get("evaluation_run_id"), "evaluation_run_id")
    run_dir = evidence_path.parent.resolve(strict=False)
    if run_dir.parent != evaluations_root or run_dir.name != run_id:
        raise RuntimeError(
            "Held-out evidence must live at "
            "brain_v2/evals/engineer/behavior_evaluations/<evaluation_run_id>/."
        )

    evidence_lesson_id = str(
        _first_value(payload, ("candidate_lesson_id", "lesson_id")) or ""
    ).strip()
    if not evidence_lesson_id:
        nested_candidate = payload.get("candidate")
        if isinstance(nested_candidate, dict):
            evidence_lesson_id = str(nested_candidate.get("lesson_id") or "").strip()
    if evidence_lesson_id != lesson_id:
        raise RuntimeError(
            f"Held-out evidence candidate lesson mismatch: expected {lesson_id}, got {evidence_lesson_id or '<missing>'}."
        )
    expected_candidate_hash = _candidate_identity_sha256(lesson)
    if str(payload.get("candidate_lesson_sha256") or "").strip().lower() != expected_candidate_hash:
        raise RuntimeError("Held-out evidence candidate lesson hash does not match.")

    declared_promotion_id = str(payload.get("promotion_id") or "").strip()
    if promotion_id and declared_promotion_id != promotion_id:
        raise RuntimeError(
            f"Held-out evidence promotion mismatch: expected {promotion_id}, "
            f"got {declared_promotion_id or '<missing>'}."
        )
    phase = _phase_value(payload)
    if phase != expected_phase:
        raise RuntimeError(
            f"Held-out evidence phase mismatch: expected {expected_phase}, "
            f"got {phase or '<missing>'}."
        )

    rows = _evidence_rows(payload)
    if not rows:
        raise RuntimeError(
            "Held-out evidence requires paired task_results with evaluator artifacts."
        )
    task_ids: list[str] = []
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    evaluator_artifacts: list[dict[str, Any]] = []
    task_fingerprints: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        row_status = str(row.get("status") or "").strip().lower()
        if row_status not in _COMPLETE_STATUSES:
            raise RuntimeError(f"Held-out task result {index} is not complete: {row_status}")
        task_id = str(_first_value(row, ("task_id", "id")) or "").strip()
        if not task_id:
            raise RuntimeError(f"Held-out task result {index} is missing task_id.")
        task_artifact = _validated_evaluation_artifact(
            row.get("task_artifact"),
            label=f"task {task_id} definition",
            run_dir=run_dir,
            evidence_path=evidence_path,
        )
        baseline_artifact = _validated_evaluation_artifact(
            row.get("baseline_output_artifact"),
            label=f"task {task_id} baseline output",
            run_dir=run_dir,
            evidence_path=evidence_path,
        )
        candidate_artifact = _validated_evaluation_artifact(
            row.get("candidate_output_artifact"),
            label=f"task {task_id} candidate output",
            run_dir=run_dir,
            evidence_path=evidence_path,
        )
        baseline_value = _score_from_mapping(row, "baseline")
        candidate_value = _score_from_mapping(row, "candidate")
        if baseline_value is None or candidate_value is None:
            raise RuntimeError(f"Held-out task result {task_id} lacks paired baseline/candidate scores.")
        task_ids.append(task_id)
        baseline_scores.append(_number(baseline_value, f"task {task_id} baseline score"))
        candidate_scores.append(_number(candidate_value, f"task {task_id} candidate score"))
        task_fingerprints.append(
            {"task_id": task_id, "task_sha256": task_artifact["sha256"]}
        )
        evaluator_artifacts.append(
            {
                "task_id": task_id,
                "task_artifact": task_artifact,
                "baseline_output_artifact": baseline_artifact,
                "candidate_output_artifact": candidate_artifact,
            }
        )

    declared_ids = _declared_task_ids(payload)
    if declared_ids:
        if task_ids and set(declared_ids) != set(task_ids):
            raise RuntimeError("Held-out task_ids do not match task_results.")
        if not task_ids:
            task_ids = declared_ids
    if not task_ids:
        raise RuntimeError("Held-out evidence contains no task ids.")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Held-out evidence contains duplicate task ids.")
    task_set_sha256 = _sha256_bytes(
        json.dumps(
            sorted(task_fingerprints, key=lambda item: item["task_id"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if str(payload.get("task_set_sha256") or "").strip().lower() != task_set_sha256:
        raise RuntimeError("Held-out evidence task_set_sha256 does not match task artifacts.")
    overlap = sorted(set(task_ids) & source_task_ids)
    if overlap:
        raise RuntimeError(
            "Held-out task ids overlap the candidate lesson source task(s): " + ", ".join(overlap)
        )

    computed_baseline: float | None = None
    computed_candidate: float | None = None
    if baseline_scores:
        computed_baseline = sum(baseline_scores) / len(baseline_scores)
        computed_candidate = sum(candidate_scores) / len(candidate_scores)
        if computed_candidate < computed_baseline:
            raise RuntimeError(
                "Candidate computed aggregate regresses below baseline "
                f"({computed_candidate} < {computed_baseline})."
            )

    declared_baseline_value, declared_candidate_value = _declared_aggregate(payload)
    declared_baseline: float | None = None
    declared_candidate: float | None = None
    if declared_baseline_value is not None or declared_candidate_value is not None:
        if declared_baseline_value is None or declared_candidate_value is None:
            raise RuntimeError("Held-out evidence must declare both baseline and candidate aggregates.")
        declared_baseline = _number(declared_baseline_value, "baseline aggregate")
        declared_candidate = _number(declared_candidate_value, "candidate aggregate")
        if declared_candidate < declared_baseline:
            raise RuntimeError(
                "Candidate declared aggregate regresses below baseline "
                f"({declared_candidate} < {declared_baseline})."
            )
    if computed_baseline is None and declared_baseline is None:
        raise RuntimeError("Held-out evidence contains no aggregate scores.")

    effective_baseline = declared_baseline if declared_baseline is not None else computed_baseline
    effective_candidate = declared_candidate if declared_candidate is not None else computed_candidate
    return {
        "schema": HELD_OUT_EVALUATION_SCHEMA,
        "ok": True,
        "lesson_id": lesson_id,
        "expected_phase": expected_phase,
        "declared_phase": phase,
        "producer": producer,
        "evaluation_run_id": run_id,
        "candidate_lesson_sha256": expected_candidate_hash,
        "task_ids": task_ids,
        "task_set_sha256": task_set_sha256,
        "evaluator_artifacts": evaluator_artifacts,
        "source_task_ids": sorted(source_task_ids),
        "task_count": len(task_ids),
        "baseline_aggregate": effective_baseline,
        "candidate_aggregate": effective_candidate,
        "computed_baseline_aggregate": computed_baseline,
        "computed_candidate_aggregate": computed_candidate,
        "declared_baseline_aggregate": declared_baseline,
        "declared_candidate_aggregate": declared_candidate,
        "candidate_non_regression": True,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256_bytes(evidence_bytes),
        "evidence_size_bytes": len(evidence_bytes),
        "validated_at": now_tz_iso(),
    }


def validate_held_out_evidence(
    candidate: dict[str, Any] | str,
    evidence_file: Path | str,
    *,
    expected_phase: str = "pre_promotion",
    promotion_id: str = "",
) -> dict[str, Any]:
    """Compatibility alias with the same fail-closed behavior."""

    return validate_held_out_evaluation(
        candidate,
        evidence_file,
        expected_phase=expected_phase,
        promotion_id=promotion_id,
    )


def _inventory(*, role: "config.RoleConfig" = None) -> dict[str, bytes | None]:
    role = role or config.ENGINEER
    inventory: dict[str, bytes | None] = {
        "canonical_behavior.md": None,
        "approved_lessons.jsonl": None,
    }
    fixed = {
        "canonical_behavior.md": Path(role.harness_files["canonical_behavior"]),
        "approved_lessons.jsonl": Path(role.harness_files["approved_lessons"]),
    }
    for logical, path in fixed.items():
        if path.is_symlink():
            raise RuntimeError(f"Behavior state path cannot be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise RuntimeError(f"Behavior state path is not a regular file: {path}")
        if path.is_file():
            inventory[logical] = path.read_bytes()

    prompts_root = Path(role.prompts_dir)
    if prompts_root.is_symlink():
        raise RuntimeError(f"Behavior prompts root cannot be a symlink: {prompts_root}")
    if prompts_root.exists() and not prompts_root.is_dir():
        raise RuntimeError(f"Behavior prompts root is not a directory: {prompts_root}")
    if prompts_root.is_dir():
        for path in sorted(prompts_root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise RuntimeError(f"Behavior prompt path cannot be a symlink: {path}")
            if path.is_file():
                relative = path.relative_to(prompts_root).as_posix()
                inventory[f"prompts/{relative}"] = path.read_bytes()
    return inventory


def _inventory_digest(inventory: dict[str, bytes | None]) -> str:
    digest = hashlib.sha256()
    digest.update(b"companybrain-behavior-state-v1\0")
    for logical in sorted(inventory):
        encoded = logical.encode("utf-8")
        data = inventory[logical]
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        if data is None:
            digest.update(b"\0missing\0")
        else:
            digest.update(b"\0file\0")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def _behavior_digest(*, role: "config.RoleConfig" = None) -> str:
    return _inventory_digest(_inventory(role=role or config.ENGINEER))


def _snapshot_behavior(snapshot_dir: Path, *, role: "config.RoleConfig" = None) -> dict[str, Any]:
    role = role or config.ENGINEER
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    files_dir = snapshot_dir / "files"
    inventory = _inventory(role=role)
    entries: list[dict[str, Any]] = []
    for logical in sorted(inventory):
        data = inventory[logical]
        source = (
            Path(role.prompts_dir) / logical.removeprefix("prompts/")
            if logical.startswith("prompts/")
            else Path(role.harness_files["canonical_behavior"])
            if logical == "canonical_behavior.md"
            else Path(role.harness_files["approved_lessons"])
        )
        entry: dict[str, Any] = {
            "path": logical,
            "source_path": str(source),
            "existed": data is not None,
            "size_bytes": len(data) if data is not None else 0,
            "sha256": _sha256_bytes(data) if data is not None else "",
            "snapshot_path": "",
            "mode": stat.S_IMODE(source.stat().st_mode) if data is not None else None,
        }
        if data is not None:
            snapshot_path = files_dir / logical
            _atomic_write_bytes(snapshot_path, data)
            entry["snapshot_path"] = str(snapshot_path.relative_to(snapshot_dir).as_posix())
        entries.append(entry)
    manifest = {
        "schema": BEHAVIOR_SNAPSHOT_SCHEMA,
        "captured_at": now_tz_iso(),
        "behavior_digest": _inventory_digest(inventory),
        "entries": entries,
    }
    _atomic_write_json(snapshot_dir / "manifest.json", manifest)
    return manifest


def _validated_snapshot(snapshot_dir: Path, expected_digest: str) -> dict[str, Any]:
    manifest = _load_json(snapshot_dir / "manifest.json", "Behavior snapshot manifest")
    if manifest.get("schema") != BEHAVIOR_SNAPSHOT_SCHEMA:
        raise RuntimeError("Behavior snapshot schema is unsupported.")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("Behavior snapshot entries are missing.")
    reconstructed: dict[str, bytes | None] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("Behavior snapshot entry is malformed.")
        logical = str(item.get("path") or "")
        if not logical or logical in reconstructed:
            raise RuntimeError("Behavior snapshot contains a missing or duplicate path.")
        if logical not in {"canonical_behavior.md", "approved_lessons.jsonl"} and not logical.startswith(
            "prompts/"
        ):
            raise RuntimeError(f"Behavior snapshot contains an unsupported path: {logical}")
        if item.get("existed") is True:
            relative = str(item.get("snapshot_path") or "")
            snapshot_path = (snapshot_dir / relative).resolve(strict=False)
            if not _inside(snapshot_path, snapshot_dir) or not snapshot_path.is_file() or snapshot_path.is_symlink():
                raise RuntimeError(f"Behavior snapshot file is missing or unsafe: {logical}")
            data = snapshot_path.read_bytes()
            if _sha256_bytes(data) != str(item.get("sha256") or ""):
                raise RuntimeError(f"Behavior snapshot hash mismatch: {logical}")
            if len(data) != int(item.get("size_bytes") or 0):
                raise RuntimeError(f"Behavior snapshot size mismatch: {logical}")
            reconstructed[logical] = data
        else:
            reconstructed[logical] = None
    digest = _inventory_digest(reconstructed)
    if digest != str(manifest.get("behavior_digest") or "") or digest != expected_digest:
        raise RuntimeError("Behavior snapshot aggregate digest mismatch.")
    return {"manifest": manifest, "inventory": reconstructed}


def _restore_snapshot(snapshot_dir: Path, expected_digest: str, *, role: "config.RoleConfig" = None) -> str:
    role = role or config.ENGINEER
    validated = _validated_snapshot(snapshot_dir, expected_digest)
    inventory: dict[str, bytes | None] = validated["inventory"]
    manifest: dict[str, Any] = validated["manifest"]
    entries_by_path = {
        str(item.get("path")): item for item in manifest.get("entries") or [] if isinstance(item, dict)
    }
    prompts_root = Path(role.prompts_dir)
    if prompts_root.is_symlink():
        raise RuntimeError("Cannot restore behavior through a symlinked prompts root.")
    prompts_root.mkdir(parents=True, exist_ok=True)

    desired_prompt_paths = {
        logical.removeprefix("prompts/")
        for logical, data in inventory.items()
        if logical.startswith("prompts/") and data is not None
    }
    current_prompt_files: list[Path] = []
    for path in sorted(prompts_root.rglob("*"), key=lambda item: item.as_posix(), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"Cannot safely restore over a symlinked prompt path: {path}")
        if path.is_file():
            current_prompt_files.append(path)
    for path in current_prompt_files:
        relative = path.relative_to(prompts_root).as_posix()
        if relative not in desired_prompt_paths:
            path.unlink()

    for logical, data in inventory.items():
        if not logical.startswith("prompts/") or data is None:
            continue
        destination = prompts_root / logical.removeprefix("prompts/")
        if destination.exists() and not destination.is_file():
            raise RuntimeError(f"Cannot restore prompt file over a non-file path: {destination}")
        _atomic_write_bytes(destination, data)
        mode = entries_by_path.get(logical, {}).get("mode")
        if isinstance(mode, int):
            try:
                destination.chmod(mode)
            except OSError:
                pass

    fixed_destinations = {
        "canonical_behavior.md": Path(role.harness_files["canonical_behavior"]),
        "approved_lessons.jsonl": Path(role.harness_files["approved_lessons"]),
    }
    for logical, destination in fixed_destinations.items():
        data = inventory.get(logical)
        if destination.is_symlink():
            raise RuntimeError(f"Cannot safely restore over a symlinked behavior path: {destination}")
        if destination.exists() and not destination.is_file():
            raise RuntimeError(f"Cannot restore behavior file over a non-file path: {destination}")
        if data is None:
            if destination.is_file():
                destination.unlink()
            continue
        _atomic_write_bytes(destination, data)
        mode = entries_by_path.get(logical, {}).get("mode")
        if isinstance(mode, int):
            try:
                destination.chmod(mode)
            except OSError:
                pass

    if prompts_root.is_dir():
        for path in sorted(
            (item for item in prompts_root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
            except OSError:
                pass
    restored_digest = _behavior_digest(role=role)
    if restored_digest != expected_digest:
        raise RuntimeError(
            f"Behavior rollback digest mismatch after restore: {restored_digest} != {expected_digest}"
        )
    return restored_digest


def _promotions_root() -> Path:
    return Path(config.ENGINEER_BEHAVIOR_PROMOTIONS_DIR)


def _active_path() -> Path:
    return _promotions_root() / "active.json"


def _transition_lock_path() -> Path:
    return _promotions_root() / ".transition.lock"


@contextmanager
def _behavior_transition_lock(
    action: str,
    target: str,
) -> Iterator[None]:
    """Allow at most one behavior-state transition across local processes.

    Exclusive file creation is supported on Windows and POSIX. A process crash
    deliberately leaves the lock behind: behavior changes fail closed until a
    human inspects the recorded owner instead of guessing that a transition was
    safe to resume.
    """

    root = _promotions_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _transition_lock_path()
    token = uuid.uuid4().hex
    payload = {
        "schema": BEHAVIOR_TRANSITION_LOCK_SCHEMA,
        "token": token,
        "action": action,
        "target": target,
        "pid": os.getpid(),
        "created_at": now_tz_iso(),
    }
    try:
        _exclusive_write_json(path, payload)
    except FileExistsError as exc:
        try:
            owner = _load_json(path, "Behavior transition lock")
        except RuntimeError:
            owner = {}
        owner_action = str(owner.get("action") or "unknown")
        owner_target = str(owner.get("target") or "unknown")
        raise RuntimeError(
            "Another behavior transition is already in progress "
            f"({owner_action} for {owner_target}); retry only after it completes."
        ) from exc

    try:
        yield
    finally:
        current = _load_json(path, "Behavior transition lock")
        if str(current.get("token") or "") != token:
            raise RuntimeError(
                "Behavior transition lock ownership changed; refusing unsafe lock release."
            )
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Behavior transition lock disappeared before its owner released it."
            ) from exc


def _serialized_behavior_transition(action: str):
    """Decorate a complete public behavior transition with the process lock."""

    def decorate(function):
        @wraps(function)
        def wrapped(target, *args, **kwargs):
            with _behavior_transition_lock(action, str(target)):
                return function(target, *args, **kwargs)

        return wrapped

    return decorate


def _promotion_dir(promotion_id: str) -> Path:
    return _promotions_root() / _safe_id(promotion_id, "promotion_id")


def _promotion_artifact_path(promotion_id: str) -> Path:
    return _promotion_dir(promotion_id) / "promotion.json"


def _load_active() -> dict[str, Any] | None:
    path = _active_path()
    if not path.exists():
        return None
    return _load_json(path, "Active behavior-promotion pointer")


def _reserve_active(promotion_id: str, artifact_path: Path) -> None:
    pointer = {
        "schema": ACTIVE_PROMOTION_SCHEMA,
        "promotion_id": promotion_id,
        "status": "reserving",
        "artifact_path": str(artifact_path),
        "created_at": now_tz_iso(),
    }
    try:
        _exclusive_write_json(_active_path(), pointer)
    except FileExistsError as exc:
        active = _load_active()
        active_id = str((active or {}).get("promotion_id") or "unknown")
        raise RuntimeError(
            f"Behavior promotion {active_id} is still active and awaiting validation or rollback."
        ) from exc


def _update_active(promotion_id: str, status: str) -> None:
    active = _load_active()
    if not active or str(active.get("promotion_id") or "") != promotion_id:
        raise RuntimeError(f"Active behavior-promotion pointer does not match {promotion_id}.")
    active["status"] = status
    active["updated_at"] = now_tz_iso()
    _atomic_write_json(_active_path(), active)


def _release_active(
    promotion_id: str,
    *,
    idempotent_cleanup: bool = False,
) -> bool:
    """Release the matching gate; missing state is an error by default."""

    active = _load_active()
    if active is None:
        if idempotent_cleanup:
            return False
        raise RuntimeError(
            f"Active behavior-promotion pointer is missing while releasing {promotion_id}."
        )
    if str(active.get("promotion_id") or "") != promotion_id:
        raise RuntimeError(
            f"Refusing to release active promotion {active.get('promotion_id')} while handling {promotion_id}."
        )
    try:
        _active_path().unlink()
    except FileNotFoundError as exc:
        if idempotent_cleanup:
            return False
        raise RuntimeError(
            f"Active behavior-promotion pointer disappeared while releasing {promotion_id}."
        ) from exc
    return True


def _new_promotion_id() -> str:
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:10]}_behavior_promotion"


def _record_consent(
    action: str,
    target: str,
    attribution: dict[str, Any],
    outcome: str,
    detail: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    consent.record_consent_event(
        action,
        target,
        attribution,
        outcome=outcome,
        detail=detail,
        metadata=metadata,
    )


def _require_yes(action: str, target: str, yes: bool, attribution: dict[str, Any]) -> None:
    if yes:
        _record_consent(
            action,
            target,
            attribution,
            "confirmed",
            "Explicit --yes was supplied; actor attribution was recorded without authentication inference.",
        )
        return
    _record_consent(
        action,
        target,
        attribution,
        "refused",
        "Explicit --yes confirmation was not supplied.",
    )
    raise RuntimeError(f"{action} requires explicit --yes confirmation.")


def _append_approved_lesson(
    candidate: dict[str, Any],
    promotion_id: str,
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = Path(config.ENGINEER_FILES["approved_lessons"])
    if path.is_symlink():
        raise RuntimeError("Approved lessons ledger cannot be a symlink.")
    before = path.read_bytes() if path.is_file() else b""
    if before and not before.endswith(b"\n"):
        raise RuntimeError("Approved lessons ledger is not newline-terminated; refusing an ambiguous append.")
    if any(item.get("parse_error") for item in read_jsonl(path)):
        raise RuntimeError("Approved lessons ledger contains malformed JSONL.")
    lesson_id = str(candidate.get("lesson_id") or "")
    if any(str(item.get("lesson_id") or "") == lesson_id for item in read_jsonl(path)):
        raise RuntimeError(f"Lesson already approved: {lesson_id}")
    approved = {
        **candidate,
        "status": "approved",
        "approved_at": now_tz_iso(),
        "behavior_promotion_id": promotion_id,
        "held_out_evidence_sha256": evidence["evidence_sha256"],
    }
    line = (json.dumps(approved, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, before + line)
    rows = read_jsonl(path)
    if len(rows) == 0 or str(rows[-1].get("behavior_promotion_id") or "") != promotion_id:
        raise RuntimeError("Approved lesson append could not be verified.")
    if sum(str(item.get("behavior_promotion_id") or "") == promotion_id for item in rows) != 1:
        raise RuntimeError("Approved lesson promotion was not appended exactly once.")
    return approved, _sha256_path(path)


def _persist_evaluation_evidence(
    promotion_dir: Path,
    filename: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Copy validated evidence bytes into the durable promotion record."""

    source = Path(str(evidence.get("evidence_path") or ""))
    data = source.read_bytes()
    expected = str(evidence.get("evidence_sha256") or "")
    observed = _sha256_bytes(data)
    if not expected or observed != expected:
        raise RuntimeError("Held-out evidence changed after validation.")
    destination = promotion_dir / filename
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or _sha256_path(destination) != observed
        ):
            raise RuntimeError(
                f"Durable held-out evidence copy already exists with different bytes: {destination}"
            )
    else:
        _fsync_write(destination, data, exclusive=True)
    return {
        "path": str(destination),
        "sha256": observed,
        "size_bytes": len(data),
    }


@_serialized_behavior_transition("engineer-promote-lesson")
def engineer_promote_lesson(
    lesson_id: str,
    evidence_file: Path | str,
    yes: bool = False,
) -> dict[str, Any]:
    """Promote one evaluated candidate and require later installed-behavior validation.

    This function is intentionally for an explicit human CLI command only.  It must
    not be registered in autonomous or MCP dispatch.
    """

    config.ensure_harness()
    lesson_key = _safe_id(lesson_id, "lesson_id")
    attribution = consent.capture_consent_attribution()
    action = "engineer-promote-lesson"
    _require_yes(action, lesson_key, yes, attribution)

    try:
        candidate = _resolve_candidate_lesson(lesson_key)
        evidence = validate_held_out_evaluation(
            candidate,
            evidence_file,
            expected_phase="pre_promotion",
        )
        if _load_active() is not None:
            active = _load_active() or {}
            raise RuntimeError(
                f"Behavior promotion {active.get('promotion_id', 'unknown')} is still active "
                "and awaiting validation or rollback."
            )

        promotion_id = _new_promotion_id()
        promotion_dir = _promotion_dir(promotion_id)
        artifact_path = _promotion_artifact_path(promotion_id)
        _promotions_root().mkdir(parents=True, exist_ok=True)
        _reserve_active(promotion_id, artifact_path)
        try:
            promotion_dir.mkdir(parents=False, exist_ok=False)
        except Exception:
            _release_active(promotion_id)
            raise

        mutation_started = False
        artifact: dict[str, Any] = {
            "schema": BEHAVIOR_PROMOTION_SCHEMA,
            "promotion_id": promotion_id,
            "lesson_id": lesson_key,
            "source_task_ids": evidence["source_task_ids"],
            "status": "preparing",
            "created_at": now_tz_iso(),
            "consent_attribution": attribution,
            "pre_promotion_evidence": evidence,
            "post_promotion_validation": None,
        }
        try:
            artifact["pre_promotion_evidence_copy"] = _persist_evaluation_evidence(
                promotion_dir,
                "pre_promotion_held_out_evidence.json",
                evidence,
            )
            snapshot_dir = promotion_dir / "snapshot"
            snapshot = _snapshot_behavior(snapshot_dir)
            artifact["snapshot_path"] = str(snapshot_dir)
            artifact["snapshot_manifest"] = snapshot
            artifact["pre_behavior_digest"] = snapshot["behavior_digest"]
            artifact["approved_lessons_pre_sha256"] = next(
                (
                    str(item.get("sha256") or "")
                    for item in snapshot["entries"]
                    if item.get("path") == "approved_lessons.jsonl"
                ),
                "",
            )
            _atomic_write_json(artifact_path, artifact)

            mutation_started = True
            approved_lesson, approved_post_sha256 = _append_approved_lesson(
                candidate,
                promotion_id,
                evidence,
            )
            post_digest = _behavior_digest()
            artifact.update(
                {
                    "status": "pending_post_promotion_validation",
                    "promoted_at": now_tz_iso(),
                    "approved_lesson": approved_lesson,
                    "approved_lessons_post_sha256": approved_post_sha256,
                    "post_behavior_digest": post_digest,
                    "active_pointer_path": str(_active_path()),
                }
            )
            _atomic_write_json(artifact_path, artifact)
            _update_active(promotion_id, "pending_post_promotion_validation")
        except Exception as exc:
            artifact["status"] = (
                "promotion_failed_requires_rollback" if mutation_started else "promotion_failed_before_mutation"
            )
            artifact["failure"] = {"error": str(exc)[:1200], "recorded_at": now_tz_iso()}
            try:
                _atomic_write_json(artifact_path, artifact)
            except Exception:
                pass
            if not mutation_started:
                _release_active(promotion_id)
            raise

        _record_consent(
            action,
            promotion_id,
            attribution,
            "completed",
            "Candidate lesson promoted and left pending post-promotion validation.",
            {"lesson_id": lesson_key, "evidence_sha256": evidence["evidence_sha256"]},
        )
        return {
            "ok": True,
            "promotion_id": promotion_id,
            "lesson_id": lesson_key,
            "status": "pending_post_promotion_validation",
            "consent_attribution": attribution,
            "approvedLesson": artifact["approved_lesson"],
            "pre_behavior_digest": artifact["pre_behavior_digest"],
            "post_behavior_digest": artifact["post_behavior_digest"],
            "path": str(artifact_path),
            "snapshotPath": str(promotion_dir / "snapshot"),
            "output": (
                f"Promoted lesson {lesson_key} as {promotion_id}; "
                "post-promotion held-out validation or rollback is required before another promotion."
            ),
        }
    except Exception as exc:
        _record_consent(action, lesson_key, attribution, "refused", str(exc))
        raise


@_serialized_behavior_transition("engineer-validate-behavior-promotion")
def engineer_validate_behavior_promotion(
    promotion_id: str,
    evidence_file: Path | str,
    yes: bool = False,
) -> dict[str, Any]:
    """Close one active promotion only after new held-out non-regression evidence."""

    config.ensure_harness()
    promotion_key = _safe_id(promotion_id, "promotion_id")
    attribution = consent.capture_consent_attribution()
    action = "engineer-validate-behavior-promotion"
    _require_yes(action, promotion_key, yes, attribution)
    try:
        active = _load_active()
        if not active or str(active.get("promotion_id") or "") != promotion_key:
            raise RuntimeError(f"Behavior promotion is not the active pending promotion: {promotion_key}")
        artifact_path = _promotion_artifact_path(promotion_key)
        artifact = _load_json(artifact_path, "Behavior promotion artifact")
        if artifact.get("status") != "pending_post_promotion_validation":
            raise RuntimeError(
                f"Behavior promotion {promotion_key} is not pending validation: {artifact.get('status')}"
            )
        expected_digest = str(artifact.get("post_behavior_digest") or "")
        current_digest = _behavior_digest()
        if not expected_digest or current_digest != expected_digest:
            raise RuntimeError(
                "Behavior state changed after promotion; validate is refused. Roll back or inspect the active promotion."
            )
        candidate = next(
            (
                item
                for item in read_jsonl(config.ENGINEER_FILES["approved_lessons"])
                if str(item.get("behavior_promotion_id") or "") == promotion_key
            ),
            None,
        )
        if not isinstance(candidate, dict):
            raise RuntimeError("Promoted lesson is missing from approved lessons.")
        validation = validate_held_out_evaluation(
            candidate,
            evidence_file,
            expected_phase="post_promotion_validation",
            promotion_id=promotion_key,
        )
        pre_evidence = artifact.get("pre_promotion_evidence") or {}
        if set(validation["task_ids"]) != set(pre_evidence.get("task_ids") or []):
            raise RuntimeError("Post-promotion validation must use the same held-out task ids.")
        pre_task_set_sha256 = str(
            pre_evidence.get("task_set_sha256") or ""
        ).strip().lower()
        post_task_set_sha256 = str(
            validation.get("task_set_sha256") or ""
        ).strip().lower()
        if (
            not pre_task_set_sha256
            or not post_task_set_sha256
            or post_task_set_sha256 != pre_task_set_sha256
        ):
            raise RuntimeError(
                "Post-promotion validation must use the identical nonempty held-out "
                "task_set_sha256 from the pre-promotion evidence."
            )
        if validation["evidence_sha256"] == pre_evidence.get("evidence_sha256"):
            raise RuntimeError("Post-promotion validation must provide new evidence, not reuse the pre-promotion file.")
        validation_copy = _persist_evaluation_evidence(
            _promotion_dir(promotion_key),
            "post_promotion_held_out_evidence.json",
            validation,
        )

        artifact.update(
            {
                "status": "validated",
                "validated_at": now_tz_iso(),
                "validation_consent_attribution": attribution,
                "post_promotion_validation": validation,
                "post_promotion_evidence_copy": validation_copy,
                "post_validation_behavior_digest": current_digest,
            }
        )
        _atomic_write_json(artifact_path, artifact)
        _release_active(promotion_key)
        _record_consent(
            action,
            promotion_key,
            attribution,
            "completed",
            "Installed behavior passed held-out non-regression validation.",
            {"evidence_sha256": validation["evidence_sha256"]},
        )
        return {
            "ok": True,
            "promotion_id": promotion_key,
            "lesson_id": artifact.get("lesson_id"),
            "status": "validated",
            "consent_attribution": attribution,
            "behavior_digest": current_digest,
            "validation": validation,
            "path": str(artifact_path),
            "output": f"Validated behavior promotion {promotion_key}; the active promotion gate is clear.",
        }
    except Exception as exc:
        _record_consent(action, promotion_key, attribution, "refused", str(exc))
        raise


@_serialized_behavior_transition("engineer-rollback-behavior")
def engineer_rollback_behavior(
    promotion_id: str,
    yes: bool = False,
) -> dict[str, Any]:
    """Restore exact pre-promotion behavior bytes and verify the aggregate digest."""

    config.ensure_harness()
    promotion_key = _safe_id(promotion_id, "promotion_id")
    attribution = consent.capture_consent_attribution()
    action = "engineer-rollback-behavior"
    _require_yes(action, promotion_key, yes, attribution)
    artifact_path = _promotion_artifact_path(promotion_key)
    try:
        artifact = _load_json(artifact_path, "Behavior promotion artifact")
        status = str(artifact.get("status") or "")
        if status == "rolled_back":
            raise RuntimeError(f"Behavior promotion already rolled back: {promotion_key}")
        if status not in {
            "pending_post_promotion_validation",
            "validated",
            "promotion_failed_requires_rollback",
        }:
            raise RuntimeError(f"Behavior promotion cannot be rolled back from status: {status}")

        active = _load_active()
        if (
            active is None
            and status
            in {
                "pending_post_promotion_validation",
                "promotion_failed_requires_rollback",
            }
        ):
            raise RuntimeError(
                "Active behavior-promotion pointer is missing; refusing rollback "
                "of a promotion that has not completed its lifecycle."
            )
        if active and str(active.get("promotion_id") or "") != promotion_key:
            raise RuntimeError(
                f"Another behavior promotion is active: {active.get('promotion_id')}; rollback it first."
            )
        approved_path = Path(config.ENGINEER_FILES["approved_lessons"])
        expected_approved_post = str(artifact.get("approved_lessons_post_sha256") or "")
        if expected_approved_post and (
            not approved_path.is_file() or _sha256_path(approved_path) != expected_approved_post
        ):
            raise RuntimeError(
                "Approved lessons changed after this promotion; exact rollback would remove later decisions."
            )

        pre_digest = str(artifact.get("pre_behavior_digest") or "")
        snapshot_path_text = str(artifact.get("snapshot_path") or "").strip()
        if not pre_digest or not snapshot_path_text:
            raise RuntimeError("Behavior promotion lacks a complete rollback snapshot.")
        snapshot_dir = Path(snapshot_path_text)
        before_rollback_digest = _behavior_digest()
        try:
            restored_digest = _restore_snapshot(snapshot_dir, pre_digest)
        except Exception as exc:
            artifact["status"] = "rollback_failed"
            artifact["rollback_failure"] = {"error": str(exc)[:1200], "recorded_at": now_tz_iso()}
            _atomic_write_json(artifact_path, artifact)
            raise

        artifact.update(
            {
                "status": "rolled_back",
                "rolled_back_at": now_tz_iso(),
                "rollback_consent_attribution": attribution,
                "before_rollback_behavior_digest": before_rollback_digest,
                "restored_behavior_digest": restored_digest,
                "rollback_digest_verified": restored_digest == pre_digest,
            }
        )
        _atomic_write_json(artifact_path, artifact)
        if active:
            _release_active(promotion_key)
        _record_consent(
            action,
            promotion_key,
            attribution,
            "completed",
            "Exact behavior snapshot restored and digest verified.",
            {"restored_behavior_digest": restored_digest},
        )
        return {
            "ok": True,
            "promotion_id": promotion_key,
            "lesson_id": artifact.get("lesson_id"),
            "status": "rolled_back",
            "consent_attribution": attribution,
            "before_rollback_behavior_digest": before_rollback_digest,
            "restored_behavior_digest": restored_digest,
            "rollback_digest_verified": True,
            "path": str(artifact_path),
            "output": f"Rolled back behavior promotion {promotion_key}; exact pre-promotion digest restored.",
        }
    except Exception as exc:
        _record_consent(action, promotion_key, attribution, "refused", str(exc))
        raise
