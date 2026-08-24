"""The per-task J-space: an append-only provenance record for one Engineer task.

The manifest is the audit spine. It stores hashes, permissions, budgets, checkpoints
and artifact pointers - never copies of selected source files. Two invariants matter:
permissions here are the only permissions (nothing in retrieved content can widen
them), and recorded hashes are revalidated before any write, so a task whose context
changed underneath it fails closed instead of patching stale code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from repository_retrieval import (
    DEFAULT_MAX_EXCERPTS as RETRIEVAL_DEFAULT_MAX_EXCERPTS,
    DEFAULT_MAX_FILE_BYTES as RETRIEVAL_DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES as RETRIEVAL_DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOTAL_CHARS as RETRIEVAL_DEFAULT_MAX_CHARS,
    MAX_FILES_HARD as RETRIEVAL_HARD_MAX_FILES,
    MAX_TOTAL_CHARS_HARD as RETRIEVAL_HARD_MAX_CHARS,
    SUPPORTED_INTENTS as RETRIEVAL_INTENTS,
    normalize_authorized_intents,
    normalize_authorized_roots,
)

from . import config
from .util import (
    file_provenance,
    now_iso,
    now_tz_iso,
    portable_repo_path,
    repo_path,
    safe_workspace_id,
    sha256_file,
)


def paths(run_id: str) -> tuple[Path, Path, Path]:
    """(workspace_dir, manifest.json, manifest.md) for a task id.

    Reads the tasks dir off `config` at call time so a caller that redirects it into
    a temp fixture is honoured.
    """
    workspace_id = safe_workspace_id(run_id)
    workspace = config.ENGINEER_J_SPACE_TASKS_DIR / workspace_id
    return workspace, workspace / "manifest.json", workspace / "manifest.md"


def pointer(run_id: str) -> dict:
    workspace, json_path, md_path = paths(run_id)
    return {
        "workspace_id": safe_workspace_id(run_id),
        "workspace_path": portable_repo_path(workspace),
        "manifest_path": portable_repo_path(json_path),
        "readable_manifest_path": portable_repo_path(md_path),
    }


def response_fields(run_id: str) -> dict:
    workspace, json_path, md_path = paths(run_id)
    if not json_path.exists():
        return {}
    return {
        "jSpaceWorkspaceId": safe_workspace_id(run_id),
        "jSpaceWorkspacePath": str(workspace),
        "jSpacePath": str(md_path),
        "jSpaceJsonPath": str(json_path),
    }


def artifact_ref(
    artifact_type: str,
    artifact_id: str,
    path: Path | str,
    json_path: Path | str | None = None,
) -> dict:
    return {
        "type": artifact_type,
        "artifact_id": artifact_id,
        "path": portable_repo_path(path),
        "json_path": portable_repo_path(json_path) if json_path else "",
    }


def manifest_markdown(manifest: dict) -> str:
    task = manifest.get("task") or {}
    sections = [
        f"# Engineer J-space: {manifest.get('workspace_id', '')}",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Current stage: `{manifest.get('current_stage', '')}`",
        f"- Created: {manifest.get('created_at', '')}",
        f"- Updated: {manifest.get('updated_at', '')}",
        "- Full selected source files copied here: no",
        "- Exact broker-returned excerpts stored here: yes, only when repository retrieval is performed",
        "",
        "## Task",
        "",
        str(task.get("request") or ""),
        "",
    ]
    for title, key in [
        ("Selected Context Provenance", "selected_context"),
        ("Harness Context Provenance", "harness_context"),
        ("Permissions", "permissions"),
        ("Budgets", "budgets"),
        ("Retrieval", "retrieval"),
        ("Grounding", "grounding"),
        ("Checkpoints", "checkpoints"),
        ("Artifact References", "artifact_refs"),
        ("Events", "events"),
    ]:
        sections.extend(
            [
                f"## {title}",
                "",
                "```json",
                json.dumps(manifest.get(key), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def write_manifest(manifest: dict) -> dict:
    workspace, json_path, md_path = paths(str(manifest.get("workspace_id") or ""))
    workspace.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = now_iso()
    json_temp = json_path.with_suffix(".json.tmp")
    md_temp = md_path.with_suffix(".md.tmp")
    json_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_temp.write_text(manifest_markdown(manifest), encoding="utf-8")
    json_temp.replace(json_path)
    md_temp.replace(md_path)
    return response_fields(str(manifest["workspace_id"]))


def load_manifest(run_id: str) -> dict:
    _, manifest_path, _ = paths(run_id)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def create_manifest(
    run_id: str,
    task: str,
    project: str,
    selected_records: list[dict],
    use_approved_lessons: bool,
    allow_web: bool,
    retrieval_roots: list[str] | None = None,
    retrieval_intents: list[str] | None = None,
    task_envelope: dict | None = None,
    selected_file_excerpt_chars_per_file: int | None = None,
) -> dict:
    """Create the workspace. Refuses to reuse an existing one so provenance is never merged."""
    workspace, json_path, _ = paths(run_id)
    if workspace.exists() or json_path.exists():
        raise RuntimeError(f"J-space workspace already exists: {workspace}")
    workspace.mkdir(parents=True, exist_ok=False)
    now = now_iso()
    selected_context = [
        file_provenance(
            Path(str(item["path"])),
            "human_selected_repo_file",
            str(item.get("input") or item.get("path") or ""),
        )
        for item in selected_records
        if item.get("exists") and Path(str(item.get("path") or "")).is_file()
    ]
    harness_context = [
        file_provenance(path, "engineer_harness_policy")
        for key, path in sorted(config.ENGINEER_FILES.items())
        if key in config.ENGINEER_PROMPT_FILE_KEYS and path.is_file()
    ]
    if use_approved_lessons and config.ENGINEER_FILES["approved_lessons"].is_file():
        harness_context.append(
            file_provenance(config.ENGINEER_FILES["approved_lessons"], "human_approved_lessons")
        )
    authorized_retrieval_roots = normalize_authorized_roots(retrieval_roots or [], config.ROOT)
    authorized_retrieval_intents = normalize_authorized_intents(
        retrieval_intents or (sorted(RETRIEVAL_INTENTS) if authorized_retrieval_roots else [])
    )
    checkpoints = []
    for stage in config.J_SPACE_STAGE_ORDER:
        status = "pending"
        if stage in {"task_intake", "context_selection"}:
            status = "complete"
        elif stage == "plan":
            status = "active"
        checkpoints.append({"stage": stage, "status": status, "updated_at": now if status != "pending" else ""})
    envelope_binding = (
        {
            "envelope_id": str(task_envelope.get("envelope_id") or ""),
            "task_id": str(task_envelope.get("task_id") or ""),
            "canonical_payload_sha256": str(
                (task_envelope.get("integrity") or {}).get("canonical_payload_sha256")
                or ""
            ),
            "expires_at": str(task_envelope.get("expires_at") or ""),
            "immutable": True,
            "in_run_expansion": False,
        }
        if isinstance(task_envelope, dict)
        else {}
    )
    envelope_budgets = (
        dict(task_envelope.get("budgets") or {})
        if isinstance(task_envelope, dict)
        else {}
    )
    from .context import resolve_excerpt_char_limit

    excerpt_chars = resolve_excerpt_char_limit(selected_file_excerpt_chars_per_file)
    manifest = {
        "schema": config.J_SPACE_SCHEMA,
        "schema_version": 1,
        "workspace_id": safe_workspace_id(run_id),
        "run_id": safe_workspace_id(run_id),
        "created_at": now,
        "updated_at": now,
        "status": "planning",
        "current_stage": "plan",
        "progress_index": config.J_SPACE_STAGE_ORDER.index("plan"),
        "task": {"request": task, "project": project},
        "task_envelope": envelope_binding,
        "data_classification": "repo_internal_explicit_context",
        "selected_context": selected_context,
        "harness_context": harness_context,
        "permissions": {
            "repo_read_scope": "explicit selected files plus versioned Engineer harness policy",
            "repo_write_mode": (
                "task_envelope_disposable_checkout_only"
                if envelope_binding
                else "human_confirmed_patch_apply_only"
            ),
            "verification_execution": (
                "task_envelope_exact_allowlist_in_disposable_checkout"
                if envelope_binding
                else "manual_opt_in_only"
            ),
            "external_retrieval": "not_integrated_stage_2" if allow_web else "disabled",
            "external_retrieval_requested": allow_web,
            "secrets": "denied",
            "canonical_memory_write": "denied",
            "lesson_promotion": "manual_only",
            "behavior_change": "never_autonomous",
            "background_execution": "denied",
            "repository_retrieval": {
                "mode": "pre_authorized" if authorized_retrieval_roots else "disabled",
                "allowed_roots": authorized_retrieval_roots,
                "allowed_intents": authorized_retrieval_intents,
                "external_retrieval": False,
            },
        },
        "budgets": {
            "selected_file_count": len(selected_context),
            "task_envelope": envelope_budgets,
            "selected_file_excerpt_chars_per_file": excerpt_chars,
            "planning_model_timeout_seconds": 45,
            "planning_model_max_output_tokens": 3000,
            "external_retrieval_requests": 0,
            "maintenance_iterations": 0,
            "background_processes": 0,
            "repository_retrieval_calls": int(
                envelope_budgets.get("repository_retrieval_calls", 18)
            ),
            "repository_retrieval_default_max_files_per_call": RETRIEVAL_DEFAULT_MAX_FILES,
            "repository_retrieval_default_max_excerpts_per_call": RETRIEVAL_DEFAULT_MAX_EXCERPTS,
            "repository_retrieval_default_max_chars_per_call": RETRIEVAL_DEFAULT_MAX_CHARS,
            "repository_retrieval_default_max_file_bytes": RETRIEVAL_DEFAULT_MAX_FILE_BYTES,
            "repository_retrieval_hard_max_files_per_call": RETRIEVAL_HARD_MAX_FILES,
            "repository_retrieval_hard_max_chars_per_call": RETRIEVAL_HARD_MAX_CHARS,
        },
        "retrieval": {
            "status": "not_performed",
            "sources": [],
            "requests": [],
            "external_retrieval": "disabled",
            "limitations": "Stage 3.1 supports deterministic repository-local retrieval only. External and semantic retrieval are disabled.",
        },
        "grounding": {
            "contract": "not_activated",
            "context_requests": [],
            "repository_map": {},
            "replans": [],
            "latest_context_sufficiency": "not_assessed",
            "external_retrieval": "disabled",
        },
        "checkpoints": checkpoints,
        "artifact_refs": [],
        "events": [
            {
                "event": "workspace_created",
                "stage": "plan",
                "status": "planning",
                "created_at": now,
                "actor": "engineer_harness",
            }
        ],
    }
    return write_manifest(manifest)


def snapshot_selected_plan_excerpts(run_id: str, selected_records: list[dict]) -> dict:
    """Store the exact excerpts the Plan model saw, so a plan can be replayed."""
    workspace, manifest_path, _ = paths(run_id)
    output_dir = workspace / "context" / "selected_excerpts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "plan_input.json"
    md_path = output_dir / "plan_input.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("Selected Plan excerpt snapshot already exists.")
    entries = []
    for item in selected_records:
        if not item.get("exists") or not item.get("content"):
            continue
        path = Path(str(item.get("path") or "")).resolve()
        content = str(item.get("content") or "")
        relative = path.relative_to(config.ROOT.resolve()).as_posix()
        full_hash = sha256_file(path)
        entries.append(
            {
                "evidence_id": f"selected-plan:{relative}:{full_hash[:12]}",
                "source_type": "human_selected_file_excerpt",
                "path": relative,
                "start_line": int(item.get("excerpt_start_line") or 1),
                "end_line": int(item.get("excerpt_end_line") or (content.count("\n") + 1)),
                "excerpt_truncated": bool(item.get("content_truncated")),
                "encoding": "utf-8",
                "full_file_sha256": full_hash,
                "excerpt_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "selection_reason": "human-selected Plan input",
                "trust": "untrusted_repository_evidence",
                "content": content,
            }
        )
    payload = {
        "schema": "companybrain.engineer.selected_plan_excerpts.v1",
        "task_id": run_id,
        "created_at": now_tz_iso(),
        "full_selected_files_copied": False,
        "entries": entries,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Selected Plan Excerpts\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("grounding", {})["selected_plan_excerpts"] = {
        "path": portable_repo_path(md_path),
        "json_path": portable_repo_path(json_path),
        "entry_count": len(entries),
        "full_selected_files_copied": False,
    }
    ref = artifact_ref("selected_plan_excerpts", "plan_input", md_path, json_path)
    manifest.setdefault("artifact_refs", []).append({**ref, "recorded_at": payload["created_at"]})
    manifest.setdefault("events", []).append(
        {
            "event": "selected_plan_excerpts_snapshotted",
            "stage": "context_selection",
            "entry_count": len(entries),
            "full_selected_files_copied": False,
            "created_at": payload["created_at"],
            "actor": "engineer_harness",
        }
    )
    write_manifest(manifest)
    return {
        "path": portable_repo_path(md_path),
        "json_path": portable_repo_path(json_path),
        "entry_count": len(entries),
    }


def update_manifest(
    run_id: str,
    stage: str,
    checkpoint_status: str,
    workspace_status: str,
    detail: str = "",
    artifact_reference: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Advance one checkpoint and append an event. Unknown stages raise."""
    if stage not in config.J_SPACE_STAGE_ORDER:
        raise RuntimeError(f"Unknown J-space checkpoint stage: {stage}")
    try:
        _, json_path, _ = paths(run_id)
    except RuntimeError:
        return {}
    if not json_path.exists():
        return {}
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    now = now_iso()
    checkpoint = next((item for item in manifest.get("checkpoints", []) if item.get("stage") == stage), None)
    if checkpoint is None:
        checkpoint = {"stage": stage}
        manifest.setdefault("checkpoints", []).append(checkpoint)
    checkpoint.update({"status": checkpoint_status, "updated_at": now})
    if detail:
        checkpoint["detail"] = detail
    if metadata:
        checkpoint["metadata"] = metadata
    if artifact_reference:
        refs = manifest.setdefault("artifact_refs", [])
        ref_key = (
            artifact_reference.get("type"),
            artifact_reference.get("artifact_id"),
            artifact_reference.get("path"),
        )
        existing_keys = {
            (item.get("type"), item.get("artifact_id"), item.get("path"))
            for item in refs
        }
        if ref_key not in existing_keys:
            refs.append({**artifact_reference, "recorded_at": now})
    stage_index = config.J_SPACE_STAGE_ORDER.index(stage)
    if stage_index >= int(manifest.get("progress_index", 0)):
        manifest["progress_index"] = stage_index
        manifest["current_stage"] = stage
        manifest["status"] = workspace_status
    manifest.setdefault("events", []).append(
        {
            "event": "checkpoint_updated",
            "stage": stage,
            "checkpoint_status": checkpoint_status,
            "workspace_status": workspace_status,
            "detail": detail,
            "metadata": metadata or {},
            "created_at": now,
            "actor": "engineer_harness",
        }
    )
    return write_manifest(manifest)


def context_hash_mismatches(
    run_id: str,
    *,
    source_root: Path | str | None = None,
) -> list[dict]:
    """Files whose on-disk hash no longer matches what this task recorded.

    Callers use a non-empty result to refuse a write: patching against context that
    changed underneath the plan is how a correct-looking diff corrupts a file.
    ``source_root`` lets the same recorded relative provenance be checked against a
    disposable checkout without moving J-space out of durable harness state.
    """
    if not run_id:
        return []
    try:
        _, manifest_path, _ = paths(run_id)
    except RuntimeError:
        return []
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: list[dict] = [
        {"source": "selected_context", "path": item.get("path"), "sha256": item.get("sha256")}
        for item in manifest.get("selected_context") or []
    ]
    expected.extend(
        {"source": "repository_retrieval", "path": item.get("path"), "sha256": item.get("sha256")}
        for item in (manifest.get("retrieval") or {}).get("sources") or []
    )
    root = Path(source_root).resolve() if source_root is not None else config.ROOT.resolve()
    mismatches = []
    seen = set()
    for item in expected:
        key = (item.get("path"), item.get("sha256"))
        if not item.get("path") or not item.get("sha256") or key in seen:
            continue
        seen.add(key)
        try:
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if path != root and root not in path.parents:
                raise RuntimeError(f"Refusing path outside source root: {path}")
        except (OSError, RuntimeError):
            mismatches.append({**item, "reason": "path_outside_repository", "actual_sha256": ""})
            continue
        actual = sha256_file(path) if path.is_file() else ""
        if actual != item["sha256"]:
            mismatches.append(
                {
                    **item,
                    "reason": "missing" if not actual else "sha256_changed",
                    "actual_sha256": actual,
                }
            )
    return mismatches


def refresh_selected_context_hashes(
    run_id: str,
    *,
    source_root: Path | str | None = None,
    only_paths: set[str] | None = None,
) -> list[dict]:
    """Re-record selected_context file hashes from disk for Apply after a passing dry-run."""

    if not run_id:
        return []
    try:
        _, manifest_path, md_path = paths(run_id)
    except RuntimeError:
        return []
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(source_root).resolve() if source_root is not None else config.ROOT.resolve()
    updated: list[dict] = []
    for item in manifest.get("selected_context") or []:
        relative = str(item.get("path") or "").replace("\\", "/")
        if not relative:
            continue
        if only_paths is not None and relative not in only_paths:
            continue
        try:
            path = (root / relative).resolve()
            if path != root and root not in path.parents:
                continue
        except OSError:
            continue
        if not path.is_file():
            continue
        actual = sha256_file(path)
        if actual and actual != item.get("sha256"):
            item["sha256"] = actual
            updated.append({"path": relative, "sha256": actual})
    if not updated:
        return []
    manifest["updated_at"] = now_tz_iso()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if md_path.is_file():
        md_path.write_text(manifest_markdown(manifest), encoding="utf-8")
    return updated


def record_retrieval(run_id: str, payload: dict, md_path: Path, json_path: Path) -> dict:
    _, manifest_path, _ = paths(run_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    now = now_tz_iso()
    retrieval = manifest.setdefault("retrieval", {})
    request_records = retrieval.setdefault("requests", [])
    status = str(payload.get("status") or "")
    counts_toward_budget = status in {"complete", "partial"}
    request_records.append(
        {
            "request_id": payload.get("request_id"),
            "status": payload.get("status"),
            "counts_toward_budget": counts_toward_budget,
            "intent": payload.get("intent"),
            "query": payload.get("query", ""),
            "allowed_roots": payload.get("allowed_roots") or [],
            "result_count": len(payload.get("results") or []),
            "rejected_count": len(payload.get("rejected_results") or []),
            "budget_used": payload.get("budget_used") or {},
            "budget": payload.get("budget") or {},
            "budget_expansion": payload.get("budget_expansion") or {},
            "budget_exhausted": payload.get("budget_exhausted", False),
            "artifact_path": portable_repo_path(md_path),
            "json_path": portable_repo_path(json_path),
            "created_at": payload.get("created_at") or now,
        }
    )
    sources = retrieval.setdefault("sources", [])
    source_keys = {(item.get("request_id"), item.get("path"), item.get("sha256")) for item in sources}
    for result in payload.get("results") or []:
        source = {
            "request_id": payload.get("request_id"),
            "path": result.get("path"),
            "sha256": result.get("sha256"),
            "selection_reason": result.get("selection_reason"),
            "excerpt_ranges": [
                {
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                    "sha256": item.get("sha256"),
                }
                for item in result.get("excerpts") or []
            ],
            "security_flags": result.get("security_flags") or [],
            "content_stored_in_j_space": True,
        }
        key = (source["request_id"], source["path"], source["sha256"])
        if key not in source_keys:
            sources.append(source)
            source_keys.add(key)
    retrieval["status"] = "performed" if payload.get("results") else str(payload.get("status") or "denied")
    retrieval["external_retrieval"] = "disabled"
    ref = artifact_ref(
        "repository_retrieval",
        str(payload.get("request_id") or ""),
        md_path,
        json_path,
    )
    refs = manifest.setdefault("artifact_refs", [])
    if not any(item.get("type") == ref["type"] and item.get("artifact_id") == ref["artifact_id"] for item in refs):
        refs.append({**ref, "recorded_at": now})
    manifest.setdefault("events", []).append(
        {
            "event": "repository_retrieval_recorded",
            "stage": manifest.get("current_stage", "context_selection"),
            "request_id": payload.get("request_id"),
            "status": payload.get("status"),
            "result_count": len(payload.get("results") or []),
            "budget_exhausted": payload.get("budget_exhausted", False),
            "security_flags": sorted(
                {
                    flag
                    for result in payload.get("results") or []
                    for flag in result.get("security_flags") or []
                }
            ),
            "created_at": now,
            "actor": "repository_retrieval_broker",
        }
    )
    return write_manifest(manifest)


def record_context_request(task_id: str, payload: dict, artifact_paths: dict) -> dict:
    _, manifest_path, _ = paths(task_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grounding = manifest.setdefault("grounding", {})
    records = grounding.setdefault("context_requests", [])
    record = {
        "request_id": payload.get("request_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary") or {},
        "path": portable_repo_path(artifact_paths["path"]),
        "json_path": portable_repo_path(artifact_paths["jsonPath"]),
        "updated_at": now_tz_iso(),
    }
    existing = next((item for item in records if item.get("request_id") == payload.get("request_id")), None)
    if existing:
        existing.update(record)
    else:
        records.append(record)
    grounding["external_retrieval"] = "disabled"
    ref = artifact_ref(
        "context_request",
        str(payload.get("request_id") or ""),
        artifact_paths["path"],
        artifact_paths["jsonPath"],
    )
    refs = manifest.setdefault("artifact_refs", [])
    if not any(item.get("type") == ref["type"] and item.get("artifact_id") == ref["artifact_id"] for item in refs):
        refs.append({**ref, "recorded_at": record["updated_at"]})
    manifest.setdefault("events", []).append(
        {
            "event": "context_request_recorded",
            "stage": "context_selection",
            "request_id": payload.get("request_id"),
            "status": payload.get("status"),
            "summary": payload.get("summary") or {},
            "created_at": record["updated_at"],
            "actor": "repository_retrieval_policy",
        }
    )
    return write_manifest(manifest)


def new_plan_id() -> str:
    """A collision-free plan id: timestamp, then uuid-suffixed if that is taken."""
    import uuid

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidates = [
        f"{stamp}_engineer_plan",
        f"{stamp}_{uuid.uuid4().hex[:8]}_engineer_plan",
    ]
    for candidate in candidates:
        workspace, _, _ = paths(candidate)
        if not workspace.exists() and not (config.ENGINEER_RUNS_DIR / f"{candidate}.json").exists():
            return candidate
    return f"{stamp}_{uuid.uuid4().hex}_engineer_plan"
