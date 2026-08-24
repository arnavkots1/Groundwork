"""Immutable, task-bound authority for unattended Engineer runs.

An envelope is declared once through the local CLI. It is not editable and cannot
be widened in-run. The JSON is authenticated with an HMAC whose nonce may live
outside the repository. The raw nonce is never embedded in the envelope JSON or
report; in the default mode its separate nonce file is repository-readable durable
state and is reported as such.

This is a capability boundary for the harness, not an identity system. A same-user
agent with arbitrary shell and nonce access can still invoke trusted CLI code.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from repository_retrieval import (
    DEFAULT_MAX_EXCERPTS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOTAL_CHARS,
    MAX_EXCERPTS_HARD,
    MAX_FILE_BYTES_HARD,
    MAX_FILES_HARD,
    MAX_TOTAL_CHARS_HARD,
    normalize_authorized_intents,
    normalize_authorized_roots,
)

from . import config
from .consent import capture_consent_attribution, record_consent_event
from .util import append_jsonl, now_tz_iso, repo_path, safe_workspace_id, sha256_file


ENVELOPE_SCHEMA = "companybrain.engineer.task_envelope.v1"
ENVELOPE_EVENT_SCHEMA = "companybrain.engineer.envelope_event.v1"
NONCE_PATH_ENV = "COMPANYBRAIN_ENVELOPE_NONCE_PATH"
NONCE_DIR_ENV = "COMPANYBRAIN_ENVELOPE_NONCE_DIR"
_BUDGET_KEYS = {
    "repository_retrieval_calls": (1, 100),
    "max_files_per_call": (1, MAX_FILES_HARD),
    "max_excerpts_per_call": (1, MAX_EXCERPTS_HARD),
    "max_chars_per_call": (1, MAX_TOTAL_CHARS_HARD),
    "max_file_bytes": (1, MAX_FILE_BYTES_HARD),
}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        unsigned["integrity"] = {
            key: value
            for key, value in integrity.items()
            if key not in {"hmac_sha256", "canonical_payload_sha256"}
        }
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _envelope_paths(task_id: str) -> tuple[Path, Path]:
    key = safe_workspace_id(task_id)
    return (
        config.ENGINEER_ENVELOPES_DIR / f"{key}.json",
        config.ENGINEER_ENVELOPES_DIR / f"{key}.md",
    )


def _nonce_path(task_id: str, locator: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    exact = os.environ.get(NONCE_PATH_ENV, "").strip()
    directory = os.environ.get(NONCE_DIR_ENV, "").strip()
    expected_mode = str((locator or {}).get("mode") or "")
    if locator is not None and expected_mode not in {
        "external_env_path",
        "external_env_directory",
        "repo_local_readable",
    }:
        raise RuntimeError("Task envelope nonce locator mode is invalid.")
    if expected_mode == "repo_local_readable":
        path = config.ENGINEER_ENVELOPES_DIR / "nonces" / f"{safe_workspace_id(task_id)}.nonce"
        mode = "repo_local_readable"
        env_var = ""
    elif expected_mode == "external_env_path":
        if not exact:
            raise RuntimeError(f"{NONCE_PATH_ENV} is required to validate this envelope.")
        path = Path(exact).expanduser().resolve()
        mode = "external_env_path"
        env_var = NONCE_PATH_ENV
    elif expected_mode == "external_env_directory":
        if not directory:
            raise RuntimeError(f"{NONCE_DIR_ENV} is required to validate this envelope.")
        path = Path(directory).expanduser().resolve() / f"{safe_workspace_id(task_id)}.nonce"
        mode = "external_env_directory"
        env_var = NONCE_DIR_ENV
    elif exact:
        path = Path(exact).expanduser().resolve()
        mode = "external_env_path"
        env_var = NONCE_PATH_ENV
    elif directory:
        path = Path(directory).expanduser().resolve() / f"{safe_workspace_id(task_id)}.nonce"
        mode = "external_env_directory"
        env_var = NONCE_DIR_ENV
    else:
        path = config.ENGINEER_ENVELOPES_DIR / "nonces" / f"{safe_workspace_id(task_id)}.nonce"
        mode = "repo_local_readable"
        env_var = ""
    try:
        outside_repo = config.ROOT.resolve() not in path.resolve().parents and path.resolve() != config.ROOT.resolve()
    except OSError:
        outside_repo = False
    return path, {
        "mode": mode,
        "env_var": env_var,
        "outside_repository": bool(outside_repo),
        "raw_nonce_recorded_in_repository": not bool(outside_repo),
        "raw_nonce_embedded_in_envelope_artifact": False,
    }


def _parse_expiry(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise RuntimeError("Envelope expiry is required.")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError("Envelope expiry must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("Envelope expiry must include a timezone offset.")
    if parsed <= datetime.now().astimezone():
        raise RuntimeError("Envelope expiry must be in the future.")
    return parsed


def _normalize_write_targets(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        resolved = repo_path(str(value))
        if not resolved.is_file():
            raise RuntimeError(f"Envelope write target must be an existing file: {value}")
        relative = resolved.relative_to(config.ROOT.resolve()).as_posix()
        from .behavior import behavior_write_block_reason

        behavior_reason = behavior_write_block_reason(relative)
        if behavior_reason:
            raise RuntimeError(f"Envelope cannot authorize behavior target {relative}: {behavior_reason}")
        if relative.startswith("brain_v2/employees/engineer/"):
            raise RuntimeError(f"Envelope cannot authorize Engineer state as a code write target: {relative}")
        normalized.append(relative)
    result = list(dict.fromkeys(normalized))
    if not result:
        raise RuntimeError("Envelope requires at least one exact write target.")
    return result


def _normalize_budgets(values: dict[str, Any] | None) -> dict[str, int]:
    defaults = {
        "repository_retrieval_calls": 18,
        "max_files_per_call": DEFAULT_MAX_FILES,
        "max_excerpts_per_call": DEFAULT_MAX_EXCERPTS,
        "max_chars_per_call": DEFAULT_MAX_TOTAL_CHARS,
        "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
    }
    raw = {**defaults, **(values or {})}
    normalized: dict[str, int] = {}
    for key, (minimum, maximum) in _BUDGET_KEYS.items():
        try:
            number = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Envelope budget {key} must be an integer.") from exc
        if not minimum <= number <= maximum:
            raise RuntimeError(f"Envelope budget {key} must be between {minimum} and {maximum}.")
        normalized[key] = number
    return normalized


def _verification_specs(commands: list[str], write_targets: list[str]) -> list[dict[str, Any]]:
    from .verify import verification_command_spec

    rows: list[dict[str, Any]] = []
    declared_commands = list(
        dict.fromkeys(str(item).strip() for item in commands if str(item).strip())
    )
    # The current non-OS-sandbox backend has exactly one autonomous-safe verifier:
    # isolated py_compile. Require complete target coverage so a passing command for
    # one file cannot bless an unverified sibling target.
    if any(Path(target).suffix.lower() != ".py" for target in write_targets):
        raise RuntimeError(
            "Every autonomous write target must currently be a Python file covered "
            "by isolated py_compile."
        )
    expected_commands = {
        f"python -m py_compile {target}" for target in write_targets
    }
    if set(declared_commands) != expected_commands or len(declared_commands) != len(
        write_targets
    ):
        missing = sorted(expected_commands - set(declared_commands))
        extra = sorted(set(declared_commands) - expected_commands)
        raise RuntimeError(
            "Envelope verification must contain exactly one isolated py_compile "
            f"command per write target; missing={missing}; extra={extra}."
        )
    for raw in declared_commands:
        spec = verification_command_spec(raw, write_targets)
        if spec is None:
            raise RuntimeError(f"Envelope verification command is not in the exact allowlist: {raw}")
        if spec.get("autonomous_safe") is not True:
            raise RuntimeError(
                "Envelope verification command can execute repository code without "
                f"an OS sandbox and is not autonomous-safe: {raw}"
            )
        cwd = Path(spec["cwd"]).resolve()
        try:
            relative_cwd = cwd.relative_to(config.ROOT.resolve()).as_posix() or "."
        except ValueError as exc:
            raise RuntimeError(f"Verification cwd escaped the repository: {cwd}") from exc
        argv = [str(item) for item in spec["argv"]]
        spec_payload = {
            "command": raw,
            "argv": argv,
            "cwd": relative_cwd,
            "autonomous_safe": True,
        }
        rows.append(
            {
                **spec_payload,
                "spec_sha256": hashlib.sha256(
                    json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    if not rows:
        raise RuntimeError("Envelope requires at least one exact allowlisted verification command.")
    return rows


def envelope_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Task Envelope: {payload.get('task_id', '')}",
            "",
            f"- Schema: `{payload.get('schema', '')}`",
            f"- Declared: `{payload.get('declared_at', '')}`",
            f"- Expires: `{payload.get('expires_at', '')}`",
            f"- Actor: `{((payload.get('declaration') or {}).get('consent_attribution') or {}).get('actor', 'unknown')}`",
            f"- Nonce storage: `{((payload.get('integrity') or {}).get('nonce_locator') or {}).get('mode', '')}`",
            "- In-run expansion: `false`",
            "- Real working-tree write during autonomy: `false`",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def record_envelope_event(
    task_id: str,
    stage: str,
    status: str,
    reason: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config.ensure_harness()
    event = {
        "schema": ENVELOPE_EVENT_SCHEMA,
        "task_id": safe_workspace_id(task_id),
        "stage": str(stage),
        "status": str(status),
        "reason": str(reason)[:1200],
        "details": details or {},
        "recorded_at": now_tz_iso(),
    }
    append_jsonl(config.ENGINEER_ENVELOPES_DIR / "events.jsonl", event)
    return event


def declare_task_envelope(
    task_id: str,
    task: str,
    allowed_roots: list[str],
    allowed_intents: list[str],
    write_targets: list[str],
    verification_commands: list[str],
    expires_at: str,
    *,
    retrieval_budget: dict[str, Any] | None = None,
    yes: bool = False,
) -> dict[str, Any]:
    """Create an envelope once. There is deliberately no update operation."""

    config.ensure_harness()
    task_key = safe_workspace_id(task_id)
    attribution = capture_consent_attribution()
    if not yes:
        record_consent_event(
            "engineer-envelope-declare",
            task_key,
            attribution,
            outcome="refused",
            detail="explicit --yes confirmation required",
        )
        raise RuntimeError("Envelope declaration requires explicit --yes confirmation.")
    try:
        request = str(task or "").strip()
        if not request:
            raise RuntimeError("Envelope task text is required.")
        expiry = _parse_expiry(expires_at)
        roots = normalize_authorized_roots(allowed_roots, config.ROOT)
        intents = normalize_authorized_intents(allowed_intents)
        if not roots or not intents:
            raise RuntimeError(
                "Envelope requires at least one allowed root and retrieval intent."
            )
        targets = _normalize_write_targets(write_targets)
        root_paths = [(config.ROOT / value).resolve() for value in roots]
        for target in targets:
            target_path = (config.ROOT / target).resolve()
            if not any(
                root == target_path or root in target_path.parents
                for root in root_paths
            ):
                raise RuntimeError(
                    "Envelope write target must be inside an allowed retrieval root: "
                    f"{target}"
                )
        budgets = _normalize_budgets(retrieval_budget)
        verification = _verification_specs(verification_commands, targets)
    except Exception as exc:
        record_consent_event(
            "engineer-envelope-declare",
            task_key,
            attribution,
            outcome="refused",
            detail=str(exc),
        )
        record_envelope_event(
            task_key,
            "declaration",
            "refused",
            str(exc),
        )
        raise
    json_path, md_path = _envelope_paths(task_key)
    if json_path.exists() or md_path.exists():
        record_consent_event(
            "engineer-envelope-declare",
            task_key,
            attribution,
            outcome="refused",
            detail="immutable envelope already exists",
        )
        raise RuntimeError(f"Immutable envelope already exists for task: {task_key}")

    nonce_path, nonce_locator = _nonce_path(task_key)
    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_bytes(32)
    try:
        with nonce_path.open("xb") as handle:
            handle.write(nonce)
    except FileExistsError as exc:
        raise RuntimeError(f"Envelope nonce already exists for task: {task_key}") from exc

    payload: dict[str, Any] = {
        "schema": ENVELOPE_SCHEMA,
        "schema_version": 1,
        "envelope_id": f"envelope_{task_key}",
        "task_id": task_key,
        "task": {
            "request": request,
            "request_sha256": hashlib.sha256(request.encode("utf-8")).hexdigest(),
        },
        "declared_at": now_tz_iso(),
        "expires_at": expiry.isoformat(timespec="seconds"),
        "declaration": {
            "confirmation_flag": True,
            "consent_attribution": attribution,
        },
        "scope": {
            "allowed_roots": roots,
            "allowed_intents": intents,
            "write_targets": targets,
        },
        "budgets": budgets,
        "verification": {"allowed_commands": verification},
        "execution": {
            "apply_target": "disposable_checkout_only",
            "real_working_tree_write": False,
            "external_retrieval": False,
            "in_run_expansion": False,
            "behavior_promotion": False,
        },
        "integrity": {
            "algorithm": "hmac-sha256",
            "nonce_locator": nonce_locator,
        },
    }
    tag = hmac.new(nonce, _canonical_bytes(payload), hashlib.sha256).hexdigest()
    payload["integrity"]["hmac_sha256"] = tag
    payload["integrity"]["canonical_payload_sha256"] = hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        with md_path.open("x", encoding="utf-8") as handle:
            handle.write(envelope_markdown(payload))
    except Exception:
        if json_path.exists():
            json_path.unlink()
        if md_path.exists():
            md_path.unlink()
        if nonce_path.exists():
            nonce_path.unlink()
        raise
    event = record_consent_event(
        "engineer-envelope-declare",
        task_key,
        attribution,
        outcome="accepted",
        metadata={
            "envelope_id": payload["envelope_id"],
            "envelope_sha256": sha256_file(json_path),
            "nonce_storage": nonce_locator["mode"],
        },
    )
    record_envelope_event(task_key, "declaration", "accepted", "immutable envelope created")
    return {
        **payload,
        "ok": True,
        "path": str(md_path),
        "jsonPath": str(json_path),
        "envelope_file_sha256": sha256_file(json_path),
        "consent_event": event,
        "output": (
            f"Declared immutable envelope {payload['envelope_id']} for {task_key}; "
            f"actor={attribution['actor']}; expires={payload['expires_at']}."
        ),
    }


def load_task_envelope(task_id: str, *, require_unexpired: bool = True) -> dict[str, Any]:
    """Load and authenticate one immutable envelope."""

    task_key = safe_workspace_id(task_id)
    json_path, _ = _envelope_paths(task_key)
    if not json_path.is_file():
        raise RuntimeError(f"Task envelope not found: {task_key}")
    payload = json.loads(json_path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(payload, dict) or payload.get("schema") != ENVELOPE_SCHEMA:
        raise RuntimeError("Task envelope schema is invalid.")
    if str(payload.get("task_id") or "") != task_key:
        raise RuntimeError("Task envelope identity does not match its filename.")
    request = str((payload.get("task") or {}).get("request") or "")
    expected_request_hash = hashlib.sha256(request.encode("utf-8")).hexdigest()
    if expected_request_hash != str((payload.get("task") or {}).get("request_sha256") or ""):
        raise RuntimeError("Task envelope request hash is invalid.")
    integrity = payload.get("integrity") or {}
    if integrity.get("algorithm") != "hmac-sha256":
        raise RuntimeError("Task envelope integrity algorithm is invalid.")
    stored_locator = integrity.get("nonce_locator")
    if not isinstance(stored_locator, dict):
        raise RuntimeError("Task envelope nonce locator is missing.")
    nonce_path, observed_locator = _nonce_path(task_key, stored_locator)
    if observed_locator != stored_locator:
        raise RuntimeError("Task envelope nonce locator metadata changed.")
    try:
        nonce = nonce_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("Task envelope nonce is unavailable.") from exc
    expected = hmac.new(nonce, _canonical_bytes(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(integrity.get("hmac_sha256") or "")):
        raise RuntimeError("Task envelope HMAC validation failed.")
    canonical_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if not hmac.compare_digest(canonical_hash, str(integrity.get("canonical_payload_sha256") or "")):
        raise RuntimeError("Task envelope canonical payload hash failed.")
    targets = write_targets(payload)
    stored_specs = (payload.get("verification") or {}).get("allowed_commands")
    if not isinstance(stored_specs, list):
        raise RuntimeError("Task envelope verification specs are missing.")
    expected_specs = _verification_specs(
        [
            str(item.get("command") or "")
            for item in stored_specs
            if isinstance(item, dict)
        ],
        targets,
    )
    if stored_specs != expected_specs:
        raise RuntimeError(
            "Task envelope verification argv/cwd specs changed after declaration."
        )
    expiry = _parse_expiry(str(payload.get("expires_at") or "")) if require_unexpired else datetime.fromisoformat(
        str(payload.get("expires_at") or "").replace("Z", "+00:00")
    )
    if require_unexpired and expiry <= datetime.now().astimezone():
        raise RuntimeError("Task envelope is expired.")
    return {
        **payload,
        "envelope_file_sha256": sha256_file(json_path),
        "path": str(json_path.with_suffix(".md")),
        "jsonPath": str(json_path),
    }


def envelope_from_manifest(manifest: dict[str, Any], *, require_unexpired: bool = True) -> dict[str, Any] | None:
    binding = manifest.get("task_envelope")
    if not isinstance(binding, dict) or not binding.get("task_id"):
        return None
    envelope = load_task_envelope(str(binding["task_id"]), require_unexpired=require_unexpired)
    if str(binding.get("envelope_id") or "") != str(envelope.get("envelope_id") or ""):
        raise RuntimeError("J-space envelope binding identity changed.")
    if str(binding.get("canonical_payload_sha256") or "") != str(
        (envelope.get("integrity") or {}).get("canonical_payload_sha256") or ""
    ):
        raise RuntimeError("J-space envelope binding hash changed.")
    return envelope


def retrieval_request_violation(envelope: dict[str, Any], request: dict[str, Any]) -> str:
    """Return the first immutable-envelope budget/scope violation, if any."""

    scope = envelope.get("scope") or {}
    budgets = envelope.get("budgets") or {}
    intent = str(request.get("intent") or "")
    if intent not in set(scope.get("allowed_intents") or []):
        return "intent_outside_task_envelope"
    try:
        roots = normalize_authorized_roots([str(item) for item in request.get("allowed_roots") or []], config.ROOT)
    except ValueError as exc:
        return str(exc)
    allowed_paths = [(config.ROOT / value).resolve() for value in scope.get("allowed_roots") or []]
    for root in roots:
        candidate = (config.ROOT / root).resolve()
        if not any(allowed == candidate or allowed in candidate.parents for allowed in allowed_paths):
            return "root_outside_task_envelope"
    limits = {
        "max_files": int(budgets.get("max_files_per_call") or 0),
        "max_excerpts": int(budgets.get("max_excerpts_per_call") or 0),
        "max_total_chars": int(budgets.get("max_chars_per_call") or 0),
        "max_file_bytes": int(budgets.get("max_file_bytes") or 0),
    }
    defaults = {
        "max_files": DEFAULT_MAX_FILES,
        "max_excerpts": DEFAULT_MAX_EXCERPTS,
        "max_total_chars": DEFAULT_MAX_TOTAL_CHARS,
        "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
    }
    for key, limit in limits.items():
        try:
            requested = int(request.get(key, defaults[key]))
        except (TypeError, ValueError):
            return f"invalid_{key}"
        if requested > limit:
            return f"{key}_outside_task_envelope"
    return ""


def verification_commands(envelope: dict[str, Any]) -> list[str]:
    return [
        str(item.get("command") or "")
        for item in ((envelope.get("verification") or {}).get("allowed_commands") or [])
        if isinstance(item, dict) and item.get("command")
    ]


def verification_command_specs(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in ((envelope.get("verification") or {}).get("allowed_commands") or [])
        if isinstance(item, dict)
    ]


def write_targets(envelope: dict[str, Any]) -> list[str]:
    return [str(item) for item in ((envelope.get("scope") or {}).get("write_targets") or [])]
