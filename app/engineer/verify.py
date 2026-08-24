"""Verification: the only place the harness executes anything.

Execution is limited to an exact command -> argv map: the enumerated
`VERIFICATION_COMMAND_SPECS` plus `python -m py_compile <target>` specs generated
only for Python files that are actually applied targets. There is no shell, no
string splitting, and no way for model output to widen the set - an unrecognised
command is recorded as skipped, never run.

A model claiming success is not verification. `verification_passed` is true only
when commands were actually executed and every executed command exited zero.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from repository_retrieval import secret_content_block_reason

from . import artifacts, config, jspace
from .util import dedupe, now_iso, repo_path, unique_artifact_id


OUTPUT_CAPTURE_LIMIT_BYTES = 1024 * 1024
DISPOSABLE_VERIFICATION_TIMEOUT_SECONDS = 60


def _source_root(source_root: Path | str | None = None) -> Path:
    root = Path(source_root) if source_root is not None else config.ROOT
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Verification source root is not a directory: {resolved}")
    return resolved


def _source_path(path_text: str, source_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = source_root / path
    resolved = path.resolve()
    if resolved != source_root and source_root not in resolved.parents:
        raise RuntimeError(f"Refusing verification path outside source root: {resolved}")
    return resolved


def _public_authorization(authorization: dict | None) -> dict:
    if not isinstance(authorization, dict):
        return {}
    allowed = {
        "kind",
        "envelope_id",
        "task_id",
        "actor",
        "method",
        "declared_at",
        "expires_at",
        "attribution",
    }
    return {key: value for key, value in authorization.items() if key in allowed}


def _bounded_stream_evidence(data: bytes) -> dict:
    retained = data[:OUTPUT_CAPTURE_LIMIT_BYTES]
    decoded = retained.decode("utf-8", errors="replace")
    secret_reason = secret_content_block_reason(decoded)
    return {
        "text": (
            f"[verification output withheld: {secret_reason}]"
            if secret_reason
            else decoded
        ),
        "base64": (
            "" if secret_reason else base64.b64encode(retained).decode("ascii")
        ),
        "bytes": len(data),
        "retained_bytes": len(retained),
        "sha256": hashlib.sha256(data).hexdigest(),
        "truncated": len(data) > len(retained),
        "capture_limit_bytes": OUTPUT_CAPTURE_LIMIT_BYTES,
        "text_encoding": "utf-8",
        "text_errors": "replace",
        "secret_content_blocked": bool(secret_reason),
        "secret_block_reason": secret_reason,
    }


def _explicit_command_spec(command: str, source_root: Path) -> dict | None:
    explicit = config.VERIFICATION_COMMAND_SPECS.get(command)
    if not explicit:
        return None
    configured_cwd = Path(explicit["cwd"]).resolve()
    real_root = config.ROOT.resolve()
    try:
        cwd_relative = configured_cwd.relative_to(real_root)
    except ValueError:
        return None
    return {
        **explicit,
        "argv": [str(value) for value in explicit["argv"]],
        "cwd": (source_root / cwd_relative).resolve(),
    }


def generated_python_compile_specs(
    target_files: list[str] | None,
    *,
    source_root: Path | str | None = None,
) -> dict[str, dict]:
    """Build exact py_compile specs only from existing repo-local Python targets."""

    root = _source_root(source_root)
    specs: dict[str, dict] = {}
    for target in target_files or []:
        try:
            resolved = _source_path(str(target), root)
            relative = resolved.relative_to(root).as_posix()
        except (RuntimeError, ValueError):
            continue
        if not resolved.is_file() or resolved.suffix.lower() != ".py":
            continue
        command = f"python -m py_compile {relative}"
        specs[command] = {
            # Isolated mode ignores PYTHON* environment variables and the current
            # directory; -S avoids site/customization imports. py_compile parses
            # target bytes but does not execute them.
            "argv": [
                sys.executable,
                "-I",
                "-S",
                "-m",
                "py_compile",
                relative,
            ],
            "cwd": root,
            "autonomous_safe": True,
        }
    return specs


def verification_command_spec(
    command: str,
    target_files: list[str] | None = None,
    *,
    source_root: Path | str | None = None,
) -> dict | None:
    root = _source_root(source_root)
    normalized_command = command.strip()
    explicit = _explicit_command_spec(normalized_command, root)
    if explicit:
        return explicit
    return generated_python_compile_specs(
        target_files, source_root=root
    ).get(normalized_command)


def is_safe_verification_command(
    command: str,
    target_files: list[str] | None = None,
    *,
    source_root: Path | str | None = None,
) -> tuple[bool, str]:
    if not command.strip():
        return False, "empty command"
    if verification_command_spec(
        command, target_files, source_root=source_root
    ):
        return True, ""
    return False, "command not in exact local allowlist for applied patch targets"


def recommended_verification_commands(
    target_files: list[str],
    *,
    source_root: Path | str | None = None,
) -> list[str]:
    root = _source_root(source_root)
    normalized: set[str] = set()
    for item in target_files:
        try:
            normalized.add(_source_path(str(item), root).relative_to(root).as_posix().lower())
        except (RuntimeError, ValueError):
            continue
    commands = list(generated_python_compile_specs(target_files, source_root=root))
    if "web-gui/server/company-brain-api.mjs" in normalized:
        commands.append("node --check web-gui/server/company-brain-api.mjs")
    if any(path.startswith("web-gui/") for path in normalized):
        commands.append("cd web-gui && npm run build")
    if any(path.startswith("web-gui/server/") for path in normalized):
        commands.append("cd web-gui && npm run test:api")
    if any(path.startswith("dogfood/ph_withholding/") for path in normalized):
        commands.append("cd dogfood/ph_withholding && npm test")
    if any(path.endswith(".py") for path in normalized):
        commands.append("python scripts/smoke_test.py")
    return dedupe(commands or ["python scripts/smoke_test.py"])


def prepare_verification_commands(
    raw_commands: object,
    target_files: list[str],
    *,
    source_root: Path | str | None = None,
) -> dict:
    """Split model-proposed commands into allowlisted and manual-only, then top up.

    Non-allowlisted commands are never silently dropped: they are recorded as
    manual suggestions so a human can see what the model wanted to run.
    """
    original = raw_commands if isinstance(raw_commands, list) else []
    original = dedupe([str(item).strip() for item in original if str(item).strip()])
    allowlisted: list[str] = []
    suggested_manual: list[dict] = []
    warnings: list[str] = []
    for command in original:
        safe, reason = is_safe_verification_command(
            command, target_files, source_root=source_root
        )
        if safe:
            allowlisted.append(command)
        else:
            suggested_manual.append({"command": command, "reason": reason})
    recommended = recommended_verification_commands(
        target_files, source_root=source_root
    )
    added = [command for command in recommended if command not in allowlisted]
    allowlisted = dedupe([*allowlisted, *added])
    if suggested_manual:
        warnings.append(
            f"Moved {len(suggested_manual)} non-allowlisted model command(s) to manual suggestions."
        )
    if added:
        warnings.append(
            "Added deterministic allowlisted replacement command(s): " + ", ".join(added)
        )
    quality = "pass" if original and not suggested_manual and not added else "warn"
    if not allowlisted:
        quality = "fail"
        warnings.append("No allowlisted verification command could be prepared.")
    return {
        "verification_commands": allowlisted,
        "verification_commands_original": original,
        "verification_commands_quality": quality,
        "verification_command_warnings": warnings,
        "verification_commands_suggested_manual": suggested_manual,
    }


def engineer_verify(
    applied_patch_id: str,
    run: bool = False,
    save: bool = False,
    *,
    source_root: Path | str | None = None,
    allowed_commands: list[str] | None = None,
    required_commands: list[str] | None = None,
    sealed_command_specs: list[dict] | None = None,
    authorization: dict | None = None,
) -> dict:
    """Prepare or execute verification against one explicit source root.

    The default root preserves the public real-worktree behavior. A different root
    is a disposable checkout: it requires task-envelope authorization, enforces the
    envelope command subset in addition to the fixed harness allowlist, and always
    persists its verification artifact in real harness state.
    """
    config.ensure_harness()
    root = _source_root(source_root)
    real_root = config.ROOT.resolve()
    disposable = root != real_root
    public_authorization = _public_authorization(authorization)
    if disposable:
        if str(public_authorization.get("kind") or "") != "task_envelope_sandbox":
            raise RuntimeError(
                "Disposable-checkout verification requires "
                "authorization.kind=task_envelope_sandbox."
            )
        if allowed_commands is None:
            raise RuntimeError(
                "Disposable-checkout verification requires explicit allowed_commands."
            )
        if required_commands is None or sealed_command_specs is None:
            raise RuntimeError(
                "Disposable-checkout verification requires the envelope's exact "
                "required commands and sealed command specs."
            )
        save = True

    allowed_command_list = dedupe(
        [str(item).strip() for item in allowed_commands or [] if str(item).strip()]
    )
    allowed_command_set = set(allowed_command_list)
    applied_json_path = artifacts.resolve_artifact_json(applied_patch_id, config.ENGINEER_APPLIED_PATCHES_DIR)
    applied = json.loads(applied_json_path.read_text(encoding="utf-8", errors="replace"))
    patch_json_path_text = str(applied.get("source_patch_json") or "")
    commands = applied.get("verification_commands") or []
    changed_files = applied.get("changed_files") or []
    patch = {}
    if patch_json_path_text:
        patch_json_path = repo_path(patch_json_path_text)
        if patch_json_path.exists():
            patch = json.loads(patch_json_path.read_text(encoding="utf-8", errors="replace"))
            if not commands:
                commands = patch.get("verification_commands") or []
    commands = [str(item).strip() for item in commands if str(item).strip()]
    if disposable:
        required_command_list = dedupe(
            [
                str(item).strip()
                for item in required_commands or []
                if str(item).strip()
            ]
        )
        if required_command_list != allowed_command_list:
            raise RuntimeError(
                "Disposable verification requires every sealed envelope command "
                "exactly once and in declared order."
            )
        sealed_rows = sealed_command_specs or []
        if len(sealed_rows) != len(required_command_list):
            raise RuntimeError("Sealed verification command count changed.")
        for command, sealed in zip(required_command_list, sealed_rows):
            if not isinstance(sealed, dict) or str(sealed.get("command") or "") != command:
                raise RuntimeError("Sealed verification command identity changed.")
            sealed_payload = {
                "command": command,
                "argv": [str(item) for item in sealed.get("argv") or []],
                "cwd": str(sealed.get("cwd") or ""),
                "autonomous_safe": bool(sealed.get("autonomous_safe")),
            }
            sealed_hash = hashlib.sha256(
                json.dumps(
                    sealed_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if sealed_hash != str(sealed.get("spec_sha256") or ""):
                raise RuntimeError(f"Sealed verification spec hash changed: {command}")
            spec = verification_command_spec(
                command, changed_files, source_root=root
            )
            if spec is None or spec.get("autonomous_safe") is not True:
                raise RuntimeError(
                    f"Verification command is not safe for in-process-isolated "
                    f"autonomy: {command}"
                )
            cwd = Path(spec["cwd"]).resolve()
            current_payload = {
                "command": command,
                "argv": [str(item) for item in spec["argv"]],
                "cwd": cwd.relative_to(root).as_posix() or ".",
                "autonomous_safe": True,
            }
            if current_payload != sealed_payload:
                raise RuntimeError(
                    f"Verification argv/cwd changed after envelope declaration: {command}"
                )
        commands = required_command_list
    suggested_manual = applied.get("verification_commands_suggested_manual") or []
    verification_id = unique_artifact_id(
        "_engineer_verification", config.ENGINEER_VERIFICATIONS_DIR
    )
    results = []
    warnings = []
    commands_run = []
    commands_skipped = []
    for command in commands:
        safe, reason = is_safe_verification_command(
            command, changed_files, source_root=root
        )
        if not safe:
            commands_skipped.append({"command": command, "reason": reason})
            warnings.append(f"Skipped command: {command} ({reason})")
            results.append({"command": command, "status": "skipped", "reason": reason})
            continue
        if allowed_commands is not None and command not in allowed_command_set:
            reason = "command not allowed by task envelope"
            commands_skipped.append({"command": command, "reason": reason})
            warnings.append(f"Skipped command: {command} ({reason})")
            results.append({"command": command, "status": "skipped", "reason": reason})
            continue
        if not run:
            results.append({"command": command, "status": "not_run", "reason": "manual_run_required"})
            continue
        try:
            spec = verification_command_spec(
                command, changed_files, source_root=root
            )
            if spec is None:
                raise RuntimeError("Verification command lost its allowlist execution spec.")
            cwd = Path(spec["cwd"]).resolve()
            if cwd != root and root not in cwd.parents:
                raise RuntimeError(
                    f"Verification cwd escaped source root: {cwd}"
                )
            argv = [str(value) for value in spec["argv"]]
            run_environment = None
            if disposable:
                run_environment = {
                    key: value
                    for key, value in os.environ.items()
                    if key.upper()
                    in {
                        "COMSPEC",
                        "PATHEXT",
                        "SYSTEMDRIVE",
                        "SYSTEMROOT",
                        "TEMP",
                        "TMP",
                        "WINDIR",
                    }
                }
            completed = subprocess.run(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                close_fds=True,
                env=run_environment,
                timeout=(
                    DISPOSABLE_VERIFICATION_TIMEOUT_SECONDS
                    if disposable
                    else None
                ),
            )
            commands_run.append(command)
            stdout_evidence = _bounded_stream_evidence(completed.stdout)
            stderr_evidence = _bounded_stream_evidence(completed.stderr)
            secret_output_blocked = bool(
                stdout_evidence["secret_content_blocked"]
                or stderr_evidence["secret_content_blocked"]
            )
            results.append(
                {
                    "command": command,
                    "argv": argv,
                    "cwd": str(cwd),
                    "cwd_authority": (
                        "disposable_checkout" if disposable else "real_worktree"
                    ),
                    "status": (
                        "error"
                        if secret_output_blocked
                        else "pass"
                        if completed.returncode == 0
                        else "fail"
                    ),
                    "exit_code": completed.returncode,
                    "stdout": stdout_evidence["text"],
                    "stderr": stderr_evidence["text"],
                    "stdout_base64": stdout_evidence["base64"],
                    "stderr_base64": stderr_evidence["base64"],
                    "stdout_bytes": stdout_evidence["bytes"],
                    "stderr_bytes": stderr_evidence["bytes"],
                    "stdout_retained_bytes": stdout_evidence["retained_bytes"],
                    "stderr_retained_bytes": stderr_evidence["retained_bytes"],
                    "stdout_sha256": stdout_evidence["sha256"],
                    "stderr_sha256": stderr_evidence["sha256"],
                    "stdout_truncated": stdout_evidence["truncated"],
                    "stderr_truncated": stderr_evidence["truncated"],
                    "secret_output_blocked": secret_output_blocked,
                    "stdout_secret_block_reason": stdout_evidence[
                        "secret_block_reason"
                    ],
                    "stderr_secret_block_reason": stderr_evidence[
                        "secret_block_reason"
                    ],
                    "output_capture_limit_bytes": OUTPUT_CAPTURE_LIMIT_BYTES,
                    "output_encoding": "utf-8",
                    "output_encoding_errors": "replace",
                }
            )
        except Exception as exc:  # noqa: BLE001
            commands_run.append(command)
            results.append(
                {
                    "command": command,
                    "status": "error",
                    "error": str(exc),
                    "cwd_authority": (
                        "disposable_checkout" if disposable else "real_worktree"
                    ),
                }
            )
    executed_results = [item for item in results if item.get("status") in {"pass", "fail", "error"}]
    verification_passed = bool(run and executed_results) and all(
        item.get("status") == "pass" for item in executed_results
    )
    verification_incomplete = bool(
        not run
        or not commands_run
        or commands_skipped
        or suggested_manual
        or any(
            item.get("stdout_truncated") or item.get("stderr_truncated")
            for item in results
        )
    )
    source_run_id = str(applied.get("source_run_id") or patch.get("source_run_id") or "")
    payload = {
        "ok": True,
        "source_run_id": source_run_id,
        "applied_patch_id": applied.get("applied_patch_id", applied_json_path.stem),
        "verification_id": verification_id,
        "j_space": jspace.pointer(source_run_id) if source_run_id else {},
        "created_at": now_iso(),
        "source_root": str(root),
        "source_root_authority": (
            "disposable_checkout" if disposable else "real_worktree"
        ),
        "application_mode": (
            "disposable_checkout"
            if disposable
            else str(applied.get("application_mode") or "real_worktree")
        ),
        "authorization": (
            public_authorization
            if disposable
            else (
                public_authorization
                or {"kind": "manual_run_opt_in" if run else "prepare_only"}
            )
        ),
        "run_commands": run,
        "commands": commands,
        "allowed_commands": (
            allowed_command_list if allowed_commands is not None else None
        ),
        "allowed_commands_enforced": allowed_commands is not None,
        "required_commands": (
            list(required_commands or []) if disposable else None
        ),
        "sealed_command_specs": (
            list(sealed_command_specs or []) if disposable else None
        ),
        "sealed_command_specs_enforced": disposable,
        "commands_run": commands_run,
        "commands_skipped": commands_skipped,
        "commands_suggested_manual": suggested_manual,
        "verification_passed": verification_passed,
        "verification_incomplete": verification_incomplete,
        "results": results,
        "warnings": warnings,
        "source_applied_patch_json": str(applied_json_path),
    }
    md_path = config.ENGINEER_VERIFICATIONS_DIR / f"{verification_id}.md"
    json_path = config.ENGINEER_VERIFICATIONS_DIR / f"{verification_id}.json"
    if save:
        md_path.write_text(artifacts.verification_markdown(payload), encoding="utf-8")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not run:
            checkpoint_status = "prepared"
            workspace_status = (
                "sandbox_verification_prepared"
                if disposable
                else "verification_prepared"
            )
        elif verification_passed and not verification_incomplete:
            checkpoint_status = "pass"
            workspace_status = "sandbox_verified" if disposable else "verified"
        elif verification_passed or verification_incomplete:
            checkpoint_status = "incomplete"
            workspace_status = (
                "sandbox_verification_incomplete"
                if disposable
                else "verification_incomplete"
            )
        else:
            checkpoint_status = "fail"
            workspace_status = (
                "sandbox_verification_failed"
                if disposable
                else "verification_failed"
            )
        j_space_fields = jspace.update_manifest(
            source_run_id,
            "verify",
            checkpoint_status,
            workspace_status,
            (
                f"run={str(run).lower()}; passed={str(verification_passed).lower()}; "
                f"incomplete={str(verification_incomplete).lower()}; commands_run={len(commands_run)}"
            ),
            jspace.artifact_ref("engineer_verification", verification_id, md_path, json_path),
            {
                "run_commands": run,
                "commands_run": commands_run,
                "commands_skipped": commands_skipped,
                "commands_suggested_manual": suggested_manual,
                "verification_passed": verification_passed,
                "verification_incomplete": verification_incomplete,
                "source_root_authority": (
                    "disposable_checkout" if disposable else "real_worktree"
                ),
                "allowed_commands": (
                    allowed_command_list if allowed_commands is not None else None
                ),
                "authorization": (
                    public_authorization
                    if disposable
                    else (
                        public_authorization
                        or {"kind": "manual_run_opt_in" if run else "prepare_only"}
                    )
                ),
            },
        ) if source_run_id else {}
    else:
        j_space_fields = jspace.response_fields(source_run_id) if source_run_id else {}
    # Only report artifact paths that actually exist on disk.
    payload["path"] = str(md_path) if save else ""
    payload["jsonPath"] = str(json_path) if save else ""
    payload.update(j_space_fields)
    ran = len(commands_run)
    payload["output"] = (
        f"Engineer verification {verification_id}: {len(commands)} command(s), "
        + (
            f"{ran} executed, passed={str(verification_passed).lower()}, incomplete={str(verification_incomplete).lower()}"
            if run
            else "prepared only (pass --run to execute), incomplete=true"
        )
        + (
            ", source=disposable_checkout"
            if disposable
            else ", source=real_worktree"
        )
    )
    # The cost ledger's human-minutes field is never inferred, only human-declared
    # (scripts/engineer_cost_ledger.py). Left to memory, it never gets filled in -
    # 109 real attempts, zero notes, as of 2026-08-12. Surface the exact command
    # here, once, right when the number is still fresh, instead of relying on
    # someone remembering to run it later.
    payload["human_cost_reminder"] = (
        (
            f"python scripts/engineer_cost_ledger.py --note-attempt {source_run_id} "
            "--human-minutes N --outcome verified|rework --rework-rounds N "
            '--note "what took the time"'
        )
        if (save and run and source_run_id and not disposable)
        else ""
    )
    return payload
