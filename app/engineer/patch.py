"""Patch proposal: generate a diff, then refuse it unless it survives every check.

Nothing here writes to the working tree. A proposal is always saved - blocked ones
too - so a refusal is auditable. `apply_allowed` is always False on generation; only
the human Apply gate can override it.

Write scope is composed only from human-selected files. Retrieval widens what the
model may read, never what it may edit.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from engineer_grounding import GROUNDING_CONTRACT, build_evidence_catalog, validate_grounded_plan

from . import artifacts, config, context, jspace, models, prompts, verify
from .apply import dry_run_patch_applicability, is_forbidden_patch_path, repair_unified_diff_hunk_headers
from .util import dedupe, normalize_repo_file, now_iso, repo_path, unique_artifact_id


FORBIDDEN_CHANGES_CHECKED = [
    "repo-local files only",
    "package.json requires --allow-package-json",
    "deletions require --allow-delete",
    "secret/env files blocked",
    "behavior prompts, canonical behavior, and approved lessons blocked",
]


def patch_target_files(plan: dict, selected_records: list[dict]) -> list[str]:
    # Retrieval and model-proposed files are read-only context. The write scope
    # is composed only from files a human explicitly selected for this task.
    files = []
    for item in selected_records:
        if item.get("input"):
            files.append(item["input"])
        elif item.get("path"):
            try:
                files.append(str(Path(item["path"]).resolve().relative_to(config.ROOT)))
            except ValueError:
                files.append(item["path"])
    existing_files = []
    for item in dedupe(files):
        try:
            resolved = repo_path(item)
        except RuntimeError:
            continue
        if resolved.exists() and resolved.is_file():
            existing_files.append(resolved.relative_to(config.ROOT).as_posix())
    return dedupe(existing_files)


def grounding_patch_gate(plan: dict) -> dict:
    """Once retrieval happened, a patch requires an evidence-backed replan.

    Retrieval means the model saw content the original plan did not. Letting a
    pre-retrieval plan drive a patch would silently un-ground the change.
    """
    run_id = str(plan.get("run_id") or "")
    try:
        manifest = jspace.load_manifest(run_id)
    except (OSError, RuntimeError, json.JSONDecodeError):
        manifest = {}
    retrieval_sources = (manifest.get("retrieval") or {}).get("sources") or []
    required = bool(retrieval_sources)
    valid_evidence_ids = {
        str(item.get("evidence_id"))
        for item in plan.get("evidence_index") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    validation = validate_grounded_plan(plan, valid_evidence_ids) if required else {
        "status": "not_required",
        "failed_rules": [],
        "warnings": [],
        "claim_ids": [],
        "action_ids": [],
    }
    failed_rules = list(validation.get("failed_rules") or [])
    if required and plan.get("grounding_contract_version") != GROUNDING_CONTRACT:
        failed_rules.append("grounded_replan_required_after_retrieval")
    return {
        "required": required,
        "status": "fail" if failed_rules else validation.get("status", "pass"),
        "failed_rules": dedupe(failed_rules),
        "warnings": validation.get("warnings") or [],
        "claim_ids": validation.get("claim_ids") or [],
        "action_ids": validation.get("action_ids") or [],
        "retrieval_source_count": len(retrieval_sources),
    }


def classify_patch_risk(
    target_files: list[str], allow_package_json: bool, allow_delete: bool
) -> tuple[str, list[str], list[str]]:
    risk = "low"
    reasons: list[str] = []
    blockers: list[str] = []
    for file_text in target_files:
        path = file_text.replace("\\", "/").lower()
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".txt"}:
            reasons.append(f"Low risk docs/copy file: {file_text}")
        elif suffix in {".css", ".scss"}:
            reasons.append(f"Low risk stylesheet file: {file_text}")
        elif suffix in {".tsx", ".jsx"}:
            risk = "medium" if risk == "low" else risk
            reasons.append(f"Medium risk React component logic/rendering file: {file_text}")
        elif suffix in {".py", ".ts", ".js", ".mjs"}:
            risk = "medium" if risk == "low" else risk
            reasons.append(f"Medium risk CLI/parser/local behavior file: {file_text}")
        if path.endswith("package.json") or path.endswith("package-lock.json") or path.endswith("pnpm-lock.yaml") or path.endswith("yarn.lock"):
            risk = "high"
            reasons.append(f"High risk package or lock file: {file_text}")
            if path.endswith("package.json") and not allow_package_json:
                blockers.append("package.json changes require --allow-package-json")
        if "server/" in path or "company-brain-api" in path:
            risk = "high"
            reasons.append(f"High risk backend/API bridge file: {file_text}")
            # Parked web-gui is the active Engineer dogfood target; allow patch
            # generation (Apply still requires human --yes). Block only non-gui bridges.
            if not path.startswith("web-gui/"):
                blockers.append("backend/API bridge patching is high risk and not enabled in v0.4")
        if ".env" in path or "providers.env" in path or "secret" in path:
            risk = "high"
            reasons.append(f"High risk secret/env file: {file_text}")
            blockers.append("secret/env files cannot be patched")
        if (
            "canonical_behavior.md" in path
            or "approved_lessons" in path
            or (
                path.startswith("brain_v2/employees/")
                and "/prompts/" in path
            )
        ):
            risk = "high"
            reasons.append(f"High risk behavior-state file: {file_text}")
            blockers.append(
                "behavior prompts, canonical behavior, and approved lessons cannot be patched"
            )
        if "autonomous" in path:
            risk = "high"
            reasons.append(f"High risk autonomous loop behavior file: {file_text}")
            blockers.append("autonomous loop behavior cannot be patched by Engineer v0.4")
    if allow_delete:
        reasons.append("--allow-delete enabled; deletion checks still require generated diff validation")
    return risk, dedupe(reasons or ["No target files identified; risk treated as high."]), dedupe(blockers)


def diff_touched_files(unified_diff: str) -> list[str]:
    files = []
    for line in unified_diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            value = line[4:].strip()
            if value == "/dev/null":
                files.append(value)
                continue
            if value.startswith("a/") or value.startswith("b/"):
                value = value[2:]
            files.append(value)
    return dedupe([item for item in files if item and item != "/dev/null"])


def diff_deletes_file(unified_diff: str) -> bool:
    return "--- " in unified_diff and "+++ /dev/null" in unified_diff


def diff_hunks(unified_diff: str) -> list[dict]:
    hunks: list[dict] = []
    current_file = ""
    file_counts: dict[str, int] = {}
    for line in unified_diff.splitlines():
        if line.startswith("+++ "):
            current_file = line[4:].strip()
            if current_file.startswith("a/") or current_file.startswith("b/"):
                current_file = current_file[2:]
        elif line.startswith("@@ ") and current_file and current_file != "/dev/null":
            file_counts[current_file] = file_counts.get(current_file, 0) + 1
            hunks.append(
                {
                    "hunk_id": f"{current_file}#hunk{file_counts[current_file]}",
                    "file": current_file,
                    "header": line,
                }
            )
    return hunks


def check_patch_payload(payload: dict, allow_package_json: bool, allow_delete: bool) -> dict:
    """Deterministic patch gate. Any failed rule blocks the proposal."""
    warnings: list[str] = []
    failed_rules: list[str] = []
    diff = payload.get("unified_diff", "") or ""
    if not diff.strip():
        failed_rules.append("empty_unified_diff")
        warnings.append("Unified diff is empty.")
    touched = diff_touched_files(diff)
    target_files = {normalize_repo_file(item) for item in payload.get("target_files", [])}
    for file_text in touched:
        if file_text == "/dev/null":
            continue
        try:
            resolved = repo_path(file_text)
            normalized = resolved.relative_to(config.ROOT).as_posix().lower()
        except (RuntimeError, ValueError):
            failed_rules.append("diff_touches_non_repo_file")
            warnings.append(f"Diff touches non-repo file: {file_text}")
            continue
        forbidden_rule = is_forbidden_patch_path(normalized)
        if forbidden_rule:
            failed_rules.append("diff_touches_forbidden_path")
            warnings.append(f"Diff touches forbidden path ({forbidden_rule}): {file_text}")
        if target_files and normalized not in target_files:
            failed_rules.append("diff_touches_unapproved_file")
            warnings.append(f"Diff touches file outside target set: {file_text}")
        lower = normalized.lower()
        if lower.endswith("package.json") and not allow_package_json:
            failed_rules.append("package_json_without_permission")
            warnings.append("Diff touches package.json without --allow-package-json.")
        if any(marker in lower for marker in [".env", "providers.env", "secret"]):
            failed_rules.append("secret_or_env_file_touched")
            warnings.append(f"Diff touches secret/env-like file: {file_text}")
        if "canonical_behavior.md" in lower or "approved_lessons" in lower:
            failed_rules.append("canonical_or_approved_memory_touched")
            warnings.append(f"Diff touches canonical/approved Engineer memory: {file_text}")
    if diff_deletes_file(diff) and not allow_delete:
        failed_rules.append("delete_without_permission")
        warnings.append("Diff deletes a file without --allow-delete.")
    verification_commands = payload.get("verification_commands") or []
    unsafe_commands = [
        str(command)
        for command in verification_commands
        if not verify.is_safe_verification_command(str(command), payload.get("target_files") or [])[0]
    ]
    if not verification_commands:
        failed_rules.append("missing_verification_commands")
        warnings.append("Patch artifact is missing verification commands.")
    if unsafe_commands:
        warnings.append(
            "Verification commands include non-allowlisted entries; replace them before verification: "
            + ", ".join(unsafe_commands)
        )
    verification_quality = str(payload.get("verification_commands_quality") or "")
    if verification_quality == "warn":
        warnings.extend(payload.get("verification_command_warnings") or [])
    if payload.get("grounding_required"):
        valid_claim_ids = set(map(str, payload.get("available_claim_ids") or []))
        valid_action_ids = set(map(str, payload.get("available_plan_action_ids") or []))
        trace = payload.get("patch_claim_trace") or []
        hunks = diff_hunks(diff)
        traced_hunks = {str(item.get("hunk_id")) for item in trace if isinstance(item, dict)}
        if not hunks or {item["hunk_id"] for item in hunks} != traced_hunks:
            failed_rules.append("incomplete_patch_hunk_trace")
            warnings.append("Every grounded patch hunk must trace to a material claim.")
        for item in trace:
            claim_ids = set(map(str, item.get("claim_ids") or [])) if isinstance(item, dict) else set()
            if not claim_ids:
                failed_rules.append("patch_hunk_without_claim")
            elif not claim_ids.issubset(valid_claim_ids):
                failed_rules.append("patch_hunk_with_unknown_claim")
            action_ids = set(map(str, item.get("plan_action_ids") or [])) if isinstance(item, dict) else set()
            if not action_ids:
                failed_rules.append("patch_hunk_without_plan_action")
            elif not action_ids.issubset(valid_action_ids):
                failed_rules.append("patch_hunk_with_unknown_plan_action")
    if diff.strip():
        applicability = dry_run_patch_applicability(diff)
        failed_rules.extend(applicability.get("failed_rules") or [])
        warnings.extend(applicability.get("warnings") or [])
    return {
        "status": "fail" if failed_rules else "pass",
        "warnings": dedupe(warnings),
        "failed_rules": dedupe(failed_rules),
        "verification_commands_quality": verification_quality or ("warn" if unsafe_commands else "pass"),
    }


def build_patch_claim_trace(
    unified_diff: str,
    claim_links: object,
    available_claim_ids: list[str],
    allow_fallback: bool = False,
    claim_action_map: dict[str, list[str]] | None = None,
) -> list[dict]:
    links = claim_links if isinstance(claim_links, list) else []
    by_file: dict[str, list[str]] = {}
    for link in links:
        if not isinstance(link, dict):
            continue
        try:
            file_text = normalize_repo_file(str(link.get("file") or ""))
        except RuntimeError:
            continue
        by_file[file_text] = dedupe([str(value) for value in link.get("claim_ids") or [] if str(value)])
    fallback = dedupe([str(value) for value in available_claim_ids if str(value)]) if allow_fallback else []
    trace = []
    for hunk in diff_hunks(unified_diff):
        claim_ids = by_file.get(normalize_repo_file(hunk["file"]), fallback)
        action_ids = dedupe(
            [
                action_id
                for claim_id in claim_ids
                for action_id in (claim_action_map or {}).get(claim_id, [])
            ]
        )
        trace.append({**hunk, "claim_ids": claim_ids, "plan_action_ids": action_ids})
    return trace


def blocked_patch_payload(
    patch_id: str,
    plan: dict,
    selected_records: list[dict],
    target_files: list[str],
    risk_level: str,
    risk_reasons: list[str],
    limitations: list[str],
    checker_status: str,
) -> dict:
    return {
        "patch_id": patch_id,
        "source_run_id": plan.get("run_id", ""),
        "j_space": jspace.pointer(str(plan.get("run_id") or "")),
        "created_at": now_iso(),
        "task": plan.get("task", ""),
        "target_files": target_files,
        "selected_file_inspections": context.selected_file_inspections(selected_records),
        "change_summary": "Patch proposal blocked before diff generation.",
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "apply_allowed": False,
        "unified_diff": "",
        "verification_commands": [],
        "verification_commands_original": [],
        "verification_commands_quality": "fail",
        "verification_command_warnings": ["No verification commands were prepared because patch generation was blocked."],
        "verification_commands_suggested_manual": [],
        "rollback_plan": "No patch was generated; no rollback required.",
        "forbidden_changes_checked": list(FORBIDDEN_CHANGES_CHECKED),
        "checker_status": checker_status,
        "limitations": limitations,
        "patch_status": "blocked",
        "patch_checker": {"status": "fail", "warnings": limitations, "failed_rules": ["blocked_patch"]},
    }


def extract_unified_diff(text: str) -> str:
    if "```" in text:
        match = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    start = text.find("diff --git ")
    if start == -1:
        start = text.find("--- ")
    return text[start:].strip() if start != -1 else ""


def _normalize_limitations(raw_limitations: object) -> list[str]:
    """Coerce a model's "limitations" field to a real list of strings.

    A model that returns "limitations" as a single string rather than a JSON
    array (a real, observed output-shape deviation, not hypothetical - found via
    a committed patch artifact whose limitations field was a list of individual
    characters) silently gets shredded downstream: every consumer of this field
    does `[*generated.get("limitations", []), ...]` / `dedupe([*...])`, and
    unpacking a string with `*` iterates its characters one at a time.
    """
    if isinstance(raw_limitations, str):
        return [raw_limitations] if raw_limitations.strip() else []
    if isinstance(raw_limitations, list):
        return [str(item) for item in raw_limitations if str(item).strip()]
    if raw_limitations:
        return [str(raw_limitations)]
    return []


def generate_patch_with_model(
    plan: dict,
    selected_records: list[dict],
    target_files: list[str],
    *,
    grounding_required: bool,
    grounding_evidence_entries: list[dict] | None = None,
) -> dict:
    """Ask a model for a diff. Both claim-link contracts live in prompt data files.

    The grounded branch stays strict: every claim_id must name a supplied material
    claim. The ungrounded branch explicitly tells the model not to invent claim_ids
    rather than leaving the field ambiguous.
    """
    contract_name = (
        "patch_claim_contract_grounded" if grounding_required else "patch_claim_contract_ungrounded"
    )
    claim_link_contract = prompts.load(contract_name).strip()
    verification_allowlist = dedupe(
        [
            *verify.generated_python_compile_specs(target_files),
            *config.VERIFICATION_COMMAND_SPECS,
        ]
    )
    prompt = prompts.render(
        "patch_proposal",
        {
            "task": plan.get("task", ""),
            "revised_codex_prompt": plan.get("revised_codex_prompt") or plan.get("codex_prompt") or "",
            "target_files": json.dumps(target_files, ensure_ascii=False, indent=2),
            "selected_file_inspections": json.dumps(selected_records, ensure_ascii=False, indent=2),
            "grounding_context": json.dumps(
                {
                    "material_claims": plan.get("material_claims") or [],
                    "plan_actions": plan.get("plan_actions") or [],
                    "evidence_index": plan.get("evidence_index") or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "evidence_entries": json.dumps(grounding_evidence_entries or [], ensure_ascii=False, indent=2),
            "claim_link_contract": claim_link_contract,
            "verification_allowlist": "; ".join(verification_allowlist),
        },
    )
    routed = models.call_tier_with_fallback(
        "Lead Engineer",
        prompt,
        tier=models._engineer_primary_tier(),
        max_cooldown_wait_seconds=0,
        cloud_request_timeout_seconds=45,
        cloud_max_retries=0,
        max_tokens=2000,
    )
    from .util import extract_json_object

    parsed = extract_json_object(routed.content)
    if not parsed:
        parsed = {
            "change_summary": "Model returned non-JSON patch output.",
            "unified_diff": extract_unified_diff(routed.content),
            "verification_commands": [],
            "rollback_plan": "Do not apply unless manually reviewed.",
            "limitations": ["Structured patch JSON parse failed."],
        }
    parsed["model_route"] = models.route_of(routed)
    parsed["model_attempts"] = models.attempts_of(routed)
    # Normalize once, here, so every downstream consumer of "limitations" can
    # assume a real list of strings - see _normalize_limitations's docstring.
    parsed["limitations"] = _normalize_limitations(parsed.get("limitations"))
    return parsed


def deterministic_literal_patch(task: str, target_files: list[str]) -> dict | None:
    """A model-free path for one exact, unambiguous single-literal replacement.

    Deliberately narrow: one file, one occurrence, no newlines, and the replacement
    must not already be present. Anything else returns None rather than guessing.
    """
    if len(target_files) != 1:
        return None
    match = re.search(
        r"\b(?:change|replace)\b[\s\S]*?\bfrom\s+(['\"])(.*?)\1\s+to\s+(['\"])(.*?)\3",
        task,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    old_text, new_text = match.group(2), match.group(4)
    if not old_text or old_text == new_text or "\n" in old_text or "\n" in new_text:
        return None
    relative = normalize_repo_file(target_files[0])
    path = repo_path(relative)
    if not path.is_file():
        return None
    original = path.read_text(encoding="utf-8", errors="strict")
    if original.count(old_text) != 1 or new_text in original:
        return None
    updated = original.replace(old_text, new_text, 1)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    ).strip()
    if not diff:
        return None
    return {
        "change_summary": f"Replace one exact literal in {relative}.",
        "unified_diff": diff,
        "verification_commands": verify.recommended_verification_commands(target_files),
        "rollback_plan": f"Restore the original literal in {relative}.",
        "limitations": [
            "Model routes were unavailable; used the narrow deterministic single-literal replacement fallback."
        ],
        "model_route": "deterministic_literal_replacement",
        "model_attempts": [],
        "generation_mode": "deterministic_literal_replacement",
    }


def _attempt_patch_followup(
    plan: dict,
    generated: dict,
    selected_records: list[dict],
    target_files: list[str],
    grounding_gate: dict,
    patch_id: str,
) -> dict:
    """Bounded one-round follow-up when the model says it needs more evidence, not
    that the change is unsafe or out of scope. Mirrors the grounded-replan follow-up
    in replan.py: the model can ask for scope, engineer_context_request classifies
    and gates it, and only a pre-authorized (or partially-authorized) request is
    ever executed - engineer_context_execute is the real per-item safety boundary.

    Deferred imports avoid a circular import: loop.py imports patch.py as patch_mod
    at module level, so patch.py cannot import loop.py at module level.
    """
    from .loop import emit_generated_context_request
    from .retrieval import engineer_context_execute

    run_id = str(plan.get("run_id") or "")
    followup_request = emit_generated_context_request(run_id, generated, label=f"patch_{patch_id}") or {}
    if followup_request.get("status") not in {"ready", "approval_required"} or not followup_request.get("request_id"):
        return {"attempted": False}
    try:
        execution = engineer_context_execute(run_id, str(followup_request.get("request_id") or ""))
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "error": str(exc)[:900]}
    if not execution.get("ok") or not execution.get("retrieval_results"):
        return {
            "attempted": True,
            "ok": False,
            "status": execution.get("status"),
            "blocked_items": execution.get("blocked_items"),
        }
    workspace, _, _ = jspace.paths(run_id)
    manifest = jspace.load_manifest(run_id)
    refreshed_evidence = build_evidence_catalog(
        manifest, workspace, config.ROOT, task_text=str(plan.get("task") or "")
    ).get("entries") or []
    try:
        second = generate_patch_with_model(
            plan,
            selected_records,
            target_files,
            grounding_required=bool(grounding_gate.get("required")),
            grounding_evidence_entries=refreshed_evidence,
        )
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "error": str(exc)[:900]}
    return {
        "attempted": True,
        "ok": True,
        "request_id": followup_request.get("request_id"),
        "retrieved": execution.get("retrieval_results"),
        "generated": second,
    }


def engineer_patch(
    run_id: str,
    selected_files: list[str] | None = None,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
) -> dict:
    """Propose a patch. Never writes to the working tree; `apply_allowed` stays False."""
    config.ensure_harness()
    cost_mark = models.ledger_mark()
    json_path, plan = artifacts.load_run_json(run_id)
    patch_id = unique_artifact_id("_engineer_patch", config.ENGINEER_PATCHES_DIR)
    checker_status = plan.get("checker_status", "")
    original_selected = [
        item.get("input") or item.get("path")
        for item in ((plan.get("context_manifest") or {}).get("selected_files") or [])
        if isinstance(item, dict) and (item.get("input") or item.get("path"))
    ]
    all_selected = dedupe([*(original_selected or []), *(selected_files or [])])
    excerpt_chars = context.resolve_excerpt_char_limit(
        plan.get("selected_file_excerpt_chars_per_file")
        if plan.get("selected_file_excerpt_chars_per_file") is not None
        else ((plan.get("context_manifest") or {}).get("selected_file_excerpt_chars_per_file"))
    )
    selected_records, missing = context.selected_context(
        all_selected, plan.get("task", ""), max_chars_per_call=excerpt_chars
    )
    target_files = patch_target_files(plan, selected_records)
    grounding_gate = grounding_patch_gate(plan)
    grounding_evidence_entries: list[dict] = []
    manifest: dict = {}
    if grounding_gate.get("required"):
        workspace, _, _ = jspace.paths(str(plan.get("run_id") or ""))
        manifest = jspace.load_manifest(str(plan.get("run_id") or ""))
        grounding_evidence_entries = build_evidence_catalog(
            manifest, workspace, config.ROOT, task_text=str(plan.get("task") or "")
        ).get("entries") or []
    risk_level, risk_reasons, risk_blockers = classify_patch_risk(target_files, allow_package_json, allow_delete)
    limitations = []
    if missing:
        limitations.append(f"Selected files missing: {', '.join(missing)}")
    if checker_status == "fail" and not override_failed_check:
        limitations.append("Refusing patch because checker_status=fail; pass --override-failed-check to override.")
    if plan.get("blocked_plan"):
        limitations.append("Refusing patch because source Engineer Plan is blocked_plan=true.")
    if grounding_gate["status"] == "fail":
        limitations.append(
            "Refusing patch because repository retrieval requires a sufficient evidence-backed replan: "
            + ", ".join(grounding_gate.get("failed_rules") or ["grounding_gate_failed"])
        )
    if risk_blockers:
        limitations.extend(risk_blockers)
    if not target_files:
        limitations.append("No repo-local target files available for patch generation.")
    patch_followup: dict = {"attempted": False}
    if limitations:
        payload = blocked_patch_payload(
            patch_id, plan, selected_records, target_files, risk_level, risk_reasons, dedupe(limitations), checker_status
        )
    else:
        generated = deterministic_literal_patch(str(plan.get("task") or ""), target_files)
        model_error = ""
        if generated is None:
            try:
                generated = generate_patch_with_model(
                    plan,
                    selected_records,
                    target_files,
                    grounding_required=bool(grounding_gate.get("required")),
                    grounding_evidence_entries=grounding_evidence_entries,
                )
            except Exception as exc:  # noqa: BLE001
                model_error = str(exc)[:900]
        if (
            generated is not None
            and not str(generated.get("unified_diff") or "").strip()
            and generated.get("insufficient_evidence")
            and generated.get("context_requests")
        ):
            patch_followup = _attempt_patch_followup(
                plan, generated, selected_records, target_files, grounding_gate, patch_id
            )
            if patch_followup.get("ok"):
                generated = patch_followup.pop("generated")
        if generated is None:
            payload = blocked_patch_payload(
                patch_id,
                plan,
                selected_records,
                target_files,
                risk_level,
                risk_reasons,
                [f"Best Available patch route failed; patch requires model route availability. Detail: {model_error}"],
                checker_status,
            )
        else:
            from .envelope import envelope_from_manifest, verification_commands

            bound_envelope = (
                envelope_from_manifest(manifest)
                if manifest.get("task_envelope")
                else None
            )
            if bound_envelope is not None:
                sealed_commands = verification_commands(bound_envelope)
                original_commands = dedupe(
                    [
                        str(item).strip()
                        for item in generated.get("verification_commands", [])
                        if str(item).strip()
                    ]
                )
                outside = [
                    command
                    for command in original_commands
                    if command not in set(sealed_commands)
                ]
                verification = {
                    "verification_commands": sealed_commands,
                    "verification_commands_original": original_commands,
                    "verification_commands_quality": (
                        "pass" if not outside else "warn"
                    ),
                    "verification_command_warnings": (
                        [
                            "The immutable envelope replaced model/recommended "
                            "verification with its exact sealed command list."
                        ]
                        if outside or original_commands != sealed_commands
                        else []
                    ),
                    "verification_commands_outside_envelope": [
                        {
                            "command": command,
                            "reason": "refused: command not declared in the immutable task envelope",
                        }
                        for command in outside
                    ],
                    "verification_commands_suggested_manual": [],
                }
            else:
                verification = verify.prepare_verification_commands(
                    generated.get("verification_commands", []),
                    target_files,
                )
            claim_ids = list(grounding_gate.get("claim_ids") or [])
            plan_actions = [item for item in plan.get("plan_actions") or [] if isinstance(item, dict)]
            action_ids = [str(item.get("action_id")) for item in plan_actions if item.get("action_id")]
            claim_action_map: dict[str, list[str]] = {}
            for action_item in plan_actions:
                for claim_id in action_item.get("claim_ids") or []:
                    claim_action_map.setdefault(str(claim_id), []).append(str(action_item.get("action_id") or ""))
            raw_diff = str(generated.get("unified_diff") or "")
            repaired_diff, repair_notes = repair_unified_diff_hunk_headers(raw_diff)
            if repair_notes and repaired_diff != raw_diff:
                generated["unified_diff"] = repaired_diff
                generated["hunk_header_repair"] = repair_notes
                generated["limitations"] = dedupe(
                    [
                        *(generated.get("limitations") or []),
                        "Deterministically recomputed malformed unified-diff hunk headers before checking.",
                    ]
                )
            elif repair_notes:
                generated["hunk_header_repair"] = repair_notes
            patch_claim_trace = build_patch_claim_trace(
                str(generated.get("unified_diff") or ""),
                generated.get("claim_links"),
                claim_ids,
                allow_fallback=generated.get("generation_mode") == "deterministic_literal_replacement",
                claim_action_map=claim_action_map,
            )
            payload = {
                "patch_id": patch_id,
                "source_run_id": plan.get("run_id", json_path.stem),
                "j_space": jspace.pointer(str(plan.get("run_id") or json_path.stem)),
                "created_at": now_iso(),
                "task": plan.get("task", ""),
                "target_files": target_files,
                "selected_file_inspections": context.selected_file_inspections(selected_records),
                "change_summary": generated.get("change_summary", ""),
                "risk_level": risk_level,
                "risk_reasons": risk_reasons,
                "apply_allowed": False,
                "unified_diff": generated.get("unified_diff", ""),
                "grounding_required": grounding_gate.get("required", False),
                "grounding_gate": grounding_gate,
                "available_claim_ids": claim_ids,
                "available_plan_action_ids": action_ids,
                "claim_links": generated.get("claim_links") or [],
                "patch_claim_trace": patch_claim_trace,
                **verification,
                "rollback_plan": generated.get("rollback_plan", "Do not apply if review fails; discard patch artifact."),
                "forbidden_changes_checked": list(FORBIDDEN_CHANGES_CHECKED),
                "checker_status": checker_status,
                "limitations": generated.get("limitations", []),
                "patch_status": "proposed",
                "model_route": generated.get("model_route", ""),
                "model_attempts": generated.get("model_attempts", []),
                "generation_mode": generated.get("generation_mode", "model"),
                "hunk_header_repair": generated.get("hunk_header_repair") or [],
            }
            patch_checker = check_patch_payload(payload, allow_package_json, allow_delete)
            payload["patch_checker"] = patch_checker
            if patch_checker["status"] == "fail":
                payload["patch_status"] = "blocked"
                payload["limitations"] = dedupe([*payload.get("limitations", []), *patch_checker["warnings"]])

    payload.setdefault("grounding_required", grounding_gate.get("required", False))
    payload.setdefault("grounding_gate", grounding_gate)
    payload.setdefault("available_claim_ids", grounding_gate.get("claim_ids") or [])
    payload.setdefault("available_plan_action_ids", grounding_gate.get("action_ids") or [])
    payload.setdefault("claim_links", [])
    payload.setdefault("patch_claim_trace", [])
    payload["patch_followup"] = patch_followup

    payload["cost_accounting"] = models.ledger_since(cost_mark)
    md_path = config.ENGINEER_PATCHES_DIR / f"{patch_id}.md"
    patch_json_path = config.ENGINEER_PATCHES_DIR / f"{patch_id}.json"
    # Patch proposals are always saved so engineer-apply-patch can locate them by patch_id.
    md_path.write_text(artifacts.patch_markdown(payload), encoding="utf-8")
    patch_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    source_run_id = str(payload.get("source_run_id") or plan.get("run_id") or json_path.stem)
    patch_status = str(payload.get("patch_status") or "blocked")
    j_space_fields = jspace.update_manifest(
        source_run_id,
        "patch",
        "complete" if patch_status == "proposed" else "blocked",
        "patch_proposed" if patch_status == "proposed" else "blocked",
        "; ".join(payload.get("limitations") or []) or str(payload.get("change_summary") or ""),
        jspace.artifact_ref("engineer_patch", patch_id, md_path, patch_json_path),
        {
            "generation_mode": payload.get("generation_mode", ""),
            "model_route": payload.get("model_route", ""),
            "risk_level": payload.get("risk_level", ""),
            "target_files": payload.get("target_files") or [],
        },
    )
    payload.update(
        {
            "ok": True,
            "path": str(md_path),
            "jsonPath": str(patch_json_path),
            "output": f"Engineer patch {patch_id}: status={payload.get('patch_status')}, risk={payload.get('risk_level')}, apply_allowed=false",
            **j_space_fields,
        }
    )
    return payload


def run_patch_followup_self_tests() -> dict:
    """Zero-model-call, zero-filesystem regression suite for _attempt_patch_followup.

    Mirrors replan.py's run_followup_self_tests. The label scheme (label=f"patch_{patch_id}",
    distinct from the plan stage's "plan" and the replan stage's "replan_{revision_id}")
    is what actually prevents the request_id collision class of bug found in the
    plan/replan follow-up rounds - that is structural (every patch_id is unique by
    construction via unique_artifact_id) rather than something a test can regress on
    its own, but these cases cover the eligibility and bounding logic around it.
    """

    import sys

    from . import loop as loop_mod
    from . import retrieval as retrieval_mod

    mod = sys.modules[__name__]
    results: list[dict] = []
    original_emit = loop_mod.emit_generated_context_request
    original_execute = retrieval_mod.engineer_context_execute
    original_generate = mod.generate_patch_with_model
    original_paths = jspace.paths
    original_manifest = jspace.load_manifest

    plan = {"run_id": "self_test_run", "task": "self test task"}
    generated = {
        "unified_diff": "",
        "insufficient_evidence": True,
        "context_requests": [{"item_id": "x", "intent": "find_symbol", "reason": "r", "allowed_roots": ["app"]}],
    }

    def _run(emit_result: dict | None, execute_result: dict | None, execute_raises: Exception | None = None):
        calls = {"emit": 0, "execute": 0, "generate": 0}

        def fake_emit(run_id: str, parsed: dict, *, label: str) -> dict | None:
            calls["emit"] += 1
            return dict(emit_result) if emit_result is not None else None

        def fake_execute(run_id: str, request_id: str) -> dict:
            calls["execute"] += 1
            if execute_raises is not None:
                raise execute_raises
            return dict(execute_result or {})

        def fake_generate(*args, **kwargs) -> dict:
            calls["generate"] += 1
            return {"unified_diff": "--- a/x\n+++ b/x\n", "change_summary": "second pass"}

        loop_mod.emit_generated_context_request = fake_emit
        retrieval_mod.engineer_context_execute = fake_execute
        mod.generate_patch_with_model = fake_generate
        jspace.paths = lambda run_id: (Path("."), Path("."), Path("."))
        jspace.load_manifest = lambda run_id: {}
        try:
            outcome = _attempt_patch_followup(plan, dict(generated), [], ["app/x.py"], {}, "self_test_patch")
        finally:
            loop_mod.emit_generated_context_request = original_emit
            retrieval_mod.engineer_context_execute = original_execute
            mod.generate_patch_with_model = original_generate
            jspace.paths = original_paths
            jspace.load_manifest = original_manifest
        return outcome, calls

    try:
        # 1. No context request emitted (empty items or emit failure): not eligible.
        outcome, calls = _run(None, None)
        results.append(
            {
                "id": "no_emitted_request_not_eligible",
                "ok": outcome == {"attempted": False} and calls == {"emit": 1, "execute": 0, "generate": 0},
                "detail": {"outcome": outcome, "calls": calls},
            }
        )

        # 2. Emitted but denied (status neither ready nor approval_required): not eligible.
        outcome, calls = _run({"request_id": "ctx_x", "status": "denied"}, None)
        results.append(
            {
                "id": "denied_not_eligible",
                "ok": outcome == {"attempted": False} and calls == {"emit": 1, "execute": 0, "generate": 0},
                "detail": {"outcome": outcome, "calls": calls},
            }
        )

        # 3. Eligible, execution succeeds: exactly one extra generate_patch_with_model call.
        outcome, calls = _run(
            {"request_id": "ctx_x", "status": "ready"},
            {"ok": True, "retrieval_results": [{"path": "app/x.py"}]},
        )
        results.append(
            {
                "id": "ready_triggers_second_pass",
                "ok": (
                    outcome.get("attempted") is True
                    and outcome.get("ok") is True
                    and outcome.get("generated", {}).get("unified_diff")
                    and calls == {"emit": 1, "execute": 1, "generate": 1}
                ),
                "detail": {"outcome": outcome, "calls": calls},
            }
        )

        # 4. Eligible, but execution comes back empty: no second pass, original stands.
        outcome, calls = _run(
            {"request_id": "ctx_x", "status": "ready"},
            {"ok": False, "retrieval_results": [], "status": "blocked"},
        )
        results.append(
            {
                "id": "execution_fails_no_second_pass",
                "ok": (
                    outcome.get("attempted") is True
                    and outcome.get("ok") is False
                    and calls == {"emit": 1, "execute": 1, "generate": 0}
                ),
                "detail": {"outcome": outcome, "calls": calls},
            }
        )
    finally:
        loop_mod.emit_generated_context_request = original_emit
        retrieval_mod.engineer_context_execute = original_execute
        mod.generate_patch_with_model = original_generate
        jspace.paths = original_paths
        jspace.load_manifest = original_manifest

    passed = sum(1 for item in results if item["ok"])
    return {"ok": passed == len(results), "passed": passed, "failed": len(results) - passed, "cases": results}
