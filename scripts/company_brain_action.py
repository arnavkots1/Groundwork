"""Thin CLI over the CompanyBrain Engineer library.

This file parses arguments, calls exactly one `engineer.api` function, and prints
JSON. All behavior lives in `app/engineer/`; see `app/engineer/api.py` for the
product API that any other surface (MCP, HTTP) should wrap instead of this script.

The public interface here is frozen: action names, flags, artifact shapes, and
stdout shape are depended on by other tools. The module-level re-exports below
exist because sibling benchmark scripts import this module and reach for harness
internals by name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from api_clients import ProviderError, call_provider  # noqa: E402
from engineer import api  # noqa: E402
from engineer import (  # noqa: E402
    apply as _apply_mod,
    config as _config,
    context as _context,
    jspace as _jspace,
    models as _models,
    patch as _patch_mod,
    verify as _verify_mod,
)
from repository_retrieval import SUPPORTED_INTENTS as REPOSITORY_RETRIEVAL_INTENTS  # noqa: E402


# --------------------------------------------------------------- public surface
# Callers import this module and use these names directly. They are re-exported,
# not reimplemented; the definitions live in app/engineer/.

engineer_task = api.run_task
engineer_plan = api.plan_change
engineer_envelope_declare = api.declare_envelope
engineer_autonomous_run = api.run_autonomous
engineer_retrieve = api.retrieve
engineer_context_request = api.request_context
engineer_context_approve = api.approve_context
engineer_context_execute = api.execute_context
engineer_context_invalidate = api.invalidate_context
engineer_repo_map = api.map_repository
engineer_replan = api.replan_with_evidence
engineer_check_run = api.recheck_plan
engineer_review = api.record_review
engineer_patch = api.propose_patch
engineer_patch_review = api.review_patch
engineer_apply_patch = api.apply_patch
engineer_revert_applied_patch = api.revert_applied_patch
engineer_trial_patch = api.trial_patch
engineer_verify = api.verify_applied
engineer_feedback = api.record_feedback
engineer_approve_lesson = api.approve_lesson
engineer_validate_behavior_promotion = api.validate_behavior_promotion
engineer_rollback_behavior = api.rollback_behavior
engineer_reject_lesson = api.reject_lesson
engineer_version = api.record_version
run_summary = api.audit_run
model_registry_show = api.show_model_registry
model_registry_refresh = api.refresh_model_registry
model_probe = api.probe_model
call_agent = api.call_model
read_file = api.read_named_file
read_run = api.read_artifact
list_runs = api.list_artifacts
save_outbox = api.save_codex_request
list_nvidia_models = api.list_nvidia_catalog
status = api.workspace_status

# Harness internals used by the sibling benchmark scripts.
ENGINEER_TASK_STAGES = _config.ENGINEER_TASK_STAGES
_selected_context = _context.selected_context
_create_j_space_manifest = _jspace.create_manifest
_snapshot_selected_plan_excerpts = _jspace.snapshot_selected_plan_excerpts
_parse_unified_diff = _apply_mod.parse_unified_diff
_apply_patch_to_text = _apply_mod.apply_patch_to_text
_dry_run_patch_applicability = _apply_mod.dry_run_patch_applicability
_plan_patch_operations = _apply_mod.plan_patch_operations
_is_forbidden_patch_path = _apply_mod.is_forbidden_patch_path
_check_patch_payload = _patch_mod.check_patch_payload
_deterministic_literal_patch = _patch_mod.deterministic_literal_patch
_generate_patch_with_model = _patch_mod.generate_patch_with_model
_patch_target_files = _patch_mod.patch_target_files
_build_patch_claim_trace = _patch_mod.build_patch_claim_trace
_is_safe_verification_command = _verify_mod.is_safe_verification_command
_verification_command_spec = _verify_mod.verification_command_spec
_prepare_verification_commands = _verify_mod.prepare_verification_commands
call_tier_with_fallback = _models.call_tier_with_fallback
call_best_available_with_fallback = _models.call_best_available_with_fallback


def _optional_bool(value: str) -> bool | None:
    from engineer.util import optional_bool

    return optional_bool(value)


def _pin_provider_from_cli(provider: str, model_id: str = "") -> None:
    """Pin Engineer lead routing when --provider is not Best Available."""
    name = (provider or "").strip()
    if not name or name.lower() == "best available":
        return
    aliases = {
        "cursor": "Cursor Agent CLI",
        "cursor agent": "Cursor Agent CLI",
        "cursor agent cli": "Cursor Agent CLI",
        "claude": "Claude Code CLI",
        "claude code cli": "Claude Code CLI",
        "codex": "Codex CLI",
        "codex cli": "Codex CLI",
    }
    pinned = aliases.get(name.lower(), name)
    os.environ["COMPANYBRAIN_PIN_PROVIDER"] = pinned
    if model_id.strip():
        if pinned == "Cursor Agent CLI":
            os.environ["CURSOR_CLI_MODEL"] = model_id.strip()
        elif pinned == "Claude Code CLI":
            os.environ["CLAUDE_CLI_MODEL"] = model_id.strip()
        elif pinned == "Codex CLI":
            os.environ["CODEX_CLI_MODEL"] = model_id.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "engineer-task", "engineer-plan", "engineer-envelope-declare", "engineer-autonomous-run", "engineer-retrieve", "engineer-context-request", "engineer-context-approve", "engineer-context-execute", "engineer-context-invalidate", "engineer-repo-map", "engineer-replan", "engineer-check-run", "engineer-review", "engineer-patch", "engineer-patch-review", "engineer-apply-patch", "engineer-revert-applied-patch", "engineer-trial-patch", "engineer-verify", "engineer-feedback", "engineer-approve-lesson", "engineer-validate-behavior-promotion", "engineer-rollback-behavior", "engineer-reject-lesson", "engineer-version", "run-summary", "engineer-summarize-run", "model-registry", "model-registry-refresh", "model-probe", "call-agent", "read-file", "read-run", "list-runs", "save-outbox", "list-nvidia-models"])
    parser.add_argument("--provider", default="Best Available")
    parser.add_argument("--role", default="CEO")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--target", default="")
    parser.add_argument("--selected-file", action="append", default=[])
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--task", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--use-approved-lessons", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-file", default="")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--map-max-files", type=int, default=500)
    parser.add_argument("--retrieval-root", action="append", default=[])
    parser.add_argument("--retrieval-intent", action="append", choices=sorted(REPOSITORY_RETRIEVAL_INTENTS), default=[])
    parser.add_argument("--write-target", action="append", default=[])
    parser.add_argument("--verification-command", action="append", default=[])
    parser.add_argument("--expires-at", default="")
    parser.add_argument("--retrieval-call-budget", type=int)
    parser.add_argument("--max-files-per-call", type=int)
    parser.add_argument("--max-excerpts-per-call", type=int)
    parser.add_argument("--max-chars-per-call", type=int)
    parser.add_argument("--max-file-bytes", type=int)
    parser.add_argument(
        "--stop-after",
        choices=ENGINEER_TASK_STAGES,
        default="patch",
        help="engineer-task: last stage to run. Apply and Verify are never run by engineer-task.",
    )
    parser.add_argument("--patch-id", default="")
    parser.add_argument("--applied-patch-id", default="")
    parser.add_argument("--usefulness", type=int, default=3)
    parser.add_argument("--correctness", type=int, default=3)
    parser.add_argument("--accepted", choices=["true", "false"], default="false")
    parser.add_argument("--notes", default="")
    parser.add_argument("--codex-success", choices=["true", "false", ""], default="")
    parser.add_argument("--compile-passed", choices=["true", "false", ""], default="")
    parser.add_argument("--tests-passed", choices=["true", "false", ""], default="")
    parser.add_argument("--files-touched-correct", choices=["true", "false", ""], default="")
    parser.add_argument("--manual-correction", choices=["", "low", "medium", "high"], default="")
    parser.add_argument("--lesson-needed", choices=["true", "false", ""], default="")
    parser.add_argument("--codex-summary", default="")
    parser.add_argument("--lesson-id", default="")
    parser.add_argument("--evidence-file", default="")
    parser.add_argument("--promotion-id", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--allow-package-json", action="store_true")
    parser.add_argument("--allow-delete", action="store_true")
    parser.add_argument("--allow-high-risk", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--override-failed-check", action="store_true")
    parser.add_argument("--keep-on-fail", action="store_true", help="engineer-trial-patch: do not auto-revert when verify fails")
    parser.add_argument("--no-approved-lessons", action="store_true")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--enable-if-ok", action="store_true")
    return parser


def _dispatch(args: argparse.Namespace) -> dict:
    """Map one parsed action onto exactly one api call. No logic beyond argument shaping."""
    action = args.action
    if action == "status":
        return api.workspace_status()
    if action == "engineer-task":
        _pin_provider_from_cli(args.provider, args.model_id)
        return api.run_task(
            args.task or args.prompt,
            selected_files=args.selected_file,
            project=args.project,
            use_approved_lessons=not args.no_approved_lessons,
            allow_web=args.allow_web,
            retrieval_roots=args.retrieval_root,
            retrieval_intents=args.retrieval_intent,
            allow_package_json=args.allow_package_json,
            allow_delete=args.allow_delete,
            override_failed_check=args.override_failed_check,
            stop_after=args.stop_after,
            max_chars_per_call=args.max_chars_per_call,
        )
    if action == "engineer-plan":
        return api.plan_change(
            args.task or args.prompt,
            selected_files=args.selected_file,
            project=args.project,
            use_approved_lessons=args.use_approved_lessons,
            allow_web=args.allow_web,
            compare_baseline=args.compare_baseline,
            retrieval_roots=args.retrieval_root,
            retrieval_intents=args.retrieval_intent,
            max_chars_per_call=args.max_chars_per_call,
        )
    if action == "engineer-envelope-declare":
        budget_values = {
            "repository_retrieval_calls": args.retrieval_call_budget,
            "max_files_per_call": args.max_files_per_call,
            "max_excerpts_per_call": args.max_excerpts_per_call,
            "max_chars_per_call": args.max_chars_per_call,
            "max_file_bytes": args.max_file_bytes,
        }
        return api.declare_envelope(
            args.task_id or args.run_id or args.target,
            args.task or args.prompt,
            args.retrieval_root,
            args.retrieval_intent,
            args.write_target or args.selected_file,
            args.verification_command,
            args.expires_at,
            retrieval_budget={
                key: value for key, value in budget_values.items() if value is not None
            },
            yes=args.yes,
        )
    if action == "engineer-autonomous-run":
        return api.run_autonomous(args.task_id or args.run_id or args.target)
    if action == "engineer-retrieve":
        return api.retrieve(
            args.task_id or args.run_id or args.target,
            request_json=args.request_json,
            request_file=args.request_file,
        )
    if action == "engineer-context-request":
        return api.request_context(
            args.task_id or args.run_id or args.target,
            args.request_json or args.prompt,
            request_file=args.request_file,
        )
    if action == "engineer-context-approve":
        return api.approve_context(
            args.task_id or args.run_id or args.target,
            args.request_id or args.prompt,
            yes=args.yes,
        )
    if action == "engineer-context-execute":
        return api.execute_context(
            args.task_id or args.run_id or args.target,
            args.request_id or args.prompt,
        )
    if action == "engineer-context-invalidate":
        return api.invalidate_context(
            args.task_id or args.run_id or args.target,
            args.request_id or args.prompt,
            yes=args.yes,
        )
    if action == "engineer-repo-map":
        return api.map_repository(
            args.task_id or args.run_id or args.target,
            max_files=args.map_max_files,
        )
    if action == "engineer-replan":
        return api.replan_with_evidence(args.run_id or args.task_id or args.target)
    if action == "engineer-check-run":
        return api.recheck_plan(args.run_id or args.target or args.prompt, write=args.write)
    if action == "engineer-review":
        return api.record_review(
            args.run_id or args.target,
            changed_files=args.changed_file,
            codex_summary=args.codex_summary or args.prompt,
            compile_passed=_optional_bool(args.compile_passed),
            tests_passed=_optional_bool(args.tests_passed),
            files_touched_correct=_optional_bool(args.files_touched_correct),
            manual_correction=args.manual_correction,
            notes=args.notes,
        )
    if action == "engineer-patch":
        _pin_provider_from_cli(args.provider, args.model_id)
        return api.propose_patch(
            args.run_id or args.target,
            selected_files=args.selected_file,
            allow_package_json=args.allow_package_json,
            allow_delete=args.allow_delete,
            override_failed_check=args.override_failed_check,
        )
    if action == "engineer-patch-review":
        return api.review_patch(args.patch_id or args.target or args.prompt)
    if action == "engineer-apply-patch":
        return api.apply_patch(
            args.patch_id or args.target or args.prompt,
            yes=args.yes,
            require_clean_git=args.require_clean_git,
            allow_package_json=args.allow_package_json,
            allow_delete=args.allow_delete,
            override_failed_check=args.override_failed_check,
            allow_high_risk=args.allow_high_risk,
        )
    if action == "engineer-revert-applied-patch":
        return api.revert_applied_patch(
            args.applied_patch_id or args.patch_id or args.target or args.prompt,
            yes=args.yes,
        )
    if action == "engineer-trial-patch":
        return api.trial_patch(
            args.patch_id or args.target or args.prompt,
            yes=args.yes,
            run_verify=args.run or True,
            save_verify=args.save or True,
            auto_revert_on_fail=not args.keep_on_fail,
            require_clean_git=args.require_clean_git,
            allow_package_json=args.allow_package_json,
            allow_delete=args.allow_delete,
            override_failed_check=args.override_failed_check,
            allow_high_risk=args.allow_high_risk,
        )
    if action == "engineer-verify":
        return api.verify_applied(
            args.applied_patch_id or args.target or args.prompt,
            run=args.run,
            save=args.save,
        )
    if action == "engineer-feedback":
        return api.record_feedback(
            args.run_id or args.target,
            usefulness=args.usefulness,
            correctness=args.correctness,
            accepted=args.accepted == "true",
            notes=args.notes or args.prompt,
            codex_success=_optional_bool(args.codex_success),
            compile_passed=_optional_bool(args.compile_passed),
            files_touched_correct=_optional_bool(args.files_touched_correct),
            manual_correction=args.manual_correction,
            lesson_needed=_optional_bool(args.lesson_needed),
        )
    if action == "engineer-approve-lesson":
        return api.approve_lesson(
            args.lesson_id or args.target or args.prompt,
            args.evidence_file,
            yes=args.yes,
        )
    if action == "engineer-validate-behavior-promotion":
        return api.validate_behavior_promotion(
            args.promotion_id or args.target or args.prompt,
            args.evidence_file,
            yes=args.yes,
        )
    if action == "engineer-rollback-behavior":
        return api.rollback_behavior(
            args.promotion_id or args.target or args.prompt,
            yes=args.yes,
        )
    if action == "engineer-reject-lesson":
        return api.reject_lesson(args.lesson_id or args.target, args.notes or args.prompt)
    if action == "engineer-version":
        return api.record_version(args.summary or args.prompt)
    if action == "run-summary":
        return api.audit_run(args.target or args.run_id or args.prompt, use_model=args.use_model)
    if action == "engineer-summarize-run":
        # Legacy alias for run-summary kept for older callers.
        return api.audit_run(args.target or args.prompt, use_model=args.use_model)
    if action == "model-registry":
        return api.show_model_registry()
    if action == "model-registry-refresh":
        return api.refresh_model_registry()
    if action == "model-probe":
        return api.probe_model(args.provider, args.model_id or args.target or args.prompt, enable_if_ok=args.enable_if_ok)
    if action == "call-agent":
        return api.call_model(args.provider, args.role, args.prompt)
    if action == "read-file":
        return api.read_named_file(args.target or args.prompt)
    if action == "read-run":
        return api.read_artifact(args.target or args.prompt)
    if action == "list-runs":
        return api.list_artifacts()
    if action == "save-outbox":
        return api.save_codex_request(args.prompt)
    return api.list_nvidia_catalog()


def main() -> int:
    args = _build_parser().parse_args()
    try:
        payload = _dispatch(args)
        print(json.dumps({"ok": True, **payload}))
        return 0
    except ProviderError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "retryable": exc.retryable, "retryAfter": exc.retry_after}))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
