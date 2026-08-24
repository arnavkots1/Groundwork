"""CompanyBrain Engineer: the product API.

Plain arguments in, plain JSON-serialisable dicts out. No argparse, no printing, no
`sys.exit`, no global CLI state. Any surface - the CLI today, an MCP server next -
can wrap these functions directly.

    import sys; sys.path.insert(0, "app")
    from engineer import api
    plan = api.plan_change("Rename the helper", selected_files=["app/env_loader.py"])
    patch = api.propose_patch(plan["run_id"])

Errors vs refusals
------------------
An *expected refusal* is a normal return value: the result carries `status`,
`reason`, and the artifact path, and the artifact is written to disk. A blocked
patch, a failing checker, and an insufficient-context replan are all refusals, not
exceptions. Exceptions are reserved for genuine faults - a missing task id, a path
outside the repo, a denied secret, an unparseable diff, a pinned provider with no
route, a missing prompt file.

Gates
-----
Three side effects are gated, each by exactly one function:

* `apply_patch` is the only function that writes to the working tree. It refuses a
  blocked patch artifact even with `yes=True`, revalidates recorded context hashes
  immediately before writing, and backs up every file it touches.
* `verify_applied(run=True)` is the only function that executes anything, and only
  commands from the exact allowlist.
* `approve_lesson` is the behavior-change gate. It requires held-out
  non-regression evidence, permits only one unvalidated promotion, and captures an
  exact rollback snapshot of prompts, canonical behavior, and approved lessons.
* `run_autonomous` may apply and verify only inside a disposable checkout bound to
  a previously declared immutable envelope. Durable evidence stays in real
  `brain_v2`; the real source tree is never its Apply target.

Model-agnosticism
-----------------
No model claim is treated as evidence. Model output is parsed permissively and then
validated strictly by deterministic checkers; verification means running a command
and reading its exit code. Swapping in a much worse model degrades output quality
only - it cannot unlock a side effect.
"""

from __future__ import annotations as _annotations

import json as _json
import re as _re
from datetime import datetime as _datetime
from pathlib import Path as _Path

import rate_limiter as _rate_limiter
from repository_retrieval import secret_content_block_reason as _secret_content_block_reason

# Implementation modules are imported under private aliases so that `dir(api)` and
# `help(api)` show the local product surface and nothing else. Remote capability
# surfaces must never reflect over ``__all__``: MCP uses the separate, explicit
# ``MCP_TOOL_NAMES`` allowlist below.
from . import config as _config, jspace as _jspace, prompts as _prompts
from .apply import engineer_apply_patch as _engineer_apply_patch
from .apply import engineer_revert_applied_patch as _engineer_revert_applied_patch
from .apply import engineer_trial_patch as _engineer_trial_patch
from .autonomous import engineer_autonomous_run as _engineer_autonomous_run
from .behavior import (
    engineer_promote_lesson as _engineer_promote_lesson,
    engineer_rollback_behavior as _engineer_rollback_behavior,
    engineer_validate_behavior_promotion as _engineer_validate_behavior_promotion,
)
from .config import RoleConfig
from .consent import CONSENT_ACTORS as _CONSENT_ACTORS
from .driver import engineer_task as _engineer_task
from .loop import (
    engineer_check_run as _engineer_check_run,
    engineer_plan as _engineer_plan,
)
from .replan import engineer_replan as _engineer_replan
from .envelope import declare_task_envelope as _declare_task_envelope
from .patch import engineer_patch as _engineer_patch
from .prompts import PromptError
from .retrieval import (
    engineer_context_approve as _engineer_context_approve,
    engineer_context_execute as _engineer_context_execute,
    engineer_context_invalidate as _engineer_context_invalidate,
    engineer_context_request as _engineer_context_request,
    engineer_repo_map as _engineer_repo_map,
    engineer_retrieve as _engineer_retrieve,
)
from .review import (
    engineer_feedback as _engineer_feedback,
    engineer_patch_review as _engineer_patch_review,
    engineer_reject_lesson as _engineer_reject_lesson,
    engineer_review as _engineer_review,
    engineer_version as _engineer_version,
)
from .verify import engineer_verify as _engineer_verify
from .workspace import (
    call_agent as _call_agent,
    list_nvidia_models as _list_nvidia_models,
    list_runs as _list_runs,
    model_probe as _model_probe,
    model_registry_refresh as _model_registry_refresh,
    model_registry_show as _model_registry_show,
    read_file as _read_file,
    read_run as _read_run,
    run_summary as _run_summary,
    save_outbox as _save_outbox,
    status as _status,
)
from .util import (
    safe_workspace_id as _safe_workspace_id,
    selected_context_block_reason as _selected_context_block_reason,
)


__all__ = [
    # Engineer loop
    "plan_change",
    "run_task",
    "declare_envelope",
    "run_autonomous",
    "recheck_plan",
    "replan_with_evidence",
    "request_context",
    "approve_context",
    "execute_context",
    "invalidate_context",
    "retrieve",
    "map_repository",
    "propose_patch",
    "review_patch",
    "apply_patch",
    "verify_applied",
    "record_review",
    "record_feedback",
    "propose_lesson",
    "approve_lesson",
    "validate_behavior_promotion",
    "rollback_behavior",
    "reject_lesson",
    "record_version",
    "audit_run",
    # Introspection
    "describe",
    "prompt_audit",
    "role_config",
    # Workspace
    "workspace_status",
    "list_artifacts",
    "read_named_file",
    "read_artifact",
    "show_model_registry",
    "refresh_model_registry",
    "probe_model",
    "call_model",
    "save_codex_request",
    "list_nvidia_catalog",
    # Errors
    "PromptError",
]


# The MCP surface is deliberately separate from ``__all__``.  ``__all__`` is the
# local Python/CLI product API and contains human-gated calls; MCP must never derive
# tools from it.  This tuple is the complete remote allowlist.
MCP_TOOL_NAMES = (
    "describe",
    "plan_change",
    "recheck_plan",
    "map_repository",
    "request_context",
    "approve_context",
    "execute_context",
    "replan_with_evidence",
    "propose_patch",
    "review_patch",
    "run_autonomous",
    "audit_run",
    "read_artifact",
    "list_artifacts",
    "prompt_audit",
)

_MCP_ALLOWED_ARGUMENTS = {
    "describe": set(),
    "plan_change": {"task", "selected_files", "project", "use_approved_lessons"},
    "recheck_plan": {"run_id"},
    "map_repository": {"task_id"},
    "request_context": {"task_id", "request"},
    "approve_context": {"task_id", "request_id"},
    "execute_context": {"task_id", "request_id"},
    "replan_with_evidence": {"run_id"},
    "propose_patch": {"run_id"},
    "review_patch": {"patch_id"},
    "run_autonomous": {"task_id"},
    "audit_run": {"target"},
    "read_artifact": {"path"},
    "list_artifacts": set(),
    "prompt_audit": set(),
}

_MCP_REQUIRED_ARGUMENTS = {
    "plan_change": {"task"},
    "recheck_plan": {"run_id"},
    "map_repository": {"task_id"},
    "request_context": {"task_id", "request"},
    "approve_context": {"task_id", "request_id"},
    "execute_context": {"task_id", "request_id"},
    "replan_with_evidence": {"run_id"},
    "propose_patch": {"run_id"},
    "review_patch": {"patch_id"},
    "run_autonomous": {"task_id"},
    "audit_run": {"target"},
    "read_artifact": {"path"},
}

_MCP_CONTEXT_REQUEST_KEYS = {
    "request_id",
    "task_id",
    "external_retrieval",
    "requested_items",
}

_MCP_CONTEXT_ITEM_KEYS = {
    "item_id",
    "intent",
    "reason",
    "requested_roots",
    "path",
    "query",
    "symbol",
    "target",
    "include_tests",
}

_MCP_CONSENT_KEYS = {
    "actor",
    "approval",
    "approve",
    "approved",
    "confirm",
    "confirmation",
    "confirmed",
    "force",
    "pre_authorized",
    "skip",
    "token",
    "yes",
}

_MCP_MAX_ARTIFACT_BYTES = 2_000_000


def _mcp_model_state_preflight() -> str:
    """Verify model calls can only touch the existing canonical cooldown artifact."""

    expected_config = (_config.ROOT / "config" / "model_rate_limits.json").absolute()
    configured_config = _Path(_rate_limiter.CONFIG_PATH).absolute()
    if configured_config != expected_config or configured_config.is_symlink():
        return "Model rate-limit configuration path is not the canonical repo file."
    if not configured_config.is_file():
        return (
            "Model rate-limit configuration is missing; MCP refuses to let the "
            "provider layer create configuration outside Engineer artifacts."
        )
    expected_state = (
        _config.ROOT / "brain" / "model_health" / "provider_rate_limit_state.json"
    ).absolute()
    configured_state = _Path(_rate_limiter.STATE_PATH).absolute()
    if configured_state != expected_state or configured_state.is_symlink():
        return "Provider cooldown state path is not the canonical operational artifact."
    return ""


# --------------------------------------------------------------------- planning

def declare_envelope(
    task_id: str,
    task: str,
    allowed_roots: list[str],
    allowed_intents: list[str],
    write_targets: list[str],
    verification_commands: list[str],
    expires_at: str,
    retrieval_budget: dict | None = None,
    yes: bool = False,
) -> dict:
    """Declare immutable authority for one unattended run.

    This is a human-operated declaration gate. It binds an exact task, retrieval
    scope and budgets, write-target set, verification-command set, and expiry.
    It cannot authorize behavior files or real-tree Apply, and it cannot be edited
    or widened after creation.
    """
    return _declare_task_envelope(
        task_id,
        task,
        allowed_roots,
        allowed_intents,
        write_targets,
        verification_commands,
        expires_at,
        retrieval_budget=retrieval_budget,
        yes=yes,
    )


def run_autonomous(task_id: str) -> dict:
    """Execute an existing envelope in a disposable code checkout.

    The sole argument is the envelope-bound task id. The run writes durable
    provenance and evidence to the real `brain_v2` tree, while shared Apply and
    Verify machinery target only a temporary checkout outside the repository.
    Because that checkout is not an OS sandbox, unattended Verify accepts only
    exact sealed ``autonomous_safe`` static specs. The final report records both
    the protected-source equality proof and the additive durable-evidence delta.
    """
    return _engineer_autonomous_run(task_id)


def plan_change(
    task: str,
    selected_files: list[str] | None = None,
    project: str = "",
    use_approved_lessons: bool = False,
    allow_web: bool = False,
    compare_baseline: bool = False,
    retrieval_roots: list[str] | None = None,
    retrieval_intents: list[str] | None = None,
    max_chars_per_call: int | None = None,
) -> dict:
    """Plan a change against explicitly selected files.

    Preconditions: `task` is non-empty; every path in `selected_files` exists, is a
    file inside the repo, and is not secret-shaped. A missing or denied file raises.

    Writes: a J-space workspace with the exact excerpts the model saw, and a plan
    artifact (`.md` + `.json`) under the runs directory - including for refusals.

    Gate: none. This function proposes only; it never edits the working tree.

    Refusal: a code-modification task with no selected files returns
    `blocked_plan=True` and `checker_status="fail"` with no executable prompt.

    Returns the plan payload: `run_id`, `checker_status`, `checker_report`,
    `path`/`jsonPath`, `cost_accounting`, and the plan fields themselves.
    """
    return _engineer_plan(
        task,
        selected_files=selected_files,
        project=project,
        use_approved_lessons=use_approved_lessons,
        allow_web=allow_web,
        compare_baseline=compare_baseline,
        retrieval_roots=retrieval_roots,
        retrieval_intents=retrieval_intents,
        max_chars_per_call=max_chars_per_call,
    )


def run_task(
    task: str,
    selected_files: list[str] | None = None,
    project: str = "",
    use_approved_lessons: bool = True,
    allow_web: bool = False,
    retrieval_roots: list[str] | None = None,
    retrieval_intents: list[str] | None = None,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
    stop_after: str = "patch",
    max_chars_per_call: int | None = None,
) -> dict:
    """Chain plan -> check -> patch in one call, stopping before the Apply gate.

    Preconditions: `stop_after` is one of "plan", "check", "patch".

    Writes: whatever the chained stages write.

    Gate: none. Apply and Verify are deliberately unreachable from here.

    Refusal: a failing checker halts the chain with `halted="blocked_by_checker"`
    unless `override_failed_check=True`. Stage errors are captured in `steps`, not
    raised, so a partial run is still reportable.
    """
    return _engineer_task(
        task,
        selected_files=selected_files,
        project=project,
        use_approved_lessons=use_approved_lessons,
        allow_web=allow_web,
        retrieval_roots=retrieval_roots,
        retrieval_intents=retrieval_intents,
        allow_package_json=allow_package_json,
        allow_delete=allow_delete,
        override_failed_check=override_failed_check,
        stop_after=stop_after,
        max_chars_per_call=max_chars_per_call,
    )


def recheck_plan(run_id: str, write: bool = False) -> dict:
    """Re-run the deterministic checker against a saved plan and current disk state.

    Preconditions: `run_id` names a plan JSON inside the repo.

    Writes: nothing unless `write=True`, which updates the plan artifact, advances
    the checker checkpoint, and may append candidate lessons.

    Gate: none.

    Returns `checkerStatus`, `warnings`, `failedRules`, `suggestedCorrections`.
    """
    return _engineer_check_run(run_id, write=write)


def replan_with_evidence(run_id: str) -> dict:
    """Revise a plan using only retrieved evidence, with claims cited to evidence ids.

    Preconditions: the task has at least one completed retrieval excerpt, and no
    recorded context hash has changed. Both raise if unmet - replanning against
    stale evidence is worse than not replanning.

    Writes: a replan revision under the J-space, plus the updated plan artifact.

    Gate: none, but this is the gate feeder - `propose_patch` refuses once retrieval
    has happened until a grounded replan exists.

    Refusal: an unparseable model response yields `context_sufficiency="unavailable"`
    and `ok=False`, which keeps patch generation blocked.
    """
    return _engineer_replan(run_id)


# -------------------------------------------------------------------- retrieval

def request_context(task_id: str, request_json: str = "", request_file: str = "") -> dict:
    """Classify a request for more repository context. Executes nothing.

    Preconditions: exactly one of `request_json` / `request_file`; the request's
    `task_id` matches `task_id` exactly. A secret-shaped request file is refused.

    Writes: a classified context-request artifact and a J-space record.

    Gate: none. Items outside pre-authorized roots are marked `request_approval` and
    wait for `approve_context`.
    """
    return _engineer_context_request(task_id, request_json=request_json, request_file=request_file)


def approve_context(task_id: str, request_id: str, yes: bool = False) -> dict:
    """Human-approve a repository-local scope expansion. Retrieves nothing.

    Preconditions: `yes=True` (raises otherwise) and at least one item is awaiting
    approval.

    Writes: widened `allowed_roots`/`allowed_intents` in the J-space permissions and
    the updated request artifact.

    Gate: this is the scope gate. Nothing else can widen retrieval permissions.
    """
    return _engineer_context_approve(task_id, request_id, yes=yes)


def execute_context(task_id: str, request_id: str) -> dict:
    """Run the retrievals for items already pre-authorized or human-approved.

    Preconditions: the request exists.

    Writes: one retrieval artifact per executed item, plus J-space provenance.

    Gate: none beyond the per-item mode check - items in any other mode are reported
    in `blocked_items` rather than executed.
    """
    return _engineer_context_execute(task_id, request_id)


def invalidate_context(task_id: str, request_id: str, yes: bool = False) -> dict:
    """Human-gated discard of one context request and its retrieval evidence.

    Preconditions: `yes=True`. The named request must exist and not already be
    invalidated.

    Writes: marks the request invalidated, removes its retrieval sources from the
    manifest, deletes linked retrieval artifacts, and records provenance.

    Gate: human confirmation only. Never auto-invalidates. Afterward the human
    may re-run `replan_with_evidence` on remaining fresh evidence.
    """
    return _engineer_context_invalidate(task_id, request_id, yes=yes)


def retrieve(task_id: str, request_json: str = "", request_file: str = "") -> dict:
    """Run one bounded repository retrieval directly.

    Preconditions: the task exists, the call budget is not exhausted (raises), and
    the request is within `allowed_roots`/`allowed_intents`.

    Writes: a retrieval artifact with per-excerpt hashes and a J-space source record.

    Gate: none. Results are untrusted evidence; injection markers are flagged and
    never obeyed. External retrieval is disabled.
    """
    return _engineer_retrieve(task_id, request_json=request_json, request_file=request_file)


def map_repository(task_id: str, max_files: int = 200) -> dict:
    """Build a navigation-only map of authorized roots: paths and metadata, no content.

    Preconditions: the task exists and has authorized roots; a second map for the
    same task raises rather than silently replacing the first snapshot.

    Writes: a repository-map artifact and a J-space record.

    Gate: none. The map grants no read or write permission.
    """
    return _engineer_repo_map(task_id, max_files=max_files)


# ------------------------------------------------------------------------ patch

def propose_patch(
    run_id: str,
    selected_files: list[str] | None = None,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
) -> dict:
    """Generate a unified diff proposal. Never writes to the working tree.

    Preconditions: `run_id` names a saved plan. Write scope is composed only from
    human-selected files - retrieval never widens it.

    Writes: a patch artifact, blocked or proposed, always.

    Gate: none; `apply_allowed` is always False on generation.

    Refusal: returns `patch_status="blocked"` with `limitations` when the checker
    failed, the plan is blocked, the grounding gate fails, risk blockers apply, no
    target files exist, or the generated diff fails the patch checker (including a
    read-only dry-run proving it actually applies).
    """
    return _engineer_patch(
        run_id,
        selected_files=selected_files,
        allow_package_json=allow_package_json,
        allow_delete=allow_delete,
        override_failed_check=override_failed_check,
    )


def review_patch(patch_id: str) -> dict:
    """Review a proposed patch: deterministic checks plus an independent model reviewer.

    Preconditions: `patch_id` resolves to a patch artifact.

    Writes: a patch-review artifact and a J-space checkpoint.

    Gate: none, and `apply_performed` is always False.

    Refusal: `review_status="fail"` when the deterministic checker fails or the
    reviewer requests changes; `"unavailable"` when no reviewer route responded -
    which is reported, never silently treated as approval.
    """
    return _engineer_patch_review(patch_id)


def apply_patch(
    patch_id: str,
    yes: bool = False,
    require_clean_git: bool = False,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
    allow_high_risk: bool = False,
) -> dict:
    """Apply a proposed patch to the working tree. THE write gate.

    Preconditions: the artifact is not blocked (unless ``override_failed_check``),
    its diff is non-empty, and every recorded context hash still matches.
    ``yes=True`` is required for real-repo writes.

    Writes: the target files, a full backup of every touched file, and an
    applied-patch artifact (always, so a write always leaves an audit trail).

    After Apply: ``git commit`` to keep, or ``revert_applied_patch`` to discard.
    """
    return _engineer_apply_patch(
        patch_id,
        yes=yes,
        require_clean_git=require_clean_git,
        allow_package_json=allow_package_json,
        allow_delete=allow_delete,
        override_failed_check=override_failed_check,
        allow_high_risk=allow_high_risk,
    )


def revert_applied_patch(applied_patch_id: str, yes: bool = False) -> dict:
    """Restore the working tree from an Apply backup."""
    return _engineer_revert_applied_patch(applied_patch_id, yes=yes)


def trial_patch(
    patch_id: str,
    yes: bool = False,
    *,
    run_verify: bool = True,
    save_verify: bool = True,
    auto_revert_on_fail: bool = True,
    require_clean_git: bool = False,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
    allow_high_risk: bool = False,
) -> dict:
    """Apply, verify, and revert automatically when verification fails."""
    return _engineer_trial_patch(
        patch_id,
        yes=yes,
        run_verify=run_verify,
        save_verify=save_verify,
        auto_revert_on_fail=auto_revert_on_fail,
        require_clean_git=require_clean_git,
        allow_package_json=allow_package_json,
        allow_delete=allow_delete,
        override_failed_check=override_failed_check,
        allow_high_risk=allow_high_risk,
    )


def verify_applied(applied_patch_id: str, run: bool = False, save: bool = False) -> dict:
    """Prepare and optionally execute verification for an applied patch. THE exec gate.

    Preconditions: `applied_patch_id` resolves to an applied-patch artifact.

    Writes: a verification artifact only when `save=True`; `path`/`jsonPath` are
    empty strings otherwise, so a caller is never handed a path that does not exist.

    Gate: this one. Only exact allowlisted commands run - the enumerated specs plus
    generated `py_compile` specs for applied Python targets. There is no shell.
    Unrecognised commands are recorded as skipped.

    `verification_passed` is True only when commands actually executed and all
    passed. `verification_incomplete` stays True whenever anything was skipped.
    """
    return _engineer_verify(applied_patch_id, run=run, save=save)


# ------------------------------------------------------------- review & lessons

def record_review(
    run_id: str,
    changed_files: list[str] | None = None,
    codex_summary: str = "",
    compile_passed: bool | None = None,
    tests_passed: bool | None = None,
    files_touched_correct: bool | None = None,
    manual_correction: str = "",
    notes: str = "",
) -> dict:
    """Record the outcome of executing a plan, and score it deterministically.

    Preconditions: `run_id` names a saved plan.

    Writes: a review artifact, an `eval_history` row, and - when the outcome warrants
    it - candidate lessons.

    Gate: none; any lesson created is a candidate awaiting `approve_lesson`.
    """
    return _engineer_review(
        run_id,
        changed_files=changed_files,
        codex_summary=codex_summary,
        compile_passed=compile_passed,
        tests_passed=tests_passed,
        files_touched_correct=files_touched_correct,
        manual_correction=manual_correction,
        notes=notes,
    )


def record_feedback(
    run_id: str,
    usefulness: int,
    correctness: int,
    accepted: bool,
    notes: str,
    codex_success: bool | None = None,
    compile_passed: bool | None = None,
    files_touched_correct: bool | None = None,
    manual_correction: str = "",
    lesson_needed: bool | None = None,
) -> dict:
    """Append human feedback for a run; see `propose_lesson` for the lesson it may create.

    Preconditions: `run_id` is non-empty.

    Writes: an `eval_history` row and, when warranted, one candidate lesson.

    Gate: none; `lesson_needed=False` suppresses lesson creation outright.
    """
    return _engineer_feedback(
        run_id,
        usefulness=usefulness,
        correctness=correctness,
        accepted=accepted,
        notes=notes,
        codex_success=codex_success,
        compile_passed=compile_passed,
        files_touched_correct=files_touched_correct,
        manual_correction=manual_correction,
        lesson_needed=lesson_needed,
    )


#: Lessons are proposed by recording feedback; there is no separate propose call,
#: because a lesson with no run behind it has no provenance.
propose_lesson = record_feedback


def approve_lesson(
    lesson_id: str,
    evidence_file: str = "",
    yes: bool = False,
) -> dict:
    """Promote one candidate lesson through the behavior-change gate.

    Promotion requires explicit confirmation plus held-out task evidence that is
    disjoint from the lesson's source task and scores no worse than baseline.
    Exactly one promotion may remain unvalidated. Before the approved set changes,
    the gate snapshots exact prompt, canonical-behavior, and approved-lesson bytes
    so `rollback_behavior` can restore them.
    """
    return _engineer_promote_lesson(lesson_id, evidence_file, yes=yes)


def validate_behavior_promotion(
    promotion_id: str,
    evidence_file: str,
    yes: bool = False,
) -> dict:
    """Validate installed behavior with fresh held-out non-regression evidence."""
    return _engineer_validate_behavior_promotion(
        promotion_id,
        evidence_file,
        yes=yes,
    )


def rollback_behavior(promotion_id: str, yes: bool = False) -> dict:
    """Restore and digest-check the exact behavior snapshot from a promotion."""
    return _engineer_rollback_behavior(promotion_id, yes=yes)


def reject_lesson(lesson_id: str, reason: str) -> dict:
    """Reject one candidate lesson with a recorded reason.

    Preconditions: the lesson exists as a candidate, is undecided, and `reason` is
    non-empty. All raise otherwise.

    Writes: a `rejected_lessons` row.

    Gate: none. Canonical behavior is not edited.
    """
    return _engineer_reject_lesson(lesson_id, reason)


def record_version(summary: str) -> dict:
    """Stamp a behavior version. Preconditions: `summary` non-empty. Writes a
    `version_history` row. Canonical behavior is not edited."""
    return _engineer_version(summary)


def audit_run(target: str, use_model: bool = False) -> dict:
    """Summarize any saved artifact - plan, patch, applied patch, verification, review.

    Preconditions: `target` resolves to an artifact id or repo path (raises if not).

    Writes: nothing.

    Gate: none. The deterministic summary is always produced; `use_model=True` adds
    an optional `utility_light` summary that never escalates to an elite route and
    falls back silently to the deterministic one.
    """
    return _run_summary(target, use_model=use_model)


# ------------------------------------------------------------------ inspection

def role_config() -> RoleConfig:
    """The active role descriptor: the only four role-specific surfaces.

    Prompt files, retrieval intents, checker rules, and the verification mechanism.
    A second role supplies a second `RoleConfig`; the loop does not change.
    """
    return _config.ENGINEER


def prompt_audit() -> dict:
    """Every prompt data file and the placeholders it declares."""
    return _prompts.audit()


def describe() -> dict:
    """Machine-readable description of the complete local Python API.

    This intentionally includes human-gated calls and therefore must never be used
    to generate the MCP surface. ``invoke_mcp_tool("describe")`` returns the remote
    allowlist instead.
    """
    import inspect

    entries = []
    for name in __all__:
        if name in {"describe", "PromptError"}:
            continue
        value = globals().get(name)
        if not callable(value):
            continue
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = (inspect.getdoc(value) or "").strip().splitlines()
        entries.append(
            {
                "name": name,
                "signature": signature,
                "summary": doc[0] if doc else "",
                "writes_working_tree": name == "apply_patch",
                "writes_disposable_checkout": name == "run_autonomous",
                "executes_commands": name in {"verify_applied", "run_autonomous"},
            }
        )
    return {
        "api": "companybrain.engineer.api",
        "role": _config.ENGINEER.name,
        "functions": entries,
        "gates": {
            "working_tree_write": "apply_patch",
            "command_execution": "verify_applied",
            "lesson_promotion": "approve_lesson",
            "behavior_validation": "validate_behavior_promotion",
            "behavior_rollback": "rollback_behavior",
            "scope_expansion": "approve_context",
            "autonomous_authority_declaration": "declare_envelope",
        },
    }


# ------------------------------------------------------------------- workspace

def workspace_status() -> dict:
    """Provider, project, and artifact counts. Writes nothing."""
    return _status()


def list_artifacts() -> dict:
    """The 30 most recently modified run and artifact files. Writes nothing."""
    return _list_runs()


def read_named_file(label: str) -> dict:
    """Read one registered Company Brain file by label. Raises on an unknown label."""
    return _read_file(label)


def read_artifact(path_text: str) -> dict:
    """Read any repo-local run file. Raises on a path outside the repo."""
    return _read_run(path_text)


def show_model_registry() -> dict:
    """The current model registry with health and routing state. Writes nothing."""
    return _model_registry_show()


def refresh_model_registry() -> dict:
    """Re-derive registry state from catalogs. Writes the registry file."""
    return _model_registry_refresh()


def probe_model(provider: str, model_id: str, enable_if_ok: bool = False) -> dict:
    """Probe one provider/model route and record the result.

    Gate: `enable_if_ok=True` is required to enable a route; a successful probe
    alone only recommends enabling it. Errors are redacted before being recorded.
    """
    return _model_probe(provider, model_id, enable_if_ok=enable_if_ok)


def call_model(provider: str, role: str, prompt: str) -> dict:
    """Free-form model call outside the Engineer loop; saves a run transcript."""
    return _call_agent(provider, role, prompt)


def save_codex_request(prompt: str) -> dict:
    """Write a Codex bridge request to the outbox. Raises on empty input."""
    return _save_outbox(prompt)


def list_nvidia_catalog() -> dict:
    """List NVIDIA-compatible models and cache them under config/."""
    return _list_nvidia_models()


# --------------------------------------------------------------- safe MCP seam

def _mcp_refusal(
    reason: str,
    *,
    status: str = "refused",
    artifact_path: str = "",
    next_action: str = "Correct the request and call a listed MCP tool; no gate was advanced.",
    **extra,
) -> dict:
    """The one refusal envelope used at the remote boundary."""

    return {
        "status": status,
        "reason": reason,
        "artifact_path": artifact_path,
        "next_action": next_action,
        **extra,
    }


def _mcp_safe_error(exc: Exception) -> str:
    """Flatten an exception without returning credentials or a traceback."""

    text = str(exc).replace("\r", " ").replace("\n", " ").strip()
    text = _re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = _re.sub(r"(?i)(api[_-]?key[=:\s]+)[^\s,;]+", r"\1[REDACTED]", text)
    text = _re.sub(r"(?i)(token[=:\s]+)[^\s,;]+", r"\1[REDACTED]", text)
    return text[:900] or "The operation was refused without an error detail."


def _mcp_role_config(role: str) -> RoleConfig:
    """Resolve the explicit current-role registry; an unknown role never falls back."""

    if not isinstance(role, str):
        raise RuntimeError("role must be a string")
    requested = role.strip() or "engineer"
    roles = {_config.ENGINEER.name: _config.ENGINEER}
    selected = roles.get(requested)
    if selected is None:
        raise RuntimeError(
            f"Unsupported role {requested!r}; configured roles: {', '.join(sorted(roles))}"
        )
    return selected


def _mcp_cli_help(tool_name: str) -> str:
    actions = {
        "plan_change": "engineer-plan",
        "recheck_plan": "engineer-check-run",
        "map_repository": "engineer-repo-map",
        "request_context": "engineer-context-request",
        "approve_context": "engineer-context-approve",
        "execute_context": "engineer-context-execute",
        "replan_with_evidence": "engineer-replan",
        "propose_patch": "engineer-patch",
        "review_patch": "engineer-patch-review",
        "audit_run": "run-summary",
    }
    action = actions.get(tool_name)
    if action:
        return f"Use the human-operated CLI for gated options: python scripts/company_brain_action.py {action} --help"
    return "Use the human-operated CLI for gated actions: python scripts/company_brain_action.py --help"


def _mcp_argument_refusal(tool_name: str, names: list[str]) -> dict:
    rendered = ", ".join(names)
    return _mcp_refusal(
        f"MCP tool {tool_name} refuses unsupported or escalation argument(s): {rendered}. "
        "Arguments are rejected rather than ignored.",
        next_action=_mcp_cli_help(tool_name),
        refused_arguments=names,
    )


def _mcp_validate_argument_types(tool_name: str, arguments: dict) -> str:
    string_fields = {
        "plan_change": {"task", "project"},
        "recheck_plan": {"run_id"},
        "map_repository": {"task_id"},
        "request_context": {"task_id"},
        "approve_context": {"task_id", "request_id"},
        "execute_context": {"task_id", "request_id"},
        "replan_with_evidence": {"run_id"},
        "propose_patch": {"run_id"},
        "review_patch": {"patch_id"},
        "run_autonomous": {"task_id"},
        "audit_run": {"target"},
        "read_artifact": {"path"},
    }.get(tool_name, set())
    for name in string_fields:
        if name not in arguments:
            continue
        value = arguments[name]
        if not isinstance(value, str) or (name != "project" and not value.strip()):
            return f"{name} must be a non-empty string"
    identifier_fields = {
        "recheck_plan": {"run_id"},
        "map_repository": {"task_id"},
        "request_context": {"task_id"},
        "approve_context": {"task_id", "request_id"},
        "execute_context": {"task_id", "request_id"},
        "replan_with_evidence": {"run_id"},
        "propose_patch": {"run_id"},
        "review_patch": {"patch_id"},
        "run_autonomous": {"task_id"},
    }.get(tool_name, set())
    for name in identifier_fields:
        value = str(arguments.get(name) or "")
        if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", value):
            return (
                f"{name} must start with a letter or number and contain only "
                "letters, numbers, dot, underscore, or hyphen"
            )
    if tool_name == "plan_change":
        selected = arguments.get("selected_files", [])
        if not isinstance(selected, list) or any(
            not isinstance(item, str) or not item.strip() for item in selected
        ):
            return "selected_files must be an array of non-empty repository path strings"
        lessons = arguments.get("use_approved_lessons", False)
        if not isinstance(lessons, bool):
            return "use_approved_lessons must be a boolean"
    if tool_name == "request_context" and not isinstance(arguments.get("request"), dict):
        return "request must be a JSON object"
    return ""


def _mcp_validate_context_request(request: dict, task_id: str) -> str:
    unexpected = sorted(set(request) - _MCP_CONTEXT_REQUEST_KEYS)
    if unexpected:
        return (
            "Context request refuses unsupported or consent-shaped field(s): "
            + ", ".join(unexpected)
        )
    if request.get("external_retrieval") not in {None, False}:
        return "Context request cannot enable external_retrieval"
    request_task = request.get("task_id")
    if not isinstance(request_task, str) or request_task.strip() != task_id.strip():
        return "request.task_id must exactly match task_id"
    if not isinstance(request.get("request_id"), str) or not str(request["request_id"]).strip():
        return "request.request_id must be a non-empty string"
    if not _re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}",
        str(request["request_id"]),
    ):
        return (
            "request.request_id must start with a letter or number and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    items = request.get("requested_items")
    if not isinstance(items, list) or not 1 <= len(items) <= 12:
        return "request.requested_items must contain between 1 and 12 objects"
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return f"request.requested_items[{index}] must be an object"
        unexpected_item = sorted(set(item) - _MCP_CONTEXT_ITEM_KEYS)
        if unexpected_item:
            return (
                f"request.requested_items[{index}] refuses unsupported or consent-shaped "
                f"field(s): {', '.join(unexpected_item)}"
            )
        consent = sorted(set(item) & _MCP_CONSENT_KEYS)
        if consent:
            return (
                f"request.requested_items[{index}] cannot supply approval field(s): "
                + ", ".join(consent)
            )
        roots = item.get("requested_roots")
        if not isinstance(roots, list) or not roots or any(
            not isinstance(value, str) or not value.strip() for value in roots
        ):
            return f"request.requested_items[{index}] must request one or more root strings"
        for name in ("item_id", "intent", "reason", "path", "query", "symbol", "target"):
            if name in item and not isinstance(item[name], str):
                return f"request.requested_items[{index}].{name} must be a string"
        item_id = str(item.get("item_id") or "")
        if item_id and not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", item_id):
            return (
                f"request.requested_items[{index}].item_id must start with a letter "
                "or number and contain only letters, numbers, dot, underscore, or hyphen"
            )
        if "include_tests" in item and not isinstance(item["include_tests"], bool):
            return f"request.requested_items[{index}].include_tests must be a boolean"
    return ""


def _mcp_artifact_roots(role: RoleConfig) -> tuple[_Path, _Path]:
    employee_root = role.prompts_dir.parent.resolve()
    eval_root = (_config.BRAIN_V2_DIR / "evals").resolve()
    return employee_root, eval_root


def _mcp_inside(path: _Path, root: _Path) -> bool:
    return path == root or root in path.parents


def _mcp_is_capability_material(path: _Path) -> bool:
    nonce_root = (_config.ENGINEER_ENVELOPES_DIR / "nonces").resolve()
    resolved = path.resolve()
    return _mcp_inside(resolved, nonce_root)


def _mcp_resolve_artifact(
    path_text: str,
    role: RoleConfig,
    *,
    allow_id: bool = False,
) -> _Path:
    text = str(path_text or "").strip()
    if not text:
        raise RuntimeError("Artifact path is required.")
    raw = _Path(text)
    if raw.is_absolute():
        raise RuntimeError("Absolute artifact paths are refused; use a repository-relative path.")
    if ".." in raw.parts:
        raise RuntimeError("Artifact path traversal is refused.")
    roots = _mcp_artifact_roots(role)
    candidates: list[_Path] = []
    if allow_id and len(raw.parts) == 1:
        names = [raw.name] if raw.suffix else [f"{raw.name}.json", f"{raw.name}.md"]
        for root in roots:
            if not root.exists():
                continue
            for name in names:
                candidates.extend(path for path in root.rglob(name) if path.is_file())
        resolved_matches = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if any(_mcp_inside(resolved, root) for root in roots) and not _mcp_is_capability_material(
                resolved
            ):
                resolved_matches.append(resolved)
        unique = sorted(set(resolved_matches), key=lambda value: str(value).lower())
        if len(unique) > 1:
            raise RuntimeError(
                f"Artifact id is ambiguous ({len(unique)} matches); provide its repository-relative path."
            )
        if unique:
            return unique[0]
    candidate = (_config.ROOT / raw).resolve()
    if not any(_mcp_inside(candidate, root) for root in roots):
        raise RuntimeError(
            "Artifact path must be under brain_v2/employees/"
            f"{role.name}/ or brain_v2/evals/."
        )
    if _mcp_is_capability_material(candidate):
        raise RuntimeError("Envelope nonce capability material is never an MCP artifact.")
    if not candidate.exists() or not candidate.is_file():
        raise RuntimeError(f"Artifact file does not exist: {text}")
    return candidate


def _mcp_read_safe_artifact(path: _Path) -> str:
    if _mcp_is_capability_material(path):
        raise RuntimeError("Envelope nonce capability material is never readable through MCP.")
    path_reason = _selected_context_block_reason(path)
    if path_reason:
        raise RuntimeError(f"Artifact secret-path denial: {path_reason}.")
    size = path.stat().st_size
    if size > _MCP_MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"Artifact exceeds the {_MCP_MAX_ARTIFACT_BYTES}-byte MCP read limit."
        )
    content = path.read_text(encoding="utf-8", errors="replace")
    content_reason = _secret_content_block_reason(content)
    if content_reason:
        raise RuntimeError(
            "Artifact content was denied before it could be returned: "
            f"{content_reason}."
        )
    return content


def _mcp_artifact_path(payload: dict) -> str:
    return str(
        payload.get("jsonPath")
        or payload.get("path")
        or payload.get("runPath")
        or ""
    )


def _mcp_result(payload: dict, *, status: str = "ok", reason: str = "", next_action: str = "") -> dict:
    result = dict(payload)
    # These four fields are the governed envelope. Never let model-produced or
    # artifact-loaded keys override the deterministic boundary decision.
    artifact_path = _mcp_artifact_path(result)
    result["status"] = status
    result["reason"] = reason
    result["artifact_path"] = artifact_path
    result["next_action"] = next_action
    return result


def _mcp_load_context_request(
    task_id: str,
    request_id: str,
    role: RoleConfig,
) -> tuple[_Path, dict]:
    run_key = _safe_workspace_id(task_id)
    request_key = _safe_workspace_id(request_id)
    workspace, _, _ = _jspace.paths(run_key)
    path = workspace / "context" / "requests" / f"{request_key}.json"
    resolved = path.resolve()
    if not _mcp_inside(resolved, role.prompts_dir.parent.resolve()):
        raise RuntimeError("Context request escaped the Engineer artifact root.")
    if not resolved.is_file():
        raise RuntimeError(f"Context request not found: {request_key}")
    # Harness-written request metadata under j_space/context/requests/. Path-shaped
    # secret denial targets repository retrieval (and MCP read_artifact); applying it
    # here would refuse legitimate request_ids that merely mention "secret".
    size = resolved.stat().st_size
    if size > _MCP_MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"Artifact exceeds the {_MCP_MAX_ARTIFACT_BYTES}-byte MCP read limit."
        )
    payload = _json.loads(resolved.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise RuntimeError("Context request artifact must contain a JSON object.")
    if str(payload.get("task_id") or "") != run_key or str(payload.get("request_id") or "") != request_key:
        raise RuntimeError("Context request artifact identity does not match the requested task/request.")
    return resolved, payload


def _mcp_context_handoff(
    task_id: str,
    request_id: str,
    role: RoleConfig,
    attempted_arguments: list[str] | None = None,
) -> dict:
    path, payload = _mcp_load_context_request(task_id, request_id, role)
    items = [item for item in payload.get("requested_items") or [] if isinstance(item, dict)]
    pending = [item for item in items if item.get("mode") == "request_approval"]
    if not pending:
        already_approved = bool(items) and all(
            item.get("mode") == "approved"
            and isinstance(item.get("approval"), dict)
            and item["approval"].get("approved") is True
            and item["approval"].get("actor") in _CONSENT_ACTORS
            for item in items
        )
        return _mcp_refusal(
            (
                "Context request is already human-approved; MCP did not change it."
                if already_approved
                else "Context request has no items awaiting human approval."
            ),
            status="already_human_approved" if already_approved else "refused",
            artifact_path=str(path),
            next_action=(
                f"Call execute_context with task_id={task_id} and request_id={request_id}."
                if already_approved
                else "Create a corrected context request; denied items cannot be approved through MCP."
            ),
            task_id=task_id,
            request_id=request_id,
            request_status=str(payload.get("status") or ""),
            refused_arguments=sorted(attempted_arguments or []),
        )
    command = (
        "python scripts/company_brain_action.py engineer-context-approve "
        f"--task-id {task_id} --request-id {request_id} --yes"
    )
    attempted = sorted(attempted_arguments or [])
    attempt_note = (
        " MCP discarded and refused affirmative argument(s): " + ", ".join(attempted) + "."
        if attempted
        else ""
    )
    return _mcp_refusal(
        "An MCP caller cannot supply human consent; approval state was not changed."
        + attempt_note,
        status="human_confirmation_required",
        artifact_path=str(path),
        next_action=command,
        task_id=task_id,
        request_id=request_id,
        request_status=str(payload.get("status") or ""),
        cli_command=command,
        refused_arguments=attempted,
    )


def _mcp_context_is_executable(
    task_id: str,
    request_id: str,
    role: RoleConfig,
) -> tuple[_Path, dict, str]:
    """Allow execute when at least one item is already authorized.

    Executable means mode=pre_authorized (within intake scope) or mode=approved with
    persisted approval records one of the observable consent-attribution actors.
    Items still in request_approval or denied stay blocked by execute itself; MCP
    must not invent consent for them.
    """

    path, payload = _mcp_load_context_request(task_id, request_id, role)
    items = payload.get("requested_items")
    if not isinstance(items, list) or not items:
        return path, payload, "Context request has no executable items."
    executable = 0
    pending_approval = 0
    for item in items:
        if not isinstance(item, dict):
            return path, payload, "Context request contains a malformed item."
        mode = str(item.get("mode") or "")
        approval = item.get("approval")
        if mode == "pre_authorized":
            executable += 1
            continue
        if (
            mode == "approved"
            and isinstance(approval, dict)
            and approval.get("approved") is True
            and approval.get("actor") in _CONSENT_ACTORS
        ):
            executable += 1
            continue
        if mode == "request_approval":
            pending_approval += 1
    if executable:
        return path, payload, ""
    if pending_approval:
        return (
            path,
            payload,
            "Every executable item still requires human CLI approval "
            "(mode=request_approval). MCP cannot supply consent.",
        )
    return path, payload, "Context request has no pre_authorized or human-approved items."


def _mcp_list_role_artifacts(role: RoleConfig) -> dict:
    rows = []
    for root in _mcp_artifact_roots(role):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
                if not _mcp_inside(resolved, root):
                    continue
                if _mcp_is_capability_material(resolved):
                    continue
                if _selected_context_block_reason(resolved):
                    continue
                stat = resolved.stat()
                rows.append(
                    {
                        "name": resolved.name,
                        "path": resolved.relative_to(_config.ROOT.resolve()).as_posix(),
                        "bytes": stat.st_size,
                        "modified": _datetime.fromtimestamp(stat.st_mtime)
                        .astimezone()
                        .isoformat(timespec="seconds"),
                    }
                )
            except (OSError, RuntimeError, ValueError):
                continue
    rows.sort(key=lambda item: (item["modified"], item["path"]), reverse=True)
    limit = 2000
    return {
        "artifacts": rows[:limit],
        "artifacts_scanned": len(rows),
        "truncated": len(rows) > limit,
        "approved_roots": [
            root.relative_to(_config.ROOT.resolve()).as_posix()
            for root in _mcp_artifact_roots(role)
        ],
    }


def _mcp_describe(role: RoleConfig) -> dict:
    return {
        "api": "companybrain.mcp",
        "role": role.name,
        "tools": list(MCP_TOOL_NAMES),
        "workflow": [
            "human CLI declare_envelope (outside MCP)",
            "run_autonomous(task_id) for envelope-bound scratch Apply/Verify",
            "or use the proposal-only sequence:",
            "plan_change",
            "recheck_plan",
            "request_context",
            "execute_context for pre_authorized items",
            "approve_context (handoff only for scope expansion)",
            "human CLI context approval (scope expansion only)",
            "map_repository (optional)",
            "execute_context after human approval when needed",
            "replan_with_evidence",
            "propose_patch",
            "review_patch",
            "human CLI Apply",
        ],
        "artifact_roots": [
            f"brain_v2/employees/{role.name}/",
            "brain_v2/evals/",
        ],
        "write_boundaries": {
            "engineer_artifacts": f"brain_v2/employees/{role.name}/",
            "evaluation_artifacts": "brain_v2/evals/",
            "shared_provider_cooldown_artifact": (
                "brain/model_health/provider_rate_limit_state.json"
            ),
            "real_working_tree_source": "unreachable",
            "disposable_checkout_source": (
                "reachable only through run_autonomous with an integrity-validated, "
                "unexpired task envelope"
            ),
        },
        "human_only_not_exposed": [
            "actual_context_approval",
            "declare_envelope",
            "apply_patch",
            "verify_applied",
            "approve_lesson",
            "validate_behavior_promotion",
            "rollback_behavior",
            "reject_lesson",
            "record_version",
            "call_model",
            "probe_model",
            "refresh_model_registry",
            "context_invalidate",
        ],
        "guarantees": {
            "real_working_tree_write": "unreachable from MCP",
            "scratch_working_tree_write": (
                "run_autonomous only; exact envelope targets through the shared applier"
            ),
            "command_execution": (
                "run_autonomous only; exact sealed autonomous-safe static specs "
                "in the disposable checkout"
            ),
            "scope_expansion": "approve_context is a non-operative CLI handoff",
            "pre_authorized_execute": "intake-scoped items execute without a human CLI round trip",
            "behavior_promotion": (
                "promotion, validation, rollback, prompts, and approved lessons are unreachable from MCP"
            ),
            "unknown_arguments": "rejected before dispatch, never silently ignored",
        },
    }


def _mcp_wrap_tool_result(tool_name: str, payload: dict) -> dict:
    if tool_name == "plan_change":
        blocked = bool(payload.get("blocked_plan")) or payload.get("checker_status") == "fail"
        reason = "; ".join(
            str(value)
            for value in (
                payload.get("limitations")
                or (payload.get("checker_report") or {}).get("failed_rules")
                or []
            )
        )
        return _mcp_result(
            payload,
            status="blocked" if blocked else "ok",
            reason=reason if blocked else "",
            next_action=(
                "Select the missing target files and create a new plan."
                if blocked
                else f"Call recheck_plan with run_id={payload.get('run_id', '')}, then request bounded context if needed."
            ),
        )
    if tool_name == "recheck_plan":
        blocked = payload.get("checkerStatus") == "fail"
        reason = "; ".join(str(value) for value in payload.get("failedRules") or [])
        return _mcp_result(
            payload,
            status="blocked" if blocked else "ok",
            reason=reason if blocked else "",
            next_action=(
                "Create a corrected plan; MCP cannot override a failed checker."
                if blocked
                else "Call request_context for missing evidence or propose_patch if context is sufficient."
            ),
        )
    if tool_name == "request_context":
        request_status = str(payload.get("status") or "")
        denied = [
            str(item.get("decision_reason") or "")
            for item in payload.get("requested_items") or []
            if isinstance(item, dict) and item.get("mode") == "denied"
        ]
        reason = "; ".join(value for value in denied if value)
        next_action = (
            f"Call approve_context with task_id={payload.get('task_id', '')} and "
            f"request_id={payload.get('request_id', '')} to obtain the human CLI handoff."
            if request_status == "approval_required"
            else (
                "Create a corrected bounded request; denied items cannot execute."
                if request_status == "denied"
                else "Call execute_context for pre_authorized items, or approve_context for scope expansion."
            )
        )
        return _mcp_result(payload, status=request_status or "refused", reason=reason, next_action=next_action)
    if tool_name == "execute_context":
        execution_status = str(payload.get("status") or "blocked")
        reason = "; ".join(
            str(item.get("reason") or "")
            for item in payload.get("blocked_items") or []
            if isinstance(item, dict)
        )
        return _mcp_result(
            payload,
            status=execution_status,
            reason=reason,
            next_action=(
                f"Call replan_with_evidence with run_id={payload.get('task_id', '')}."
                if payload.get("retrieval_results")
                else "Resolve the recorded refusal; MCP cannot approve or expand this request."
            ),
        )
    if tool_name == "replan_with_evidence":
        blocked = not bool(payload.get("ok")) or payload.get("checkerStatus") == "fail"
        failed = ((payload.get("checker_report") or {}).get("failed_rules") or [])
        return _mcp_result(
            payload,
            status="blocked" if blocked else "ok",
            reason="; ".join(str(value) for value in failed) if blocked else "",
            next_action=(
                "Request the missing evidence or repair the grounded plan; no override is available."
                if blocked
                else f"Call propose_patch with run_id={payload.get('run_id', '')}."
            ),
        )
    if tool_name == "propose_patch":
        patch_status = str(payload.get("patch_status") or "blocked")
        limitations = payload.get("limitations") or []
        result = _mcp_result(
            payload,
            status=patch_status,
            reason="; ".join(str(value) for value in limitations) if patch_status == "blocked" else "",
            next_action=(
                "Resolve the recorded checker, grounding, scope, or applicability failure."
                if patch_status == "blocked"
                else ""
            ),
        )
        patch_id = str(payload.get("patch_id") or "")
        if patch_status != "blocked" and patch_id:
            command = (
                "python scripts/company_brain_action.py engineer-apply-patch "
                f"--patch-id {patch_id} --yes"
            )
            result["human_apply_command"] = command
            result["next_action"] = command
        return result
    if tool_name == "review_patch":
        review_status = str(payload.get("review_status") or "unavailable")
        patch_id = str(payload.get("patch_id") or "")
        command = (
            "python scripts/company_brain_action.py engineer-apply-patch "
            f"--patch-id {patch_id} --yes"
            if patch_id
            else ""
        )
        return _mcp_result(
            payload,
            status=review_status,
            reason=str(payload.get("reason") or ""),
            next_action=(
                command
                if review_status == "pass" and command
                else "Revise or re-review the patch; an unavailable or failing review is not approval."
            ),
        )
    if tool_name == "run_autonomous":
        run_status = str(payload.get("status") or "refused")
        ready = (
            payload.get("ok") is True
            and run_status == "verified_pending_human_acceptance"
        )
        refusal_reason = "; ".join(
            str(item.get("detail") or "")
            for item in payload.get("refusals") or []
            if isinstance(item, dict) and item.get("detail")
        )
        return _mcp_result(
            payload,
            status=run_status,
            reason="" if ready else refusal_reason or "Autonomous run did not reach verified status.",
            next_action=(
                "Inspect the standalone report, then use the human CLI Apply gate if the "
                "verified patch should be promoted to the real working tree."
                if ready
                else "Inspect the recorded refusal and declare a new immutable envelope if scope must change."
            ),
        )
    next_actions = {
        "map_repository": "Call request_context for specific bounded source evidence.",
        "audit_run": "Inspect the proven fields; absent historical fields remain unknown.",
        "read_artifact": "Inspect the artifact and follow its recorded next action.",
        "list_artifacts": "Pass one returned path to read_artifact or audit_run.",
        "prompt_audit": "Review prompt provenance through the normal repository workflow.",
    }
    return _mcp_result(payload, next_action=next_actions.get(tool_name, ""))


def invoke_mcp_tool(
    tool_name: str,
    arguments: dict | None = None,
    role: str = "engineer",
) -> dict:
    """Invoke exactly one fail-closed MCP tool.

    This is the sole remote dispatch seam. It is intentionally absent from
    ``__all__`` so the local API cannot be reflected into a remote capability set.
    All unlisted arguments reach this function and are refused before an underlying
    Engineer function is called.
    """

    if not isinstance(tool_name, str) or tool_name not in MCP_TOOL_NAMES:
        return _mcp_refusal(
            f"Unknown or unexposed MCP tool: {tool_name!r}.",
            next_action="Call describe or tools/list; no unlisted Python API function is remotely callable.",
        )
    if arguments is None:
        args = {}
    elif isinstance(arguments, dict):
        args = dict(arguments)
    else:
        return _mcp_refusal("Tool arguments must be a JSON object.")
    try:
        role_descriptor = _mcp_role_config(role)
    except Exception as exc:  # noqa: BLE001
        return _mcp_refusal(
            _mcp_safe_error(exc),
            next_action="Use role='engineer'; no fallback role was selected.",
        )

    allowed = _MCP_ALLOWED_ARGUMENTS[tool_name]
    unexpected = sorted(set(args) - allowed)
    missing = sorted(_MCP_REQUIRED_ARGUMENTS.get(tool_name, set()) - set(args))
    if missing:
        return _mcp_refusal(
            "Missing required argument(s): " + ", ".join(missing),
            next_action=f"Call {tool_name} again with the required fields.",
        )
    type_error = _mcp_validate_argument_types(tool_name, args)
    if type_error:
        return _mcp_refusal(type_error, next_action=f"Correct the {tool_name} argument types and retry.")
    if tool_name == "approve_context":
        try:
            return _mcp_context_handoff(
                str(args["task_id"]),
                str(args["request_id"]),
                role_descriptor,
                attempted_arguments=unexpected,
            )
        except Exception as exc:  # noqa: BLE001
            return _mcp_refusal(
                _mcp_safe_error(exc),
                next_action="Create or inspect the context request; MCP still cannot approve it.",
                refused_arguments=unexpected,
            )
    if unexpected:
        return _mcp_argument_refusal(tool_name, unexpected)
    if tool_name in {
        "plan_change",
        "replan_with_evidence",
        "propose_patch",
        "review_patch",
        "run_autonomous",
    }:
        model_state_error = _mcp_model_state_preflight()
        if model_state_error:
            return _mcp_refusal(
                model_state_error,
                next_action=(
                    "Restore config/model_rate_limits.json and the canonical provider "
                    "cooldown-state path before retrying; MCP will not create or redirect them."
                ),
            )

    try:
        if tool_name == "describe":
            return _mcp_result(
                _mcp_describe(role_descriptor),
                next_action="Call plan_change to start a governed proposal.",
            )
        if tool_name == "plan_change":
            payload = plan_change(
                str(args["task"]),
                selected_files=list(args.get("selected_files") or []),
                project=str(args.get("project") or ""),
                use_approved_lessons=bool(args.get("use_approved_lessons", False)),
                allow_web=False,
                compare_baseline=False,
                retrieval_roots=None,
                retrieval_intents=None,
            )
        elif tool_name == "recheck_plan":
            payload = recheck_plan(str(args["run_id"]), write=False)
        elif tool_name == "map_repository":
            payload = map_repository(str(args["task_id"]), max_files=200)
        elif tool_name == "request_context":
            request = dict(args["request"])
            request_error = _mcp_validate_context_request(request, str(args["task_id"]))
            if request_error:
                return _mcp_refusal(
                    request_error,
                    next_action=(
                        "Submit only a structured context request. It may ask for scope, "
                        "but it cannot claim approval or pre-authorization."
                    ),
                )
            payload = request_context(
                str(args["task_id"]),
                request_json=_json.dumps(request, ensure_ascii=False),
                request_file="",
            )
        elif tool_name == "execute_context":
            artifact_path, _request, approval_error = _mcp_context_is_executable(
                str(args["task_id"]),
                str(args["request_id"]),
                role_descriptor,
            )
            if approval_error:
                return _mcp_refusal(
                    approval_error,
                    status="human_confirmation_required",
                    artifact_path=str(artifact_path),
                    next_action=(
                        "Call approve_context to obtain the exact human CLI approval command. "
                        "MCP cannot advance this gate."
                    ),
                )
            payload = execute_context(str(args["task_id"]), str(args["request_id"]))
        elif tool_name == "replan_with_evidence":
            payload = replan_with_evidence(str(args["run_id"]))
        elif tool_name == "propose_patch":
            run_id = str(args["run_id"])
            stale = _jspace.context_hash_mismatches(run_id)
            if stale:
                changed = [str(item.get("path") or "") for item in stale]
                return _mcp_refusal(
                    "Patch proposal blocked because recorded planning context is stale: "
                    + ", ".join(changed[:8]),
                    status="blocked_stale_context",
                    next_action="Restore the recorded files or create a new plan against current hashes.",
                    stale_paths=changed,
                )
            payload = propose_patch(
                run_id,
                selected_files=None,
                allow_package_json=False,
                allow_delete=False,
                override_failed_check=False,
            )
        elif tool_name == "review_patch":
            payload = review_patch(str(args["patch_id"]))
        elif tool_name == "run_autonomous":
            payload = run_autonomous(str(args["task_id"]))
        elif tool_name == "audit_run":
            artifact = _mcp_resolve_artifact(
                str(args["target"]), role_descriptor, allow_id=True
            )
            _mcp_read_safe_artifact(artifact)
            payload = audit_run(str(artifact), use_model=False)
        elif tool_name == "read_artifact":
            artifact = _mcp_resolve_artifact(str(args["path"]), role_descriptor)
            payload = {
                "path": artifact.relative_to(_config.ROOT.resolve()).as_posix(),
                "content": _mcp_read_safe_artifact(artifact),
            }
        elif tool_name == "list_artifacts":
            payload = _mcp_list_role_artifacts(role_descriptor)
        elif tool_name == "prompt_audit":
            payload = _prompts.audit(role_descriptor.prompts_dir)
        else:  # The tuple/map coupling is asserted by MCP acceptance tests.
            return _mcp_refusal(f"MCP dispatch is not implemented for {tool_name}.")
        if not isinstance(payload, dict):
            return _mcp_refusal("Engineer API returned a non-object result.")
        return _mcp_wrap_tool_result(tool_name, payload)
    except Exception as exc:  # noqa: BLE001
        return _mcp_refusal(
            _mcp_safe_error(exc),
            next_action=_mcp_cli_help(tool_name),
        )
