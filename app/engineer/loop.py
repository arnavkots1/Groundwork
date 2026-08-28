"""The Engineer loop: plan and deterministic re-check.

Every stage writes an artifact before it returns, including refusals - a refusal
that leaves no trace is indistinguishable from a step that never ran.

Grounded replan lives in `replan`; the one-command driver lives in `driver`.
"""

from __future__ import annotations

import json
from pathlib import Path

from repository_retrieval import (
    SUPPORTED_INTENTS as RETRIEVAL_INTENTS,
    normalize_authorized_intents,
    normalize_authorized_roots,
)

from . import artifacts, checker, config, context, jspace, models, patch as patch_mod, prompts, retrieval, review
from .fallbacks import blocked_engineer_output, fallback_engineer_output
from .util import (
    dedupe,
    extract_json_object,
    now_iso,
    portable_repo_path,
    read_jsonl,
    read_text,
    repo_path,
    safe_workspace_id,
)


def run_checker(parsed: dict, selected_records: list[dict], raw_output: str = "", task: str = "") -> dict:
    return checker.run_checker(parsed, selected_records, raw_output, task)


def emit_generated_context_request(run_id: str, parsed: dict, *, label: str) -> dict | None:
    """Turn a model's context_requests into a real, classified request artifact.

    The model can ask for scope; it cannot grant it. The request is classified and
    anything beyond pre-authorized roots waits for a human approval step.

    `label` must be unique per call on a given run_id: the underlying request store
    (`write_context_request(..., create=True)`) raises if the request_id already
    exists, and the initial plan stage and every grounded-replan revision (including
    the bounded follow-up round) all call this on the same run_id. Reusing "ctx_{run_id}"
    unconditionally used to make every replan-stage call collide with the plan-stage
    request, so the follow-up round's own context request silently failed and was
    swallowed into `status: "invalid"` - which made it permanently ineligible without
    ever showing up as an error.
    """
    items = parsed.get("context_requests") or []
    if isinstance(items, dict):
        items = items.get("requested_items") or []
    if not isinstance(items, list) or not items:
        return None
    context_request_id = f"ctx_{run_id}_{label}"
    try:
        result = retrieval.engineer_context_request(
            run_id,
            json.dumps(
                {
                    "request_id": context_request_id,
                    "task_id": run_id,
                    "requested_items": items,
                    "external_retrieval": False,
                }
            ),
        )
        return {
            "request_id": context_request_id,
            "status": result.get("status"),
            "path": portable_repo_path(result.get("path", "")),
            "json_path": portable_repo_path(result.get("jsonPath", "")),
        }
    except Exception as exc:  # noqa: BLE001
        return {"request_id": context_request_id, "status": "invalid", "error": str(exc)[:900]}


# Cheap/fast per-provider model used only for the plan stage's own model call.
# Deliberately not replan or patch: replan re-derives the whole plan against
# retrieved evidence and is where the real grounding checks (claim_without_evidence,
# context_incomplete, etc.) live, and patch has to emit an exact unified diff - both
# are where a weaker model's mistakes actually cost something. Plan's output is
# fully re-verified by replan against real evidence regardless of how it was
# produced, so a weaker first guess here is genuinely recoverable downstream, not
# just assumed to be. Keyed by provider so whichever route call_tier_with_fallback
# actually picks gets the right override; PLAN_STAGE_CHEAP_MODEL_ENV_VAR mirrors
# the same per-provider model env vars agent_cli_argv() already reads.
PLAN_STAGE_CHEAP_MODEL_ENV_VAR: dict[str, str] = {
    "Cursor Agent CLI": "CURSOR_CLI_MODEL",
    "Codex CLI": "CODEX_CLI_MODEL",
    "Claude Code CLI": "CLAUDE_CLI_MODEL",
}
# Hardcoded fallback only - PLAN_STAGE_MODEL_OVERRIDE_ENV_VAR below lets a caller
# pick a different cheap model per provider (or disable routing entirely) without
# editing source, same as every other model choice in this codebase is tunable.
PLAN_STAGE_CHEAP_MODEL: dict[str, str] = {
    "Cursor Agent CLI": "composer-2.5-fast",
    "Codex CLI": "gpt-5.6-luna",
    "Claude Code CLI": "claude-haiku-4-5-20251001",
}
PLAN_STAGE_MODEL_OVERRIDE_ENV_VAR: dict[str, str] = {
    "Cursor Agent CLI": "PLAN_STAGE_MODEL_CURSOR",
    "Codex CLI": "PLAN_STAGE_MODEL_CODEX",
    "Claude Code CLI": "PLAN_STAGE_MODEL_CLAUDE",
}
# Set to disable plan-stage model routing entirely (plan then uses whatever model
# replan/patch use) - e.g. for a controlled A/B comparison of the routing itself.
DISABLE_PLAN_STAGE_CHEAP_MODEL_ENV_VAR = "DISABLE_PLAN_STAGE_CHEAP_MODEL"


class _plan_stage_cheap_model:
    """Temporarily override every known CLI provider's model env var for the
    plan-stage call, then restore whatever was there before - same set/restore
    pattern already used for COMPANYBRAIN_PIN_PROVIDER elsewhere in this codebase.
    Set for every provider's env var (not just the pinned one) because the caller
    doesn't know in advance which route call_tier_with_fallback will pick; each
    provider's var is independent, so this is a normal, harmless no-op for
    providers not actually used this call.

    DISABLE_PLAN_STAGE_CHEAP_MODEL=1 makes __enter__/__exit__ a true no-op (no
    env var touched, not even a no-op set/restore), for a clean controlled
    comparison against routing turned on.
    """

    def __init__(self) -> None:
        import os

        self._disabled = bool(os.environ.get(DISABLE_PLAN_STAGE_CHEAP_MODEL_ENV_VAR, "").strip())
        self._previous: dict[str, str | None] = {}

    def __enter__(self) -> "_plan_stage_cheap_model":
        import os

        if self._disabled:
            return self
        for provider, env_var in PLAN_STAGE_CHEAP_MODEL_ENV_VAR.items():
            self._previous[env_var] = os.environ.get(env_var)
            override_var = PLAN_STAGE_MODEL_OVERRIDE_ENV_VAR[provider]
            os.environ[env_var] = os.environ.get(override_var, "").strip() or PLAN_STAGE_CHEAP_MODEL[provider]
        return self

    def __exit__(self, *exc_info: object) -> None:
        import os

        if self._disabled:
            return

        for env_var, value in self._previous.items():
            if value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = value


def engineer_plan(
    task: str,
    selected_files: list[str] | None = None,
    project: str = "",
    use_approved_lessons: bool = False,
    allow_web: bool = False,
    compare_baseline: bool = False,
    retrieval_roots: list[str] | None = None,
    retrieval_intents: list[str] | None = None,
    task_id: str = "",
    task_envelope: dict | None = None,
    max_chars_per_call: int | None = None,
) -> dict:
    """Produce and save a checked plan for a task plus its human-selected files."""
    config.ensure_harness()
    cost_mark = models.ledger_mark()
    task = task.strip()
    if not task:
        raise RuntimeError("engineer-plan requires --task.")
    excerpt_chars = context.resolve_excerpt_char_limit(max_chars_per_call)
    selected_records, missing = context.selected_context(
        selected_files or [], task, max_chars_per_call=excerpt_chars
    )
    if missing:
        raise RuntimeError(f"Hard stop: selected files are missing: {', '.join(missing)}")
    if bool(task_id) != bool(task_envelope):
        raise RuntimeError("A predeclared task_id and validated task_envelope must be supplied together.")
    if task_envelope:
        envelope_task_id = str(task_envelope.get("task_id") or "")
        envelope_task = str((task_envelope.get("task") or {}).get("request") or "")
        if task_id != envelope_task_id or task != envelope_task:
            raise RuntimeError("Plan task identity/text does not match the immutable task envelope.")
        envelope_targets = [
            str(item)
            for item in ((task_envelope.get("scope") or {}).get("write_targets") or [])
        ]
        selected_inputs = [
            str(item.get("input") or item.get("path") or "")
            for item in selected_records
            if item.get("input") or item.get("path")
        ]
        normalized_selected = {
            repo_path(item).relative_to(config.ROOT.resolve()).as_posix()
            for item in selected_inputs
        }
        if normalized_selected != set(envelope_targets):
            raise RuntimeError("Plan selected files must exactly match envelope write targets.")
        retrieval_roots = list((task_envelope.get("scope") or {}).get("allowed_roots") or [])
        retrieval_intents = list((task_envelope.get("scope") or {}).get("allowed_intents") or [])

    root_values = list(retrieval_roots or [])
    if not root_values and selected_records:
        # Intake scope: parent directories of human-selected files are pre-authorized
        # for retrieval. MCP cannot pass retrieval_roots, so this is how in-scope
        # reads become pre_authorized without a human CLI round trip.
        inferred: list[str] = []
        for item in selected_records:
            relative = str(item.get("input") or "").replace("\\", "/").strip()
            if not relative:
                try:
                    relative = portable_repo_path(str(item.get("path") or ""))
                except Exception:  # noqa: BLE001
                    continue
            parent = Path(relative).parent.as_posix()
            if parent and parent not in {".", ""}:
                inferred.append(parent)
        root_values = dedupe(inferred)
    authorized_retrieval_roots = normalize_authorized_roots(root_values, config.ROOT)
    authorized_retrieval_intents = normalize_authorized_intents(
        retrieval_intents or (sorted(RETRIEVAL_INTENTS) if authorized_retrieval_roots else [])
    )

    approved_lessons = read_jsonl(config.ENGINEER_FILES["approved_lessons"]) if use_approved_lessons else []
    created_at = now_iso()
    run_id = safe_workspace_id(task_id) if task_envelope else jspace.new_plan_id()
    j_space_fields = jspace.create_manifest(
        run_id,
        task,
        project,
        selected_records,
        use_approved_lessons,
        allow_web,
        authorized_retrieval_roots,
        authorized_retrieval_intents,
        task_envelope=task_envelope,
        selected_file_excerpt_chars_per_file=excerpt_chars,
    )
    selected_excerpt_snapshot = jspace.snapshot_selected_plan_excerpts(run_id, selected_records)
    j_space = jspace.pointer(run_id)
    context_manifest = {
        "task": task,
        "project": project,
        "created_at": created_at,
        "j_space_workspace_id": run_id,
        "j_space_manifest": j_space["manifest_path"],
        "selected_file_excerpt_chars_per_file": excerpt_chars,
        "selected_files": [
            {key: value for key, value in item.items() if key != "content"} for item in selected_records
        ],
        "selected_file_inspections": context.selected_file_inspections(selected_records),
        "selected_plan_excerpt_snapshot": selected_excerpt_snapshot,
        "approved_lessons_loaded": use_approved_lessons,
        "approved_lesson_count": len(approved_lessons),
        "candidate_lessons_loaded": False,
        "legacy_brain_auto_loaded": False,
        "old_idea_runs_auto_loaded": False,
        "web_enabled": allow_web,
        "repository_retrieval": {
            "mode": "pre_authorized" if authorized_retrieval_roots else "disabled",
            "allowed_roots": authorized_retrieval_roots,
            "allowed_intents": authorized_retrieval_intents,
            "external_retrieval": False,
        },
        "task_envelope": (
            {
                "envelope_id": task_envelope.get("envelope_id"),
                "task_id": task_envelope.get("task_id"),
                "canonical_payload_sha256": (task_envelope.get("integrity") or {}).get(
                    "canonical_payload_sha256"
                ),
                "expires_at": task_envelope.get("expires_at"),
                "immutable": True,
            }
            if task_envelope
            else {}
        ),
        "excluded_context": [
            "legacy idea runs",
            "historical agent examples",
            "candidate lessons",
            "old product winners and losers",
            "automatic web browsing" if not allow_web else "random web browsing",
        ],
        "checkpoints": [
            {"stage": "task_intake", "status": "complete"},
            {"stage": "context_selection", "status": "complete"},
            {"stage": "plan", "status": "pending"},
            {"stage": "checker", "status": "pending"},
            {"stage": "codex_prompt", "status": "pending"},
            {"stage": "execution_review", "status": "pending"},
        ],
    }
    if checker.is_code_modification_task(task) and not selected_records:
        return _save_blocked_plan(
            run_id, task, j_space, context_manifest, selected_records, selected_excerpt_snapshot
        )

    bundle = {
        key: read_text(path)
        for key, path in config.ENGINEER_FILES.items()
        if key in config.ENGINEER_PROMPT_FILE_KEYS
    }
    if approved_lessons:
        bundle["approved_lessons"] = "\n".join(json.dumps(item, ensure_ascii=False) for item in approved_lessons)
    if selected_records:
        bundle["selected_files"] = "\n\n".join(
            f"## {item['path']}\nBEGIN UNTRUSTED SELECTED FILE EXCERPT\n{item.get('content', '')}\nEND UNTRUSTED SELECTED FILE EXCERPT"
            for item in selected_records
        )

    prompt = prompts.render(
        "plan",
        {
            "task": task,
            "context_manifest": json.dumps(context_manifest, ensure_ascii=False, indent=2),
            "harness_bundle": "\n\n".join(f"## {name}\n{content}" for name, content in bundle.items()),
        },
    )
    route_attempts = []
    raw_content = ""
    final_route = ""
    route_status = "not_attempted"
    structured_parse_status = "not_attempted"
    literal_candidate = patch_mod.deterministic_literal_patch(
        task,
        [str(item.get("input") or item.get("path") or "") for item in selected_records],
    )
    if literal_candidate:
        final_route = "deterministic_literal_replacement"
        route_status = "deterministic"
        structured_parse_status = "deterministic"
        parsed = fallback_engineer_output(
            task,
            "",
            "Exact single-file literal replacement validated; deterministic planning path used.",
            selected_records,
        )
    else:
        try:
            with _plan_stage_cheap_model():
                routed = models.call_tier_with_fallback(
                    "Lead Engineer",
                    prompt,
                    tier=models._engineer_primary_tier(),
                    max_cooldown_wait_seconds=0,
                    cloud_request_timeout_seconds=45,
                    cloud_max_retries=0,
                    max_tokens=3000,
                )
            raw_content = routed.content
            route_attempts = models.attempts_of(routed)
            final_route = models.route_of(routed)
            route_status = "succeeded"
            parsed = extract_json_object(routed.content)
            if not parsed:
                structured_parse_status = "failed"
                parsed = fallback_engineer_output(
                    task,
                    routed.content,
                    "Model route succeeded, but structured JSON parsing failed; deterministic fallback used.",
                    selected_records,
                )
            else:
                structured_parse_status = "parsed"
        except Exception as exc:  # noqa: BLE001
            raw_content = str(exc)
            final_route = "Best Available failed; deterministic fallback used"
            route_status = "failed"
            structured_parse_status = "not_attempted"
            route_attempts = [{"provider": "Best Available", "model": "", "status": "failed", "detail": str(exc)[:1200]}]
            parsed = fallback_engineer_output(
                task,
                raw_content,
                "No model route succeeded; deterministic fallback used.",
                selected_records,
            )

    original_plan = dict(parsed)
    parsed.setdefault(
        "verified_facts",
        [f"Selected file exists: {item.get('input')}" for item in selected_records if item.get("exists")],
    )
    parsed.setdefault("unverified_assumptions", parsed.get("assumptions", []))
    parsed.setdefault("repo_grounding_score", 3 if selected_records else 1)
    parsed.setdefault("revised_codex_prompt", parsed.get("codex_prompt", ""))

    checker_report = checker.run_checker(parsed, selected_records, raw_content, task)
    checker_warnings = checker_report["warnings"]
    checker_status = checker_report["status"]
    repair_result = None
    checker_adjusted_plan = {}
    if checker_status in {"warn", "fail"} and route_status == "succeeded":
        try:
            repair_result = _repair_plan(task, original_plan, checker_report, context_manifest, selected_records)
            if repair_result.get("parsed"):
                checker_adjusted_plan = repair_result["parsed"]
                parsed.update(checker_adjusted_plan)
                parsed.setdefault("revised_codex_prompt", parsed.get("codex_prompt", ""))
                checker_report = checker.run_checker(parsed, selected_records, repair_result.get("raw_output", ""), task)
                checker_warnings = checker_report["warnings"]
                checker_status = checker_report["status"]
        except Exception as exc:  # noqa: BLE001
            repair_result = {
                "route": "",
                "attempts": [],
                "raw_output": str(exc),
                "parsed": {},
                "parse_status": "failed",
                "error": str(exc),
            }

    baseline_output = ""
    baseline_attempts = []
    if compare_baseline:
        try:
            baseline = models.call_best_available_with_fallback(
                "Lead Engineer",
                prompts.render(
                    "baseline_plan",
                    {
                        "task": task,
                        "selected_files": json.dumps(selected_records, ensure_ascii=False, indent=2),
                    },
                ),
                max_cooldown_wait_seconds=0,
            )
            baseline_output = baseline.content
            baseline_attempts = models.attempts_of(baseline)
        except Exception as exc:  # noqa: BLE001
            baseline_output = f"Baseline route failed: {exc}"
            baseline_attempts = [{"provider": "Best Available", "model": "", "status": "failed", "detail": str(exc)[:1200]}]

    payload = {
        "run_id": run_id,
        "task": task,
        "j_space": j_space,
        "context_manifest": context_manifest,
        "selected_file_excerpt_chars_per_file": excerpt_chars,
        "selected_file_inspections": context.selected_file_inspections(selected_records),
        "selectedContextPath": str(repo_path(selected_excerpt_snapshot["path"])),
        "selectedContextJsonPath": str(repo_path(selected_excerpt_snapshot["json_path"])),
        "model_routes_attempted": route_attempts,
        "final_route": final_route,
        "route_status": route_status,
        "structured_parse_status": structured_parse_status,
        "raw_output_parse_failed": structured_parse_status == "failed",
        "checker_status": checker_status,
        "checker_report": checker_report,
        "checker_warnings": checker_warnings,
        "original_plan": original_plan,
        "checker_adjusted_plan": checker_adjusted_plan,
        "repair_result": repair_result,
        "raw_output": raw_content,
        "baseline_output": baseline_output,
        "baseline_model_routes_attempted": baseline_attempts,
        "candidate_lessons_from_checker": [],
        **parsed,
    }
    payload["candidate_lessons_from_checker"] = review.candidate_lessons_from_checker(run_id, checker_report)
    context_manifest["checkpoints"] = [
        {"stage": "task_intake", "status": "complete"},
        {"stage": "context_selection", "status": "complete"},
        {"stage": "plan", "status": "complete"},
        {"stage": "checker", "status": "complete"},
        {"stage": "codex_prompt", "status": "complete"},
        {"stage": "execution_review", "status": "template_saved"},
    ]

    md_path = config.ENGINEER_RUNS_DIR / f"{run_id}.md"
    json_path = config.ENGINEER_RUNS_DIR / f"{run_id}.json"
    # Plan artifacts are always saved so downstream steps (check-run, patch, review)
    # can locate them by run_id and the loop stays auditable.
    md_path.write_text(artifacts.plan_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_ref = jspace.artifact_ref("engineer_plan", run_id, md_path, json_path)
    jspace.update_manifest(
        run_id,
        "plan",
        "complete",
        "planned",
        "Plan artifact saved.",
        artifact_ref,
        {
            "final_route": final_route,
            "route_status": route_status,
            "structured_parse_status": structured_parse_status,
            "selected_file_count": len(selected_records),
        },
    )
    j_space_fields = jspace.update_manifest(
        run_id,
        "checker",
        checker_status,
        "blocked" if checker_status == "fail" else "planned",
        "; ".join(checker_warnings),
        artifact_ref,
    )
    generated_context_request = emit_generated_context_request(run_id, parsed, label="plan")
    if generated_context_request is not None:
        payload["generated_context_request"] = generated_context_request
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(artifacts.plan_markdown(payload), encoding="utf-8")
    payload["cost_accounting"] = models.ledger_since(cost_mark)
    payload.update(
        {
            "ok": True,
            "path": str(md_path),
            "jsonPath": str(json_path),
            "output": artifacts.plan_markdown(payload),
            **j_space_fields,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _save_blocked_plan(
    run_id: str,
    task: str,
    j_space: dict,
    context_manifest: dict,
    selected_records: list[dict],
    selected_excerpt_snapshot: dict,
) -> dict:
    """A code task with no selected files is refused, but the refusal is still saved."""
    parsed = blocked_engineer_output(task)
    checker_report = checker.run_checker(parsed, selected_records, "", task)
    checker_report["status"] = "fail"
    if "code_task_missing_selected_files" not in checker_report["failed_rules"]:
        checker_report["failed_rules"].append("code_task_missing_selected_files")
    checker_report["warnings"] = dedupe(
        [
            *checker_report["warnings"],
            "Code modification task has no selected files; blocked planning artifact saved.",
        ]
    )
    context_manifest["checkpoints"] = [
        {"stage": "task_intake", "status": "complete"},
        {"stage": "context_selection", "status": "blocked_missing_selected_files"},
        {"stage": "plan", "status": "blocked"},
        {"stage": "checker", "status": "complete"},
        {"stage": "codex_prompt", "status": "blocked"},
        {"stage": "execution_review", "status": "template_saved"},
    ]
    payload = {
        "run_id": run_id,
        "task": task,
        "j_space": j_space,
        "context_manifest": context_manifest,
        "selected_file_excerpt_chars_per_file": context_manifest.get(
            "selected_file_excerpt_chars_per_file"
        ),
        "selected_file_inspections": [],
        "selectedContextPath": str(repo_path(selected_excerpt_snapshot["path"])),
        "selectedContextJsonPath": str(repo_path(selected_excerpt_snapshot["json_path"])),
        "model_routes_attempted": [],
        "final_route": "not attempted; blocked before model routing",
        "route_status": "blocked",
        "structured_parse_status": "blocked",
        "raw_output_parse_failed": False,
        "checker_status": "fail",
        "checker_report": checker_report,
        "checker_warnings": checker_report["warnings"],
        "original_plan": parsed,
        "checker_adjusted_plan": {},
        "repair_result": None,
        "raw_output": "",
        "baseline_output": "",
        "baseline_model_routes_attempted": [],
        "candidate_lessons_from_checker": review.candidate_lessons_from_checker(run_id, checker_report),
        **parsed,
    }
    md_path = config.ENGINEER_RUNS_DIR / f"{run_id}.md"
    json_path = config.ENGINEER_RUNS_DIR / f"{run_id}.json"
    md_path.write_text(artifacts.plan_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_ref = jspace.artifact_ref("engineer_plan", run_id, md_path, json_path)
    jspace.update_manifest(
        run_id,
        "plan",
        "blocked",
        "blocked",
        "Missing selected files.",
        artifact_ref,
        {"route_status": "blocked", "final_route": "not attempted"},
    )
    j_space_fields = jspace.update_manifest(
        run_id,
        "checker",
        "fail",
        "blocked",
        "; ".join(checker_report.get("warnings") or []),
        artifact_ref,
    )
    payload.update(
        {
            "ok": True,
            "path": str(md_path),
            "jsonPath": str(json_path),
            "output": artifacts.plan_markdown(payload),
            **j_space_fields,
        }
    )
    return payload


def _repair_plan(
    task: str, original_plan: dict, checker_report: dict, context_manifest: dict, selected_records: list[dict]
) -> dict:
    prompt = prompts.render(
        "repair_plan",
        {
            "task": task,
            "context_manifest": json.dumps(context_manifest, ensure_ascii=False, indent=2),
            "selected_file_inspections": json.dumps(
                context.selected_file_inspections(selected_records), ensure_ascii=False, indent=2
            ),
            "checker_report": json.dumps(checker_report, ensure_ascii=False, indent=2),
            "original_plan": json.dumps(original_plan, ensure_ascii=False, indent=2),
        },
    )
    # Deliberately NOT wrapped in _plan_stage_cheap_model: fixing a plan that
    # already failed the checker needs real reasoning about why it failed, not a
    # cheap first guess - unlike the initial plan, there's no later grounded-replan
    # stage that re-derives this from scratch against evidence to catch a weak
    # repair attempt.
    routed = models.call_tier_with_fallback(
        "Lead Engineer",
        prompt,
        tier=models._engineer_primary_tier(),
        max_cooldown_wait_seconds=0,
        cloud_request_timeout_seconds=45,
        cloud_max_retries=0,
        max_tokens=3000,
    )
    parsed = extract_json_object(routed.content)
    return {
        "route": models.route_of(routed),
        "attempts": models.attempts_of(routed),
        "raw_output": routed.content,
        "parsed": parsed,
        "parse_status": "parsed" if parsed else "failed",
    }


def engineer_check_run(run_id: str, write: bool = False) -> dict:
    """Re-run the checker against a saved plan and current disk state."""
    config.ensure_harness()
    json_path, payload = artifacts.load_run_json(run_id)
    selected_records = context.records_from_saved_payload(payload)
    checker_meta = {
        "checker_status",
        "checker_report",
        "checker_warnings",
        "candidate_lessons_from_checker",
        "checker_rerun_at",
        "output",
    }
    plan = {key: value for key, value in payload.items() if key not in checker_meta}
    if payload.get("checker_adjusted_plan"):
        plan.update(payload.get("checker_adjusted_plan") or {})
    checker_report = checker.run_checker(plan, selected_records, payload.get("raw_output", ""), payload.get("task", ""))
    output_lines = [
        f"checker_status: {checker_report['status']}",
        "warnings:",
        *[f"- {warning}" for warning in checker_report.get("warnings", [])],
    ]
    candidate_lessons = []
    source_run_id = payload.get("run_id", json_path.stem)
    j_space_fields = jspace.response_fields(source_run_id)
    if write:
        payload["selected_file_inspections"] = context.selected_file_inspections(selected_records)
        if payload.get("context_manifest"):
            payload["context_manifest"]["selected_file_inspections"] = payload["selected_file_inspections"]
        payload["checker_status"] = checker_report["status"]
        payload["checker_report"] = checker_report
        payload["checker_warnings"] = checker_report["warnings"]
        payload["checker_rerun_at"] = now_iso()
        candidate_lessons = review.candidate_lessons_from_checker(payload.get("run_id", json_path.stem), checker_report)
        payload["candidate_lessons_from_checker"] = [
            *(payload.get("candidate_lessons_from_checker") or []),
            *candidate_lessons,
        ]
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = json_path.with_suffix(".md")
        md_path.write_text(artifacts.plan_markdown(payload), encoding="utf-8")
        j_space_fields = jspace.update_manifest(
            source_run_id,
            "checker",
            checker_report["status"],
            "blocked" if checker_report["status"] == "fail" else "planned",
            "; ".join(checker_report.get("warnings") or []),
            jspace.artifact_ref("engineer_plan", source_run_id, md_path, json_path),
            {
                "failed_rules": checker_report.get("failed_rules") or [],
                "warning_count": len(checker_report.get("warnings") or []),
                "persisted": True,
            },
        )
    return {
        "ok": True,
        "run_id": source_run_id,
        "checkerStatus": checker_report["status"],
        "warnings": checker_report["warnings"],
        "failedRules": checker_report["failed_rules"],
        "suggestedCorrections": checker_report["suggested_corrections"],
        "candidateLessons": candidate_lessons,
        "path": str(json_path),
        "output": "\n".join(output_lines),
        **j_space_fields,
    }
