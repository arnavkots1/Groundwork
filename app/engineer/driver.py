"""One-command driver for the Engineer loop.

Chains plan -> check -> patch and stops. Apply and Verify mutate the working tree or
execute commands, so they stay separate, explicitly human-confirmed calls and are
deliberately unreachable from here.
"""

from __future__ import annotations

from datetime import datetime

from . import config, models, patch as patch_mod
from .loop import engineer_check_run, engineer_plan
from .util import portable_repo_path


def engineer_task(
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
    """Drive the Engineer loop from one command, up to but never through the Apply gate.

    Chains plan -> check -> patch. Apply and Verify are deliberately excluded: they
    mutate the working tree, so they remain separate, explicitly human-confirmed
    commands. This driver only ever reads and proposes.
    """
    if stop_after not in config.ENGINEER_TASK_STAGES:
        raise RuntimeError(f"--stop-after must be one of: {', '.join(config.ENGINEER_TASK_STAGES)}")
    stop_index = config.ENGINEER_TASK_STAGES.index(stop_after)

    steps: list[dict] = []
    started_at = datetime.now()
    cost_mark = models.ledger_mark()

    def _step(name: str, status: str, detail: str, began: datetime, **extra: object) -> dict:
        entry: dict = {
            "step": name,
            "status": status,
            "detail": detail,
            "seconds": round((datetime.now() - began).total_seconds(), 2),
        }
        entry.update(extra)
        steps.append(entry)
        return entry

    def _finish(halted: str, run_id: str = "", patch_id: str = "") -> dict:
        elapsed = round((datetime.now() - started_at).total_seconds(), 2)
        cost = models.ledger_since(cost_mark)
        completed = [s["step"] for s in steps if s["status"] == "ok"]
        next_commands: list[str] = []
        if patch_id:
            next_commands = [
                f"python scripts/company_brain_action.py engineer-patch-review --patch-id {patch_id}",
                f"python scripts/company_brain_action.py engineer-apply-patch --patch-id {patch_id} --yes",
                "python scripts/company_brain_action.py engineer-verify --applied-patch-id <applied_patch_id> --run --save",
            ]
        elif run_id:
            next_commands = [
                f"python scripts/company_brain_action.py engineer-check-run --run-id {run_id} --write",
                f"python scripts/company_brain_action.py engineer-replan --run-id {run_id}",
            ]
        lines = [
            "# Engineer Task Run",
            "",
            f"- Task: {task.strip()[:300]}",
            f"- Stages completed: {', '.join(completed) if completed else 'none'}",
            f"- Halted: {halted}",
            f"- Run id: {run_id or 'n/a'}",
            f"- Patch id: {patch_id or 'n/a'}",
            f"- Elapsed: {elapsed}s",
            f"- Model calls: {cost['model_calls']} "
            f"(lead {cost['lead_calls']}, worker {cost['worker_calls']}); "
            f"~{cost['prompt_tokens_est'] + cost['completion_tokens_est']} tokens, "
            f"~${cost['cost_usd_est']} estimated",
            "",
            "## Steps",
            "",
        ]
        for entry in steps:
            lines.append(f"- `{entry['step']}` **{entry['status']}** ({entry['seconds']}s): {entry['detail']}")
        if next_commands:
            lines.extend(["", "## Next (human-confirmed, not run by this command)", ""])
            lines.extend(f"    {cmd}" for cmd in next_commands)
        lines.extend(
            [
                "",
                "Apply and Verify are never executed by engineer-task. Review the patch before applying.",
            ]
        )
        return {
            "ok": halted == "completed",
            "run_id": run_id,
            "patch_id": patch_id,
            "halted": halted,
            "steps": steps,
            "seconds": elapsed,
            "cost_accounting": cost,
            "next_commands": next_commands,
            "apply_performed": False,
            "verification_performed": False,
            "output": "\n".join(lines),
        }

    # Stage 1: plan
    began = datetime.now()
    try:
        plan_payload = engineer_plan(
            task,
            selected_files=selected_files,
            project=project,
            use_approved_lessons=use_approved_lessons,
            allow_web=allow_web,
            retrieval_roots=retrieval_roots,
            retrieval_intents=retrieval_intents,
            max_chars_per_call=max_chars_per_call,
        )
    except Exception as exc:  # noqa: BLE001
        _step("plan", "error", str(exc)[:900], began)
        return _finish("plan_failed")

    run_id = str(plan_payload.get("run_id") or "")
    plan_checker = str(plan_payload.get("checker_status") or "unknown")
    route_status = str(plan_payload.get("route_status") or "")
    _step(
        "plan",
        "ok",
        f"checker={plan_checker}, route={plan_payload.get('final_route') or route_status or 'unknown'}",
        began,
        run_id=run_id,
        artifact=portable_repo_path(str(plan_payload.get("path") or "")),
        checker_status=plan_checker,
    )
    if stop_index == 0:
        return _finish("stopped_after_plan", run_id=run_id)

    # Stage 2: independent re-check of the saved artifact
    began = datetime.now()
    try:
        check_payload = engineer_check_run(run_id)
    except Exception as exc:  # noqa: BLE001
        _step("check", "error", str(exc)[:900], began)
        return _finish("check_failed", run_id=run_id)

    checker_status = str(check_payload.get("checkerStatus") or "unknown")
    failed_rules = check_payload.get("failedRules") or []
    _step(
        "check",
        "ok" if checker_status != "fail" else "fail",
        f"checker={checker_status}" + (f", failed_rules={len(failed_rules)}" if failed_rules else ""),
        began,
        checker_status=checker_status,
        failed_rules=failed_rules,
    )
    if checker_status == "fail" and not override_failed_check:
        return _finish("blocked_by_checker", run_id=run_id)
    if stop_index == 1:
        return _finish("stopped_after_check", run_id=run_id)

    # Stage 3: patch proposal (never applied)
    began = datetime.now()
    try:
        patch_payload = patch_mod.engineer_patch(
            run_id,
            selected_files=selected_files,
            allow_package_json=allow_package_json,
            allow_delete=allow_delete,
            override_failed_check=override_failed_check,
        )
    except Exception as exc:  # noqa: BLE001
        _step("patch", "error", str(exc)[:900], began)
        return _finish("patch_failed", run_id=run_id)

    patch_id = str(patch_payload.get("patch_id") or "")
    patch_status = str(patch_payload.get("patch_status") or "blocked")
    _step(
        "patch",
        "ok" if patch_status == "proposed" else "blocked",
        f"status={patch_status}, risk={patch_payload.get('risk_level') or 'unknown'}, "
        f"targets={len(patch_payload.get('target_files') or [])}",
        began,
        patch_id=patch_id,
        artifact=portable_repo_path(str(patch_payload.get("path") or "")),
        patch_status=patch_status,
    )
    if patch_status != "proposed":
        return _finish("patch_blocked", run_id=run_id, patch_id=patch_id)
    return _finish("completed", run_id=run_id, patch_id=patch_id)
