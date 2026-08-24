"""Post-execution review, independent patch review, feedback, and the lesson gate.

Lessons are always created as candidates. Approval is a separate human action and
canonical behavior is never edited automatically - that separation is what keeps a
bad run from teaching the harness a bad habit.
"""

from __future__ import annotations

import json
import uuid

from . import artifacts, config, jspace, models, prompts
from .patch import check_patch_payload
from .util import append_jsonl, dedupe, extract_json_object, normalize_repo_file, now_iso, now_tz_iso, read_jsonl, unique_artifact_id


def planned_engineer_files(plan: dict) -> list[str]:
    planned: list[str] = []
    for item in plan.get("files_likely_involved") or []:
        if isinstance(item, str):
            planned.append(item)
    selected = ((plan.get("context_manifest") or {}).get("selected_files") or [])
    for item in selected:
        candidate = item.get("input") or item.get("path") if isinstance(item, dict) else ""
        if candidate:
            planned.append(candidate)
    return dedupe([item for item in planned if item])


def files_match(planned_files: list[str], changed_files: list[str]) -> str | bool:
    if not changed_files:
        return False
    planned = {normalize_repo_file(item) for item in planned_files if item}
    changed = {normalize_repo_file(item) for item in changed_files if item}
    if not planned:
        return False
    overlap = planned & changed
    if changed and changed <= planned:
        return True
    if overlap:
        return "partial"
    return False


def forbidden_changes_touched(forbidden_changes: list, changed_files: list[str]) -> list[str]:
    forbidden_text = " ".join(str(item).lower() for item in forbidden_changes)
    changed_text = " ".join(normalize_repo_file(item) for item in changed_files)
    hits = []
    checks = {
        "legacy brain": ["brain/", "brain\\", "legacy"],
        "candidate lessons": ["candidate_lessons"],
        "canonical behavior": ["canonical_behavior.md"],
        "backend/api": ["server/", "api", "company-brain-api"],
        "dependencies": ["package.json", "package-lock.json"],
    }
    for label, markers in checks.items():
        if label in forbidden_text and any(marker in changed_text for marker in markers):
            hits.append(label)
    return hits


def _new_candidate_lessons(statements: list[str], build: object) -> list[dict]:
    """Append candidate lessons, skipping statements already pending review."""
    lessons = []
    existing = {
        item.get("statement")
        for item in read_jsonl(config.ENGINEER_FILES["candidate_lessons"])
        if item.get("status") == "candidate"
    }
    for statement in dedupe(statements):
        if statement in existing:
            continue
        lesson = build(statement)  # type: ignore[operator]
        append_jsonl(config.ENGINEER_FILES["candidate_lessons"], lesson)
        lessons.append(lesson)
    return lessons


def candidate_lessons_from_checker(run_id: str, checker_report: dict) -> list[dict]:
    now = now_iso()
    rules = set(checker_report.get("failed_rules") or [])
    warnings = " ".join(checker_report.get("warnings") or []).lower()
    candidates: list[str] = []
    if rules & {"proposed_dependency_without_permission", "new_dependency_conflicts_with_task"} or "dependency mentioned" in warnings:
        candidates.append("Before suggesting library-specific changes, verify the dependency exists in selected files or package metadata.")
    if rules & {"mentions_missing_file", "mentions_unverified_file"}:
        candidates.append("Before naming files in a Codex prompt, verify each file exists or was selected.")
    if "unsupported_rendering_assumption" in rules:
        candidates.append("Before relying on rendering behavior, verify it from selected UI files.")
    if "backend_change_for_frontend_task" in rules:
        candidates.append("When a task says frontend-only or no backend behavior change, keep implementation steps and prompts out of backend/API files.")
    if "missing_acceptance_tests" in rules or "verification command" in warnings:
        candidates.append("Every Engineer plan should include concrete acceptance tests and at least one executable verification command.")

    return _new_candidate_lessons(
        candidates,
        lambda statement: {
            "lesson_id": f"lesson_{uuid.uuid4().hex[:12]}",
            "source_run_id": run_id,
            "lesson_type": "checker_failure",
            "statement": statement,
            "applies_to": "Engineer Employee planning behavior",
            "confidence": "medium",
            "status": "candidate",
            "created_at": now,
        },
    )


def candidate_lessons_from_review(review: dict) -> list[dict]:
    now = now_iso()
    statements = []
    if review.get("compile_passed") is False:
        statements.append("Engineer plans should include verification commands that catch compile failures before a prompt is treated as useful.")
    if review.get("tests_passed") is False:
        statements.append("Engineer plans should identify at least one executable test or smoke check aligned with the intended change.")
    if review.get("files_touched_correct") is False or review.get("files_match") is False:
        statements.append("Engineer prompts should constrain file scope tightly enough that execution touches only selected or planned files.")
    if review.get("manual_correction") in {"medium", "high"}:
        statements.append("When manual correction is medium or high, capture the missing context or prompt ambiguity before reusing the plan pattern.")
    if review.get("checker_missed"):
        statements.append("Checker misses from execution review should become candidate lessons before they influence future prompts.")
    if review.get("forbidden_changes_touched"):
        statements.append("If execution touches forbidden areas, future prompts must state those forbidden paths as hard stop constraints.")

    return _new_candidate_lessons(
        statements,
        lambda statement: {
            "lesson_id": f"lesson_{uuid.uuid4().hex[:12]}",
            "source_run_id": review.get("source_run_id", ""),
            "source_review_id": review.get("review_id", ""),
            "lesson_type": "execution_review",
            "statement": statement,
            "applies_to": "Engineer Employee planning and execution review behavior",
            "confidence": "medium",
            "status": "candidate",
            "created_at": now,
        },
    )


def engineer_review(
    run_id: str,
    changed_files: list[str] | None = None,
    codex_summary: str = "",
    compile_passed: bool | None = None,
    tests_passed: bool | None = None,
    files_touched_correct: bool | None = None,
    manual_correction: str = "",
    notes: str = "",
) -> dict:
    """Record what actually happened when a plan was executed.

    Scores are derived from the reported outcomes, not from a model's opinion.
    """
    config.ensure_harness()
    json_path, plan = artifacts.load_run_json(run_id)
    changed_files = changed_files or []
    now = now_iso()
    review_id = unique_artifact_id("_engineer_review", config.ENGINEER_REVIEWS_DIR)
    planned_files = planned_engineer_files(plan)
    selected_files = [
        item.get("input") or item.get("path")
        for item in ((plan.get("context_manifest") or {}).get("selected_files") or [])
        if isinstance(item, dict)
    ]
    forbidden_changes = plan.get("forbidden_changes") or []
    match = files_match(planned_files, changed_files)
    forbidden_hits = forbidden_changes_touched(forbidden_changes, changed_files)
    checker_status = plan.get("checker_status", "")
    process_violation = checker_status == "fail" and bool(changed_files or codex_summary or notes)

    rules_triggered = []
    if compile_passed is False:
        rules_triggered.append("compile_failed_decision_not_useful")
    if tests_passed is False:
        rules_triggered.append("tests_failed")
    if files_touched_correct is False:
        rules_triggered.append("files_touched_incorrect_decision_not_useful")
    if manual_correction == "high":
        rules_triggered.append("high_manual_correction")
    if forbidden_hits:
        rules_triggered.append("forbidden_changes_touched")
    if process_violation:
        rules_triggered.append("checker_failed_but_execution_ran")

    plan_followed_score = 5
    if match == "partial":
        plan_followed_score -= 1
    elif match is False:
        plan_followed_score -= 2
    if files_touched_correct is False:
        plan_followed_score -= 2
    if forbidden_hits or process_violation:
        plan_followed_score = 1
    plan_followed_score = max(1, min(5, plan_followed_score))

    prompt_quality_score = 5
    if checker_status == "warn":
        prompt_quality_score -= 1
    if checker_status == "fail" or plan.get("blocked_plan"):
        prompt_quality_score -= 2
    if manual_correction == "medium":
        prompt_quality_score -= 1
    elif manual_correction == "high":
        prompt_quality_score -= 2
    if compile_passed is False or tests_passed is False:
        prompt_quality_score -= 1
    prompt_quality_score = max(1, min(5, prompt_quality_score))

    what_failed = []
    if compile_passed is False:
        what_failed.append("Compile failed.")
    if tests_passed is False:
        what_failed.append("Tests failed.")
    if match is False:
        what_failed.append("Changed files did not match planned/selected files.")
    elif match == "partial":
        what_failed.append("Changed files partially matched planned/selected files.")
    if forbidden_hits:
        what_failed.append(f"Forbidden change areas were touched: {', '.join(forbidden_hits)}.")
    if process_violation:
        what_failed.append("Checker status was fail, but execution appears to have proceeded.")
    checker_missed = []
    if files_touched_correct is False and checker_status != "fail":
        checker_missed.append("Checker did not prevent wrong-file execution.")
    if compile_passed is False and checker_status == "pass":
        checker_missed.append("Checker passed a plan that later failed compilation.")
    if forbidden_hits and checker_status != "fail":
        checker_missed.append("Checker did not block a plan that led to forbidden changes.")

    if forbidden_hits or compile_passed is False or files_touched_correct is False:
        decision = "failed"
    elif manual_correction == "high" or tests_passed is False or process_violation:
        decision = "needs_revision"
    elif manual_correction == "medium" or match == "partial" or checker_missed:
        decision = "needs_revision"
    else:
        decision = "useful"

    review = {
        "review_id": review_id,
        "source_run_id": plan.get("run_id", json_path.stem),
        "source_run_json": str(json_path),
        "j_space": jspace.pointer(str(plan.get("run_id") or json_path.stem)),
        "created_at": now,
        "execution_summary": codex_summary or "No Codex/Cursor summary provided.",
        "planned_files": planned_files,
        "selected_files": selected_files,
        "changed_files": changed_files,
        "files_match": match,
        "compile_passed": compile_passed,
        "tests_passed": tests_passed,
        "files_touched_correct": files_touched_correct,
        "manual_correction": manual_correction,
        "plan_followed_score": plan_followed_score,
        "prompt_quality_score": prompt_quality_score,
        "missed_context": notes if manual_correction in {"medium", "high"} else "",
        "what_worked": [
            "Changed files matched the planned/selected scope." if match is True else "",
            "Compile passed." if compile_passed is True else "",
            "Tests passed." if tests_passed is True else "",
        ],
        "what_failed": [item for item in what_failed if item],
        "checker_missed": checker_missed,
        "forbidden_changes_touched": forbidden_hits,
        "review_rules_triggered": rules_triggered,
        "checker_status": checker_status,
        "acceptance_tests": plan.get("acceptance_tests", []),
        "forbidden_changes": forbidden_changes,
        "revised_codex_prompt": plan.get("revised_codex_prompt", ""),
        "notes": notes,
        "proposed_candidate_lessons": [],
        "next_prompt_improvement": "",
        "decision": decision,
    }
    if decision != "useful":
        review["next_prompt_improvement"] = "Tighten selected-file scope, verification commands, and hard constraints before rerunning Codex/Cursor."
    elif manual_correction == "low":
        review["next_prompt_improvement"] = "Keep the same structure; minor wording improvements only."
    else:
        review["next_prompt_improvement"] = "Use this review as a passing baseline for similar tasks."

    if decision != "useful" or manual_correction in {"medium", "high"} or files_touched_correct is False or checker_missed:
        review["proposed_candidate_lessons"] = candidate_lessons_from_review(review)

    md_path = config.ENGINEER_REVIEWS_DIR / f"{review_id}.md"
    review_json_path = config.ENGINEER_REVIEWS_DIR / f"{review_id}.json"
    # Review artifacts are always saved so eval history and lesson provenance stay auditable.
    md_path.write_text(artifacts.review_markdown(review), encoding="utf-8")
    review_json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    append_jsonl(
        config.ENGINEER_FILES["eval_history"],
        {
            "type": "engineer_review",
            "review_id": review_id,
            "source_run_id": review["source_run_id"],
            "decision": decision,
            "plan_followed_score": plan_followed_score,
            "prompt_quality_score": prompt_quality_score,
            "compile_passed": compile_passed,
            "tests_passed": tests_passed,
            "files_touched_correct": files_touched_correct,
            "manual_correction": manual_correction,
            "created_at": now,
            "path": str(review_json_path),
        },
    )

    j_space_fields = jspace.update_manifest(
        review["source_run_id"],
        "review",
        decision,
        "reviewed" if decision == "useful" else "needs_revision",
        (
            f"decision={decision}; plan_followed_score={plan_followed_score}; "
            f"prompt_quality_score={prompt_quality_score}"
        ),
        jspace.artifact_ref("engineer_review", review_id, md_path, review_json_path),
        {
            "decision": decision,
            "plan_followed_score": plan_followed_score,
            "prompt_quality_score": prompt_quality_score,
            "compile_passed": compile_passed,
            "tests_passed": tests_passed,
            "files_touched_correct": files_touched_correct,
        },
    )

    review.update(
        {
            "ok": True,
            "path": str(md_path),
            "jsonPath": str(review_json_path),
            "output": f"Engineer review {review_id}: decision={decision}, plan_followed_score={plan_followed_score}, prompt_quality_score={prompt_quality_score}",
            **j_space_fields,
        }
    )
    return review


def engineer_patch_review(patch_id: str) -> dict:
    """Deterministic check plus an independent model reviewer. Never applies anything.

    The model reviewer is advisory: an unavailable reviewer degrades the status to
    "unavailable", but the deterministic checker alone can still fail the patch.
    """
    config.ensure_harness()
    patch_path = artifacts.resolve_artifact_json(patch_id, config.ENGINEER_PATCHES_DIR)
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    review_id = unique_artifact_id("_engineer_patch_review", config.ENGINEER_REVIEWS_DIR)
    deterministic = check_patch_payload(patch, allow_package_json=False, allow_delete=False)
    prompt = prompts.render(
        "patch_review",
        {
            "patch_artifact": json.dumps(
                {
                    key: patch.get(key)
                    for key in [
                        "task",
                        "target_files",
                        "change_summary",
                        "unified_diff",
                        "verification_commands",
                        "grounding_required",
                        "available_claim_ids",
                        "patch_claim_trace",
                        "limitations",
                        "model_route",
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            "deterministic_checker": json.dumps(deterministic, ensure_ascii=False, indent=2),
        },
    )
    model_route = ""
    model_attempts = []
    try:
        routed = models.call_tier_with_fallback(
            "QA/Release",
            prompt,
            tier="reasoning_elite",
            max_cooldown_wait_seconds=0,
            cloud_request_timeout_seconds=45,
            cloud_max_retries=0,
            max_tokens=2200,
        )
        model_route = models.route_of(routed)
        model_attempts = models.attempts_of(routed)
        model_review = extract_json_object(routed.content)
    except Exception as exc:  # noqa: BLE001
        model_review = {
            "verdict": "unavailable",
            "risk_level": patch.get("risk_level") or "high",
            "findings": [f"Independent model review unavailable: {str(exc)[:700]}"],
            "unsupported_hunks": [],
            "missing_tests": [],
            "review_summary": "Deterministic review completed; independent model review was unavailable.",
        }
        model_attempts = [{"status": "failed", "detail": str(exc)[:900]}]
    if not isinstance(model_review, dict):
        model_review = {"verdict": "unavailable", "findings": ["Reviewer returned invalid JSON."]}
    generation_route = str(patch.get("model_route") or "")
    payload = {
        "review_id": review_id,
        "patch_id": patch.get("patch_id") or patch_path.stem,
        "source_run_id": patch.get("source_run_id") or "",
        "created_at": now_tz_iso(),
        "deterministic_checker": deterministic,
        "model_review": model_review,
        "generation_model_route": generation_route,
        "review_model_route": model_route,
        "model_route_independent": bool(model_route and generation_route and model_route != generation_route),
        "model_attempts": model_attempts,
        "apply_performed": False,
    }
    payload["review_status"] = (
        "fail"
        if deterministic.get("status") == "fail" or model_review.get("verdict") == "request_changes"
        else "unavailable"
        if model_review.get("verdict") == "unavailable"
        else "pass"
    )
    md_path = config.ENGINEER_REVIEWS_DIR / f"{review_id}.md"
    json_path = config.ENGINEER_REVIEWS_DIR / f"{review_id}.json"
    md_path.write_text(
        "# Independent Engineer Patch Review\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_run_id = str(payload.get("source_run_id") or "")
    j_space_fields = {}
    if source_run_id:
        j_space_fields = jspace.update_manifest(
            source_run_id,
            "patch_review",
            payload["review_status"],
            "patch_proposed" if payload["review_status"] == "pass" else "blocked",
            str(model_review.get("review_summary") or "Independent patch review recorded."),
            jspace.artifact_ref("engineer_patch_review", review_id, md_path, json_path),
            {
                "patch_id": payload["patch_id"],
                "review_model_route": model_route,
                "model_route_independent": payload["model_route_independent"],
                "apply_performed": False,
            },
        )
    return {
        **payload,
        "ok": payload["review_status"] == "pass",
        "path": str(md_path),
        "jsonPath": str(json_path),
        "output": (
            f"Patch review {review_id}: status={payload['review_status']}; "
            f"model_independent={payload['model_route_independent']}; apply_performed=false"
        ),
        **j_space_fields,
    }


def engineer_feedback(
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
    """Append human feedback and, when warranted, one candidate lesson."""
    config.ensure_harness()
    run_id = run_id.strip()
    if not run_id:
        raise RuntimeError("engineer-feedback requires --run-id.")
    now = now_iso()
    from datetime import datetime

    feedback_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + "_engineer_feedback"
    feedback = {
        "type": "engineer_feedback",
        "feedback_id": feedback_id,
        "run_id": run_id,
        "usefulness": usefulness,
        "correctness": correctness,
        "accepted": accepted,
        "notes": notes,
        "codex_success": codex_success,
        "compile_passed": compile_passed,
        "files_touched_correct": files_touched_correct,
        "manual_correction": manual_correction,
        "lesson_needed": lesson_needed,
        "created_at": now,
    }
    append_jsonl(config.ENGINEER_FILES["eval_history"], feedback)
    lesson = None
    correction_high = manual_correction in {"medium", "high"}
    should_create_lesson = notes.strip() and lesson_needed is not False and (
        usefulness >= 4
        or correctness <= 2
        or not accepted
        or codex_success is False
        or compile_passed is False
        or files_touched_correct is False
        or correction_high
        or lesson_needed is True
    )
    if should_create_lesson:
        lesson = {
            "lesson_id": f"lesson_{uuid.uuid4().hex[:12]}",
            "source_run_id": run_id,
            "lesson_type": "feedback",
            "statement": notes.strip(),
            "applies_to": "Engineer Employee planning behavior",
            "confidence": "medium" if accepted and usefulness >= 4 else "low",
            "status": "candidate",
            "created_at": now,
        }
        append_jsonl(config.ENGINEER_FILES["candidate_lessons"], lesson)
    feedback_ref = jspace.artifact_ref(
        "engineer_feedback",
        feedback_id,
        config.ENGINEER_FILES["eval_history"],
    )
    jspace.update_manifest(
        run_id,
        "feedback",
        "complete",
        "feedback_recorded",
        (
            f"usefulness={usefulness}; correctness={correctness}; accepted={str(accepted).lower()}; "
            f"lesson_created={str(bool(lesson)).lower()}"
        ),
        feedback_ref,
        {
            "feedback_id": feedback_id,
            "usefulness": usefulness,
            "correctness": correctness,
            "accepted": accepted,
            "lesson_created": bool(lesson),
        },
    )
    if lesson:
        j_space_fields = jspace.update_manifest(
            run_id,
            "lesson_gate",
            "manual_review_required",
            "awaiting_lesson_decision",
            f"Candidate lesson {lesson['lesson_id']} requires a human decision.",
            jspace.artifact_ref(
                "engineer_candidate_lesson",
                lesson["lesson_id"],
                config.ENGINEER_FILES["candidate_lessons"],
            ),
        )
    else:
        j_space_fields = jspace.update_manifest(
            run_id,
            "lesson_gate",
            "not_required",
            "complete",
            "Feedback recorded no durable lesson candidate.",
            feedback_ref,
        )
    return {
        "ok": True,
        "run_id": run_id,
        "feedback_id": feedback_id,
        "output": "Feedback recorded. Candidate lesson created." if lesson else "Feedback recorded. No candidate lesson created.",
        "path": str(config.ENGINEER_FILES["eval_history"]),
        "jsonPath": str(config.ENGINEER_FILES["candidate_lessons"]) if lesson else "",
        "candidateLesson": lesson,
        **j_space_fields,
    }


def _resolve_candidate_lesson(lesson_id: str) -> dict:
    candidates = read_jsonl(config.ENGINEER_FILES["candidate_lessons"])
    selected = next((item for item in candidates if item.get("lesson_id") == lesson_id), None)
    if not selected:
        raise RuntimeError(f"Candidate lesson not found: {lesson_id}")
    if any(item.get("lesson_id") == lesson_id for item in read_jsonl(config.ENGINEER_FILES["approved_lessons"])):
        raise RuntimeError(f"Lesson already approved: {lesson_id}")
    if any(item.get("lesson_id") == lesson_id for item in read_jsonl(config.ENGINEER_FILES["rejected_lessons"])):
        raise RuntimeError(f"Lesson already rejected: {lesson_id}")
    return selected


def engineer_approve_lesson(
    lesson_id: str,
    evidence_file: str = "",
    yes: bool = False,
) -> dict:
    """Compatibility seam routed through the sole behavior-promotion gate."""
    from .behavior import engineer_promote_lesson

    return engineer_promote_lesson(lesson_id, evidence_file, yes=yes)


def engineer_reject_lesson(lesson_id: str, reason: str) -> dict:
    config.ensure_harness()
    lesson_id = lesson_id.strip()
    reason = reason.strip()
    if not lesson_id:
        raise RuntimeError("engineer-reject-lesson requires --lesson-id.")
    if not reason:
        raise RuntimeError("engineer-reject-lesson requires --notes with the rejection reason.")
    selected = _resolve_candidate_lesson(lesson_id)
    rejected = {
        **selected,
        "status": "rejected",
        "rejection_reason": reason,
        "rejected_at": now_iso(),
    }
    append_jsonl(config.ENGINEER_FILES["rejected_lessons"], rejected)
    source_run_id = str(selected.get("source_run_id") or "")
    j_space_fields = jspace.update_manifest(
        source_run_id,
        "lesson_gate",
        "rejected",
        "complete",
        f"Human rejected candidate lesson {lesson_id}: {reason}",
        jspace.artifact_ref("engineer_rejected_lesson", lesson_id, config.ENGINEER_FILES["rejected_lessons"]),
    ) if source_run_id else {}
    return {
        "ok": True,
        "output": f"Rejected lesson {lesson_id}. Canonical behavior was not edited.",
        "path": str(config.ENGINEER_FILES["rejected_lessons"]),
        "rejectedLesson": rejected,
        **j_space_fields,
    }


def engineer_version(summary: str) -> dict:
    config.ensure_harness()
    summary = summary.strip()
    if not summary:
        raise RuntimeError("engineer-version requires --summary.")
    history = read_jsonl(config.ENGINEER_FILES["version_history"])
    previous = history[-1].get("version") if history else None
    version = f"engineer-v{len(history) + 1}"
    record = {
        "version": version,
        "previous_version": previous,
        "summary": summary,
        "approved_lesson_count": len(read_jsonl(config.ENGINEER_FILES["approved_lessons"])),
        "created_at": now_iso(),
    }
    append_jsonl(config.ENGINEER_FILES["version_history"], record)
    return {
        "ok": True,
        "output": f"Created {version}. Canonical behavior was not edited.",
        "path": str(config.ENGINEER_FILES["version_history"]),
        "version": version,
    }
