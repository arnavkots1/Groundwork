"""Deterministic plan payloads used when no model plan is available or allowed.

These are structured refusals, not pretend answers. Each one carries an explicit
`limitations`/`context_sufficiency` signal so downstream gates keep blocking, and
neither ever produces a Codex-ready implementation prompt it cannot justify.
"""

from __future__ import annotations

from .util import truncate


def fallback_engineer_output(
    task: str, raw_output: str, reason: str, selected_records: list[dict] | None = None
) -> dict:
    """A plan skeleton for runs where the model route failed or returned no JSON.

    The one case treated as sufficient is the deterministic literal-replacement path,
    where the change was already validated against the file without a model.
    """
    selected_paths = [item["input"] for item in selected_records or [] if item.get("exists")]
    deterministic_exact = "deterministic planning path" in reason.lower()
    return {
        "task_understanding": f"Produce a scoped Engineer Employee plan for: {task}",
        "verified_facts": [
            f"Selected file exists: {path}" for path in selected_paths
        ],
        "unverified_assumptions": [
            "No model-grounded implementation plan is available for this run.",
        ],
        "repo_grounding_score": 2 if selected_paths else 1,
        "grounding_contract_version": "selected_context_only",
        "context_sufficiency": {
            "status": "sufficient" if deterministic_exact and selected_paths else "unavailable",
            "known_unknowns": [] if deterministic_exact else ["No grounded model plan is available."],
            "unresolved_questions": [] if deterministic_exact else [{"question": "What repository context is required?", "risk": "high"}],
            "assumptions": [],
        },
        "context_requests": [],
        "material_claims": [],
        "assumptions": [
            reason,
            "Only explicit user task text and harness files should be treated as context.",
        ],
        "files_likely_involved": selected_paths or [
            "brain_v2/system/",
            "brain_v2/employees/engineer/",
            "scripts/company_brain_action.py",
        ],
        "implementation_plan": [
            "Confirm the task intent and any explicitly selected files.",
            "Build or inspect the context manifest before planning.",
            "Verify candidate lessons are excluded and approved lessons load only when requested.",
            "Check that the run writes Markdown and JSON artifacts under brain_v2/employees/engineer/runs/.",
            "Review route attempts and limitations before using the Codex prompt.",
        ],
        "risks": [
            "Provider cooldowns can make model-backed planning unavailable.",
            "Hidden legacy context would bias the new Employee Lab if loaded automatically.",
            "Feedback could become canonical behavior if approval remains unclear.",
        ],
        "forbidden_changes": [
            "Do not delete or rewrite the existing brain/ system.",
            "Do not auto-load legacy idea runs or candidate lessons.",
            "Do not approve lessons or edit canonical behavior automatically.",
            "Do not add autonomous background learning or database infrastructure.",
        ],
        "acceptance_tests": [
            "python -m py_compile scripts/company_brain_action.py",
            "python scripts/company_brain_action.py status",
            "python scripts/company_brain_action.py engineer-plan --task \"Create a small test plan for the Engineer Employee harness\" --save",
            "Confirm the saved JSON has context_manifest.legacy_brain_auto_loaded=false and candidate_lessons_loaded=false.",
            "Confirm feedback appends eval_history.jsonl and creates only candidate lessons when warranted.",
        ],
        "codex_prompt": f"Review this task and produce a scoped implementation plan with acceptance tests:\n\n{task}",
        "revised_codex_prompt": f"Review this task using only selected files and produce a grounded plan with verified facts, assumptions, and acceptance tests:\n\n{task}",
        "self_review_checklist": [
            "Context manifest excludes hidden legacy memory.",
            "Selected files exist or the run hard-stops.",
            "Plan is scoped to the Engineer Employee harness.",
            "Acceptance tests are executable.",
            "Lesson approval remains manual.",
        ],
        "post_run_review_template": "Run result:\nTests passed:\nTests failed:\nFiles changed:\nUnexpected context used:\nCandidate lesson to propose, if any:\nDo not approve automatically:",
        "rubric_self_score": {
            "task_understanding": 3,
            "implementation_specificity": 3,
            "file_path_clarity": 3,
            "scope_control": 4,
            "risk_detection": 4,
            "acceptance_test_quality": 3,
            "codex_prompt_quality": 2,
            "likely_execution_success": 3,
            "avoids_unnecessary_refactors": 5,
            "respects_constraints": 5,
        },
        "limitations": [reason, truncate(raw_output, 1200)],
    }


def blocked_engineer_output(task: str) -> dict:
    """A code-modification task with no selected files: refuse, and say what is needed."""
    context_request = (
        "Select the likely files before generating an implementation prompt. "
        "For UI work, include package metadata, the relevant React/component file, styles, and any bridge/test files that may be touched."
    )
    return {
        "blocked_plan": True,
        "context_request": context_request,
        "task_understanding": f"Code-modification task needs selected repo files before planning: {task}",
        "verified_facts": [],
        "unverified_assumptions": [
            "The task appears to modify code, UI, frontend, backend, or files, but no selected files were provided.",
        ],
        "repo_grounding_score": 1,
        "assumptions": [
            "No file-specific implementation plan should be generated without selected repo evidence.",
        ],
        "files_likely_involved": [],
        "implementation_plan": [
            "Ask the user to rerun engineer-plan with selected files relevant to the intended code change.",
            "After files are selected, inspect the selected file previews, symbols, package metadata, and task matches before writing a Codex prompt.",
        ],
        "risks": [
            "A code plan without selected files can invent paths, dependencies, rendering behavior, functions, or tests.",
        ],
        "forbidden_changes": [
            "Do not emit a Codex-ready implementation prompt for this blocked run.",
            "Do not assume file paths, dependencies, backend behavior, or rendering libraries.",
        ],
        "acceptance_tests": [
            "python -m py_compile scripts/company_brain_action.py",
            "python scripts/company_brain_action.py engineer-plan --task \"Change the web GUI artifact pane styling and markdown rendering\" --save",
            "Confirm the saved JSON has blocked_plan=true, checker_status=fail, repo_grounding_score=1, and no Codex-ready prompt.",
        ],
        "codex_prompt": "",
        "revised_codex_prompt": "",
        "self_review_checklist": [
            "No selected files means no executable prompt.",
            "Context request names the type of files needed.",
            "The artifact is saved for audit but marked blocked.",
        ],
        "post_run_review_template": "Blocked run:\nFiles user should select:\nReason no Codex prompt was emitted:\n",
        "rubric_self_score": {
            "task_understanding": 4,
            "repo_grounding": 1,
            "assumption_discipline": 5,
            "selected_file_fidelity": 5,
            "codex_prompt_quality": 1,
            "respects_constraints": 5,
        },
        "limitations": [
            "No selected files were provided for a likely code-modification task.",
        ],
    }
