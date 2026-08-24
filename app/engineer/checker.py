"""Deterministic plan checker: the role-specific ruleset.

No model output is evidence here. Every rule is a text/filesystem predicate over the
plan the model produced, so a worse model produces a worse plan that this still
catches - it cannot talk its way past a rule. `failed_rules` blocks; `warnings` does not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from engineer_grounding import GROUNDING_CONTRACT, validate_grounded_plan

from . import config
from .util import dedupe


DEPENDENCY_TERMS = [
    "react-markdown",
    "marked",
    "remark",
    "rehype",
    "highlight.js",
    "dompurify",
    "markdown-it",
]

CODE_MARKERS = [
    "change", "modify", "update", "implement", "fix", "add", "remove", "refactor",
    "style", "styling", "rendering", "code", "ui", "frontend", "backend", "api",
    "server", "component", "file", "files",
]
NON_CODE_MARKERS = ["explain", "summarize", "review only", "brainstorm", "compare"]


def run_checker(parsed: dict, selected_records: list[dict], raw_output: str = "", task: str = "") -> dict:
    warnings: list[str] = []
    failed_rules: list[str] = []
    suggested_corrections: list[str] = []
    selected_paths = {Path(item["path"]).as_posix().lower() for item in selected_records if item.get("path")}
    selected_names = {Path(item["path"]).name.lower() for item in selected_records if item.get("path")}
    actionable_text = checker_actionable_text(parsed)
    text = actionable_text.lower()
    task_lower = task.lower()
    package_deps = package_dependencies()

    if is_code_modification_task(task) and not selected_records:
        failed_rules.append("code_task_missing_selected_files")
        warnings.append("Code modification task has no selected files; blocking Codex-ready implementation prompt.")
        suggested_corrections.append("Ask the user to select likely files such as package metadata, entry components, styles, backend bridge files, or tests.")

    if selected_records:
        likely_files = parsed.get("files_likely_involved", [])
        if isinstance(likely_files, list):
            for file_text in likely_files:
                if not isinstance(file_text, str):
                    continue
                normalized = file_text.replace("\\", "/").lower()
                file_name = Path(normalized).name.lower()
                if normalized and normalized not in selected_paths and file_name not in selected_names:
                    resolved = config.ROOT / file_text
                    if resolved.exists():
                        warnings.append(f"Mentions existing file that was not selected: {file_text}")
                        suggested_corrections.append(f"Either select `{file_text}` or keep the plan scoped to selected files.")
                    else:
                        warnings.append(
                            f"Mentions nonexistent or unverified file path (non-blocking): {file_text}"
                        )
                        suggested_corrections.append(
                            f"Drop `{file_text}` from the plan or replace it with a path that exists."
                        )

    for file_text in mentioned_repo_paths(text):
        resolved = config.ROOT / file_text
        normalized = file_text.replace("\\", "/").lower()
        if normalized not in selected_paths and Path(normalized).name.lower() not in selected_names:
            if resolved.exists():
                warnings.append(f"Plan references unselected existing file: {file_text}")
            elif obvious_repo_path_exists(file_text):
                warnings.append(f"Plan references unselected existing file by name: {file_text}")
            else:
                warnings.append(
                    f"Plan references missing or unverified file path (non-blocking): {file_text}"
                )
                suggested_corrections.append(
                    f"Remove `{file_text}` or verify it exists before treating it as required evidence."
                )

    for dep in DEPENDENCY_TERMS:
        if dep in text and dep not in package_deps and proposes_dependency(text, dep):
            failed_rules.append("proposed_dependency_without_permission")
            warnings.append(f"Final actionable plan proposes unavailable dependency: {dep}")
            suggested_corrections.append(f"Do not suggest `{dep}` unless package metadata or selected files verify it is available.")
    if proposes_any_new_dependency(text) and ("no depend" in task_lower or "without adding" in task_lower or "do not add depend" in task_lower):
        failed_rules.append("new_dependency_conflicts_with_task")
        warnings.append("Plan proposes dependency/package changes even though task forbids new dependencies.")

    if "already renders markdown as html" in text or "renders markdown as html" in text:
        failed_rules.append("unsupported_rendering_assumption")
        warnings.append("Plan assumes artifact Markdown is already rendered as HTML; current pane may display raw text.")
    if "dangerouslysetinnerhtml" in text:
        warnings.append("Plan mentions dangerouslySetInnerHTML; verify current component before using DOM injection.")
    if "record_track_run" in text:
        failed_rules.append("wrong_feedback_storage")
        warnings.append("Plan mentions record_track_run; fresh Engineer feedback should use brain_v2 engineer eval/candidate lesson files.")
    if "brain_v2/evals/engineer/eval_history" in text.replace("\\", "/"):
        failed_rules.append("wrong_eval_history_path")
        warnings.append("Plan uses wrong eval path; current Engineer eval history is brain_v2/employees/engineer/eval_history.jsonl.")
    if raw_output and not parsed:
        warnings.append("Model returned content, but no valid top-level JSON object could be parsed.")

    if any(term in task_lower for term in ["frontend-only", "without changing backend", "do not change backend", "no backend"]):
        backend_change_terms = [
            "modify backend",
            "change backend",
            "backend change",
            "server/",
            "api route",
            "endpoint",
            "company-brain-api",
            "web-gui/server",
            "api bridge",
            "request payload",
            "endpoint",
        ]
        if any(proposes_backend_change(text, term) for term in backend_change_terms):
            failed_rules.append("backend_change_for_frontend_task")
            warnings.append("Plan mentions backend/API changes even though task says not to change backend behavior.")
            suggested_corrections.append("Remove backend/API/server changes from plan and Codex prompt.")
    if any(term in text for term in ["browse", "web search", "external web", "fetch url", "research broker"]) and "web_enabled" not in text:
        warnings.append("Plan may imply web/API usage; ensure web remains disabled unless explicitly enabled.")
    if any(term in text for term in ["legacy brain", "brain/ideas", "old idea", "candidate lesson"]):
        if "candidate lessons loaded" not in text:
            warnings.append("Plan references legacy/candidate context; verify it is explicitly selected before use.")

    steps = parsed.get("implementation_plan", [])
    if not isinstance(steps, list) or not steps:
        failed_rules.append("missing_implementation_plan")
        warnings.append("Implementation plan is missing or not a list.")
    else:
        vague = [step for step in steps if isinstance(step, str) and len(step.strip()) < 32]
        if len(vague) >= max(2, len(steps) // 2):
            warnings.append("Implementation plan contains multiple vague steps.")
            suggested_corrections.append("Revise implementation steps to name concrete files, functions, and checks.")

    tests = parsed.get("acceptance_tests", [])
    if not isinstance(tests, list) or not tests:
        failed_rules.append("missing_acceptance_tests")
        warnings.append("Acceptance tests are missing.")
    else:
        test_text = " ".join(str(item).lower() for item in tests)
        if not any(cmd in test_text for cmd in ["npm", "python", "py_compile", "tsc", "build", "pytest", "compile"]):
            warnings.append("Acceptance tests do not include a concrete verification command.")
            suggested_corrections.append("Add at least one concrete command such as `python -m py_compile ...` or `npm run build`.")

    verified_facts = parsed.get("verified_facts", [])
    unverified = parsed.get("unverified_assumptions", [])
    if not isinstance(verified_facts, list) or not verified_facts:
        warnings.append("verified_facts is missing or empty.")
        suggested_corrections.append("Add facts grounded in selected file inspections and context manifest.")
    if not isinstance(unverified, list):
        warnings.append("unverified_assumptions must be a list.")

    if parsed.get("grounding_contract_version") == GROUNDING_CONTRACT:
        evidence_ids = {
            str(item.get("evidence_id") or "")
            for item in parsed.get("evidence_index") or []
            if isinstance(item, dict) and item.get("evidence_id")
        }
        grounding = validate_grounded_plan(parsed, evidence_ids)
        failed_rules.extend(grounding.get("failed_rules") or [])
        warnings.extend(grounding.get("warnings") or [])

    status = "fail" if failed_rules else "warn" if warnings else "pass"
    return {
        "status": status,
        "warnings": dedupe(warnings),
        "failed_rules": dedupe(failed_rules),
        "suggested_corrections": dedupe(suggested_corrections),
    }


def package_dependencies() -> set[str]:
    deps = set()
    package_path = config.ROOT / "web-gui" / "package.json"
    if not package_path.exists():
        return deps
    try:
        data = json.loads(package_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return deps
    for section in ["dependencies", "devDependencies"]:
        deps.update((data.get(section) or {}).keys())
    return deps


def mentioned_repo_paths(text: str) -> list[str]:
    patterns = [
        r"(?:web-gui|scripts|app|brain_v2|brain|config|docs)/[A-Za-z0-9_./-]+",
        r"\b[A-Za-z0-9_-]+\.(?:jsonl|tsx|jsx|mjs|json|css|py|ts|js|md)\b",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(match.group(0).strip("`'\".,)") for match in re.finditer(pattern, text))
    return dedupe(paths)


def checker_actionable_text(parsed: dict) -> str:
    """The text the checker judges: whatever would actually be handed to an executor."""
    adjusted = parsed.get("checker_adjusted_plan") if isinstance(parsed.get("checker_adjusted_plan"), dict) else {}
    candidates = [
        adjusted.get("revised_codex_prompt"),
        adjusted.get("codex_prompt"),
        parsed.get("revised_codex_prompt"),
        parsed.get("codex_prompt"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    steps = parsed.get("implementation_plan", [])
    if isinstance(steps, list):
        return "\n".join(str(step) for step in steps)
    return str(steps) if steps else ""


def is_code_modification_task(task: str) -> bool:
    task_lower = task.lower()
    return any(marker in task_lower for marker in CODE_MARKERS) and not any(
        marker in task_lower for marker in NON_CODE_MARKERS
    )


def negated_dependency_mention(text: str, dep: str) -> bool:
    escaped = re.escape(dep)
    negation_patterns = [
        rf"do not (?:import|use|add|install)\s+{escaped}",
        rf"don't (?:import|use|add|install)\s+{escaped}",
        rf"without (?:adding|installing|importing|using)\s+{escaped}",
        rf"avoid\s+{escaped}",
        rf"{escaped}\s+is not installed",
        rf"{escaped}\s+is unavailable",
        rf"package\.json does not include\s+{escaped}",
        rf"does not include\s+{escaped}",
        rf"not include\s+{escaped}",
    ]
    return any(re.search(pattern, text) for pattern in negation_patterns)


def proposes_dependency(text: str, dep: str) -> bool:
    if negated_dependency_mention(text, dep):
        return False
    escaped = re.escape(dep)
    proposal_patterns = [
        rf"\b(?:npm|pnpm|yarn)\s+(?:install|add)\s+[^.\n]*{escaped}",
        rf"\binstall\s+{escaped}",
        rf"\badd\s+{escaped}",
        rf"\bimport\s+[^.\n]*{escaped}",
        rf"\buse\s+{escaped}\s+as\s+(?:an?\s+)?implementation dependency",
        rf"modify\s+package\.json\s+(?:to\s+)?(?:include|add)\s+{escaped}",
        rf"package\.json\s+.*(?:include|add)\s+{escaped}",
    ]
    return any(re.search(pattern, text) for pattern in proposal_patterns)


def proposes_any_new_dependency(text: str) -> bool:
    if re.search(r"\bdo not (?:add|install|use|import)\b", text) or "without adding" in text or "avoid " in text:
        return False
    return any(
        re.search(pattern, text)
        for pattern in [
            r"\b(?:npm|pnpm|yarn)\s+(?:install|add)\b",
            r"\badd dependency\b",
            r"\bnew dependency\b",
            r"modify\s+package\.json\s+(?:to\s+)?(?:include|add)",
        ]
    )


def proposes_backend_change(text: str, term: str) -> bool:
    escaped = re.escape(term)
    change_verbs = r"(?:modify|change|edit|update|add|remove|rewrite|create|wire|alter)"
    if re.search(rf"\bdo not\s+{change_verbs}?\s*[^.\n]*{escaped}", text):
        return False
    patterns = [
        rf"\b{change_verbs}\s+[^.\n]*{escaped}",
        rf"{escaped}[^.\n]*\b{change_verbs}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def obvious_repo_path_exists(file_text: str) -> bool:
    candidate = Path(file_text)
    if len(candidate.parts) > 1:
        return False
    try:
        return next(config.ROOT.rglob(file_text), None) is not None
    except OSError:
        return False
