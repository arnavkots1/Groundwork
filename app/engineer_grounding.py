"""Stage 3A context requests, evidence catalogs, and repository maps.

This module is deterministic and repository-local. It does not call models,
execute commands, perform network access, or grant write permission.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from repository_retrieval import (
    DEFAULT_FILE_TYPES,
    DEFAULT_MAX_EXCERPTS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOTAL_CHARS,
    IGNORED_DIRECTORIES,
    SUPPORTED_INTENTS,
    _inside,
    _is_config_path,
    _is_test_path,
    _iter_files,
    _path_block_reason,
    _read_candidate,
    normalize_authorized_intents,
    normalize_authorized_roots,
)


CONTEXT_REQUEST_SCHEMA = "companybrain.engineer.context_request.v1"
REPOSITORY_MAP_SCHEMA = "companybrain.engineer.repository_map.v1"
GROUNDING_CONTRACT = "companybrain.engineer.grounding.stage3a.v1"
CONTEXT_STATUSES = {"sufficient", "incomplete", "contradictory", "unavailable"}
CLAIM_INFLUENCES = {
    "scope",
    "architecture",
    "safety",
    "dependency",
    "proposed_edit",
    "test_selection",
}


def _safe_identifier(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"{field} must contain only letters, numbers, dot, underscore, or hyphen")
    return text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _root_mode(
    requested_roots: list[str],
    requested_intent: str,
    authorized_roots: list[str],
    authorized_intents: list[str],
    repo_root: Path,
) -> str:
    root = repo_root.resolve()
    requested_paths = [(root / value).resolve() for value in requested_roots]
    authorized_paths = [(root / value).resolve() for value in authorized_roots]
    roots_authorized = all(any(_inside(path, allowed) for allowed in authorized_paths) for path in requested_paths)
    return "pre_authorized" if roots_authorized and requested_intent in authorized_intents else "request_approval"


def classify_context_request(
    request: dict,
    repo_root: Path,
    authorized_roots: list[str],
    authorized_intents: list[str],
) -> dict:
    """Classify a structured context request without exposing repository content."""

    root = repo_root.resolve()
    request_id = _safe_identifier(request.get("request_id"), "request_id")
    task_id = _safe_identifier(request.get("task_id"), "task_id")
    if request.get("external_retrieval") not in {None, False}:
        raise ValueError("Stage 3A context requests cannot enable external retrieval")
    raw_items = request.get("requested_items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 12:
        raise ValueError("requested_items must contain between 1 and 12 structured items")

    normalized_authorized_roots = normalize_authorized_roots(authorized_roots, root)
    normalized_authorized_intents = normalize_authorized_intents(authorized_intents)
    items: list[dict] = []
    for index, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"requested_items[{index - 1}] must be an object")
        item_id = _safe_identifier(raw.get("item_id") or f"item_{index:02d}", "item_id")
        intent = str(raw.get("intent") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if intent not in SUPPORTED_INTENTS:
            mode = "denied"
            decision_reason = "unsupported_intent"
            requested_roots: list[str] = []
        elif not reason:
            mode = "denied"
            decision_reason = "reason_required"
            requested_roots = []
        else:
            raw_roots = raw.get("allowed_roots") or raw.get("requested_roots")
            if not isinstance(raw_roots, list) or not raw_roots:
                mode = "denied"
                decision_reason = "allowed_roots_required"
                requested_roots = []
            else:
                try:
                    requested_roots = normalize_authorized_roots([str(value) for value in raw_roots], root)
                    denied_reason = ""
                    for value in requested_roots:
                        candidate = (root / value).resolve()
                        denied_reason = _path_block_reason(candidate, root)
                        if candidate.name.lower() in IGNORED_DIRECTORIES:
                            denied_reason = "generated_or_vendor_directory_denied"
                        if denied_reason:
                            break
                    path_value = str(raw.get("path") or "").strip()
                    if path_value:
                        candidate = (root / path_value).resolve()
                        if not _inside(candidate, root):
                            denied_reason = "exact_path_outside_repository"
                        elif not candidate.exists():
                            denied_reason = "path_does_not_exist"
                        else:
                            denied_reason = _path_block_reason(candidate, root)
                        if not denied_reason and not any(
                            _inside(candidate, (root / value).resolve()) for value in requested_roots
                        ):
                            denied_reason = "exact_path_outside_requested_roots"
                    if denied_reason:
                        mode = "denied"
                        decision_reason = denied_reason
                    else:
                        mode = _root_mode(
                            requested_roots,
                            intent,
                            normalized_authorized_roots,
                            normalized_authorized_intents,
                            root,
                        )
                        decision_reason = (
                            "within_task_intake_scope"
                            if mode == "pre_authorized"
                            else "repository_local_scope_expansion_requires_human_approval"
                        )
                except ValueError as exc:
                    mode = "denied"
                    decision_reason = str(exc)
                    requested_roots = []

        estimated = raw.get("estimated_budget") if isinstance(raw.get("estimated_budget"), dict) else {}
        requested_budget = {
            "max_files": int(estimated.get("max_files", DEFAULT_MAX_FILES)),
            "max_excerpts": int(estimated.get("max_excerpts", DEFAULT_MAX_EXCERPTS)),
            "max_total_chars": int(estimated.get("max_chars", estimated.get("max_total_chars", DEFAULT_MAX_TOTAL_CHARS))),
            "max_file_bytes": int(estimated.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)),
        }
        expanded_fields = [
            field
            for field, default in {
                "max_files": DEFAULT_MAX_FILES,
                "max_excerpts": DEFAULT_MAX_EXCERPTS,
                "max_total_chars": DEFAULT_MAX_TOTAL_CHARS,
                "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
            }.items()
            if requested_budget[field] > default
        ]
        budget_reason = str(raw.get("budget_reason") or "").strip()
        if mode != "denied" and expanded_fields:
            if not budget_reason:
                mode = "denied"
                decision_reason = "expanded_budget_reason_required"
            else:
                mode = "request_approval"
                decision_reason = "expanded_budget_requires_human_approval"
        retrieval_request = {
            "request_id": f"{request_id}.{item_id}",
            "task_id": task_id,
            "intent": intent,
            "query": str(raw.get("query") or raw.get("symbol") or raw.get("target") or "").strip(),
            "allowed_roots": requested_roots,
            "file_types": raw.get("file_types") or sorted(DEFAULT_FILE_TYPES),
            **requested_budget,
            "include_tests": raw.get("include_tests", True) is not False,
            "external_retrieval": False,
        }
        if raw.get("path"):
            retrieval_request["path"] = str(raw["path"])
        if raw.get("start_line") is not None or raw.get("offset_line") is not None:
            retrieval_request["start_line"] = raw.get("start_line", raw.get("offset_line"))
        if budget_reason:
            retrieval_request["budget_reason"] = budget_reason
        items.append(
            {
                "item_id": item_id,
                "intent": intent,
                "reason": reason,
                "mode": mode,
                "decision_reason": decision_reason,
                "requested_roots": requested_roots,
                "budget_expansion": {"expanded": bool(expanded_fields), "expanded_fields": expanded_fields},
                "retrieval_request": retrieval_request,
                "approval": {"required": mode == "request_approval", "approved": False},
                "execution": {"status": "not_run"},
            }
        )

    modes = {item["mode"] for item in items}
    status = "denied" if modes == {"denied"} else "approval_required" if "request_approval" in modes else "ready"
    return {
        "schema": CONTEXT_REQUEST_SCHEMA,
        "request_id": request_id,
        "task_id": task_id,
        "status": status,
        "requested_items": items,
        "summary": {
            "pre_authorized": sum(item["mode"] == "pre_authorized" for item in items),
            "request_approval": sum(item["mode"] == "request_approval" for item in items),
            "denied": sum(item["mode"] == "denied" for item in items),
        },
        "external_retrieval": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


HUMAN_TASK_PREMISE_EVIDENCE_ID = "human_task_premise"


def build_evidence_catalog(
    manifest: dict,
    workspace: Path,
    repo_root: Path,
    selected_excerpt_chars: int = 3500,
    max_prompt_chars: int = 30_000,
    task_text: str = "",
) -> dict:
    """Build the exact evidence pack used for grounded replanning.

    A material claim is sometimes not derived from repository evidence at all -
    it restates a premise the human put directly in the task text (e.g. "observed
    lead-model CLI calls take up to 213s"). Before this, such a claim had no
    evidence_id it could honestly cite and failed closed as claim_without_evidence,
    indistinguishable from a model inventing a fact. There is now exactly one
    additional citable evidence entry, ``human_task_premise``, whose content is the
    literal task text. It is not verified any more strongly than repository
    evidence is - the checker never confirms a claim's content actually matches its
    cited evidence for either kind - but citing it is a mechanically checkable,
    honest label a human reviewer can compare against the same task text recorded
    in the plan, exactly as they would check a citation against a file excerpt.
    """

    root = repo_root.resolve()
    entries: list[dict] = []
    chars_used = 0
    truncated = False
    task_text = str(task_text or "").strip()
    if task_text:
        entries.append(
            {
                "evidence_id": HUMAN_TASK_PREMISE_EVIDENCE_ID,
                "source_type": "human_task_premise",
                "trust": "human_asserted_task_premise",
                "selection_reason": "Literal task text supplied by the human/operator, not repository content.",
                "content": task_text,
            }
        )
    selected_snapshot = workspace / "context" / "selected_excerpts" / "plan_input.json"
    snapshotted_selected = []
    if selected_snapshot.is_file() and workspace.resolve() in selected_snapshot.resolve().parents:
        snapshot_payload = json.loads(selected_snapshot.read_text(encoding="utf-8"))
        snapshotted_selected = snapshot_payload.get("entries") or []
    for selected in snapshotted_selected:
        content = str(selected.get("content") or "")
        if chars_used + len(content) > max_prompt_chars:
            truncated = True
            break
        entries.append({**selected, "source_type": "selected_file"})
        chars_used += len(content)
    for selected in ([] if snapshotted_selected else (manifest.get("selected_context") or [])):
        path_text = str(selected.get("path") or "")
        path = (root / path_text).resolve()
        if not path.is_file() or not _inside(path, root) or _sha256(path) != selected.get("sha256"):
            continue
        content = path.read_text(encoding="utf-8", errors="replace")[:selected_excerpt_chars]
        if chars_used + len(content) > max_prompt_chars:
            truncated = True
            break
        evidence_id = f"selected:{path_text}:{str(selected.get('sha256') or '')[:12]}"
        entries.append(
            {
                "evidence_id": evidence_id,
                "source_type": "selected_file",
                "path": path_text,
                "line_range": "selected excerpt",
                "full_file_sha256": selected.get("sha256"),
                "excerpt_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "selection_reason": "human-selected task context",
                "trust": "untrusted_repository_evidence",
                "content": content,
            }
        )
        chars_used += len(content)

    retrieval_dir = workspace / "context" / "retrieval"
    for request_record in (manifest.get("retrieval") or {}).get("requests") or []:
        request_id = str(request_record.get("request_id") or "")
        snapshot = retrieval_dir / f"{request_id}.json"
        if not snapshot.is_file() or workspace.resolve() not in snapshot.resolve().parents:
            continue
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        for result in payload.get("results") or []:
            for excerpt in result.get("excerpts") or []:
                content = str(excerpt.get("content") or "")
                if chars_used + len(content) > max_prompt_chars:
                    truncated = True
                    break
                evidence_id = (
                    f"{request_id}:{result.get('path')}:{excerpt.get('start_line')}-{excerpt.get('end_line')}"
                )
                entries.append(
                    {
                        "evidence_id": evidence_id,
                        "source_type": "retrieved_excerpt",
                        "retrieval_id": request_id,
                        "path": result.get("path"),
                        "line_range": f"{excerpt.get('start_line')}-{excerpt.get('end_line')}",
                        "full_file_sha256": result.get("sha256"),
                        "excerpt_sha256": excerpt.get("sha256"),
                        "selection_reason": result.get("selection_reason"),
                        "security_flags": result.get("security_flags") or [],
                        "trust": "untrusted_repository_evidence",
                        "content": content,
                    }
                )
                chars_used += len(content)
            if truncated:
                break
        if truncated:
            break
    return {
        "contract": GROUNDING_CONTRACT,
        "entries": entries,
        "evidence_ids": [item["evidence_id"] for item in entries],
        "chars_used": chars_used,
        "max_prompt_chars": max_prompt_chars,
        "truncated": truncated,
    }


def public_evidence_index(catalog: dict) -> list[dict]:
    return [{key: value for key, value in item.items() if key != "content"} for item in catalog.get("entries") or []]


def _unresolved_question_blocks_patch(item: object) -> bool:
    """Return True when an unresolved question must keep Patch blocked.

    Bare strings remain blocking (fail closed for legacy model output). Structured
    questions may declare themselves non-blocking when they are about the wider
    repository and are not required for the proposed change.
    """

    if isinstance(item, str):
        return bool(item.strip())
    if not isinstance(item, dict):
        return True
    blocks = item.get("blocks_patch")
    if isinstance(blocks, bool):
        return blocks
    if str(blocks or "").strip().lower() in {"true", "1", "yes"}:
        return True
    if str(blocks or "").strip().lower() in {"false", "0", "no"}:
        return False
    risk = str(item.get("risk") or "").strip().lower()
    about = str(item.get("about") or item.get("scope") or "").strip().lower()
    if risk == "high":
        return True
    if about in {"broader_repo", "out_of_scope", "optional", "convention", "documentation_convention"}:
        return False
    if about in {"retrieved_evidence", "change_target", "selected_file", "caller", "test", "required"}:
        return risk != "low"
    # Unspecified structured questions: only high risk already handled; medium/empty block.
    return risk in {"", "medium", "high"}


def resolve_evidence_citation(cited: object, valid_evidence_ids: set[str]) -> dict:
    """Resolve a model citation to a canonical catalog evidence_id.

    Exact match wins. Otherwise accept a unique catalog id whose segment before the
    first ``:`` equals the citation, or that starts with ``cited:``. Zero matches
    are unknown; more than one are ambiguous (candidates named). Does not weaken
    grounding: callers must still fail unknown and ambiguous citations.
    """

    text = str(cited or "").strip()
    valid = {str(item) for item in valid_evidence_ids}
    if not text:
        return {"status": "unknown", "cited": text, "canonical": "", "candidates": []}
    if text in valid:
        return {"status": "resolved", "cited": text, "canonical": text, "candidates": [text]}
    prefix = f"{text}:"
    candidates = sorted(
        item
        for item in valid
        if item.startswith(prefix) or item.split(":", 1)[0] == text
    )
    if len(candidates) == 1:
        return {
            "status": "resolved",
            "cited": text,
            "canonical": candidates[0],
            "candidates": candidates,
        }
    if not candidates:
        return {"status": "unknown", "cited": text, "canonical": "", "candidates": []}
    return {
        "status": "ambiguous",
        "cited": text,
        "canonical": "",
        "candidates": candidates,
    }


def validate_grounded_plan(plan: dict, valid_evidence_ids: set[str]) -> dict:
    warnings: list[str] = []
    failed_rules: list[str] = []
    sufficiency = plan.get("context_sufficiency")
    if not isinstance(sufficiency, dict):
        failed_rules.append("missing_context_sufficiency")
    else:
        status = str(sufficiency.get("status") or "")
        if status not in CONTEXT_STATUSES:
            failed_rules.append("invalid_context_sufficiency_status")
        unresolved = sufficiency.get("unresolved_questions") or []
        if not isinstance(unresolved, list):
            failed_rules.append("invalid_unresolved_questions")
            unresolved = []
        blocking_unresolved = [item for item in unresolved if _unresolved_question_blocks_patch(item)]
        # Only a *blocking* high-risk question fails the checker. The prompt tells
        # the model context_requests are for status != sufficient, and lets it mark
        # a broader-repo question blocks_patch=false while still honestly flagging
        # it risk=high. Gating on risk alone (ignoring blocks_patch) made that
        # combination unpassable by design: the model is told not to request more
        # context when sufficient, then fails anyway for the question it correctly
        # judged non-blocking. blocks_patch is the model's own gating signal here -
        # trust it, same as the context_incomplete branch below already does.
        high_risk_blocking = any(
            isinstance(item, dict) and str(item.get("risk") or "").lower() == "high"
            for item in blocking_unresolved
        )
        if high_risk_blocking:
            failed_rules.append("high_risk_context_question_unresolved")
        if status == "sufficient":
            pass
        elif status == "incomplete":
            if blocking_unresolved:
                failed_rules.append("context_incomplete")
            else:
                warnings.append(
                    "Context marked incomplete, but unresolved questions are non-blocking for the proposed change."
                )
        elif status in CONTEXT_STATUSES:
            failed_rules.append(f"context_{status}")

    claims = plan.get("material_claims")
    if not isinstance(claims, list) or not claims:
        failed_rules.append("missing_material_claims")
        claims = []
    seen_claims: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            failed_rules.append("invalid_material_claim")
            continue
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id or claim_id in seen_claims:
            failed_rules.append("invalid_or_duplicate_claim_id")
        seen_claims.add(claim_id)
        evidence_ids = claim.get("evidence_ids") or claim.get("evidence") or []
        if not isinstance(evidence_ids, list) or not evidence_ids:
            failed_rules.append(f"claim_without_evidence:{claim_id or index}")
        else:
            resolved_ids: list[str] = []
            claim_failed = False
            for item in evidence_ids:
                resolution = resolve_evidence_citation(item, valid_evidence_ids)
                if resolution["status"] == "resolved":
                    resolved_ids.append(str(resolution["canonical"]))
                elif resolution["status"] == "ambiguous":
                    candidates = "|".join(str(value) for value in resolution["candidates"])
                    failed_rules.append(
                        f"claim_with_ambiguous_evidence:{claim_id or index}:{resolution['cited']}->{candidates}"
                    )
                    claim_failed = True
                else:
                    failed_rules.append(f"claim_with_unknown_evidence:{claim_id or index}")
                    claim_failed = True
            if not claim_failed and resolved_ids:
                # Rewrite citations to canonical full catalog ids so downstream
                # gates (patch) see exact evidence_index strings.
                claim["evidence_ids"] = resolved_ids
                if "evidence" in claim:
                    claim["evidence"] = list(resolved_ids)
        influences = claim.get("influences") or []
        if not isinstance(influences, list) or not set(map(str, influences)) & CLAIM_INFLUENCES:
            warnings.append(f"Claim {claim_id or index} has no recognized material influence.")
        if str(claim.get("confidence") or "").lower() not in {"low", "medium", "high"}:
            warnings.append(f"Claim {claim_id or index} has no valid confidence level.")

    actions = plan.get("plan_actions")
    if not isinstance(actions, list) or not actions:
        failed_rules.append("missing_plan_actions")
        actions = []
    seen_actions: set[str] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            failed_rules.append("invalid_plan_action")
            continue
        action_id = str(action.get("action_id") or "")
        if not action_id or action_id in seen_actions:
            failed_rules.append("invalid_or_duplicate_action_id")
        seen_actions.add(action_id)
        claim_ids = action.get("claim_ids") or []
        if not isinstance(claim_ids, list) or not claim_ids:
            failed_rules.append(f"plan_action_without_claim:{action_id or index}")
        elif any(str(item) not in seen_claims for item in claim_ids):
            failed_rules.append(f"plan_action_with_unknown_claim:{action_id or index}")

    return {
        "status": "fail" if failed_rules else "warn" if warnings else "pass",
        "failed_rules": sorted(set(failed_rules)),
        "warnings": sorted(set(warnings)),
        "claim_ids": sorted(seen_claims),
        "action_ids": sorted(seen_actions),
    }


def _symbols(text: str, suffix: str) -> list[str]:
    patterns = []
    if suffix == ".py":
        patterns = [r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"]
    elif suffix in {".js", ".jsx", ".mjs", ".ts", ".tsx"}:
        patterns = [
            r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
        ]
    found: list[str] = []
    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match and match.group(1) not in found:
                found.append(match.group(1))
                if len(found) >= 40:
                    return found
    return found


def _imports(text: str, suffix: str) -> list[str]:
    patterns = []
    if suffix == ".py":
        patterns = [r"^\s*from\s+([A-Za-z0-9_.]+)\s+import", r"^\s*import\s+([A-Za-z0-9_.]+)"]
    elif suffix in {".js", ".jsx", ".mjs", ".ts", ".tsx"}:
        patterns = [r"\bfrom\s+[\"']([^\"']+)[\"']", r"\brequire\([\"']([^\"']+)[\"']\)"]
    found: list[str] = []
    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match and len(match.group(1)) <= 120 and match.group(1) not in found:
                found.append(match.group(1))
                if len(found) >= 40:
                    return found
    return found


def build_repository_map(
    repo_root: Path,
    authorized_roots: list[str],
    max_files: int = 200,
) -> dict:
    """Create navigation metadata for authorized roots without returning source text."""

    if not 1 <= max_files <= 500:
        raise ValueError("repository map max_files must be between 1 and 500")
    root = repo_root.resolve()
    normalized_roots = normalize_authorized_roots(authorized_roots, root)
    if not normalized_roots:
        raise ValueError("repository map requires at least one pre-authorized root")
    root_paths = [(root / value).resolve() for value in normalized_roots]
    files: list[dict] = []
    rejected_counts: dict[str, int] = {}
    frameworks: set[str] = set()
    exhausted = False
    for path in _iter_files(root_paths, root, set(DEFAULT_FILE_TYPES)):
        if len(files) >= max_files:
            exhausted = True
            break
        text, _, reason = _read_candidate(path, root, DEFAULT_MAX_FILE_BYTES)
        if reason:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        relative = _portable(path, root)
        suffix = path.suffix.lower()
        imports = _imports(text, suffix)
        name = path.name.lower()
        if name == "package.json":
            try:
                package = json.loads(text)
                dependencies = set((package.get("dependencies") or {})) | set((package.get("devDependencies") or {}))
                for candidate, label in {
                    "react": "React",
                    "vite": "Vite",
                    "@playwright/test": "Playwright",
                    "typescript": "TypeScript",
                }.items():
                    if candidate in dependencies:
                        frameworks.add(label)
            except json.JSONDecodeError:
                rejected_counts["invalid_package_json"] = rejected_counts.get("invalid_package_json", 0) + 1
        if suffix == ".py":
            frameworks.add("Python")
        if suffix in {".js", ".mjs"}:
            frameworks.add("Node.js")
        files.append(
            {
                "path": relative,
                "directory": path.parent.relative_to(root).as_posix(),
                "kind": "test" if _is_test_path(path) else "config" if _is_config_path(path) else "source",
                "size_bytes": path.stat().st_size,
                "symbols": _symbols(text, suffix),
                "imports": imports,
                "test_targets": [Path(value).stem for value in imports] if _is_test_path(path) else [],
            }
        )

    directories: dict[str, dict] = {}
    for item in files:
        directory = item["directory"]
        record = directories.setdefault(directory, {"path": directory, "file_count": 0, "source_files": 0, "test_files": 0, "config_files": 0})
        record["file_count"] += 1
        record[f"{item['kind']}_files"] += 1
    return {
        "schema": REPOSITORY_MAP_SCHEMA,
        "authorized_roots": normalized_roots,
        "ownership_boundaries": [{"root": value, "permission": "navigation_metadata_only"} for value in normalized_roots],
        "directories": sorted(directories.values(), key=lambda item: item["path"]),
        "files": files,
        "frameworks": sorted(frameworks),
        "ignored_directories": sorted(IGNORED_DIRECTORIES),
        "rejected_counts": rejected_counts,
        "budget": {"max_files": max_files},
        "budget_used": {"files": len(files)},
        "budget_exhausted": exhausted,
        "contains_source_content": False,
        "external_retrieval": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
