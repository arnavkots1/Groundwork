"""The one applier. Both the dry-run check and the human-confirmed Apply use it.

`plan_patch_operations` is the single source of truth for "does this diff apply".
The patch checker calls it read-only to reject an unappliable diff early; Apply calls
it again immediately before writing. Sharing it is deliberate - two appliers would
mean the thing that was checked is not the thing that runs.

Apply is the only function in the package that mutates the working tree. It refuses
blocked patches even with --yes, revalidates recorded context hashes immediately
before the write, and backs up every touched file first.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import artifacts, behavior, config, consent, jspace
from .util import allocate_applied_patch_id, dedupe, normalize_repo_file, now_iso, repo_path


FORBIDDEN_MARKERS = [".env", "providers.env", "secret", "api_key", "apikey", ".git/"]


def _source_root(source_root: Path | str | None = None) -> Path:
    """Resolve one explicit patch target root without changing global state."""

    root = Path(source_root) if source_root is not None else config.ROOT
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Patch source root is not a directory: {resolved}")
    return resolved


def _source_path(path_text: str, source_root: Path) -> Path:
    """Resolve a diff path beneath ``source_root`` and refuse traversal."""

    path = Path(path_text)
    if not path.is_absolute():
        path = source_root / path
    resolved = path.resolve()
    if resolved != source_root and source_root not in resolved.parents:
        raise RuntimeError(f"Refusing path outside source root: {resolved}")
    return resolved


def _relative_source_path(path_text: str, source_root: Path) -> str:
    return _source_path(path_text, source_root).relative_to(source_root).as_posix()


def _public_authorization(authorization: dict | None) -> dict:
    """Persist attribution fields without copying nonce/token material."""

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


def is_forbidden_patch_path(path_text: str) -> str | None:
    """Name the rule that forbids writing this path, or None if writing is allowed."""
    normalized = normalize_repo_file(path_text)
    lower = normalized.lower()
    if not lower:
        return "empty_path"
    behavior_reason = behavior.behavior_write_block_reason(normalized)
    if behavior_reason:
        return behavior_reason
    if any(marker in lower for marker in FORBIDDEN_MARKERS):
        return "secret_or_unsafe_path"
    if lower.startswith("brain/"):
        return "legacy_brain_write_blocked"
    if lower.startswith("brain_v2/employees/engineer/j_space/"):
        return "j_space_audit_state_blocked"
    if "canonical_behavior.md" in lower:
        return "canonical_behavior_blocked"
    if "approved_lessons.jsonl" in lower:
        return "approved_lessons_blocked"
    if lower.endswith("config/model-routing.yaml") or lower.endswith("brain/03_model_routing.md"):
        return "provider_routing_blocked"
    return None


def clean_diff_path(value: str) -> str:
    path = value.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def parse_unified_diff(unified_diff: str) -> list[dict]:
    """Strict unified-diff parser. Anything it does not understand raises."""
    lines = unified_diff.splitlines()
    patches: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            index += 1
            continue
        if not line.startswith("--- "):
            index += 1
            continue
        old_path = clean_diff_path(line[4:].strip())
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise RuntimeError("Malformed unified diff: missing +++ line.")
        new_path = clean_diff_path(lines[index][4:].strip())
        index += 1
        hunks: list[dict] = []
        while index < len(lines):
            hunk_line = lines[index]
            if hunk_line.startswith("diff --git ") or hunk_line.startswith("--- "):
                break
            if not hunk_line.startswith("@@ "):
                index += 1
                continue
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", hunk_line)
            if not match:
                raise RuntimeError(f"Unsupported hunk header: {hunk_line}")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            index += 1
            hunk_lines: list[str] = []
            while index < len(lines):
                body = lines[index]
                if body.startswith("@@ ") or body.startswith("diff --git ") or body.startswith("--- "):
                    break
                if body.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not body or body[0] not in {" ", "+", "-"}:
                    raise RuntimeError(f"Unsupported unified diff line: {body}")
                hunk_lines.append(body)
                index += 1
            hunks.append(
                {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": hunk_lines,
                }
            )
        patches.append({"old_path": old_path, "new_path": new_path, "hunks": hunks})
    if not patches:
        raise RuntimeError("Unified diff contains no file patches.")
    return patches


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _locate_hunk_start(file_lines: list[str], hunk_body: list[str]) -> int | None:
    """Find 1-based old_start by matching leading context/delete lines."""

    needle: list[str] = []
    for line in hunk_body:
        if not line or line[0] not in {" ", "-"}:
            if needle:
                break
            continue
        needle.append(line[1:])
        if len(needle) >= 3:
            break
    if not needle:
        return None
    for index in range(0, max(0, len(file_lines) - len(needle) + 1)):
        if file_lines[index : index + len(needle)] == needle:
            return index + 1
    for index, line in enumerate(file_lines):
        if line == needle[0]:
            return index + 1
    return None


def _locate_new_hunk_start(file_lines: list[str], hunk_body: list[str]) -> int | None:
    """Find 1-based new_start by matching leading context/insert lines."""

    needle: list[str] = []
    for line in hunk_body:
        if not line or line[0] not in {" ", "+"}:
            if needle:
                break
            continue
        needle.append(line[1:])
        if len(needle) >= 3:
            break
    if not needle:
        return None
    for index in range(0, max(0, len(file_lines) - len(needle) + 1)):
        if file_lines[index : index + len(needle)] == needle:
            return index + 1
    for index, line in enumerate(file_lines):
        if line == needle[0]:
            return index + 1
    return None


def _relocate_hunk_starts_in_unified_diff(unified_diff: str) -> tuple[str, list[str]]:
    """Re-anchor each hunk header by matching body lines against cumulative file text.

    Cursor/Composer often emit valid hunk bodies with wrong @@ line numbers. Relocate
    against the current on-disk file (and simulated post-hunk text) so dry-run Apply
    can succeed without changing delete/context lines.
    """

    text = str(unified_diff or "")
    if not text.strip():
        return text, []
    try:
        parsed = parse_unified_diff(text)
    except RuntimeError:
        return text, []
    out: list[str] = []
    notes: list[str] = []
    relocated_count = 0
    for file_patch in parsed:
        old_path = file_patch["old_path"]
        new_path = file_patch["new_path"]
        out.append(f"--- a/{old_path}" if not old_path.startswith("/") else f"--- {old_path}")
        out.append(f"+++ b/{new_path}" if not new_path.startswith("/") else f"+++ {new_path}")
        display = old_path if old_path != "/dev/null" else new_path
        file_text = ""
        if old_path != "/dev/null":
            try:
                source = repo_path(old_path)
                if source.is_file():
                    file_text = source.read_text(encoding="utf-8", errors="replace")
            except RuntimeError:
                file_text = ""
        old_cur = file_text
        new_cur = file_text
        for hunk in file_patch["hunks"]:
            hunk_body = hunk["lines"]
            old_count = sum(1 for item in hunk_body if item[0] in {" ", "-"})
            new_count = sum(1 for item in hunk_body if item[0] in {" ", "+"})
            old_start = _locate_hunk_start(old_cur.splitlines(), hunk_body)
            new_start = _locate_new_hunk_start(new_cur.splitlines(), hunk_body)
            if not old_start:
                notes.append(f"relocate_aborted:missing_old_start:{display}")
                return text, notes
            if not new_start:
                new_start = old_start
            declared_old = int(hunk.get("old_start") or 0)
            declared_new = int(hunk.get("new_start") or 0)
            if old_start != declared_old or new_start != declared_new:
                relocated_count += 1
                notes.append(f"relocated_start:{display}:{declared_old}->{old_start}:{declared_new}->{new_start}")
            out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@")
            out.extend(hunk_body)
            if old_path != "/dev/null":
                hunk_apply = {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": hunk_body,
                }
                try:
                    old_cur = apply_patch_to_text(old_cur, [hunk_apply], display)
                except RuntimeError as exc:
                    notes.append(f"relocate_aborted:apply_failed:{display}:{exc}")
                    return text, notes
                new_cur = old_cur
    repaired = "\n".join(out)
    if text.endswith("\n"):
        repaired += "\n"
    try:
        parse_unified_diff(repaired)
    except RuntimeError as exc:
        notes.append(f"relocate_failed_parse:{exc}")
        return text, notes
    if relocated_count:
        notes.insert(0, f"relocated_hunk_starts:{relocated_count}")
    return repaired, notes


def recompute_unified_diff_hunk_counts(unified_diff: str) -> tuple[str, list[str]]:
    """Recompute @@ old/new line counts from each hunk body.

    Composer/Cursor often emit a valid-looking body with wrong counts; ``git apply``
    then fails with ``patch fragment without header``. This is count-only repair —
    it does not invent missing context or relocate hunks.
    """

    text = str(unified_diff or "")
    if not text.strip():
        return text, []
    lines = text.splitlines()
    out: list[str] = []
    notes: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            out.append(line)
            index += 1
            continue
        index += 1
        hunk_body: list[str] = []
        while index < len(lines):
            body = lines[index]
            if body.startswith("@@") or body.startswith("diff --git ") or body.startswith("--- "):
                break
            if body.startswith("\\ No newline at end of file"):
                index += 1
                continue
            if not body or body[0] not in {" ", "+", "-"}:
                # Leave unsupported lines alone; caller/git apply decide.
                hunk_body.append(body)
                index += 1
                continue
            hunk_body.append(body)
            index += 1
        old_count = sum(1 for item in hunk_body if item and item[0] in {" ", "-"})
        new_count = sum(1 for item in hunk_body if item and item[0] in {" ", "+"})
        old_start = int(match.group(1))
        new_start = int(match.group(3))
        claimed_old = int(match.group(2) or "1")
        claimed_new = int(match.group(4) or "1")
        trail = line[match.end() :]
        if claimed_old != old_count or claimed_new != new_count:
            notes.append(f"recomputed_counts:{old_start}:{claimed_old}->{old_count}:{claimed_new}->{new_count}")
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{trail}")
        out.extend(hunk_body)
    repaired = "\n".join(out)
    if text.endswith("\n"):
        repaired += "\n"
    return repaired, notes


def repair_unified_diff_hunk_headers(unified_diff: str) -> tuple[str, list[str]]:
    """Recompute malformed @@ headers from the hunk body when the body is valid.

    Returns (diff, notes). Always recomputes wrong line counts first. If the
    original already parses after count repair, returns that. Otherwise attempts
    fuller header reconstruction. Callers must still dry-run and block unverified diffs.
    """

    text = str(unified_diff or "")
    if not text.strip():
        return text, []
    recounted, count_notes = recompute_unified_diff_hunk_counts(text)
    notes = list(count_notes)
    try:
        parse_unified_diff(recounted)
        if dry_run_patch_applicability(recounted)["ok"]:
            return recounted, notes
        relocated, relocate_notes = _relocate_hunk_starts_in_unified_diff(recounted)
        notes.extend(relocate_notes)
        if relocated != recounted and dry_run_patch_applicability(relocated)["ok"]:
            return relocated, notes
        text = relocated if relocated != recounted else recounted
    except RuntimeError:
        text = recounted

    lines = text.splitlines()
    out: list[str] = []
    index = 0
    repaired_count = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            out.append(line)
            index += 1
            continue
        out.append(line)
        old_path = clean_diff_path(line[4:].strip())
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            notes.append("repair_aborted:missing_plus_plus_line")
            return text, notes
        out.append(lines[index])
        index += 1
        display = old_path if old_path != "/dev/null" else clean_diff_path(out[-1][4:].strip())
        file_text = ""
        if old_path != "/dev/null":
            try:
                source = repo_path(old_path)
                if source.is_file():
                    file_text = source.read_text(encoding="utf-8", errors="replace")
            except RuntimeError:
                file_text = ""
        file_lines = file_text.splitlines()
        while index < len(lines):
            hunk_line = lines[index]
            if hunk_line.startswith("diff --git ") or hunk_line.startswith("--- "):
                break
            if not hunk_line.startswith("@@"):
                out.append(hunk_line)
                index += 1
                continue
            index += 1
            hunk_body: list[str] = []
            while index < len(lines):
                body = lines[index]
                if body.startswith("@@") or body.startswith("diff --git ") or body.startswith("--- "):
                    break
                if body.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not body or body[0] not in {" ", "+", "-"}:
                    notes.append(f"repair_aborted:unsupported_body_line:{body[:80]}")
                    return text, notes
                hunk_body.append(body)
                index += 1
            old_count = sum(1 for item in hunk_body if item[0] in {" ", "-"})
            new_count = sum(1 for item in hunk_body if item[0] in {" ", "+"})
            match = _HUNK_HEADER_RE.match(hunk_line)
            if match:
                old_start = int(match.group(1))
                new_start = int(match.group(3))
                if int(match.group(2) or "1") != old_count or int(match.group(4) or "1") != new_count:
                    repaired_count += 1
                    notes.append(f"recomputed_counts:{display}")
            else:
                old_start = _locate_hunk_start(file_lines, hunk_body) or 1
                new_start = old_start
                repaired_count += 1
                notes.append(f"recomputed_header:{display}:{hunk_line[:80]}")
            out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@")
            out.extend(hunk_body)
    repaired = "\n".join(out)
    if text.endswith("\n"):
        repaired += "\n"
    try:
        parse_unified_diff(repaired)
    except RuntimeError as exc:
        notes.append(f"repair_failed_parse:{exc}")
        return text, notes
    if repaired_count:
        notes.insert(0, f"repaired_hunk_headers:{repaired_count}")
    return repaired, notes


def apply_patch_to_text(original_text: str, hunks: list[dict], file_path: str) -> str:
    """Apply hunks in memory. Every context and delete line must match exactly."""
    lines = original_text.splitlines()
    delta = 0
    for hunk_index, hunk in enumerate(hunks, start=1):
        cursor = hunk["old_start"] - 1 + delta
        if cursor < 0:
            raise RuntimeError(f"Invalid hunk position for {file_path} (hunk {hunk_index}).")
        for line in hunk["lines"]:
            sign = line[0]
            content = line[1:]
            if sign == " ":
                if cursor >= len(lines) or lines[cursor] != content:
                    raise RuntimeError(
                        f"Context mismatch while applying patch to {file_path} (hunk {hunk_index})."
                    )
                cursor += 1
            elif sign == "-":
                if cursor >= len(lines) or lines[cursor] != content:
                    raise RuntimeError(
                        f"Delete mismatch while applying patch to {file_path} (hunk {hunk_index})."
                    )
                lines.pop(cursor)
                delta -= 1
            elif sign == "+":
                lines.insert(cursor, content)
                cursor += 1
                delta += 1
            else:
                raise RuntimeError(f"Unsupported hunk op in {file_path} (hunk {hunk_index}): {sign}")
    output = "\n".join(lines)
    if original_text.endswith("\n"):
        output += "\n"
    return output


def plan_patch_operations(
    unified_diff: str,
    *,
    source_root: Path | str | None = None,
) -> list[dict]:
    """Plan Apply operations using the exact same text applier Apply uses.

    Reads repository files as needed but never writes, creates backups, or mutates
    the working tree. Apply still re-validates hashes and writes separately.
    """

    root = _source_root(source_root)
    parsed_patches = parse_unified_diff(unified_diff)
    operations: list[dict] = []
    for file_patch in parsed_patches:
        old_path = file_patch["old_path"]
        new_path = file_patch["new_path"]
        display = new_path if old_path == "/dev/null" else old_path
        if old_path == "/dev/null":
            target = _source_path(new_path, root)
            if target.exists():
                raise RuntimeError(f"Patch creates a file that already exists: {new_path}")
            new_text = apply_patch_to_text("", file_patch["hunks"], display)
            operations.append({"op": "create", "path": target, "new_text": new_text, "display": display})
            continue
        source = _source_path(old_path, root)
        if not source.exists():
            raise RuntimeError(f"Cannot apply patch; file missing: {old_path}")
        original_text = source.read_text(encoding="utf-8", errors="replace")
        if new_path == "/dev/null":
            operations.append({"op": "delete", "path": source, "old_text": original_text, "display": display})
            continue
        target = _source_path(new_path, root)
        current_text = original_text
        for hunk_index, hunk in enumerate(file_patch["hunks"], start=1):
            located = _locate_hunk_start(current_text.splitlines(), hunk["lines"])
            hunk_apply = dict(hunk)
            if located:
                hunk_apply["old_start"] = located
            try:
                current_text = apply_patch_to_text(current_text, [hunk_apply], display)
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}") from exc
        new_text = current_text
        operations.append(
            {
                "op": "modify",
                "path": target,
                "old_text": original_text,
                "new_text": new_text,
                "display": display,
            }
        )
    return operations


def dry_run_patch_applicability(
    unified_diff: str,
    *,
    source_root: Path | str | None = None,
) -> dict:
    """Early filter: prove the diff applies in memory before Patch is marked proposed."""

    if not str(unified_diff or "").strip():
        return {"ok": True, "failed_rules": [], "warnings": []}
    try:
        plan_patch_operations(unified_diff, source_root=source_root)
    except RuntimeError as exc:
        message = str(exc)
        hunk_match = re.search(r"while applying patch to (.+?) \(hunk (\d+)\)", message)
        missing_match = re.search(r"file missing: (.+)$", message)
        exists_match = re.search(r"already exists: (.+)$", message)
        invalid_match = re.search(r"Invalid hunk position for (.+?) \(hunk (\d+)\)", message)
        if hunk_match:
            path_text = hunk_match.group(1).replace("\\", "/")
            rule = f"unappliable_hunk:{path_text}#hunk{hunk_match.group(2)}"
        elif invalid_match:
            path_text = invalid_match.group(1).replace("\\", "/")
            rule = f"unappliable_hunk:{path_text}#hunk{invalid_match.group(2)}"
        elif missing_match:
            rule = f"unappliable_hunk:{missing_match.group(1).replace(chr(92), '/')}#missing_file"
        elif exists_match:
            rule = f"unappliable_hunk:{exists_match.group(1).replace(chr(92), '/')}#create_exists"
        else:
            rule = "unappliable_hunk:parse_or_apply_failure"
        return {"ok": False, "failed_rules": [rule], "warnings": [message]}
    return {"ok": True, "failed_rules": [], "warnings": []}


def git_working_tree_clean(*, source_root: Path | str | None = None) -> bool:
    root = _source_root(source_root)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("git is not available but --require-clean-git was requested.")
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip() or result.stdout.strip()}")
    return not result.stdout.strip()


def engineer_apply_patch(
    patch_id: str,
    yes: bool = False,
    require_clean_git: bool = False,
    allow_package_json: bool = False,
    allow_delete: bool = False,
    override_failed_check: bool = False,
    allow_high_risk: bool = False,
    *,
    source_root: Path | str | None = None,
    authorization: dict | None = None,
    allowed_write_targets: list[str] | None = None,
) -> dict:
    """The single write gate. Raises rather than partially applying.

    The default target is the real repository and always requires ``yes=True``.
    A different target root is a disposable checkout and requires an explicit
    ``task_envelope_sandbox`` authorization plus an exact allowed-write target set.
    Both modes use the same parser, applicability planner, stale-hash check, backup
    path, and write implementation.
    """
    config.ensure_harness()
    root = _source_root(source_root)
    real_root = config.ROOT.resolve()
    real_apply = root == real_root
    public_authorization = _public_authorization(authorization)
    consent_attribution = consent.capture_consent_attribution() if real_apply else {}
    consent_event: dict = {}
    source_run_id = ""

    try:
        if real_apply and not yes:
            consent_event = consent.record_consent_event(
                "engineer_apply_patch",
                patch_id,
                consent_attribution,
                outcome="refused",
                detail="Real working-tree Apply requires explicit --yes.",
                metadata={"application_mode": "real_worktree"},
            )
            raise RuntimeError(
                "Real working-tree Apply requires explicit confirmation. "
                "Pass --yes to confirm."
            )
        if not real_apply:
            if str(public_authorization.get("kind") or "") != "task_envelope_sandbox":
                raise RuntimeError(
                    "Disposable-checkout Apply requires authorization.kind=task_envelope_sandbox."
                )
            if not allowed_write_targets:
                raise RuntimeError(
                    "Disposable-checkout Apply requires non-empty allowed_write_targets."
                )

        patch_json_path = artifacts.resolve_artifact_json(patch_id, config.ENGINEER_PATCHES_DIR)
        patch = json.loads(patch_json_path.read_text(encoding="utf-8", errors="replace"))
        unified_diff = (patch.get("unified_diff") or "").strip()
        if patch.get("patch_status") == "blocked" and not override_failed_check:
            raise RuntimeError(
                "Refusing blocked patch artifact. Pass --override-failed-check if you "
                "accept the risk after manual review."
            )
        if not unified_diff:
            raise RuntimeError("Refusing patch with empty unified_diff.")
        unified_diff, hunk_repair_notes = repair_unified_diff_hunk_headers(unified_diff)
        if require_clean_git and not git_working_tree_clean(source_root=root):
            raise RuntimeError("Working tree is not clean; refusing due to --require-clean-git.")

        source_run_id = str(patch.get("source_run_id") or "")
        applicability = dry_run_patch_applicability(unified_diff, source_root=root)
        stale_context = jspace.context_hash_mismatches(source_run_id, source_root=root)
        if stale_context and applicability.get("ok"):
            touched = {
                (fp["old_path"] if fp["old_path"] != "/dev/null" else fp["new_path"]).replace("\\", "/")
                for fp in parse_unified_diff(unified_diff)
            }
            jspace.refresh_selected_context_hashes(
                source_run_id,
                source_root=root,
                only_paths=touched,
            )
            stale_context = jspace.context_hash_mismatches(source_run_id, source_root=root)
        if stale_context:
            changed = ", ".join(str(item.get("path") or "") for item in stale_context[:5])
            raise RuntimeError(
                "Stale J-space context; refusing Apply before writes. "
                f"Re-retrieve and regenerate the patch: {changed}"
            )

        parsed_patches = parse_unified_diff(unified_diff)
        touched_files: list[str] = []
        blocked_reasons: list[str] = []
        for file_patch in parsed_patches:
            old_path = file_patch["old_path"]
            new_path = file_patch["new_path"]
            candidates = [item for item in [old_path, new_path] if item != "/dev/null"]
            for candidate in candidates:
                try:
                    normalized = _relative_source_path(candidate, root)
                except RuntimeError:
                    boundary = "repo" if real_apply else "source root"
                    blocked_reasons.append(f"diff path outside {boundary}: {candidate}")
                    continue
                touched_files.append(normalized)
                rule = is_forbidden_patch_path(normalized)
                if rule:
                    blocked_reasons.append(f"{normalized}: {rule}")
                lower = normalized.lower()
                if (
                    lower.endswith("package.json") or lower.endswith("package-lock.json")
                ) and not allow_package_json:
                    blocked_reasons.append(
                        f"{normalized}: package metadata change requires --allow-package-json"
                    )
            if new_path == "/dev/null" and not allow_delete:
                blocked_reasons.append("Patch includes deletion; pass --allow-delete.")
        if patch.get("risk_level") == "high" and not (
            allow_high_risk or allow_package_json or allow_delete
        ):
            blocked_reasons.append(
                "Patch risk_level=high requires --allow-high-risk or another explicit override flag."
            )

        allowed_targets: list[str] = []
        if allowed_write_targets is not None:
            for target in allowed_write_targets:
                try:
                    allowed_targets.append(_relative_source_path(str(target), root))
                except RuntimeError:
                    blocked_reasons.append(f"allowed write target outside source root: {target}")
            allowed_keys = {item.casefold() for item in allowed_targets}
            outside_allowed = [
                item for item in dedupe(touched_files) if item.casefold() not in allowed_keys
            ]
            if outside_allowed:
                blocked_reasons.append(
                    "Diff touches target(s) outside allowed_write_targets: "
                    + ", ".join(outside_allowed)
                )
        if blocked_reasons:
            raise RuntimeError("; ".join(dedupe(blocked_reasons)))

        # Same in-memory applier the patch checker dry-runs. Hash revalidation
        # happened above; writes begin only after this complete plan succeeds.
        operations = plan_patch_operations(unified_diff, source_root=root)
        operation_files = dedupe(
            [str(op["path"].relative_to(root).as_posix()) for op in operations]
        )
        if allowed_write_targets is not None:
            allowed_keys = {item.casefold() for item in allowed_targets}
            if any(item.casefold() not in allowed_keys for item in operation_files):
                raise RuntimeError(
                    "Planned operation escaped allowed_write_targets after applicability planning."
                )

        application_mode = "real_worktree" if real_apply else "disposable_checkout"
        status = "applied" if real_apply else "sandbox_applied"
        suffix = (
            "_engineer_applied_patch"
            if real_apply
            else "_engineer_sandbox_applied_patch"
        )
        applied_patch_id, backup_dir = allocate_applied_patch_id(
            suffix, config.ENGINEER_APPLIED_PATCHES_DIR
        )

        if real_apply:
            consent_event = consent.record_consent_event(
                "engineer_apply_patch",
                patch_id,
                consent_attribution,
                outcome="authorized",
                detail="Consent was attributed immediately before real working-tree writes.",
                metadata={
                    "application_mode": application_mode,
                    "changed_files": operation_files,
                },
            )

        for op in operations:
            path_obj: Path = op["path"]
            relative = path_obj.relative_to(root)
            backup_path = backup_dir / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(path_obj.read_bytes() if path_obj.exists() else b"")

        for op in operations:
            path_obj = op["path"]
            if op["op"] == "delete":
                if path_obj.exists():
                    path_obj.unlink()
            else:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.write_text(op["new_text"], encoding="utf-8")

        changed_files = operation_files
        file_operations = [
            {
                "path": str(op["path"].relative_to(root).as_posix()),
                "op": op["op"],
            }
            for op in operations
        ]
        payload = {
            "ok": True,
            "patch_id": patch.get("patch_id", patch_json_path.stem),
            "applied_patch_id": applied_patch_id,
            "source_run_id": source_run_id,
            "j_space": jspace.pointer(source_run_id) if source_run_id else {},
            "created_at": now_iso(),
            "source_patch_json": str(patch_json_path),
            "source_root": str(root),
            "source_root_authority": (
                "real_worktree" if real_apply else "task_envelope_sandbox"
            ),
            "application_mode": application_mode,
            "authorization": (
                {
                    "kind": "attributed_cli_consent",
                    "attribution": consent_attribution,
                    "consent_event": consent_event,
                }
                if real_apply
                else public_authorization
            ),
            "allowed_write_targets": dedupe(allowed_targets),
            "changed_files": changed_files,
            "file_operations": file_operations,
            "backup_dir": str(backup_dir),
            "hunk_repair_notes": hunk_repair_notes,
            "revertible": True,
            "keep_status": "pending_review",
            "verification_commands": patch.get("verification_commands") or [],
            "verification_commands_quality": patch.get(
                "verification_commands_quality", ""
            ),
            "verification_command_warnings": patch.get(
                "verification_command_warnings"
            )
            or [],
            "verification_commands_suggested_manual": patch.get(
                "verification_commands_suggested_manual"
            )
            or [],
            "status": status,
            "warnings": [],
            "safety_checks": {
                "require_clean_git": require_clean_git,
                "allow_package_json": allow_package_json,
                "allow_delete": allow_delete,
                "allow_high_risk": allow_high_risk,
                "override_failed_check": override_failed_check,
                "manual_yes_override": bool(yes) if real_apply else False,
                "manual_confirmation": bool(yes) if real_apply else False,
                "context_hashes_revalidated": True,
                "write_target_subset_validated": allowed_write_targets is not None,
                "application_mode": application_mode,
            },
        }
        md_path = config.ENGINEER_APPLIED_PATCHES_DIR / f"{applied_patch_id}.md"
        json_path = config.ENGINEER_APPLIED_PATCHES_DIR / f"{applied_patch_id}.json"
        # Both real and sandbox Apply artifacts live in durable real harness state.
        md_path.write_text(artifacts.applied_patch_markdown(payload), encoding="utf-8")
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        detail = (
            f"Consent-attributed real Apply changed {len(changed_files)} file(s)."
            if real_apply
            else f"Envelope-authorized sandbox Apply changed {len(changed_files)} file(s)."
        )
        j_space_fields = (
            jspace.update_manifest(
                source_run_id,
                "apply",
                "complete",
                status,
                detail,
                jspace.artifact_ref(
                    (
                        "engineer_applied_patch"
                        if real_apply
                        else "engineer_sandbox_applied_patch"
                    ),
                    applied_patch_id,
                    md_path,
                    json_path,
                ),
                {
                    "changed_files": changed_files,
                    "manual_confirmation": bool(yes) if real_apply else False,
                    "require_clean_git": require_clean_git,
                    "application_mode": application_mode,
                    "authorization": (
                        consent_attribution if real_apply else public_authorization
                    ),
                    "allowed_write_targets": dedupe(allowed_targets),
                },
            )
            if source_run_id
            else {}
        )
        payload["path"] = str(md_path)
        payload["jsonPath"] = str(json_path)
        payload.update(j_space_fields)
        payload["output"] = (
            f"Engineer {status} patch {applied_patch_id}: "
            f"{len(changed_files)} file(s) changed, backup at {backup_dir.name}. "
            "Like it: git commit. Don't: engineer-revert-applied-patch --yes."
        )
        return payload
    except Exception as exc:
        if real_apply and not consent_event:
            consent.record_consent_event(
                "engineer_apply_patch",
                patch_id,
                consent_attribution,
                outcome="refused",
                detail=str(exc),
                metadata={"application_mode": "real_worktree"},
            )
        raise


def engineer_revert_applied_patch(applied_patch_id: str, yes: bool = False) -> dict:
    """Restore the working tree from an Apply backup (git-like discard).

    Gate: ``yes=True`` required for real-repo revert. Idempotent refusal if already
    reverted.
    """
    config.ensure_harness()
    root = config.ROOT.resolve()
    if not yes:
        raise RuntimeError(
            "Revert requires explicit confirmation. Pass --yes to restore backup files."
        )

    json_path = artifacts.resolve_artifact_json(
        applied_patch_id, config.ENGINEER_APPLIED_PATCHES_DIR
    )
    payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    status = str(payload.get("status") or "")
    if status == "reverted":
        raise RuntimeError(f"Applied patch {applied_patch_id} was already reverted.")
    if status not in {"applied", "sandbox_applied"}:
        raise RuntimeError(f"Refusing revert for status={status!r}.")

    backup_dir = Path(str(payload.get("backup_dir") or ""))
    if not backup_dir.is_dir():
        raise RuntimeError(f"Backup directory missing: {backup_dir}")

    file_operations = payload.get("file_operations") or []
    restored: list[str] = []
    removed: list[str] = []

    if file_operations:
        for entry in file_operations:
            relative = str(entry.get("path") or "").replace("\\", "/")
            op = str(entry.get("op") or "modify")
            target = root / relative
            backup_path = backup_dir / relative
            if op == "create":
                if target.exists():
                    target.unlink()
                    removed.append(relative)
                continue
            if not backup_path.is_file():
                raise RuntimeError(f"Backup missing for {relative}: {backup_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(backup_path.read_bytes())
            restored.append(relative)
    else:
        for backup_path in backup_dir.rglob("*"):
            if not backup_path.is_file():
                continue
            relative = backup_path.relative_to(backup_dir).as_posix()
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(backup_path.read_bytes())
            restored.append(relative)

    payload["status"] = "reverted"
    payload["keep_status"] = "discarded"
    payload["reverted_at"] = now_iso()
    payload["revert_restored_files"] = dedupe(restored)
    payload["revert_removed_files"] = dedupe(removed)
    md_path = config.ENGINEER_APPLIED_PATCHES_DIR / f"{applied_patch_id}.md"
    md_path.write_text(artifacts.applied_patch_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    source_run_id = str(payload.get("source_run_id") or "")
    j_space_fields = (
        jspace.update_manifest(
            source_run_id,
            "apply",
            "complete",
            "reverted",
            f"Reverted applied patch {applied_patch_id}; restored {len(restored)} file(s).",
            jspace.artifact_ref(
                "engineer_applied_patch",
                applied_patch_id,
                md_path,
                json_path,
            ),
            {"restored_files": dedupe(restored), "removed_files": dedupe(removed)},
        )
        if source_run_id
        else {}
    )
    result = {
        "ok": True,
        "applied_patch_id": applied_patch_id,
        "status": "reverted",
        "restored_files": dedupe(restored),
        "removed_files": dedupe(removed),
        "backup_dir": str(backup_dir),
        "path": str(md_path),
        "jsonPath": str(json_path),
        "output": (
            f"Reverted {applied_patch_id}: restored {len(restored)} file(s)"
            + (f", removed {len(removed)} created file(s)" if removed else "")
            + "."
        ),
        **j_space_fields,
    }
    return result


def engineer_trial_patch(
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
    """Apply → verify → optionally revert. Git-like try before you commit.

    On verification failure with ``auto_revert_on_fail=True`` (default), restores the
    pre-Apply backup automatically. On success, changes stay in the working tree for
    you to ``git commit`` when satisfied.
    """
    apply_result = engineer_apply_patch(
        patch_id,
        yes=yes,
        require_clean_git=require_clean_git,
        allow_package_json=allow_package_json,
        allow_delete=allow_delete,
        override_failed_check=override_failed_check,
        allow_high_risk=allow_high_risk,
    )
    applied_patch_id = str(apply_result.get("applied_patch_id") or "")
    if not run_verify:
        return {
            "ok": True,
            "trial": True,
            "reverted": False,
            "applied_patch_id": applied_patch_id,
            "apply": apply_result,
            "output": (
                f"Trial apply complete ({applied_patch_id}). "
                "Run verify, then commit or revert."
            ),
        }

    from .verify import engineer_verify

    verify_result = engineer_verify(applied_patch_id, run=True, save=save_verify)
    passed = bool(verify_result.get("verification_passed"))
    reverted = False
    revert_result: dict | None = None
    if not passed and auto_revert_on_fail:
        revert_result = engineer_revert_applied_patch(applied_patch_id, yes=True)
        reverted = True

    return {
        "ok": passed and not reverted,
        "trial": True,
        "reverted": reverted,
        "verification_passed": passed,
        "applied_patch_id": applied_patch_id,
        "apply": apply_result,
        "verify": verify_result,
        "revert": revert_result,
        "output": (
            f"Trial kept changes ({applied_patch_id}); verification passed — git commit when ready."
            if passed and not reverted
            else (
                f"Trial reverted ({applied_patch_id}); verification failed."
                if reverted
                else f"Trial kept changes ({applied_patch_id}); verification failed — revert manually."
            )
        ),
    }
