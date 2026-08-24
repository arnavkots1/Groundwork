"""Repo-path safety, file IO, and permissive-but-bounded model output parsing.

The JSON extraction here is the model-agnostic seam: it accepts fenced blocks,
prose-wrapped objects, and trailing commentary, but it only ever returns a plain
dict. Nothing downstream trusts the shape; the checker validates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import config

# Acceptance suites freeze the stamp clock via this env var so same-second
# collisions are deterministic. Empty / unset means wall clock.
STAMP_NOW_ENV = "COMPANYBRAIN_TEST_NOW"


def repo_path(path_text: str) -> Path:
    """Resolve a path and refuse anything outside the repository root."""
    path = Path(path_text)
    if not path.is_absolute():
        path = config.ROOT / path
    resolved = path.resolve()
    root_resolved = config.ROOT.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise RuntimeError(f"Refusing path outside repo: {resolved}")
    return resolved


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line, "parse_error": True})
    return rows


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def portable_repo_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def normalize_repo_file(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        resolved = repo_path(path_text)
        return resolved.relative_to(config.ROOT).as_posix().lower()
    except (RuntimeError, ValueError):
        return path_text.replace("\\", "/").strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_context_block_reason(path: Path) -> str:
    """Path-shaped secret denial. Runs before any read, so a denied file is never opened."""
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if name == ".env" or name.startswith(".env.") or name == "providers.env":
        return "environment files are excluded from Engineer context"
    if path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}:
        return "key and certificate files are excluded from Engineer context"
    if {"secret", "secrets", ".secrets"} & parts or any(
        marker in name for marker in ["secret", "credential", "api_key", "apikey"]
    ):
        return "secret or credential-like files are excluded from Engineer context"
    return ""


def file_provenance(path: Path, source_type: str, requested_path: str = "") -> dict:
    stat = path.stat()
    return {
        "source_type": source_type,
        "requested_path": requested_path or portable_repo_path(path),
        "path": portable_repo_path(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "content_stored_in_j_space": False,
    }


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated for synthesis]"


def optional_bool(value: str | bool | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def extract_json_object(text: str) -> dict:
    """Parse permissively; return {} rather than guessing a shape.

    An empty dict is a real signal downstream ("structured parse failed"), which is
    why this never fabricates fields to make output look well-formed.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        for candidate in balanced_json_candidates(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {}


def balanced_json_candidates(text: str) -> list[str]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return sorted(candidates, key=len, reverse=True)


def safe_workspace_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError(f"Invalid J-space workspace id: {value!r}")
    return value


def stamp_now() -> datetime:
    """Wall clock, or COMPANYBRAIN_TEST_NOW when set (acceptance freeze seam)."""
    override = os.environ.get(STAMP_NOW_ENV, "").strip()
    if override:
        return datetime.fromisoformat(override)
    return datetime.now()


def stamp_prefix() -> str:
    return stamp_now().strftime("%Y-%m-%d_%H%M%S")


def stamp(suffix: str) -> str:
    """Second-resolution id with no uniqueness check.

    Prefer `unique_artifact_id` or `allocate_applied_patch_id` for durable
    artifacts. Kept for callers that only need a readable timestamp label.
    """
    return stamp_prefix() + suffix


def unique_artifact_id(
    suffix: str,
    directory: Path,
    *,
    is_free: Callable[[str], bool] | None = None,
) -> str:
    """Collision-free artifact id: clean timestamp, uuid-suffixed if taken.

    Uniqueness is decided against *what already exists on disk in directory*,
    not an in-process counter. The suffix stays at the end so globs like
    `*_engineer_patch.md` keep matching (`{stamp}_{uuid8}_engineer_patch`).
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = stamp_prefix()

    def default_free(candidate: str) -> bool:
        return not (directory / f"{candidate}.json").exists() and not (
            directory / f"{candidate}.md"
        ).exists()

    check = is_free or default_free
    candidates = [
        f"{base}{suffix}",
        f"{base}_{uuid.uuid4().hex[:8]}{suffix}",
    ]
    for candidate in candidates:
        if check(candidate):
            return candidate
    candidate = f"{base}_{uuid.uuid4().hex}{suffix}"
    if check(candidate):
        return candidate
    raise RuntimeError(f"Unable to allocate unique artifact id with suffix {suffix!r}")


def allocate_applied_patch_id(suffix: str, directory: Path) -> tuple[str, Path]:
    """Allocate an applied-patch id and reserve its backup directory together.

    An id is free only if both the record paths (`.json`/`.md`) and the
    `_backup` directory are free. Reservation uses `mkdir(exist_ok=False)` on
    the backup directory so a concurrent same-second Apply cannot share it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = stamp_prefix()

    def try_reserve(candidate: str) -> Path | None:
        json_path = directory / f"{candidate}.json"
        md_path = directory / f"{candidate}.md"
        backup_dir = directory / f"{candidate}_backup"
        if json_path.exists() or md_path.exists() or backup_dir.exists():
            return None
        try:
            backup_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            return None
        return backup_dir

    candidates = [
        f"{base}{suffix}",
        f"{base}_{uuid.uuid4().hex[:8]}{suffix}",
    ]
    for candidate in candidates:
        reserved = try_reserve(candidate)
        if reserved is not None:
            return candidate, reserved
    for _ in range(8):
        candidate = f"{base}_{uuid.uuid4().hex}{suffix}"
        reserved = try_reserve(candidate)
        if reserved is not None:
            return candidate, reserved
    raise RuntimeError(
        f"Unable to allocate unique applied patch id with suffix {suffix!r}"
    )


def now_iso() -> str:
    return stamp_now().isoformat(timespec="seconds")


def now_tz_iso() -> str:
    return stamp_now().astimezone().isoformat(timespec="seconds")
