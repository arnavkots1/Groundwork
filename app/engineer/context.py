"""Human-selected file context: denial, excerpting, and cheap static inspection.

The write scope of the whole harness is derived from what a human selected here.
Retrieval and model-proposed files are read-only context; they never widen this set.
Secret denial happens twice - once on the path before any read, once on the content -
so a denied file is neither opened nor echoed into an artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

from repository_retrieval import MAX_TOTAL_CHARS_HARD, secret_content_block_reason

from . import config
from .util import read_text, repo_path, selected_context_block_reason


EXCERPT_CHAR_LIMIT = 3500
# Same hard ceiling as repository retrieval so --max-chars-per-call is one knob.
EXCERPT_CHAR_HARD_LIMIT = MAX_TOTAL_CHARS_HARD


def resolve_excerpt_char_limit(max_chars_per_call: int | None = None) -> int:
    """Default selected-file excerpt size, or an explicit CLI override within hard bounds."""
    if max_chars_per_call is None:
        return EXCERPT_CHAR_LIMIT
    try:
        value = int(max_chars_per_call)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("--max-chars-per-call must be an integer.") from exc
    if not 1 <= value <= EXCERPT_CHAR_HARD_LIMIT:
        raise RuntimeError(
            f"--max-chars-per-call must be between 1 and {EXCERPT_CHAR_HARD_LIMIT}."
        )
    return value


def selected_context(
    selected_files: list[str],
    task: str = "",
    *,
    max_chars_per_call: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Resolve, deny-check, read, and excerpt every human-selected file.

    Raises on a denied path, a non-file, or secret-looking content. Returns
    (records, missing_paths); a caller treats a non-empty `missing` as a hard stop.
    """
    excerpt_limit = resolve_excerpt_char_limit(max_chars_per_call)
    selected = []
    missing = []
    for item in selected_files:
        if not item:
            continue
        resolved = repo_path(item)
        record = {
            "input": item,
            "path": str(resolved),
            "exists": resolved.exists(),
            "legacy_brain": config.ROOT.resolve() / "brain" in resolved.parents,
        }
        if resolved.exists():
            blocked_reason = selected_context_block_reason(resolved)
            if blocked_reason:
                raise RuntimeError(f"Hard stop: selected context denied for {item}: {blocked_reason}.")
            if not resolved.is_file():
                raise RuntimeError(f"Hard stop: selected context must be a file: {resolved}")
            content = read_text(resolved)
            content_block_reason = secret_content_block_reason(content)
            if content_block_reason:
                raise RuntimeError(
                    f"Hard stop: selected context denied for {item}: {content_block_reason}."
                )
            record.update(inspect_selected_file(resolved, content, task))
            excerpt, truncated, start_line, end_line = task_relevant_excerpt(
                content, task, limit=excerpt_limit
            )
            record["content"] = excerpt
            record["content_truncated"] = truncated
            record["excerpt_start_line"] = start_line
            record["excerpt_end_line"] = end_line
            record["excerpt_char_limit"] = excerpt_limit
            selected.append(record)
        else:
            missing.append(str(resolved))
            selected.append(record)
    return selected, missing


def task_relevant_excerpt(content: str, task: str, limit: int = EXCERPT_CHAR_LIMIT) -> tuple[str, bool, int, int]:
    """Prefer a task-relevant window over the file head when selected files must be truncated.

    Evidence for a change should include the symbols the task names, not only the first N chars.
    """

    lines = content.splitlines(keepends=True)
    total_lines = len(lines) if lines else 1
    if len(content) <= limit:
        return content, False, 1, total_lines

    plain_lines = [line.rstrip("\r\n") for line in lines]
    matches = task_matches(plain_lines, task)
    if not matches:
        excerpt = content[:limit]
        end_line = excerpt.count("\n") + (0 if excerpt.endswith("\n") else 1)
        return excerpt, True, 1, max(1, end_line)

    # Prefer definition-like matches for task terms (def/class/function/const).
    ranked: list[tuple[int, int]] = []
    for match in matches:
        line_no = int(match.get("line") or 0)
        text = str(match.get("text") or "")
        score = len(match.get("terms") or [])
        if re.match(r"^\s*(?:async\s+)?(?:def|class|function|const|let|var|interface|type|enum)\b", text):
            score += 10
        ranked.append((score, line_no))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    anchor = ranked[0][1]
    # Grow a char-bounded window around the anchor line.
    start_idx = max(0, anchor - 1)
    end_idx = start_idx
    char_count = len(lines[start_idx]) if start_idx < len(lines) else 0
    while end_idx + 1 < len(lines) and char_count + len(lines[end_idx + 1]) <= limit:
        end_idx += 1
        char_count += len(lines[end_idx])
    while start_idx > 0 and char_count + len(lines[start_idx - 1]) <= limit:
        start_idx -= 1
        char_count += len(lines[start_idx])
    excerpt = "".join(lines[start_idx : end_idx + 1])
    if len(excerpt) > limit:
        excerpt = excerpt[:limit]
    return excerpt, True, start_idx + 1, end_idx + 1


def inspect_selected_file(path: Path, content: str, task: str = "") -> dict:
    lines = content.splitlines()
    return {
        "file_size": path.stat().st_size if path.exists() else 0,
        "content_preview": "\n".join(lines[:40])[:1800],
        "extracted_symbols": extract_symbols(path, lines),
        "detected_technologies": detect_technologies(path, content),
        "relevant_matches": task_matches(lines, task),
    }


def extract_symbols(path: Path, lines: list[str]) -> list[str]:
    suffix = path.suffix.lower()
    patterns = []
    if suffix == ".py":
        patterns = [r"^\s*def\s+([A-Za-z_]\w*)", r"^\s*class\s+([A-Za-z_]\w*)"]
    elif suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs"}:
        patterns = [
            r"^\s*(?:export\s+)?function\s+([A-Za-z_]\w*)",
            r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)",
            r"^\s*type\s+([A-Za-z_]\w*)",
            r"^\s*interface\s+([A-Za-z_]\w*)",
        ]
    elif suffix in {".css", ".scss"}:
        patterns = [r"^\s*([.#][A-Za-z0-9_-]+)\s*[,{]"]
    symbols = []
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                symbols.append(match.group(1))
                break
        if len(symbols) >= 40:
            break
    return symbols


def detect_technologies(path: Path, content: str) -> list[str]:
    suffix = path.suffix.lower()
    tech = []
    markers = {
        "python": suffix == ".py",
        "typescript": suffix in {".ts", ".tsx"},
        "react": suffix in {".tsx", ".jsx"} or "react" in content.lower(),
        "javascript": suffix in {".js", ".jsx", ".mjs"},
        "css": suffix in {".css", ".scss"},
        "vite": "vite" in content.lower(),
        "lucide-react": "lucide-react" in content,
    }
    for name, present in markers.items():
        if present:
            tech.append(name)
    return tech


def task_matches(lines: list[str], task: str) -> list[dict]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "into", "without", "change", "changes", "add"}
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", task) if term.lower() not in stop]
    matches = []
    seen = set()
    for index, line in enumerate(lines, start=1):
        lower = line.lower()
        matched = [term for term in terms if term in lower]
        if matched:
            key = (index, line.strip())
            if key in seen:
                continue
            seen.add(key)
            matches.append({"line": index, "terms": matched[:6], "text": line.strip()[:240]})
        if len(matches) >= 20:
            break
    return matches


def selected_file_inspections(selected_records: list[dict]) -> list[dict]:
    return [
        {
            "path": item.get("path"),
            "exists": item.get("exists", False),
            "file_size": item.get("file_size", 0),
            "content_preview": item.get("content_preview", ""),
            "extracted_symbols": item.get("extracted_symbols", []),
            "detected_technologies": item.get("detected_technologies", []),
            "relevant_matches": item.get("relevant_matches", []),
        }
        for item in selected_records
    ]


def records_from_saved_payload(payload: dict) -> list[dict]:
    """Rebuild selected-file records from a saved plan so a re-check reads current disk state."""
    records = []
    task = payload.get("task", "")
    excerpt_limit = resolve_excerpt_char_limit(
        payload.get("selected_file_excerpt_chars_per_file")
        if payload.get("selected_file_excerpt_chars_per_file") is not None
        else ((payload.get("context_manifest") or {}).get("selected_file_excerpt_chars_per_file"))
    )
    selected = ((payload.get("context_manifest") or {}).get("selected_files") or [])
    for item in selected:
        input_path = item.get("input") or item.get("path") or ""
        if not input_path:
            continue
        resolved = repo_path(input_path)
        record = {
            "input": input_path,
            "path": str(resolved),
            "exists": resolved.exists(),
            "legacy_brain": config.ROOT.resolve() / "brain" in resolved.parents,
        }
        if resolved.exists():
            content = read_text(resolved)
            record.update(inspect_selected_file(resolved, content, task))
            excerpt, truncated, start_line, end_line = task_relevant_excerpt(
                content, task, limit=excerpt_limit
            )
            record["content"] = excerpt
            record["content_truncated"] = truncated
            record["excerpt_start_line"] = start_line
            record["excerpt_end_line"] = end_line
            record["excerpt_char_limit"] = excerpt_limit
        records.append(record)
    return records
