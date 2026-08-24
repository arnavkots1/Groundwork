"""Load prompts from data files and substitute named placeholders literally.

Prompt files are UNTRUSTED DATA. Substitution is `str.replace` over an explicit
set of `{{name}}` tokens - never `eval`, never `str.format`, never an f-string over
file content. A file that is missing, or whose placeholder set does not exactly
match the supplied keys, raises rather than rendering a half-filled prompt.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config


PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


class PromptError(RuntimeError):
    """A prompt file is missing, empty, or its placeholders do not match the call."""


def prompt_path(name: str, prompts_dir: Path | None = None) -> Path:
    directory = prompts_dir or config.ENGINEER_PROMPTS_DIR
    return directory / f"{name}.md"


def load(name: str, prompts_dir: Path | None = None) -> str:
    path = prompt_path(name, prompts_dir)
    if not path.is_file():
        raise PromptError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptError(f"Prompt file is empty: {path}")
    return text


def placeholders(template: str) -> set[str]:
    return set(PLACEHOLDER_PATTERN.findall(template))


def render(name: str, values: dict[str, str], prompts_dir: Path | None = None) -> str:
    """Render a prompt file, requiring an exact placeholder/key match.

    Raises PromptError when the file is missing, when the template names a
    placeholder the caller did not supply, or when the caller supplies a key the
    template does not use. Both directions are errors because either one means the
    prompt and the code have drifted apart, and a silently half-substituted prompt
    is exactly the kind of quiet degradation this harness must not produce.
    """
    template = load(name, prompts_dir)
    found = placeholders(template)
    supplied = set(values)
    missing = sorted(found - supplied)
    unused = sorted(supplied - found)
    if missing or unused:
        detail = []
        if missing:
            detail.append(f"template placeholders with no supplied value: {', '.join(missing)}")
        if unused:
            detail.append(f"supplied values with no template placeholder: {', '.join(unused)}")
        raise PromptError(f"Prompt '{name}' placeholder mismatch; " + "; ".join(detail))

    # One left-to-right pass over the TEMPLATE. Substituted values are never
    # rescanned, so untrusted repository content that happens to contain `{{name}}`
    # is emitted literally instead of being re-substituted or raising. Validating
    # the template up front is what guarantees no placeholder is left behind.
    rendered = PLACEHOLDER_PATTERN.sub(lambda match: str(values[match.group(1)]), template)
    return rendered.strip("\n")


def audit(prompts_dir: Path | None = None) -> dict:
    """Report every prompt file and the placeholders it declares."""
    directory = prompts_dir or config.ENGINEER_PROMPTS_DIR
    entries = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        entries.append(
            {
                "name": path.stem,
                "path": path.relative_to(config.ROOT).as_posix(),
                "placeholders": sorted(placeholders(text)),
                "chars": len(text),
            }
        )
    return {"prompts_dir": directory.relative_to(config.ROOT).as_posix(), "prompts": entries}
