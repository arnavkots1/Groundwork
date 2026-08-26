"""Harness-on vs harness-off A/B benchmark.

This is the decisive experiment for CompanyBrain's product claim:

    the same lead model, given the same task, produces better and cheaper-to-review
    work through the harness than it does from a bare single call.

Arm A (baseline / harness-off): one call to the lead model with the task and the raw
contents of the selected files. No retrieval, no deterministic checker, no grounding,
no replan. This is "what you get if you just paste files into Claude or GPT".

Arm B (harness-on): the full Engineer pipeline - plan, bounded context request,
retrieval execution, grounded replan, deterministic check, patch proposal.

Both arms are scored on the same objective checks. Neither arm applies a patch, runs
a verification command, promotes a lesson, or touches the working tree.

Execution is opt-in twice: --run to do any work at all, and
--allow-private-source-export to acknowledge that bounded private repository
excerpts are sent to the configured lead-model provider.

    python scripts/harness_ab_benchmark.py --run --allow-private-source-export
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT / "scripts", ROOT / "app"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import company_brain_action as action  # noqa: E402

EVAL_DIR = ROOT / "brain_v2" / "evals" / "engineer"
SANDBOX_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".tmp",
    "dist",
    "build",
    "coverage",
    "*.pyc",
    "brain_v2",
    "web-gui",
)

# Per-million-token prices in USD. NVIDIA NIM free tier is 0 by definition.
# Update these when provider pricing changes; they only affect the cost column.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "Anthropic API": (3.0, 15.0),
    "OpenAI API": (2.5, 10.0),
    "Gemini": (0.0, 0.0),
    "NVIDIA NIM": (0.0, 0.0),
    "Kimi / Moonshot": (0.0, 0.0),
}

# Lowered from 60k. The baseline arm pastes whole files, so this is the single
# biggest driver of usage. 20k is still far more context than these tasks need.
BASELINE_MAX_FILE_CHARS = 20_000

BASELINE_SYSTEM = (
    "You are a senior software engineer. You will be given a task and the full contents of "
    "one or more repository files. Produce a short plan, then a unified diff patch limited "
    "strictly to the files you were given. Do not invent file paths. End with a single "
    "verification command that would prove the change works.\n\n"
    "Return the patch inside a fenced block labelled diff."
)

# CHEAP is the default set: small real files with real callers and tests, chosen so a
# full sweep costs a handful of short calls instead of a session's worth of quota.
# The FULL set targets bigger files (company_brain_action.py alone is ~247k chars);
# use it only when you have budget to spare.
#
# HARD tasks create the retrieval asymmetry the product claims: both arms receive the
# same single selected file, but a correct edit requires a fact that lives outside that
# file and is reachable via find_references / find_tests / find_symbol. hidden_fact is
# audit metadata for humans and dry-run reports; it is never included in model prompts.
CHEAP_TASKS = [
    {
        "id": "rate_limiter_slot",
        "task": (
            "Trace acquire_slot in the rate limiter through its callers. If a caller can "
            "acquire a slot without recording success or failure, close that gap."
        ),
        "symbol": "acquire_slot",
        "selected": ["app/rate_limiter.py"],
        "roots": ["app"],
    },
    {
        "id": "catalog_model_ranking",
        "task": (
            "Trace smartest_chat_models and tighten its handling of an empty or malformed "
            "catalog so it degrades to an explicit empty result instead of a partial one."
        ),
        "symbol": "smartest_chat_models",
        "selected": ["app/nvidia_catalog.py"],
        "roots": ["app"],
    },
    {
        "id": "env_loader_precedence",
        "task": (
            "Trace load_local_env and make the precedence between an already-set process "
            "environment variable and a file-provided value explicit and documented."
        ),
        "symbol": "load_local_env",
        "selected": ["app/env_loader.py"],
        "roots": ["app"],
    },
]

HARD_TASKS = [
    {
        "id": "rate_limiter_slot_context",
        # Hidden fact: app/api_clients.py catches RateLimitError from acquire_slot and
        # re-raises without calling record_failure; record_failure is only used after a
        # slot was acquired and the provider call failed. A context manager that records
        # failure whenever acquire_slot raises RateLimitError invents cooldown pressure
        # the live callers deliberately avoid.
        "hidden_fact": (
            "api_clients.call_provider: RateLimitError from acquire_slot is re-raised as "
            "ProviderError without record_failure; record_failure runs only after acquire."
        ),
        "task": (
            "Add a context-manager helper around rate-limit slot acquisition that records "
            "the terminal outcome for a wrapped call. Preserve acquire_slot's existing "
            "signature and do not change how acquisition-time rate limits are surfaced."
        ),
        "symbol": "acquire_slot",
        "selected": ["app/rate_limiter.py"],
        "roots": ["app"],
        "behavior_test": {
            "command": [
                sys.executable,
                "-c",
                "import ast, sys\n"
                "from pathlib import Path\n"
                "from contextlib import contextmanager\n"
                "from unittest import mock\n"
                "src = Path('app/rate_limiter.py').read_text(encoding='utf-8')\n"
                "tree = ast.parse(src)\n"
                "helpers = []\n"
                "for node in tree.body:\n"
                "    if not isinstance(node, ast.FunctionDef):\n"
                "        continue\n"
                "    for dec in node.decorator_list:\n"
                "        name = ''\n"
                "        if isinstance(dec, ast.Name): name = dec.id\n"
                "        elif isinstance(dec, ast.Attribute): name = dec.attr\n"
                "        if name == 'contextmanager':\n"
                "            helpers.append(node.name)\n"
                "if not helpers:\n"
                "    raise SystemExit(1)\n"
                "sys.path.insert(0, 'app')\n"
                "import rate_limiter\n"
                "helper = getattr(rate_limiter, helpers[0])\n"
                "acquire_calls = []\n"
                "success_calls = []\n"
                "failure_calls = []\n"
                "def fake_acquire(provider, model):\n"
                "    acquire_calls.append((provider, model))\n"
                "    raise rate_limiter.RateLimitError('cooling down', retry_after=1)\n"
                "with mock.patch.object(rate_limiter, 'acquire_slot', fake_acquire), \\\n"
                "     mock.patch.object(rate_limiter, 'record_success', lambda *a, **k: success_calls.append(a)), \\\n"
                "     mock.patch.object(rate_limiter, 'record_failure', lambda *a, **k: failure_calls.append(a)):\n"
                "    try:\n"
                "        with helper('NVIDIA NIM', 'demo-model'):\n"
                "            pass\n"
                "    except rate_limiter.RateLimitError:\n"
                "        pass\n"
                "if not acquire_calls or failure_calls or success_calls:\n"
                "    raise SystemExit(1)\n"
                "raise SystemExit(0)\n",
            ]
        },
    },
    {
        "id": "grounding_reject_duplicate_evidence",
        # Hidden fact: scripts/engineer_acceptance.py::_stage3a_grounding_contract_case
        # requires high_risk_context_question_unresolved and bare-string context_incomplete
        # to keep failing. A rewrite that only adds duplicate detection and drops those
        # rules fails the acceptance contract reachable via find_tests on validate_grounded_plan.
        "hidden_fact": (
            "engineer_acceptance stage3a_grounding_contract still requires "
            "high_risk_context_question_unresolved and bare-string context_incomplete."
        ),
        "task": (
            "Reject material claims that cite the same evidence id more than once inside one "
            "claim. Keep every existing grounding failure mode intact."
        ),
        "symbol": "validate_grounded_plan",
        "selected": ["app/engineer_grounding.py"],
        "roots": ["app", "scripts"],
        "behavior_test": {
            "command": [
                sys.executable,
                "-c",
                "import sys\n"
                "sys.path.insert(0, 'app')\n"
                "from engineer_grounding import validate_grounded_plan\n"
                "evidence = {'e1'}\n"
                "dup = {\n"
                "  'context_sufficiency': {'status': 'sufficient', 'known_unknowns': [], "
                "'unresolved_questions': [], 'assumptions': []},\n"
                "  'material_claims': [{'claim_id': 'c1', 'claim': 'x', 'evidence_ids': ['e1', 'e1'], "
                "'confidence': 'high', 'influences': ['scope']}],\n"
                "  'plan_actions': [{'action_id': 'a1', 'action': 'y', 'claim_ids': ['c1'], "
                "'files': ['README.md']}],\n"
                "}\n"
                "dup_rules = validate_grounded_plan(dup, evidence).get('failed_rules') or []\n"
                "dup_ok = any('duplicate' in str(rule).lower() for rule in dup_rules)\n"
                "risky = {\n"
                "  'context_sufficiency': {'status': 'incomplete', 'known_unknowns': [], "
                "'unresolved_questions': [{'question': 'q', 'risk': 'high'}], 'assumptions': []},\n"
                "  'material_claims': [{'claim_id': 'c1', 'claim': 'x', 'evidence_ids': ['e1'], "
                "'confidence': 'high', 'influences': ['scope']}],\n"
                "  'plan_actions': [{'action_id': 'a1', 'action': 'y', 'claim_ids': ['c1'], "
                "'files': ['README.md']}],\n"
                "}\n"
                "risky_rules = validate_grounded_plan(risky, evidence).get('failed_rules') or []\n"
                "preserve = 'high_risk_context_question_unresolved' in risky_rules\n"
                "raise SystemExit(0 if dup_ok and preserve else 1)\n",
            ]
        },
    },
    {
        "id": "resilient_no_empty_refill",
        # Hidden fact: call_nvidia_with_fallback currently appends available_models_for_role
        # when smartest_chat_models returns fewer than limit. Those helpers live in
        # nvidia_catalog.py (outside the selected file). Fail-closed empty ranking must not
        # silently refill from availability.
        "hidden_fact": (
            "resilient_calls.call_nvidia_with_fallback refills from "
            "nvidia_catalog.available_models_for_role when smartest_chat_models is short."
        ),
        "task": (
            "When the preferred ranked chat fleet is empty, fail closed instead of inventing "
            "substitute chat candidates before attempting provider calls."
        ),
        "symbol": "call_nvidia_with_fallback",
        "selected": ["app/resilient_calls.py"],
        "roots": ["app"],
        "behavior_test": {
            "command": [
                sys.executable,
                "-c",
                "import sys\n"
                "sys.path.insert(0, 'app')\n"
                "import resilient_calls as rc\n"
                "from api_clients import ProviderError\n"
                "available_calls = []\n"
                "rc.smartest_chat_models = lambda role=None, limit=10: []\n"
                "def fake_available(role):\n"
                "    available_calls.append(role)\n"
                "    return [{'id': 'fake/model', 'endpoint_type': 'chat'}]\n"
                "rc.available_models_for_role = fake_available\n"
                "rc.call_provider = lambda *a, **k: (_ for _ in ()).throw("
                "ProviderError('blocked', retryable=True))\n"
                "try:\n"
                "    rc.call_nvidia_with_fallback('CEO', 'ping', limit=3)\n"
                "except Exception:\n"
                "    pass\n"
                "raise SystemExit(0 if not available_calls else 1)\n",
            ]
        },
    },
    {
        "id": "env_loader_empty_process_key",
        # Hidden fact: app/api_clients.py calls load_local_env() at import, then reads
        # provider API keys from os.environ. An empty-string process value must keep
        # winning over file values; callers treat empty as "unset credential" rather than
        # a missing key to fill from .env.
        "hidden_fact": (
            "api_clients imports load_local_env before reading provider env keys; empty "
            "process values must not be overwritten by file-provided secrets."
        ),
        "task": (
            "Make load_local_env treat an already-present process environment key as "
            "authoritative even when its value is an empty string, and document that rule."
        ),
        "symbol": "load_local_env",
        "selected": ["app/env_loader.py"],
        "roots": ["app"],
        "behavior_test": {
            "command": [
                sys.executable,
                "-c",
                "import os, sys, tempfile\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, 'app')\n"
                "import env_loader\n"
                "key = 'COMPANYBRAIN_AB_EMPTY_KEY'\n"
                "os.environ[key] = ''\n"
                "td = tempfile.TemporaryDirectory()\n"
                "root = Path(td.name)\n"
                "cfg = root / 'config'\n"
                "cfg.mkdir()\n"
                "(cfg / 'providers.env').write_text(f'{key}=from-file\\n', encoding='utf-8')\n"
                "env_loader.ROOT = root\n"
                "env_loader.ENV_PATHS = [cfg / 'providers.env', root / '.env']\n"
                "env_loader.load_local_env()\n"
                "ok_behavior = os.environ.get(key) == ''\n"
                "doc = Path('app/env_loader.py').read_text(encoding='utf-8').lower()\n"
                "ok_doc = 'empty' in doc and ('process' in doc or 'already' in doc)\n"
                "td.cleanup()\n"
                "raise SystemExit(0 if ok_behavior and ok_doc else 1)\n",
            ]
        },
    },
]

FULL_TASKS = [
    {
        "id": "api_argument_routing",
        "task": (
            "Trace how buildArgs routes Engineer context actions from the API bridge, including its tests. "
            "Identify any contract mismatch and fix it."
        ),
        "symbol": "buildArgs",
        "selected": ["web-gui/server/company-brain-api.mjs", "web-gui/server/company-brain-api.test.mjs"],
        "roots": ["web-gui/server", "web-gui/src", "web-gui/tests"],
    },
    {
        "id": "retrieval_broker",
        "task": (
            "Trace run_repository_retrieval through its callers and acceptance tests. Tighten any root, "
            "intent, budget, or secret-rejection boundary that is weaker than the others."
        ),
        "symbol": "run_repository_retrieval",
        "selected": ["app/repository_retrieval.py", "scripts/engineer_acceptance.py"],
        "roots": ["app", "scripts"],
    },
    {
        "id": "grounding_duplicate_evidence",
        "task": (
            "Improve validate_grounded_plan so duplicate evidence_ids within one material claim fail "
            "deterministically, and extend the existing acceptance case to prove the rejection. "
            "Keep the change scoped to the selected files and add no dependencies."
        ),
        "symbol": "validate_grounded_plan",
        "selected": ["app/engineer_grounding.py", "scripts/engineer_acceptance.py"],
        "roots": ["app", "scripts"],
    },
    {
        "id": "verification_quality",
        "task": (
            "Trace verification command preparation through Patch and Verify. Make unsupported commands "
            "fail loudly at preparation time instead of being silently skipped at execution time."
        ),
        "symbol": "_prepare_verification_commands",
        # prepare_verification_commands() itself lives in app/engineer/verify.py;
        # scripts/company_brain_action.py only re-exports it
        # (_prepare_verification_commands = _verify_mod.prepare_verification_commands).
        # Without the real implementation file in the write scope, the model is
        # asked to change behavior it structurally cannot write a patch for -
        # confirmed live: every replan blocked on "prepare_verification_commands
        # implementation body and return/error semantics" as an unresolved
        # known_unknown, and roots=["scripts"] meant it could never be retrieved
        # either.
        "selected": ["app/engineer/verify.py", "scripts/company_brain_action.py", "scripts/smoke_test.py"],
        "roots": ["app", "scripts"],
    },
    {
        "id": "stale_source_guard",
        "task": (
            "Verify that a selected file mutated after planning blocks Apply before the first write. "
            "If any path can reach a write without a fresh-hash check, close it."
        ),
        "symbol": "engineer_apply_patch",
        # Same class of misconfiguration found and fixed for verification_quality
        # in this same commit: the real fresh-hash / stale-context check lives in
        # app/engineer/apply.py (engineer_apply_patch, called through
        # app/engineer/api.py's apply_patch), not in
        # scripts/company_brain_action.py (engineer_apply_patch = api.apply_patch,
        # a thin re-export). Without the real implementation file in the write
        # scope, the model cannot write a patch to the code that actually needs
        # tightening.
        "selected": ["app/engineer/apply.py", "scripts/company_brain_action.py"],
        "roots": ["app", "scripts"],
    },
]

TASK_SETS: dict[str, list[dict]] = {"cheap": CHEAP_TASKS, "hard": HARD_TASKS, "full": FULL_TASKS}


def select_specs(task_set: str, only: str = "") -> list[dict]:
    """Filter a task set by --only, which accepts one id or a comma-separated
    list (e.g. "retrieval_broker,verification_quality") - a targeted subset
    to run together under real parallel dispatch (run_benchmark's
    ProcessPoolExecutor), not the whole set and not one task run alone.
    """
    if not only:
        return list(TASK_SETS[task_set])
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    return [s for s in TASK_SETS[task_set] if s["id"] in wanted]


# --------------------------------------------------------------------------- cost


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Exact char counts are also recorded, so this only
    needs to be consistent across arms, not perfectly accurate."""
    return max(1, len(text) // 4)


def estimate_cost_usd(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = PRICING_USD_PER_MTOK.get(provider, (0.0, 0.0))
    return round((prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000, 6)


# ------------------------------------------------------------------------- scoring

SCHEMA = "companybrain.engineer.harness_ab.v2"


def diff_target_paths(diff_text: str) -> list[str]:
    paths: set[str] = set()
    for line in diff_text.splitlines():
        match = re.match(r"^(?:\+\+\+|---)\s+[ab]/(.+?)\s*$", line)
        if match and match.group(1) not in {"dev/null"}:
            paths.add(match.group(1).replace("\\", "/"))
    return sorted(paths)


VERIFY_CMD = r"(?:npm|npx|node|python|python3|pytest|tsc|cargo|go)\s+[^\n`\"']+"


def extract_verification_commands(text: str) -> list[str]:
    """Pull plausible verification commands out of free-form model text.

    Deliberately generous: it matches commands in backticks, in fenced blocks, and at
    line start. The baseline arm is scored on this, so under-matching would make the
    harness look better than it is.
    """
    found: list[str] = []
    for pattern in (rf"`\s*(?:\$\s*)?({VERIFY_CMD})\s*`", rf"(?m)^\s*(?:\$\s*)?({VERIFY_CMD})$"):
        for match in re.findall(pattern, text):
            cleaned = match.strip()
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found[:5]


def _looks_like_hunk(block: str) -> bool:
    """A diff body without file headers: has +/- edit lines, not just prose."""
    lines = block.splitlines()
    edits = sum(1 for line in lines if line[:1] in {"+", "-"} and line[:3] not in {"---", "+++"})
    return edits > 0


def extract_diff(text: str, selected: list[str] | None = None) -> tuple[str, bool]:
    """Return (diff_text, headerless).

    Models frequently emit a bare hunk inside a ```diff fence with no --- / +++
    headers. That is still a usable patch, and rejecting it scores a real answer as
    "produced no patch" - which silently understates whichever arm happens to
    format that way. When the task selected exactly one file, attribution of a
    headerless hunk is unambiguous, so headers are synthesised for it.
    """
    fenced = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL)
    for block in fenced:
        if "---" in block and "+++" in block:
            return block.strip(), False
    if "--- a/" in text and "+++ b/" in text:
        start = text.index("--- a/")
        return text[start:].strip(), False

    for block in fenced:
        if _looks_like_hunk(block) and selected and len(selected) == 1:
            target = selected[0].replace("\\", "/")
            return f"--- a/{target}\n+++ b/{target}\n{block.strip()}", True
    return "", False


class QuotaExhausted(RuntimeError):
    """The lead provider refused further calls. Continuing would fabricate data."""


QUOTA_MARKERS = (
    "session limit",
    "rate limit",
    "quota",
    "usage limit",
    "too many requests",
    "429",
)


def is_quota_error(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


def releasable_diff(arm: dict) -> tuple[str, bool]:
    """Return (diff_for_scoring, blocked_but_nonempty).

    A patch counts as produced only when it is releasable. Harness arms that keep a
    non-empty unified_diff under patch_status != 'proposed' (blocked, etc.) must score
    identically to an empty diff. The raw diff and patch_status stay on the arm for
    diagnosis; blocked-but-nonempty is reported separately so the information is not lost.
    Baseline arms have no patch_status and are releasable whenever the diff is non-empty.
    """
    raw = arm.get("unified_diff") or ""
    status = arm.get("patch_status")
    nonempty = bool(str(raw).strip())
    if status is None:
        return (raw if nonempty else ""), False
    if status == "proposed":
        return (raw if nonempty else ""), False
    return "", nonempty


def working_tree_fingerprint(root: Path = ROOT) -> str:
    """Hash tracked source under app/, scripts/, config/ for mutation guards."""
    digest = hashlib.sha256()
    for rel_root in ("app", "scripts", "config"):
        base = root / rel_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part == "__pycache__" or str(part).endswith(".pyc") for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _run_behavior_command(command: list[str], cwd: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    detail = ((completed.stdout or "") + (completed.stderr or ""))[-800:]
    return completed.returncode == 0, detail


def _apply_diff_in_sandbox(sandbox: Path, diff_text: str) -> None:
    """Apply a unified diff inside a disposable tree. Never touches ROOT."""
    parsed = action._parse_unified_diff(diff_text)  # noqa: SLF001
    for file_patch in parsed:
        display = (file_patch.get("new_path") or file_patch.get("old_path") or "").replace("\\", "/")
        if not display or display == "dev/null":
            raise RuntimeError(f"sandbox apply refused empty path in patch: {file_patch!r}")
        target = sandbox / display
        old_path = (file_patch.get("old_path") or "").replace("\\", "/")
        if old_path in {"", "dev/null"}:
            if target.exists():
                raise RuntimeError(f"sandbox create already exists: {display}")
            target.parent.mkdir(parents=True, exist_ok=True)
            new_text = action._apply_patch_to_text("", file_patch["hunks"], display)  # noqa: SLF001
            target.write_text(new_text, encoding="utf-8", newline="\n")
            continue
        if not target.is_file():
            raise RuntimeError(f"sandbox file missing: {display}")
        original = target.read_text(encoding="utf-8", errors="replace")
        new_text = action._apply_patch_to_text(original, file_patch["hunks"], display)  # noqa: SLF001
        target.write_text(new_text, encoding="utf-8", newline="\n")


def evaluate_patch_applies(diff_text: str) -> bool | None:
    """Dry-run applicability against the live tree; never writes."""
    if not str(diff_text or "").strip():
        return None
    result = action._dry_run_patch_applicability(diff_text)  # noqa: SLF001
    return bool(result.get("ok"))


def evaluate_behavior_correct(diff_text: str, behavior_test: dict | None) -> bool | None:
    """Score behavioral correctness in a disposable repo copy only.

    Returns True only when the oracle fails before the patch and passes after.
    Returns None (excluded from denominator) when there is no oracle, no releasable
    patch, or the oracle is invalid (already passes before the patch).
    """
    if not behavior_test or not str(diff_text or "").strip():
        return None
    command = behavior_test.get("command")
    if not isinstance(command, list) or not command:
        return None

    before_fp = working_tree_fingerprint(ROOT)
    sandbox_parent = tempfile.mkdtemp(prefix="companybrain_ab_")
    sandbox = Path(sandbox_parent) / "repo"
    try:
        shutil.copytree(ROOT, sandbox, ignore=SANDBOX_IGNORE, dirs_exist_ok=False)
        pre_ok, pre_detail = _run_behavior_command(command, sandbox)
        if pre_ok:
            # Oracle must fail on current code; otherwise the task cannot measure correction.
            return None
        _apply_diff_in_sandbox(sandbox, diff_text)
        post_ok, post_detail = _run_behavior_command(command, sandbox)
        _ = pre_detail, post_detail
        return bool(post_ok)
    except Exception:  # noqa: BLE001
        return False
    finally:
        shutil.rmtree(sandbox_parent, ignore_errors=True)
        after_fp = working_tree_fingerprint(ROOT)
        if after_fp != before_fp:
            raise RuntimeError(
                "Working tree mutated during behavior_correct scoring; aborting. "
                f"before={before_fp[:12]} after={after_fp[:12]}"
            )


def score_arm(arm: dict, selected: list[str], spec: dict | None = None) -> dict:
    # An arm that never got a model response is not a zero - it is missing data.
    # Scoring it as zero silently converts an outage into evidence, which is how a
    # broken run masquerades as a verdict.
    if not arm.get("ok", True):
        arm["checks"] = {}
        arm["score"] = None
        arm["score_max"] = None
        arm["scored"] = False
        arm["blocked_patch_nonempty"] = False
        return arm

    scoring_diff, blocked_nonempty = releasable_diff(arm)
    arm["blocked_patch_nonempty"] = blocked_nonempty
    arm["scoring_unified_diff"] = scoring_diff
    targets = diff_target_paths(scoring_diff)
    selected_norm = {p.replace("\\", "/") for p in selected}
    out_of_scope = [p for p in targets if p not in selected_norm]
    nonexistent = [p for p in targets if not (ROOT / p).is_file()]
    has_patch = bool(scoring_diff.strip())

    behavior_test = (spec or {}).get("behavior_test") if spec else None
    checks: dict[str, bool | None] = {
        "produced_patch": has_patch,
        "scope_respected": (bool(targets) and not out_of_scope) if has_patch else False,
        # Vacuous "no paths => no hallucinations" previously floored failed arms at 1/4.
        "no_hallucinated_paths": (not nonexistent) if has_patch else None,
        "verification_command_present": bool(arm.get("verification_commands")) if has_patch else False,
        "patch_applies": evaluate_patch_applies(scoring_diff),
        "behavior_correct": evaluate_behavior_correct(scoring_diff, behavior_test),
    }
    # produced_patch/scope_respected/no_hallucinated_paths/patch_applies only test that a
    # diff is well-formed and in scope, not that it works. behavior_correct is the only
    # check that runs the oracle and proves the change does what the task asked. Weighting
    # every check equally let a plausible-but-wrong guess (form checks pass, behavior_correct
    # fails) outscore a correct refusal to guess (form checks score 0/None, nothing to be
    # wrong about). Weighting behavior_correct higher makes the score track "did it work"
    # instead of "did it look like an attempt".
    check_weights = {
        "produced_patch": 1,
        "scope_respected": 1,
        "no_hallucinated_paths": 1,
        "verification_command_present": 1,
        "patch_applies": 1,
        "behavior_correct": 3,
    }
    scored_items = [(key, value) for key, value in checks.items() if value is not None]
    arm["target_paths"] = targets
    arm["out_of_scope_paths"] = out_of_scope
    arm["nonexistent_paths"] = nonexistent
    arm["checks"] = checks
    arm["check_weights"] = check_weights
    arm["score"] = sum(check_weights[key] for key, value in scored_items if value)
    arm["score_max"] = sum(check_weights[key] for key, _ in scored_items)
    arm["scored"] = True
    return arm


# ---------------------------------------------------------------------------- arms


def build_baseline_prompt(spec: dict, max_file_chars: int) -> tuple[str, list[str]]:
    sections = [f"# Task\n\n{spec['task']}\n", "# Files\n"]
    truncated: list[str] = []
    for rel in spec["selected"]:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_file_chars:
            text = text[:max_file_chars]
            truncated.append(rel)
        sections.append(f"\n## {rel}\n\n```\n{text}\n```\n")
    return "\n".join(sections), truncated


def ensure_selected_files_fit(specs: list[dict], max_file_chars: int) -> int:
    """Hard tasks require the full selected file; truncation would fake the asymmetry."""
    needed = max_file_chars
    for spec in specs:
        for rel in spec.get("selected") or []:
            path = ROOT / rel
            if path.is_file():
                needed = max(needed, len(path.read_text(encoding="utf-8", errors="replace")))
    return needed


def estimate_run(specs: list[dict], max_file_chars: int, repeat: int = 1) -> dict:
    """Cost preview. Makes zero model calls."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    HARNESS_CALLS_PER_TASK = 3  # plan + replan + patch, sometimes a 4th repair pass
    rows = []
    total_tokens = 0
    for spec in specs:
        prompt, truncated = build_baseline_prompt(spec, max_file_chars)
        baseline_tokens = estimate_tokens(prompt)
        # Harness prompts are bounded by retrieval budgets (~8k chars per call).
        harness_tokens = estimate_tokens("x" * 8000) * HARNESS_CALLS_PER_TASK
        total_tokens += baseline_tokens + harness_tokens
        rows.append(
            {
                "task_id": spec["id"],
                "baseline_prompt_chars": len(prompt),
                "baseline_tokens_est": baseline_tokens,
                "harness_tokens_est": harness_tokens,
                "truncated": truncated,
                "calls": 1 + HARNESS_CALLS_PER_TASK,
                "selected": list(spec["selected"]),
                "has_behavior_test": bool(spec.get("behavior_test")),
                # Audit only — never sent to models.
                "hidden_fact": spec.get("hidden_fact"),
            }
        )
    estimate = {
        "tasks": rows,
        "total_calls_est": sum(r["calls"] for r in rows),
        "total_prompt_tokens_est": total_tokens,
    }
    if repeat == 1:
        return estimate

    for row in rows:
        row["baseline_tokens_est"] *= repeat
        row["harness_tokens_est"] *= repeat
        row["calls"] *= repeat
    estimate["total_calls_est"] *= repeat
    estimate["total_prompt_tokens_est"] *= repeat
    estimate["repeat"] = repeat
    return estimate


def run_baseline(spec: dict, provider: str, model: str, max_file_chars: int = BASELINE_MAX_FILE_CHARS) -> dict:
    began = datetime.now()
    arm: dict = {"arm": "harness_off", "provider": provider, "model": model}
    prompt, truncated = build_baseline_prompt(spec, max_file_chars)
    arm["files_truncated"] = truncated
    try:
        content = action.call_provider(provider, "Lead Engineer", prompt, model_override=model or None)
        arm["ok"] = True
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {str(exc)[:900]}"
        arm.update({"ok": False, "error": message})
        if is_quota_error(message):
            raise QuotaExhausted(message) from exc
        content = ""
    # Keep a bounded slice of the raw response. Without it, "no diff produced" is
    # undiagnosable without paying for the whole run again.
    arm["raw_response_excerpt"] = content[:2000]
    diff_text, headerless = extract_diff(content, spec["selected"])
    arm["unified_diff"] = diff_text
    arm["diff_headerless"] = headerless
    arm["verification_commands"] = extract_verification_commands(content)
    arm["raw_response_chars"] = len(content)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(content)
    arm.update(
        {
            "prompt_chars": len(prompt),
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "cost_usd_est": estimate_cost_usd(provider, prompt_tokens, completion_tokens),
            "seconds": round((datetime.now() - began).total_seconds(), 2),
            "model_calls": 1,
        }
    )
    return arm


def run_harness(spec: dict, provider: str = "") -> dict:
    began = datetime.now()
    arm: dict = {"arm": "harness_on", "pinned_provider": provider}
    # Pin the harness arm to the same provider the baseline used. Without this the
    # registry picks by tier priority and the two arms can run different models,
    # which measures the model rather than the harness.
    previous_pin = os.environ.get("COMPANYBRAIN_PIN_PROVIDER")
    if provider:
        os.environ["COMPANYBRAIN_PIN_PROVIDER"] = provider
    try:
        plan = action.engineer_plan(
            spec["task"],
            selected_files=spec["selected"],
            project="CompanyBrain harness A/B benchmark",
            use_approved_lessons=True,
            retrieval_roots=spec["roots"],
            retrieval_intents=["find_symbol", "find_references", "find_tests"],
        )
        run_id = str(plan.get("run_id") or "")
        request_id = f"ctx_ab_{spec['id']}"
        request_payload = {
            "request_id": request_id,
            "task_id": run_id,
            "requested_items": [
                {
                    "item_id": suffix,
                    "intent": intent,
                    "reason": f"Ground the {suffix} needed for {spec['id']}.",
                    "requested_roots": spec["roots"],
                    "query": spec["symbol"],
                    "include_tests": True,
                    "estimated_budget": {"max_files": 3, "max_excerpts": 6, "max_chars": 8000},
                }
                for suffix, intent in [
                    ("definition", "find_symbol"),
                    ("callers", "find_references"),
                    ("tests", "find_tests"),
                ]
            ],
            "external_retrieval": False,
        }
        action.engineer_context_request(run_id, json.dumps(request_payload))
        execution = action.engineer_context_execute(run_id, request_id)
        replan = action.engineer_replan(run_id)
        check = action.engineer_check_run(run_id)
        patch = action.engineer_patch(run_id, selected_files=spec["selected"])

        retrieved = sorted(
            {
                str(item.get("path"))
                for retrieval in (execution.get("retrieval_results") or [])
                for item in retrieval.get("results") or []
                if item.get("path")
            }
        )
        # Keep unified_diff and patch_status even when blocked so diagnosis is possible.
        # score_arm treats anything other than patch_status=="proposed" as no patch.
        arm.update(
            {
                "ok": True,
                "run_id": run_id,
                "patch_id": patch.get("patch_id"),
                "plan_checker": plan.get("checker_status"),
                "replan_checker": replan.get("checkerStatus"),
                "replan_failed_rules": replan.get("failedRules") or [],
                "replan_followup": replan.get("followup") or {"attempted": False},
                "final_checker": check.get("checkerStatus"),
                "context_sufficiency": replan.get("contextSufficiency"),
                "retrieved_paths": retrieved,
                "patch_status": patch.get("patch_status"),
                "patch_checker_failed_rules": (patch.get("patch_checker") or {}).get("failed_rules") or [],
                "patch_followup": patch.get("patch_followup") or {"attempted": False},
                "unified_diff": patch.get("unified_diff") or "",
                "verification_commands": patch.get("verification_commands") or [],
                "evidence_trace_present": bool(patch.get("patch_claim_trace")),
                "model_route": patch.get("model_route"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {str(exc)[:900]}"
        arm.update({"ok": False, "error": message, "unified_diff": ""})
        if isinstance(exc, QuotaExhausted) or is_quota_error(message):
            raise QuotaExhausted(message) from exc
    finally:
        if provider:
            if previous_pin is None:
                os.environ.pop("COMPANYBRAIN_PIN_PROVIDER", None)
            else:
                os.environ["COMPANYBRAIN_PIN_PROVIDER"] = previous_pin

    arm["seconds"] = round((datetime.now() - began).total_seconds(), 2)
    arm.setdefault("verification_commands", [])
    return arm


# -------------------------------------------------------------------------- report

# Cap on concurrent task processes. Each one spawns its own CLI subprocess
# (Cursor/Codex/Claude), so this also bounds how many of those run at once -
# kept modest to stay well under any provider-side concurrency/rate limits.
MAX_PARALLEL_TASKS = 4


def _run_one_task(spec: dict, provider: str, model: str, max_file_chars: int) -> dict:
    """Run one task's baseline+harness arms to completion. Runs in its own OS
    process, not a thread: run_harness() temporarily mutates the process-global
    os.environ["COMPANYBRAIN_PIN_PROVIDER"] for the duration of its call, and
    concurrent threads sharing one process would clobber each other's pin.
    Separate processes each get their own environment copy, so this hazard
    doesn't need to be touched at all to parallelize safely. Each call's
    disposable checkout already comes from tempfile.mkdtemp() (unique per
    call, process-safe), so no other shared-state hazard applies here.

    Baseline and harness are run sequentially, deliberately, even though the
    two arms don't read each other's output: run_deterministic_self_tests()
    encodes a real, intentional guarantee that a quota error on baseline means
    harness is never even attempted (see "baseline_quota_aborts_immediately" -
    its mock is named never_harness for a reason). Once you know a provider is
    out of quota, making the second call anyway just spends more of it for a
    result you're going to discard. A concurrent version was tried and
    reverted: both calls start before either can signal "stop, we're out,"
    so the two are fundamentally in tension - concurrency wins on speed,
    sequential wins on not wasting calls after a known failure. Kept
    sequential because that failure-path guarantee is the one that matters
    when it matters (an exhausted or rate-limited provider), not the common
    case this would have sped up.

    Returns a normal result row, or {"aborted_reason": ...} if quota-exhausted
    partway through - the caller distinguishes the two by "task_id" absence.
    """
    try:
        baseline = score_arm(run_baseline(spec, provider, model, max_file_chars), spec["selected"], spec)
        harness = score_arm(run_harness(spec, provider), spec["selected"], spec)
    except QuotaExhausted as exc:
        return {"aborted_reason": str(exc)}
    both_scored = baseline.get("scored") and harness.get("scored")
    return {
        "task_id": spec["id"],
        "task": spec["task"],
        "selected_files": spec["selected"],
        "harness_off": baseline,
        "harness_on": harness,
        "comparable": bool(both_scored),
        "harness_wins": bool(both_scored) and harness["score"] > baseline["score"],
        "tie": bool(both_scored) and harness["score"] == baseline["score"],
        "blocked_patch_nonempty_on": bool(harness.get("blocked_patch_nonempty")),
    }


def run_benchmark(
    provider: str,
    model: str,
    only: str = "",
    task_set: str = "cheap",
    max_file_chars: int = BASELINE_MAX_FILE_CHARS,
    repeat: int = 1,
    _artifact_suffix: str = "",
    parallel: bool = True,
) -> dict:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if repeat > 1:
        return run_repeated_benchmark(
            provider,
            model,
            repeat,
            only=only,
            task_set=task_set,
            max_file_chars=max_file_chars,
            parallel=parallel,
        )

    specs = select_specs(task_set, only)
    if task_set == "hard":
        max_file_chars = ensure_selected_files_fit(specs, max_file_chars)
    rows = []
    aborted = ""
    tree_before = working_tree_fingerprint(ROOT)
    # Each task is already fully self-contained - its own run_id, its own
    # tempfile.mkdtemp() disposable checkout, no shared mutable state that
    # survives past _run_one_task's own process - so nothing about running
    # them one at a time was ever load-bearing; it was just how the loop
    # happened to be written. Parallel execution turns "N tasks * one CLI
    # call at a time" into "N tasks concurrently", which is the actual
    # bottleneck given each CLI call is real network-bound wall-clock time.
    #
    # parallel=False keeps the original sequential in-process loop, and
    # exists only so run_deterministic_self_tests() can keep monkeypatching
    # mod.run_baseline/mod.run_harness the way it always has - a
    # ProcessPoolExecutor worker re-imports this module fresh in a new
    # interpreter, so a parent-process monkeypatch never reaches it. This
    # path is not a fallback for real runs, it is what the self-tests
    # actually exercise, so it needs to keep working exactly as before.
    if parallel:
        max_workers = max(1, min(len(specs), MAX_PARALLEL_TASKS))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one_task, spec, provider, model, max_file_chars) for spec in specs]
            results = [future.result() for future in futures]
        for result in results:
            if "task_id" not in result:
                aborted = aborted or result.get("aborted_reason", "")
                continue
            rows.append(result)
    else:
        for spec in specs:
            result = _run_one_task(spec, provider, model, max_file_chars)
            if "task_id" not in result:
                aborted = aborted or result.get("aborted_reason", "")
                break
            rows.append(result)

    tree_after = working_tree_fingerprint(ROOT)
    if tree_after != tree_before:
        raise RuntimeError("Working tree mutated during harness A/B benchmark; refusing to report.")

    comparable = [r for r in rows if r["comparable"]]
    wins = sum(1 for r in comparable if r["harness_wins"])
    ties = sum(1 for r in comparable if r["tie"])
    losses = len(comparable) - wins - ties
    off_score = sum(r["harness_off"]["score"] for r in comparable)
    on_score = sum(r["harness_on"]["score"] for r in comparable)
    off_cost = round(sum(r["harness_off"].get("cost_usd_est", 0.0) or 0.0 for r in rows), 6)
    blocked_nonempty = sum(1 for r in rows if r.get("blocked_patch_nonempty_on"))

    report = {
        "benchmark": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lead_provider": provider,
        "lead_model": model,
        "task_set": task_set,
        "tasks_total": len(rows),
        "tasks_attempted": len(specs),
        "tasks_comparable": len(comparable),
        "aborted_reason": aborted,
        "harness_wins": wins,
        "ties": ties,
        "harness_losses": losses,
        "score_harness_off": off_score,
        "score_harness_on": on_score,
        "score_max": sum((r["harness_off"]["score_max"] or 0) for r in comparable),
        "baseline_cost_usd_est": off_cost,
        "blocked_patch_nonempty_count": blocked_nonempty,
        # A verdict requires enough tasks where BOTH arms actually produced output.
        # Fewer than three comparable pairs is an outage report, not a finding.
        "verdict": (
            "inconclusive_insufficient_data"
            if aborted or len(comparable) < 3
            else "harness_earns_its_complexity"
            if on_score > off_score
            else "harness_not_yet_justified"
        ),
        "tasks": rows,
        "safety": {
            "patch_applied": False,
            "verification_commands_run": False,
            "lessons_promoted": False,
            "external_retrieval_used": False,
            "working_tree_mutated": False,
            "working_tree_fingerprint": tree_after,
        },
        "caveats": [
            "Token counts are estimated from character length, not provider usage headers.",
            "Null checks (no patch / no behavior oracle) are excluded from each arm's denominator.",
            "Blocked patches retain patch_status and unified_diff but score as no patch.",
            "behavior_correct runs only inside a disposable repo copy; the working tree is never written.",
        ],
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = EVAL_DIR / f"{stamp}_harness_ab_benchmark{_artifact_suffix}.json"
    md_path = EVAL_DIR / f"{stamp}_harness_ab_benchmark{_artifact_suffix}.md"

    lines = [
        "# Harness A/B Benchmark",
        "",
        f"- Lead model: `{provider}` / `{model or 'provider default'}`",
        f"- Task set: `{task_set}`",
        f"- Verdict: **{report['verdict']}**",
        f"- Objective score: harness-off {off_score} vs harness-on {on_score} (max {report['score_max']})",
        f"- Wins {wins} / ties {ties} / losses {losses}",
        f"- Blocked-but-nonempty harness patches (scored as no patch): {blocked_nonempty}",
        f"- Estimated baseline spend: ${off_cost}",
        "",
        "| task | off | on | produced (off/on) | applies (off/on) | behavior (off/on) | blocked nonempty |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        off, on = row["harness_off"], row["harness_on"]
        if not row["comparable"]:
            reason = (off.get("error") or on.get("error") or "incomplete")[:60]
            lines.append(f"| {row['task_id']} | n/a | n/a | not comparable | | | {reason} |")
            continue
        lines.append(
            f"| {row['task_id']} | {off['score']}/{off['score_max']} | {on['score']}/{on['score_max']} "
            f"| {off['checks'].get('produced_patch')}/{on['checks'].get('produced_patch')} "
            f"| {off['checks'].get('patch_applies')}/{on['checks'].get('patch_applies')} "
            f"| {off['checks'].get('behavior_correct')}/{on['checks'].get('behavior_correct')} "
            f"| {row.get('blocked_patch_nonempty_on')} |"
        )
    lines.extend(["", "```json", json.dumps(report, ensure_ascii=False, indent=2), "```", ""])

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    report["path"] = str(md_path)
    report["jsonPath"] = str(json_path)
    return report


def _score_statistics(reports: list[dict], score_key: str) -> dict:
    scores = [report[score_key] for report in reports]
    if not scores:
        return {"sample_size": 0, "mean": None, "min": None, "max": None, "spread": None}
    return {
        "sample_size": len(scores),
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
    }


def _repeat_run_complete(report: dict, expected_tasks: int) -> bool:
    return bool(
        not report.get("aborted_reason")
        and report.get("tasks_total") == expected_tasks
        and report.get("tasks_comparable") == expected_tasks
    )


def run_repeated_benchmark(
    provider: str,
    model: str,
    repeat: int,
    only: str = "",
    task_set: str = "cheap",
    max_file_chars: int = BASELINE_MAX_FILE_CHARS,
    parallel: bool = True,
) -> dict:
    """Run complete benchmark samples sequentially and summarize score variance."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    specs = select_specs(task_set, only)
    runs = []
    completed_reports = []
    for run_index in range(1, repeat + 1):
        result = run_benchmark(
            provider,
            model,
            only=only,
            task_set=task_set,
            max_file_chars=max_file_chars,
            repeat=1,
            _artifact_suffix=f"_run_{run_index:03d}",
            parallel=parallel,
        )
        complete = _repeat_run_complete(result, len(specs))
        runs.append({"run_index": run_index, "complete": complete, "result": result})
        if complete:
            completed_reports.append(result)
        if result.get("aborted_reason"):
            break

    attempted = len(runs)
    completed = len(completed_reports)
    incomplete = completed != repeat
    off_statistics = _score_statistics(completed_reports, "score_harness_off")
    on_statistics = _score_statistics(completed_reports, "score_harness_on")
    enough_tasks = bool(completed_reports) and all(
        result["tasks_comparable"] >= 3 for result in completed_reports
    )
    off_spread = off_statistics.get("spread")
    on_spread = on_statistics.get("spread")
    within_arm_spread = None
    between_arm_gap = None
    gap_exceeds_within_arm_spread = None
    if (
        off_statistics.get("mean") is not None
        and on_statistics.get("mean") is not None
        and off_spread is not None
        and on_spread is not None
    ):
        within_arm_spread = max(off_spread, on_spread)
        between_arm_gap = abs(on_statistics["mean"] - off_statistics["mean"])
        gap_exceeds_within_arm_spread = between_arm_gap > within_arm_spread

    if incomplete or not enough_tasks:
        verdict = "inconclusive_insufficient_data"
    elif gap_exceeds_within_arm_spread is False:
        # A gap smaller than the within-arm error bar is not a verdict.
        verdict = "inconclusive_within_noise"
    elif on_statistics["mean"] > off_statistics["mean"]:
        verdict = "harness_earns_its_complexity"
    else:
        verdict = "harness_not_yet_justified"
    aborted_reason = next(
        (entry["result"]["aborted_reason"] for entry in runs if entry["result"].get("aborted_reason")),
        "",
    )

    report = {
        "benchmark": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lead_provider": provider,
        "lead_model": model,
        "task_set": task_set,
        "repeat": {
            "requested": repeat,
            "attempted": attempted,
            "completed": completed,
            "incomplete": incomplete,
        },
        "aborted_reason": aborted_reason,
        "verdict": verdict,
        "score_statistics": {
            "harness_off": off_statistics,
            "harness_on": on_statistics,
            "between_arm_gap": between_arm_gap,
            "within_arm_spread": within_arm_spread,
            "gap_exceeds_within_arm_spread": gap_exceeds_within_arm_spread,
        },
        "baseline_cost_usd_est": round(
            sum(entry["result"].get("baseline_cost_usd_est", 0.0) or 0.0 for entry in runs),
            6,
        ),
        "runs": runs,
        "safety": {
            "patch_applied": False,
            "verification_commands_run": False,
            "lessons_promoted": False,
            "external_retrieval_used": False,
            "working_tree_mutated": False,
        },
        "caveats": [
            "Token counts are estimated from character length, not provider usage headers.",
            "Null checks (no patch / no behavior oracle) are excluded from each arm's denominator.",
            "Blocked patches retain patch_status and unified_diff but score as no patch.",
            "Repeat statistics include only runs where every selected task produced a comparable A/B pair.",
            "If the between-arm gap does not exceed within-arm spread, verdict is inconclusive_within_noise.",
        ],
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = EVAL_DIR / f"{stamp}_harness_ab_benchmark_repeat_{repeat}.json"
    md_path = EVAL_DIR / f"{stamp}_harness_ab_benchmark_repeat_{repeat}.md"
    gap_line = (
        f"- Between-arm gap {between_arm_gap} vs within-arm spread {within_arm_spread}: "
        f"{'exceeds noise' if gap_exceeds_within_arm_spread else 'within noise / inconclusive'}"
        if between_arm_gap is not None
        else "- Between-arm gap vs within-arm spread: n/a (insufficient complete runs)"
    )
    lines = [
        "# Harness A/B Benchmark (Repeated)",
        "",
        f"- Lead model: `{provider}` / `{model or 'provider default'}`",
        f"- Task set: `{task_set}`",
        f"- Repetitions: {completed}/{repeat} complete ({attempted} attempted)",
        f"- Verdict: **{verdict}**",
        (
            f"- harness_off mean/min/max/spread: "
            f"{off_statistics.get('mean')}/{off_statistics.get('min')}/"
            f"{off_statistics.get('max')}/{off_statistics.get('spread')}"
        ),
        (
            f"- harness_on mean/min/max/spread: "
            f"{on_statistics.get('mean')}/{on_statistics.get('min')}/"
            f"{on_statistics.get('max')}/{on_statistics.get('spread')}"
        ),
        gap_line,
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    report["path"] = str(md_path)
    report["jsonPath"] = str(json_path)
    return report


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _stub_ok_arm(arm_name: str) -> dict:
    return {
        "arm": arm_name,
        "ok": True,
        "unified_diff": (
            "--- a/app/env_loader.py\n+++ b/app/env_loader.py\n"
            "@@ -1 +1 @@\n"
            "-from __future__ import annotations\n"
            "+from __future__ import annotations\n"
        ),
        "verification_commands": ["python scripts/smoke_test.py"],
        "cost_usd_est": 0.0,
        "seconds": 0.01,
        "model_calls": 1,
    }


def run_deterministic_self_tests() -> dict:
    """Zero-model-call regression suite for repeat accounting and quota abort."""

    import sys

    mod = sys.modules[__name__]
    results: list[dict] = []
    observed_calls: dict[str, dict] = {}
    original_baseline = mod.run_baseline
    original_harness = mod.run_harness
    original_benchmark = mod.run_benchmark
    original_plan = action.engineer_plan

    try:
        base = estimate_run(CHEAP_TASKS, BASELINE_MAX_FILE_CHARS, repeat=1)
        triple = estimate_run(CHEAP_TASKS, BASELINE_MAX_FILE_CHARS, repeat=3)
        results.append(
            {
                "id": "repeat_estimate_scales_exactly",
                "ok": (
                    triple["total_calls_est"] == base["total_calls_est"] * 3
                    and triple["total_prompt_tokens_est"] == base["total_prompt_tokens_est"] * 3
                    and all(
                        row3["calls"] == row1["calls"] * 3
                        and row3["baseline_tokens_est"] == row1["baseline_tokens_est"] * 3
                        and row3["harness_tokens_est"] == row1["harness_tokens_est"] * 3
                        for row1, row3 in zip(base["tasks"], triple["tasks"])
                    )
                ),
                "detail": {
                    "base_calls": base["total_calls_est"],
                    "triple_calls": triple["total_calls_est"],
                    "base_tokens": base["total_prompt_tokens_est"],
                    "triple_tokens": triple["total_prompt_tokens_est"],
                },
            }
        )

        call_counts = {"baseline": 0, "harness": 0}

        def fake_baseline(spec, provider, model, max_file_chars=BASELINE_MAX_FILE_CHARS):
            call_counts["baseline"] += 1
            return _stub_ok_arm("harness_off")

        def fake_harness(spec, provider=""):
            call_counts["harness"] += 1
            return _stub_ok_arm("harness_on")

        mod.run_baseline = fake_baseline
        mod.run_harness = fake_harness
        single = mod.run_benchmark(
            "Stub Provider", "", only="env_loader_precedence", task_set="cheap", repeat=1, parallel=False
        )
        results.append(
            {
                "id": "repeat_one_single_run_shape",
                "ok": (
                    "repeat" not in single
                    and single.get("benchmark") == SCHEMA
                    and isinstance(single.get("tasks"), list)
                    and single.get("aborted_reason") == ""
                    and call_counts == {"baseline": 1, "harness": 1}
                ),
                "detail": {"keys": sorted(single.keys()), "calls": dict(call_counts)},
            }
        )
        observed_calls["repeat_one"] = dict(call_counts)

        stats = _score_statistics(
            [
                {"score_harness_off": 1, "score_harness_on": 8},
                {"score_harness_off": 3, "score_harness_on": 3},
                {"score_harness_off": 8, "score_harness_on": 1},
            ],
            "score_harness_off",
        )
        on_stats = _score_statistics(
            [
                {"score_harness_off": 1, "score_harness_on": 8},
                {"score_harness_off": 3, "score_harness_on": 3},
                {"score_harness_off": 8, "score_harness_on": 1},
            ],
            "score_harness_on",
        )
        results.append(
            {
                "id": "score_statistics_known_set",
                "ok": stats
                == {"sample_size": 3, "mean": 4.0, "min": 1, "max": 8, "spread": 7}
                and on_stats == {"sample_size": 3, "mean": 4.0, "min": 1, "max": 8, "spread": 7},
                "detail": {"off": stats, "on": on_stats},
            }
        )

        invalid_ok = True
        invalid_detail = {}
        for bad in (0, -1):
            try:
                estimate_run(CHEAP_TASKS, BASELINE_MAX_FILE_CHARS, repeat=bad)
                invalid_ok = False
                invalid_detail[str(bad)] = "estimate_accepted"
            except ValueError:
                invalid_detail[str(bad)] = "estimate_rejected"
            try:
                mod.run_benchmark("Stub", "", repeat=bad)
                invalid_ok = False
                invalid_detail[f"run_{bad}"] = "run_accepted"
            except ValueError:
                invalid_detail[f"run_{bad}"] = "run_rejected"
        try:
            _positive_int("nope")
            invalid_ok = False
            invalid_detail["non_integer"] = "accepted"
        except argparse.ArgumentTypeError:
            invalid_detail["non_integer"] = "rejected"
        results.append({"id": "invalid_repeat_rejected", "ok": invalid_ok, "detail": invalid_detail})

        ordinary_calls = {"baseline": 0, "harness": 0}

        def ordinary_baseline(spec, provider, model, max_file_chars=BASELINE_MAX_FILE_CHARS):
            ordinary_calls["baseline"] += 1
            return _stub_ok_arm("harness_off")

        def ordinary_harness(spec, provider=""):
            ordinary_calls["harness"] += 1
            # Mirror real run_harness non-quota behavior: failed arm, no raise.
            return {
                "arm": "harness_on",
                "ok": False,
                "error": "RuntimeError: temporary filesystem glitch in stub harness",
                "unified_diff": "",
                "verification_commands": [],
                "seconds": 0.01,
            }

        mod.run_baseline = ordinary_baseline
        mod.run_harness = ordinary_harness
        ordinary = mod.run_benchmark(
            "Stub Provider", "", only="env_loader_precedence", task_set="cheap", repeat=1, parallel=False
        )
        results.append(
            {
                "id": "ordinary_harness_failure_continues",
                "ok": (
                    ordinary.get("aborted_reason") == ""
                    and ordinary.get("tasks_total") == 1
                    and ordinary["tasks"][0]["harness_on"].get("ok") is False
                    and ordinary["tasks"][0]["comparable"] is False
                    and ordinary_calls == {"baseline": 1, "harness": 1}
                ),
                "detail": {"calls": dict(ordinary_calls), "aborted_reason": ordinary.get("aborted_reason")},
            }
        )
        observed_calls["ordinary_failure"] = dict(ordinary_calls)

        baseline_quota_calls = {"baseline": 0, "harness": 0}

        def quota_baseline(spec, provider, model, max_file_chars=BASELINE_MAX_FILE_CHARS):
            baseline_quota_calls["baseline"] += 1
            raise QuotaExhausted("HTTP 429 Too Many Requests: quota exceeded on baseline")

        def never_harness(spec, provider=""):
            baseline_quota_calls["harness"] += 1
            return _stub_ok_arm("harness_on")

        mod.run_baseline = quota_baseline
        mod.run_harness = never_harness
        baseline_abort = mod.run_benchmark("Stub Provider", "", task_set="cheap", repeat=1, parallel=False)
        results.append(
            {
                "id": "baseline_quota_aborts_immediately",
                "ok": (
                    bool(baseline_abort.get("aborted_reason"))
                    and "429" in baseline_abort.get("aborted_reason", "")
                    and baseline_quota_calls == {"baseline": 1, "harness": 0}
                    and baseline_abort.get("tasks_total") == 0
                ),
                "detail": {
                    "calls": dict(baseline_quota_calls),
                    "aborted_reason": baseline_abort.get("aborted_reason"),
                },
            }
        )
        observed_calls["baseline_quota"] = dict(baseline_quota_calls)

        harness_quota_calls = {"baseline": 0, "plan": 0}

        def ok_baseline(spec, provider, model, max_file_chars=BASELINE_MAX_FILE_CHARS):
            harness_quota_calls["baseline"] += 1
            return _stub_ok_arm("harness_off")

        def boom_plan(*_args, **_kwargs):
            harness_quota_calls["plan"] += 1
            raise RuntimeError("ProviderError: HTTP 429 rate limit / quota exceeded")

        mod.run_baseline = ok_baseline
        mod.run_harness = original_harness
        action.engineer_plan = boom_plan  # type: ignore[assignment]
        harness_abort = mod.run_benchmark(
            "Stub Provider", "", only="env_loader_precedence", task_set="cheap", repeat=1, parallel=False
        )
        results.append(
            {
                "id": "harness_quota_aborts_immediately",
                "ok": (
                    bool(harness_abort.get("aborted_reason"))
                    and "429" in harness_abort.get("aborted_reason", "").lower()
                    and harness_abort.get("tasks_total") == 0
                    and harness_quota_calls == {"baseline": 1, "plan": 1}
                ),
                "detail": {
                    "calls": dict(harness_quota_calls),
                    "aborted_reason": harness_abort.get("aborted_reason"),
                    "tasks_total": harness_abort.get("tasks_total"),
                },
            }
        )
        observed_calls["harness_quota"] = dict(harness_quota_calls)

        repeat_calls = {"benchmark": 0}

        def aborting_benchmark(*_args, **kwargs):
            repeat_calls["benchmark"] += 1
            if kwargs.get("repeat", 1) != 1:
                raise AssertionError("run_repeated_benchmark must call run_benchmark with repeat=1")
            if repeat_calls["benchmark"] == 1:
                return {
                    "benchmark": SCHEMA,
                    "aborted_reason": "HTTP 429 quota",
                    "tasks_total": 0,
                    "tasks_comparable": 0,
                    "score_harness_off": 0,
                    "score_harness_on": 0,
                    "baseline_cost_usd_est": 0.0,
                    "path": "",
                    "jsonPath": "",
                }
            raise AssertionError("run_repeated_benchmark continued after abort")

        mod.run_benchmark = aborting_benchmark  # type: ignore[assignment]
        repeated = mod.run_repeated_benchmark("Stub", "", repeat=3, task_set="cheap", parallel=False)
        results.append(
            {
                "id": "repeated_benchmark_stops_on_abort",
                "ok": (
                    repeated.get("aborted_reason") == "HTTP 429 quota"
                    and repeated["repeat"]["attempted"] == 1
                    and repeated["repeat"]["completed"] == 0
                    and repeat_calls["benchmark"] == 1
                ),
                "detail": {"repeat": repeated.get("repeat"), "calls": dict(repeat_calls)},
            }
        )
        observed_calls["repeated_abort"] = dict(repeat_calls)

        # Before: blocked nonempty diffs scored produced_patch=True (and often 4/4).
        # After: identical to empty diff; no_hallucinated_paths is null and excluded.
        legacy_blocked = {
            "arm": "harness_on",
            "ok": True,
            "patch_status": "blocked",
            "unified_diff": (
                "--- a/app/nvidia_catalog.py\n+++ b/app/nvidia_catalog.py\n"
                "@@ -1 +1 @@\n"
                "-from __future__ import annotations\n"
                "+from __future__ import annotations\n"
            ),
            "verification_commands": ["python -m py_compile app/nvidia_catalog.py"],
        }
        scored_blocked = score_arm(dict(legacy_blocked), ["app/nvidia_catalog.py"], None)
        empty_arm = score_arm(
            {
                "arm": "harness_on",
                "ok": True,
                "patch_status": "blocked",
                "unified_diff": "",
                "verification_commands": [],
            },
            ["app/nvidia_catalog.py"],
            None,
        )
        results.append(
            {
                "id": "blocked_nonempty_scores_as_no_patch",
                "ok": (
                    scored_blocked.get("blocked_patch_nonempty") is True
                    and scored_blocked["checks"].get("produced_patch") is False
                    and scored_blocked["checks"].get("no_hallucinated_paths") is None
                    and scored_blocked["checks"].get("scope_respected") is False
                    and scored_blocked["checks"].get("verification_command_present") is False
                    and scored_blocked["checks"].get("patch_applies") is None
                    and scored_blocked["checks"].get("behavior_correct") is None
                    and scored_blocked["score"] == 0
                    and scored_blocked["score_max"] == 3
                    and empty_arm["score"] == scored_blocked["score"]
                    and empty_arm["score_max"] == scored_blocked["score_max"]
                    and empty_arm["checks"] == scored_blocked["checks"]
                ),
                "detail": {
                    "blocked_score": f"{scored_blocked['score']}/{scored_blocked['score_max']}",
                    "blocked_checks": scored_blocked.get("checks"),
                    "legacy_would_have_produced": True,
                },
            }
        )

        # Failed/no-patch arm must approach 0, not floor at 1 via vacuous no_hallucinated_paths.
        no_patch = score_arm(
            {
                "arm": "harness_off",
                "ok": True,
                "unified_diff": "",
                "verification_commands": [],
            },
            ["app/env_loader.py"],
            None,
        )
        results.append(
            {
                "id": "failed_arm_approaches_zero",
                "ok": (
                    no_patch["score"] == 0
                    and no_patch["score_max"] == 3
                    and no_patch["checks"].get("no_hallucinated_paths") is None
                    and no_patch["checks"].get("produced_patch") is False
                ),
                "detail": {"score": f"{no_patch['score']}/{no_patch['score_max']}", "checks": no_patch["checks"]},
            }
        )

        # Gap inside within-arm spread => inconclusive_within_noise even if on > off.
        noise_reports = [
            {"score_harness_off": 5, "score_harness_on": 6, "tasks_comparable": 3, "aborted_reason": ""},
            {"score_harness_off": 5, "score_harness_on": 3, "tasks_comparable": 3, "aborted_reason": ""},
            {"score_harness_off": 3, "score_harness_on": 6, "tasks_comparable": 3, "aborted_reason": ""},
        ]
        # ON means 5.0, OFF means ~4.333, gap ~0.667; spreads are 2 and 3 => within noise.
        off_n = _score_statistics(noise_reports, "score_harness_off")
        on_n = _score_statistics(noise_reports, "score_harness_on")
        within = max(off_n["spread"], on_n["spread"])
        gap = abs(on_n["mean"] - off_n["mean"])
        results.append(
            {
                "id": "repeat_noise_gap_rule",
                "ok": gap <= within and on_n["mean"] > off_n["mean"],
                "detail": {"off": off_n, "on": on_n, "gap": gap, "within": within},
            }
        )

        fp_before = working_tree_fingerprint(ROOT)
        # Exercise sandbox path with empty behavior (null) — must not mutate tree.
        _ = evaluate_behavior_correct("", {"command": [sys.executable, "-c", "raise SystemExit(1)"]})
        fp_after = working_tree_fingerprint(ROOT)
        results.append(
            {
                "id": "behavior_scoring_leaves_working_tree",
                "ok": fp_before == fp_after,
                "detail": {"before": fp_before[:16], "after": fp_after[:16]},
            }
        )
    finally:
        mod.run_baseline = original_baseline
        mod.run_harness = original_harness
        mod.run_benchmark = original_benchmark
        action.engineer_plan = original_plan  # type: ignore[assignment]

    passed = sum(1 for item in results if item["ok"])
    return {
        "ok": passed == len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "cases": results,
        "observed_calls": observed_calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the benchmark.")
    parser.add_argument(
        "--allow-private-source-export",
        action="store_true",
        help="Acknowledge that bounded private repository excerpts are sent to the lead-model provider.",
    )
    parser.add_argument("--lead-provider", default=os.environ.get("LEAD_PROVIDER", "Codex CLI"))
    parser.add_argument("--lead-model", default=os.environ.get("LEAD_MODEL", ""))
    parser.add_argument(
        "--only",
        default="",
        help="Run a task id, or a comma-separated list of task ids (e.g. "
        "'retrieval_broker,verification_quality') to run that subset together "
        "under real parallel dispatch instead of the whole task set.",
    )
    parser.add_argument(
        "--task-set",
        choices=sorted(TASK_SETS),
        default="cheap",
        help="cheap = 3 small files (default). hard = retrieval-asymmetry set. full = large-file set.",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=BASELINE_MAX_FILE_CHARS,
        help="Truncate each baseline file to this many characters.",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="Run each selected benchmark task this many times sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the estimated call and token cost without calling any model.",
    )
    args = parser.parse_args()

    specs = select_specs(args.task_set, args.only)
    if not specs:
        print(json.dumps({"ok": False, "error": f"No task matched --only {args.only!r}."}))
        return 2

    max_file_chars = args.max_file_chars
    if args.task_set == "hard":
        max_file_chars = ensure_selected_files_fit(specs, max_file_chars)

    if args.dry_run:
        estimate = estimate_run(specs, max_file_chars, args.repeat)
        # Prove correctness scoring does not touch the working tree.
        fp_before = working_tree_fingerprint(ROOT)
        demo_no_patch = score_arm(
            {
                "arm": "harness_on",
                "ok": True,
                "patch_status": "blocked",
                "unified_diff": (
                    "--- a/app/env_loader.py\n+++ b/app/env_loader.py\n"
                    "@@ -1 +1 @@\n"
                    "-from __future__ import annotations\n"
                    "+from __future__ import annotations\n"
                ),
                "verification_commands": ["python -m py_compile app/env_loader.py"],
            },
            ["app/env_loader.py"],
            specs[0] if specs else None,
        )
        fp_after = working_tree_fingerprint(ROOT)
        hard_audit = [
            {"id": t["id"], "selected": t["selected"], "hidden_fact": t.get("hidden_fact")}
            for t in HARD_TASKS
        ]
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "task_set": args.task_set,
                    "schema": SCHEMA,
                    **estimate,
                    "hard_task_audit": hard_audit,
                    "scoring_self_check": {
                        "blocked_nonempty_score": f"{demo_no_patch['score']}/{demo_no_patch['score_max']}",
                        "blocked_nonempty_checks": demo_no_patch["checks"],
                        "blocked_patch_nonempty": demo_no_patch.get("blocked_patch_nonempty"),
                        "working_tree_untouched": fp_before == fp_after,
                    },
                },
                indent=2,
            )
        )
        print("\nNo model was called. Remove --dry-run to execute.")
        return 0

    if not args.run:
        print(json.dumps({"ok": False, "error": "Pass --run to opt in. No work was performed."}))
        return 2
    if not args.allow_private_source_export:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Private source export was not approved. Pass --allow-private-source-export only after "
                        "explicitly authorizing bounded private repository excerpts to the lead-model provider. "
                        "No work was performed."
                    ),
                }
            )
        )
        return 2

    estimate = estimate_run(specs, max_file_chars, args.repeat)
    repeat_description = "" if args.repeat == 1 else f", {args.repeat} repetitions"
    print(
        f"Running {len(specs)} task(s) from the '{args.task_set}' set via {args.lead_provider}: "
        f"~{estimate['total_calls_est']} calls, ~{estimate['total_prompt_tokens_est']} prompt tokens{repeat_description}.\n"
    )
    report = run_benchmark(
        args.lead_provider,
        args.lead_model,
        only=args.only,
        task_set=args.task_set,
        max_file_chars=max_file_chars,
        repeat=args.repeat,
    )
    hidden_report_keys = {"tasks"}
    if args.repeat > 1:
        hidden_report_keys.add("runs")
    print(
        json.dumps({k: v for k, v in report.items() if k not in hidden_report_keys}, ensure_ascii=False, indent=2)
    )
    print(f"\nFull report: {report['path']}")
    return 0 if report["verdict"] == "harness_earns_its_complexity" else 1


if __name__ == "__main__":
    raise SystemExit(main())
