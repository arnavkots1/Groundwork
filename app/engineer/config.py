"""Paths, constants, and the role descriptor for the Engineer harness.

Only four things are role-specific: prompt files, retrieval intents, checker rules,
and the verification mechanism. `RoleConfig` names exactly those four so a second
role is a second `RoleConfig` instance rather than a fork of the loop.

Module-level names in the INJECTABLE set below are read through the module
(`config.ENGINEER_PATCHES_DIR`, never `from config import ENGINEER_PATCHES_DIR`)
because callers rebind them to redirect artifact writes into temp fixtures.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"

RUNS_DIR = ROOT / "brain" / "runs"
OUTBOX_DIR = ROOT / "brain" / "codex_bridge" / "outbox"
BRAIN_V2_DIR = ROOT / "brain_v2"
ENGINEER_DIR = BRAIN_V2_DIR / "employees" / "engineer"

# INJECTABLE: rebound by callers to redirect artifact writes. Always read via the
# module object so a rebind is observed by the code that writes artifacts.
ENGINEER_RUNS_DIR = ENGINEER_DIR / "runs"
ENGINEER_REVIEWS_DIR = ENGINEER_DIR / "reviews"
ENGINEER_PATCHES_DIR = ENGINEER_DIR / "patches"
ENGINEER_APPLIED_PATCHES_DIR = ENGINEER_DIR / "applied_patches"
ENGINEER_VERIFICATIONS_DIR = ENGINEER_DIR / "verifications"
ENGINEER_J_SPACE_DIR = ENGINEER_DIR / "j_space"
ENGINEER_J_SPACE_TASKS_DIR = ENGINEER_J_SPACE_DIR / "tasks"
ENGINEER_ENVELOPES_DIR = ENGINEER_DIR / "envelopes"
ENGINEER_AUTONOMOUS_RUNS_DIR = ENGINEER_DIR / "autonomous_runs"
ENGINEER_BEHAVIOR_PROMOTIONS_DIR = ENGINEER_DIR / "behavior_promotions"
ENGINEER_CONSENT_EVENTS_FILE = ENGINEER_DIR / "consent_events.jsonl"
ENGINEER_BEHAVIOR_EVALUATIONS_DIR = (
    BRAIN_V2_DIR / "evals" / "engineer" / "behavior_evaluations"
)

# Prompt templates are role data, not code: they live beside the role's other policy
# files so a prompt change is reviewable without touching Python.
# `RoleConfig.prompts_dir` keeps the location overridable per role.
ENGINEER_PROMPTS_DIR = ENGINEER_DIR / "prompts"

J_SPACE_SCHEMA = "companybrain.engineer.j_space.task.v1"
J_SPACE_STAGE_ORDER = [
    "task_intake",
    "context_selection",
    "plan",
    "checker",
    "patch",
    "patch_review",
    "apply",
    "verify",
    "review",
    "feedback",
    "lesson_gate",
]

ENGINEER_PROMPT_FILE_KEYS = {
    "product_thesis",
    "operating_principles",
    "no_bias_policy",
    "harness_contract",
    "employee_profile",
    "context_policy",
    "tool_policy",
    "rubric",
    "canonical_behavior",
    "experimental_behavior",
}

# The verification mechanism: an exact command -> argv/cwd map. Nothing outside this
# map (plus generated py_compile specs for applied Python targets) is ever executed,
# so a worse model can only pick a worse member of this set, never widen it.
VERIFICATION_COMMAND_SPECS = {
    "node --check web-gui/server/company-brain-api.mjs": {
        "argv": ["node", "--check", "web-gui/server/company-brain-api.mjs"],
        "cwd": ROOT,
    },
    "cd web-gui && npm run build": {
        "argv": ["npm.cmd" if os.name == "nt" else "npm", "run", "build"],
        "cwd": ROOT / "web-gui",
    },
    "cd web-gui && npm run test:api": {
        "argv": ["npm.cmd" if os.name == "nt" else "npm", "run", "test:api"],
        "cwd": ROOT / "web-gui",
    },
    "cd dogfood/ph_withholding && npm test": {
        "argv": ["npm.cmd" if os.name == "nt" else "npm", "test"],
        "cwd": ROOT / "dogfood" / "ph_withholding",
    },
    "python scripts/smoke_test.py": {
        "argv": [sys.executable, "scripts/smoke_test.py"],
        "cwd": ROOT,
    },
}

FILES = {
    "Company OS": ROOT / "brain" / "00_company_os.md",
    "Context Index": ROOT / "brain" / "01_context_index.md",
    "Decision Log": ROOT / "brain" / "02_decision_log.md",
    "Agent Registry": ROOT / "brain" / "04_agent_registry.md",
    "Model Routing": ROOT / "brain" / "03_model_routing.md",
    "Agent Model Routing": ROOT / "brain" / "models" / "agent_model_routing.md",
    "NVIDIA Utilization": ROOT / "brain" / "model_health" / "nvidia_utilization_plan.md",
    "Project Index": ROOT / "brain" / "projects" / "index.md",
    "Source Index": ROOT / "brain" / "research" / "source_index.md",
    "Build Workflow": ROOT / "brain" / "workflows" / "build_anything.md",
    "Loop Engineering Workflow": ROOT / "brain" / "workflows" / "loop_engineering.md",
    "Codex Bridge": ROOT / "brain" / "workflows" / "codex_bridge.md",
}

ENGINEER_FILES = {
    "product_thesis": BRAIN_V2_DIR / "system" / "product_thesis.md",
    "operating_principles": BRAIN_V2_DIR / "system" / "operating_principles.md",
    "no_bias_policy": BRAIN_V2_DIR / "system" / "no_bias_policy.md",
    "harness_contract": BRAIN_V2_DIR / "system" / "harness_contract.md",
    "employee_profile": ENGINEER_DIR / "employee_profile.md",
    "context_policy": ENGINEER_DIR / "context_policy.md",
    "tool_policy": ENGINEER_DIR / "tool_policy.md",
    "rubric": ENGINEER_DIR / "rubric.md",
    "canonical_behavior": ENGINEER_DIR / "canonical_behavior.md",
    "experimental_behavior": ENGINEER_DIR / "experimental_behavior.md",
    "approved_lessons": ENGINEER_DIR / "approved_lessons.jsonl",
    "candidate_lessons": ENGINEER_DIR / "candidate_lessons.jsonl",
    "rejected_lessons": ENGINEER_DIR / "rejected_lessons.jsonl",
    "failures": ENGINEER_DIR / "failures.jsonl",
    "version_history": ENGINEER_DIR / "version_history.jsonl",
    "eval_history": ENGINEER_DIR / "eval_history.jsonl",
}

ENGINEER_TASK_STAGES = ["plan", "check", "patch"]


@dataclass(frozen=True)
class RoleConfig:
    """The complete set of role-specific surfaces.

    Everything else in `app/engineer` is role-agnostic and reads what it needs from
    here. A second role supplies a second instance; it does not copy the loop.
    """

    name: str
    prompts_dir: Path
    harness_files: dict
    prompt_file_keys: set
    verification_command_specs: dict
    checker_rules: object = None  # `checker.EngineerRuleset` by default; set lazily
    default_retrieval_intents: tuple = field(default_factory=tuple)


ENGINEER = RoleConfig(
    name="engineer",
    prompts_dir=ENGINEER_PROMPTS_DIR,
    harness_files=ENGINEER_FILES,
    prompt_file_keys=ENGINEER_PROMPT_FILE_KEYS,
    verification_command_specs=VERIFICATION_COMMAND_SPECS,
)


def ensure_harness() -> None:
    """Create every directory and append-only ledger the harness writes to."""
    for directory in [
        BRAIN_V2_DIR / "system",
        ENGINEER_DIR,
        ENGINEER_RUNS_DIR,
        ENGINEER_REVIEWS_DIR,
        ENGINEER_PATCHES_DIR,
        ENGINEER_APPLIED_PATCHES_DIR,
        ENGINEER_VERIFICATIONS_DIR,
        ENGINEER_J_SPACE_DIR,
        ENGINEER_J_SPACE_TASKS_DIR,
        ENGINEER_ENVELOPES_DIR,
        ENGINEER_ENVELOPES_DIR / "nonces",
        ENGINEER_AUTONOMOUS_RUNS_DIR,
        ENGINEER_BEHAVIOR_PROMOTIONS_DIR,
        ENGINEER_BEHAVIOR_EVALUATIONS_DIR,
        ENGINEER_J_SPACE_DIR / "evidence",
        ENGINEER_J_SPACE_DIR / "scratch",
        ENGINEER_J_SPACE_DIR / "evals",
        ENGINEER_J_SPACE_DIR / "maintenance",
        ENGINEER_J_SPACE_DIR / "archive",
        BRAIN_V2_DIR / "runs" / "engineer",
        BRAIN_V2_DIR / "evals" / "engineer",
        BRAIN_V2_DIR / "research" / "evidence_notes",
        BRAIN_V2_DIR / "artifacts",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for key in [
        "candidate_lessons",
        "rejected_lessons",
        "failures",
        "version_history",
        "eval_history",
    ]:
        ENGINEER_FILES[key].touch(exist_ok=True)
    ENGINEER_CONSENT_EVENTS_FILE.touch(exist_ok=True)
