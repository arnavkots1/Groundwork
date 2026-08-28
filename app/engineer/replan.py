"""Grounded replan: revise a plan against retrieved evidence only.

Split out of `loop` because it is the gate feeder for patching. Once retrieval has
happened, the model has seen content the original plan did not, so `propose_patch`
refuses until a replan re-derives the plan from cited evidence.
"""

from __future__ import annotations

import json
from datetime import datetime

from engineer_grounding import (
    GROUNDING_CONTRACT,
    build_evidence_catalog,
    public_evidence_index,
    validate_grounded_plan,
)

from . import artifacts, checker, config, context, jspace, models, patch as patch_mod, prompts
from .loop import emit_generated_context_request
from .retrieval import engineer_context_execute
from .util import dedupe, extract_json_object, now_tz_iso, portable_repo_path


# Rules that mean "the model correctly said it hasn't seen enough evidence yet",
# as opposed to a rule that means the model's reasoning itself is ungrounded
# (invented a claim, cited unknown evidence, left an action untraced). Only the
# former is safe to resolve by fetching more of what the model already asked for;
# the latter needs a genuinely different plan, not more context.
_CONTEXT_ONLY_FAILURE_RULES = frozenset(
    {"context_incomplete", "high_risk_context_question_unresolved", "invalid_unresolved_questions"}
)


def engineer_replan(run_id: str, *, _followup_round: int = 0) -> dict:
    """Re-plan against retrieved evidence. Requires at least one completed retrieval.

    If the replan is blocked purely because the model reported incomplete context
    (not because its reasoning was ungrounded) and it emitted a well-formed follow-up
    context request that resolves to already-authorized scope, execute that one
    request and replan once more. This uses machinery (`generatedContextRequest`,
    `engineer_context_execute`) that already existed but that no caller drove to
    completion - the model's honest "I need X" was being discarded instead of
    serviced. Bounded to exactly one extra round; anything requiring new human
    approval or touching real grounding-violation rules still fails closed exactly
    as before.
    """
    result = _engineer_replan_once(run_id)
    if _followup_round > 0:
        return result
    failed_rules = set(result.get("failedRules") or [])
    followup_request = result.get("generatedContextRequest") or {}
    # "approval_required" is included, not just "ready": a follow-up request with
    # several items (e.g. one full-file read plus three already-authorized
    # find_symbol/find_references/find_tests items) is classified approval_required
    # in aggregate as soon as any single item needs new scope, even when most items
    # are already pre-authorized. `engineer_context_execute` only ever executes the
    # items actually marked pre_authorized/approved and blocks the rest with their
    # reasons recorded - that per-item gate is the real safety boundary, not this
    # aggregate label - so attempting execution here does not bypass approval for
    # anything that actually needs it.
    eligible = (
        result.get("checkerStatus") == "fail"
        and result.get("contextSufficiency") == "incomplete"
        and failed_rules
        and failed_rules <= _CONTEXT_ONLY_FAILURE_RULES
        and followup_request.get("status") in {"ready", "approval_required"}
        and followup_request.get("request_id")
    )
    if not eligible:
        result["followup"] = {"attempted": False}
        return result
    try:
        execution = engineer_context_execute(run_id, str(followup_request.get("request_id") or ""))
    except Exception as exc:  # noqa: BLE001
        result["followup"] = {"attempted": True, "ok": False, "error": str(exc)[:900]}
        return result
    if not execution.get("ok") or not execution.get("retrieval_results"):
        result["followup"] = {
            "attempted": True,
            "ok": False,
            "status": execution.get("status"),
            "blocked_items": execution.get("blocked_items"),
        }
        return result
    second = engineer_replan(run_id, _followup_round=1)
    second["followup"] = {
        "attempted": True,
        "ok": True,
        "request_id": followup_request.get("request_id"),
        "retrieved": execution.get("retrieval_results"),
        "first_pass_checker_status": result.get("checkerStatus"),
        "first_pass_revision_id": result.get("revision_id"),
    }
    return second


def _engineer_replan_once(run_id: str) -> dict:
    config.ensure_harness()
    json_path, plan = artifacts.load_run_json(run_id)
    source_run_id = str(plan.get("run_id") or json_path.stem)
    stale_context = jspace.context_hash_mismatches(source_run_id)
    if stale_context:
        raise RuntimeError(
            "Grounded replan blocked because J-space context is stale: "
            + ", ".join(str(item.get("path") or "") for item in stale_context)
        )
    workspace, manifest_path, _ = jspace.paths(source_run_id)
    manifest = jspace.load_manifest(source_run_id)
    catalog = build_evidence_catalog(manifest, workspace, config.ROOT, task_text=str(plan.get("task") or ""))
    retrieved_entries = [item for item in catalog.get("entries") or [] if item.get("source_type") == "retrieved_excerpt"]
    if not retrieved_entries:
        raise RuntimeError("Grounded replan requires at least one completed repository retrieval excerpt.")
    map_path = workspace / "context" / "repository_map" / "repository_map.json"
    repository_map = json.loads(map_path.read_text(encoding="utf-8")) if map_path.is_file() else {}
    evidence_ids = [str(item) for item in (catalog.get("evidence_ids") or [])]
    evidence_id_list = "\n".join(f"`{item}`" for item in evidence_ids) if evidence_ids else "(none)"
    prompt = prompts.render(
        "grounded_replan",
        {
            "task": plan.get("task", ""),
            "current_plan": json.dumps(
                {
                    key: plan.get(key)
                    for key in [
                        "task_understanding",
                        "files_likely_involved",
                        "implementation_plan",
                        "risks",
                        "acceptance_tests",
                        "limitations",
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            "repository_map": json.dumps(repository_map, ensure_ascii=False, indent=2),
            "evidence_id_list": evidence_id_list,
            "evidence_entries": json.dumps(catalog.get("entries") or [], ensure_ascii=False, indent=2),
        },
    )
    route_attempts = []
    route = ""
    raw_output = ""
    selected_records = context.records_from_saved_payload(plan)
    target_files = patch_mod.patch_target_files(plan, selected_records)
    literal_candidate = patch_mod.deterministic_literal_patch(
        str(plan.get("task") or ""),
        target_files,
    )
    retrieved_for_targets = [
        item
        for item in retrieved_entries
        if str(item.get("path") or "") in set(target_files)
    ]
    if literal_candidate and len(target_files) == 1 and retrieved_for_targets:
        evidence_id = str(retrieved_for_targets[0].get("evidence_id") or "")
        route = "deterministic_literal_replacement"
        parsed = {
            key: plan.get(key)
            for key in [
                "task_understanding",
                "verified_facts",
                "unverified_assumptions",
                "repo_grounding_score",
                "files_likely_involved",
                "implementation_plan",
                "risks",
                "forbidden_changes",
                "acceptance_tests",
                "codex_prompt",
                "revised_codex_prompt",
                "self_review_checklist",
                "post_run_review_template",
                "rubric_self_score",
                "limitations",
            ]
        }
        parsed.update(
            {
                "verified_facts": [
                    *(plan.get("verified_facts") or []),
                    (
                        f"Retrieved evidence {evidence_id} records the exact current bytes "
                        f"of the sole write target {target_files[0]}."
                    ),
                ],
                "repo_grounding_score": max(3, int(plan.get("repo_grounding_score") or 0)),
                "material_claims": [
                    {
                        "claim_id": "C1",
                        "claim": (
                            f"The declared exact-literal edit is grounded in the retrieved "
                            f"current bytes of {target_files[0]}."
                        ),
                        "evidence_ids": [evidence_id],
                        "confidence": "high",
                        "influences": ["scope", "proposed_edit", "test_selection"],
                    }
                ],
                "plan_actions": [
                    {
                        "action_id": "A1",
                        "action": "Apply only the declared exact single-literal replacement.",
                        "claim_ids": ["C1"],
                        "files": target_files,
                    }
                ],
                "context_sufficiency": {
                    "status": "sufficient",
                    "known_unknowns": [],
                    "unresolved_questions": [],
                    "assumptions": [],
                },
                "context_requests": [],
            }
        )
        raw_output = json.dumps(parsed, ensure_ascii=False)
    else:
        try:
            routed = models.call_tier_with_fallback(
                "Lead Engineer",
                prompt,
                tier=models._engineer_primary_tier(),
                max_cooldown_wait_seconds=0,
                cloud_request_timeout_seconds=45,
                cloud_max_retries=0,
                max_tokens=4000,
            )
            route = models.route_of(routed)
            route_attempts = models.attempts_of(routed)
            raw_output = routed.content
            parsed = extract_json_object(routed.content)
        except Exception as exc:  # noqa: BLE001
            parsed = {}
            raw_output = str(exc)
            route_attempts = [{"provider": "Best Available", "model": "", "status": "failed", "detail": str(exc)[:1200]}]
    if not parsed:
        parsed = _unavailable_replan(plan)
    parsed["grounding_contract_version"] = GROUNDING_CONTRACT
    parsed["evidence_index"] = public_evidence_index(catalog)
    parsed["evidence_pack_truncated"] = bool(catalog.get("truncated"))
    grounding_validation = validate_grounded_plan(parsed, set(catalog.get("evidence_ids") or []))
    checker_report = checker.run_checker(parsed, selected_records, raw_output, str(plan.get("task") or ""))
    combined_failed = dedupe([*checker_report.get("failed_rules", []), *grounding_validation.get("failed_rules", [])])
    combined_warnings = dedupe([*checker_report.get("warnings", []), *grounding_validation.get("warnings", [])])
    checker_report.update(
        {
            "status": "fail" if combined_failed else "warn" if combined_warnings else "pass",
            "failed_rules": combined_failed,
            "warnings": combined_warnings,
            "grounding_validation": grounding_validation,
        }
    )
    protected = {
        "run_id",
        "task",
        "j_space",
        "context_manifest",
        "model_routes_attempted",
        "original_plan",
        "raw_output",
        "created_at",
    }
    for key, value in parsed.items():
        if key not in protected:
            plan[key] = value
    revision_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_grounded_replan"
    revision = {
        "schema": GROUNDING_CONTRACT,
        "revision_id": revision_id,
        "source_run_id": source_run_id,
        "created_at": now_tz_iso(),
        "model_route": route,
        "model_attempts": route_attempts,
        "checker_report": checker_report,
        "evidence_index": public_evidence_index(catalog),
        "evidence_pack_truncated": bool(catalog.get("truncated")),
        "context_sufficiency": parsed.get("context_sufficiency") or {},
        "material_claims": parsed.get("material_claims") or [],
        "plan_actions": parsed.get("plan_actions") or [],
        "context_requests": parsed.get("context_requests") or [],
        "plan": {key: value for key, value in parsed.items() if key not in {"evidence_index"}},
        "raw_output": raw_output,
    }
    output_dir = workspace / "context" / "replans"
    output_dir.mkdir(parents=True, exist_ok=True)
    revision_json = output_dir / f"{revision_id}.json"
    revision_md = output_dir / f"{revision_id}.md"
    revision_json.write_text(json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    revision_md.write_text(
        "# Grounded Replan\n\n```json\n" + json.dumps(revision, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    plan.setdefault("plan_revisions", []).append(
        {
            "revision_id": revision_id,
            "path": portable_repo_path(revision_md),
            "json_path": portable_repo_path(revision_json),
            "model_route": route,
            "checker_status": checker_report["status"],
        }
    )
    plan["checker_status"] = checker_report["status"]
    plan["checker_report"] = checker_report
    plan["checker_warnings"] = checker_report["warnings"]
    plan["grounded_replan_at"] = revision["created_at"]
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    json_path.with_suffix(".md").write_text(artifacts.plan_markdown(plan), encoding="utf-8")

    manifest = jspace.load_manifest(source_run_id)
    grounding = manifest.setdefault("grounding", {})
    grounding["contract"] = GROUNDING_CONTRACT
    grounding["latest_context_sufficiency"] = str((parsed.get("context_sufficiency") or {}).get("status") or "unavailable")
    grounding.setdefault("replans", []).append(
        {
            "revision_id": revision_id,
            "path": portable_repo_path(revision_md),
            "json_path": portable_repo_path(revision_json),
            "checker_status": checker_report["status"],
            "model_route": route,
            "created_at": revision["created_at"],
        }
    )
    ref = jspace.artifact_ref("grounded_replan", revision_id, revision_md, revision_json)
    manifest.setdefault("artifact_refs", []).append({**ref, "recorded_at": revision["created_at"]})
    manifest.setdefault("events", []).append(
        {
            "event": "grounded_replan_recorded",
            "stage": "plan",
            "revision_id": revision_id,
            "context_sufficiency": grounding["latest_context_sufficiency"],
            "checker_status": checker_report["status"],
            "created_at": revision["created_at"],
            "actor": "engineer_harness",
        }
    )
    jspace.write_manifest(manifest)
    j_space_fields = jspace.update_manifest(
        source_run_id,
        "checker",
        checker_report["status"],
        "blocked" if checker_report["status"] == "fail" else "planned",
        "; ".join(checker_report.get("warnings") or checker_report.get("failed_rules") or []),
        ref,
        {"grounding_contract": GROUNDING_CONTRACT, "context_sufficiency": grounding["latest_context_sufficiency"]},
    )
    return {
        "ok": checker_report["status"] != "fail",
        "run_id": source_run_id,
        "revision_id": revision_id,
        "checkerStatus": checker_report["status"],
        "failedRules": checker_report.get("failed_rules") or [],
        "contextSufficiency": grounding["latest_context_sufficiency"],
        "modelRoute": route,
        "materialClaims": parsed.get("material_claims") or [],
        "planActions": parsed.get("plan_actions") or [],
        "generatedContextRequest": emit_generated_context_request(source_run_id, parsed, label=f"replan_{revision_id}"),
        "path": str(revision_md),
        "jsonPath": str(revision_json),
        "runPath": str(json_path.with_suffix(".md")),
        "output": (
            f"Grounded replan {revision_id}: checker={checker_report['status']}; "
            f"context={grounding['latest_context_sufficiency']}; claims={len(parsed.get('material_claims') or [])}"
        ),
        **j_space_fields,
    }


def _unavailable_replan(plan: dict) -> dict:
    """Structured refusal when the replan route produced nothing parseable."""
    return {
        "task_understanding": plan.get("task_understanding", ""),
        "verified_facts": [],
        "unverified_assumptions": ["Grounded replan model route did not return valid structured output."],
        "repo_grounding_score": 1,
        "files_likely_involved": plan.get("files_likely_involved") or [],
        "implementation_plan": plan.get("implementation_plan") or [],
        "risks": plan.get("risks") or [],
        "forbidden_changes": plan.get("forbidden_changes") or [],
        "acceptance_tests": plan.get("acceptance_tests") or [],
        "codex_prompt": "",
        "revised_codex_prompt": "",
        "self_review_checklist": [],
        "post_run_review_template": "",
        "rubric_self_score": {},
        "limitations": ["Grounded replan unavailable; patch generation must remain blocked."],
        "material_claims": [],
        "plan_actions": [],
        "context_sufficiency": {
            "status": "unavailable",
            "known_unknowns": ["No valid grounded replan output."],
            "unresolved_questions": [{"question": "Can a grounded plan be generated?", "risk": "high"}],
            "assumptions": [],
        },
        "context_requests": [],
    }


def run_followup_self_tests() -> dict:
    """Zero-model-call, zero-filesystem regression suite for the bounded follow-up round.

    Monkeypatches `_engineer_replan_once` and `engineer_context_execute` (both
    module-level names in this file) so every scenario is deterministic. Exercises
    the two failure modes tonight's A/B benchmark actually hit: a replan blocked
    purely on incomplete context should get exactly one extra retrieval+replan
    round when the follow-up resolves to already-authorized scope (including the
    partially-authorized "approval_required" case), and should stay fail-closed
    exactly as before for anything else - a real grounding violation, a fully
    denied follow-up, a failed execution, or a second round that would need a
    third.
    """

    import sys

    mod = sys.modules[__name__]
    results: list[dict] = []
    original_once = mod._engineer_replan_once
    original_execute = mod.engineer_context_execute

    def _pass(checker_status: str, context_sufficiency: str, failed_rules: list[str], request_status: str | None) -> dict:
        return {
            "ok": checker_status != "fail",
            "run_id": "self_test_run",
            "revision_id": "rev",
            "checkerStatus": checker_status,
            "failedRules": failed_rules,
            "contextSufficiency": context_sufficiency,
            "generatedContextRequest": (
                {"request_id": "ctx_self_test_run", "status": request_status} if request_status else None
            ),
            "output": "stub",
        }

    def _run(name: str, once_sequence: list[dict], execute_result: dict | None, execute_raises: Exception | None = None):
        calls = {"once": 0, "execute": 0}

        def fake_once(run_id: str) -> dict:
            calls["once"] += 1
            index = min(calls["once"] - 1, len(once_sequence) - 1)
            return dict(once_sequence[index])

        def fake_execute(run_id: str, request_id: str) -> dict:
            calls["execute"] += 1
            if execute_raises is not None:
                raise execute_raises
            return dict(execute_result or {})

        mod._engineer_replan_once = fake_once
        mod.engineer_context_execute = fake_execute
        try:
            outcome = mod.engineer_replan("self_test_run")
        finally:
            mod._engineer_replan_once = original_once
            mod.engineer_context_execute = original_execute
        return outcome, calls

    try:
        # 1. Clean pass: no follow-up should even be considered.
        outcome, calls = _run(
            "pass_no_followup",
            [_pass("pass", "sufficient", [], None)],
            None,
        )
        results.append(
            {
                "id": "pass_no_followup",
                "ok": outcome["followup"] == {"attempted": False} and calls == {"once": 1, "execute": 0},
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 2. A real grounding violation mixed with a context-only rule must still
        # fail closed - more retrieval cannot fix an invented/unresolvable claim.
        outcome, calls = _run(
            "real_violation_no_followup",
            [_pass("fail", "incomplete", ["context_incomplete", "claim_with_unknown_evidence:C1"], "ready")],
            None,
        )
        results.append(
            {
                "id": "real_violation_no_followup",
                "ok": outcome["followup"] == {"attempted": False} and calls == {"once": 1, "execute": 0},
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 3. Purely context-incomplete with a fully pre-authorized follow-up: should
        # execute once and replan once more.
        outcome, calls = _run(
            "ready_triggers_followup",
            [
                _pass("fail", "incomplete", ["context_incomplete", "high_risk_context_question_unresolved"], "ready"),
                _pass("pass", "sufficient", [], None),
            ],
            {"ok": True, "retrieval_results": [{"path": "app/example.py"}], "status": "executed"},
        )
        results.append(
            {
                "id": "ready_triggers_followup",
                "ok": (
                    outcome["checkerStatus"] == "pass"
                    and outcome["followup"]["attempted"] is True
                    and outcome["followup"]["ok"] is True
                    and calls == {"once": 2, "execute": 1}
                ),
                "detail": {"checkerStatus": outcome.get("checkerStatus"), "followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 4. A follow-up request classified "approval_required" in aggregate (e.g.
        # one item needs new scope while three others are already pre-authorized)
        # must still be attempted - engineer_context_execute is the real per-item
        # safety boundary, this aggregate label is not.
        outcome, calls = _run(
            "approval_required_triggers_followup",
            [
                _pass("fail", "incomplete", ["context_incomplete"], "approval_required"),
                _pass("pass", "sufficient", [], None),
            ],
            {"ok": True, "retrieval_results": [{"path": "app/example.py"}], "status": "partially_executed"},
        )
        results.append(
            {
                "id": "approval_required_triggers_followup",
                "ok": (
                    outcome["checkerStatus"] == "pass"
                    and outcome["followup"]["attempted"] is True
                    and calls == {"once": 2, "execute": 1}
                ),
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 5. Fully denied follow-up: nothing pre-authorized anywhere, must not attempt.
        outcome, calls = _run(
            "denied_no_followup",
            [_pass("fail", "incomplete", ["context_incomplete"], "denied")],
            None,
        )
        results.append(
            {
                "id": "denied_no_followup",
                "ok": outcome["followup"] == {"attempted": False} and calls == {"once": 1, "execute": 0},
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 6. Eligible, but the retrieval itself comes back empty: must not fabricate
        # a second pass or loop - the original (still-failing) result stands.
        outcome, calls = _run(
            "execution_fails_no_second_pass",
            [_pass("fail", "incomplete", ["context_incomplete"], "ready")],
            {"ok": False, "retrieval_results": [], "status": "blocked"},
        )
        results.append(
            {
                "id": "execution_fails_no_second_pass",
                "ok": (
                    outcome["checkerStatus"] == "fail"
                    and outcome["followup"]["attempted"] is True
                    and outcome["followup"]["ok"] is False
                    and calls == {"once": 1, "execute": 1}
                ),
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )

        # 7. Bounded to exactly one extra round: if the second pass is *also*
        # blocked purely on incomplete context with a ready follow-up, there must
        # be no third round.
        outcome, calls = _run(
            "bounded_to_one_round",
            [
                _pass("fail", "incomplete", ["context_incomplete"], "ready"),
                _pass("fail", "incomplete", ["context_incomplete"], "ready"),
            ],
            {"ok": True, "retrieval_results": [{"path": "app/example.py"}], "status": "executed"},
        )
        results.append(
            {
                "id": "bounded_to_one_round",
                "ok": (
                    outcome["checkerStatus"] == "fail"
                    and outcome["followup"]["attempted"] is True
                    and calls == {"once": 2, "execute": 1}
                ),
                "detail": {"followup": outcome.get("followup"), "calls": calls},
            }
        )
    finally:
        mod._engineer_replan_once = original_once
        mod.engineer_context_execute = original_execute

    passed = sum(1 for item in results if item["ok"])
    return {"ok": passed == len(results), "passed": passed, "failed": len(results) - passed, "cases": results}


if __name__ == "__main__":
    import json as _json

    _summary = run_followup_self_tests()
    print(_json.dumps(_summary, indent=2))
    raise SystemExit(0 if _summary["ok"] else 1)
