You are repairing an Engineer Employee plan. Return only valid JSON. Do not introduce new files, dependencies, web/API usage, backend changes, legacy brain context, or candidate lessons. Separate verified_facts from unverified_assumptions. Produce a revised_codex_prompt that only uses selected files.

Task:
{{task}}

Context manifest:
{{context_manifest}}

Selected file inspections:
{{selected_file_inspections}}

Checker report:
{{checker_report}}

Original plan:
{{original_plan}}

Required JSON fields: task_understanding, verified_facts, unverified_assumptions, repo_grounding_score, files_likely_involved, implementation_plan, risks, forbidden_changes, acceptance_tests, codex_prompt, revised_codex_prompt, self_review_checklist, post_run_review_template, rubric_self_score, limitations.
