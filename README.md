# Groundwork

Groundwork is a harness for AI coding agents that refuses to let a patch ship
unless every material claim behind it cites real, retrieved evidence from the
repository — and it proves the harness is worth its complexity with a held-out
A/B benchmark against the same lead model running with no harness at all.

Most agent harnesses assume more scaffolding is better. Groundwork doesn't
assume that. It measures it.

## The pipeline

```
human-selected files + task
  -> plan
  -> bounded repository retrieval
  -> grounded replan (every claim must cite an evidence_id, or fail closed)
  -> checked patch proposal (never applied automatically)
  -> disposable-copy verification (the real working tree is never touched)
  -> evidence report
```

Two bounded, one-shot "follow-up" rounds exist in this pipeline: if the model
says it genuinely lacks evidence (not that the change is unsafe or out of
scope), it can ask for exactly one more bounded retrieval round, at the
replan stage and at the patch-generation stage. Every other refusal — unsafe,
out of scope, ungrounded claim, unresolved high-risk blocking question — fails
closed, permanently, for that run.

## Why this design

Every claim a plan or patch makes must resolve to an `evidence_id` pulled
from actual retrieved repository content, or explicitly cite
`human_task_premise` when it's restating something the human said directly.
A claim with no citation is indistinguishable from an invented one, so it's
rejected the same way. This is enforced by `app/engineer_grounding.py` and
checked before every patch proposal.

## Does it actually help?

That's an empirical question, not an assumption — `scripts/harness_ab_benchmark.py`
runs the same task through two arms: the harness's full pipeline, and a raw
single-shot call to the same lead model with the same files. Both arms are
scored against the same rubric (produced a patch, stayed in scope, no
hallucinated paths, a verification command, the patch actually applies, and
a real behavior test where one exists — behavior tests run only inside a
disposable repo copy, the real working tree is never written). Run it
yourself:

```bash
python scripts/company_brain_action.py --help
python scripts/harness_ab_benchmark.py --dry-run --task-set cheap
python scripts/harness_ab_benchmark.py --run --allow-private-source-export --task-set cheap --repeat 3
```

`--repeat` runs the same task set multiple times and reports whether the
harness's advantage over the raw baseline exceeds the noise between repeats
— a single run proves nothing; repeated runs with a gap larger than the
within-arm spread do.

## Model routing

Groundwork calls out to whatever lead model you configure — `app/api_clients.py`
supports direct API providers (Anthropic, OpenAI) and CLI-based subscription
tools (Claude Code CLI, Codex CLI, Cursor Agent CLI) behind one interface.
Copy `config/providers.example.env`, fill in what you have, and pick a
provider with `--lead-provider`.

## Safety design

See [`docs/threat_model.md`](docs/threat_model.md) for the actual security
boundary: what's protected, what "Apply" means (it never touches your real
working tree without an explicit human step), and what's explicitly out of
scope (an agent with raw shell access on the same OS account is outside this
harness's boundary — Groundwork constrains its own dispatcher, not the
operating system).

## What's here vs. what isn't

This is the harness core: planning, retrieval, grounded replanning, patch
proposal, the safety checker, model routing, and the benchmark that proves
(or disproves) any of it. It does not include the private evidence history
this project accumulated while being built and debugged — that's specific
to one repository's history, not to the harness itself. Every claim in this
README about what the harness does is verifiable by reading the code in
`app/engineer/` and running the benchmark yourself.

## Status

This is a working harness with a real, repeated, measured advantage over
raw model use on two of three internal benchmark categories tested so far
(a gap that exceeds run-to-run noise, not a single lucky run). It is not
validated at scale across arbitrary codebases yet — that's the benchmark's
job, and you're encouraged to point it at your own tasks and see what it
finds, including where it's still wrong.
