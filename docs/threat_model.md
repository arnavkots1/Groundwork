# CompanyBrain threat model

## Scope

This document describes the security boundary of the governed Engineer loop, its
stdio MCP server, and the envelope-bound autonomous driver.

The boundary is surface-specific. CompanyBrain can restrict calls made through its
own CLI, Python API, and MCP dispatcher. It does not remove Bash, PowerShell,
filesystem, Python-import, or other tools already available to the calling agent.
An agent with arbitrary shell access under the same operating-system account is
outside the high-assurance boundary.

In this document, **Apply** means promotion of a proposed patch into the real
repository working tree. An autonomous run may materialize that patch in a newly
created disposable code checkout under the operating system's temporary directory,
outside the repository. Scratch materialization is not Apply and cannot be promoted
into the real working tree by MCP. The temporary checkout is path separation, not
an OS sandbox.

## Assets and trust boundaries

The protected assets are:

- the real repository source and configuration;
- the human-declared task envelope;
- selected and retrieved evidence, including its hashes;
- the proposed diff and evidence-to-claim-to-hunk trace;
- verification commands and their recorded outcomes;
- canonical behavior, approved lessons, governed prompt files, and behavior
  versions;
- the durable audit trail under `brain_v2/`.

Repository files, retrieved excerpts, model output, prompts supplied by a caller,
and generated diffs are untrusted data. They never become permission state merely
because they contain approval-like text.

The local operating system, CompanyBrain implementation, Python runtime, configured
model-provider executables, and verification executable are trusted dependencies.
Autonomous verification does not run repository tests or builds. Sealing a static
command shape does not make a compromised interpreter safe.

## Consent attribution

Consent is attributable, not unforgeable. Consent-bearing artifacts contain an
attribution object, and governed consent actions append
`companybrain.engineer.consent_event.v1` events. The attribution records:

- `actor`: `human`, `human_via_agent`, or `unknown`;
- `method`: `interactive_tty`, `non_interactive_or_subprocess`, or `unknown`;
- `stdin_isatty`: `true`, `false`, or `null`;
- `recorded_at`;
- `parent_process`: PID, name, executable, and availability evidence;
- `attribution_is_authentication=false`.

The actor is derived only from the TTY/console observation:

- `stdin.isatty() is True` records `actor="human"` only after
  `GetConsoleMode` also confirms a real console handle on Windows;
- `stdin.isatty() is False` records `actor="human_via_agent"`;
- an unavailable or failed observation records `actor="unknown"`.

Parent-process information is evidence, not an identity allowlist. A process name
does not prove who operated it. A same-user agent can invoke a CLI command directly
and may be able to create a pseudo-terminal. Therefore `actor="human"` means an
interactive TTY was observed; it is not cryptographic proof of a particular person.
Historical records that predate these attribution fields must not be inferred to be
direct-human consent from `--yes`, `manual_confirmation`, or an old `actor` string.

`human` and `human_via_agent` approvals remain valid governed-CLI actions and are
counted separately. `unknown` is never silently relabeled. High-assurance operation
uses an independently operated interactive terminal for envelope declaration and
final acceptance.

## Task envelope

`engineer-envelope-declare` creates one immutable
`companybrain.engineer.task_envelope.v1` artifact before a run. It is bound to one
`task_id` and records:

- the task text and its SHA-256;
- declaration time, expiry, confirmation flag, and consent attribution;
- allowed repository roots and retrieval intents;
- explicit write targets;
- retrieval call and per-call output budgets;
- exact sealed verification specifications, each marked `autonomous_safe`;
- `apply_target="disposable_checkout_only"`;
- `real_working_tree_write=false`;
- `external_retrieval=false`;
- `in_run_expansion=false`;
- the integrity mode and integrity evidence.

Runtime events are stored separately; they do not update the envelope. There is no
in-run widening. A new root, intent, target, budget, command, or expiry requires a
new task envelope and is therefore a new declaration act. Missing, malformed,
task-mismatched, expired, or integrity-invalid envelopes fail closed and the
refusal is recorded.

An envelope is also single-use for autonomy. The driver creates an exclusive
attempt record before the governed loop; any second attempt with the same
`task_id` is refused. A retry requires a newly declared envelope.

The envelope uses HMAC-SHA-256 integrity evidence. When its nonce is stored under
the repository, this detects accidental or governed-surface mutation but is not
tamper-proof against a shell-capable same-user agent. High-assurance operation
stores the nonce outside the repository and supplies its path to the trusted
process through `COMPANYBRAIN_ENVELOPE_NONCE_PATH`. The envelope artifact stores
the HMAC and canonical-payload hash, never the raw nonce. In the default
`repo_local_readable` mode, however, the raw nonce file does live under the
Engineer envelope evidence directory and is explicitly reported as repository
readable. External placement helps only when the agent cannot read that path,
environment value, or file. If the same shell-capable agent
inherits the secret or can read the external file, external placement is not a
security boundary.

## Autonomous execution boundary

`engineer-autonomous-run` and MCP `run_autonomous` load an existing envelope. The
only authority-bearing MCP input is the envelope-bound `task_id`; the generic
optional `role` must resolve to the sole configured value, `engineer`. MCP cannot
supply or alter roots, targets, intents, budgets, verification commands, expiry,
attribution, or nonce material.

An autonomous run performs planning, bounded context selection and retrieval,
grounded replanning, patch proposal, disposable-copy application, verification,
and reporting. Every source mutation and verification command runs in the
disposable copy. The existing shared applier is still used, with the scratch root
as its explicit authority. The real source working tree is never the autonomous
applier target.

Only the code checkout is disposable. The real repository intentionally
accumulates the run's durable state under `brain_v2/`, including plans, J-space
provenance, patches, sandbox Apply records, verifications, reviews, reports,
evaluation history, lesson evidence, version history, and value evidence. Existing
durable evidence may not be deleted or rebuilt: new artifacts and
prefix-preserving JSONL appends are additive; any removal or other replacement
marks the run unsafe.
Model-capable stages may also update the fixed shared provider cooldown artifact at
`brain/model_health/provider_rate_limit_state.json`. A protected-source hash must
therefore publish its exact included surface and exclusions. Equality of before and
after hashes proves equal end-state bytes for that declared surface; it does not
prove that no transient write occurred between the two observations.

Because the temporary checkout is not an OS sandbox, autonomous verification is
restricted to exact envelope-sealed specifications carrying
`autonomous_safe=true`. Every sealed specification must execute once in declaration
order. The current safe set is only isolated
`python -I -S -m py_compile <declared-target.py>`; `py_compile` compiles the target
source without importing or executing it. Each write target must be covered by
exactly one sealed compile specification. The resulting review may record compile
success but must leave test status unknown. Repository tests, builds, scripts, and
other code-executing checks are excluded from autonomy and remain human-run.

The disposable verification process receives a reduced environment containing only
basic operating-system and temporary-directory variables. Its bounded stdout and
stderr are hashed and checked for known secret patterns. On a match, readable text
and base64 are withheld and verification is marked erroneous. Stdin and inherited
file descriptors are closed, and the child has a fixed 60-second timeout. The
scanner is heuristic and the reduced environment is not a process sandbox.

A successful autonomous report uses
`status="verified_pending_human_acceptance"` and
`promotion_status="pending_human_acceptance"`; refused and unsafe reports are not
promotion-eligible. MCP cannot promote any report. Final Apply remains the separate
governed `engineer-apply-patch` action, which revalidates source hashes immediately
before writing.

## What is defended

Within the governed surfaces and trust assumptions above:

- **Unattended escalation:** each autonomous step must be a subset of one valid,
  immutable, unexpired, task-bound envelope. Out-of-envelope requests fail closed.
- **MCP-surface bypass:** the explicit MCP allowlist rejects unknown tools,
  escalation arguments, envelope declaration or mutation, production Apply,
  behavior promotion, and equivalent aliases.
- **Injected instructions:** repository text is flagged and treated as untrusted
  evidence. It cannot directly change code-held permissions, envelope fields, or
  checkpoint state.
- **Secret disclosure:** known secret-shaped paths and content are denied before
  they are returned in MCP/model context or persisted in a readable report;
  verification output matching the scanner is withheld.
- **Stale-source writes:** selected and retrieved hashes are revalidated before
  scratch materialization and again before final Apply.
- **Out-of-scope edits:** patch paths must be explicit envelope write targets and
  pass the existing forbidden-path and applicability checks.
- **Silent harness side effects:** harness-directed retrievals, refusals, scratch
  writes, verification commands, exit codes, artifact changes, and declared
  protected-source end-state hashes are recorded.
- **Durable evidence replacement:** evidence deletion, non-prefix JSONL rewrites,
  and other rebuilds are detected and make the autonomous result unsafe.

## What is not defended

CompanyBrain does not claim to defend against:

- an agent with arbitrary shell access invoking the human CLI directly;
- a same-user process editing unsigned repository artifacts or simulating a TTY;
- administrator, kernel, runtime, dependency, provider-binary, or test-binary
  compromise;
- every possible secret format or covert disclosure channel—secret detection is
  heuristic and conservative, not complete data-loss prevention;
- prompt injection influencing the quality or semantics of a model proposal;
- a malicious but in-scope patch whose defect is not caught by deterministic
  checks or verification;
- transient writes that are reverted before the after-hash;
- unobserved side effects internal to trusted provider or verification executables;
- process containment, because the external temporary checkout is not an OS
  sandbox;
- production-grade concurrency or filesystem transaction isolation;
- semantic correctness merely because verification commands exited zero.

The report lets a third party reconstruct and inspect the recorded envelope,
evidence, diff, command execution, refusals, and end-state hashes without rerunning
the task. It does not independently authenticate the human, prove the model's
reasoning, or prove that the patch is correct.

## Behavior changes are never autonomous

Autonomous and MCP runs may create candidate lessons, but they cannot approve,
activate, version, roll back, or directly edit governed behavior. No envelope
authorizes a behavior change.

Behavior promotion is a separate consent-attributed CLI workflow with these
mandatory properties. An interactive invocation records `actor="human"`; a
noninteractive `--yes` remains valid but records `actor="human_via_agent"`:

1. At most one promoted behavior change may be pending validation.
2. Held-out non-regression evidence, produced from tasks not used to derive the
   candidate, is required before activation.
3. An exact prior byte snapshot is retained for the governed rollback set—every
   file under the Engineer `prompts/` directory, `canonical_behavior.md`, and
   `approved_lessons.jsonl`—with pre/post aggregate digests and the post-promotion
   approved-ledger hash.
4. Rollback restores the exact prior bytes and appends a durable rollback record.
5. A second promotion is refused while one remains unvalidated.
6. Promote, validate, and rollback transitions are serialized across local
   processes; a missing or mismatched active pointer fails closed.

Pre- and post-promotion evidence must be an evaluator artifact rooted at
`brain_v2/evals/engineer/behavior_evaluations/<evaluation_run_id>/evaluation.json`.
It declares `held_out=true`, `status="complete"`, producer name, version,
evaluation run, generation time, phase, and candidate identity. Each held-out row
hash-binds a task artifact and paired baseline/candidate output artifacts; the
harness recomputes the task-set digest, rejects source-task overlap, and checks
recorded aggregate non-regression. Post-promotion validation must bind the
identical task-definition digest as the pre-promotion evidence; matching IDs alone
are insufficient.

This validates artifact bytes, bindings, and recorded provenance, not the truth of
the evaluation. The producer identity is not authenticated, and CompanyBrain
cannot prove that the evaluator chose genuinely independent tasks, produced honest
outputs, or assigned correct scores.

The current controlled promotion surface handles evaluated candidate lessons only.
The exact governed rollback set is the Engineer prompt directory,
`canonical_behavior.md`, and `approved_lessons.jsonl`. Ordinary and autonomous
Apply reject all `brain_v2/employees/**/prompts/**` plus canonical-behavior and
approved-lesson targets. Ordinary Apply also rejects the behavior-promotion state
and behavior-evaluation evidence roots so a patch cannot forge the gate's state or
inputs. Those two roots remain in the autonomous protected source/behavior hash
rather than the writable evidence surface. There is no generic prompt or
canonical-behavior editor.

An iteration count, lesson count, or version label is not evidence that behavior
improved.

## High-assurance operating recommendation

For high-assurance use:

1. Give the coding agent only the CompanyBrain MCP surface, not repository shell or
   direct filesystem access.
2. Declare the envelope out of band from a separate trusted interactive terminal.
3. Keep the external HMAC nonce outside the repository with restrictive OS access,
   and do not expose its path or value to the agent process.
4. Pin and review provider and verification executables; use an OS sandbox or
   container when executable side effects are a concern.
5. Review the self-contained evidence bundle and diff before final Apply.
6. Apply only while the final source hashes still match the report.
