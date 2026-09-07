# ADR-0001: Separate code and harness history

- Status: accepted
- Date: 2026-09-08

## Context

Plans, gate definitions, decisions, and run evidence for libmuffintin
accumulated in the git-excluded `scratch/` directory of the `main` checkout:
four version plans, two design-decision notes, three Python fixture studies
with results, an experiment log, and a rank-probe artifact record, none of
them versioned. The user asked for the `graft-rs` model, an orphan `harness`
branch holding a tracker and a ledger, while avoiding two failure modes seen
there: file sprawl (JSONL streams, per-run report directories, templates,
fixture copies) and over-verification (hashes of everything, toolchain
strings, ten-thousand-case property counts).

## Decision

Keep code and records on unrelated Git histories in the same repository.
`main` owns code and source-owned documentation. `harness` owns `STATUS.md`,
`ledger.md`, `gates.md`, `plans/`, `decisions/`, and `evidence/`. Bind every
evidence entry to an exact `main` revision, an exact command, and a log
pointer. Numerical acceptance follows Stop That Digit. The record set is the
one in `README.md` and does not grow without a decision.

## Consequences

- The branches are never merged or rebased; hosts show two root commits.
- `scratch/` on `main` remains local and git-excluded for bulky external
  material only; plans and evidence no longer live there.
- Result logs produced by examples stay on `main` next to the example; the
  ledger points to them instead of copying them.
- Moving a fixture from `evidence/` into a source test is an explicit commit
  on `main`; harness history alone never changes code.
