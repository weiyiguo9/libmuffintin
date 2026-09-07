# libmuffintin harness

This orphan branch is the tracker and ledger for the `main` branch of
libmuffintin. It follows the code/harness separation of the sibling
`graft-rs` repository with a deliberately smaller record set.

- `main` owns code, tests, fixtures, the numbered derivations under `doc/`,
  example READMEs, and the result logs next to the examples that produce
  them.
- `harness` owns intent and state: plans, the tracker, the append-only
  ledger, the gate registry, decisions, and scratch evidence that has no home
  on `main`.
- Nothing here is runtime content. `main` never depends on this branch and
  this branch never carries code.

## Branch rules

`harness` has no merge base with `main`. Never merge or rebase one onto the
other. Use adjacent worktrees when both are needed:

```sh
git worktree add ../libmuffintin-harness harness   # from a main checkout
git worktree add ../libmuffintin main              # from a harness checkout
```

Commits follow Conventional Commits 1.0.0 with scope `harness`
(`docs(harness): …` for records, `test(harness): …` for evidence scripts)
and the same repo-local Git identity as `main`.

## Layout

```text
STATUS.md         tracker, generated: one row per workstream (navigation, not authority)
update-status.py  regenerates STATUS.md; run before every harness commit
ledger.md         append-only events and evidence, newest last
gates.md          acceptance gates: quantity, reference, bound, where enforced
plans/<ws>/       plan.vN.md; an accepted version is immutable, changes make v(N+1)
decisions/        ADR-NNNN-<slug>.md
evidence/         <date>-<topic>/ scripts, logs, and results with no home on main
```

A ledger entry that changes a workstream's state carries
`- state: <workstream> = <state>` and optionally `- note: <workstream> = <text>`;
`update-status.py` renders `STATUS.md` from the last such lines, the plan
headers, and the `main` tip. `STATUS.md` is never edited by hand.

## Authority order

1. Live `main` source, tests, and fresh output.
2. The accepted plan for scoped intent.
3. `ledger.md` as the current view.
4. `STATUS.md` as navigation only.

## Record discipline

This is what keeps the branch small.

- One ledger with two entry kinds, `evt` (state change) and `evd` (gate run).
  An entry is one heading line and a few lines of body. There are no JSONL
  streams, per-run report directories, templates, or fixture copies.
- An evidence entry is a Stop That Digit stamp (quantity, reference, bound,
  measured difference, verdict), the code revision, the exact command, and a
  pointer to the log that produced it. Logs stay where they are produced:
  `examples/*/results/` on `main`, or `evidence/` here when `main` has no
  place for them.
- Not recorded: hashes of every file, toolchain strings unless the result
  depends on them, reruns of passing checks, property-test case counts,
  resource statistics. A pass is closed. A fail gets the diagnostics its plan
  lists, then a handoff.
- Bounds are fixed in `gates.md` before the first run and are not revised
  after a result is seen. A different bound is a new gate ID with an event.
- Supersede, never edit. A wrong entry is answered by a later entry that
  names it.
- A plan is one file. A plan directory holds `plan.vN.md` and at most the
  reference data the plan needs.

## Not tracked here

Bulky external material stays in the git-excluded `scratch/` directory of
the `main` checkout: the Savrasov and Stuttgart LMTO archives, arXiv sources
(2012.04992, 2510.20826), and generated figures. A plan that uses such
material names it by URL or arXiv identifier.

## Start of every task

Read `STATUS.md`, the active plan, and the tail of `ledger.md`, then inspect
the live `main` worktree. Append an event when a workstream changes state and
an evidence entry when a gate is run. Stop That Digit
(`~/.codex/skills/stop-that-digit/SKILL.md`, summarized in `gates.md`)
governs every numerical acceptance.
