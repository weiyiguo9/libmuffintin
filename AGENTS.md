# Harness branch guidance

- This is the orphan `harness` branch of libmuffintin: tracker, ledger,
  plans, gates, decisions, evidence. Code lives on `main` in the adjacent
  worktree. Never merge or rebase the two branches.
- Start by reading `STATUS.md`, the active plan, and the tail of `ledger.md`.
  Then inspect the live `main` worktree; never infer code state from records.
- `ledger.md` is append-only. Add `evt-NNNN` entries for state changes and
  `evd-NNNN` entries for gate runs, newest last. Supersede a wrong entry with
  a later one that names it. An entry that changes a workstream's state
  carries `- state: <workstream> = <state>` (and optionally `- note:`).
- `STATUS.md` is generated. After appending to the ledger or changing a plan
  header run `python3 update-status.py` and commit the result; never edit
  `STATUS.md` by hand. `python3 update-status.py --check` exits 1 when stale.
- An accepted plan is immutable; write `plan.v(N+1).md` for a change of
  direction. `proposed` and `draft` plans may be edited in place.
- Numerical acceptance follows Stop That Digit
  (`~/.codex/skills/stop-that-digit/SKILL.md`): fix quantity, reference,
  bound, and budget in `gates.md` before the first run; a pass is closed; a
  fail gets its listed diagnostics and then a handoff. Do not sweep, tighten,
  or add checks to explain accepted digits.
- Keep records small: no new directories, templates, streams, or copies of
  files that exist on `main`. Point to logs, do not duplicate them.
- Commits: Conventional Commits 1.0.0, scope `harness`, no attribution
  trailers, repo-local Git identity.
- On MSI (`D:/projects/libmuffintin-harness`): commit only on `harness-msi`,
  use ledger IDs `evt-1001`+ and `evd-1001`+, merge `github/harness` before
  writing, regenerate `STATUS.md` through WSL Python, push `harness-msi` to
  the `github` remote. Never commit to or push `harness`; the Mac merges.
- Markdown mathematics and en-dash conventions of `main`'s `AGENTS.md` apply
  to every file here.
