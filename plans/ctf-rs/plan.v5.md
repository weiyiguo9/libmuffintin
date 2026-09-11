# ctf-rs audit remediation plan

- Workstream ID: `ctf-rs`
- Plan version: 5
- Approval: proposed 2026-09-12, not accepted; executor Codex on MSI, `D:/projects/ctf-rs`, if accepted
- Supersedes: nothing. plan.v3 and plan.v4 are closed (evd-1015 to evd-1019). This plan carries the closed list of the audit `evidence/2026-09-12-ctf-rs-audit/findings.md` (evt-0073) and nothing else.
- Repository: rustnumgum/ctf-rs, `origin/master` at `3dafffc`
- Upstream: cc4s/ctf `f69cbb46`, unchanged
- Decision: ADR-0007 stands; no new ADR
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` section "A1 audit remediation"; one evd entry here with the ctf-rs SHA

## Objective

Remove the debts the audit found without changing any numerical result:
make the acceptance scripts able to record a partial failure and cover
every gating target, delete the untested executors, merge the duplicated
panel executors, and bring the reader-facing documents to the S1-close
state. No new capability, no tolerance change, no fixture change.

## 1. Contract

```text
digit: Q=every driver's own metric, unchanged ref=pinned f69cbb46 tol=unchanged class=R
runs: the full scripts/acceptance-wsl.sh (with the added targets) once at 1, 2, 4 on
      the final tree; scripts/acceptance-native.ps1 -BuildOnly once; the thirteen S1
      targets once on native at 1, 2, 4 through the revised native script
budget: a driver whose result changes is a FAIL of the refactor, not a diagnostic
        case; at most the rank split and the dense twin of plan.v3 section 5, then
        DIGIT / HANDOFF
records: docs/validation.md "A1 audit remediation"; one evd entry here
```

## 2. Work, in order

| ID | Work | Audit IDs | Acceptance |
|---|---|---|---|
| A1.1 | Acceptance scripts: add the thirteen S1 targets, `dgtog_redistribution`, `model_io` to both scripts; list `upstream_bench_contraction` and `upstream_model_trainer` as deliberate exclusions; `cargo test --no-fail-fast` with a per-target `RUN_EXIT` line in the pattern of the S1 run scripts; one target manifest file both scripts read, with a check against `cargo metadata` that every `[[test]]` and every auto-discovered test file is either listed or excluded by name; `[[example]]` entries for the five undeclared examples; the MS-MPI path as a parameter | E1, E2, E4, E5 | the manifest check passes; the full WSL run records one `RUN_EXIT` per target and rank |
| A1.2 | One-line fixes: the `DIGIT / ` prefix in `tests/distributed_random_fill.rs:112`; `#[must_use]` on `Context::split` and `split_shared`; delete `symmetric_reshuffle::plan` and the three unused imports; escape the eight rustdoc links; `.gitignore`; `rust-version = "1.89"` after one `cargo check` on that toolchain, or the lowest toolchain that passes, recorded; `#![allow(dead_code)]` on the shared test modules | E3, D2, A1, A3, A4 | `cargo check --all-targets` warning-free; `cargo doc --no-deps` clean under `-D warnings` |
| A1.3 | Delete the three uncalled executors `contract_sparse_dense_from_selected`, `sum_sparse_function_from_selected`, `accumulate_sparse_function_from_selected` and the `Pattern::SparseDenseSparse` variant with everything only they use; the commit body names the audit ID and states that re-adding needs an upstream driver | C2 | builds; class R run at the close |
| A1.4 | Merge the seven panel executors into one generic executor over an operand-fetch closure and an empty-block closure, widening the `sparse_2d` helpers to `pub(crate)`; factor the Hadamard recursion into one helper; pass `LabelMetadata` through instead of rebuilding it. One commit per item | C1, C3, C4 | class R run at the close; line count of `src/sparse_2d.rs` plus `src/sparse_contract_general.rs` reported before and after, informational |
| A1.5 | `// SAFETY:` comment on every `unsafe` block, count-and-datatype invariants first; a comment on `redistribute_ror` stating why the raw non-blocking calls stay, or the migration to the safe rsmpi calls if it is a drop-in | D3, D4 | review only |
| A1.6 | Documents to the S1-close state: the five stale surfaces (F1), the retracted sentence and the pointer (F2), the phase-4 row (F3), the inventory rows (B1, B2, B3, B5, B6), the README scope note on multi-term chain ordering (B4), the README native and S1-set pointers and the macOS line (F5, A5), and a top index in `docs/validation.md` naming the authoritative section per driver without editing any existing section (F4) | B, F | review only; the audit's stale lines no longer grep |
| Close | The section 1 runs on the final tree | all | `G-CTF-A1` |

## 3. Boundaries

- No numerical, tolerance, fixture, seed, or expression change; a changed
  driver result fails the refactor.
- No new capability: the SDS executor is deleted, not wired; no dense X
  path for the Hadamard rewrite; no planner or kernel redesign beyond the
  named merges.
- History in `docs/validation.md` is indexed, never rewritten.
- No ctf-rs push from MSI; the Mac publishes. libmuffintin and the fftw
  fork are not edited.
