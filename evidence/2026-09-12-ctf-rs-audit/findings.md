# ctf-rs global audit, 2026-09-12

- Revision: rustnumgum/ctf-rs `3dafffc` (origin/master), the S1 close
- Method: read-only. Mac build with rustc 1.97.1 and Homebrew Open MPI
  (`cargo check --all-targets`, `cargo clippy --all-targets`,
  `cargo doc --no-deps` under `-D warnings`), plus five parallel read-only
  reviews (upstream coverage, dead weight, ADR-0007 MPI contract, tests and
  acceptance scripts, documentation) against the pinned upstream tree copied
  from `D:/projects/ctf-upstream-f69`. Every high finding below was
  re-verified by hand on the Mac.
- Size: 37,579 lines in 143 `src/*.rs`; 36,575 lines in 241 test files;
  214 `[[test]]` targets; 191 drivers print a pass line.
- Verdict in one line: the port is complete for its declared scope and its
  numerical record is honest; the debts are one structural duplication,
  three untested execution paths, an acceptance script that cannot record a
  partial failure, and a closing commit that left five reader-facing
  documents stating the old handoffs.

Severity: H = misleads a reader or hides a failure; M = real cost or risk,
not urgent; L = hygiene.

## A. Build, lint, and repository hygiene (Mac)

| ID | Sev | Finding | Action |
|---|---|---|---|
| A1 | L | 25 rustc warnings. Real ones: `src/symmetric_reshuffle.rs:13` `plan` never used; unused imports `src/ffi/mpi.rs:9`, `src/ffi/scalapack.rs:7`, `src/symmetric_reshuffle.rs:142`; two never-read assignments `tests/upstream/bench/model_trainer.rs:259,278`. The other 19 are per-binary "never used" in shared test modules (`tests/upstream/examples/mis_common.rs`, `moldynamics.rs`, `btwn_central_kernels.rs`, `tests/upstream/test/test_suite.rs`) | delete the dead function and imports; mark the shared test modules `#![allow(dead_code)]` |
| A2 | L | 226 clippy warnings: about 90 `too_many_arguments` (source signatures; 44 sites already allowed), 33 `needless_range_loop`, 24 manual reimplementations, 4 unused imports; densest in `sparse_contract_general.rs` (14), `tensor.rs` (10), `tensor_svd.rs` (8), `sparse_2d.rs` (8) | fix the non-signature classes; keep the argument-count allows |
| A3 | L | `cargo doc` fails under `-D warnings`: 8 unresolved intra-doc links from bracketed text in doc comments (`multilinear.rs:38,145,146`, `node_aware.rs:48`, `symmetric_sy_sum.rs:124,125`, `flop_counter.rs:42`) | escape the brackets |
| A4 | L | No `.gitignore` (`.DS_Store` shows as untracked); no `rust-version` (edition 2024 implies 1.85 or newer; libmuffintin's floor is 1.89); version 0.1.0 with no tag; stale `origin/wip/sparse-search-cache` (never compiled) | add both files; set `rust-version = "1.89"` after one check on that toolchain; tag and branch cleanup are the user's call |
| A5 | ok | The crate builds and type-checks on macOS with Homebrew Open MPI; nothing in the README says so | one README line |

## B. Upstream coverage (`docs/upstream-inventory.tsv`, `docs/coverage.md`)

| ID | Sev | Finding | Action |
|---|---|---|---|
| B1 | H | Rows 53, 55, 96, 97 (`back_comp.h`, `common.h`, `world.cxx`, `world.h`) name `src/runtime.rs`, removed in R1; the code is `src/context.rs` (`Context::world` :16, `from_communicator` :25, `split`/`close` :93-105). `coverage.md` itself says "no Runtime wrapper" | point the four rows at `src/context.rs` |
| B2 | H | Rows for `interface/sparse_tensor.cxx/.h` name `src/interface` (no such module) and say pending, while `Tensor::write_scaled` (`tensor.rs:615`), `SparseTensor::write_scaled` (`sparse.rs:303`), `Tensor::read` (`tensor.rs:1278`) carry the semantics | point the rows there; state that the operator sugar is not retained |
| B3 | M | Rows 247, 248 (`bench_nosym_transp.cxx`, `bench_redistribution.cxx`) name `tests/upstream/bench/*.rs` files that do not exist; `examples/bench_*.rs` are single-size probes without timing statistics | correct the destinations; say the examples are partial |
| B4 | M | The expression-template layer (`term.cxx`, `idx_tensor.cxx`, `fun_term.cxx`) is marked "replaced", but its automatic ordering of multi-term contraction chains (`Sum_Term::estimate_time` `term.cxx:426`, `execute` :486-518) has no Rust equivalent; a chain written as successive `contract_from` calls gets no reordering | disclose in README and `coverage.md` as a scope limit |
| B5 | M | Rows 209-212, 219 (`block_sparse`, `btwn_central*`, `force_integration_sparse`) still say "native runtime remains S1c"; the S1c native table shows all three PASS at 1, 2, 4 | update the five rows |
| B6 | L | `decomposition.cxx` row says hosvd "remains pending" while the `examples/hosvd.cxx` row is ported and passing; the phase-5 multilinear headline in `coverage.md` omits the inventory's open items (scalar kernels, memory diagnostics); root `Makefile`, `configure`, `include/ctf.hpp` have no rows | align the wording; add three one-line rows |
| B7 | ok | Every upstream file in `src/`, `examples/`, `test/`, `test/python`, `bench/`, `studies/`, `scalapack_tests/` has a row; about fifteen function-level spot checks across contraction, summation, mapping, ScaLAPACK, symmetry, interface matched; the "175 MPI drivers" figure equals the script's array | none |

## C. Dead weight (`src/`)

| ID | Sev | Finding | Action |
|---|---|---|---|
| C1 | H | The sparse 2D panel state machine (comm / inner==0 / outer==1 / else, contribution accumulation) is written six times in `src/sparse_2d.rs` (`execute_csr` :296, `execute_ccsr_dense` :375, `execute_coo_dense` :468, `execute_csr_dense` :518, `execute_csr_sparse_dense` :592, `execute_pairs_dense` :668) and a seventh time in `src/sparse_contract_general.rs:918` with private copies of `sparse_2d` helpers (`custom_schedule` :867 = `schedule` :155, `custom_positions` :881 = `Panel::operand_positions` :43, `custom_csr_operand` :902, `empty_csr` :912 = `csr_empty` :173). Cause: the helpers are private. About 550 lines that one executor generic over an operand fetch and an empty-block closure would replace with about 150 | widen the helpers to `pub(crate)` and merge into one executor; class R regression |
| C2 | H | Three executors have no caller in `src/`, `tests/`, or `examples/` and no test: `contract_sparse_dense_from_selected` (`sparse_contract_general.rs:1634-1712`, the only use of `Pattern::SparseDenseSparse`), `sum_sparse_function_from_selected` and `accumulate_sparse_function_from_selected` (`sparse_functions.rs:251, 267`) | delete them and the pattern variant; re-adding needs an upstream driver that exercises the case |
| C3 | M | The Hadamard-index recursion of S1d is written three times (`sparse_contract_general.rs:1082, 1482, 1715`); about 55 lines removable through the file's own `define_mapped_contraction!` style | factor into one helper |
| C4 | M | On the unfolded sparse-A dense-B path `LabelMetadata::new` runs twice per call: in `sparse_weigh_index` (:122) and again in `contract_sparse_from_mapped` (:1139) | pass the metadata through |
| C5 | L | `int_timer.rs:137,158` and `partition.rs:61` are unreferenced small methods; 44 `#[allow(clippy::too_many_arguments)]` | leave the allows; drop the methods if no consumer appears in T1 |
| C6 | ok | No never-constructed error variants, no infallible `Result`s, no debug prints, no `#[allow(dead_code)]`; the repeated `ptr::eq(context)` assertions guard independently public entries | none |

## D. MPI contract (ADR-0007)

| ID | Sev | Finding | Action |
|---|---|---|---|
| D1 | ok | No `MPI_Init`/`MPI_Finalize` in `src/`; `Context` borrows `&'u Universe` (`context.rs:16-31`); no `impl Drop` anywhere in `src/`; `Comm::close` (`ffi/mpi.rs:292-298`) is the only free, through `ManuallyDrop` on the split variant, so a forgotten close leaks and never frees implicitly; `mpi` has default features off, no `libffi` in `Cargo.lock`, `mpi-sys` only transitive; `check_thread` (`context.rs:33-49`) checks `threading_support()` and `MPI_Is_thread_main`; `PhantomData<Rc<()>>` markers on `Comm` and `scalapack::Grid`, no `unsafe impl Send/Sync`; 198 of 198 drivers initialize, check the level, and close | none |
| D2 | M | `Context::split` (`context.rs:93`) returns `Option<Context>`; the `#[must_use]` on the struct does not fire through `Option` (verified with rustc 1.97.1: `split();` compiles silently), so a discarded split leaks a communicator without a warning | `#[must_use]` on `split` and `split_shared` |
| D3 | M | Zero `// SAFETY:` comments over about 91 `unsafe` sites (`ffi/mpi.rs` 21, `ffi/scalapack.rs` 38, `ffi/linalg.rs` 20, `memcontrol.rs` 6, others) | one line per block, starting with the count-and-datatype invariants |
| D4 | L | `redistribute_ror` (`ffi/mpi.rs:115-183`) hand-rolls `MPI_Irecv`/`MPI_Isend`/`MPI_Waitall` where rsmpi 0.8.2 offers `immediate_receive_into_with_tag`, `immediate_send_with_tag`, `RequestCollection::wait_all`; permitted by the ADR | migrate or record why not |
| D5 | ok | Remaining raw calls are legitimately raw: `MPI_Is_thread_main`, MPI-IO in `ffi/binary_io.rs` and `ffi/mpi_io.rs`, BLACS `Csys2blacs_handle` in `ffi/scalapack.rs` | none |

## E. Tests and acceptance scripts

| ID | Sev | Finding | Action |
|---|---|---|---|
| E1 | H | 17 `[[test]]` targets are in neither `scripts/acceptance-wsl.sh` nor `scripts/acceptance-native.ps1`: the thirteen S1 targets, `dgtog_redistribution`, `model_io`, and two informational ones (`upstream_bench_contraction` prints INFO; `upstream_model_trainer` needs `-write`) | add the fifteen gating targets; list the two exclusions in the scripts |
| E2 | H | Both scripts batch every MPI target into one `cargo test` per rank count (`acceptance-wsl.sh:109-112`) under `set -euo pipefail` / `$ErrorActionPreference = 'Stop'` with no `--no-fail-fast` and no per-target record; one failing binary ends the run and voids the evidence of everything after it | `--no-fail-fast`; per-target exit lines as the S1 run scripts already do |
| E3 | H | `tests/distributed_random_fill.rs:112` prints `PASS …` without the `DIGIT / ` prefix; 1 of 191 | one-line fix |
| E4 | M | The two scripts keep hand-typed lists (currently identical, 221 names); 24 auto-discovered `harness = true` files have no `[[test]]` entry; 5 of 6 `examples/*.rs` are undeclared and build only because the linalg features are default | one target manifest both scripts read, checked against `cargo metadata` |
| E5 | L | 16 foundational drivers run world only (no parity split), honestly labeled; `acceptance-native.ps1:9` hard-codes the MS-MPI path | leave; parameterize the path |
| E6 | ok | 0 `#[ignore]`, 0 tautological assertions, 0 duplicated bodies among 117 ordinary tests; all 37 sampled `DIGIT` bounds equal the code's assertions; every `Generator::new` is rank-seeded; `process::id()` only names temp files | none |

## F. Documentation

| ID | Sev | Finding | Action |
|---|---|---|---|
| F1 | H | `README.md:55-58`, `docs/native-windows.md:6-9`, `docs/sparse-output.md:46-49`, and inventory rows 121-122 (`spctr_tsr.cxx/.h`) still state the three S1 handoffs that `3dafffc` closed | propagate the S1-close wording |
| F2 | H | `docs/automatic-planning.md:13` "sparse node-aware integration remains pending" is retracted by `docs/source-node-aware-boundary.md:12-13` but never removed; `automatic-planning.md:279` still points at "source-restricted HANDOFF cases" | delete the sentence; reword the pointer |
| F3 | M | `docs/coverage.md:19` phase-4 row keeps "with bounded HANDOFF cases" after the sibling rows were rewritten | reword |
| F4 | M | `docs/validation.md` is 3,008 lines and 163 sections, not chronological: the S1 close sits at line 111 above the S1a to S1c sections it supersedes, and nothing in those sections points forward | a top index, one row per driver, naming its authoritative section |
| F5 | M | README "Running the current subset" documents only the WSL path; nothing on native Windows or on running the S1 set | add both pointers and the macOS note (A5) |
| F6 | ok | The two remaining known-failure entries are honestly scoped (one executed reference run, one labeled static); design-doc claims matched the code on spot checks | none |

## What the audit did not do

No numerical check was run or repeated; no performance measurement; no
review of the `studies/` and `scalapack_tests/` ports beyond confirming the
Rust twins exist, are in the WSL script, and carry inventory rows.
