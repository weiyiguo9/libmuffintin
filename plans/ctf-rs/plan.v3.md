# ctf-rs C1 and S1 delivery plan

- Workstream ID: `ctf-rs`
- Plan version: 3
- Approval: accepted 2026-09-11 (immutable; direction changes make plan.v4); executor Codex on MSI, `D:/projects/ctf-rs`
- Supersedes: `plans/ctf-rs/plan.v2.md` (R1 closed under it, evd-1013; the dgtog_redistribution replica contract closed at evd-1014). S1 is carried over from `plans/ctf-rs/plan.v1.md` section 3 with its driver list unchanged and split into ordered batches here.
- Repository: rustnumgum/ctf-rs, `origin/master` at `b9990fa`; the MSI checkout `D:/projects/ctf-rs` executes and is clean at that SHA; the Mac clone `~/tmp/ctf-rs` is in sync and does the publishing
- Upstream: cc4s/ctf `f69cbb46`, checked out on MSI at `D:/projects/ctf-upstream-f69` (Windows) and built as the WSL reference under `/home/xylxp/ctf-rs-reference`
- Decision: ADR-0007 stands; no new ADR
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` section per batch; one evd entry here per milestone close with the ctf-rs commit SHA

## Objective

Close the port's own remaining scope. C1 is the closed list of items that
`docs/coverage.md` and `docs/upstream-inventory.tsv` mark pending and that no
earlier milestone named. S1 is sparse automatic planning, unchanged from
plan.v1 section 3, on the rsmpi entry. After both, the README scope (a CPU/MPI
port of the pinned reference without CUDA, Python, or C++ compatibility) is
complete; muffin-tin integration is a separate workstream and not part of
this plan.

## 1. State at plan time (ctf-rs `b9990fa`, 2026-09-11)

- R1 closed: 175 WSL drivers at 1, 2, 4 ranks, the native build, and the D6
  native set pass on the rsmpi entry (evd-1013).
- Sparse: every sparse `src/*.cxx` inventory row is `partial`; none of the
  eight S1 acceptance drivers exists under `tests/` or `examples/`;
  `test/python/test_sparse.py` is `pending`. The 2026-09-08 WIP on
  `wip/sparse-search-cache` (115 lines) failed to compile at B0 (missing
  `sgemm_`, `cgemm_`, `zgemm_` declarations; `f64` node counts into `usize`
  fields) and is reference material, not a base.
- Already ported on the sparse side and gated at WSL 1, 2, 4: `speye`,
  `sptensor_sum`, `endomorphism_cust_sp`, the `upstream_sparse_mp3` pair, and
  about forty `tests/*sparse*` drivers written against the Rust API.
- Dense-side rows still `partial` with a named pending item: `shared/model`
  (model I/O), `symmetry/symmetrization` (tensor symmetrize/desymmetrize),
  `contraction/sym_seq_ctr` (custom-function folded BLAS specialization),
  `summation/sym_seq_sum` (optimized folded/custom-function branches),
  `interface/graph_io_aux` (native runtime), and the coverage phase 3 note
  that the selector allgather is an assertion stub.

## 2. Common contract

```text
digit: Q=each listed upstream driver's own metric and tolerance
       ref=pinned upstream f69cbb46 tol=upstream, unchanged class=R (A where a step says so)
runs: WSL Ubuntu-26.04 at 1, 2, 4 ranks, once; native compile/link per batch;
      native runtime once, at the C1 close and at the S1 close
budget: per failing driver at most the three diagnostics in section 5, then
        DIGIT / HANDOFF; a pass is closed
records: ctf-rs docs/validation.md section per batch; coverage and inventory
         rows updated as a consequence of the work; one evd entry here per
         milestone close with the ctf-rs commit SHA
```

A driver whose upstream expectation is itself wrong is recorded in
`docs/upstream-known-failures.md` as today and stays unaccepted; no tolerance
or semantic change hides it.

## 3. Milestones, in order

| ID | Scope | Acceptance |
|---|---|---|
| C1 | The closed list in section 4 | section 4 per item, then the full `scripts/acceptance-wsl.sh` at 1, 2, 4 once, `scripts/acceptance-native.ps1 -BuildOnly` once, and the C1.5 native run |
| S1a | Sparse contraction planning: folded k1 to k5 candidate assembly and selection on the twelve CPU leaf models already ported in `spctr_tsr`; sparse B and C output combinations including sparse–sparse–sparse; raw sparse search-cache integration in `contraction`/`sparse_search` | `block_sparse`, `apsp`, `algebraic_multigrid`; `test_sparse.py` semantics `test_einsum_hadamard`, `test_scaled_expression`, `test_complex` |
| S1b | Compressed-symmetry automatic sparse plans; folded custom-function sparse kernels (`sp_seq_ctr` "folded custom functions" and "other general sparse combinations") | `force_integration_sparse` (with `moldynamics.h`), `btwn_central` with `btwn_central_kernels`; `test_sparse.py` semantics `test_sparse_SY` |
| S1c | Sparse summation planner and communication (`spsum_tsr`, `spr_seq_sum` optimized mapped kernels and custom-function paths); optimized sparse read/write communication kernels (`sparse_rw`); sparse tensor persistence (`write_sparse_to_file`, `read_sparse_from_file`) | `checkpoint_sparse`, `mis`, `mis2`; `test_sparse.py` semantics `test_sample`; then the whole S1 set once on native Windows at 1, 2, 4 ranks |

Rules for the S1 batches:

- A driver listed under one batch that turns out to need a later batch's
  feature moves to that batch with an evt line; it is not dropped. The S1
  close needs all eight drivers and all five Python semantics.
- Each Python semantic is one Rust test with the same shapes, sparsity, and
  expression as the Python body; Q is that test's own comparison rule
  (`numpy.allclose` defaults, rtol 1e-5 and atol 1e-8) against the dense
  computation of the same expression in Rust. No Python API is built.
- S1a is the first sparse implementation; no sparse planning code lands
  under C1.

## 4. C1 closed list

| Item | Work | Acceptance |
|---|---|---|
| C1.1 | Model coefficient I/O onto the context-owned registry: `write_all_models`, `load_all_models`, `dump_all_models`, `print_all_models` (upstream `shared/model.h:27-31`) in the source file format; the `CTF_MODEL_FILE` read in `world.cxx:271` becomes an explicit load call on the registry, not an environment read | class A: write then load returns every registered model's coefficients; Q = coefficients, ref = the in-memory values before the write, tol = the printed precision of the source format, stated in the commit body before the run; plus the ported `model_trainer` once with a write and a load, one timing, informational |
| C1.2 | Tensor `symmetrize`/`desymmetrize` on the contraction path (upstream `contraction.cxx:5186-5201`, `symmetrization.h:16-25`) | class R rerun inside the C1 full WSL run of `gemm_4D` all branches, `sy_times_ns`, `multi_tsr_sym`, `diag_sym`, `weigh_4D` |
| C1.3 | Dense custom-function folded BLAS specialization in `sym_seq_ctr`; optimized folded and custom-function branches in `sym_seq_sum` | class R rerun inside the C1 full WSL run of `endomorphism*`, `univar_function`, `bivar_function`, `bivar_transform`, `ccsdt_t3_to_t2` |
| C1.4 | The coverage phase 3 note "selector allgather is an assertion stub": implement the source agreement (`contraction.cxx:2993-2997` and `:3164-3168`, `Bcast` of the best rank, topology, time, memory; `:3130` `Allreduce` of the valid-mapping count) | class R rerun inside the C1 full WSL run of `gemm_4D`, `ccsdt_map_test`, `subworld_gemm`, `permute_multiworld` NS |
| C1.5 | `graph_io_aux` native runtime | the drivers that gate `src/sparse_text.rs` (`tests/sparse_text_io.rs`, `tests/distributed_sparse_io.rs`) once each on native Windows at 1, 2, 4 ranks; Q = their own assertions, class R |
| C1.6 | Bookkeeping: README "Running the current subset" no longer says the WSL script covers only the foundation and local-linalg subset (R1 ran 175 drivers through it); every sparse `.h` inventory row takes the status of its `.cxx`; coverage rows for C1 items updated | none; recorded in the C1 evd entry |

Documentation close: C1.2 and C1.4 may close without code only if the
closing note in `docs/validation.md` names, for every upstream function in
the item, the Rust function that already carries its semantics and one
existing driver at WSL 1, 2, 4 that exercises it. Anything less is an
implementation.

Outside C1, by name, so they are not rediscovered: multilinear sparse-path
optimization (performance, no driver), `interface/matrix` and
`shared/lapack_symbs` "remaining routine families" (no driver demands them),
`interface/decomposition` HoSVD (upstream itself incomplete), the faer
replaceability requirement (a constraint, not work), CUDA/offload, Python/C++
API compatibility, BG/Q topology discovery, `mapping` "repeated-label
candidate generation" and `topology` "full responsibility audit" (they close
only through an S1 driver that needs them).

## 5. Diagnostics

At most three per failing driver, each named for the two explanations it
separates:

1. Rank-count split: run the failing driver at 1 rank; a pass there against
   a failure at 2 or 4 separates a local kernel or planning error from a
   communication or layout error.
2. Dense twin: run the same expression through the dense path on the same
   fixture at the failing rank count; agreement with the dense path
   separates a sparse storage, key, or padding error from a wrong reference
   expectation.
3. Candidate pin: force the planner to the source's selected candidate for
   that contraction (the source prints it under its verbosity flag) and run
   once; agreement separates a wrong candidate choice from a wrong kernel.

## 6. Boundaries

- No tolerance, metric, fixture, or driver change; bounded fixtures stay as
  they are.
- No gather-based replacement of a distributed algorithm and no plugin
  framework; no implicit collective on `Drop`; ctf-rs never initializes or
  finalizes MPI (plan.v2 section 2 stands in full).
- No sparse planning code before C1 closes; no dense redesign under S1.
- Benchmarks record one timing per driver; no sweeps.
- libmuffintin and the fftw fork are not edited. ctf-rs is not pushed from
  MSI; the Mac publishes.
