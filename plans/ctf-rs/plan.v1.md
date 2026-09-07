# ctf-rs dense-first delivery plan

- Workstream ID: `ctf-rs`
- Plan version: 1
- Approval: accepted 2026-09-08 (immutable; direction changes make plan.v2)
- Supersedes: `plans/ctf-slate-binding/plan.v1.md` (frozen)
- Repository: `~/tmp/ctf-rs` on the Mac, `D:/projects/ctf-rs` on MSI; upstream
  cc4s/ctf pinned at `f69cbb46e23bc2f39cda5722ce096f56301dab4f`
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` and `docs/upstream-inventory.tsv`
  remain the per-run record and are not duplicated here; this ledger records
  milestone transitions with the ctf-rs commit SHA

## Objective

Finish the dense algebra of the port before continuing sparse automatic
planning. "Finished" is the closed list in section 3, verified by the
upstream drivers that are already ported, not by new studies.

## 1. Why dense first

- libmuffintin consumes the dense line only: distributed redistribution into
  ScaLAPACK block-cyclic layouts and the Phase 5 decompositions that already
  exist (four-type Cholesky, QR, SVD, eigh). Sparse automatic planning has no
  consumer there yet.
- The dense correctness drivers under upstream `test/` are already ported
  (all but `test_suite.cxx`, the SY/AS branches of `gemm_4D`, and the upstream
  failure in `permute_multiworld`). The missing pieces are subsystems, so the
  acceptance tools exist before the work starts.
- The sparse remainder is one self-contained layer (section 4, S1). Doing it
  after the dense close causes no rework.

## 2. State at plan time (ctf-rs `f22da3a`, 2026-09-08)

Inventory rows marked `pending`: 139, of which 61 are headers whose `.cxx`
is already partial. Pending `src/*.cxx`: 26, grouped as dense core 11
(`scaling/*` 3, `redistribution/*` 6, `contraction/ctr_comm` 1, plus
`dgtog_calc_cnt`), shared infrastructure 4, interface 12 of which three
(`term`, `idx_tensor`, `fun_term`) are the expression interface the README
declares replaced. No sparse `.cxx` is pending; every sparse file is partial.
Native Windows has compile/link evidence only; WSL 1/2/4 carries every pass.

## 3. Milestones

Common contract for every milestone:

```text
digit: Q=each listed upstream driver's own metric and tolerance
       ref=pinned upstream f69cbb46 tol=upstream, unchanged class=R
runs: WSL Ubuntu-26.04 at 1, 2, 4 ranks, once; native compile/link per batch;
      native runtime once, at D6
budget: per failing driver at most three diagnostic computations, then
        DIGIT / HANDOFF; a pass is closed
records: ctf-rs docs/validation.md section per batch; inventory rows updated;
         one evd entry here per milestone close with the ctf-rs commit SHA
```

| ID | Scope | Acceptance drivers |
|---|---|---|
| B0 | Bookkeeping before any port. (1) WIP `f22da3a` (raw sparse search cache, "unverified coverage"): run `tests/distributed_sparse_search_cache.rs` at WSL 1/2/4 once and record it, or move the commit to a branch and return `main` to `da5354b`. (2) Mark `term`, `idx_tensor`, `fun_term` rows `replaced` so the pending count is honest. | none; two evt lines |
| D1 | `scaling/{scaling,scale_tsr,strp_tsr,sym_seq_scl}`: the dense scale and strip operations, unported today | rerun the ported `scalar`, `diag_sym`, `weigh_4D`, `dft` once (class R regression); exact integer checks for strip and packed scaling taken from source semantics |
| D2 | Optimized redistribution: `dgtog_redist` with `dgtog_calc_cnt`, `dgtog_bucket`, `dgtog_redist_ror`; `glb_cyclic_reshuffle`; `nosym_transp`; `redist`; `slice`; `pad`. This is also the layout path libmuffintin needs. | `readwrite_test`, `readall_test`, `repack`, `permute_multiworld` NS, `reduce_bcast`, `subworld_gemm`; `bench_redistribution` and `bench_nosym_transp` one timing each, informational |
| D3 | Contraction close: `ctr_comm.cxx`; folded nested panels for all four scalar types (f64 only today); `gemm_4D` SY/AS branches; automatic symmetric plan integration in `ctr_tsr` and `sum_tsr` | `gemm_4D` all branches, `weigh_4D`, `sy_times_ns`, `ccsdt_t3_to_t2`, `ccsdt_map_test`, `multi_tsr_sym`, the seven `studies/fast_*` drivers |
| D4 | Infrastructure: `memcontrol` (process memory discovery, low-memory path), `int_timer`, `util`, `blas_symbs`, `flop_counter` | exact unit checks; `examples/mpi_low_memory_bench.rs` one run, informational |
| D5 | Interface surface: `partition` (Idx_Partition), `vector`, `world`, `scalar`, `set`, `monoid`, `ring`, remaining `common.h` items | `fft_with_idx_partition`, `fft`, `dft_3D`; rerun `endomorphism*`, `univar_function`, `bivar_*` |
| D6 | Dense drivers and native close: `test_suite.cxx` dense subset; examples `matmul`, `recursive_matmul`, `ccsd`, `ao_mo_transf`, `neural_network`, `bitonic_sort`, `checkpoint`, `force_integration`, `particle_interaction` with `moldynamics.h`, `qinformatics`, `mttkrp`; `bench_contraction`, `model_trainer` one timing each | each driver's upstream criterion at WSL 1/2/4; then the whole dense set once on native Windows at 1, 2, 4 ranks |
| S1 | Sparse automatic planning, after D6: folded k1 to k5 candidate assembly and selection, sparse B/C output combinations, compressed-symmetry automatic plans, search-cache integration; then sparse sum planner and communication | `block_sparse`, `checkpoint_sparse`, `force_integration_sparse`, `apsp`, `mis`, `mis2`, `btwn_central`, `algebraic_multigrid`; `test/python/test_sparse.py` semantics |

Order is by dependency: D2 feeds D3 and D6 and is the first thing
libmuffintin can use; D4 must precede the low-memory acceptance in D6.

## 4. Definition of done for the dense phase

- No `src/` inventory row that is dense-only remains `pending`; sparse rows
  keep their `partial` status untouched.
- Every D1 to D6 driver passes at WSL 1, 2, 4 ranks with its upstream metric,
  and the D6 dense set passes once on native Windows.
- ctf-rs `docs/coverage.md` phase rows 1, 2, 3, 5 no longer list a dense
  item as pending. The file is updated as a consequence of the work, not as
  planning input.

## 5. Boundaries

- No sparse implementation before D6 except B0.
- No tolerance, metric, or fixture-size change; bounded fixtures already in
  use (n=3, n=4, n=5) stay as they are.
- No gather-based replacement of a distributed algorithm and no plugin
  framework, as the ctf-rs README already requires.
- Benchmarks record one timing per driver; no sweeps.
- A native runtime failure at D6 is a HANDOFF with the driver named, not a
  reason to drop the native gate or to relabel WSL evidence.
