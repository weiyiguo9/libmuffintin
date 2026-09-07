# Ledger

Append-only, newest last. `evt` entries record state changes, `evd` entries
record gate runs in the form given in `gates.md`. A wrong entry is
superseded by a later entry that names it; nothing is edited or deleted.

## 2026-09-08 · evt-0001 · harness initialized · actor: claude

Orphan `harness` branch created from the `graft-rs` model with the reduced
record set of `README.md` and ADR-0001. No gate is claimed by this entry.

## 2026-09-08 · evt-0002 · scratch plans and decisions imported · actor: claude

Imported from the git-excluded `scratch/` of the `main` checkout, bodies
unchanged, each with a header naming its workstream, approval, and source:
`v0.1-lapw-foundation` (accepted, historical, closed), `v0.2-isdf-thc`
(accepted; M-A to M-Kc closed on `main`, M-L open), `v0.3-mto-family`
(draft), `v0.4-emto-nmto` (draft), `ctf-slate-binding` (draft), `h2-hf`
(proposed). Design notes imported as ADR-0002 (orbital configuration V2,
closed 2026-08-24), ADR-0003 (MTO density, Poisson, and kinetic axes,
superseded 2026-08-27 by the v0.3 and v0.4 plans), and ADR-0004
(relativistic core–valence exchange, recorded 2026-08-28). Scratch evidence
imported under `evidence/` with its original dates. Approval states are the
ones the source files carried; no plan was accepted by this import.

## 2026-09-07 · evd-0001 · h2-lda G-H2-LDA-1 G-H2-LDA-2 · main a1fdb87 (XC change), recorded through 9c870eb860f07e27abd34b761b90154450cd4adc

```text
DIGIT / PASS
Q: HOMO eigenvalue, total energy (Ha); class: A; ref: examples/h2_dft/periodic-reference.json (10 Bohr box)
bound: 1e-3 / 2e-2; Delta: 1.71e-4 / 8.12e-4; d: 0.17 / 0.04
command: cargo run --release -p libmuffintin-runtime --features fft-fftw --example h2_dft -- <out> 10 6 18 137035.9895 78
log: examples/h2_dft/results/box10-field18-grid78-step.log (main)
scope: box 10, orbital cutoff 6, field cutoff 18, truncated-step interstitial XC. The
0.8 mHa total residual was decomposed on main (examples/h2_dft/README.md): orbital
cutoff 6 to 7 removes 0.74 mHa, field cutoff is converged to 0.1 mHa, PySCF-side
errors are below 0.02 mHa. No HF or Kr claim.
```

## 2026-09-08 · evt-0003 · h2-hf plan.v1 proposed · actor: claude

`plans/h2-hf/plan.v1.md` written as Stop That Digit contracts (steps A0,
A1, A1v, A2, B, Bv; gates G-H2-HF-0 to G-H2-HF-5). Reference data in
`plans/h2-hf/pyscf-rhf-references.json`. Awaiting the user's acceptance
event before Codex starts; no run has been made.

## 2026-09-08 · evt-0004 · kr-hf registered as open · actor: claude

Workstream registered without a plan on record. State as reported on
2026-09-05 and not re-verified here: converged frozen-SRA Kr total
−2787.682 Ha against GTO 4c-DC-HF −2788.884 Ha; the VV exchange sector is
0.82 Ha too weak, CV 0.12 Ha too strong, CC matches after the smoothed
Spencer–Alavi kernel change; every run used box 8 and omega 0.8. The `h2-hf`
workstream is the sector-targeted test for this gap.

## 2026-09-08 · evt-0005 · ctf-slate-binding superseded · actor: user

The CTF/SLATE binding plan is retired. Its role is taken by the standalone
`ctf-rs` repository (`~/tmp/ctf-rs`, a Rust port of cc4s CTF pinned at
`f69cbb46e23bc2f39cda5722ce096f56301dab4f`) with its own coverage inventory
and validation record; nothing from it is tracked on this branch. The plan
file stays as history with a superseded header.

## 2026-09-08 · evt-0006 · ctf-rs plan.v1 proposed; ctf-slate-binding frozen · actor: user

The user asked for the dense-first order to live here as a plan rather than
as an edit of ctf-rs `docs/coverage.md`. `plans/ctf-rs/plan.v1.md` (B0, D1 to
D6, S1; class R contracts against the pinned upstream) awaits acceptance.
`plans/ctf-slate-binding/plan.v1.md` is frozen: superseded, no further edits.
This restates evt-0005 with the new plan path.

- state: ctf-rs = proposed
- note: ctf-rs = plan.v1 dense-first; per-run evidence stays in ctf-rs docs/validation.md
- state: ctf-slate-binding = frozen
- note: ctf-slate-binding = superseded by plans/ctf-rs/plan.v1.md

## 2026-09-08 · evt-0007 · STATUS.md generated from ledger state lines · actor: claude

`update-status.py` renders STATUS.md from `- state:` and `- note:` lines in
this ledger, the plan headers, and the main tip. The current states are
restated here so the generator has a complete input; nothing changes state.

- state: v0.1-lapw-foundation = closed
- note: v0.1-lapw-foundation = superseded by v0.2
- state: v0.2-isdf-thc = active (M-A to M-Kc closed, M-L open)
- note: v0.2-isdf-thc = first M-L probe in evidence/2026-08-28-ml0-adaptive-rank-probe/
- state: v0.3-mto-family = draft, not authorized
- state: v0.4-emto-nmto = draft, not authorized
- note: v0.4-emto-nmto = inherits v0.3
- state: h2-lda = closed 2026-09-07
- note: h2-lda = evd-0001; gates in examples/h2_dft/README.md on main
- state: h2-hf = proposed
- note: h2-hf = awaiting user acceptance before Codex starts
- state: kr-hf = open
- note: kr-hf = 1.2 Ha total-energy gap in the VV sector as reported 2026-09-05; box and omega axes untested

## 2026-09-08 · evt-0008 · ctf-rs plan.v1 accepted · actor: user

The user accepted `plans/ctf-rs/plan.v1.md` and handed it to Codex on MSI
(`D:/projects/ctf-rs`, harness mirror read-only). Authorized: B0, then D1 to
D6 in order; S1 stays deferred. Milestone closes and handoffs arrive as
ctf-rs commits plus `docs/validation.md` sections and are recorded here as
`evd` entries with the ctf-rs SHA.

- state: ctf-rs = active: B0 authorized, D1 to D6 in order; S1 deferred
- note: ctf-rs = plan.v1 accepted 2026-09-08; Codex on MSI executes; evidence in ctf-rs docs/validation.md

## 2026-09-08 · evt-0009 · two-writer protocol adopted · actor: user

Codex on MSI may now write harness records itself instead of handing lines
back: it commits on `harness-msi` in `D:/projects/libmuffintin-harness` with
ledger IDs from 1001, pushes that branch, and the Mac merges it into
`harness`. `ledger.md` and `STATUS.md` are union-merged and `STATUS.md` is
regenerated after each merge (README "Two writers"). The ctf-rs milestone
records of evt-0008 therefore arrive as `evd-1xxx` entries.

## 2026-09-08 · evd-1001 · ctf-rs B0 bookkeeping close · ctf-rs 4eb0a4b5e6efbde23446ac2a8093c5769a1315e2

```text
DIGIT / FAIL
milestone: B0; commit: 4eb0a4b5e6efbde23446ac2a8093c5769a1315e2
validation.md section: B0 sparse WIP disposition and expression inventory (2026-09-08)
Q: distributed_sparse_search_cache exact i64 results, layouts and cache statistics; class: R; ref: pinned f69cbb46
bound: exact; observed: WIP compilation failed before runtime (missing s/c/z GEMM declarations and five f64-to-usize mismatches)
runs: WSL compilation attempt before 1/2/4 execution; native not applicable
open: none; prescribed fallback retained f22da3a on wip/sparse-search-cache and returned delivery branch to da5354b
command: cargo test --test distributed_sparse_search_cache
```

- state: ctf-rs = B0 closed
- note: ctf-rs = rejected sparse cache WIP retained on its branch; expression rows reconciled; D1 next
## 2026-09-08 · evt-0010 · h2-hf plan.v1 accepted · actor: user

`plans/h2-hf/plan.v1.md` is accepted and immutable. Codex on the Mac
executes it against `main` at 69dafb8: deliverables 1 to 4, then steps A0,
A1, A1v, A2, B, Bv as Stop That Digit contracts. Gates G-H2-HF-0 to 5 are
open with the bounds fixed in `gates.md`. Verify steps arrive as `evd`
entries from evd-0002; the closing `evt` carries `closed` or `handoff`.

- state: h2-hf = active: deliverables 1 to 4, then A0 to Bv in order
- note: h2-hf = plan.v1 accepted 2026-09-08; Codex on the Mac executes; logs in examples/h2_dft/results/hf-*.log

## 2026-09-08 · evd-1002 · ctf-rs D1 dense scaling close · ctf-rs 43a06586567a20e3c533c8ab59656bd83eee318e

```text
DIGIT / PASS
milestone: D1; commit: 43a06586567a20e3c533c8ab59656bd83eee318e
validation.md section: D1 dense scaling and strip close (2026-09-08)
Q: scaling exact strip, virtual and packed checks; scalar; diag_sym; weigh_4D; dft; class: R; ref: pinned f69cbb46
bound: exact for integer/index/layout; upstream per driver otherwise; observed: all within
runs: scaling exact checks once on final source; WSL driver world/parity at 1/2/4 once; native compile/link
open: none
command: cargo test --test scaling; MPI runner cargo test --test upstream_scalar --test upstream_diag_sym --test upstream_weigh4d --test upstream_dft
```

- state: ctf-rs = D1 closed
- note: ctf-rs = dense scaling, virtual traversal, strip/restore and packed indexed scaling closed; D2 next
## 2026-09-08 · evd-0002 · h2-hf G-H2-HF-0 · main f65c193b03bfa941573b4c689605185b922ee219

```text
DIGIT / HANDOFF
Q: exchange/eigenvalue/total identity residuals (Ha); class: A; ref: 0
bound: 1e-8; Delta: unavailable because no outer iteration completed
checks: process killed with exit 137 at 515 plane waves, spinor dimension 1030; runs: 1
unresolved: can A0 complete within the local 24 GB resource boundary?
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0.log
log: examples/h2_dft/results/hf-a0.log (main)
scope: A0 stopped under the immutable plan's resource-kill rule. A1, A1v, A2,
B, and Bv were not run; no downstream evidence entry is claimed.
```

## 2026-09-08 · evt-0011 · h2-hf handed off at A0 · actor: codex

All four deliverables were committed on `main`, but A0 was killed before its
first completed outer iteration. Per plan.v1, the spinor dimension is recorded
and numerical execution stops without spending a diagnostic row or starting a
downstream study.

- state: h2-hf = handoff: can A0 complete within the local 24 GB resource boundary?
- note: h2-hf = evd-0002; deliverables complete on main; A1 through Bv not run

## 2026-09-08 · evd-1005 · ctf-rs D4 shared infrastructure close · ctf-rs 3acda59

```text
DIGIT / PASS
milestone: D4; commit: 3acda59
validation.md section: D4 shared infrastructure close (2026-09-08)
Q: exact memcontrol, timer, util, four-type SYR and flop-counter checks; class: R; ref: pinned f69cbb46
bound: exact integer/index/layout/BLAS/flop checks and timer invariants; observed: all within
runs: WSL world/parity at 1/2/4 once; mpi_low_memory_bench 0.002395 s once at four ranks; native compile/link
open: none
command: MPI runner cargo test for three D4 drivers; mpirun -n 4 release mpi_low_memory_bench; acceptance-native.ps1 -BuildOnly
```

- state: ctf-rs = D4 closed
- note: ctf-rs = process memory budgets, low-memory path, timers, util, four-type SYR and flop snapshots closed; D5 next

## 2026-09-08 · evt-0011 · hf-input proposed; ADR-0005 recorded · actor: user

The user decided that Hartree–Fock enters through the same input file and
`dft-scf` task as DFT, as `xc.kind = "exact-exchange"`, with the driver
family chosen from `xc`, `relativity`, and the core channels
(`decisions/ADR-0005-hf-is-an-xc-kind-of-dft-scf.md`). The refactor is
`plans/hf-input/plan.v1.md`: schema, lowering, an exact spec-equality test,
and the two HF examples reduced to input files plus printing. Gates G-HFI-1
and G-HFI-2 are class R. It starts after `h2-hf` closes or hands off;
`h2-hf` deliverables 1 and 2 are already on `main` (465be6e, 3ef6bca).

- state: hf-input = proposed
- note: hf-input = ADR-0005; starts after h2-hf is closed or handoff; no physics gates

## 2026-09-08 · evd-1003 · ctf-rs D2 optimized redistribution close · ctf-rs e2e60f96f7442e85ddc71aca0f62686503b8e68d

```text
DIGIT / PASS
milestone: D2; commit: e2e60f96f7442e85ddc71aca0f62686503b8e68d
validation.md section: D2 optimized dense redistribution close (2026-09-08)
Q: readwrite_test; readall_test; repack; permute_multiworld NS; reduce_bcast; subworld_gemm; class: R; ref: pinned f69cbb46
bound: upstream per driver; observed: all within
runs: WSL world/parity at 1/2/4 once; bench_redistribution 0.000077 s and bench_nosym_transp 0.000656 s once at four ranks; native compile/link
open: none
command: MPI runner cargo test for the six D2 drivers; mpirun -n 4 each release benchmark once
```

- state: ctf-rs = D2 closed
- note: ctf-rs = DGTOG ROR, block/global reshuffle, transpose, slice and packed padding closed; D3 next

## 2026-09-08 · evd-1004 · ctf-rs D3 dense contraction close · ctf-rs 00a4d12

```text
DIGIT / PASS
milestone: D3; commit: 00a4d12
validation.md section: D3 dense contraction close (2026-09-08)
Q: gemm_4D all branches; weigh_4D; sy_times_ns; ccsdt_t3_to_t2; ccsdt_map_test; multi_tsr_sym; seven studies/fast_* drivers; class: R; ref: pinned f69cbb46
bound: upstream per driver; observed: all within
runs: WSL world/parity at 1/2/4 once; native compile/link
open: none
command: MPI runner cargo test for the thirteen D3 drivers; acceptance-native.ps1 -BuildOnly
```

- state: ctf-rs = D3 closed
- note: ctf-rs = replication, four-type folded panels, automatic compressed contraction/sum planning and seven fast studies closed; D4 next
## 2026-09-08 · evt-0012 · h2-hf handoff ID collision superseded · actor: codex

The h2-hf handoff entry above reused `evt-0011` after that ID had been assigned
concurrently to the hf-input proposal. This append-only correction supersedes
the h2-hf entry's heading ID as `evt-0012`; its evidence, state, and unresolved
question remain unchanged.

- state: h2-hf = handoff: can A0 complete within the local 24 GB resource boundary?
- note: h2-hf = evd-0002; deliverables complete on main; A1 through Bv not run

## 2026-09-08 · evt-0013 · h2-hf handoff question answered: time, not memory · actor: claude

Two probes of the A0 command (`evidence/2026-09-08-h2-hf-a0-probe/`)
replace the unresolved question of evd-0002/evt-0012. Peak RSS was 2.06 GB
on the 24 GB machine; the process ran one core inside
`spinor_mpb::contract_interstitial_selections` (`fft-fftw` variant, one
TBLIS einsum per band pair, threads waiting at barriers). With timings on,
the first MPB rebuild spent 49.7 s in `vv.mt_contraction` and was still in
`vv.interstitial` after 300 s; the Fock loop may rebuild 40 times per outer
iteration, and `h2_hf.rs` prints nothing until the driver returns, so the
4400 s silence was expected behavior of a slow path, not a hang or a kill
for memory. A0 resumes unchanged once the rebuild is fast enough to finish
the run; no bound, row, or setting of plan.v1 changes.

- state: h2-hf = handoff: A0 blocked by the fft-fftw interstitial MPB contraction (>300 s per rebuild, one core); memory peak 2.1 GB
- note: h2-hf = evd-0002 handoff re-read by evt-0013; perf fix on main needed before A0 can complete

## 2026-09-08 · evt-0014 · h2-hf resumes after a perf fix on main · actor: user

The user authorized a performance commit on `main` before A0 is rerun:
batch the `fft-fftw` variant of `spinor_mpb::contract_interstitial_selections`
per left band (the non-fftw variant already batches 64 selections per
einsum) and name where `vv.mt_contraction` spends its 49.7 s. Acceptance
is class R only: the `gamma_valence_hf` identities stay at 1e-8 and the
fftw and non-fftw vertices agree to 1e-10. Codex on the Mac executes, then
resumes plan.v1 at A0 with the exact evd-0002 command; the plan is
unchanged. Codex writes `evd-0003`+ and `evt-0015`+.

- state: h2-hf = active: perf fix on the fft-fftw interstitial contraction, then A0 to Bv as in plan.v1
- note: h2-hf = evt-0013 diagnosis; Codex on the Mac executes; evd-0003+ / evt-0015+

## 2026-09-08 · evd-1006 · ctf-rs D5 native interface close · ctf-rs 36fe4f7

```text
DIGIT / PASS
milestone: D5; commit: 36fe4f7
validation.md section: D5 native interface and FFT close (2026-09-08)
Q: partition, algebra, vector/scalar, common helpers, FFT/DFT and endomorphism/function drivers; class: R; ref: pinned f69cbb46
bound: exact interface checks and upstream per-driver numerical bounds; observed: all within
runs: WSL world/parity at 1/2/4 once; native compile/link
open: none
command: MPI runner cargo test for the twelve D5 drivers; acceptance-native.ps1 -BuildOnly
```

- state: ctf-rs = D5 closed
- note: ctf-rs = native partition/value/algebra/common interfaces and FFT/DFT drivers closed; D6 next

## 2026-09-08 · evd-1007 · ctf-rs D6 native runtime handoff · ctf-rs 17402f3c8cfbbfd3467d8092682ae77eabe4fbfb

```text
DIGIT / HANDOFF
milestone: D6; commit: 17402f3c8cfbbfd3467d8092682ae77eabe4fbfb
validation.md section: D6 dense drivers and native runtime handoff (2026-09-08)
Q: dense test_suite subset, eleven examples, dense low-memory path, two informational benchmarks and native dense runtime; class: R; ref: pinned f69cbb46
bound: upstream per driver; observed: all WSL values within, native produced no value
runs: WSL world/parity at 1/2/4 once; benchmarks once at four ranks; native compile/link passed; native runtime stopped before first driver
open: can Microsoft MPI mpiexec be installed so the native 1/2/4-rank dense runtime can execute?
command: MPI runner cargo test for D6 drivers; acceptance-native.ps1 -BuildOnly; acceptance-native.ps1 -D6Only
```

- state: ctf-rs = D6 handoff: d4_blas_flops never executed because mpiexec was not found
- note: ctf-rs = D6 code and WSL 1/2/4 closed; native compile/link passed; install Microsoft MPI launcher before the one remaining runtime gate

## 2026-09-08 · evd-0003 · h2-hf FFTW interstitial batching · main 44fd1886a60bc6ee2b1ecb8cf303ab6cc4abc644

```text
DIGIT / PASS
Q: exchange/eigenvalue/total identity residuals (Ha); class: R; ref: gamma_valence_hf fixture
bound: 1e-8 unchanged; Delta: every residual passed the existing assertions
checks: focused test once without and once with fft-fftw; runs: 2; numerical verification closed

DIGIT / PASS
Q: first-build interstitial vertex coefficients; class: R; ref: non-fftw build
bound: 1e-10 absolute; Delta: 0; d: 0
checks: 432 complex coefficients, no key mismatches; comparisons: 1; numerical verification closed

STUDY / REPORT
Q: first-rebuild vv.interstitial seconds; class: P; ref: >300 s before
bound: none; observed: >564 s after, first rebuild incomplete at 600 s process cap
context: vv.mt_contraction was 49.667629 s before and 35.201104 s after
commands: cargo test -p libmuffintin-runtime --test gamma_valence_hf; cargo test -p libmuffintin-runtime --features fft-fftw --test gamma_valence_hf; gawk coefficient comparison recorded in the evidence README
log: evidence/2026-09-08-h2-hf-fft-batching/timing-after.log
scope: coefficient dumps and their exact comparison command are in evidence/2026-09-08-h2-hf-fft-batching/. The temporary test hook and timing example were never committed.
```

## 2026-09-08 · evd-0004 · h2-hf G-H2-HF-0 rerun · main 44fd1886a60bc6ee2b1ecb8cf303ab6cc4abc644, recorded through a5bd71dbc9a604c094ee60abfa1ab8541fce24a5

```text
DIGIT / HANDOFF
Q: exchange/eigenvalue/total identity residuals (Ha); class: A; ref: 0
bound: 1e-8; Delta: unavailable because no outer iteration completed
checks: exact A0 rerun reached the authorized 1800 s wall limit; runs: 1
unresolved: what performance change beyond per-left-band projection batching is needed for A0 to return within 30 minutes?
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0.log
supervisor: /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s
log: examples/h2_dft/results/hf-a0.log (main)
timing log: evidence/2026-09-08-h2-hf-fft-batching/timing-after.log
scope: A0 stopped under the user-authorized wall-time rule. A1, A1v, A2, B, and Bv were not run; no downstream evidence entry is claimed.
```

## 2026-09-08 · evt-0015 · h2-hf handed off after bounded A0 rerun · actor: codex

The requested `fft-fftw` projection batching is on `main` and passed both
class-R checks. Its single timing probe still did not complete the first MPB
rebuild, and the unchanged committed A0 example did not return within the
authorized 30-minute wall limit. Numerical execution therefore stops before
A1 without changing plan.v1.

- state: h2-hf = handoff: what performance change beyond per-left-band projection batching is needed for A0 to return within 30 minutes?
- note: h2-hf = evd-0003 batching checks passed; evd-0004 A0 timed handoff; A1 through Bv not run

## 2026-09-08 · evt-0016 · h2-hf: the rebuild selects all 1030² band pairs · actor: claude

Answers the evt-0015 handoff question. A scratch build printed the size
at the start of `vv.interstitial`: 1 060 900 selections (1030²), 3 809 raw
components, 515 product waves, a 21³ pair FFT grid
(`evidence/2026-09-08-h2-hf-a0-probe/README.md`, "After the batching
commit"). `rebuild_exchange` on the Gamma valence path selects every
`(left_band, right_band)` pair of the spinor window; the relaxed-core frame
already restricts `left_band` to occupied bands. The perf change needed is
that restriction on the valence path, with the same occupation criterion
the exchange assembly uses; it removes roughly 97 % of the pairs at the A0
settings (about 30 bands carry a nonzero Fermi tail at 1 mHa) and more if
the assembly ignores tails below its own threshold. `vv.mt_contraction`
scales with the same selection list.

- state: h2-hf = handoff: rebuild_exchange builds 1030² pair vertices; restrict left bands to occupied, then rerun A0
- note: h2-hf = evt-0016 diagnosis; next perf task not yet assigned

## 2026-09-08 · evt-0017 · h2-hf perf fix assigned to the herdr claude-worker pane · actor: user

The user assigned the evt-0016 change (restrict `rebuild_exchange` left
bands to occupied bands on the Gamma valence path) and the A0 to Bv rerun
to the Claude session in the herdr `claude-worker` pane on the Mac. Task
text: the session scratchpad file `worker-task-h2-hf-perf.md`. Class R
acceptance only for the change; plan.v1 unchanged. The worker writes
evd-0005+ and evt-0018+.

- state: h2-hf = active: perf fix (occupied left bands) by claude-worker, then A0 to Bv as in plan.v1
- note: h2-hf = evt-0016 diagnosis; claude-worker executes; evd-0005+ / evt-0018+
