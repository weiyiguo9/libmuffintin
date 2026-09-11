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

## 2026-09-08 · evd-0005 · h2-hf occupied left bands · main 78b84917cb961d2265009582ff39df2d8c6070a8

```text
DIGIT / PASS
Q: exchange/eigenvalue/total identity residuals (Ha); class: R; ref: gamma_valence_hf fixture
bound: 1e-8 unchanged; Delta: every residual passed the existing assertions
checks: focused test once without and once with fft-fftw; runs: 2; numerical verification closed

DIGIT / PASS
Q: fixture total_energy and exchange_energy (Ha); class: R; ref: the same fixture at a5bd71d
bound: 1e-10; Delta: 0; d: 0
checks: total -5.32924235801359503e-1 and exchange -1.71511931745922745e-5 before and after, in both builds; runs: 2 before and 3 after, one after run repeated to confirm the incremental rebuild; numerical verification closed

STUDY / REPORT
Q: first-rebuild selection count and vv.interstitial seconds at the A0 settings; class: P; ref: 1 060 900 and >420 s
bound: none; observed: 39 140 selections (38 occupied left bands of 1030) and vv.interstitial 27.281 s
context: vv.mt_contraction fell from 35.201 s to 0.613 s and the whole first vv.mpb_rebuild took 28.658 s
commands: cargo test -p libmuffintin-runtime --test gamma_valence_hf; cargo test -p libmuffintin-runtime --features fft-fftw --test gamma_valence_hf; the A0 command against a scratch timing copy of the example
log: evidence/2026-09-08-h2-hf-a0-probe/timing-occupied.log
scope: `build_spinor_mpb_exchange` requires every square-layout column, so the
call moved to `contract_selected_spinor_mpb_exchange_with_operators`, the
crate-internal VV consumer the relaxed-core frame already uses, with
`assemble_coulomb` in the caller. Both routes end in the same
`contract_rectangular_exchange` over an auxiliary basis independent of the
selection list, which is why the fixture energies are bit identical. The A0
gate value under a5bd71d is unavailable because that build never returned.
The temporary fixture print, the scratch example, and the selection-count
print were never committed.
```

## 2026-09-08 · evt-0018 · h2-hf perf fix committed, A0 rerun · actor: claude

The evt-0016 restriction is on `main` at 78b8491 and passed both class-R
checks with zero change in the fixture energies. The first MPB rebuild at
the A0 settings dropped from an unfinished 1 060 900-vertex build to 39 140
vertices in 28.7 s, and the committed A0 example now returns in 251 s.

- state: h2-hf = active: A0 rerun after the occupied-left-band perf fix
- note: h2-hf = evd-0005 class-R checks passed; A0 verify next

## 2026-09-08 · evd-0006 · h2-hf G-H2-HF-0 A0 identity handoff · main 78b84917cb961d2265009582ff39df2d8c6070a8, recorded through 05b6d520f7c04c7855abf655a3731b90f141771b

```text
DIGIT / HANDOFF
Q: valence eigenvalue identity residual (Ha); class: A; ref: 0
bound: 1e-8; Delta: 3.4089473164444770e-4; d: 3.4e4
checks: the exchange and total identity gates are never reached, the driver returns on the first one that fails; runs: 1 + the 3 diagnostics plan.v1 lists for A0
diagnostic 1: --fock-max-iterations 256 gives the identical residual, so the Fock iteration limit is not the cause
diagnostic 2: orbital 3 / product 3 fails the same identity at 1.4380668996653856e-4, so it does not change which identity fails
diagnostic 3: the gamma_valence_hf fixture at product_g_max 4 holds every identity at 1e-8, so the driver itself is sound at that product cutoff
unresolved: is the 3.4e-4 residual only the example's own 1e-5 Fock exit tolerance showing through the 2e-8 identity gate, or a defect of the two-site setup under fractional occupation tails?
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0.log
supervisor: /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s
log: examples/h2_dft/results/hf-a0.log and hf-a0-diag1-fock256.log, hf-a0-diag2-orb3prod3.log, hf-a0-diag3-fixture-productg4.log (main)
scope: A0 returned in 251 s and failed its gate; the driver errors before the
example prints hf_energy_terms_ha, so the row has no E, HOMO, E_H, or E_x.
Diagnostic 3 does not separate a two-site setup error from a driver defect
under fractional occupation tails: the fixture carries one fully occupied
band while A0 spreads 38 fractionally occupied bands over the 1 mHa tail. The
example exits its Fock loop at fock_density_tolerance 1e-5 and
fock_feedback_tolerance 1e-5 Ha; the fixture that passes at 1e-8 exits at
1e-7 and 1e-8 Ha. A1, A1v, A2, B, and Bv were not run and claim no evidence.
Nothing in plan.v1 authorizes moving the tolerance, so the plan stops here.
```

## 2026-09-08 · evt-0019 · h2-hf handed off on the A0 identity gate · actor: claude

The perf fix answered the evt-0016 wall-time question and A0 is now a
numerical result rather than a timeout. It fails plan.v1's class-A digit,
its three listed diagnostics are recorded in evd-0006, and the plan stops
before A1 without any tolerance being changed.

- state: h2-hf = handoff: A0 fails the valence eigenvalue identity at 3.4e-4 against 1e-8; is that the example's 1e-5 Fock exit tolerance or a two-site defect?
- note: h2-hf = evd-0005 perf checks passed; evd-0006 A0 handoff; A1 through Bv not run

## 2026-09-08 · evt-0020 · h2-hf Fock exit tolerance fix authorized, assigned to claude-worker · actor: claude

The user accepted the evt-0019 handoff reading: evd-0006 diagnostic 1
(Fock limit doubled) reproduced the 3.4e-4 residual to every digit, so the
Fock loop exits normally at the example's 1e-5 density and feedback
tolerances, which the passing fixture sets to 1e-7 and 1e-8. The user
authorized one fix(examples) commit adopting the fixture's Fock exit
tolerances (solver settings plan.v1 does not fix; every bound stays as
written) and one rerun of A0 under the unchanged A0 contract: a pass
continues A1 through Bv, a failure is a driver defect handoff. The
claude-worker pane executes; its records continue at evt-0021 and evd-0007.

- state: h2-hf = active: A0 rerun with the fixture's Fock exit tolerances (claude-worker)
- note: h2-hf = evt-0019 handoff accepted as an example defect; plan.v1 bounds unchanged
## 2026-09-08 · evd-1008 · ctf-rs D6 native runtime close · ctf-rs 354cf7dc1e2f675d8ed88352e6f579aa70185970

```text
DIGIT / PASS
milestone: D6; commit: 354cf7dc1e2f675d8ed88352e6f579aa70185970
validation.md section: D6 dense drivers and native runtime close (2026-09-08)
Q: dense test_suite subset, eleven examples, dense low-memory path, two informational benchmarks and native dense runtime; class: R; ref: pinned f69cbb46
bound: exact or upstream per driver; observed: all within
runs: WSL world/parity at 1/2/4 once; benchmarks once at four ranks; native compile/link once; native runtime at 1/2/4 once after Microsoft MPI installation
open: none
command: MPI runner cargo test for D6 drivers; acceptance-native.ps1 -BuildOnly; acceptance-native.ps1 -D6Only
```

This entry supersedes the native-runtime HANDOFF in evd-1007 after the required
Microsoft MPI launcher became available; no acceptance bound or fixture changed.

- state: ctf-rs = D6 closed
- note: ctf-rs = D6 WSL and native 1/2/4 passed; dense-first objective complete; S1 not started

## 2026-09-08 · evd-0007 · h2-hf G-H2-HF-0 A0 identity handoff after the Fock exit tolerance fix · main 30cc770931069be0527ff2f31424efb09001d1f7, recorded through 815fe4c028e97e197ea73dd301069f29da42ade3

```text
DIGIT / HANDOFF
Q: valence eigenvalue identity residual (Ha); class: A; ref: 0
bound: 1e-8; Delta: 1.6477445782814293e-7; d: 16.5
checks: the gates run in the order electron count, exchange identity, valence eigenvalue identity, total identity, so the exchange identity passed the driver 2e-8 check without printing its value and the total identity was never reached; runs: 1
budget: the single diagnostic the rerun contract allowed was conditional on the run stopping at the Fock iteration limit; the driver reaches its identity gate only after the Fock loop has converged, so no diagnostic was authorized and none was spent
unresolved: the residual is no longer the example's Fock exit tolerance showing through the gate, so the eigenvalue identity of the two site Gamma valence path does not close at 1e-8 for reasons inside the driver
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0.log
supervisor: /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s
log: examples/h2_dft/results/hf-a0.log (main); the evd-0006 run is preserved as examples/h2_dft/results/hf-a0-fock1e-5.log
scope: 30cc770 adopts the gamma_valence_hf fixture's fock_density_tolerance
1e-7 and fock_feedback_tolerance 1e-8 Ha in crates/mt-runtime/examples/h2_hf.rs
and changes nothing else; FOCK_MAX_ITERATIONS stays 128 and every plan.v1 bound
stays as written. The rerun moved the residual by a factor of 2069 and still
fails, so the evt-0019 question is answered in the negative: the 1e-5 exit
tolerance was most of the 3.4e-4, but 1.6e-7 survives it. The driver returns
before the example prints hf_energy_terms_ha, so this row has no E, HOMO, E_H,
E_x, and no printed Fock iteration count; the wall time is 282 s measured from
the output directory to the log, not from an hf_final line. A1, A1v, A2, B, and
Bv were not run and claim no evidence.
```

## 2026-09-08 · evt-0021 · h2-hf handed off again on the A0 identity gate · actor: claude

The evt-0020 authorization is spent. The fix landed as one fix(examples)
commit, the A0 rerun used the unchanged evd-0002 command, and A0 still fails
its class A digit at 1.6477445782814293e-7 against 1e-8. The residual is
16.5 times the bound and 16.5 times the 1e-8 Ha feedback tolerance the Fock
loop exited on, so it is not a residue of the exit setting that was changed.
Per evt-0020 this is a driver defect and was not investigated further.

- state: h2-hf = handoff: A0 still fails the valence eigenvalue identity, now at 1.6e-7 against 1e-8, with the example's Fock exit tolerances at the fixture's values
- note: h2-hf = evd-0007 supersedes the evd-0006 reading; main 30cc770 fix and 815fe4c docs; A1 through Bv not run

## 2026-09-08 · evt-0022 · h2-hf second Fock exit tolerance step authorized, assigned to claude-worker · actor: claude

Orchestrator reading of evd-0007, accepted by the user: the valence
eigenvalue identity residual tracks the example's Fock exit tolerance
(feedback 1e-5 to 1e-8 moved it from 3.409e-4 to 1.648e-7, a factor 2069
for a factor 1000), with a prefactor of 16 to 34 between the residual and
the feedback tolerance. The evt-0021 inference "16.5 times the feedback
tolerance, therefore not a residue" divides two different quantities and
does not follow. The driver-side finding is a contract mismatch, not
two-site physics: the Gamma valence path gates its identities at a fixed
2e-8 regardless of the caller's exit tolerances, while the relaxed-core
path gates at fock_feedback_tolerance and the KH+SOC path at
max(2e-8, N_val * feedback); the Gate error also returns before the
iteration diagnostics are recorded, so the actual exit residuals are never
printed. Both are deferred to the hf-input refactor (plans/hf-input). The
user authorized one more fix(examples) commit, density 1e-9 and feedback
1e-10, and one A0 rerun under the unchanged A0 contract with a prediction
fixed before the run: residual in [1e-9, 4e-9], at least 40 times below
1.648e-7; a residual within a factor 3 of 1.648e-7 is a floor and a
genuine driver defect. A pass continues A1 through Bv. The claude-worker
pane executes; its records continue at evt-0023 and evd-0008.

- state: h2-hf = active: second A0 rerun, Fock exit tolerances 1e-9 / 1e-10, prediction 1e-9 to 4e-9 (claude-worker)
- note: h2-hf = evd-0007 read as a convergence residue; driver gate versus exit tolerance mismatch deferred to hf-input

## 2026-09-08 · evd-0008 · h2-hf G-H2-HF-0 A0 wall cap handoff at Fock exit tolerances 1e-9 / 1e-10 · main 08f7c21c9e71b0382363ae32d0347f3ce38abbd3, recorded through 421a44750468e8d34a375a6dea387f07fa82da4e

```text
DIGIT / HANDOFF
Q: valence eigenvalue identity residual (Ha); class: A; ref: 0
bound: 1e-8; Delta: unavailable, no outer iteration completed
prediction: untested; the run produced no residual, so it neither confirms the predicted 1e-9 to 4e-9 nor shows a floor near 1.6477e-7
checks: killed by the supervisor at the 1800 s cap, exit 124, after writing only its two header lines; runs: 1
budget: plan.v1 diagnostic 1 was conditional on the run ending in a Fock not-converged error; it ended in a wall clock kill, so no diagnostic was authorized and none was spent
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0.log
supervisor: /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s
log: examples/h2_dft/results/hf-a0.log (main), two lines, header only; the evd-0007 run is preserved as examples/h2_dft/results/hf-a0-fock1e-8.log
scope: 08f7c21c9e71b0382363ae32d0347f3ce38abbd3 sets FOCK_DENSITY_TOLERANCE 1e-9 and
FOCK_FEEDBACK_TOLERANCE_HARTREE 1e-10 in crates/mt-runtime/examples/h2_hf.rs
with a two line comment and changes nothing else; FOCK_MAX_ITERATIONS stays 128
and every plan.v1 bound stays as written. This is not the plan.v1 memory kill
case: the command, the plane wave dimension and the spinor basis are identical
to the evd-0007 run, which reached the identity gate in 282 s, and the only
change is the exit tolerance, so the loop was still inside its first Fock cycle
when the cap arrived. The loop's own limit is above the cap, since
FOCK_MAX_ITERATIONS is 128 and evd-0005 measured one exchange rebuild at these
settings at 28.7 s; what the loop would have done with more time is not
claimed. A1, A1v, A2, B, and Bv were not run and claim no evidence.
```

## 2026-09-08 · evt-0023 · h2-hf handed off on the A0 wall cap · actor: claude

The evt-0022 authorization is spent. The fix landed as one fix(examples)
commit and the A0 rerun used the unchanged evd-0002 command, but the run was
killed at the 1800 s cap without completing an outer iteration. evd-0008
therefore answers neither the evt-0022 prediction nor the residue versus floor
question: no residual was produced. Per the assignment the failure was not
investigated.

- state: h2-hf = handoff: A0 at Fock exit tolerances 1e-9 / 1e-10 was killed at the 1800 s cap with no residual; can the Fock loop reach 1e-10 Ha within the cap at the A0 dimension, or does the residue versus floor test need a longer cap or an intermediate tolerance?
- note: h2-hf = evd-0008; prediction untested, no diagnostic authorized; main 08f7c21 fix and 421a447 docs; A1 through Bv not run

## 2026-09-08 · evt-0024 · v0.2-isdf-thc M-L implementation closed; cross-code acceptance deferred · actor: user

The plan header imported on 2026-09-08 still said "M-L open" because it was
copied from the scratch plan last modified 2026-08-27; the M-L deliverables
landed on main between 2026-08-29 and 2026-09-03: frozen spinor product
input 9ae3d3f, spinor THC Coulomb bridge 24a2d42, bounded Sm THC lane on
the SPEX snapshot b45d9b9 (8670401), MLDUMP HDF5 payloads 9880886 and
045bb26 plus the CoQuí Cholesky ERI writer, frozen-orbital ISDF exchange
3700ee4, core-aware four-sector THC bdfff88 with the exact-MPB sector gates
7b2c7a5, frozen core-valence actions d145dff, explicit Dirac product modes
760ce03. External evidence: evidence/2026-08-28-thc-smdy-experiment/RESULTS.md
(THC against LCUT=6 SPEX, tight rank 7.1 to 7.3 orbitals) and the ML0
weighted-QRCP rank probe (verdict PASS). Deferred, not claimed: the span
principal-angle metric of doc 08, the Dy bcc demo (no DFT input,
tests/dy_bcc_material_evidence.toml), and an independent cross-code number
for the magnetic plus SOC 4c first-variation case. Successor workstream:
hf-thc-scf (THC exchange inside the Fock loop), to be proposed.

- state: v0.2-isdf-thc = closed (M-L implementation closed 2026-09-08; cross-code acceptance deferred)
- note: v0.2-isdf-thc = deferred: span metric, Dy bcc demo, magnetic+SOC 4c cross-code; successor hf-thc-scf

## 2026-09-08 · evt-0025 · h2-hf pair-FFT perf and pair-level MPI authorized, assigned to the codex pane · actor: claude

The user chose speed over a fourth tolerance step. The evd-0008 cap kill
came from a Fock loop whose rebuild costs 28.7 s, of which 27.3 s is the
serial fft-fftw interstitial pair correlation: 234,840 transforms of 21³ per
rebuild on one core, two forward and one inverse per pair and spin. Work
authorized, in order: (1) perf(runtime), cache the per-orbital spectra and
sum spins before one inverse transform, rayon over left bands with one plan
per worker; (2) rerun A0 with the exact evd-0002 command under a 75 minute
cap so the loop reaches its own verdict, with a Progress-level per-iteration
residual print; (3) feat(runtime), pair-level MPI under an optional `mpi`
feature per ADR-0006. The Stop That Digit contracts are fixed in the brief;
the codex pane w1:p2 executes; records continue at evt-0026 and evd-0009;
timing logs go to evidence/2026-09-08-h2-hf-pair-fft-perf/.

- state: h2-hf = active: pair-FFT perf, then A0 to the loop's own limit, then pair-level MPI (codex pane)
- note: h2-hf = evd-0008 cap kill traced to the serial pair FFT; ADR-0006 fixes MPI at pair level

## 2026-09-08 · evt-0026 · h2-hf pair-FFT implementation started · actor: codex

Verified clean main at 421a44750468e8d34a375a6dea387f07fa82da4e and clean
harness at 96fb7fdadf0200076a07401f1bb5e3329603f27b, including evt-0025 and
ADR-0006. Task 1 uses the brief's fixed class-R bounds: fixture identities
1e-8 Ha, fixture energy preservation 1e-10 Ha, and maximum absolute A0
interstitial vertex difference 1e-10. Budget: two fixture runs before and
two after, two vertex dumps, two after timings; no failure diagnostics.
No push is authorized.

- state: h2-hf = active: capturing pair-FFT baseline and implementing spectrum caching
- note: h2-hf = evt-0025 fixed contracts; task 2 and MPI wait for task 1

## 2026-09-08 · evd-0009 · h2-hf pair-FFT preservation and timing · main 814003278024a560de4fef737749015215b36b74

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals (Ha); class: R; ref: fixture
bound: 1e-8; Delta: maximum 3.42933147157165052e-16
Q: fixture total and exchange energy preservation (Ha); class: R; ref: 421a447
bound: 1e-10; Delta: 0 for both energies with and without fft-fftw
Q: maximum absolute first-rebuild A0 interstitial vertex difference; class: R; ref: 421a447 fft-fftw
bound: 1e-10; Delta: 6.50521335639941991e-19 over 20,157,100 finite entries
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw -- --nocapture
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
command: /opt/homebrew/bin/python3 compare_vertices.py /tmp/h2-hf-perf-before.bin /tmp/h2-hf-perf-after.bin 1e-10
log: evidence/2026-09-08-h2-hf-pair-fft-perf/{fixture-before-fftw,fixture-before-direct,fixture-after-fftw,fixture-after-direct,fixture-compare,vertex-compare}.log
scope: fixed task-1 gates closed. Four fixture runs, two vertex dumps, two changed timing samples (one overlaps the changed dump); seven numerical executions total. No diagnostics.
```

The README in that evidence directory records the exact scratch build/run
commands, environment, and log mapping. First-rebuild class-P report-only
timings (`vv.interstitial` / `vv.mpb_rebuild` seconds): old evd-0005 reference
27.281043 / 28.657877; captured 421a447 one-thread baseline 34.261638 /
37.398714; changed one-thread 21.380939 / 24.576575; changed ten-thread
9.604797 / 11.903934. Transform count is 234,840 before versus 41,276 after.
The right-spectrum cache uses 305,242,560 bytes at A0 and disjoint right-band
chunks above 4,000,000,000 bytes. MPI remains unimplemented and unrun.

## 2026-09-08 · evt-0027 · h2-hf pair-FFT gates closed, A0 own-limit run started · actor: codex

Task 1 closed at evd-0009. Main adds Progress residual output in `7efe82c`
and the default-quiet `--verbosity` option in `ca2e5ed`. A0 retains the exact
evd-0002 numerical settings, adds `--verbosity 1`, and is supervised by
`/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 4500s`. Class A:
three identities against zero within 1e-8 Ha; validity requires convergence,
finite energies, and electron count 2 within 1e-8. One run, no diagnostics.
The existing cap-1800 log is retained as `hf-a0-fock1e-10-cap1800.log`.

- state: h2-hf = active: pair-FFT passed; A0 rerun to the existing 128-iteration limit
- note: h2-hf = evd-0009 closed; A0 one run under 4500 s, no diagnostics; MPI conditional on outcome

## 2026-09-08 · evt-0028 · h2-hf A0 passed, A1 base row started · actor: codex

A0 outcome (a): converged in 8 outer iterations, 7 Fock iterations each,
56 rebuilds, driver wall 1934.680438 s (supervisor wall 1935.39 s), exit 0.
Maximum three-identity residual across all outer iterations is
1.7276811092870048e-9 Ha; final exchange/eigenvalue/total residuals are
0 / 7.2523410887814777e-10 / 7.2523409500035996e-10 Ha. All are below
1e-8 Ha, all reported energies are finite, and final electron count is
2.0000000000000058. A0 is closed; no A0 diagnostic or repeat is authorized.
The first-outer eigenvalue residual 1.7276809149979755e-9 fits evt-0022's
prediction; the final residual is below its predicted interval.

Proceed to the immutable A1 base row: orbital 5, field 12, product 6,
product lmax 4, overlap tolerance 1e-4, common box-8 settings. This is class P,
report-only hartree_exchange against zero with no study bound; every run
still requires converged finite output, electron count 2 within 1e-8, and
the three identities within 1e-8 Ha. One execution per authorized row,
supervised at 1800 s, no added diagnostics.

- state: h2-hf = active: A0 passed; A1 base row under 1800 s cap
- note: h2-hf = A0 outcome a; no further A0 checks; downstream study follows plan.v1

## 2026-09-08 · evd-0010 · h2-hf A0 pass and A1 base resource handoff · main ca2e5ed, recorded through 35c200cdcc8283c3ad6744267a7bbd49684045af

```text
DIGIT / PASS
Q: driver exchange/eigenvalue/total identity residuals (Ha); class: A; ref: 0
bound: 1e-8; Delta: maxima 0 / 1.7276809149979755e-9 / 1.7276811092870048e-9
checks: converged, finite energies; final electron_count=2.0000000000000058 within 1e-8 of 2; exit 0
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 4500s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a0.log
log: examples/h2_dft/results/hf-a0.log (main)
scope: A0 outcome (a), closed; one run, no diagnostics. E=-0.59291354371542793, HOMO=-0.12617886551775270, E_H=0.37293092552423746, E_x=-0.023151590595997563 Ha; final identities=0 / 7.2523410887814777e-10 / 7.2523409500035996e-10 Ha; hartree_exchange=0.16331387216612117 Ha. Eight outer iterations, seven Fock iterations each, 56 total rebuilds; driver wall=1934.680438 s, supervisor wall=1935.39 s.

DIGIT / HANDOFF
Q: A1 base hartree_exchange (Ha); class: P; ref: 0
bound: none (study); Delta: unavailable
checks: no completed outer iteration or converged result; required three identities within 1e-8 Ha and final electron count 2 within 1e-8 not established
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a1-row1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a1-row1.log
log: examples/h2_dft/results/hf-a1-row1.log (main)
scope: one run, supervisor exit 124 at 1800.83 s; two completed Fock iterations. Last density=1.0988849307297540e-4 and feedback=1.3734823798679154e-4 Ha. No diagnostic; no convergence-floor claim.
unresolved: the A1 base row cannot finish within its authorized 1800 s cap.
```

The first-outer A0 eigenvalue residual fits evt-0022's [1e-9, 4e-9]
prediction and is 95.37 times below 1.6477445782814293e-7. The final residual
is below that predicted interval. This is not an external-energy or
product-basis-completeness pass. Main preserves the old A0 log under
`hf-a0-fock1e-10-cap1800.log`. All exact scratch commands and timing logs
remain in `evidence/2026-09-08-h2-hf-pair-fft-perf/`.

## 2026-09-08 · evt-0029 · h2-hf handed off at the A1 base cap · actor: codex

A0 and Task 1 remain closed. Stop at the A1 resource-budget handoff;
A1 rows 2–10, A1v, A2, B, Bv, and Task 3 pair-level MPI were not run.
No MPI implementation or rank-count evidence is claimed. The brief used
nine numerical executions: four fixtures, two vertex dumps, one separate
timing, A0, and A1 base. Two changed timing samples include the changed dump.
No numerical diagnostics, repeated passes, or pushes were performed.

- state: h2-hf = handoff: A0 passed; A1 base exceeds the authorized 1800 s cap
- note: h2-hf = evd-0009 and evd-0010; main 35c200c; downstream rows and MPI unrun

## 2026-09-08 · evt-0030 · h2-hf A1 cap traced to the uncached Gamma exchange rebuild; profile and conditional cache assigned, MPI held · actor: claude

evd-0010 closed A0 (identities 0 / 7.3e-10 / 7.3e-10 Ha, 8 outer × 7 Fock
iterations, 1935 s) and stopped A1 row 1 at the 1800 s cap after two Fock
iterations, about 900 s each at orbital 5, product G 6, product l 4. After
the pair-FFT perf the A0 rebuild is 11.9 s of a 34.5 s Fock iteration, so
the pair transforms are no longer the largest term. Reading of the code:
`rebuild_exchange` in the Gamma valence path calls `build_spinor_mpb` and
`assemble_coulomb` on every Fock iteration, while the relaxed-core, scalar,
and second-variation paths keep a `CoreExchangeCache` /
`ScalarExchangeCache` / `SecondVariationExchangeCache` of compiled bases and
Coulomb operators across the Fock iterations of one outer step.
`compile_spinor_mpb_basis` and `build_spinor_mpb_from_basis` already split
the spinor build. Assigned to the codex pane: phase timers for the Gamma
Fock iteration, a timed A0 and A1 row 1 (class P), and, only if basis
compile plus Coulomb assembly is at least 30 percent of an A1 Fock
iteration, a `GammaExchangeCache` on the existing pattern with class R
checks, then A1 row 1 again under the plan cap. The pair-level MPI task of
evt-0025 is held until the profile is read, because it addresses only the
vertex build. Records continue at evt-0031 and evd-0011.

- state: h2-hf = active: Gamma Fock phase profile at A0 and A1 row 1, conditional exchange cache, then A1 (codex pane); MPI held
- note: h2-hf = A0 closed by evd-0010; A1 row 1 900 s per Fock iteration; Gamma path rebuilds basis and Coulomb every iteration

## 2026-09-08 · evt-0031 · h2-hf Gamma profiling started · actor: codex

Verified clean main `35c200cdcc8283c3ad6744267a7bbd49684045af` and clean
harness `17c615f72330343bd6b2f5fc7c7096415068ef78`, including evt-0030.
Task A changes only Gamma phase timers. Task B is class P, report-only:
two example runs at A0 (240 s cap) and A1 base (2400 s cap), with verbosity 2.
Task C is authorized only if A1 basis compilation plus Coulomb assembly
accounts for at least 30 percent of a completed Fock iteration. If enabled,
its fixed class-R bounds are fixture identities 1e-8 Ha, fixture energy
preservation 1e-10 Ha, and A0 E/HOMO/E_H/E_x preservation 1e-10 Ha against
evd-0010, with identities still within 1e-8. Budget: two fixtures before,
two after, one A0 at 2400 s; no failure diagnostics. MPI remains held.

- state: h2-hf = active: adding Gamma timers for the two bounded profiles
- note: h2-hf = evt-0030 conditional cache contract; no push; MPI held

## 2026-09-08 · evd-0011 · h2-hf Gamma phase profiles · main 79623b8b4765b76ce6341be30716f9052a4b339d

```text
STUDY / REPORT
Q: completed Fock phase seconds and shares; class: P; ref: none
bound: none; budget: two runs, A0 cap 240 s and A1 base cap 2400 s
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/gamma-profile-a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
log: evidence/2026-09-08-h2-hf-gamma-profile/a0.stdout.log and a0.stderr.log
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/gamma-profile-a1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
log: evidence/2026-09-08-h2-hf-gamma-profile/a1.stdout.log and a1.stderr.log
scope: A0 iterations 1 and 2 and A1 iteration 1 tabulated in the evidence README; neither profile is a plan row or SCF acceptance result. Both exited 124 at their caps (240.12 and 2400.31 s). Build passed; one static review of the delegated timer-only changes; no numerical tests beyond the two requested profile runs.
```

A0 iteration totals: 17.479391 and 15.698071 s. MPB: 8.108799 / 7.799805 s;
contraction: 4.875244 / 4.592004 s. A1 iteration total: 482.082668 s;
contraction: 290.551110 s (60.270%); MPB: 151.026190 s (31.328%);
spinor solve: 10.895720 s (2.260%); Coulomb assembly: 8.563591 s (1.776%).
The A1 vertex rebuild is 143.437000 s, so compile plus Coulomb is
151.026190 − 143.437000 + 8.563591 = 16.152781 s, **3.351%**, below 30%.
Task C is skipped. There are no Task C fixture checks or A0 rerun results.

## 2026-09-08 · evt-0032 · h2-hf cache condition failed; advance to the prescribed A1 retry · actor: codex

The fixed 30% cache threshold was not met (evd-0011: 3.351%). No cache,
contraction, pair-FFT, or MPI implementation is made. Task D reruns A1 base
once with its unchanged numerical settings and 1800 s cap, preserving the
prior log as `hf-a1-row1-cap1800.log`. A repeated base-row cap stops the task;
the budget does not authorize a diagnostic or extension.

- state: h2-hf = active: profile complete; cache skipped; A1 base retry under 1800 s cap
- note: h2-hf = evd-0011 compile plus Coulomb 3.351% below 30%; MPI held

## 2026-09-08 · evd-0012 · h2-hf A1 base profiled retry handoff · main 79623b8b4765b76ce6341be30716f9052a4b339d, recorded through 10f6b3925ed32fb4c84564c50dae3a84d47b50d1

```text
DIGIT / HANDOFF
Q: A1 base hartree_exchange (Ha); class: P; ref: 0
bound: none (study); Delta: unavailable
checks: supervisor exit 124 at 1800.76 s; three completed Fock iterations and no completed outer iteration; required finite converged energies, three identities within 1e-8 Ha, and final electron count 2 within 1e-8 not established
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a1-row1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a1-row1.log
log: examples/h2_dft/results/hf-a1-row1.log (main); prior evd-0010 log preserved as hf-a1-row1-cap1800.log
scope: one Task D run, no diagnostics. Last density residual 6.3664758060341759e-5; feedback residual 7.9480430635287458e-5 Ha. No convergence-floor claim.
unresolved: A1 base still cannot finish within its authorized 1800 s cap.
```

The controlling profile (evd-0011) measured basis compilation 7.589190 s
plus Coulomb assembly 8.563591 s = 16.152781 s of 482.082668 s, or 3.351%,
below the fixed 30% cache threshold. Contraction was 290.551110 s (60.270%).
Task C was skipped; there are no new fixture or A0 rerun results, and no
after-cache A0 wall time. The earlier A0 pass remains closed.

## 2026-09-08 · evt-0033 · h2-hf profile handed off after repeated A1 base cap · actor: codex

Task A timer-only change is main `79623b8`; main `10f6b39` records the profile
and retry and preserves the prior cap log. Task B completed its two capped
class-P profiles. Task C was skipped by its fixed condition. Task D stopped
on the repeated base-row cap, as explicitly required. A1 rows 2–10, A1v,
A2, B, and Bv were not run. MPI remains held; no MPI work was started.
Three numerical executions total, no diagnostics or repeated acceptance
checks, and no pushes.

- state: h2-hf = handoff: A1 base still exceeds 1800 s; Gamma cache condition not met
- note: h2-hf = evd-0011/0012; compile plus Coulomb 3.351%, contraction 60.270%; main 10f6b39; MPI held

## 2026-09-08 · evt-0034 · h2-hf A1 time read from the Gamma profile: exchange contraction, then vertex projection; BLAS-3 perf assigned, MPI still held · actor: claude

evd-0011 answers evt-0030 in the negative for the cache (basis compile plus
Coulomb assembly is 3.4 percent of an A1 Fock iteration) and names the
terms: `gamma.rebuild.contraction` 290.6 s of 482.1 s (60 percent) and
`gamma.rebuild.mpb` 151.0 s (31 percent, vertices 143.4 s) at A1 row 1;
at A0 the same two are 4.9 s and 8.1 s of 17.5 s. The A1 profile ran at
3.1 cores on average (user 7362 s over 2400 s wall). Reading of the code:
`CoulombVertexContractor::weighted_occupied_quadratic_sum`
(crates/mt-coulomb/src/operator.rs) gathers every (occupied, target)
vertex into one [n_aux, n_occ·n_target] tensor by a strided scatter (about
5 GB at A1), runs `einsum("ab,bj->aj")` through TBLIS, copies the result
and a conjugated weighted copy of the columns back to host, and finishes
with `einsum("aoi,aoj->ij")`; nothing is chunked, and the plain matrix
products do not go through the threaded faer matmul rstsr provides. The
interstitial projection at A1 has the same shape (`vv.interstitial.gemm`,
[pairs, n_raw] × [n_raw, n_pw] per chunk of 64). Assigned to the codex
pane: perf on both, per occupied band, threaded matmul, no multi-GB
intermediates, with class R checks against the first-rebuild exchange
feedback and evd-0010's A0 energies. Feasibility, stated for the user's
decision: at the exit tolerances A0's 1e-8 identity gate forced (evt-0022),
a run costs 56 rebuilds; even at a fifth of today's A1 iteration the A1
study is about 1.5 hours per row and A2 and B grow with product G and box,
so the ladder needs either a per-row identity bound below 1e-8 for study
rows (plan v2) or more machines running rows in parallel. MPI stays held.

- state: h2-hf = active: BLAS-3 perf on the exchange contraction and vertex projection (codex pane); ladder feasibility decision pending
- note: h2-hf = evd-0011 profile: contraction 60 percent, vertices 30 percent of an A1 Fock iteration at 3 cores

## 2026-09-08 · evt-0035 · h2-hf contraction and projection performance work started · actor: codex

Verified clean main `10f6b3925ed32fb4c84564c50dae3a84d47b50d1` and clean
harness `7864b80eacdc04a9157859d9534badf524de5c32`, including evt-0034.
The fixed class-R contracts are fixture identities within 1e-8 Ha; fixture
total/exchange energies within 1e-10 Ha of 10f6b39; maximum absolute
first-rebuild band-space feedback difference within 1e-10 of 10f6b39; and
A0 E/HOMO/E_H/E_x within 1e-10 Ha of evd-0010, with identities within 1e-8.
Budget: two baseline fixtures, two after Task 1, two after Task 2; two feedback
dumps; one A0 at 2400 s; one report-only A1 timing at 1200 s. A failure
hands off without diagnostics. MPI remains held; no A1 ladder or push.

- state: h2-hf = active: BLAS-3 contraction implementation and baseline capture
- note: h2-hf = evt-0034 fixed preservation bounds and finite run budget; MPI and ladder held

## 2026-09-08 · evt-0036 · h2-hf Task 1 fixture gates passed, projection stage started · actor: codex

The Task 1-only scratch snapshot passed the two prescribed fixture runs,
with and without fft-fftw. Class R: maximum identity residual
4.83771009363032078e-16 Ha against 1e-8; total/exchange energy differences
from 10f6b39 are 0 / 4.20128341838132968e-19 Ha against 1e-10. These
stage-specific passes are closed. The first-rebuild baseline feedback dump
contains one 1030-by-1030 k block (1,060,900 complex entries).
Proceed to Task 2: the existing FFTW-only projection uses the same borrowed
RSTSR/faer matmul route with 256 right bands per projection chunk; the
spectrum cache and Rayon structure remain unchanged.

- state: h2-hf = active: Task 1 fixtures closed; checking Task 2 projection
- note: h2-hf = fixed six-fixture budget; no extra diagnostics; MPI and ladder held

## 2026-09-08 · evt-0037 · h2-hf staged fixtures and feedback preservation closed · actor: codex

Task 2 passed its two prescribed fixture runs; its total/exchange differences
and maximum identity residual equal the accepted Task 1 values in evt-0036.
The after-both first-rebuild feedback comparison passed over all 1,060,900
finite entries: maximum absolute difference 1.21430643318376497e-16 against
the fixed 1e-10 bound, reference 10f6b39. Six fixture executions and both
feedback dumps are complete; none will be repeated.
Run the single A0 preservation check under 2400 s, comparing E/HOMO/E_H/E_x
to evd-0010 within 1e-10 Ha and retaining the plan's 1e-8 identities, then
the one report-only A1 timing under 1200 s. No A1 ladder, MPI, or diagnostics.

- state: h2-hf = active: fixtures and feedback passed; final A0 and A1 timing remain
- note: h2-hf = eight numerical executions used; two authorized runs remain

## 2026-09-08 · evt-0038 · h2-hf A0 preservation passed; final A1 timing started · actor: codex

The single A0 run exited 0, converged in eight outer iterations and 56 Fock
iterations, and took 1166.331817 driver seconds (1166.76 supervisor seconds).
Class-R absolute differences from evd-0010 for E/HOMO/E_H/E_x are
1.22124532708767219e-15 / 1.76803016671556179e-14 /
2.52575738102223113e-14 / 3.95516952522712018e-15 Ha, all below 1e-10.
Maximum driver identity residual is 1.72767455897115951e-9 Ha, below 1e-8;
final electron-count error is 5.32907051820075139e-15. This pass is closed.
Run only the remaining class-P first-iteration A1 timing under 1200 s,
including whole-run user/real utilization. It is not an A1 ladder row.

- state: h2-hf = active: all contraction preservation gates passed; A1 timing remains
- note: h2-hf = nine numerical executions used; final report-only timing running; no further checks

## 2026-09-08 · evd-0013 · h2-hf Task 1 contraction fixture preservation · main 16ff43682537c752902ecbb1440f43ae612b8d55

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identities (Ha); class: R; ref: fixture
bound: 1e-8; Delta: maximum 4.83771009363032078e-16
Q: fixture total/exchange energy preservation (Ha); class: R; ref: 10f6b39
bound: 1e-10; Delta: 0 / 4.20128341838132968e-19 in both feature variants
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw -- --nocapture
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
log: evidence/2026-09-08-h2-hf-contraction-perf/fixture-before-{fftw,direct}.log and fixture-task1-{fftw,direct,compare}.log
scope: two baseline and two Task 1-only scratch fixture executions; temporary prints only, no assertion changes. Source snapshot and environment are documented in the evidence README. Pass closed; no repeat or diagnostic.
```

## 2026-09-08 · evd-0014 · h2-hf Task 2 fixtures and full feedback preservation · main 700fa1f7c1ba467edff41363f8b5f51d2ee9589d

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identities (Ha); class: R; ref: fixture
bound: 1e-8; Delta: maximum 4.83771009363032078e-16
Q: fixture total/exchange energies (Ha); class: R; ref: 10f6b39
bound: 1e-10; Delta: 0 / 4.20128341838132968e-19 in both feature variants
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw -- --nocapture
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
log: evidence/2026-09-08-h2-hf-contraction-perf/fixture-task2-{fftw,direct,compare}.log
scope: two after-both fixture runs; stage-specific acceptance closed.

DIGIT / PASS
Q: every entry of first-rebuild band-space exchange feedback; class: R; ref: 10f6b39
bound: 1e-10 absolute; Delta: maximum 1.21430643318376497e-16
command: /opt/homebrew/bin/python3 compare_feedback.py feedback-before.bin feedback-after.bin
log: evidence/2026-09-08-h2-hf-contraction-perf/feedback-{before,after,compare}.log and feedback-{before,after}.bin
scope: one ordered 1030-by-1030 k block, 1,060,900 finite complex entries; two scratch first-feedback probes, each under 2400 s, both exit 0. Exact build/run commands are in the evidence README. Pass closed, no diagnostics.
```

## 2026-09-08 · evd-0015 · h2-hf A0 full-run preservation · main 700fa1f7c1ba467edff41363f8b5f51d2ee9589d, recorded through 5f40aac4c88cb36805fcba1d7494a6bec22a3514

```text
DIGIT / PASS
Q: A0 E/HOMO/E_H/E_x (Ha); class: R; ref: evd-0010
bound: 1e-10; Delta: 1.22124532708767219e-15 / 1.76803016671556179e-14 / 2.52575738102223113e-14 / 3.95516952522712018e-15
checks: converged, finite energies; maximum driver identity residual 1.72767455897115951e-9 Ha <= 1e-8; final electron-count error 5.32907051820075139e-15
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0-contract.log
command: /opt/homebrew/bin/python3 ../libmuffintin-harness/evidence/2026-09-08-h2-hf-contraction-perf/compare_a0.py examples/h2_dft/results/hf-a0-contract.log
log: examples/h2_dft/results/hf-a0-contract.log (main); evidence/2026-09-08-h2-hf-contraction-perf/a0-compare.log
scope: one run, exit 0, eight outer iterations and 56 Fock iterations. Driver wall 1166.331817 s, supervisor 1166.76 s; reference driver wall 1934.680438 s. Preservation pass closed, no diagnostic; no new external-energy or product-basis acceptance claim.
```

## 2026-09-08 · evd-0016 · h2-hf A1 first-iteration timing and utilization · main 700fa1f7c1ba467edff41363f8b5f51d2ee9589d

```text
STUDY / REPORT
Q: first-iteration contraction/MPB/Fock seconds and whole-run user/real; class: P; ref: evd-0011
bound: none; before: 290.551110 / 151.026190 / 482.082668 s, about 3.1 cores; after: 179.435197 / 96.477896 / 313.415279 s, 3.213291 cores
command: RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1200s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/contract-a1-timing --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
log: evidence/2026-09-08-h2-hf-contraction-perf/a1-timing.stdout.log, a1-timing.stderr.log, a1-summary.log
scope: one report-only run, cap exit 124; user 3857.62 s / real 1200.52 s, system 195.48 s. RSTSR/faer has a ten-thread pool; observed whole-run utilization does not establish sustained ten-core execution. No extra profiling was authorized or run. This is not a ladder row.
```

## 2026-09-08 · evt-0039 · h2-hf BLAS-3 performance task completed; ladder policy still held · actor: codex

Both scoped performance changes and their requested evidence are complete:
`16ff436` per-occupied-band faer GEMMs with 1 GB target tiles; `700fa1f`
256-right-band projection matmul with resident theta; `5f40aac` examples
records. All class-R gates passed; the report-only timing is recorded as
measured, including limited average core utilization. Ten numerical executions
used the exact budget, with no diagnostics or repeated acceptance checks.
MPI and the A1 ladder remain held. No push was performed.

- state: h2-hf = handoff: BLAS-3 perf gates closed; A1 tolerance-policy decision pending
- note: h2-hf = evd-0013 through evd-0016; A1 iteration 313.415279 s at 3.213 average cores; main 5f40aac; MPI held

## 2026-09-08 · evt-0040 · h2-hf pair-level MPI released (user decision), assigned to the codex pane · actor: user

The user chose to build the MPI layer now to cut wall time, with Snellius
job submission as the target, before deciding the A1 tolerance policy.
ADR-0006's first step is refined on the evd-0011 profile: occupied left
bands are distributed across ranks, each rank builds and contracts only its
own pair vertices, and one Allreduce of the band-space feedback per k
replaces any vertex allgather; binding rule unchanged (rsmpi `mpi` 0.8,
binary initializes at Funneled, library only checks `is_initialized`).
The codex pane executes with class R rank-independence checks and a class
P timing at ranks 1, 2, 4 on the Mac; Snellius runs are documented, not
executed, in this task. The A1 ladder stays held pending the tolerance
policy. Records continue at evt-0041 and evd-0017.

- state: h2-hf = active: pair-level MPI feature (codex pane); A1 ladder held pending the tolerance policy
- note: h2-hf = MPI released by the user for Snellius wall time; ADR-0006 first step refined to local contraction plus Allreduce

## 2026-09-08 · evt-0041 · h2-hf bounded pair-level MPI implementation started · actor: codex

Verified clean main `5f40aac4c88cb36805fcba1d7494a6bec22a3514` and clean
harness `476237f1f96f5937ca4b453f73b67208c5e117f6`, including evt-0040 and
the refined ADR-0006. The optional `mpi` feature uses safe rsmpi 0.8 over
mpi-sys 0.2; the binary owns `MPI_Init_thread` at Funneled and the library
never initializes or finalizes MPI. Distribution is round-robin over occupied
left bands; vertices and contractions stay local and one band-feedback
Allreduce per k restores identical SPMD state.

Fixed class-R contracts: fixture identities within 1e-8 for no feature,
one-process mpi feature, and `mpirun -n 2`; maximum first-rebuild feedback
difference for ranks 2 and 4 versus rank 1 within 1e-12; A0 rank-2
E/HOMO/E_H/E_x within 1e-10 Ha of evd-0010, identities within 1e-8, and
identical rank-final state. Class-P timing list: A1 first Fock iteration at
(1,10), (2,5), and (4,2), 900 s cap each. Budget: three fixtures, three
feedback dumps, one A0, three A1 timings; no diagnostics. The A1 ladder is
held and no push is authorized.

- state: h2-hf = active: implementing occupied-band MPI and fixed rank-count gates
- note: h2-hf = evt-0040; ten-run budget; A1 ladder held; no push

## 2026-09-08 · evt-0042 · h2-hf three fixture ranks passed; feedback dumps started · actor: codex

The complete three-run fixture set passed: ordinary default build, one-process
`fft-fftw,mpi`, and the MPI-enabled test binary under `mpirun -n 2` with
five Rayon threads per rank. The fixture initializes MPI at Funneled under
the feature and retains its Universe; the two-rank command launches the
already built test binary, not Cargo. Every rank ran the one exact fixture
and passed its unchanged 1e-8 identity assertions. This class-R set is closed.

Proceed to the one-build feedback dumps at ranks 1, 2, and 4. Only rank zero
writes the first-rebuild post-Allreduce block; all ranks stop together after
a scratch-only barrier. Compare ranks 2 and 4 with rank 1 over every finite
entry at the fixed 1e-12 absolute bound. No diagnostics or repeat on failure.

- state: h2-hf = active: fixture MPI gate passed; first-feedback rank dumps running
- note: h2-hf = three of ten executions complete; A1 ladder held; no push

## 2026-09-08 · evd-0017 · h2-hf MPI fixture identities · main 33a761d032540972126e6e7bdb6fb8b4f364b9ab, recorded through a0218d3577798187dc34c174162df77d57226db8

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: fixture
bound: 1e-8; Delta: every unchanged assertion passed
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw,mpi -- --nocapture
command: RAYON_NUM_THREADS=5 /opt/homebrew/bin/mpirun -n 2 target/release/deps/gamma_valence_hf-5a476f84e231c5b9 --exact gamma_hydrogen_rebuilds_full_vv_feedback_and_rejects_stale_orbitals --nocapture --test-threads=1
log: evidence/2026-09-08-h2-hf-mpi/fixture-default.log, fixture-mpi-n1.log, fixture-mpi-n2.log
scope: three runs; the feature-gated fixture initializes MPI at Funneled, registers world, and retains Universe. The two-rank command launches the built test binary, not two Cargo processes. Identity gate closed; no additional fixture check.
```

## 2026-09-08 · evd-0018 · h2-hf first-feedback probe execution handoff · main 33a761d032540972126e6e7bdb6fb8b4f364b9ab

```text
DIGIT / HANDOFF
Q: every first-rebuild band-space feedback entry at n=2 and n=4 versus n=1; class: R; ref: n=1
bound: 1e-12 absolute; Delta: unavailable
command: RAYON_NUM_THREADS=10 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n1.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 1 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n1 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
log: evidence/2026-09-08-h2-hf-mpi/feedback-n1.log and feedback-n1.bin
checks: rank zero flushed one 1030-by-1030 block with 1,060,900 complex entries after the first Allreduce; scratch process then bypassed Universe drop and MPI_Finalize through process::exit, so Open MPI returned exit 1. n=2 and n=4 were not run.
runs: one feedback probe; no diagnostic or replacement run authorized
unresolved: obtain clean MPI finalization for a first-rebuild-only probe without exceeding the fixed run budget.
```

## 2026-09-08 · evt-0043 · h2-hf MPI numerical acceptance handed off at the scratch probe · actor: codex

Implementation is on main: `f189669` optional binding, `33a761d` occupied-band
distribution and Allreduce, `88b212d` Snellius documentation, and `a0218d3`
the fixture/pass and probe/handoff record. The production example retains its
Universe and finalizes normally; the defect is confined to the untracked
first-rebuild hook. Stop That Digit forbids accepting an execution failure,
and the task permits no failure diagnostic. Therefore n=2/n=4 feedback,
n=2 A0, and n=1/n=2/n=4 A1 timing are unrun. The A1 ladder remains held.
Four numerical executions total; no push.

- state: h2-hf = handoff: MPI implementation built and fixtures pass; scratch first-feedback probe did not finalize
- note: h2-hf = evd-0017/0018; main a0218d3; rank-independence, A0, and timings unrun; A1 ladder held

## 2026-09-08 · evt-0044 · h2-hf MPI handoff answered: probe defect is the scratch exit, remaining rank checks re-authorized · actor: claude

Reviewed main `f189669`, `33a761d`, `88b212d`, `a0218d3` (no attribution
trailers; default-off `mpi` feature; binary-only Funneled initialization;
world-only setter via `SimpleCommunicator::world()` instead of `FromRaw`,
accepted as the first step since rsmpi's owning constructor would free
`MPI_COMM_WORLD`; round-robin occupied bands, local vertices, one Allreduce
per k; rank-zero output with an exact final-state agreement check). Both
branches are pushed at these tips. Ruling on evd-0018: the exit-1 came from
the untracked scratch hook calling `process::exit(0)` after its dump, which
skips the retained `Universe` drop and therefore `MPI_Finalize`; the
committed example finalizes normally. `feedback-n1.bin` was flushed after
the completed Allreduce and stands as the rank-1 reference; no rerun of
n=1. The hook is to return an error after the dump and the scratch main is
to map that error to a normal return, so every rank drops its `Universe`.
The unchanged contracts of evt-0041 are re-authorized with a budget of six
runs: n=2 and n=4 feedback dumps against `feedback-n1.bin` (1e-12, exit
code 0 required), A0 under `mpirun -n 2` against evd-0010 (1e-10, identities
1e-8, identical rank-final state), and the class P A1 first-iteration
timings at (1,10), (2,5), (4,2). No diagnostics; a failure hands off with
the value. Separately noted for the user's decision: the A0 trajectory in
evd-0010 shows every outer iteration restarting the Fock loop from the
exchange-free bands with `previous_global_feedback = None`, so outer
iterations 2..8 each spend seven rebuilds re-converging an already known
fixed point; loosening the exit tolerance from 1e-9 to 1e-7 would save one
inner iteration in seven. That reading supersedes the feasibility sentence
of evt-0034; the candidate lever is a warm start across outer iterations,
not a plan v2 on tolerances.

- state: h2-hf = active: MPI rank-independence, A0 n=2, and A1 timings re-authorized (codex pane); A1 ladder held
- note: h2-hf = evd-0018 answered: scratch-hook exit only; six runs re-authorized; warm-start reading supersedes evt-0034 feasibility

## 2026-09-08 · evd-0019 · h2-hf rank-two first-feedback mismatch · main a0218d3577798187dc34c174162df77d57226db8, recorded through c46223c

```text
DIGIT / HANDOFF
Q: every first-rebuild band-space feedback entry at n=2 versus n=1; class: R; ref: evidence/2026-09-08-h2-hf-mpi/feedback-n1.bin accepted by evt-0044
bound: 1e-12 absolute; Delta: maximum 2.91207674214817233e-2 at (k=0,row=0,column=3)
command: RAYON_NUM_THREADS=5 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n2.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 2 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
command: /opt/homebrew/bin/python3 compare_feedback.py feedback-n1.bin feedback-n2.bin | tee compare-feedback-n2.log
log: evidence/2026-09-08-h2-hf-mpi/feedback-n2.log, feedback-n2.bin, and compare-feedback-n2.log
checks: the scratch-only hook returned through main on both ranks, both retained Universes finalized, mpirun exited 0, and all 1,060,900 entries were finite and compared
runs: one of six re-authorized numerical runs; no diagnostic
unresolved: rank-two feedback is not rank-independent within the fixed bound. Per the stop rule, n=4 feedback, n=2 A0, and n=1/n=2/n=4 A1 timings are unrun; the A1 ladder remains held.
```

## 2026-09-08 · evt-0045 · h2-hf MPI continuation handed off on rank-two feedback · actor: codex

Evt-0044 authorized the scratch-only finalization fix and six remaining runs.
The hook now returns an error after dumping on every rank, and the scratch
example maps that probe error to a normal return so its retained Universe is
dropped. The first authorized run, n=2 with five Rayon threads per rank,
finalized normally and exited 0. Its first-rebuild feedback differs from the
accepted rank-one reference by `2.91207674214817233e-02` at
`(k=0,row=0,column=3)`, exceeding the fixed absolute `1e-12` bound. The
contract authorizes no diagnostics, so n=4 feedback, n=2 A0, and all three A1
timings were not run. The A1 ladder remains held. Main documentation is
`c46223c`; no push.

- state: h2-hf = handoff: rank-two first-rebuild feedback fails the fixed rank-independence bound; A1 ladder held
- note: h2-hf = evd-0019; one of six re-authorized runs used; n=4/A0/A1 unrun; no diagnostics; main c46223c

## 2026-09-08 · evt-0046 · h2-hf rank-two feedback mismatch is band gauge, not distribution; contract re-posed on the spectrum · actor: claude

Read of evd-0019 from the two dumps: the diagonal of the n=2 block agrees
with n=1 to 1.6e-15 and the sorted eigenvalues agree to 9.9e-16, while
off-diagonal entries such as (0,3) differ by 2.9e-2 with equal Frobenius
norm per degenerate block. That is a unitary rotation inside degenerate
band subspaces (Kramers pairs plus cubic-box degeneracies), whose gauge the
eigensolver does not reproduce across thread counts (ten threads at n=1,
five per rank at n=2). The distributed sum is therefore the same operator;
the evt-0041 quantity "every band-space entry" was ill-posed for degenerate
bands, and that was the coordinator's error in the brief. Re-posed
contract, fixed before any further run: Q = sorted eigenvalues of each
first-rebuild band-feedback block, ref n=1, bound 1e-12 absolute, class R,
checked by `evidence/2026-09-08-h2-hf-mpi/compare_feedback_spectrum.py`.
On the existing dumps n=2 passes (`compare-feedback-spectrum-n2.log`,
maximum 9.853e-16); no rerun. Remaining runs re-authorized under the
otherwise unchanged evt-0041 contracts: n=4 dump with the spectrum check,
A0 under `mpirun -n 2` against evd-0010, and the A1 first-iteration timings
at (1,10), (2,5), (4,2). Five runs, no diagnostics. Main `c46223c` and this
harness tip are pushed.

- state: h2-hf = active: MPI n=4 spectrum check, A0 n=2, and A1 timings re-authorized (codex pane); A1 ladder held
- note: h2-hf = evd-0019 read as band gauge; spectrum contract fixed; n=2 passes at 9.9e-16

## 2026-09-08 · evd-0020 · h2-hf rank-four feedback spectrum pass · main c46223c673c6a5e1d3595f2a7e7205cddf8b9260, recorded through b282484

```text
DIGIT / PASS
Q: sorted eigenvalues of each first-rebuild band-feedback block at n=4 versus n=1; class: R; ref: evidence/2026-09-08-h2-hf-mpi/feedback-n1.bin
bound: 1e-12 absolute; Delta: maximum 5.62050406216485499e-16
command: RAYON_NUM_THREADS=2 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n4.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 4 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n4 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
command: /opt/homebrew/bin/python3 compare_feedback_spectrum.py feedback-n1.bin feedback-n4.bin | tee compare-feedback-spectrum-n4.log
log: evidence/2026-09-08-h2-hf-mpi/feedback-n4.log, feedback-n4.bin, and compare-feedback-spectrum-n4.log
checks: mpirun exit 0; one 1030-dimensional feedback block; spectrum comparison passed. The existing n=2 dump separately passes at maximum 9.85322934354826430e-16 under evt-0046 with no rerun.
runs: one; spectrum contract closed without diagnostics or repeats
```

## 2026-09-08 · evd-0021 · h2-hf MPI A0 cap handoff · main c46223c673c6a5e1d3595f2a7e7205cddf8b9260, recorded through b282484

```text
DIGIT / HANDOFF
Q: A0 E/HOMO/E_H/E_x (Ha) and three identity residuals (Ha); class: R; ref: evd-0010
bound: 1e-10 for E/HOMO/E_H/E_x and 1e-8 for identities; Delta: unavailable
command: RAYON_NUM_THREADS=5 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s /opt/homebrew/bin/mpirun -n 2 target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-mpi-n2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
log: examples/h2_dft/results/hf-a0-mpi-n2.log on main b282484
checks: the 1800 s supervisor cap returned exit 124 before convergence, final energies, identities, driver wall_s, or the example's rank-agreement check. Forty exchange rebuilds completed; supervisor wall was 1800.12 s. compare_a0.py was not run.
runs: one; no diagnostic or repeat
unresolved: the rank-two MPI A0 acceptance quantities were not produced within the fixed cap. The three A1 timings are unrun under the stop rule; the A1 ladder remains held.
```

## 2026-09-08 · evt-0047 · h2-hf spectrum gates pass; MPI A0 handed off at the fixed cap · actor: codex

The re-posed spectrum contract passes for both remaining rank counts: the
existing rank-two dump differs from rank one by at most
`9.85322934354826430e-16`, and the new rank-four dump by at most
`5.62050406216485499e-16`, both below the absolute `1e-12` bound. The
rank-four launcher finalized normally and exited 0. The next authorized run,
A0 at two ranks and five Rayon threads per rank, completed 40 exchange
rebuilds but reached its 1800 s cap before producing final acceptance
quantities or executing the example's rank-agreement check. It returned exit
124 with supervisor wall 1800.12 s. Per the fixed stop rule, no comparison,
diagnostic, repeat, or A1 timing followed. Two of five authorized runs were
used; the A1 ladder remains held. Main documentation is `b282484`; no push.

- state: h2-hf = handoff: MPI feedback spectrum passes at n=2/n=4; A0 n=2 exceeds the fixed 1800 s cap; A1 ladder held
- note: h2-hf = evd-0020/0021; two of five runs used; A1 timings unrun; no diagnostics; main b282484

## 2026-09-08 · evt-0048 · h2-hf MPI rank checks read; A0 under two ranks deferred behind the warm start, which the user authorized · actor: user

Reviewed main `b282484` and harness `6aabfd1` (records only, no trailers);
both pushed. Reading of evd-0020/0021: the spectrum contract passes at n=2
(9.9e-16) and n=4 (5.6e-16), so the distributed first rebuild is the same
operator at every rank count. The A0 run under `mpirun -n 2` with five
threads per rank completed 40 rebuilds in 1800 s, 45 s per rebuild against
20.8 s at one process with ten threads (evd-0010, 1166 s / 56). On one
ten-core machine splitting cores across ranks halves the threads of every
redundant phase and shares the memory bandwidth, so a per-node loss was
expected; the feature's value is on nodes where ranks add cores, which is
what the A1 timing table is for. Its residual trajectory tracks evd-0010 to
about 1e-13 relative at every printed iteration, which is a reading, not a
stamp. The user authorized the warm start (evt-0044 reading) and assigned it
to the codex pane ahead of the remaining MPI runs: Gamma valence path only,
carry the final mixed global feedback across outer iterations and seed the
inner loop with one spinor solve, fixed point unchanged; class R fixture
identities 1e-8, A0 energies within 2e-8 Ha of evd-0010 (twice the outer
exit tolerance, the reason a changed inner path can end elsewhere inside
that floor), identities 2e-8; class P rebuild count with the pre-stated
prediction 12 to 24; two runs, no diagnostics. After it lands, the MPI A1
first-iteration timings at (1,10), (2,5), (4,2) and the A0 two-rank stamp
against the warm-start A0 log at 1e-10 with the cap raised to 3600 s.

- state: h2-hf = active: Fock-loop warm start (codex pane); MPI A1 timings and two-rank A0 stamp queued behind it; A1 ladder held
- note: h2-hf = MPI spectrum passes at n=2 and n=4; two-rank A0 45 s per rebuild on one machine; warm start authorized by the user

## 2026-09-08 · evd-0022 · h2-hf Gamma valence warm-start fixture pass and A0 cap handoff · main a5d52b3, recorded through 199eb49

The pre-run prediction was 12–24 exchange rebuilds and 7–9 outer iterations.
A result at or above 40 rebuilds was fixed as a handoff before execution.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: crates/mt-runtime/tests/gamma_valence_hf.rs
bound: 1e-8; Delta: every unchanged assertion passed
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw
log: evidence/2026-09-08-h2-hf-warm-start/fixture-fftw.log
checks: one release fft-fftw fixture passed
runs: one; fixture contract closed

DIGIT / HANDOFF
Q: A0 E/HOMO/E_H/E_x and three driver identities (Ha); class: R; ref: evd-0010
bound: 2e-8 absolute for energies and identities; Delta: unavailable
command: RAYON_NUM_THREADS=10 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-warm --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
log: evidence/2026-09-08-h2-hf-warm-start/a0.stdout.log and examples/h2_dft/results/hf-a0-warm.log on main
checks: ten-thread supervisor exit 124 at 1800.24 s before convergence, final electron count, or final energy and identity output
runs: one; no diagnostic or repeat
unresolved: A0 did not produce the class-R quantities within the fixed cap

DIGIT / HANDOFF
Q: A0 exchange_rebuilds, outer_iterations, wall_s; class: P; ref: 56, 8, 1166.331817 s
bound: report-only, with 12–24 rebuild prediction and mandatory handoff at 40 or more; Delta: 43 completed rebuilds, 7 completed outer iterations with outer 8 in progress, final driver wall_s unavailable
checks: supervisor wall 1800.24 s; rebuild threshold exceeded
runs: same A0 run; no diagnostic or repeat
unresolved: the warm start did not reduce rebuilds below the predeclared handoff threshold
```

## 2026-09-08 · evt-0049 · h2-hf warm start landed; A0 handed off at 43 rebuilds and the fixed cap · actor: codex

Main `a5d52b3` carries the final mixed Gamma valence global-basis feedback
between outer iterations and applies it once to the fresh H0/S bands before
the ordinary rebuild and CDIIS path. The first outer iteration, fixed-point
criteria, tolerances, gates, mixer, and other HF paths are unchanged. The
authorized fft-fftw fixture passed. The ten-thread A0 run then completed 43
rebuilds through outer iteration 8, above the predeclared 40-rebuild handoff
threshold, and reached the 1800 s cap before convergence or final acceptance
output. It exited 124 at supervisor wall 1800.24 s. The two-run budget is
exhausted; no diagnostic, repeat, MPI work, MPI-evidence edit, or A1-ladder
run followed. Main records are `199eb49`; no push.

- state: h2-hf = handoff: Gamma warm-start fixture passes, but A0 reaches 43 rebuilds and the 1800 s cap; A1 ladder held
- note: h2-hf = evd-0022; two-run budget exhausted; final A0 energies unavailable; main 199eb49; MPI evidence untouched

## 2026-09-08 · evt-0050 · h2-hf warm start read: the starting point moved, the CDIIS start-up still costs three iterations; follow-up assigned · actor: claude

Reviewed main `a5d52b3` (17 lines, Gamma path only, doc/23 sentence
updated) and `199eb49`; harness `05c31f4`; all pushed. Reading of
evd-0022 (`hf-a0-warm.log`): the warm start takes effect, outer 2 starts
at density residual 1.0e-4 instead of 1.2e-3 and outer 7 at 2.9e-8, yet
the inner loops still take 6, 6, 6, 5, 5, 5 iterations, 43 rebuilds by
outer 8 iteration 3 against 52 at the same point in evd-0010. The
trajectory shows why: `FeedbackMixer` runs `startup_steps` = 2 damped
mixes at weight 0.5 that are not pushed into the history, then one plain
step with a one-record history, and only then extrapolates; at outer 7 the
residual sits at 2.9e-8, 1.7e-8, 1.9e-8 for three iterations and then
drops to 3.1e-9 and 9.8e-13 in two. The pre-run prediction of 12 to 24
rebuilds assumed the inner loop converges in one or two steps from a close
start and was wrong for that reason. Wall time is not readable from this
run: the machine carried a Zoom call with screen sharing during it (load
average 71 over 15 minutes on 10 cores), and the 42 s per rebuild against
20.8 s in evd-0010 is that load, not the change. Follow-up assigned to the
codex pane under the same authorization: in warm-started inner loops
construct the mixer with zero start-up steps so extrapolation begins at
the second record; cold outer 1 unchanged. Prediction stated now: A0 at 22
to 34 rebuilds. Same class R contracts as evt-0048 (fixture 1e-8; A0
energies within 2e-8 Ha of evd-0010; identities 2e-8), cap raised to
3600 s because of the shared machine, load average recorded before and
after the run as context.

- state: h2-hf = active: warm-start mixer follow-up (codex pane); MPI A1 timings and two-rank A0 stamp queued behind it; A1 ladder held
- note: h2-hf = evd-0022 read: warm start effective, CDIIS start-up costs three iterations per outer; wall polluted by machine load

## 2026-09-09 · evd-0023 · h2-hf zero-start-up warm-loop fixture and A0 pass · main 762c210, recorded through 9d64c5d

The pre-run prediction was 22–34 exchange rebuilds, 7–9 outer iterations,
and 2–4 iterations per warm inner loop. A result at or above 40 rebuilds was
fixed as a handoff before execution.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: crates/mt-runtime/tests/gamma_valence_hf.rs
bound: 1e-8; Delta: every unchanged assertion passed
command: cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw
log: evidence/2026-09-08-h2-hf-warm-start/fixture-warm2-fftw.log
checks: one release fft-fftw fixture passed
runs: one; fixture contract closed

DIGIT / PASS
Q: A0 E/HOMO/E_H/E_x and three driver identities (Ha); class: R; ref: evd-0010
bound: 2e-8 absolute for energies and identities; Delta: 2.48245868306185002e-13 / 3.09803882547754483e-9 / 1.62302338235775778e-10 / 8.68805014042628443e-10 for E/HOMO/E_H/E_x
command: RAYON_NUM_THREADS=10 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 3600s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-warm2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
log: evidence/2026-09-08-h2-hf-warm-start/a0-warm2.stdout.log and compare-a0-warm2.log; examples/h2_dft/results/hf-a0-warm2.log on main
checks: finite converged result; exit 0; maximum identity 4.76458923703848569e-9; electron-count error 6.66133814775093924e-15
runs: one; class-R contract closed

DIGIT / PASS
Q: A0 exchange_rebuilds, outer_iterations, per-outer Fock iterations, wall_s; class: P; ref: evd-0010 and evd-0022
bound: report-only, with 22–34 rebuild prediction and mandatory handoff at 40 or more; result: 32 rebuilds, 8 outer iterations, 7 / 5 / 4 / 4 / 3 / 3 / 3 / 3 iterations, wall_s 1228.894974
checks: rebuild and outer-count predictions passed; first warm loop took 5 iterations, one above the report-only 2–4 prediction; all later warm loops took 3–4; supervisor wall 1229.77 s
load: before 10.78 / 28.99 / 37.53; after 27.81 / 27.57 / 28.24
runs: same A0 run; closed without diagnostics or repeats
```

## 2026-09-09 · evt-0051 · h2-hf zero-start-up warm-loop follow-up passes · actor: codex

Main `762c210` sets `startup_steps` to zero only for warm-started Gamma
valence `CommutatorDiis` and `QuasiNewtonDiis` loops. Cold outer iteration 1,
the carried-feedback history boundary, configured damping and history, all
gates and tolerances, spec structs, and other HF paths remain unchanged. The
authorized fft-fftw fixture passed. The ten-thread A0 run converged in 32
rebuilds over eight outer iterations, with per-outer counts
7 / 5 / 4 / 4 / 3 / 3 / 3 / 3. Its energies pass the fixed 2e-8 Ha bounds,
the maximum identity is `4.76458923703848569e-9`, and the electron-count
error is `6.66133814775093924e-15`. Driver wall is 1228.894974 s and
supervisor wall is 1229.77 s, with load averages recorded in evd-0023. Two
of two runs were used. Numerical verification is closed; no diagnostic,
repeat, MPI work, MPI-evidence edit, or A1-ladder run followed. Main records
are `9d64c5d`; no push.

- state: h2-hf = active: zero-start-up Gamma warm loop accepted; MPI A1 timings and two-rank A0 stamp remain queued; A1 ladder held
- note: h2-hf = evd-0023; A0 passes in 32 rebuilds and 8 outers; first warm loop 5 iterations, later loops 3–4; main 9d64c5d

## 2026-09-09 · evt-0051 · ctf-rs plan.v2 proposed: one breaking change to rsmpi ownership; ADR-0007; gate G-CTF-R1 · actor: claude

Read-only comparison of the three MPI users (libmuffintin `main`,
rustnumgum/fftw `fe731a02`, ctf-rs `origin/master` `f2039d3`): one
`mpi-sys` 0.2.4 per build, three ownership models. ctf-rs's
`Runtime::initialize` calls `MPI_Init` at Single and asserts MPI is
uninitialized, so it cannot share a process with a host `Universe`. The
user's direction: unify initialization and communicator ownership under
rsmpi in one breaking change; hand-written wrappers may coexist; keep the
thread markers and check the provided level; no libffi; MPI-IO and any
variable-count reduce-scatter stay on `mpi::ffi`; explicit `close` stays the
only collective free. Recorded as `plans/ctf-rs/plan.v2.md` (proposed,
supersedes v1, whose D1 to D6 are closed) and ADR-0007, which supersedes the
ADR-0006 lines on raw-handle crossing and on a raw adopt constructor. Gate
G-CTF-R1 added: class R, upstream tolerances unchanged, WSL 1/2/4 once,
native build once, D6 native set once. No code was changed and no test was
run. The Mac clone `~/tmp/ctf-rs` is stale (`f22da3a`, diverged) and is not
the baseline. `update-status.py` now orders state lines by entry date and
then number; before this the MSI id range always won, so this entry's
state would not have appeared in `STATUS.md`.

- state: ctf-rs = plan.v2 proposed: R1 rsmpi binding (one breaking change) before S1; awaiting acceptance
- note: ctf-rs = D1 to D6 closed under v1; R1 gate G-CTF-R1 proposed; executor MSI at origin/master f2039d3

## 2026-09-09 · evt-0052 · h2-hf warm-loop acceptance ID collision superseded · actor: claude

The codex entry "h2-hf zero-start-up warm-loop follow-up passes" above
reused `evt-0051` after that ID had been assigned concurrently to the
ctf-rs plan.v2 proposal by another Mac writer. This append-only correction
supersedes the h2-hf entry's heading ID as `evt-0052`; its evidence
(evd-0023), state, and note remain unchanged. Two Claude sessions and one
codex pane now write this worktree on the Mac; each is to read the ledger
tail immediately before appending.

- state: h2-hf = active: zero-start-up Gamma warm loop accepted; MPI A1 timings and two-rank A0 stamp remain queued; A1 ladder held
- note: h2-hf = evd-0023; A0 passes in 32 rebuilds and 8 outers; main 9d64c5d

## 2026-09-09 · evt-0053 · h2-hf warm-start step closed at 32 rebuilds; precision state; herdr callbacks blocked · actor: claude

Reviewed main `762c210` (mixer start-up override for warm loops only,
doc/23 sentence) and `9d64c5d`; both pushed with harness. Reading of
evd-0023: the A0 run needs 32 rebuilds (7/5/4/4/3/3/3/3) against 56 in
evd-0010, a 43 percent cut at the same tolerances and gates, with energies
inside the outer floor (largest deviation HOMO 3.1e-9 Ha) and identities at
4.8e-9. Wall 1229 s is not comparable: load average 28 to 37 on ten cores
throughout; at the idle 20.8 s per rebuild of evd-0010 the run would be
about 11 minutes, and an A1 row at 313 s per rebuild about 2.8 hours on
this machine. Precision state for the user: the driver identities hold at
1e-9 at A0; the physical numbers at A0's smoke settings (orbital 4, product
4, l 2) are E −0.5929, HOMO −0.1262, E_H 0.3729, E_x −0.0232 against the
plan's box-8 reference −0.8124, −0.2641, 0.5951, −0.2976, and the
hartree_exchange identity misses by 0.163 Ha; no row at A1 settings has
completed, so whether that miss is product-basis convergence or a defect is
still the open question A1 and A1v were written to answer. Coordination
constraint: the herdr client on this machine is at protocol 22 and the
running server at 20 after an upgrade, so `claude-report` callbacks and
`herdr agent prompt` dispatches fail until the server is restarted, which
ends every pane; the user will decide when. MPI-4 (A1 first-iteration
timings at (1,10), (2,5), (4,2) and the two-rank A0 stamp against
`hf-a0-warm2.log` at 1e-10) and the A1 ladder wait on that and on the
tolerance-policy decision, which the warm start has narrowed to: run the
ten A1 rows as written, in parallel where machines allow.

- state: h2-hf = held: warm start closed (32 rebuilds); MPI-4 and the A1 ladder wait for the herdr restart and the row-placement decision
- note: h2-hf = A0 smoke numbers 0.2 Ha from the box-8 reference, hartree_exchange 0.163 Ha; nothing at A1 settings yet; herdr protocol 22 vs 20

## 2026-09-09 · evt-0054 · ctf-rs plan.v2 accepted; R1 rsmpi binding assigned to Codex on MSI · actor: user

The user accepted `plans/ctf-rs/plan.v2.md` ("deploy the migration to
rsmpi on MSI, the whole migration") and ADR-0007 with it; gate G-CTF-R1 is
accepted. Executor: Codex on MSI in `D:/projects/ctf-rs` at `origin/master`
`f2039d3`, clean at assignment; the Mac clone stays out of it. Execution
follows Stop That Digit exactly as the plan writes it: R1 as one commit
series in the plan's order, the class R acceptance runs once each
(`scripts/acceptance-wsl.sh` at 1, 2, 4 ranks, `scripts/acceptance-native.ps1
-BuildOnly`, the D6 native runtime set at 1, 2, 4), at most the three named
diagnostics per failing driver, then `DIGIT / HANDOFF`; no tolerance,
metric, fixture, or driver change; no dual API; no push of ctf-rs without a
separate request. Records: ctf-rs `docs/validation.md` section "rsmpi
binding"; on `harness-msi` (after `git merge github/harness`) evd-1009 and
the next free evt in the MSI range, pushed to `github/harness-msi`; the Mac
merges. Because the Mac's herdr server is down for an upgrade, the report
channel is that push, not `claude-report`.

- state: ctf-rs = active: R1 rsmpi binding (Codex on MSI); G-CTF-R1 accepted
- note: ctf-rs = plan.v2 accepted 2026-09-09; executor MSI at f2039d3; report by harness-msi push

## 2026-09-09 · evd-1009 · ctf-rs R1 rsmpi binding acceptance handoff · ctf-rs 813d90a8f76a4f67e21c5b28340d014f4ce2dfc3

Gate `G-CTF-R1`, accepted plan.v2 and ADR-0007. Implementation ends at
`f072971`; validation/ownership records end at `813d90a`. The ordered local
master series starts after `f2039d3`; ctf-rs was not pushed. All 184 driver
entry points changed mechanically; no algorithm, metric, tolerance, fixture,
driver body, or acceptance-script changes. Host communicator ownership and
Universe lifetimes are explicit; no Runtime, libffi, or collective Drop.

```text
DIGIT / HANDOFF
Q: each WSL driver's existing metric and invariants; class: R
ref: pinned upstream f69cbb46; bound: upstream per driver, unchanged
Delta: cyclic_reshuffle, two ranks, rank 1 local offset 0: actual 0,
       expected -5; absolute difference 5 against exact bound 0
checks: all 175 one-rank drivers completed; eight two-rank drivers passed,
        ninth cyclic_reshuffle failed; rank 4 and subsequent local/7-rank calls not reached
runs: acceptance-wsl.sh once (its own 1/2/4 loop); script exit 1 after the
      failed launcher's blocked MPI peers were terminated
diagnostics: one computation, named diagnostic 1 at two ranks; both ranks
             size 2, Funneled, main-thread check passed; same mismatch before
             explicit close/drop; 60-second supervisor exit 124 after panic
scope: no numerical repair, baseline rerun, sweep, changed bound, or repeated pass
open: what R1-scoped resolution is authorized for cyclic_reshuffle's exact
      replica/layout mismatch (0 versus -5) with correct rank/thread setup?

DIGIT / PASS
Q: native Windows GNU compile/link of all tests and examples; class: R
ref: plan.v2 build gate; bound: compile/link success, exit 0
Delta: no compile/link failure; numerical delta not applicable
runs: acceptance-native.ps1 -BuildOnly once

DIGIT / PASS
Q: native D6 drivers' existing metrics and invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: all unchanged assertions passed; emitted numerical deltas retained in log
checks: 50 dense drivers at each of 1/2/4 ranks and four local scaling tests passed
runs: acceptance-native.ps1 -D6Only once; each rank once; exit 0; no diagnostics
closed: native numerical verification; overall R1 remains HANDOFF
```

Exact commands, run from `D:/projects/ctf-rs`:

```powershell
wsl -d Ubuntu-26.04 -- bash -lc 'cd /mnt/d/projects/ctf-rs && bash scripts/sync-wsl.sh && cd /home/xylxp/ctf-rs-work && bash -x scripts/acceptance-wsl.sh > /mnt/d/projects/runs/ctf-rs-r1/wsl.log 2>&1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -BuildOnly > D:/projects/runs/ctf-rs-r1/native-build.log 2>&1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -D6Only > D:/projects/runs/ctf-rs-r1/native-d6.log 2>&1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1/diagnostic-1.sh > D:/projects/runs/ctf-rs-r1/diagnostic-1.log 2>&1
```

Logs and commands remain under `D:/projects/runs/ctf-rs-r1/`; `commands.md`
records the diagnostic instrumentation and exact nested command, the tooling
quoting correction before its one execution, stale Linux copy cleanup, and the
single keepalive startup. Temporary diagnostic printing was removed from the
Linux work copy; delivery source was not instrumented. ctf-rs
`docs/validation.md` section "rsmpi binding" carries these stamps. Total:
three acceptance-script invocations, one named diagnostic computation, no
repeated passing driver at the same platform/rank configuration.

## 2026-09-09 · evt-1001 · ctf-rs R1 implementation recorded with WSL gate handoff · actor: codex

The ordered R1 commit series is local on MSI master through `813d90a`.
Native build and the native D6 1/2/4 set pass. WSL one-rank acceptance passes,
but two-rank cyclic_reshuffle fails exactly with 0 instead of -5; the named
initialization diagnostic reproduces it with valid rank/thread setup. Per the
immutable plan, no numerical or driver repair followed and R1 is not closed.
S1 has not started. Evidence is evd-1009; this harness-msi push is the report
channel, not claude-report, and no ctf-rs push was made.

- state: ctf-rs = handoff: R1 rsmpi implementation committed; WSL two-rank cyclic_reshuffle fails exact 0 versus -5; native build and D6 1/2/4 pass; S1 held
- note: ctf-rs = evd-1009; local master 813d90a; one named diagnostic, valid Funneled/main-thread setup; no numerical repair or ctf-rs push

## 2026-09-09 · evt-0055 · ctf-rs R1 handoff answered: two-rank cyclic_reshuffle is an MPI-layer translation defect; diagnostic 3 sharpened on Context::exchange, fix and one rerun authorized · actor: claude

Merged `harness-msi` c10b6d3 (evd-1009, evt-1001) into `harness`. Reading
of the handoff: R1 changed only the MPI layer (`src/ffi/mpi.rs`),
`Context` construction, and the test and example mains; no reshuffle,
pack, or unpack code changed, and 175 one-rank drivers plus eight
two-rank drivers pass, including `custom_reduce` (user operation),
`ctr_2d` (broadcast and root Reduce), and `contract_remap`. A two-rank
value of 0 where −5 is expected at rank 1 offset 0 therefore comes from a
wrapper whose two-rank form the passing drivers do not exercise; the
cyclic reshuffle reaches MPI through `Context::exchange`, whose rsmpi form
(`all_to_all_into` for counts, `all_to_all_varcount_into` with
`Partition`/`PartitionMut` for byte payloads) is the first candidate. The
other wrappers read faithful to their raw predecessors: `redistribute_ror`
still issues raw Isend/Irecv, `send_receive`, `replace_wire`, `split`, and
`reduce_monoid` map argument for argument. Ruling: diagnostic 2 does not
apply to a mid-run wrong value and is not run; diagnostic 3 is run as the
plan names it, sharpened to a per-call trace of counts and bucket hashes
in `exchange` at 813d90a and at f2039d3 on the failing test at two ranks;
the first divergent line names the wrapper. A translation defect so found
is R1-scoped and is fixed in one `fix(mpi)` commit, followed by the R1
acceptance rerun once as plan.v2 section 4 states. Identical traces are a
handoff with the traces. The report channel is `claude-report` again (the
Mac herdr server was restarted to 0.9.0, protocol 22); the harness-msi
push stays the durable record.

- state: ctf-rs = active: R1 diagnostic 3 on Context::exchange, fix and one acceptance rerun authorized (Codex on MSI)
- note: ctf-rs = evd-1009 handoff read as an MPI-layer translation defect; exchange is the first candidate; S1 held

## 2026-09-09 · evd-1010 · ctf-rs R1 diagnostic 3 identical exchange traces and baseline failure · ctf-rs 622ef106fdff963e1f8f3d3e43a28b98f9162182

Executed evt-0055 / BRIEF-2's sharpened diagnostic on scratch archives of
candidate `813d90a` and raw-MPI baseline `f2039d3`, with separate WSL target
directories. Delivery source was not instrumented. Only the bool/complex/Word
calls were omitted to select i8; no assertions, layouts, initialization,
splits, or cleanup changed. The five original R1 commits stand.

```text
DIGIT / HANDOFF
Q: per-call/rank send_counts, recv_counts and sent/received bucket FNV-1a-64
   hashes in Comm::exchange; existing exact i8 cyclic_reshuffle assertions
class: R; ref: f2039d3; candidate: 813d90a
bound: exact trace equality and unchanged exact driver bound 0
Delta: zero differing fields across two trace rows, call 0 at ranks 0 and 1
driver result: both revisions fail rank 1 local offset 0, actual 0, expected -5;
               absolute difference 5 against exact bound 0
runs: two total, one two-rank diagnostic run per revision; no repeats
exits: candidate 124 at the 60-second supervisor after panic; baseline 101
       after the same assertion panic and MPI process failure
finding: identical traces, no first divergent line, no wrapper named for repair;
         the pre-R1 baseline reproduces the same driver failure
scope: no wrapper fix, no further diagnostics, diagnostic 2 not applicable
open: what scope is authorized for the baseline failure with identical
      exchange traces, rather than an R1 wrapper translation defect?
```

Both attached traces contain these same complete tuples, sorted by call/rank:

```text
call=0 rank=0 send_counts=[10, 0] recv_counts=[10, 0] sent_hashes=[3bc129cffea72d2d, cbf29ce484222325] received_hashes=[3bc129cffea72d2d, cbf29ce484222325]
call=0 rank=1 send_counts=[0, 10] recv_counts=[0, 10] sent_hashes=[cbf29ce484222325, 4a98c824fe6e7321] received_hashes=[cbf29ce484222325, 4a98c824fe6e7321]
```

Attachments: [candidate trace](evidence/2026-09-09-ctf-rs-r1-2/current.trace),
[baseline trace](evidence/2026-09-09-ctf-rs-r1-2/baseline.trace),
[candidate log](evidence/2026-09-09-ctf-rs-r1-2/current.log),
[baseline log](evidence/2026-09-09-ctf-rs-r1-2/baseline.log),
[commands and instrumentation](evidence/2026-09-09-ctf-rs-r1-2/commands.md),
[exact nested commands](evidence/2026-09-09-ctf-rs-r1-2/diagnostic-3.sh).
Raw logs are UTF-16 PowerShell output; extraction joined wrapped whitespace
across each full tuple without dropping counts or hashes. Originals remain
under `D:/projects/runs/ctf-rs-r1-2/`.

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh current > D:/projects/runs/ctf-rs-r1-2/current.log 2>&1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh baseline > D:/projects/runs/ctf-rs-r1-2/baseline.log 2>&1
```

The script builds the single test without execution first, then runs it once
with `mpirun --oversubscribe -n 2`, a 60-second timeout and five-second kill
grace. The existing keepalive was reused; no diagnostic processes remain.
This establishes the baseline failure, not global R1 behavioral equivalence.

## 2026-09-09 · evd-1011 · ctf-rs R1 conditional acceptance rerun not executed · ctf-rs 622ef106fdff963e1f8f3d3e43a28b98f9162182

```text
DIGIT / HANDOFF
Q: G-CTF-R1 acceptance; class: R; ref: pinned upstream f69cbb46
bound: upstream metrics/tolerances unchanged; Delta: no new acceptance value
runs: zero acceptance reruns in CTF-R1-2
reason: BRIEF-2 mandates handoff for identical diagnostic traces; evd-1010
        identifies no translation fix, so the after-a-fix rerun is not triggered
closed evidence: evd-1009 native build and D6 1/2/4 passes remain closed;
                 no new pass or full R1 closure is claimed
```

## 2026-09-09 · evt-1002 · ctf-rs R1 diagnostic 3 hands off identical traces · actor: codex

The two authorized computations produced identical exchange traces and the
same exact i8 failure at `813d90a` and `f2039d3`. Per BRIEF-2's identical-trace
branch, no `fix(mpi)` commit or acceptance rerun was made, and numerical work
stopped. ctf-rs `622ef10` only adds the finding/stamps to "rsmpi binding" in
validation.md, without rewriting the five R1 commits or pushing ctf-rs.
Both traces and raw logs accompany evd-1010; evd-1011 records the untriggered
rerun. This harness-msi push is the durable report; claude-report is notified
after the pushed tip is known.

- state: ctf-rs = handoff: R1 diagnostic 3 traces identical at 813d90a and f2039d3; both fail exact 0 versus -5; no wrapper fix or acceptance rerun; S1 held
- note: ctf-rs = evd-1010 and evd-1011; two diagnostic runs; local master 622ef10; baseline failure reproduced; traces attached; no ctf-rs push

## 2026-09-09 · evt-0056 · ctf-rs R1 faithful on cyclic_reshuffle; the two-rank failure predates R1 and is opened as B1 against the ungated formatting commit · actor: claude

Merged `harness-msi` ecfc4a8 (evd-1010, evd-1011, evt-1002). Reading:
diagnostic 3 gives identical `Context::exchange` traces at 813d90a and at
the raw-MPI baseline f2039d3, and both fail the same exact assertion
(rank 1 offset 0, 0 against −5). R1 therefore reproduces the baseline on
this driver and is not the cause; no wrapper translation is at fault. The
baseline itself is the regression: the D6 close 354cf7d passed the WSL
gate at 1, 2, 4 ranks (evd-1008), and the following commit f2039d3,
labelled "style: finalize restored Rust formatting", changed 64 files
with 6605 insertions and 2256 deletions including `src/tensor.rs`,
`src/mapping.rs`, `src/map_tensor.rs`, and `src/context.rs`, and was never
gated. Opened as item B1 under the ctf-rs workstream with a fixed budget:
one two-rank run of `tests/cyclic_reshuffle.rs` at 354cf7d; if it passes,
a non-whitespace diff of f2039d3 on the redistribute path names the hunk
and one run with that hunk reverted verifies the naming; then one `fix`
commit restoring the 354cf7d semantics on top of 622ef10, and the R1
acceptance once in full (the WSL script running past every driver at 1,
2, 4, native build, D6 native set). A pass closes R1 and B1 together. If
354cf7d also fails, handoff. Lesson recorded: a commit labelled style
with thousands of changed lines is not exempt from the gate; the
executor is to run the WSL gate on every commit that reaches master.

- state: ctf-rs = active: B1 bisect of the ungated formatting commit f2039d3, then the R1 acceptance rerun (Codex on MSI)
- note: ctf-rs = R1 faithful on cyclic_reshuffle (identical traces); failure predates R1; suspect f2039d3; S1 held

## 2026-09-09 · evt-0057 · ctf-rs B1 bisect stopped by the user; the two-rank cyclic_reshuffle defect is to be fixed directly · actor: user

The user stopped the B1 bisect ("fix the bug directly, do not look for
the commit; I broke it"). The history search of evt-0056 is withdrawn;
what it produced so far (354cf7d not passing in today's environment, the
683bfd8 endpoint not building without three BLAS declarations) is noted
here and not pursued. The executor now finds the defect in the current
tree from the failing assertion along the redistribute path (tensor,
mapping, glb_cyclic_reshuffle, dgtog_redist), with `Context::exchange`
already cleared by evd-1010, fixes it as one commit on top of 622ef10
without touching the test, confirms the single test at 1, 2, 4 ranks
once, and then runs the R1 acceptance once in full. A pass closes R1; B1
closes with the fix. Stop That Digit unchanged for the gates.

- state: ctf-rs = active: direct fix of the two-rank cyclic_reshuffle defect, then the R1 acceptance rerun (Codex on MSI)
- note: ctf-rs = bisect withdrawn by the user; fix in the current tree; R1 faithful per evd-1010; S1 held

## 2026-09-09 · evd-1012 · ctf-rs direct replica-restoration fix and single-test confirmation · ctf-rs a558e30c57bcb0e12abf18c8d2906237d41ccf90

The withdrawn B1 bisect scratch trees, build patches and target directories
were discarded without further history investigation.

Current-tree inspection and one scratch-only two-rank diagnostic identified
missing replica restoration in `Tensor::redistribute`. Both optimized paths
(equal-phase block reshuffle and DGTOG ROR) return primary-layer-only buffers;
the public Tensor layout requires values on every owning rank. In the first
failing `[3,5]` cyclic-to-unmapped transition, rank 1 has `receive_root=false`,
receive counts `[10,5]` and 15 valid but unpopulated local slots, so offset 0
contains 0 instead of key 0's i8 value -5. Rank 0 unpacks correctly; the MPI
exchange remains cleared by evd-1010.

Fix `a558e30` changes only `src/tensor.rs`: after either optimized path, group
ranks by mapped physical residues, order by original rank so the canonical
owner is subgroup rank 0, broadcast that local block and explicitly close
the subgroup. This restores scalar/partial/full replicas and preserves zero
padding, low-level primary-only primitives and the legacy path. No test,
driver, fixture, tolerance or acceptance script changed. The fix is on top
of `622ef10`, with no rewritten commits and no ctf-rs push.

```text
DIGIT / PASS
Q: cyclic_reshuffle exact values/padding/replicas, virtual phases, empty/scalar,
   i8/bool/complex/non-Copy Wire, world/parity contexts; class: R
ref: unchanged test assertions; bound: exact equality, 0
Delta: before fix rank 1 offset 0 difference 5, actual 0 versus expected -5;
       after fix all unchanged assertions pass, exact difference 0
runs: one pre-fix two-rank diagnostic, supervisor exit 124 after assertion panic;
      one post-fix confirmation at each of 1/2/4 ranks, exit 0
closed: single-test confirmation, no further diagnostics after the fix
```

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/diagnostic.sh
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/confirm.sh
```

Logs: `D:/projects/runs/ctf-rs-r1-4/diagnostic.log` and `confirmation.log`.
The scripts and `commands.md` beside them give exact nested commands and
scratch instrumentation. Only scratch code was instrumented, not delivery
source. The existing keepalive was reused.

## 2026-09-09 · evd-1013 · ctf-rs full R1 acceptance passes after replica restoration · ctf-rs 2b684061b7f5bf2699d646ec4113c04084789f71

Implementation `a558e30`, records `2b68406`, gate G-CTF-R1 under plan.v2 and
BRIEF-4. All prescribed scripts ran unchanged, including the WSL script's
complete selection and post-loop checks.

```text
DIGIT / PASS
Q: every WSL driver's own metric and required invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: all unchanged assertions passed; emitted numerical values preserved in log
checks: all 175 MPI drivers at each of 1/2/4 ranks, followed by all prescribed
        library/local checks and the seven-rank Strassen invocation
runs: acceptance-wsl.sh once, each configured rank set once; exit 0

DIGIT / PASS
Q: native Windows GNU compile/link of all tests/examples; class: R
ref: plan.v2 native build gate; bound: successful compile/link, exit 0
Delta: no compile/link failure; numerical delta not applicable
runs: acceptance-native.ps1 -BuildOnly once

DIGIT / PASS
Q: native D6 drivers' own metrics/invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: all unchanged assertions passed; emitted numerical values preserved in log
checks: 50 dense drivers at each of 1/2/4 ranks, plus four local scaling tests;
        150 driver PASS stamps, 151 target executions
runs: acceptance-native.ps1 -D6Only once, each rank set once; exit 0
closed: G-CTF-R1 and direct replica-restoration defect; no post-gate diagnostics
```

Exact commands (native commands from `D:/projects/ctf-rs`):

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/acceptance.sh
$env:CARGO_BUILD_JOBS='2'
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -BuildOnly > D:\projects\runs\ctf-rs-r1-4\native-build.log 2>&1"
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -D6Only > D:\projects\runs\ctf-rs-r1-4\native-d6.log 2>&1"
```

Logs stay outside source under `D:/projects/runs/ctf-rs-r1-4/`: `wsl.log`,
`native-build.log`, `native-d6.log`; `commands.md` and `acceptance.sh` retain
the exact nested commands. ctf-rs validation.md "rsmpi binding" records
the stamps and coverage.md's phase 1 row now records R1 acceptance.

Scope caveat: the separate `dgtog_redistribution` target is not selected by
the prescribed runtime scripts; its root-only Tensor scalar assertion
conflicts with the replica-populated contract explicitly required by
cyclic_reshuffle. It was left unchanged as instructed, and no runtime pass
is claimed for it. Low-level primary-only primitive semantics are unchanged.

## 2026-09-09 · evt-1003 · ctf-rs R1 closes after direct Tensor replica restoration · actor: codex

The missing public-layer replica fill is fixed in `a558e30`, the unchanged
cyclic_reshuffle test passes its separate 1/2/4 confirmation, and the full
prescribed R1 acceptance passes without skipping any WSL driver. Numerical
verification is closed. One diagnostic, three single-test confirmations and
three acceptance-script invocations were used; no extra post-pass check was
run. Records are `2b68406`; no ctf-rs push or S1 implementation followed.
The non-gated root-only Tensor test conflict is explicitly recorded in evd-1013.
This harness-msi push is the durable record; claude-report is notified after
the pushed tip is known.

- state: ctf-rs = R1 closed: replica restoration fixed; cyclic_reshuffle 1/2/4 and full WSL/native acceptance pass; S1 not started
- note: ctf-rs = evd-1012 and evd-1013; fix a558e30, local records 2b68406; no ctf-rs push; non-gated dgtog_redistribution assertion conflict recorded

## 2026-09-09 · evt-0058 · ctf-rs R1 closed: rsmpi binding accepted; replica-fill defect fixed; ctf-rs push and the dgtog_redistribution test contract await the user · actor: claude

Merged `harness-msi` 4b291a2 (evd-1012, evd-1013, evt-1003). Gate
G-CTF-R1 passed: all 175 WSL drivers at 1, 2, 4 ranks plus the local and
seven-rank checks, the native build, and the D6 native set at 1, 2, 4,
each once, on ctf-rs `2b68406` (fix `a558e30` on top of the five R1
commits). The defect behind the two-rank `cyclic_reshuffle` failure was
not in MPI at all: `Tensor::redistribute`'s two optimized paths (the
equal-phase block reshuffle and the DGTOG ROR path) returned
primary-layer-only storage and left every replica slot at the additive
identity, so a rank that owns a copy but is not the receive root saw
zeros. The fix groups ranks by their mapped physical residues, broadcasts
the canonical owner's block inside each replica subgroup, and closes the
subgroup; 23 lines in `src/tensor.rs`, no test or tolerance change. R1
and B1 are closed together. Two items for the user: `docs/validation.md`
notes that the non-gated `dgtog_redistribution` test asserts the old
root-only scalar contract and now contradicts `cyclic_reshuffle`'s
replica contract; the coordinator's recommendation is to update that
assertion to the replica contract as a `test(...)` commit and run it once
at 1, 2, 4. And ctf-rs `2b68406` is unpushed on MSI; the README requires a
separate request to push. Next under plan.v2 is S1 (sparse automatic
planning) as written in plan.v1 section 3, and the ADR-0007 consequence
for libmuffintin (`set_hf_mpi_communicator` versus taking a `&Universe`)
is a later decision.

- state: ctf-rs = R1 closed (G-CTF-R1 passed, evd-1013); ctf-rs 2b68406 unpushed pending the user's request; dgtog_redistribution test contract open; S1 next
- note: ctf-rs = replica-fill defect in Tensor::redistribute fixed by a558e30; 175 drivers at 1/2/4, native build, D6 native all pass

## 2026-09-09 · evt-0059 · integrate periodic full-potential NMTO with Python MPI · actor: codex

User authorized integrating the retained diamond production worktree into
pymuffintin main and publishing the paired native array bindings. Source
pymuffintin is f4261a9 plus its retained periodic/full-potential changes;
target is 456f8e0 plus the already verified MPI XC/core/radial blocks.
Native target is 9d64c5d; only the source radial inverse-mass export is
imported, not unrelated HF work. Source worktrees remain untouched.

Verification is a bounded integration check: existing periodic mathematical
tests keep their original tolerances; serial production versus integrated
serial/MPI bands, occupations and regional density use rtol=1e-10,
atol=1e-12 and finite outputs. Run focused tests and 1/4-rank cases once
after integration; failures get at most three discriminating computations.
No diamond basis study, SPEX build or material-accuracy acceptance this slice.

- state: nmto-periodic-mpi = integrating production periodic/full-potential path with Python-owned shared-memory MPI
- note: nmto-periodic-mpi = user authorized main publication; focused numerical equivalence only, not material acceptance

## 2026-09-09 · evt-0059 · ctf-rs pushed to GitHub at 2b68406 on the user's request; dgtog_redistribution test contract assigned · actor: user

The user requested the ctf-rs push ("pull ctf-rs back, then push") and
the test-contract change ("assert the replica contract, one test commit,
1/2/4 once"), and ruled that merging `harness-msi` into `harness` is
enough for STATUS (no ordering change). The MSI checkout
`D:/projects/ctf-rs` had no remote at all, so the eight commits
f2039d3..2b68406 travelled as a git bundle to the Mac clone
`~/tmp/ctf-rs`, whose own diverged commit f22da3a (sparse search-cache
WIP, 2026-09-08) is kept on branch `wip/sparse-search-cache-20260908`;
master was reset to 2b68406 and pushed to `rustnumgum/ctf-rs` (public)
with the Mac's gh credentials: `f2039d3..2b68406 master -> master`. The
MSI checkout now has `origin` = the GitHub URL and `origin/master` at
2b68406. The `dgtog_redistribution` replica-contract test change is
assigned to Codex on MSI (evd-1014, evt-1004 expected); the Mac pushes
it the same way when it lands.

- state: ctf-rs = active: dgtog_redistribution test onto the replica contract (Codex on MSI); R1 pushed at 2b68406; S1 next
- note: ctf-rs = GitHub master 2b68406; Mac WIP kept on wip/sparse-search-cache-20260908; MSI origin remote added
## 2026-09-09 · evd-1014 · dgtog_redistribution replica-contract test passes · ctf-rs b9990fa5b7e67fb105a9dc55da5f36d65be9d8b8

BRIEF-5's authorized test-only change is `cccd5e8`: the scalar now asserts
`[29]` on every replica instead of zero on nonroots, and the local-storage
helper asserts `key + 11` on every valid global key, zero only for padding.
All empty/high-order and other exact assertions remain. No `src/` file or
low-level root-only primitive test changed. Records `b9990fa` replace the
obsolete validation.md "Scope note" with this outcome; no ctf-rs push.

```text
DIGIT / PASS
Q: every assertion of tests/dgtog_redistribution.rs under the replica contract
class: R; ref: test committed at cccd5e8; bound: exact equality, 0
Delta: every assertion passed, exact difference 0
checks: populated replicas, padding, virtual/scalar/empty/high-order storage,
        world and parity contexts at each configured rank count
runs: WSL 1/2/4 once each, three runs total; exit 0; no diagnostics
closed: dgtog_redistribution contract; no full R1 or native rerun
```

Exact command:

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-dgtog-test/confirm.sh
```

Log: `D:/projects/runs/ctf-dgtog-test/confirmation.log`. The adjacent
`confirm.sh` and `commands.md` retain the exact nested commands: sync to the
Linux work copy, compile the single test with `--no-run`, then run it once
per rank using `mpirun --oversubscribe -n 1/2/4` under a 60-second supervisor.
The existing keepalive was reused. Prior G-CTF-R1 passes remain closed.

## 2026-09-09 · evt-1004 · dgtog_redistribution test contract closes · actor: codex

The public Tensor replica contract is now stated by the formerly root-only
test and passes the authorized three runs. Test commit `cccd5e8`, records
`b9990fa`; no source change, additional diagnostic, native test, full R1 rerun,
or ctf-rs push. Publication of ctf-rs remains with the Mac as BRIEF-5 directs.
This harness-msi push is the durable record; claude-report is notified after
the pushed tip is known.

- state: ctf-rs = R1 and dgtog_redistribution test contract closed; WSL single-test 1/2/4 passed; publication left to the Mac; S1 not started
- note: ctf-rs = evd-1014; test cccd5e8, local records b9990fa; three exact passes, no diagnostics or ctf-rs push

## 2026-09-09 · evt-0060 · ctf-rs dgtog_redistribution test contract closed and pushed; R1 line complete; S1 next · actor: claude

Merged `harness-msi` b176e65 (evd-1014, evt-1004). The test-only commit
cccd5e8 (12 lines in `tests/dgtog_redistribution.rs`: the scalar asserts
29 on every replica instead of zero on non-roots, the local-storage
helper asserts the value for every valid global key with zero only for
padding) passed at 1, 2, 4 ranks once each; records b9990fa. Pulled to
the Mac clone by bundle and pushed: `2b68406..b9990fa master -> master`
on rustnumgum/ctf-rs; the MSI checkout's `origin/master` is b9990fa. With
this the rsmpi binding line is complete: R1 closed under G-CTF-R1, the
replica-fill defect fixed, both redistribution tests stating the same
contract, everything on GitHub. Next under plan.v2 is S1 (sparse
automatic planning, plan.v1 section 3) when the user starts it; the
libmuffintin `set_hf_mpi_communicator` versus `&Universe` question stays
a later ADR.

- state: ctf-rs = idle: R1 line complete and pushed (b9990fa); S1 not started
- note: ctf-rs = GitHub master b9990fa; dgtog_redistribution on the replica contract; MSI and Mac clones in sync

## 2026-09-09 · evd-0024 · periodic full-potential NMTO MPI integration · main 5f5dc92c7bbc9d562bfc53e65e23cc9082a5f34e

DIGIT / PASS
Q: fixed H first-iteration bands, coefficients, occupations, chemical potential,
normalization and regional density; class: R relative to retained production
implementation f4261a9 plus periodic/full-potential working-tree changes.
Bound: elementwise rtol=1e-10, atol=1e-12, finite outputs required.
Delta: 1-rank maximum absolute difference 0; 4-rank maximum 2.220446049250313e-16
in MT density, with all other recorded quantities identical.
Python publication: 0c7bf7a84bc545cc3cb7804dcde91f761a60cba1.

Logs and fixed source-reference NPZ:
`/Users/zerozaki07/tmp/pymuffintin/local_experiment/generated/periodic_mpi_merge_20260909_xhB4hC/`.
`native-build.log`: FFTW release extension build passed.
`focused.log`: 21 focused tests passed. `mpi1.log`: four MPI/native block
checks passed. `mpi4.log`: eight MPI/native/control-flow checks passed per
rank. `source-vs-mpi1.log` and `source-vs-mpi4.log` retain the source comparison.
No failed numerical checks, additional diagnostics, or post-pass reruns.

Commands from `/Users/zerozaki07/tmp/pymuffintin`, after `unalias python`
and `unalias python3`, with `PYTHONPATH=$PWD/src:$PWD/.venv/lib/python3.14/site-packages:/Users/zerozaki07/tmp/libmuffintin/python`
and OMP/OPENBLAS/TBLIS/VECLIB_MAXIMUM/RAYON thread counts set to 1:

```sh
/opt/homebrew/bin/python3 -m pytest tests/test_periodic_nmto.py tests/test_nmto.py tests/test_nmto_regional_density.py tests/test_mto_hydrogen_checkpoint.py tests/test_nmto_scf_input.py tests/test_nmto_scf_loop.py -q
mpiexec --oversubscribe -n 1 /opt/homebrew/bin/python3 -m mpi4py -m pytest tests/test_nmto_mpi.py tests/test_nmto_potential_mpi.py -q
mpiexec --oversubscribe -n 4 /opt/homebrew/bin/python3 -m mpi4py -m pytest tests/test_nmto_mpi.py tests/test_nmto_potential_mpi.py tests/test_nmto_scf_loop.py -q
mpiexec --oversubscribe -n 1 /opt/homebrew/bin/python3 -m mpi4py local_experiment/generated/periodic_mpi_merge_20260909_xhB4hC/compare.py
mpiexec --oversubscribe -n 4 /opt/homebrew/bin/python3 -m mpi4py local_experiment/generated/periodic_mpi_merge_20260909_xhB4hC/compare.py
```

Build used `uvx maturin develop --release --features fft-fftw` from paired
`libmuffintin/python`, with Homebrew TBLIS/HDF5, `LIBRARY_PATH=/opt/homebrew/opt/fftw/lib`,
`CARGO_BUILD_JOBS=2`, and the pymuffintin virtualenv.
Timeouts: 600 seconds for focused tests, 300 seconds per MPI command.
Scope: integration equivalence, strict periodic mathematical tests and native
shared-array/mass-field parity; not diamond convergence, scaling or SPEX agreement.

## 2026-09-09 · evt-0061 · periodic full-potential NMTO MPI integrated and published · actor: codex

Both main branches were pushed: native 5f5dc92 and Python 0c7bf7a.
Production source trees remain untouched; unrelated `uv.lock` was restored.
Review corrected serialized k-owner preparation and duplicated per-rank
potential quadrature: k geometries now compute concurrently, root alone
prepares FFT/quadrature data, and shared read-only arrays feed fixed blocks.
The new input contract replaces minimum-cells with explicit periodic cutoffs
and quadrature, and preserves linear/Pulay selection and KH inverse masses.
The two older evt-0059 headings resulted from concurrent Mac ledger writers;
this closes the one titled "integrate periodic full-potential NMTO with Python MPI"
without rewriting either historical entry.

- state: nmto-periodic-mpi = integrated and published; focused serial/source/MPI checks passed; material study not run
- note: nmto-periodic-mpi = evd-0024; pymuffintin 0c7bf7a and libmuffintin 5f5dc92; SPEX/diamond band comparison remains separate

## 2026-09-09 · evt-0062 · ctf-rs push entry ID collision superseded · actor: claude

The entry "ctf-rs pushed to GitHub at 2b68406 on the user's request;
dgtog_redistribution test contract assigned" above reused `evt-0059` after
that ID had already been taken, in this worktree's uncommitted ledger tail,
by the nmto-periodic-mpi entry "integrate periodic full-potential NMTO with
Python MPI"; commit 74eccd2 then carried both headings. This append-only
correction supersedes the ctf-rs entry's heading ID as `evt-0062`; its
text, state, and note stand as written, and evt-0060 already closed the
item it opened. `evt-0059` now names the nmto-periodic-mpi entry alone,
which is how evt-0061 already reads it. The state lines below restate
evt-0060 unchanged.

- state: ctf-rs = idle: R1 line complete and pushed (b9990fa); S1 not started
- note: ctf-rs = GitHub master b9990fa; dgtog_redistribution on the replica contract; MSI and Mac clones in sync

## 2026-09-10 · evt-0063 · share harness across libmuffintin and pymuffintin; ingest OMT recommendations · actor: codex

User decision: libmuffintin and pymuffintin will use this single libmuffintin
`harness` branch for plans, tracking, decisions, and evidence. README and
agent guidance now state this shared ownership. Existing accepted plans are
unchanged; this event does not authorize their draft successors.

Source: the user's supplied OMT improvement report in this task. The numbers
below are user-reported observations, not a new gate run or independent
verification. Current checkout HEADs at ingestion: libmuffintin
`5f5dc92c7bbc9d562bfc53e65e23cc9082a5f34e`, pymuffintin
`303beef281a03d264903436d36b4ce78b174a26e`. These are context revisions,
not established revisions of the reported experiments. Exact run commands,
logs, error metric definitions, and reference provenance were not supplied.

| Reported improvement | Reported result and cost | Follow-up interpretation |
|---|---|---|
| Cover the target energy window with `energy-mesh = [-0.35, 0.25, 0.85]`, N=2 | OMT conduction-band error 0.44 → 0.12 eV; valence unchanged; about +50% time | Main lever for conduction bands; the existing two nodes suffice if only valence is targeted |
| Set `l-max = 3` | Conduction 0.12 → 0.11 eV; valence 0.244 → 0.238 eV; time ×1.7 | Smaller improvement at higher cost |
| Keep overlap at about 1.2a or below | 1.3a and 1.4a lower fit RMS without band improvement; valence plateaus near 0.24 eV | Report-specific recommendation, not a universal radius guard; preserve the report's undefined `a` convention pending provenance |

Reported diagnosis: the remaining roughly 0.24 eV valence error
(Γ25′ +0.26 eV, Γ1 +0.22 eV) is insensitive to nodes, overlap, l-max, and
the numerical parameters tried. The report attributes this to the minimal
basis at fixed potential: a large nonspherical intratomic potential
(l=3 component about 0.2 Ha at the sphere surface), frozen spd radial shapes,
and spherical OMT shells that cannot address that part. This causal
interpretation is retained as reported, not newly proven here.

Candidate structural remedies, not authorized implementation:
- Empty spheres with s/p functions in diamond/silicon interstitial sites.
  The report says libmuffintin checkpoint validation rejects
  `atomic_number == 0`; checkpoint, core solving, and species handling need
  cross-repository work. Verify those code boundaries when this work starts.
- Extra radial freedom (a second NMTO set or derivative-like local orbitals),
  explicitly distinguished from standard NMTO.
- A self-consistent comparison with SPEX may look better because the NMTO
  potential can partly compensate; this is an expectation, not acceptance
  and not a substitute for the fixed-potential comparison.

Separate reported issue: libmuffintin's 50-point Fibonacci XC grid affects
both LAPW and NMTO. The report expects its correction not to change the
comparison above, while changing absolute bands, including an approximately
0.2 eV Γ25′ splitting. No grid fix or cancellation claim was verified here.

Pending follow-ups from the report: add the conduction-window recommendation
and three-node example to pymuffintin README's "Production recipe"; switch
scratch `validate_compare.py` to the l6g7 reference. This task records these
items only: neither file is changed and no numerical run is started.
Before a future run, fix quantity, reference, tolerance, and finite budget
under Stop That Digit; preserve fixed-potential versus self-consistent scope.

- state: diamond-omt = reported study ingested; recipe/reference updates and structural remedies pending
- note: diamond-omt = evt-0063; shared libmuffintin/pymuffintin scope; user-reported numbers, no new numerical acceptance

## 2026-09-10 · evt-0064 · record production MT XC Lebedev repair plan for publication · actor: codex

User requested that the supplied MT XC symmetry-breaking diagnosis and
repair plan enter the shared harness and be committed and pushed together
with the pending shared-harness/OMT records. Added
`plans/mt-xc-angular/plan.v1.md`: weighted explicit-degree Lebedev quadrature,
precision independent of potential l_max, consistent regional/core XC
weights, and native/Python/TOML controls. Unrelated THC grids are excluded.

The supplied Fibonacci reproduction, forbidden coefficients, allowed-channel
error, and Γ splittings are recorded as reported evidence, not a newly run
gate. Implementation and numerical execution are not authorized by this
documentation request. Exact API/default rule, reference artifacts,
tolerances, and finite budgets must be settled before implementation/runs
as specified by the plan. No source changes or numerical tests were made.

Clarifies evt-0063: common LAPW/NMTO quadrature does not establish error
cancellation, and the fraction of the native LAPW–SPEX 0.14 eV discrepancy
caused by this bug remains unquantified. Preserve fixed-density integration
checks, fixed-potential spectra, and self-consistent comparisons as distinct
acceptance questions.

- state: mt-xc-angular = repair plan recorded; implementation and numerical gates pending authorization
- note: mt-xc-angular = evt-0064; weighted Lebedev shared-path fix; reference, tolerances, default rule, and finite budget unresolved

## 2026-09-10 · evt-0065 · record target-window basis freedom and published NMTO recipe notes · actor: codex

User-requested tracking point: in some cases a minimal muffin-tin/NMTO basis
may need additional degrees of freedom to match the target bands above the
Fermi level. The user's parenthetical target is retained as unoccupied /
conduction bands, not valence bands. This is a case-dependent basis-adequacy
concern, not a demonstrated limitation of every muffin-tin basis.

Distinguish covering that energy window with NMTO interpolation nodes from
enriching the spatial/radial basis. The reported three-node improvement in
evt-0063 motivates the former; it does not prove that additional radial
functions or empty-sphere channels are necessary or sufficient. If a residual
target-band mismatch remains, those are candidate extra degrees of freedom,
not authorized implementation or accepted remedies. Keep the separately
reported occupied-valence plateau distinct from this unoccupied-band target.

Publication update to evt-0063: pymuffintin README's "Production recipe"
energy-window example and fixed-potential probe discussion were committed and
pushed to main as `40136cb312a941bf315d205597717306dacef2cb`.
libmuffintin main remains `5f5dc92c7bbc9d562bfc53e65e23cc9082a5f34e`.
These are current context revisions, not newly established probe provenance.
The user reports that gitignored scratch `validate_compare.py` now uses
`lapw_l6g7.npz` and includes condMIN and s13/s14; that script was not part of
the README commit and its execution was not independently verified here.

This event records the concern and publication only. No numerical run,
universal basis-limit claim, or new acceptance is added; future comparisons
still require an explicit target band window, reference, tolerance, and budget.

- state: diamond-omt = recipe notes published; reference-script update reported; target-band basis enrichment remains a candidate
- note: diamond-omt = evt-0065; pymuffintin 40136cb; above-Fermi target distinguished from valence; no new numerical acceptance

## 2026-09-11 · evt-0066 · ctf-rs plan.v3 accepted; C1 and S1 assigned to Codex on MSI · actor: user

The user ordered S1 and the port's own remaining pending items deployed to
MSI ("把S1和第一层任务部署给MSI"). `plans/ctf-rs/plan.v3.md` carries S1
from plan.v1 section 3 with its eight drivers and the five
`test_sparse.py` semantics unchanged, split into the ordered batches S1a,
S1b, S1c, and adds the closed C1 list (model I/O, contraction-path
symmetrize/desymmetrize, dense custom-function folded kernels, the selector
agreement collective, the `graph_io_aux` native run, bookkeeping). C1 runs
first and re-gates the whole WSL set once; S1a is the first sparse
planning code. Gates: G-CTF-C1 new, G-CTF-S1 accepted against the plan.v3
list; the G-CTF-D1 to D6 row, still `proposed` although evd-1002 to
evd-1008 closed it, is corrected to closed. Executor: Codex on MSI in
`D:/projects/ctf-rs` at `b9990fa`, clean; upstream at
`D:/projects/ctf-upstream-f69`; brief at
`C:/Users/xylxp/projects/ctf-rs-s1/BRIEF.md`; records on `harness-msi`
(evd-1015 upward, evt-1005 upward) merged here; ctf-rs published by the
Mac only. The libmuffintin `set_hf_mpi_communicator` versus `&Universe`
question and the exchange-contraction integration stay outside this plan.

- state: ctf-rs = active: C1 then S1a to S1c (Codex on MSI); G-CTF-C1 and G-CTF-S1 accepted
- note: ctf-rs = plan.v3 accepted 2026-09-11; executor MSI at b9990fa; D1 to D6 gate row corrected to closed

## 2026-09-11 · evt-0067 · ctf-tiled plan.v1 drafted as the successor to ctf-rs S1 · actor: user

The user kept S1 running and asked for the tile-sparsity direction to be
preserved as its own plan ("把这个保留成 tiled-ay 计划，作为 S1 的后续").
`plans/ctf-tiled/plan.v1.md` (draft, not authorized) records the two
analyses of 2026-09-11: keep the CTF skeleton of ctf-rs and add
TiledArray-style tile semantics (`TileId`, `Tiling`,
`Distribution<TileId>`, block-sparse storage, tile-level join,
`TileKernel`), with the DBT plus LibRI (atom, R) line on the MTO family as
the consumer target. First milestone T1: 2D/3D block-sparse tensors,
sparse tile contraction, MPI tile ownership over the existing cyclic
mapping, toy acceptance against the dense path; the norm filter (T2), the
atom-pair distribution (T3), and the R-join (T4) follow. Gates G-CTF-T1
and G-CTF-T2 to T4 proposed. Starts after ctf-rs S1 closes unless the
user orders otherwise; an ADR and the possible rename to riir come with
T1's authorization. ctf-rs itself is unchanged: C1 then S1a to S1c stay
assigned (evt-0066).

- state: ctf-tiled = draft: successor to ctf-rs S1; not authorized
- note: ctf-tiled = the "tiled-ay" plan; CTF skeleton kept, tile sparsity added; T1 block-sparse tiles first; T4 blocked on the libmuffintin atom-centered aux
