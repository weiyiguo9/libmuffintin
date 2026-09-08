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
