# Gamma valence Fock phase profile

Authorization: evt-0030 and H2-HF-PROFILE. Baseline main `35c200c`;
starting harness `17c615f`. MPI is held and neither branch may be pushed.

Task B is a report-only class-P study, with two runs: A0 at 240 s and A1
base at 2400 s. These are not plan rows. Only completed phases are measured;
a supervisor timeout is expected budget enforcement, not SCF acceptance.
Tables report A0 Fock iterations 1 and 2 and A1 Fock iteration 1.
Nested phase shares overlap and must not be summed.

Task C requires `(gamma.rebuild.mpb - vv.mpb_rebuild +
gamma.rebuild.coulomb_assembly) / gamma.fock.iteration >= 0.30` at A1.
If authorized by that measurement, its class-R bounds and finite run budget
are recorded in evt-0031; they are not changed in response to results.

Build environment and command:

```sh
export TBLIS_DIR="$(brew --prefix tblis)" HDF5_DIR="$(brew --prefix hdf5)"
export RUSTFLAGS="-L $(brew --prefix fftw)/lib"
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" RAYON_NUM_THREADS=10
```

Commands run from the main checkout; `EVIDENCE` is
`../libmuffintin-harness/evidence/2026-09-08-h2-hf-gamma-profile`:

```sh
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/gamma-profile-a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2 > "$EVIDENCE/a0.stdout.log" 2> "$EVIDENCE/a0.stderr.log"
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/gamma-profile-a1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2 > "$EVIDENCE/a1.stdout.log" 2> "$EVIDENCE/a1.stderr.log"
```

`phase_tables.py` extracts completed iteration timings from the stderr logs,
summing repeated occupation scopes within the same iteration. Incomplete
iterations have no full-iteration denominator and are not used for Task C.

## A0 profile

Main `79623b8` (timer-only change); build log `build-timers.log`. The A0
supervisor exited 124 at its 240 s cap. Ten complete Fock iterations were
logged; only the requested first two are tabulated. The interrupted eleventh
iteration has no complete denominator and is not a convergence result.

| Phase | Iteration 1 (s) | Share | Iteration 2 (s) | Share |
|---|---:|---:|---:|---:|
| `gamma.fock.iteration` | 17.479391 | 100.000% | 15.698071 | 100.000% |
| `gamma.rebuild.inputs` | 0.014049 | 0.080% | 0.013861 | 0.088% |
| `gamma.rebuild.mpb` | 8.108799 | 46.391% | 7.799805 | 49.686% |
| `vv.mpb_rebuild` (inside MPB) | 6.334591 | 36.240% | 6.052506 | 38.556% |
| basis compilation (MPB minus vertices) | 1.774208 | 10.150% | 1.747299 | 11.131% |
| `gamma.rebuild.coulomb_assembly` | 0.973493 | 5.569% | 1.035745 | 6.598% |
| `gamma.rebuild.contraction` | 4.875244 | 27.891% | 4.592004 | 29.252% |
| `gamma.fock.feedback_lift` | 0.208948 | 1.195% | 0.212254 | 1.352% |
| `gamma.fock.mix` | not called | – | 0.053651 | 0.342% |
| `gamma.fock.spinor_solve` | 1.177446 | 6.736% | 0.880135 | 5.607% |
| `gamma.fock.occupations` (two calls) | 0.000070 | 0.000% | 0.000055 | 0.000% |
| `gamma.fock.density_residual` | 0.852210 | 4.876% | 0.631527 | 4.023% |

Compile plus Coulomb is 2.747701 s (15.720%) and 2.783044 s (17.729%).
These A0 fractions do not decide Task C; the contract uses A1.

## A1 base profile and Task C decision

Same main `79623b8`. Supervisor exit 124 at 2400.31 s; four complete Fock
iterations were logged, and the fifth was interrupted in exchange contraction.
Only the requested first iteration is used below.

| Phase | Seconds | Share of iteration |
|---|---:|---:|
| `gamma.fock.iteration` | 482.082668 | 100.000% |
| `gamma.rebuild.inputs` | 0.099336 | 0.021% |
| `gamma.rebuild.mpb` | 151.026190 | 31.328% |
| `vv.mpb_rebuild` (inside MPB) | 143.437000 | 29.754% |
| basis compilation (MPB minus vertices) | 7.589190 | 1.574% |
| `gamma.rebuild.coulomb_assembly` | 8.563591 | 1.776% |
| `gamma.rebuild.contraction` | 290.551110 | 60.270% |
| `gamma.fock.feedback_lift` | 2.616436 | 0.543% |
| `gamma.fock.mix` | not called | – |
| `gamma.fock.spinor_solve` | 10.895720 | 2.260% |
| `gamma.fock.occupations` (two calls) | 0.001246 | 0.000% |
| `gamma.fock.density_residual` | 2.214952 | 0.459% |

**Task C skipped:** (151.026190 − 143.437000 + 8.563591) / 482.082668
= 3.351%, below the fixed 30% threshold. Compilation plus Coulomb assembly
costs 16.152781 s. The largest top-level measured phase is exchange contraction
at 290.551110 s (60.270%); MPB costs 151.026190 s (31.328%), including
126.803971 s (26.303% of the iteration) in `vv.interstitial`.
No `GammaExchangeCache`, Task C fixture run, or Task C A0 rerun is authorized
by this profile. No contraction or pair-FFT change is made. MPI remains held.

These are two report-only class-P runs, not numerical acceptance rows.
The earlier A0 pass at evd-0010 remains closed. A0 profile wall is 240.12 s;
there is no after-Task-C A0 wall because Task C is skipped.

## Task D command

The prior A1 base log is moved on main to `hf-a1-row1-cap1800.log`.
Run the unchanged base row once, with the same build/run environment:

```sh
set -o pipefail
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a1-row1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a1-row1.log
```

Class P: report `hartree_exchange` against zero, no study bound. Required
validity remains convergence, finite energies, electron count 2 within 1e-8,
and three driver identities within 1e-8 Ha. Task D explicitly stops if the
base row reaches its cap again; no diagnostic or threshold change is allowed.

### Task D outcome

Main computation `79623b8`, records committed on main `10f6b39`. The retry
exited 124 at 1800.76 s after three completed Fock iterations and no completed
outer iteration. Last printed density residual: 6.3664758060341759e-5;
feedback residual: 7.9480430635287458e-5 Ha. No converged energy, HOMO,
Hartree/exchange energy, three-identity row, final electron count, or
`hartree_exchange` was produced. Log: `examples/h2_dft/results/hf-a1-row1.log`
on main; previous log: `hf-a1-row1-cap1800.log`.

```text
DIGIT / HANDOFF
Q: A1 base hartree_exchange (Ha); class: P; ref: 0
bound: none (study); Delta: unavailable
checks: required converged finite result, three identities within 1e-8 Ha, and electron count 2 within 1e-8 not established; cap exit 124
profile: compile + Coulomb 16.152781 / 482.082668 s = 3.351%; contraction 290.551110 s = 60.270%
runs: 1 Task D retry; no diagnostics
unresolved: A1 base still cannot finish within its authorized 1800 s cap
```

H2-HF-PROFILE used three numerical executions: two class-P profiles and the
Task D retry. Task C was skipped, so its four fixture runs and A0 rerun were
not performed. A1 rows 2–10, A1v, A2, B, and Bv remain unrun under Task D's
explicit repeated-base-cap stopping rule. MPI remains held. No pushes.
