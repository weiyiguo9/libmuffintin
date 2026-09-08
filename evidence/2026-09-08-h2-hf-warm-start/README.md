# Gamma valence cross-outer Fock warm start

Authorization: user evt-0048 and H2-HF-WARM. Baseline main `b282484`;
starting harness `a6cc377`. The A1 ladder is held, MPI evidence is out of
scope, and neither branch may be pushed.

Main `a5d52b3` adds one Gamma valence-only warm start. A completed fixed-
potential solve returns its final mixed global-basis feedback. The outer loop
moves that feedback into the next fixed-potential solve, which first solves
the freshly built H0/S bands with it and then enters the ordinary rebuild,
feedback-residual, and CDIIS path. Outer iteration 1 and its two identity gates
are unchanged. No cache, MPI, tolerance, mixer, gate, or other HF path changed.

Before either numerical run, the A0 prediction was fixed at 12–24 exchange
rebuilds and 7–9 outer iterations, with energies inside the outer loop's
convergence floor. A result at or above 40 rebuilds was fixed as a handoff.
The numerical budget was two runs with no diagnostics.

Common build environment:

```sh
export TBLIS_DIR="$(brew --prefix tblis)"
export HDF5_DIR="$(brew --prefix hdf5)"
export RUSTFLAGS="-L $(brew --prefix fftw)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib"
```

## Fixture

```sh
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw
```

Log: `fixture-fftw.log`.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: gamma_valence_hf fixture
bound: 1e-8; Delta: every unchanged assertion passed
checks: release fft-fftw fixture; one test passed
runs: 1; contract closed
```

## A0

The example was built from `a5d52b3` with:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
```

Build log: `build.log`. The sole authorized A0 command was:

```sh
RAYON_NUM_THREADS=10 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-warm --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
```

Combined stdout/stderr is `a0.stdout.log`; the same log is committed on main
as `examples/h2_dft/results/hf-a0-warm.log`. The supervisor returned exit 124
at 1800.24 s. The run completed 43 exchange rebuilds: outer iterations 1–7
completed, and outer iteration 8 reached Fock iteration 3. It produced no final
energy, identity, electron-count, rebuild-count, or driver-wall record.

```text
DIGIT / HANDOFF
Q: A0 E/HOMO/E_H/E_x and three driver identities (Ha); class: R; ref: evd-0010
bound: 2e-8 for energies and identities; Delta: unavailable
checks: ten-thread run reached the 1800 s cap and exited 124 before convergence, final electron count, or final energy and identity output
runs: 1; no diagnostic or repeat
unresolved: A0 did not produce the class-R quantities within the fixed cap

DIGIT / HANDOFF
Q: A0 exchange_rebuilds, outer_iterations, wall_s; class: P; ref: 56, 8, 1166.331817 s
bound: report-only, with 12–24 rebuild prediction and mandatory handoff at 40 or more; Delta: 43 completed rebuilds, 7 completed outer iterations with outer 8 in progress, final driver wall_s unavailable
checks: supervisor wall 1800.24 s; rebuild threshold exceeded
runs: same A0 run; no diagnostic or repeat
unresolved: the warm start did not reduce rebuilds below the predeclared handoff threshold
```

The two-run budget is exhausted. No diagnostic, repeat, A1-ladder, MPI run,
or MPI-evidence edit followed. Main records are complete through `199eb49`.
No push.

## Zero-start-up warm-loop follow-up

Evt-0050 authorized a second two-run contract after reading the first warm
trajectory. Main `762c210` copies `spec.fock_mixing` inside the Gamma valence
fixed-potential solve and changes `startup_steps` to zero only when carried
feedback is present and the variant is `CommutatorDiis` or
`QuasiNewtonDiis`. The cold first outer loop retains its configured start-up;
other variants, damping, history, gates, tolerances, spec structs, and all
other HF paths are unchanged. The carried feedback seeds the solve but is not
inserted into the new mixer history.

Before either follow-up run, the prediction was fixed at 22–34 exchange
rebuilds, 7–9 outer iterations, and 2–4 iterations per warm inner loop. A
result at or above 40 rebuilds remained a mandatory handoff. The numerical
budget was two runs with no diagnostics.

The fixture command was unchanged:

```sh
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw
```

Log: `fixture-warm2-fftw.log`.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: gamma_valence_hf fixture
bound: 1e-8; Delta: every unchanged assertion passed
checks: release fft-fftw fixture; one test passed
runs: 1; contract closed
```

The example was built from `762c210` with:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
```

Build log: `build-warm2.log`. Immediately before the A0 run, `uptime` was:

```text
23:56  up 2 days,  5:30, 5 users, load averages: 10.78 28.99 37.53
```

The sole authorized A0 command was:

```sh
RAYON_NUM_THREADS=10 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 3600s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-warm2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
```

It converged and exited 0. Immediately afterward, `uptime` was:

```text
0:17  up 2 days,  5:50, 5 users, load averages: 27.81 27.57 28.24
```

Logs: `a0-warm2.stdout.log`, `uptime-before.txt`, `uptime-after.txt`, and
`compare-a0-warm2.log`; main preserves the same A0 output as
`examples/h2_dft/results/hf-a0-warm2.log`.

```text
DIGIT / PASS
Q: A0 E/HOMO/E_H/E_x and three driver identities (Ha); class: R; ref: evd-0010
bound: 2e-8 absolute for energies and identities; Delta: 2.48245868306185002e-13 / 3.09803882547754483e-9 / 1.62302338235775778e-10 / 8.68805014042628443e-10 for E/HOMO/E_H/E_x
checks: maximum identity 4.76458923703848569e-9; electron-count error 6.66133814775093924e-15; finite converged result; exit 0
runs: 1; class-R contract closed

DIGIT / PASS
Q: A0 exchange_rebuilds, outer_iterations, per-outer Fock iterations, wall_s; class: P; ref: evd-0010 and evd-0022
bound: report-only, with 22–34 rebuild prediction and mandatory handoff at 40 or more; result: 32 rebuilds, 8 outer iterations, 7 / 5 / 4 / 4 / 3 / 3 / 3 / 3 iterations, wall_s 1228.894974
checks: rebuild and outer-count predictions passed; the first warm loop took 5 iterations, one above the report-only 2–4 prediction; all later warm loops took 3–4; supervisor wall 1229.77 s
runs: same A0 run; closed without diagnostics or repeats
```

Wall time is report-only and is read with the recorded load context. The
follow-up used exactly two numerical runs. No diagnostic, repeat, MPI work,
MPI-evidence edit, or A1-ladder run followed. Main records are complete through
`9d64c5d`. No push.
