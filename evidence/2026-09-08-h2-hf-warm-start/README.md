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
