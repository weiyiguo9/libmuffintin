# H2 HF pair-level MPI

Authorization: evt-0040 and refined ADR-0006. Baseline main `5f40aac`;
starting harness `476237f`. The A1 ladder is held; no push is authorized.

The fixed contracts and ten-run budget are recorded in evt-0041. Open MPI
5.0.10 (`/opt/homebrew/bin/mpirun`, `mpicc`) is used locally. The optional
runtime `mpi` feature binds to safe rsmpi 0.8, resolving to mpi-sys 0.2.
The example initializes `Threading::Funneled`; library code never initializes
or finalizes MPI. Full commands and results are appended after implementation.

Common environment:

```sh
export TBLIS_DIR="$(brew --prefix tblis)" HDF5_DIR="$(brew --prefix hdf5)"
export RUSTFLAGS="-L $(brew --prefix fftw)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib"
```

`compare_feedback.py` checks every finite complex entry and the fixed 1e-12
rank-independence bound. `compare_a0.py` checks the fixed evd-0010 energies,
plan identities, electron count, and rank-final state. `phase_summary.py`
extracts only the first completed Fock iteration; timings are report-only.

## Implementation and fixture gate

Main `f189669` adds the default-off `mpi` feature: rsmpi 0.8.2 and
mpi-sys 0.2.4, MSRV still 1.89. Main `33a761d` initializes only in binaries
at `Threading::Funneled`; the library validates the raw `MPI_COMM_WORLD`
handle and records rank/size atomically. It obtains safe world views only on
the main thread for collectives. It deliberately does not invoke rsmpi
`FromRaw`: that owning constructor forbids and would free `MPI_COMM_WORLD`,
and the crate retains `#![forbid(unsafe_code)]`.

The Gamma build assigns occupied bands by `position % size == rank`, builds
and contracts only local selections, and Allreduces the full partial
`[n_target,n_target]` block once per k. Basis compilation and Coulomb assembly
remain redundant. Ranks with no occupied bands contribute zero blocks.
Diagnostics and example output are rank-zero-only. Before final output, the
example min/max-reduces its final counters, energies, electron count, and
identities and requires exact rank agreement; it max-reduces wall time.
Main `88b212d` documents Snellius module/build/launch and threading rules.

Compile-only checks performed by the implementation worker passed for the
default example and the `fft-fftw,mpi` example/test targets. The bounded
fixture commands run from main with the common environment above:

```sh
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw,mpi -- --nocapture
RAYON_NUM_THREADS=5 /opt/homebrew/bin/mpirun -n 2 target/release/deps/gamma_valence_hf-5a476f84e231c5b9 --exact gamma_hydrogen_rebuilds_full_vv_feedback_and_rejects_stale_orbitals --nocapture --test-threads=1
```

Logs: `fixture-default.log`, `fixture-mpi-n1.log`, `fixture-mpi-n2.log`.
The feature-gated fixture owns Funneled initialization and communicator
registration, so the last command exercises the actual two-rank distribution
and Allreduce path. Each of the three executions passed all unchanged identity
assertions at 1e-8. No energy-difference contract applies to this fixture set.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identity residuals; class: R; ref: fixture
bound: 1e-8; Delta: every unchanged assertion passed in all three runs
checks: default n=1; fft-fftw,mpi n=1; initialized MPI test binary under mpirun n=2
runs: 3; numerical verification closed
```

## Feedback spectrum and remaining-run outcome

The scratch-only finalization correction from evt-0044 remains unchanged: the
hook returns after dumping on every rank and the scratch example converts that
probe error to a normal return, allowing every retained MPI Universe to drop.
Evt-0046 re-posed the comparison after reading evd-0019. The large entrywise
difference is a unitary band-gauge rotation inside degenerate subspaces, not a
different distributed operator. The fixed class-R quantity is therefore the
sorted spectrum of each feedback block.

On the existing rank-two dump, without a rerun:

```text
DIGIT / PASS
Q: sorted first-rebuild feedback eigenvalues at n=2 versus n=1; class: R; ref: feedback-n1.bin
bound: 1e-12 absolute; Delta: maximum 9.85322934354826430e-16
checks: one 1030-dimensional block; compare_feedback_spectrum.py passed
runs: 0 new; spectrum contract closed
```

The first authorized run used the unchanged scratch target:

```sh
RAYON_NUM_THREADS=2 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n4.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 4 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n4 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
/opt/homebrew/bin/python3 compare_feedback_spectrum.py feedback-n1.bin feedback-n4.bin | tee compare-feedback-spectrum-n4.log
```

```text
DIGIT / PASS
Q: sorted first-rebuild feedback eigenvalues at n=4 versus n=1; class: R; ref: feedback-n1.bin
bound: 1e-12 absolute; Delta: maximum 5.62050406216485499e-16
checks: mpirun exit 0; one 1030-dimensional block; compare_feedback_spectrum.py passed
runs: 1; spectrum contract closed
```

Logs: `feedback-n4.log` and `compare-feedback-spectrum-n4.log`. Main
`c46223c` was then built with the common environment:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw,mpi --example h2_hf
```

Build log: `build-main-mpi.log`. The second authorized run was:

```sh
RAYON_NUM_THREADS=5 /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s /opt/homebrew/bin/mpirun -n 2 target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0-mpi-n2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1
```

It reached the 1800 s cap before convergence and returned exit 124. Forty
exchange rebuilds completed through outer iteration 6, Fock iteration 5;
supervisor wall was 1800.12 s. There was no final result, driver `wall_s`, or
rank-agreement check, so `compare_a0.py` was not run.

```text
DIGIT / HANDOFF
Q: A0 E/HOMO/E_H/E_x and three identities (Ha); class: R; ref: evd-0010
bound: 1e-10 for energies and 1e-8 for identities; Delta: unavailable
checks: supervisor exit 124 at 1800.12 s before convergence and final rank agreement; 40 exchange rebuilds completed
runs: 1; no diagnostic or repeat
unresolved: the MPI A0 acceptance quantities were not produced within the fixed cap
```

Per the explicit stop rule, the three class-P A1 timing runs were not started:

| Ranks | Rayon threads/rank | `gamma.rebuild.mpb` | `gamma.rebuild.contraction` | `gamma.fock.iteration` |
|---:|---:|---:|---:|---:|
| 1 | 10 | not run | not run | not run |
| 2 | 5 | not run | not run | not run |
| 4 | 2 | not run | not run | not run |

This continuation used two of five authorized numerical runs. No diagnostic
or rerun was made, and the A1 ladder was not run. Main record: `b282484`. No
push.
