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

## First-feedback probe handoff

`/tmp/h2-hf-mpi-probe` is an untracked `git archive 88b212d` scratch copy,
with `.local-deps` symlinked from main. Its only instrumentation writes the
first post-Allreduce `exchange_feedback` block on rank zero, calls a barrier,
and invokes `process::exit(0)` on every rank. The scratch build command was:

```sh
export CARGO_TARGET_DIR=/tmp/h2-hf-mpi-probe-target
cargo build --release -p libmuffintin-runtime --features fft-fftw,mpi --example h2_hf
```

The initial launch command was issued before that background build completed,
so Open MPI could not access the not-yet-created executable and no application
or numerical code ran. After the successful build, the fixed rank-one command
was executed once:

```sh
RAYON_NUM_THREADS=10 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n1.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 1 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n1 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
```

The hook flushed `feedback-n1.bin`: one 1030-by-1030 block, 1,060,900
complex entries. It then bypassed the example's retained `Universe` destructor,
so `MPI_Finalize` was not called. Open MPI diagnosed improper rank-zero exit
and returned exit 1 (real 53.51 s). This is an execution failure even though
the requested rank-one bytes were written after the completed Allreduce.
The fixed contract authorizes no failure diagnostics or replacement run.

```text
DIGIT / HANDOFF
Q: first-rebuild band-space exchange feedback at n=2 and n=4 versus n=1; class: R; ref: n=1
bound: 1e-12 absolute; Delta: unavailable
checks: n=1 dump produced, but its scratch process exited 1 without MPI_Finalize; n=2 and n=4 not run
runs: 1 feedback probe; no diagnostic
unresolved: obtain a cleanly finalized first-rebuild probe without spending an unapproved replacement run
```

Per the explicit stop rule, the rank-two and rank-four feedback dumps, rank-two
A0, and all three A1 timing runs were not started. `compare_feedback.py`,
`compare_a0.py`, and `phase_summary.py` therefore have no result logs. The
A1 ladder was not run. Total numerical executions: four (three fixtures and
one feedback probe); an earlier no-executable launcher error ran no numerical
code. Main implementation remains committed, but numerical MPI acceptance is
handed off. No push.
