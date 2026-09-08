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

## First-feedback probe continuation

Evt-0044 accepted the flushed `feedback-n1.bin` as the rank-one reference and
authorized six remaining runs. The scratch-only hook was changed from a
barrier followed by `process::exit(0)` to returning
`GammaValenceHfError::QTopology` on every rank after the dump. The scratch
example maps that error to `Ok(())` only when `H2_MPI_FEEDBACK_DUMP` is set:

```diff
-crate::hf_communicator::barrier();
-std::process::exit(0);
+return Err(GammaValenceHfError::QTopology);
```

```diff
-let result = run_gamma_valence_hf(&mut physics, &spec)?;
+let result = match run_gamma_valence_hf(&mut physics, &spec) {
+    Ok(result) => result,
+    Err(_) if std::env::var_os("H2_MPI_FEEDBACK_DUMP").is_some() => {
+        eprintln!("probe_done rank={mpi_rank}");
+        return Ok(());
+    }
+    Err(error) => return Err(error.into()),
+};
```

The scratch target was rebuilt with the common environment and:

```sh
export CARGO_TARGET_DIR=/tmp/h2-hf-mpi-probe-target
cargo build --release -p libmuffintin-runtime --features fft-fftw,mpi --example h2_hf
```

Build log: `build-probe-2.log`. The first re-authorized command was:

```sh
RAYON_NUM_THREADS=5 H2_MPI_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-mpi/feedback-n2.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 900s /opt/homebrew/bin/mpirun -n 2 /tmp/h2-hf-mpi-probe-target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/mpi-feedback-n2 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
```

Both ranks printed `probe_done`, MPI finalized normally, and `mpirun` exited
0. The run produced one 1030-by-1030 block with 1,060,900 finite complex
entries in `feedback-n2.bin`. The fixed comparison was:

```sh
/opt/homebrew/bin/python3 compare_feedback.py feedback-n1.bin feedback-n2.bin
```

It found a maximum absolute difference of
`2.91207674214817233e-02` at `(k=0,row=0,column=3)`.

```text
DIGIT / HANDOFF
Q: first-rebuild band-space exchange feedback at n=2 versus n=1; class: R; ref: n=1
bound: 1e-12 absolute; Delta: maximum 2.91207674214817233e-2 at (k=0,row=0,column=3)
checks: n=2 exited 0 after normal finalization; all 1,060,900 finite entries compared
runs: 1 of 6 re-authorized numerical runs; no diagnostic
unresolved: rank-two feedback is not rank-independent within the fixed bound
```

Logs: `feedback-n2.log` and `compare-feedback-n2.log`. Per the explicit stop
rule, the rank-four feedback run, rank-two A0 run, and all three A1 timing runs
were not started. The A1 ladder was not run. The continuation used one
numerical execution, and the study remains handed off. No push.
