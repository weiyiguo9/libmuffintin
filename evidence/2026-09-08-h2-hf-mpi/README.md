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
