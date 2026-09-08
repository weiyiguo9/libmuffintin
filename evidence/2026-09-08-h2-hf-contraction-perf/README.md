# H2 HF contraction and projection performance

Authorization: evt-0034, H2-HF-CONTRACT. Baseline main
`10f6b3925ed32fb4c84564c50dae3a84d47b50d1`; starting harness `7864b80`.
MPI and the A1 ladder remain held. No push is authorized.

## Fixed contracts and execution setup

Class R: fixture exchange/eigenvalue/total identities within 1e-8 Ha;
fixture total/exchange energies within 1e-10 Ha of 10f6b39; maximum absolute
first-rebuild band-space feedback difference within 1e-10 of 10f6b39; final
A0 E/HOMO/E_H/E_x within 1e-10 Ha of evd-0010, with identities within 1e-8.
Budget: two fixtures before, two after Task 1, two after Task 2; two feedback
dumps; one A0 at 2400 s; one report-only A1 timing at 1200 s. No failure
diagnostics. The two feedback probes stop after the first `exchange_feedback`
output, before lifting, and each has a 2400 s safety supervisor.

Build/run environment:

```sh
export TBLIS_DIR="$(brew --prefix tblis)" HDF5_DIR="$(brew --prefix hdf5)"
export RUSTFLAGS="-L $(brew --prefix fftw)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" RAYON_NUM_THREADS=10
export CARGO_TARGET_DIR=/Users/zerozaki07/tmp/libmuffintin/target
```

Untracked scratch trees are `git archive 10f6b39` extractions with the main
checkout's `.local-deps` symlinked. `/tmp/h2-hf-contract-before` supplies
the baseline; `/tmp/h2-hf-contract-task1` will receive only Task 1 production
changes. Temporary fixture prints expose `CONTRACT_FIXTURE` energies and
`CONTRACT_ID` residuals without changing assertions. Fixture commands:

```sh
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw -- --nocapture
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
```

Baseline logs: `fixture-before-fftw.log`, `fixture-before-direct.log`.
Both passed. Total and exchange energies were
−5.32924235801359503e-1 and −1.71511931745922745e-5 Ha in both builds.
Maximum printed identity residual was 3.42933147157165052e-16 Ha.

The scratch-only feedback hook writes all k blocks immediately after the
first `exchange_feedback(&driver)` and exits 0. It is never committed as
production source. `compare_feedback.py` checks every finite complex entry,
block order, and dimensions at the fixed 1e-10 bound. Dump format: little-endian
u64 block count, then each block's u64 k index and dimension followed by the
full row-major matrix as f64 real/imaginary pairs. Dumps stay beside this README.

Baseline probe build, from `/tmp/h2-hf-contract-before`:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
H2_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-contraction-perf/feedback-before.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s "$CARGO_TARGET_DIR/release/examples/h2_hf" /tmp/libmuffintin-h2-hf/contract-before --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
```

Logs: `build-before-feedback.log`, `feedback-before.log`.

## Implementation and staged checks

Task 1 processes one occupied band at a time. Vertex columns are copied
contiguously into a column-major `[n_aux, n_target]` tensor. Two GEMMs form
`V C` and `C† (V C)`; the existing weights are applied and accumulated into
the same row-major output. Target tiling uses
`floor(1_000_000_000 / (n_aux * sizeof(Complex64)))`, limited to the target
count and at least one column, so ordinary blocks stay below 1 GB. A few
blocks, not all occupied bands, are resident at once. No factorization or
half-cost Hermitian reformulation is used.

The new borrowed rank-2 helpers call RSTSR 0.7.10 `rt::matmul_f` using the
existing shared `DeviceFaer`. The adjoint helper conjugates and transposes the
left block before that same route. Static thread configuration evidence:
RSTSR `device_faer/device.rs` makes the default device use `new(0)`;
`feature_rayon/device.rs` resolves zero to `rayon::current_num_threads()`;
`device_faer/matmul_impl.rs` passes the pool size as `faer::Par::Rayon`.
The run environment sets `RAYON_NUM_THREADS=10` before first tensor creation.
Actual utilization is reported from the single authorized A1 timing below.

Task 1 scratch copied only `operator.rs` and the two tensor helper files.
The same two fixture commands ran from `/tmp/h2-hf-contract-task1`, producing
`fixture-task1-fftw.log` and `fixture-task1-direct.log`. `compare_fixture.py task1`
reported total difference 0 and exchange difference 4.20128341838132968e-19 Ha
in both variants (bound 1e-10); maximum identity residual
4.83771009363032078e-16 Ha (bound 1e-8). Both class-R gates are closed.

Task 2 changes only the FFTW projection from 64-right-band einsum chunks to
256-right-band borrowed matmul chunks. Theta remains resident and is passed
by reference without per-chunk import or cloning. The spectra cache and
Rayon left-band structure remain unchanged. `/tmp/h2-hf-contract-after`
adds this source file to the Task 1 snapshot and retains the identical
temporary fixture prints. Its fixture logs are `fixture-task2-fftw.log`
and `fixture-task2-direct.log`.

Task 2's two fixtures passed with the same numerical results as Task 1.
The class-R fixture gates are closed (`fixture-task2-compare.log`).

The after-both scratch hook is identical to the baseline hook. Its build and
probe commands ran from `/tmp/h2-hf-contract-after`:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
H2_FEEDBACK_DUMP=/Users/zerozaki07/tmp/libmuffintin-harness/evidence/2026-09-08-h2-hf-contraction-perf/feedback-after.bin /usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s "$CARGO_TARGET_DIR/release/examples/h2_hf" /tmp/libmuffintin-h2-hf/contract-after --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 2
```

Logs: `build-after-feedback.log`, `feedback-after.log`. Both probes exited 0.
Comparison command from this evidence directory:

```sh
unalias python
unalias python3
/opt/homebrew/bin/python3 compare_feedback.py feedback-before.bin feedback-after.bin
```

`feedback-compare.log`: one k block, 1,060,900 finite entries, maximum
absolute difference 1.21430643318376497e-16 at `(k=0, row=1, column=1)`.
Class R, reference 10f6b39, bound 1e-10; PASS and closed.

## Final A0 preservation command

The main checkout is built with the same environment and the ordinary
`cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf`
command (`build-production.log`). No temporary instrumentation is present in
the main source. The only authorized A0 execution uses the evd-0002 settings:

```sh
set -o pipefail
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 2400s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 2>&1 | tee examples/h2_dft/results/hf-a0-contract.log
```

The reference `hf-a0.log` is not replaced. `compare_a0.py` checks the four
fixed evd-0010 values and the driver identities; it does not adjust a bound
or launch another calculation. The two performance commits will be made
after the authorized timing, allowing both bodies to cite actual user/real
thread evidence without rewriting previously recorded commit IDs.
