# H2 HF pair-FFT performance

Authorization: evt-0025, ADR-0006, and the H2-HF-PERF task brief. Baseline:
main `421a44750468e8d34a375a6dea387f07fa82da4e`. No push authorized.

## Scratch setup and fixed contracts

The baseline is an untracked `git archive 421a447` extraction at
`/tmp/h2-hf-perf-baseline`, with `.local-deps` symlinked to the main checkout.
The fixture has temporary `PERF_FIXTURE` energy and `PERF_ID` diagnostic
prints; no assertions or numerical settings were changed. Scratch instrumentation
is not part of the production commits.

Class R bounds: all fixture identities 1e-8 Ha; before/after fixture total
and exchange energies 1e-10 Ha; maximum absolute difference over every A0
interstitial selection and plane-wave entry 1e-10. Timing is a report-only
class P study, not physical acceptance. Budget: two baseline and two changed
fixture runs, two vertex dumps, two changed timings; no numerical diagnostics.

Common build and run environment:

```sh
export TBLIS_DIR="$(brew --prefix tblis)" HDF5_DIR="$(brew --prefix hdf5)"
export RUSTFLAGS="-L $(brew --prefix fftw)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" RAYON_NUM_THREADS=10
export CARGO_TARGET_DIR=/Users/zerozaki07/tmp/libmuffintin/target
```

Baseline fixture commands, from `/tmp/h2-hf-perf-baseline`:

```sh
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf --features fft-fftw -- --nocapture
cargo test --release -p libmuffintin-runtime --test gamma_valence_hf -- --nocapture
```

Outputs: `fixture-before-fftw.log`, `fixture-before-direct.log`. An initial
Cargo invocation could not load the archive's absent `.local-deps` directory;
it executed no numerical code. The symlink above fixed the scratch setup.

`compare_vertices.py` checks dimensions, ordered `(k, left_band, right_band)`
labels, finite real and imaginary components, and the maximum complex absolute
difference over every entry. Binary format: little-endian u64 selection count
and width, followed by each row's three u64 labels and width pairs of f64
real and imaginary components. Bulky binary dumps stay in `/tmp`, not Git.

## Task 1 execution

Changed scratch tree: `/tmp/h2-hf-perf-after`, from the same archive with the
three production files copied from the working main tree. The identical
temporary fixture prints were copied from the baseline. The two fixture
commands above were run once each there, producing `fixture-after-fftw.log`
and `fixture-after-direct.log`. `fixture-compare.log` records zero total and
exchange energy differences for both feature variants; maximum reported
fixture identity residual is 3.42933147157165052e-16 Ha. Both gates are closed.

The scratch examples set `HfVerbosity::Timings`; scratch library instrumentation
drops the assembly and rebuild timers before writing the vertex buffer, then
exits successfully after the first rebuild. This is not an SCF convergence
claim. Each scratch example was built from its own directory with:

```sh
cargo build --release -p libmuffintin-runtime --features fft-fftw --example h2_hf
```

Build logs: `build-before-probe.log`, `build-after-probe.log`. First-rebuild
commands (the build directory is shared, so each uses its immediately preceding
scratch build):

```sh
RAYON_NUM_THREADS=1 H2_VERTEX_DUMP=/tmp/h2-hf-perf-before.bin "$CARGO_TARGET_DIR/release/examples/h2_hf" /tmp/h2-hf-perf-baseline/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65
RAYON_NUM_THREADS=1 H2_VERTEX_DUMP=/tmp/h2-hf-perf-after.bin "$CARGO_TARGET_DIR/release/examples/h2_hf" /tmp/h2-hf-perf-after/a0-t1 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65
```

Logs: `vertices-before-t1.log`, `vertices-after-t1.log`. The changed one-thread
dump also supplies the first of the two authorized changed timing samples.
Baseline selection count is 39,140 and width is 515 (20,157,100 complex entries).
The implementation count is 2 left-spin forwards per occupied band plus
2 right-spin forwards per distinct right band plus one inverse per pair:
76 + 2060 + 39140 = 41,276, versus 234,840 before. Right spectra use
305,242,560 bytes at A0; larger callers are partitioned into right-band cache
chunks at a 4,000,000,000-byte limit, with each band's forwards still done once.

The vertex comparison command was:

```sh
unalias python
unalias python3
/opt/homebrew/bin/python3 compare_vertices.py /tmp/h2-hf-perf-before.bin /tmp/h2-hf-perf-after.bin 1e-10
```

`vertex-compare.log`: maximum difference 6.50521335639941991e-19 at selection
30,930, column 0; all 20,157,100 entries finite. Class R, bound 1e-10, PASS;
numerical verification is closed, without extra diagnostics.

The second changed timing command (no dump) is:

```sh
RAYON_NUM_THREADS=10 "$CARGO_TARGET_DIR/release/examples/h2_hf" /tmp/h2-hf-perf-after/a0-t10 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65
```

Log: `timing-after-t10.log`. Production implementation: main `8140032`.
One combined static review of the delegated production changes and scratch
probe preceded the changed gates. No additional tests or numerical diagnoses
were run.

| First rebuild | Transforms | `vv.interstitial` (s) | `vv.mpb_rebuild` (s) |
|---|---:|---:|---:|
| evd-0005 reference | 234840 | 27.281043 | 28.657877 |
| baseline 421a447, 1 thread | 234840 | 34.261638 | 37.398714 |
| cached, 1 thread | 41276 | 21.380939 | 24.576575 |
| cached, 10 threads | 41276 | 9.604797 | 11.903934 |

Task 1 runs: four fixtures, two dump runs (the changed dump is also the
one-thread timing), and one additional ten-thread timing: seven executions.
All numerical gates passed; no MPI runs belong to this entry.

## Task 2 A0 command

Production main `ca2e5ed` includes the Task 1 implementation and the two
Progress/verbosity commits. The first compile of the progress-only change
reported a missing `hf_progress` import before any execution; the calls were
qualified and placed in `solve_fixed_potential` before rebuilding. No numerical
run was spent on that compile failure. `build-a0.log` is the successful build.

```sh
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" RAYON_NUM_THREADS=10
set -o pipefail
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 4500s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a0 --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a0.log
```

The command runs from the main checkout. The prior log is preserved at
`examples/h2_dft/results/hf-a0-fock1e-10-cap1800.log` on main. Class A bounds
and validity are those recorded in evt-0027: one run, no diagnostics.

A0 exited 0: eight outer iterations, seven Fock iterations per outer,
56 exchange rebuilds, driver wall 1934.680438 s and supervisor wall 1935.39 s.
The full accepted row is committed in the main example README at `35c200c`.
Maximum identity residual across the run is 1.7276811092870048e-9 Ha;
final electron-count error is approximately 5.8e-15. A0 outcome (a), PASS.

The next authorized A1 base row used the same environment and binary:

```sh
/usr/bin/time -p /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 1800s target/release/examples/h2_hf /tmp/libmuffintin-h2-hf/a1-row1 --box 8 --orbital-g 5 --field-g 12 --product-g 6 --product-lmax 4 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 --verbosity 1 2>&1 | tee examples/h2_dft/results/hf-a1-row1.log
```

A1 base exited 124 at supervisor wall 1800.83 s, after two completed Fock
iterations with no completed outer iteration. Last printed density and feedback
residuals: 1.0988849307297540e-4 and 1.3734823798679154e-4 Ha. There is no
converged energy, final electron count, identity row, or `hartree_exchange`.
The run was not a `FockNotConverged` error at 128, and establishes no floor.
Class P, report-only bound, DIGIT / HANDOFF at the authorized resource limit.

Task 2 used two executions: the passed A0 and timed-out A1 base. Task 3 MPI
was not started; its fixture/dump/timing counts are all zero. A1 rows 2–10,
A1v, A2, B, and Bv remain unrun. Total numerical executions across the brief:
nine, with no numerical diagnostics or repeats after a pass.
