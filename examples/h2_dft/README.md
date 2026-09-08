# H₂ DFT acceptance

Fixed geometry: two point hydrogen nuclei separated by 1.4 Bohr, two electrons,
nonmagnetic LDA-PW92. The production LAPW scalar route uses Koelling–Harmon
radials and periodic electrostatics. The nonrelativistic acceptance input sets
`speed-of-light = 137035.9895`, 1000 times the physical value, which suppresses
the leading relativistic corrections by a factor of one million without adding
a second solver. The value is owned by the calculation and serialized in
checkpoint metadata.

## Gates

The LAPW run is compared with a PySCF calculation in the same periodic box
(`periodic_reference.py`), not with the isolated molecule. Absolute total
energies of an LAPW box and a Gaussian density-fitted box carry independent
finite-size, basis-truncation and fitting errors of order 1 mHa each: the
periodic PySCF energy itself moves by 1.56 mHa between the 10 and 12 Bohr
boxes and is still 0.27 mHa from the isolated value at 12 Bohr. The occupied
eigenvalue shares the box, the $G=0$ electrostatic convention and most of the
truncation error, so it is the quantity that is expected to agree closely.

| Quantity | Gate |
|---|---|
| SCF total-energy change | at most 1e-8 Ha |
| SCF density RMS | at most 1e-7 |
| HOMO eigenvalue vs matched periodic PySCF | absolute difference at most 1 mHa |
| Total energy vs matched periodic PySCF | absolute difference at most 20 mHa |
| Required validity | finite results, two electrons, successful SCF |

`compare.py <lapw log> <periodic json> [<isolated json>]` evaluates the two
physical gates from the `energy_terms_ha` line and prints $E_{xc}$, $E-E_{xc}$
and the isolated-molecule comparison as diagnostics. With two electrons in one
Fermi–Dirac occupied level at 1 mHa, the band energy is twice the HOMO
eigenvalue to far below 1e-10 Ha.

## References

`reference.py` (isolated) and `periodic_reference.py` (Gamma-only cubic cell,
Gaussian density fitting) use PySCF 2.14.0 with aug-cc-pV5Z and grid level 7.
Their explicit `LDA_X,LDA_C_PW_MOD` functional matches the PW92 coefficient
`0.0310907` in `crates/mt-dft/src/xc.rs`; plain `LDA` would select a different
correlation functional. Both scripts run at `verbose = 5`, fail if the SCF does
not converge, and write the energy, HOMO/LUMO and PySCF's `scf_summary`
decomposition (`e1` kinetic plus nuclear attraction, `coul`, `exc`, `nuc`) to
their sole command-line argument.

| Reference | Energy (Ha) | HOMO (Ha) | $E_{xc}$ (Ha) |
|---|---:|---:|---:|
| isolated | −1.1373018 | −0.3772929 | −0.6528929 |
| 10 Bohr box | −1.1391305 | −0.3728033 | −0.6487011 |
| 12 Bohr box | −1.1375729 | −0.3728961 | −0.6520516 |

## LAPW invocation

The complete [molecule input](molecule.toml) constructs the neutral atomic
start directly from Cartesian coordinates; it does not require a checkpoint:

```sh
RUSTFLAGS="-L native=$(brew --prefix fftw)/lib" \
  HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo build --release -p libmuffintin-runtime --features fft-fftw --bin muffintin
RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib" \
  target/release/muffintin examples/h2_dft/molecule.toml
```

`molecule.toml` generates the 10 Bohr cell; `molecule-box12.toml` is the
corresponding 12 Bohr / 96³ input. Both mix with Pulay–Anderson
(β = 0.4, history 6), which converges in seven iterations.

For the parameterized Rust experiment, explicit checkpoint output and the
`energy_terms_ha` line that `compare.py` reads:

```sh
HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo run --release -p libmuffintin-runtime --features fft-fftw --example h2_dft -- \
  <output dir> 10 6 18 137035.9895 78
```

The arguments are output directory, cubic box side in Bohr, orbital reciprocal
cutoff, density/potential reciprocal cutoff in inverse Bohr, speed of light in
atomic units, cubic XC integration grid size, and an optional muffin-tin
radius in Bohr (default 0.65; the spheres must not overlap). The example uses
401 radial points, orbital angular cutoff 8, field angular cutoff 8, and
explicit −0.4 Ha linearization energies. It writes the
atomic-start checkpoint and Input V3 before SCF, and the accepted restart only
after SCF converges.

## Interstitial XC

The interstitial XC kernel is evaluated at every point of the midpoint grid,
including the smooth plane-wave continuation inside the spheres, and the
interstitial region enters through the analytic step function truncated to the
density layout. The energy contractions weight each point by that truncated
step and the operator-facing potential is its grid product with $V_{xc}$, so
the potential coefficients are the exact derivative of the discrete energy and
the represented region does not depend on which grid points fall inside a
sphere. Local spin densities driven below zero by finite Fourier or harmonic
truncation are clamped to zero before the kernel. The earlier real-space mask
dropped grid points inside the spheres, which made $E_{xc}$ move by 0.95 mHa
between 39³ and 78³ grids and turned vacuum densities of −3e-11 Bohr⁻³ into
hard SCF failures; both effects are gone.

## Current result

All runs: 10 Bohr box, orbital cutoff 6, field cutoff 18, `c = 137035.9895`,
compared with the 10 Bohr periodic reference. The first row is the superseded
real-space mask, kept for scale (its log is in the history of this directory).

| XC integration | Energy (Ha) | HOMO (Ha) | $E_{xc}$ (Ha) | ΔHOMO (mHa) | ΔE (mHa) | Δ$E_{xc}$ (mHa) | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| real-space mask, 78³ | −1.1379437513 | −0.3724715 | −0.6476333 | 0.332 | 1.187 | 1.068 | 408.6 |
| truncated step, 78³ | −1.1383185408 | −0.3726322 | −0.6481157 | 0.171 | 0.812 | 0.585 | 458.4 |
| truncated step, 60³ | −1.1383185408 | −0.3726322 | −0.6481157 | 0.171 | 0.812 | 0.585 | 457.3 |

Both truncated-step runs converge in seven iterations and pass every gate, and
their energies agree to 3e-13 Ha, against the 0.95 mHa that the mask moved
between 39³ and 78³: the XC grid is no longer a physical parameter here.

## Where the remaining offset sits

The 0.8 mHa offset of the 78³ row was split by varying one parameter at a
time at 10 Bohr, all against the 10 Bohr periodic reference. Differences are
LAPW minus PySCF in mHa; "rest" is $E-E_{xc}$.

| RMT (Bohr) | orbital cutoff | field cutoff | Energy (Ha) | ΔHOMO | ΔE | Δ$E_{xc}$ | Δrest | Wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.65 | 6 | 12 | −1.1407820830 | 0.546 | −1.652 | 1.053 | −2.705 | 466 |
| 0.65 | 6 | 18 | −1.1383185408 | 0.171 | 0.812 | 0.585 | 0.227 | 458 |
| 0.65 | 6 | 24 | −1.1381825688 | 0.154 | 0.948 | 0.564 | 0.384 | 555 |
| 0.50 | 6 | 18 | −1.1368094215 | 0.757 | 2.321 | 1.660 | 0.662 | 459 |
| 0.60 | 6 | 18 | −1.1380697975 | 0.323 | 1.061 | 0.873 | 0.188 | 437 |
| 0.65 | 7 | 18 | −1.1390599351 | 0.050 | 0.071 | 0.188 | −0.117 | 1314 |

- Field cutoff: 12 → 18 moves the energy by 2.46 mHa, 18 → 24 by 0.14 mHa,
  so 18 is converged to about 0.1 mHa and the remaining offset does not sit
  there. Note that the field layout now also truncates the XC step function.
- Sphere radius at fixed plane-wave cutoff 6: 0.50 → 0.60 → 0.65 lowers the
  energy by 1.26 and 0.25 mHa, with ΔHOMO and Δ$E_{xc}$ shrinking in step.
  That is the plane-wave basis becoming more complete (RKmax 3.0 → 3.9), not
  an independent sphere effect.
- Orbital cutoff 6 → 7 (RKmax 3.9 → 4.55) lowers the energy by 0.74 mHa and
  leaves 0.07 mHa in the total, 0.05 mHa in the HOMO and 0.19 mHa in
  $E_{xc}$. The variational basis was the offset. Cutoff 8 (about 8600
  plane waves) exceeded the memory of the 10-thread development machine and
  was not completed.

The reference side was checked independently: Gaussian density fitting with
the same auxiliary basis changes the isolated energy by 0.001 mHa; the
periodic reference converges to the isolated exact value as −1.83, −0.27,
−0.04 and −0.006 mHa at 10, 12, 14 and 16 Bohr; and aug-cc-pV6Z lowers the
aug-cc-pV5Z energy by 0.014 mHa both isolated and in the 10 Bohr box. The
periodic HOMO stays 4.5 to 2.2 mHa below the isolated value from 10 to 16 Bohr
because of the $1/L^3$ average-potential offset, which is why the eigenvalue
gate compares boxes of the same size. Raw logs are the
`results/box10-*-step.log` files.

## Execution cost

RSTSR/TBLIS and Faer use the existing Rayon pool; `RAYON_NUM_THREADS=10` makes
the machine's 10-thread setting explicit. With `fft-fftw`, the interstitial
physical metric is evaluated as an alias-free zero-padded Fourier correlation,
reducing its reciprocal-space work from quadratic to FFT complexity. The
non-FFTW build retains a row-parallel direct contraction. Exactly identical
up/down scalar inputs also share the basis, eigensolve, and density contraction;
distinct spin channels retain their independent path. Unused extended-core/XC
spherical averaging is skipped for the all-explicit, no-core H2 route. The XC
grid is not the cost driver: the 12 Bohr basis (about 6300 plane waves at
cutoff 6) dominates, and each 12 Bohr iteration at field cutoff 18 takes well
over 1000 s.

## Hartree–Fock

The H₂ Hartree–Fock acceptance uses the Gamma-only spinor-first valence driver,
with no core states and the same neutral atomic-superposition start as the DFT
example. The Dirac radial tag is retained on this path while
`speed-of-light = 137035.9895` suppresses the relativistic correction. The
periodic finite-body kernel is compared only with the same-box PySCF
`exxdiv=None` reference; the sharp and smoothed Spencer–Alavi kernels are
compared with the isolated reference.

| Reference | $E$ (Ha) | HOMO (Ha) | $E_H$ (Ha) | $E_x$ (Ha) | $E_x/E_H$ |
|---|---:|---:|---:|---:|---:|
| isolated | −1.1336107 | −0.5946525 | 1.3171828 | −0.6585914 | −0.5000000 |
| box 8, `exxdiv=None` | −0.8124204 | −0.2640980 | 0.5951402 | −0.2975701 | −0.5000000 |
| box 10, `exxdiv=None` | −0.8624143 | −0.3183767 | 0.7534239 | −0.3767120 | −0.5000000 |

Build the parameterized example with the FFTW backend:

```sh
RUSTFLAGS="-L native=$(brew --prefix fftw)/lib" \
  HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo build --release -p libmuffintin-runtime --features fft-fftw \
  --example h2_hf
```

The first positional argument is the output directory. The remaining controls
are named options: `--box`, `--orbital-g`, `--field-g`, `--product-g`,
`--product-lmax`, `--overlap-tolerance`, `--exchange-coulomb`,
`--fock-fourier-g`, `--fock-smoothing-omega`, `--lexp`, `--speed-of-light`,
`--rmt`, `--verbosity` (0 quiet, the default; 1 progress; 2 timings), and the
A0 diagnostic control `--fock-max-iterations`.

### FFTW interstitial batching

The initial `fft-fftw` batching change retained FFT correlations per band pair but
projects up to 64 pairs sharing a left band in one TBLIS contraction. The
focused fixture passed once in each build. Its 432 first-build interstitial
coefficients were dumped from an untracked test hook and compared as
$\max_i\lvert z_i^{\mathrm{FFTW}}-z_i^{\mathrm{direct}}\rvert$.

```text
DIGIT / PASS
Q: exchange/eigenvalue/total identity residuals (Ha); class: R; ref: fixture
bound: 1e-8; Delta: every residual passed the unchanged fixture assertions
checks: default and fft-fftw focused tests passed; runs: 2; numerical verification closed

DIGIT / PASS
Q: first-build interstitial vertex coefficients; class: R; ref: non-fftw
bound: 1e-10 absolute; Delta: 0; d: 0
checks: 432 complex coefficients, no key mismatches; comparisons: 1; numerical verification closed
```

At the A0 settings, the pre-change timing had `vv.mt_contraction = 49.668 s`
and `vv.interstitial > 300 s`. The single post-change scratch timing had
`vv.mt_contraction = 35.201 s` and `vv.interstitial > 564 s` when the process
cap was reached at 600 s, so the first MPB rebuild still did not complete.
The muffin-tin time is spent in the already band-batched dense site contraction
`einsum("il,aij,jr->alr")`; the same batching fix does not apply there.

### Occupied left bands

The batching above could not help because the pair count, not the per pair
cost, was the problem: the Gamma valence rebuild selected every
`(left_band, right_band)` pair of the 1030 band spinor window. The exchange
assembly reads a column only when its weight $w_{k-q} f_{k-q,l}$ is nonzero,
so the left band is now restricted to bands the assembly can weight. The
assembly has no separate occupation threshold, and the k weights are
validated positive, so its criterion is exactly `occupation != 0.0`, the
same one the relaxed core frame already applied. Because
`build_spinor_mpb_exchange` requires every square layout column, the call now
goes through the crate internal VV consumer the relaxed core frame uses, over
an auxiliary basis that does not depend on the selection list.

```text
DIGIT / PASS
Q: exchange/eigenvalue/total identity residuals (Ha); class: R; ref: fixture
bound: 1e-8; Delta: every residual passed the unchanged fixture assertions
checks: default and fft-fftw focused tests passed; runs: 2; numerical verification closed

DIGIT / PASS
Q: fixture total_energy and exchange_energy (Ha); class: R; ref: before the change
bound: 1e-10; Delta: 0; d: 0
checks: total -5.32924235801359503e-1 and exchange -1.71511931745922745e-5, identical in both builds before and after; runs: 2 before and 3 after, one after run repeated to confirm the incremental rebuild; numerical verification closed

STUDY / REPORT
Q: first rebuild selection count and vv.interstitial seconds at the A0 settings; class: P; ref: 1 060 900 selections and > 420 s
bound: none; observed: 39 140 selections (38 occupied left bands of 1030) and vv.interstitial 27.281 s
context: vv.mt_contraction fell from 35.201 s to 0.613 s; the whole first vv.mpb_rebuild took 28.658 s, and the committed A0 example returned in 251 s instead of exceeding 1800 s
```

The timing log is
`evidence/2026-09-08-h2-hf-a0-probe/timing-occupied.log` on the `harness`
branch. The scratch example and the temporary selection count print were
never committed.

### Cached pair spectra and parallel left bands

Main `8140032` caches each selected band's two spin spectra, sums the spin
products before one inverse per pair, and uses a Rayon-local `PairFft`
workspace. Projection still contracts up to 64 right bands at once. At A0,
38 left bands and 1030 right bands need 41,276 transforms instead of
234,840. The right cache occupies 305,242,560 bytes; callers above the 4 GB
cache limit use disjoint right-band chunks.

| First rebuild | Transforms | `vv.interstitial` (s) | `vv.mpb_rebuild` (s) |
|---|---:|---:|---:|
| earlier evd-0005 reference | 234840 | 27.281043 | 28.657877 |
| 421a447 baseline, 1 thread | 234840 | 34.261638 | 37.398714 |
| cached, 1 thread | 41276 | 21.380939 | 24.576575 |
| cached, 10 threads | 41276 | 9.604797 | 11.903934 |

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identities (Ha); class: R; ref: fixture
bound: 1e-8; Delta: maximum 3.42933147157165052e-16
checks: one changed run with fft-fftw and one without; closed

DIGIT / PASS
Q: fixture total_energy and exchange_energy (Ha); class: R; ref: 421a447
bound: 1e-10; Delta: 0 in both feature variants
checks: two baseline and two changed fixture runs; closed

DIGIT / PASS
Q: maximum absolute first-rebuild A0 interstitial vertex difference; class: R; ref: 421a447 fft-fftw
bound: 1e-10; Delta: 6.50521335639941991e-19
checks: 20,157,100 finite complex entries with matching selection labels; two dumps; closed
```

Timing is class P, report only. Evidence and the comparison script are under
`evidence/2026-09-08-h2-hf-pair-fft-perf/` on `harness` (evd-0009).
There were seven executions: four fixtures, two dumps, and one additional
timing; the changed dump also supplied the one-thread timing. No additional
diagnostic or numerical verification followed the passes.

### A0 smoke

A0 was previously run three times with the same command: orbital cutoff 4, field
cutoff 12, product cutoff 4, product $l_{max}=2$, overlap tolerance $10^{-4}$,
box 8, and the periodic finite-body kernel, under an 1800 s supervisor. Only
the example's Fock exit tolerances differ between the runs. None of the three
produced a passing row: the driver returns its gate error before the example
prints `hf_energy_terms_ha`, so no run carries $E$, HOMO, $E_H$, or $E_x$, and
none prints a Fock iteration count or an `hf_final` wall time of its own.

| Run | Fock exit tolerances (density, feedback Ha) | Outcome | Eigenvalue identity (Ha) | Wall (s) | Log |
|---|---|---|---:|---:|---|
| first | $10^{-5}$, $10^{-5}$ | gate error | $3.4089473164444770\times10^{-4}$ | 251 | [`results/hf-a0-fock1e-5.log`](results/hf-a0-fock1e-5.log) |
| second | $10^{-7}$, $10^{-8}$ | gate error | $1.6477445782814293\times10^{-7}$ | 282 | [`results/hf-a0-fock1e-8.log`](results/hf-a0-fock1e-8.log) |
| third | $10^{-9}$, $10^{-10}$ | killed at the cap | none produced | > 1800 | [`results/hf-a0-fock1e-10-cap1800.log`](results/hf-a0-fock1e-10-cap1800.log) |

The three diagnostics the plan allows, spent on the first run:

| # | Diagnostic | Question | Answer | Log |
|---:|---|---|---|---|
| 1 | `--fock-max-iterations 256` | convergence limit or defect? | identical residual $3.4089473164444770\times10^{-4}$; the Fock loop already exits before the limit, so this is not a limit | [`results/hf-a0-diag1-fock256.log`](results/hf-a0-diag1-fock256.log) |
| 2 | orbital 3 / product 3 | dimension dependent defect or setup error? | the same identity fails, at $1.4380668996653856\times10^{-4}$; it does not change which identity fails | [`results/hf-a0-diag2-orb3prod3.log`](results/hf-a0-diag2-orb3prod3.log) |
| 3 | `gamma_valence_hf` fixture at `product_g_max` 4 | does the driver hold its identities at this product cutoff? | yes, every identity stays $\le 10^{-8}$ on the one atom fixture | [`results/hf-a0-diag3-fixture-productg4.log`](results/hf-a0-diag3-fixture-productg4.log) |

The first two runs place the residual at 16 to 34 times the feedback tolerance
the Fock loop exited on, which is why the tolerance was stepped twice: the
example had inherited the Kr example's $10^{-5}$ and $10^{-5}$ Ha, then took
the `gamma_valence_hf` fixture's $10^{-7}$ and $10^{-8}$ Ha, and now asks for
$10^{-9}$ and $10^{-10}$ Ha against the driver's fixed $2\times10^{-8}$
identity gate. The third run never reached that gate. It was killed by the
1800 s supervisor with exit 124 after writing only its header, so it has no
residual and it neither confirms nor refutes the prediction that the residual
would fall to between $10^{-9}$ and $4\times10^{-9}$. The command, the
dimension, and the spinor basis are identical to the second run, which reached
the gate in 282 s; the only change is the exit tolerance, so the loop was still
inside its first Fock cycle when the cap arrived. The loop's own limit is above
the cap: `FOCK_MAX_ITERATIONS` is 128 and one exchange rebuild at these
settings was measured at 28.7 s.

Historical third-run handoff (superseded by the fourth run below):

```text
DIGIT / HANDOFF
Q: valence eigenvalue identity residual (Ha); class: A; ref: 0
bound: 1e-8; Delta: unavailable, no outer iteration completed
checks: killed at the 1800 s supervisor cap, exit 124; runs: 1; no diagnostic authorized, the run ended in a wall clock kill and not in a Fock not-converged error
prediction: untested
unresolved: can the Fock loop reach 1e-10 Ha within the 1800 s cap at the A0 dimension, or does the residue versus floor question need a longer cap or an intermediate tolerance to be answerable?
```

The fourth A0 run, after spectrum caching (`8140032`), Progress residual
output (`7efe82c`), and the default-quiet verbosity flag (`ca2e5ed`), retained
the third run's numerical settings. It added `--verbosity 1` and used a
4500 s supervisor. Outcome **(a)**: converged and passed. Every inner Fock
solve took seven iterations; eight outer iterations used 56 exchange rebuilds.

| $E$ (Ha) | HOMO (Ha) | $E_H$ (Ha) | $E_x$ (Ha) | Exchange id. (Ha) | Eigenvalue id. (Ha) | Total id. (Ha) | $\lvert E_x+E_H/2\rvert$ (Ha) | Fock iter. | Wall (s) | Log |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| −0.59291354371542793 | −0.12617886551775270 | 0.37293092552423746 | −0.023151590595997563 | 0 | 7.2523410887814777e-10 | 7.2523409500035996e-10 | 0.16331387216612117 | 7 each, 56 total | 1934.680438 | [`results/hf-a0.log`](results/hf-a0.log) |

```text
DIGIT / PASS
Q: driver exchange/eigenvalue/total identity residuals (Ha); class: A; ref: 0
bound: 1e-8; Delta: maxima across all outer iterations 0 / 1.7276809149979755e-9 / 1.7276811092870048e-9
checks: converged, finite energies; final electron_count=2.0000000000000058 (bound 1e-8); outer density RMS=2.9068395716484560e-8; exit 0
runs: 1; no diagnostics; A0 numerical verification closed
prediction: first-outer eigenvalue residual is inside [1e-9, 4e-9] and 95.37 times below the old 1.6477445782814293e-7 residual; final residual is below that interval
```

The final inner-loop density and feedback residuals were
1.7911055967367636e-11 and 1.3830918181578777e-11 Ha, respectively. Supervisor
wall time was 1935.39 s. `hartree_exchange` is recorded, not judged by A0;
this pass does not claim product-basis or external-energy acceptance.

### A1 identity-floor study

Base settings are orbital cutoff 5, field cutoff 12, product cutoff 6,
product $l_{max}=4$, overlap tolerance $10^{-4}$, box 8, and the periodic
finite-body kernel. Each row changes one base setting.

| Row | Change | $E$ (Ha) | HOMO (Ha) | $E_H$ (Ha) | $E_x$ (Ha) | Exchange id. (Ha) | Eigenvalue id. (Ha) | Total id. (Ha) | $\lvert E_x+E_H/2\rvert$ (Ha) | Fock iter. | Wall (s) | Log |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | base | unavailable (cap, profiled retry) | – | – | – | – | – | – | – | 3 completed | 1800.76 | [`results/hf-a1-row1.log`](results/hf-a1-row1.log) |
| 2 | product $G=4$ | not run | – | – | – | – | – | – | – | – | – | – |
| 3 | product $G=8$ | not run | – | – | – | – | – | – | – | – | – | – |
| 4 | product $G=10$ | not run | – | – | – | – | – | – | – | – | – | – |
| 5 | product $l_{max}=2$ | not run | – | – | – | – | – | – | – | – | – | – |
| 6 | product $l_{max}=6$ | not run | – | – | – | – | – | – | – | – | – | – |
| 7 | overlap tolerance $10^{-5}$ | not run | – | – | – | – | – | – | – | – | – | – |
| 8 | overlap tolerance $10^{-6}$ | not run | – | – | – | – | – | – | – | – | – | – |
| 9 | `lexp=18` | not run | – | – | – | – | – | – | – | – | – | – |
| 10 | field cutoff 18 | not run | – | – | – | – | – | – | – | – | – | – |

After A0 passed, the first A1 base attempt ran under its authorized 1800 s cap.
The supervisor returned exit 124 at 1800.83 s after two completed Fock
iterations and no completed outer iteration. The last printed density and
feedback residuals were 1.0988849307297540e-4 and 1.3734823798679154e-4 Ha,
respectively. No energy, HOMO, three-identity row, or `hartree_exchange` was
produced; SCF convergence and final electron count were not established.
That first log is preserved at
[`results/hf-a1-row1-cap1800.log`](results/hf-a1-row1-cap1800.log).

Historical first-attempt stamp:

```text
DIGIT / HANDOFF
Q: A1 base hartree_exchange (Ha); class: P; ref: 0
bound: none (study); Delta: unavailable
checks: required converged finite output, three identities within 1e-8 Ha, and electron count 2 within 1e-8 not established; supervisor exit 124
runs: 1; no diagnostics; unresolved: the A1 base row cannot finish within the authorized 1800 s cap
```

A1 rows 2–10, A1v, A2, B, and Bv were not run after this resource-budget
handoff. The optional pair-level MPI task was not started; no MPI result or
speedup is claimed. This does not reopen A0 or the Task 1 preservation passes.

### Gamma Fock phase profile

Main `79623b8` adds `gamma.fock.*` and `gamma.rebuild.*` timers, enabled only
by `--verbosity 2`, without changing numerical operations. Two capped
report-only class-P runs measured A0 Fock iterations 1 and 2 (17.479391 and
15.698071 s) and A1 base iteration 1 (482.082668 s). The profile runs are
not plan rows and do not reopen the accepted A0 result.

The conditional cache change was **skipped**: A1 basis compilation took
7.589190 s and Coulomb assembly 8.563591 s, together 16.152781 s or 3.351%
of the Fock iteration, below the fixed 30% threshold. The largest measured
phase was exchange contraction, 290.551110 s (60.270%); MPB construction
was 151.026190 s (31.328%), including the 143.437000 s vertex rebuild.
No cache, contraction, or pair-FFT change was made. There are no Task C
fixture results or after-cache A0 timings; MPI remains held.

Full phase tables, stdout/stderr logs, commands, and the extraction script
are in `evidence/2026-09-08-h2-hf-gamma-profile/` on `harness` (evd-0011).
The A0 and A1 profile supervisors returned exit 124 at 240.12 and 2400.31 s,
respectively, as bounded profiling runs rather than converged calculations.

The prescribed A1 base retry then used the unchanged evd-0010 command and
1800 s cap (`--verbosity 1`). It exited 124 at 1800.76 s after three Fock
iterations and no completed outer iteration. Last printed density/feedback
residuals were 6.3664758060341759e-5 / 7.9480430635287458e-5 Ha. The table
above now records this retry; the previous cap log remains preserved.

```text
DIGIT / HANDOFF
Q: A1 base hartree_exchange (Ha); class: P; ref: 0
bound: none (study); Delta: unavailable
checks: supervisor exit 124 at 1800.76 s; three completed Fock iterations, no converged energy/identity row or final electron count
profile: compile plus Coulomb 16.152781 / 482.082668 s = 3.351%, below the 30% cache condition; contraction 290.551110 s = 60.270%
runs: one Task D retry; three executions for H2-HF-PROFILE including its two profiles; no diagnostics
unresolved: A1 base still cannot finish within its authorized 1800 s cap; stop before rows 2–10, A1v, A2, B, and Bv
```

No numerical acceptance is inferred from the timing data, and the earlier
A0 pass remains closed. No Task C fixtures or A0 rerun were performed.

### BLAS-3 contraction and projection

Main `16ff436` contracts one occupied band at a time with RSTSR 0.7.10/faer
GEMMs, gathering contiguous column-major vertex blocks and tiling target
columns above 1 GB. It preserves weights, sign, and row-major output.
Main `700fa1f` uses the same borrowed matmul route for FFTW projection in
256-right-band chunks, retaining resident theta, spectra caching, and the
existing Rayon left-band structure. The Fock loop and kernels are unchanged.

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identities (Ha); class: R; ref: fixture
bound: 1e-8; Delta: maximum 4.83771009363032078e-16 after Task 1 and after both tasks
checks: two baseline fixtures, two after contraction, two after projection, with and without fft-fftw; closed

DIGIT / PASS
Q: fixture total_energy and exchange_energy (Ha); class: R; ref: 10f6b39
bound: 1e-10; Delta: 0 / 4.20128341838132968e-19 at both stages and both feature variants; closed

DIGIT / PASS
Q: first-rebuild band-space exchange feedback; class: R; ref: 10f6b39
bound: 1e-10 absolute; Delta: maximum 1.21430643318376497e-16
checks: all 1,060,900 finite entries in the 1030-by-1030 k block; two dumps; closed

DIGIT / PASS
Q: A0 E/HOMO/E_H/E_x (Ha); class: R; ref: evd-0010
bound: 1e-10; Delta: 1.22124532708767219e-15 / 1.76803016671556179e-14 / 2.52575738102223113e-14 / 3.95516952522712018e-15
checks: converged, finite energies, maximum driver identity 1.72767455897115951e-9 Ha <= 1e-8; final electron-count error 5.32907051820075139e-15
runs: one A0; closed without diagnostics or repeats
```

A0 again used eight outer iterations and 56 Fock iterations. Driver wall was
1166.331817 s (supervisor 1166.76 s), versus evd-0010's 1934.680438 s.
The new log is [`results/hf-a0-contract.log`](results/hf-a0-contract.log);
the evd-0010 reference log is unchanged.

| A1 first-iteration timing | Before, evd-0011 (s) | After both changes (s) |
|---|---:|---:|
| `gamma.rebuild.contraction` | 290.551110 | 179.435197 |
| `gamma.rebuild.mpb` | 151.026190 | 96.477896 |
| `gamma.fock.iteration` | 482.082668 | 313.415279 |

This was one class-P report-only timing under the 1200 s cap, exit 124,
not an A1 plan row. Whole-run user/real was 3857.62 / 1200.52 = 3.213 cores,
versus approximately 3.1 before. The faer pool is configured for ten threads,
but the measurement does **not** establish sustained ten-core utilization.
No additional profiling was run to explain the remaining utilization gap.

The full evidence, feedback dumps, and comparison scripts are under
`evidence/2026-09-08-h2-hf-contraction-perf/` on `harness`. Ten numerical
executions exhausted the agreed list: six fixtures, two feedback probes,
one A0, and one A1 timing. MPI and the A1 ladder remain held; the study
tolerance policy is still a separate user decision.

### Pair-level MPI

Main `f189669` adds the default-off runtime `mpi` feature through rsmpi 0.8.2
and mpi-sys 0.2.4, with MSRV 1.89 unchanged. Main `33a761d` assigns occupied
left bands round-robin across ranks, constructs and contracts only rank-local
vertices, and Allreduces one band-feedback block per k. Basis compilation,
Coulomb assembly, spinor solves, occupations, and density remain redundant.
The `h2_hf` binary owns Funneled initialization; the library never initializes
or finalizes MPI. Output is rank-zero-only, with an exact final-state agreement
check before printing.

The communicator setter accepts the raw `MPI_COMM_WORLD` handle and verifies
it, but reconstructs safe world views rather than calling rsmpi `FromRaw`.
The latter is an owning constructor that explicitly cannot adopt or free a
system communicator. Root [`README.md`](../../README.md) documents Snellius
build, launch, module, and per-rank Rayon-thread rules (`88b212d`).

```text
DIGIT / PASS
Q: fixture exchange/eigenvalue/total identities; class: R; ref: fixture
bound: 1e-8; Delta: every unchanged assertion passed
checks: default single process, fft-fftw+mpi single process, and the MPI-enabled test binary under mpirun -n 2; genuine Funneled initialization in the fixture
runs: 3; closed

DIGIT / HANDOFF
Q: first-rebuild band-space feedback at ranks 2 and 4 versus rank 1; class: R; ref: rank 1
bound: 1e-12 absolute; Delta: unavailable
checks: rank-one scratch probe wrote all 1,060,900 entries, then exited 1 because process::exit bypassed Universe drop and MPI_Finalize
runs: 1 feedback probe; ranks 2 and 4 not run; no diagnostics authorized
unresolved: obtain a cleanly finalized first-rebuild probe without spending an unapproved replacement run
```

The feedback failure belongs to the scratch evidence hook, after its flushed
rank-one dump, not to the committed distribution path. Nevertheless, MPI
execution failure cannot be accepted numerically. Per the fixed stop rule,
rank-two/rank-four dumps, the rank-two A0 run, and all three report-only A1
timings were not run. The A1 ladder was not restarted. Evidence is under
`evidence/2026-09-08-h2-hf-mpi/` on `harness`.

### B kernel study

The B rows use the accepted A1v settings with orbital cutoff 5.

| Row | Kernel | Box | Fock Fourier $G$ | Omega | $E$ (Ha) | HOMO (Ha) | $E_H$ (Ha) | $E_x$ (Ha) | Exchange id. (Ha) | Eigenvalue id. (Ha) | Total id. (Ha) | $\lvert E_x+E_H/2\rvert$ (Ha) | Fock iter. | Wall (s) | Log |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | sharp Spencer–Alavi | 8 | product $G$ | – | not run | – | – | – | – | – | – | – | – | – | – |
| 2 | sharp Spencer–Alavi | 8 | $2\times$ product $G$ | – | not run | – | – | – | – | – | – | – | – | – | – |
| 3 | sharp Spencer–Alavi | 10 | product $G$ | – | not run | – | – | – | – | – | – | – | – | – | – |
| 4 | sharp Spencer–Alavi | 12 | product $G$ | – | not run | – | – | – | – | – | – | – | – | – | – |
| 5 | smoothed Spencer–Alavi | 8 | product $G$ | 0.8 | not run | – | – | – | – | – | – | – | – | – | – |
| 6 | smoothed Spencer–Alavi | 8 | product $G$ | 1.6 | not run | – | – | – | – | – | – | – | – | – | – |
| 7 | smoothed Spencer–Alavi | 8 | product $G$ | 3.2 | not run | – | – | – | – | – | – | – | – | – | – |
| 8 | smoothed Spencer–Alavi | 12 | product $G$ | 0.8 | not run | – | – | – | – | – | – | – | – | – | – |

The closed B row list and Bv verdict were not run after the A1 base-row
handoff. No Bv stamp exists because no box 12 sharp kernel value was
produced.
