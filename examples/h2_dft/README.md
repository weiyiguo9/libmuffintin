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
  $E_{xc}$. The variational basis was the offset.

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
