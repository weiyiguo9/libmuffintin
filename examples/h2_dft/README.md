# H₂ DFT acceptance

Fixed geometry: two point hydrogen nuclei separated by 1.4 Bohr, two electrons,
nonmagnetic LDA-PW92. The production LAPW scalar route uses Koelling–Harmon
radials and periodic electrostatics; the molecular reference is isolated.
The nonrelativistic acceptance input sets `speed-of-light = 137035.9895`,
1000 times the normal value. This suppresses leading relativistic corrections
by a factor of one million without adding a second solver. The value is owned
by the calculation and serialized in checkpoint metadata; rebuilding or
modifying a global source constant is no longer required. Periodic image
effects still belong to the energy comparison.

| Quantity | Acceptance |
|---|---|
| SCF total-energy change | at most 1e-8 Ha |
| SCF density RMS | at most 1e-7 |
| Total energy relative to the independent reference | absolute difference at most 1e-3 Ha per molecule |
| Required validity | finite results, two electrons, successful SCF |

`reference.py` uses PySCF with aug-cc-pV5Z and grid level 7. Its explicit
`LDA_X,LDA_C_PW_MOD` functional matches the PW92 coefficient `0.0310907` in
`crates/mt-dft/src/xc.rs`; plain `LDA` would select a different correlation
functional. See the [PySCF functional syntax](https://pyscf.org/user/dft.html).
It writes a JSON result to its sole command-line argument and fails if its
SCF does not converge.

The independent PySCF 2.14.0 reference at the declared geometry is
**−1.1373018421841654 Ha** (nonrelativistic, isolated, converged).

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
corresponding 12 Bohr / 94³ acceptance input.

For the parameterized Rust experiment and explicit checkpoint output:

```sh
HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo run --release -p libmuffintin-runtime --features fft-fftw --example h2_dft -- \
  /tmp/h2-dft-run 10 6 12 137035.9895 78
```

The arguments are output directory, cubic box side in Bohr, orbital reciprocal
cutoff, density/potential reciprocal cutoff in inverse Bohr, speed of light in
atomic units, and cubic XC integration grid size. The example
uses a 0.65 Bohr muffin-tin radius, 401 radial points, orbital angular cutoff 8,
field angular cutoff 8, and explicit −0.4 Ha linearization energies. It writes
the atomic-start checkpoint and Input V3 before SCF, and the accepted restart
only after SCF converges.

## Current result

The earlier fixed-density `h2_xc_probe` established the relevant XC-grid
sensitivity:

| XC integration grid | XC energy (Ha) | Direct density–XC-potential contraction (Ha) |
|---|---:|---:|
| 39 × 39 × 39 | −0.6487456968912186 | −0.8473727302045191 |
| 78 × 78 × 78 | −0.6478004517964453 | −0.8461302216974157 |

The final bounded comparison keeps the orbital and density cutoffs fixed and
uses independent periodic Gamma-cell PySCF references matching each boundary:

| Cell / XC grid | LAPW energy (Ha) | SCF ΔE (Ha) | Density RMS | Periodic reference (Ha) | Absolute error (mHa) | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 10 Bohr / 78³ | −1.1404078111569622 | 5.48873e-9 | 3.64957e-8 | −1.1391304636363049 | 1.277348 | 341.54 |
| 12 Bohr / 94³ | −1.1394902806948766 | 8.62415e-9 | 1.11770e-8 | −1.1375728843514892 | 1.917396 | 2172.07 |

Both runs converged in seven iterations and pass the SCF thresholds. Neither
passes the fixed 1 mHa physical-energy gate, so the physical acceptance result
is **fail**; no further box, grid, or backend sweep is part of this check.

The initial 12 Bohr path had exposed two separate negative-density defects.
Whole-field positive normalization replaced the additive zero-mode charge
shift, and the atomic guess now sums complete autocorrelations of independent
band-limited square-root amplitudes. For a mathematically positive density at
an FFT node, only a negative eigenvalue within the coefficient-scale bound for
the three-transform FFT chain is projected to zero; larger negatives, including
the earlier −2.10694e-7 Bohr⁻³ defect, remain hard errors. Missing reciprocal
difference support is also still an error.

## Execution cost

RSTSR/TBLIS and Faer use the existing Rayon pool; `RAYON_NUM_THREADS=10` makes
the machine's 10-thread setting explicit. With `fft-fftw`, the interstitial
physical metric is evaluated as an alias-free zero-padded Fourier correlation,
reducing its reciprocal-space work from quadratic to FFT complexity. The
non-FFTW build retains a row-parallel direct contraction. Exactly identical
up/down scalar inputs also share the basis, eigensolve, and density contraction;
distinct spin channels retain their independent path. Unused extended-core/XC
spherical averaging is skipped for the all-explicit, no-core H2 route.

For 10 Bohr / 78³ these changes reduce wall time from 1163.78 s to 341.54 s,
a 3.41× speedup, while changing the converged energy by only about 1.1e-11 Ha.
The 12 Bohr basis and 94³ grid remain substantially more expensive. Raw final
logs are `results/box10-grid78-fft.log` and `results/box12-grid94-fft.log`.
