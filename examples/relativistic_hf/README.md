# Relativistic atomic Hartree–Fock references

`compare_relativistic_hf.py` compares NR-HF, spin-free X2C1e-HF,
spin-dependent X2C1e-HF, and four-component Dirac–Coulomb HF in the same
all-electron Gaussian basis.

The `kr_point_dyall_v2z`, `kr_point_dyall_v3z`, and
`kr_point_dyall_v4z` JSON/CSV pairs use a point nucleus, no Gaunt or Breit
term, `conv_tol=1e-11`, `max_cycle=200`, and one PySCF thread. The JSON is the
authoritative self-contained report: it embeds the normalized PySCF basis
content and its SHA-256, software versions, calculation settings, method
summaries, deltas, and every occupied electronic orbital. The CSV repeats the
same provenance and method summary on one row per occupied orbital.

Spatial RHF entries carry their actual PySCF occupation (two for these
closed-shell calculations). X2C and four-component entries are the actual
PySCF spinor-array entries with occupation one. Kramers partners are recorded
separately with their original MO indices; the files do not collapse a pair
into a degeneracy-weighted row. For four-component DHF, “positive energy”
means the electronic solution half used by PySCF rather than a positive-valued
bound-state orbital energy.

## Krypton point-nucleus run summary

| Basis | 4c overlap condition number | 4c method time (s) | Process maximum RSS (MiB) |
| --- | ---: | ---: | ---: |
| Dyall v2z | 9.022949230311605e7 | 38.34 | 224.5 |
| Dyall v3z | 1.1349864190257856e10 | 169.10 | 377.5 |
| Dyall v4z | 1.6596900070332846e11 | 556.84 | 1162.5 |

The condition numbers and method times are the `4c-DC-HF` fields in the JSON
reports. Maximum RSS is not a JSON field: it is the maximum resident set size
for each complete script process, measured externally by macOS
`/usr/bin/time -lp` and converted from bytes to MiB.

## libmuffintin Kr molecule-in-box smoke harness

The runtime example constructs a neutral point-nucleus Kr atom at the center of
an $8$ bohr cubic cell and derives the 28 core and 8 valence electrons from
`fleur_default_atomic_configuration`. Its default `spinor-first` route runs
Gamma relaxed-core HF with explicit finite-body exchange:

```sh
cargo run -p libmuffintin-runtime --example kr_relaxed_core_hf -- \
  --out kr-relaxed-core-hf-p0 \
  --relativity spinor-first \
  --box 8 --orbital-g 1 --field-g 4.5 --orbital-lmax 1 \
  --product-g 1 --product-lmax 2 --lexp 2 \
  --rmt 2 --radial-points 2401 --hdlo none
```

These defaults are the deliberately loose P0 smoke profile: no HDLOs,
$T=0.02$ Hartree, at most two outer and core steps, and at most 32 Fock
iterations. The fixed-potential Fock loop uses global-basis commutator CDIIS
with history eight. `--fock-mixing quasi-newton-cdiis` selects the optional
diagonal orbital-energy preconditioned variant; `--fock-diis-history` changes
the history and `--fock-diis-level-shift` changes its nonnegative level shift
in Hartree. The outer loop mixes the complete total regional density. Relaxed
core orbitals are still solved freshly at every outer step; their fresh density
is subtracted from the mixed total density to recover the next valence input.
Completion demonstrates that the production pipeline executed; it is not a
claim of physical convergence.
`--field-g 4.5` sets the independent
atomic-start regional-field cutoff in inverse bohr, while `--orbital-g 1`
remains the orbital-basis cutoff. The field angular cutoff is
`2 * (orbital_lmax + 1)` so the Dirac small-component products retain their
complete channel layout. The 2401-point muffin-tin mesh resolves the separate
full-space and muffin-tin core norm quadratures used by the spill gate.

`--relativity kh-soc` selects scalar Koelling–Harmon HF followed by SOC second
variation. This route freezes the 28 core electrons in the immutable atomic
checkpoint potential, keeps their density in the total Hartree/mixing state,
adds exact spherical core exchange to scalar Fock iterations, and adds the
SOC-resolved core operator before second-variation diagonalization. It does not
put CV, VC, or CC products into the VV MPB. Scalar KH accepts `linear` or
`pulay` Fock mixing; `--soc-bands` is the explicit number of lowest scalar
source bands retained:

```sh
cargo run --release -p libmuffintin-runtime --example kr_relaxed_core_hf -- \
  --out kr-kh-soc-hf-production \
  --relativity kh-soc --soc-bands 48 \
  --box 8 --orbital-g 2 --field-g 4.5 --orbital-lmax 4 \
  --product-g 2 --product-lmax 4 --lexp 4 \
  --rmt 2 --radial-points 2401 --hdlo all --temperature 1e-6 \
  --outer-mixing pulay --outer-mixing-alpha 0.1 --outer-mixing-history 8 \
  --outer-max-iterations 64 --outer-energy-tolerance 1e-7 \
  --outer-density-tolerance 1e-6 \
  --fock-max-iterations 128 --fock-mixing pulay \
  --fock-mixing-alpha 0.5 --fock-diis-history 8
```

The KH plus SOC `result.toml` records raw energies, the explicit HOMO shift,
HOMO-shifted Hartree/eV energies, adjacent Kramers-pair splittings, occupations,
and spin-resolved scalar-source-band mixing weights. This is the table intended
for the X2C1e/4c-DC comparison; it does not by itself establish box or basis
convergence. For closed-shell Kr, the HOMO is the highest state with occupation
at least 0.5; this excludes finite-temperature occupation tails in the loose P0
profile.

For a bounded production-convergence attempt, set the outer and core gates
explicitly together with the larger orbital and product bases:

```sh
cargo run --release -p libmuffintin-runtime --example kr_relaxed_core_hf -- \
  --out kr-relaxed-core-hf-production \
  --relativity spinor-first \
  --box 8 --orbital-g 2 --field-g 4.5 --orbital-lmax 4 \
  --product-g 2 --product-lmax 4 --lexp 4 \
  --rmt 2 --radial-points 2401 --hdlo all --temperature 1e-6 \
  --outer-mixing pulay --outer-mixing-alpha 0.1 --outer-mixing-history 8 \
  --outer-max-iterations 64 --outer-energy-tolerance 1e-7 \
  --outer-density-tolerance 1e-6 \
  --core-max-iterations 32 --core-energy-tolerance 1e-8 \
  --core-radial-tolerance 1e-8 \
  --fock-max-iterations 128 --fock-mixing cdiis --fock-diis-history 8
```

The program returns a result only after the configured energy, total-density,
inner-Fock, and core gates pass. Output status
`configured_convergence_reached` therefore means the recorded controls passed;
whether those controls are physically adequate remains part of the basis, box,
and tolerance convergence study.

The output directory contains:

- `manifest.toml`: units, complete input and derived parameters, the FLEUR
  core-state split, and Git SHA/dirty provenance;
- `initial-checkpoint.toml`: the canonical atomic-superposition restart written
  before HF;
- `iterations.toml`: every completed outer/Fock iteration diagnostic for the
  selected route;
- `result.toml`: route-specific total/exchange terms, core/valence diagnostics,
  orbital energies and occupations, and core-shell energies, norms, and spill;
- `final-checkpoint.toml`: the canonical restart built from the final total
  density and potential.

`--hdlo all` adds one derivative-order-2 atomic HDLO request per orbital $l$.
The harness does not assign an AO matching tolerance or a cross-method pass/fail
acceptance threshold. For the KH plus SOC route, “Fermi shift” is recorded
explicitly as the finite-system HOMO reference, not as a periodic chemical
potential.
