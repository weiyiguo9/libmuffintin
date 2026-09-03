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
an $8$ bohr cubic cell, derives the 28 core and 8 valence electrons from
`fleur_default_atomic_configuration`, and runs the public Gamma relaxed-core HF
path with explicit finite-body exchange:

```sh
cargo run -p libmuffintin-runtime --example kr_relaxed_core_hf -- \
  --out kr-relaxed-core-hf-p0 \
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
in Hartree. The outer loop mixes only 0.1 of the fresh valence density; the core
density is still replaced without mixing. Completion demonstrates that the
production pipeline executed; it is not a claim of physical convergence.
`--field-g 4.5` sets the independent
atomic-start regional-field cutoff in inverse bohr, while `--orbital-g 1`
remains the orbital-basis cutoff. The field angular cutoff is
`2 * (orbital_lmax + 1)` so the Dirac small-component products retain their
complete channel layout. The 2401-point muffin-tin mesh resolves the separate
full-space and muffin-tin core norm quadratures used by the spill gate.

The output directory contains:

- `manifest.toml`: units, complete input and derived parameters, the FLEUR
  core-state split, and Git SHA/dirty provenance;
- `initial-checkpoint.toml`: the canonical atomic-superposition restart written
  before HF;
- `iterations.toml`: every completed relaxed-core HF iteration diagnostic;
- `result.toml`: total and exchange-sector energies, four sector traces,
  core/valence diagnostics, all orbital energies and occupations, and core-shell
  energies, norms, and spill;
- `final-checkpoint.toml`: the canonical restart built from the final total
  density and potential.

`--hdlo all` adds one derivative-order-2 atomic HDLO request per orbital $l$.
The harness does not assign an AO matching tolerance, HOMO or vacuum reference,
or a pass/fail acceptance threshold.
