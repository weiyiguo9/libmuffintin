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
