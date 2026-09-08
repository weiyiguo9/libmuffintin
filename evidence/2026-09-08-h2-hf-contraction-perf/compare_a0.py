"""Check the prescribed A0 preservation quantities against evd-0010."""

import math
from pathlib import Path
import sys


lines = Path(sys.argv[1]).read_text().splitlines()


def fields(line):
    return dict(field.split("=", 1) for field in line.split() if "=" in field)


final = fields(next(line for line in lines if line.startswith("hf_final ")))
energy = fields(next(line for line in lines if line.startswith("hf_energy_terms_ha ")))
diagnostics = [fields(line) for line in lines if line.startswith("hf_iteration=")]
references = {
    "E": -0.59291354371542793,
    "HOMO": -0.12617886551775270,
    "E_H": 0.37293092552423746,
    "E_x": -0.023151590595997563,
}
actual = dict(zip(references, map(float, (
    energy["total"], final["homo_ha"], energy["electron_hartree"], energy["exchange"]
))))
assert all(math.isfinite(value) for value in actual.values())
differences = {key: abs(actual[key] - reference) for key, reference in references.items()}
for key in references:
    print(f"{key}={actual[key]:.17e} reference={references[key]:.17e} Delta={differences[key]:.17e}")
identities = [float(row[key]) for row in diagnostics for key in (
    "exchange_identity_ha", "eigenvalue_identity_ha", "total_identity_ha"
)]
assert identities and all(math.isfinite(value) for value in identities)
count_error = abs(float(final["electron_count"]) - 2.0)
passed = max(differences.values()) <= 1e-10 and max(identities) <= 1e-8 and count_error <= 1e-8
print(f"class=R ref=evd-0010 energy_bound=1e-10 identity_max={max(identities):.17e} identity_bound=1e-8 electron_count_error={count_error:.17e}")
print(f"wall_s={final['wall_s']} outer_iterations={final['outer_iterations']} fock_iterations_total={sum(int(row['fock_iterations']) for row in diagnostics)}")
print("DIGIT / PASS" if passed else "DIGIT / HANDOFF")
sys.exit(0 if passed else 1)
