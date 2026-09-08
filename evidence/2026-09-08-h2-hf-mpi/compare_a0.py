"""Check rank-two A0 preservation and identical rank-final state."""

import math
from pathlib import Path
import sys


lines = Path(sys.argv[1]).read_text().splitlines()


def fields(line):
    return dict(field.split("=", 1) for field in line.split() if "=" in field)


final_lines = [line for line in lines if line.startswith("hf_final ")]
assert final_lines, "rank-zero final line missing"
assert len(set(final_lines)) == 1, "hf_final lines differ across printed ranks"
final = fields(final_lines[0])
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
deltas = {key: abs(actual[key] - value) for key, value in references.items()}
for key in references:
    print(f"{key}={actual[key]:.17e} ref={references[key]:.17e} Delta={deltas[key]:.17e}")
identities = [float(row[key]) for row in diagnostics for key in (
    "exchange_identity_ha", "eigenvalue_identity_ha", "total_identity_ha"
)]
count_error = abs(float(final["electron_count"]) - 2.0)
passed = max(deltas.values()) <= 1e-10 and max(identities) <= 1e-8 and count_error <= 1e-8
print(f"identity_max={max(identities):.17e} bound=1e-8 electron_count_error={count_error:.17e}")
print(f"hf_final_unique={len(set(final_lines))} printed_lines={len(final_lines)} wall_s={final['wall_s']}")
print("DIGIT / PASS" if passed else "DIGIT / HANDOFF")
sys.exit(0 if passed else 1)
