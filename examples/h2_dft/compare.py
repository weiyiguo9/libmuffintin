"""Gate an LAPW H2 run against the matched PySCF references.

Arguments: LAPW run log (from `h2_dft` or `muffintin`), matched periodic
PySCF JSON, optional isolated PySCF JSON. Exit status is nonzero when a
gate fails.
"""

import json
import re
import sys

HARTREE_TO_MILLI = 1.0e3
GATES = {
    # Cancellation-friendly: the periodic eigenvalue shares the box, the
    # G = 0 electrostatic convention, and most basis-truncation error.
    "homo_mha": 1.0,
    # Absolute total energy between an LAPW box and a Gaussian GDF box.
    "total_mha": 20.0,
}


def parse_terms(path):
    text = open(path).read()
    terms = {}
    match = re.search(r"energy_terms_ha (.*)", text)
    if match:
        for key, value in re.findall(r"(\w+)=([-+0-9.eE]+)", match.group(1)):
            terms[key] = float(value)
    final = re.search(r"(scf_final|task \S+ scf) .*total_energy_ha=([-+0-9.eE]+)", text)
    if final:
        terms["total"] = float(final.group(2))
    return terms


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    terms = parse_terms(sys.argv[1])
    periodic = json.load(open(sys.argv[2]))
    isolated = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    if "total" not in terms:
        raise SystemExit("no converged total energy in the LAPW log")
    rows = []
    failed = False
    if "band" in terms:
        # Two electrons in one Fermi-Dirac occupied band at T = 1 mHa: the
        # band energy is twice the HOMO eigenvalue to far below 1e-10 Ha.
        homo = 0.5 * terms["band"]
        delta = HARTREE_TO_MILLI * (homo - periodic["homo_hartree"])
        ok = abs(delta) <= GATES["homo_mha"]
        failed |= not ok
        rows.append(("HOMO vs periodic PySCF", homo, periodic["homo_hartree"], delta, GATES["homo_mha"], ok))
    delta = HARTREE_TO_MILLI * (terms["total"] - periodic["energy_hartree"])
    ok = abs(delta) <= GATES["total_mha"]
    failed |= not ok
    rows.append(("total vs periodic PySCF", terms["total"], periodic["energy_hartree"], delta, GATES["total_mha"], ok))
    diagnostics = []
    if "exchange_correlation" in terms:
        exc = periodic["components_hartree"]["exc"]
        diagnostics.append(("E_xc vs periodic PySCF", terms["exchange_correlation"], exc, HARTREE_TO_MILLI * (terms["exchange_correlation"] - exc)))
        diagnostics.append((
            "E - E_xc vs periodic PySCF",
            terms["total"] - terms["exchange_correlation"],
            periodic["energy_hartree"] - exc,
            HARTREE_TO_MILLI * ((terms["total"] - terms["exchange_correlation"]) - (periodic["energy_hartree"] - exc)),
        ))
    if isolated is not None:
        diagnostics.append(("total vs isolated PySCF", terms["total"], isolated["energy_hartree"], HARTREE_TO_MILLI * (terms["total"] - isolated["energy_hartree"])))
        diagnostics.append(("periodic minus isolated PySCF", periodic["energy_hartree"], isolated["energy_hartree"], HARTREE_TO_MILLI * (periodic["energy_hartree"] - isolated["energy_hartree"])))
    print(f"{'gate':32s} {'LAPW (Ha)':>20s} {'reference (Ha)':>20s} {'delta (mHa)':>12s} {'limit':>6s} result")
    for name, lapw, reference, delta, limit, ok in rows:
        print(f"{name:32s} {lapw:20.12f} {reference:20.12f} {delta:12.4f} {limit:6.1f} {'pass' if ok else 'FAIL'}")
    print(f"{'diagnostic':32s} {'LAPW (Ha)':>20s} {'reference (Ha)':>20s} {'delta (mHa)':>12s}")
    for name, lapw, reference, delta in diagnostics:
        print(f"{name:32s} {lapw:20.12f} {reference:20.12f} {delta:12.4f}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
