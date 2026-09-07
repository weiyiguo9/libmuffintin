"""Gate an LAPW H2 run against the matched PySCF references.

Arguments: LAPW run log (from `h2_dft` or `muffintin`), matched periodic
PySCF JSON, optional isolated PySCF JSON. Exit status is nonzero when a
gate fails.
"""

import json
import math
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
HF_IDENTITY_GATE = 1.0e-8
HF_HARTREE_EXCHANGE_GATE = 5.0e-4
HF_PERIODIC_GATES = {
    "electron_hartree": 5.0e-4,
    "total": 2.0e-3,
    "homo": 1.0e-3,
}
HF_ISOLATED_EXCHANGE_GATE = 1.0e-3


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


def dft_main():
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


def parse_hf_record(text, prefix, fields):
    matches = re.findall(rf"^{prefix} (.*)$", text, re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {prefix} line")
    values = {
        key: float(value)
        for key, value in re.findall(r"(\w+)=([-+0-9.eE]+)", matches[0])
    }
    if set(values) != set(fields):
        raise SystemExit(f"{prefix} fields must be {', '.join(fields)}")
    return values


def parse_hf(path):
    text = open(path).read()
    terms = parse_hf_record(
        text,
        "hf_energy_terms_ha",
        [
            "h0",
            "electron_hartree",
            "nuclear_hartree",
            "exchange",
            "occupation_correction",
            "band",
            "total",
        ],
    )
    identities = parse_hf_record(
        text,
        "hf_identity_ha",
        ["exchange", "eigenvalue", "total", "hartree_exchange"],
    )
    finals = re.findall(
        r"^hf_final .*electron_count=([-+0-9.eE]+) homo_ha=([-+0-9.eE]+) wall_s=([-+0-9.eE]+)$",
        text,
        re.MULTILINE,
    )
    if len(finals) != 1:
        raise SystemExit("expected exactly one hf_final line")
    electron_count, homo, wall = map(float, finals[0])
    values = list(terms.values()) + list(identities.values()) + [electron_count, homo, wall]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("HF log contains a non-finite result")
    return terms, identities, electron_count, homo


def hf_main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: compare.py --hf identities|a1v LOG | --hf a2|bv LOG REFERENCE_JSON"
        )
    mode = sys.argv[2]
    expected_arguments = 4 if mode in ("identities", "a1v") else 5
    if mode not in ("identities", "a1v", "a2", "bv") or len(sys.argv) != expected_arguments:
        raise SystemExit(
            "usage: compare.py --hf identities|a1v LOG | --hf a2|bv LOG REFERENCE_JSON"
        )
    terms, identities, electron_count, homo = parse_hf(sys.argv[3])
    rows = []
    for name in ("exchange", "eigenvalue", "total"):
        delta = identities[name]
        rows.append((f"{name} identity", delta, 0.0, delta, HF_IDENTITY_GATE))
    if mode == "a1v":
        delta = identities["hartree_exchange"]
        rows.append(("Hartree-exchange identity", delta, 0.0, delta, HF_HARTREE_EXCHANGE_GATE))
    elif mode == "a2":
        reference = json.load(open(sys.argv[4]))
        rows.extend(
            [
                (
                    "E_H vs periodic PySCF",
                    terms["electron_hartree"],
                    reference["e_hartree"],
                    terms["electron_hartree"] - reference["e_hartree"],
                    HF_PERIODIC_GATES["electron_hartree"],
                ),
                (
                    "total vs periodic PySCF",
                    terms["total"],
                    reference["energy_hartree"],
                    terms["total"] - reference["energy_hartree"],
                    HF_PERIODIC_GATES["total"],
                ),
                (
                    "HOMO vs periodic PySCF",
                    homo,
                    reference["homo_hartree"],
                    homo - reference["homo_hartree"],
                    HF_PERIODIC_GATES["homo"],
                ),
            ]
        )
    elif mode == "bv":
        reference = json.load(open(sys.argv[4]))
        rows.append(
            (
                "exchange vs isolated PySCF",
                terms["exchange"],
                reference["e_exchange"],
                terms["exchange"] - reference["e_exchange"],
                HF_ISOLATED_EXCHANGE_GATE,
            )
        )
    failed = abs(electron_count - 2.0) > 1.0e-8
    print(
        f"{'gate':32s} {'value (Ha)':>20s} {'reference (Ha)':>20s} "
        f"{'delta (Ha)':>16s} {'limit (Ha)':>12s} result"
    )
    for name, value, reference, delta, limit in rows:
        passed = abs(delta) <= limit
        failed |= not passed
        print(
            f"{name:32s} {value:20.12f} {reference:20.12f} {delta:16.8e} "
            f"{limit:12.4e} {'pass' if passed else 'FAIL'}"
        )
    print(
        f"electron-count validity: value={electron_count:.16e} "
        f"delta={electron_count - 2.0:.3e} limit=1.0e-8 "
        f"{'pass' if abs(electron_count - 2.0) <= 1.0e-8 else 'FAIL'}"
    )
    raise SystemExit(1 if failed else 0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--hf":
        hf_main()
    dft_main()


if __name__ == "__main__":
    main()
