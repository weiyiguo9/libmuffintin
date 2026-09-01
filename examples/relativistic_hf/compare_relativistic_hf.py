#!/usr/bin/env python3
"""Toy relativistic Hartree--Fock comparison with PySCF.

The four calculations use the same all-electron Gaussian basis and nuclear
model:

  NR-HF          nonrelativistic HF
  sf-X2C1e-HF    scalar-relativistic one-electron X2C HF
  X2C1e-HF       spin-dependent two-component one-electron X2C HF
  4c-DC-HF       four-component Dirac--Coulomb HF

This is a molecular/atomic quantum-chemistry comparison.  X2C1e is not the
same approximation as Koelling--Harmon plus first-variation SOC, and 4c-DC-HF
is not the same numerical representation as FRA LAPW augmentation.  The useful
diagnostic is nevertheless the positive-energy HF difference between the 2c
and 4c descriptions on an identical basis.

Example
-------
    python -m pip install "pyscf[bse]>=2.8"
    python compare_relativistic_hf.py --atom Xe --basis dyall-v2z

For a faster smoke test:
    python compare_relativistic_hf.py --atom Kr --basis dyall-v2z --skip-nr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HARTREE_TO_EV = 27.211386245981
HARTREE_TO_MEV = 1000.0 * HARTREE_TO_EV


@dataclass
class Result:
    method: str
    energy_hartree: float
    homo_hartree: float
    converged: bool
    matrix_dimension: int
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NR, scalar-X2C, 2c-X2C, and 4c Dirac-Coulomb HF."
    )
    parser.add_argument("--atom", default="Xe", help="Atomic symbol (default: Xe)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="PySCF spin = N_alpha - N_beta = 2S (default: 0)",
    )
    parser.add_argument("--basis", default="dyall-v2z")
    parser.add_argument(
        "--point-nucleus",
        action="store_true",
        help="Use a point nucleus instead of the default Gaussian nucleus.",
    )
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--max-memory", type=int, default=4000, help="MB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--skip-nr", action="store_true")
    parser.add_argument("--skip-scalar", action="store_true")
    interaction_group = parser.add_mutually_exclusive_group()
    interaction_group.add_argument(
        "--gaunt",
        action="store_true",
        help="Add the Gaunt term to 4c DHF (off for a clean Dirac-Coulomb comparison).",
    )
    interaction_group.add_argument(
        "--breit",
        action="store_true",
        help="Add the Breit interaction to 4c DHF; implies a different Hamiltonian.",
    )
    parser.add_argument("--csv", type=Path, help="Optional raw-result CSV path")
    parser.add_argument("--json", type=Path, help="Optional report JSON path")
    return parser.parse_args()


def make_molecule(args: argparse.Namespace):
    try:
        from pyscf import gto
    except ImportError as exc:
        raise SystemExit(
            'PySCF is required. Install it with: python -m pip install "pyscf[bse]>=2.8"'
        ) from exc

    # Dyall bases are already designed for relativistic all-electron work.
    # Prefix a conventional contracted basis with "unc-" if decontraction is
    # desired, e.g. --basis unc-ano-rcc.
    return gto.M(
        atom=f"{args.atom} 0 0 0",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        symmetry=False,
        nucmod=0 if args.point_nucleus else "gaussian",
        max_memory=args.max_memory,
        verbose=args.verbose,
    )


def configure(mf, args: argparse.Namespace):
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.chkfile = None
    return mf


def run_method(name: str, factory: Callable[[], object], args: argparse.Namespace) -> Result:
    mf = configure(factory(), args)
    t0 = time.perf_counter()
    energy = float(mf.kernel())
    elapsed = time.perf_counter() - t0
    mo_energies = np.asarray(mf.mo_energy).reshape(-1)
    mo_occupations = np.asarray(mf.mo_occ).reshape(-1)
    occ_energies = [
        float(e.real) for e, occ in zip(mo_energies, mo_occupations) if occ > 1e-8
    ]
    if not occ_energies:
        raise RuntimeError(f"{name}: no occupied electronic orbitals found")
    matrix_dimension = int(mf.get_ovlp().shape[0])
    return Result(
        method=name,
        energy_hartree=energy,
        homo_hartree=max(occ_energies),
        converged=bool(mf.converged),
        matrix_dimension=matrix_dimension,
        seconds=elapsed,
    )


def delta(a: Result, b: Result) -> dict[str, float]:
    """Return a - b in units useful for total and orbital energies."""
    de = a.energy_hartree - b.energy_hartree
    dh = a.homo_hartree - b.homo_hartree
    return {
        "energy_hartree": de,
        "energy_millihartree": 1000.0 * de,
        "energy_ev": HARTREE_TO_EV * de,
        "homo_ev": HARTREE_TO_EV * dh,
        "homo_mev": HARTREE_TO_MEV * dh,
    }


def print_results(results: list[Result], deltas: dict[str, dict[str, float]]) -> None:
    print("\nResults")
    print(
        f"{'method':<13} {'E / Eh':>22} {'HOMO / Eh':>16} "
        f"{'dim':>7} {'conv':>6} {'time / s':>11}"
    )
    for r in results:
        print(
            f"{r.method:<13} {r.energy_hartree:>22.12f} {r.homo_hartree:>16.9f} "
            f"{r.matrix_dimension:>7d} {str(r.converged):>6} {r.seconds:>11.2f}"
        )

    print("\nDifferences (left minus right)")
    print(f"{'comparison':<25} {'dE / mEh':>14} {'dE / eV':>14} {'dHOMO / meV':>16}")
    for label, d in deltas.items():
        print(
            f"{label:<25} {d['energy_millihartree']:>14.6f} "
            f"{d['energy_ev']:>14.6f} {d['homo_mev']:>16.3f}"
        )


def write_csv(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def main() -> int:
    args = parse_args()
    try:
        from pyscf import lib, scf
    except ImportError as exc:
        raise SystemExit(
            'PySCF is required. Install it with: python -m pip install "pyscf[bse]>=2.8"'
        ) from exc

    lib.num_threads(args.threads)
    mol = make_molecule(args)
    scalar_factory = scf.RHF if args.spin == 0 else scf.UHF

    jobs: list[tuple[str, Callable[[], object]]] = []
    if not args.skip_nr:
        jobs.append(("NR-HF", lambda: scalar_factory(mol)))
    if not args.skip_scalar:
        jobs.append(("sf-X2C1e-HF", lambda: scalar_factory(mol).sfx2c1e()))
    jobs.append(("X2C1e-HF", lambda: scf.X2C(mol)))

    def make_dhf():
        mf = scf.DHF(mol)
        mf.with_ssss = True
        mf.with_gaunt = args.gaunt and not args.breit
        mf.with_breit = args.breit
        return mf

    if args.breit:
        four_component_label = "4c-DCB"
        four_component_hamiltonian = "Dirac-Coulomb-Breit"
    elif args.gaunt:
        four_component_label = "4c-DCG"
        four_component_hamiltonian = "Dirac-Coulomb-Gaunt"
    else:
        four_component_label = "4c-DC"
        four_component_hamiltonian = "Dirac-Coulomb"
    four_component_method = f"{four_component_label}-HF"
    jobs.append((four_component_method, make_dhf))

    results: list[Result] = []
    for name, factory in jobs:
        print(f"\n=== {name} ===")
        result = run_method(name, factory, args)
        results.append(result)
        if not result.converged:
            print(f"WARNING: {name} did not converge", file=sys.stderr)

    by_name = {r.method: r for r in results}
    deltas: dict[str, dict[str, float]] = {}
    if "NR-HF" in by_name and "sf-X2C1e-HF" in by_name:
        deltas["sf-X2C1e - NR"] = delta(by_name["sf-X2C1e-HF"], by_name["NR-HF"])
    if "sf-X2C1e-HF" in by_name:
        deltas["X2C1e - sf-X2C1e"] = delta(
            by_name["X2C1e-HF"], by_name["sf-X2C1e-HF"]
        )
    four_component_delta = f"{four_component_label} - X2C1e"
    deltas[four_component_delta] = delta(
        by_name[four_component_method], by_name["X2C1e-HF"]
    )
    if "NR-HF" in by_name:
        deltas[f"{four_component_label} - NR"] = delta(
            by_name[four_component_method], by_name["NR-HF"]
        )

    print_results(results, deltas)

    report = {
        "input": {
            "atom": args.atom,
            "charge": args.charge,
            "spin": args.spin,
            "basis": args.basis,
            "nuclear_model": "point" if args.point_nucleus else "gaussian",
            "gaunt": args.gaunt,
            "breit": args.breit,
        },
        "results": [asdict(r) for r in results],
        "deltas": deltas,
        "interpretation": {
            four_component_delta: (
                f"Residual between full {four_component_hamiltonian} HF and "
                "one-electron X2C HF; "
                "it includes missing two-electron picture-change effects in X2C1e."
            ),
            "warning": (
                "This is not a direct Koelling-Harmon-versus-FRA LAPW benchmark."
            ),
        },
    }
    if args.csv:
        write_csv(args.csv, results)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not all(r.converged and math.isfinite(r.energy_hartree) for r in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
