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
import hashlib
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
class OccupiedOrbital:
    mo_index: int
    spin_channel: str | None
    energy_hartree: float
    occupation: float
    reported_multiplicity: int


@dataclass
class Result:
    method: str
    energy_hartree: float
    homo_hartree: float
    converged: bool
    matrix_dimension: int
    overlap_condition_number: float
    seconds: float
    pyscf_object: str
    mo_energy_shape: list[int]
    mo_occ_shape: list[int]
    orbital_representation: str
    occupation_semantics: str
    spinor_degeneracy_recording: str
    positive_energy_selection: str
    occupied_orbitals: list[OccupiedOrbital]


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


def run_method(
    name: str,
    factory: Callable[[], object],
    orbital_representation: str,
    args: argparse.Namespace,
) -> Result:
    mf = configure(factory(), args)
    t0 = time.perf_counter()
    energy = float(mf.kernel())
    elapsed = time.perf_counter() - t0

    mo_energies = np.asarray(mf.mo_energy)
    mo_occupations = np.asarray(mf.mo_occ)
    if mo_energies.shape != mo_occupations.shape:
        raise RuntimeError(
            f"{name}: mo_energy shape {mo_energies.shape} differs from "
            f"mo_occ shape {mo_occupations.shape}"
        )

    occupied_orbitals: list[OccupiedOrbital] = []
    if mo_energies.ndim == 1:
        positive_energy_start = (
            mo_energies.size // 2
            if orbital_representation == "four-component spinor"
            else 0
        )
        for mo_index, (orbital_energy, occupation) in enumerate(
            zip(mo_energies, mo_occupations)
        ):
            if mo_index >= positive_energy_start and occupation > 1e-8:
                occupied_orbitals.append(
                    OccupiedOrbital(
                        mo_index=mo_index,
                        spin_channel=None,
                        energy_hartree=float(orbital_energy.real),
                        occupation=float(occupation),
                        reported_multiplicity=1,
                    )
                )
    elif mo_energies.ndim == 2 and mo_energies.shape[0] == 2:
        for channel_index, spin_channel in enumerate(("alpha", "beta")):
            for mo_index, (orbital_energy, occupation) in enumerate(
                zip(mo_energies[channel_index], mo_occupations[channel_index])
            ):
                if occupation > 1e-8:
                    occupied_orbitals.append(
                        OccupiedOrbital(
                            mo_index=mo_index,
                            spin_channel=spin_channel,
                            energy_hartree=float(orbital_energy.real),
                            occupation=float(occupation),
                            reported_multiplicity=1,
                        )
                    )
    else:
        raise RuntimeError(f"{name}: unsupported mo_energy shape {mo_energies.shape}")

    if not occupied_orbitals:
        raise RuntimeError(f"{name}: no occupied electronic orbitals found")

    if orbital_representation == "spatial orbital":
        occupation_semantics = (
            "Each entry is one PySCF spatial MO; occupation is the electron count "
            "stored in that mo_occ entry."
        )
        spinor_degeneracy_recording = (
            "Not applicable: the entries are spatial orbitals, not spinors."
        )
        positive_energy_selection = (
            "All mo_occ entries greater than 1e-8; this method has no "
            "negative-energy Dirac branch."
        )
    elif orbital_representation == "alpha/beta spatial orbital":
        occupation_semantics = (
            "Each entry is one PySCF alpha or beta spatial MO; occupation is "
            "stored separately in the corresponding mo_occ row."
        )
        spinor_degeneracy_recording = (
            "Not applicable: the entries are alpha/beta spatial orbitals, not spinors."
        )
        positive_energy_selection = (
            "All mo_occ entries greater than 1e-8; this method has no "
            "negative-energy Dirac branch."
        )
    else:
        occupation_semantics = (
            "Each entry is one actual PySCF spinor MO and its stored mo_occ value; "
            "no occupation is multiplied by a degeneracy."
        )
        spinor_degeneracy_recording = (
            "Kramers partners remain separate PySCF mo_energy/mo_occ entries; "
            "every reported entry has multiplicity 1 and no pair is collapsed."
        )
        if orbital_representation == "four-component spinor":
            positive_energy_selection = (
                "Occupied entries from the PySCF positive-energy solution half: "
                "mo_index >= len(mo_energy) // 2 and mo_occ > 1e-8."
            )
        else:
            positive_energy_selection = (
                "All mo_occ entries greater than 1e-8; the X2C MO array contains "
                "only the electronic branch."
            )

    overlap = np.asarray(mf.get_ovlp())
    overlap_condition_number = float(np.linalg.cond(overlap))
    if not math.isfinite(overlap_condition_number) or overlap_condition_number <= 0:
        raise RuntimeError(
            f"{name}: invalid overlap condition number {overlap_condition_number}"
        )
    matrix_dimension = int(overlap.shape[0])
    return Result(
        method=name,
        energy_hartree=energy,
        homo_hartree=max(orbital.energy_hartree for orbital in occupied_orbitals),
        converged=bool(mf.converged),
        matrix_dimension=matrix_dimension,
        overlap_condition_number=overlap_condition_number,
        seconds=elapsed,
        pyscf_object=f"{type(mf).__module__}.{type(mf).__qualname__}",
        mo_energy_shape=list(mo_energies.shape),
        mo_occ_shape=list(mo_occupations.shape),
        orbital_representation=orbital_representation,
        occupation_semantics=occupation_semantics,
        spinor_degeneracy_recording=spinor_degeneracy_recording,
        positive_energy_selection=positive_energy_selection,
        occupied_orbitals=occupied_orbitals,
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


def write_csv(path: Path, results: list[Result], provenance: dict[str, object]) -> None:
    software = provenance["software"]
    basis = provenance["basis"]
    calculation = provenance["calculation"]
    fieldnames = [
        "python_version",
        "pyscf_version",
        "basis_set_exchange_version",
        "numpy_version",
        "scipy_version",
        "basis_name",
        "basis_normalized_sha256",
        "nuclear_model",
        "light_speed_au",
        "conv_tol",
        "max_cycle",
        "threads",
        "gaunt",
        "breit",
        "method",
        "energy_hartree",
        "homo_hartree",
        "converged",
        "matrix_dimension",
        "overlap_condition_number",
        "seconds",
        "pyscf_object",
        "mo_energy_shape",
        "mo_occ_shape",
        "orbital_representation",
        "occupation_semantics",
        "spinor_degeneracy_recording",
        "positive_energy_selection",
        "mo_index",
        "spin_channel",
        "orbital_energy_hartree",
        "occupation",
        "reported_multiplicity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for orbital in result.occupied_orbitals:
                writer.writerow(
                    {
                        "python_version": software["python"],
                        "pyscf_version": software["pyscf"],
                        "basis_set_exchange_version": software[
                            "basis_set_exchange"
                        ],
                        "numpy_version": software["numpy"],
                        "scipy_version": software["scipy"],
                        "basis_name": basis["name"],
                        "basis_normalized_sha256": basis["normalized_sha256"],
                        "nuclear_model": calculation["nuclear_model"],
                        "light_speed_au": calculation["light_speed_au"],
                        "conv_tol": calculation["conv_tol"],
                        "max_cycle": calculation["max_cycle"],
                        "threads": calculation["threads"],
                        "gaunt": calculation["gaunt"],
                        "breit": calculation["breit"],
                        "method": result.method,
                        "energy_hartree": result.energy_hartree,
                        "homo_hartree": result.homo_hartree,
                        "converged": result.converged,
                        "matrix_dimension": result.matrix_dimension,
                        "overlap_condition_number": (
                            result.overlap_condition_number
                        ),
                        "seconds": result.seconds,
                        "pyscf_object": result.pyscf_object,
                        "mo_energy_shape": json.dumps(result.mo_energy_shape),
                        "mo_occ_shape": json.dumps(result.mo_occ_shape),
                        "orbital_representation": result.orbital_representation,
                        "occupation_semantics": result.occupation_semantics,
                        "spinor_degeneracy_recording": (
                            result.spinor_degeneracy_recording
                        ),
                        "positive_energy_selection": result.positive_energy_selection,
                        "mo_index": orbital.mo_index,
                        "spin_channel": orbital.spin_channel,
                        "orbital_energy_hartree": orbital.energy_hartree,
                        "occupation": orbital.occupation,
                        "reported_multiplicity": orbital.reported_multiplicity,
                    }
                )


def main() -> int:
    args = parse_args()
    try:
        import basis_set_exchange
        import pyscf
        import scipy
        from pyscf import lib, scf
    except ImportError as exc:
        raise SystemExit(
            'PySCF is required. Install it with: python -m pip install "pyscf[bse]>=2.8"'
        ) from exc

    lib.num_threads(args.threads)
    mol = make_molecule(args)
    scalar_factory = scf.RHF if args.spin == 0 else scf.UHF
    scalar_representation = (
        "spatial orbital" if args.spin == 0 else "alpha/beta spatial orbital"
    )

    jobs: list[tuple[str, Callable[[], object], str]] = []
    if not args.skip_nr:
        jobs.append(("NR-HF", lambda: scalar_factory(mol), scalar_representation))
    if not args.skip_scalar:
        jobs.append(
            (
                "sf-X2C1e-HF",
                lambda: scalar_factory(mol).sfx2c1e(),
                scalar_representation,
            )
        )
    jobs.append(("X2C1e-HF", lambda: scf.X2C(mol), "two-component spinor"))

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
    jobs.append((four_component_method, make_dhf, "four-component spinor"))

    results: list[Result] = []
    for name, factory, orbital_representation in jobs:
        print(f"\n=== {name} ===")
        result = run_method(name, factory, orbital_representation, args)
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

    normalized_basis = json.dumps(
        mol._basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    provenance = {
        "software": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "python_executable": sys.executable,
            "pyscf": pyscf.__version__,
            "basis_set_exchange": basis_set_exchange.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "basis": {
            "name": args.basis,
            "normalized_format": (
                "PySCF Mole._basis serialized as UTF-8 JSON with "
                "sort_keys=True and separators=(',', ':')"
            ),
            "normalized_sha256": hashlib.sha256(
                normalized_basis.encode("utf-8")
            ).hexdigest(),
            "normalized_content": json.loads(normalized_basis),
        },
        "calculation": {
            "nuclear_model": "point" if args.point_nucleus else "gaussian",
            "light_speed_au": float(lib.param.LIGHT_SPEED),
            "conv_tol": args.conv_tol,
            "max_cycle": args.max_cycle,
            "threads": int(lib.num_threads()),
            "gaunt": args.gaunt,
            "breit": args.breit,
        },
    }

    report = {
        "provenance": provenance,
        "input": {
            "atom": args.atom,
            "charge": args.charge,
            "spin": args.spin,
            "basis": args.basis,
            "nuclear_model": "point" if args.point_nucleus else "gaussian",
            "light_speed_au": float(lib.param.LIGHT_SPEED),
            "conv_tol": args.conv_tol,
            "max_cycle": args.max_cycle,
            "threads": int(lib.num_threads()),
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
        write_csv(args.csv, results, provenance)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not all(r.converged and math.isfinite(r.energy_hartree) for r in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
