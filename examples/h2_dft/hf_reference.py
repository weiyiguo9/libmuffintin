"""Independent H2 RHF reference: OUTPUT_JSON [BOX_BOHR]."""

import json
import sys

import numpy as np
import pyscf
from pyscf import gto, lib, scf

lib.num_threads(4)


def isolated():
    mol = gto.M(
        atom="H 0 0 -0.7; H 0 0 0.7",
        unit="Bohr",
        basis="aug-cc-pv5z",
        charge=0,
        spin=0,
        verbose=5,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    energy = mf.kernel()
    return mol, mf, energy, {
        "boundary": "isolated molecule",
    }


def periodic(box):
    from pyscf.pbc import gto as pbc_gto
    from pyscf.pbc import scf as pbc_scf

    center = box / 2
    cell = pbc_gto.Cell()
    cell.atom = [
        ("H", (center - 0.7, center, center)),
        ("H", (center + 0.7, center, center)),
    ]
    cell.unit = "Bohr"
    cell.a = np.eye(3) * box
    cell.basis = "aug-cc-pv5z"
    cell.spin = 0
    cell.charge = 0
    cell.precision = 1e-9
    cell.verbose = 5
    cell.max_memory = 4000
    cell.build()
    mf = pbc_scf.RHF(cell, exxdiv=None).density_fit(auxbasis="aug-cc-pv5z-jkfit")
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    energy = mf.kernel()
    return cell, mf, energy, {
        "boundary": "3D periodic cubic cell, Gamma only",
        "box_bohr": box,
        "auxbasis": mf.with_df.auxbasis,
        "exxdiv": None,
    }


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    if len(sys.argv) == 2:
        system, mf, energy, boundary = isolated()
    else:
        system, mf, energy, boundary = periodic(float(sys.argv[2]))
    if not mf.converged:
        raise RuntimeError("H2 RHF reference SCF did not converge")
    density = mf.make_rdm1()
    vj, vk = mf.get_jk(system, density)
    e_hartree = float((0.5 * np.einsum("ij,ji", vj, density)).real)
    e_exchange = float((-0.25 * np.einsum("ij,ji", vk, density)).real)
    occupied = int(round(system.nelectron / 2))
    result = {
        "pyscf_version": pyscf.__version__,
        "system": "H2",
        "bond_bohr": 1.4,
        **boundary,
        "hamiltonian": "nonrelativistic all-electron point nuclei",
        "method": "RHF",
        "basis": system.basis,
        "energy_hartree": float(energy),
        "homo_hartree": float(mf.mo_energy[occupied - 1]),
        "lumo_hartree": float(mf.mo_energy[occupied]),
        "e_hartree": e_hartree,
        "e_exchange": e_exchange,
        "exchange_ratio": e_exchange / e_hartree,
        "components_hartree": {
            key: float(value.real) for key, value in mf.scf_summary.items()
        },
        "electron_count": system.nelectron,
        "converged": mf.converged,
    }
    with open(sys.argv[1], "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
