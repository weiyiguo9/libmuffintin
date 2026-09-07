"""Matched Gamma-cell diagnostic: output JSON path and box side in Bohr."""

import json
import sys

import numpy as np
import pyscf
from pyscf import lib
from pyscf.pbc import dft, gto

lib.num_threads(4)
box = float(sys.argv[2])
center = box / 2
cell = gto.Cell()
cell.atom = [("H", (center - 0.7, center, center)), ("H", (center + 0.7, center, center))]
cell.unit = "Bohr"
cell.a = np.eye(3) * box
cell.basis = "aug-cc-pv5z"
cell.spin = 0
cell.charge = 0
cell.precision = 1e-9
cell.verbose = 4
cell.max_memory = 4000
cell.build()
mf = dft.RKS(cell).density_fit(auxbasis="aug-cc-pv5z-jkfit")
mf.xc = "LDA_X,LDA_C_PW_MOD"
mf.grids.level = 7
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-8
mf.max_cycle = 100
energy = mf.kernel()
if not mf.converged:
    raise RuntimeError("Periodic H2 reference SCF did not converge")
result = {
    "pyscf_version": pyscf.__version__,
    "system": "H2",
    "bond_bohr": 1.4,
    "boundary": "3D periodic cubic cell, Gamma only",
    "box_bohr": box,
    "hamiltonian": "nonrelativistic all-electron point nuclei",
    "xc": mf.xc,
    "basis": cell.basis,
    "auxbasis": mf.with_df.auxbasis,
    "grid_level": mf.grids.level,
    "energy_hartree": energy,
    "electron_count": cell.nelectron,
    "converged": mf.converged,
}
with open(sys.argv[1], "w") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
