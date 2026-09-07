"""Independent nonrelativistic H2 LDA-PW92 molecular reference (Hartree/Bohr)."""

import json
import sys

import pyscf
from pyscf import dft, gto, lib

lib.num_threads(4)
mol = gto.M(
    atom="H 0 0 -0.7; H 0 0 0.7",
    unit="Bohr",
    basis="aug-cc-pv5z",
    charge=0,
    spin=0,
    verbose=4,
)
mf = dft.RKS(mol)
# Match mt-dft's PW92 coefficient A=0.0310907 (Libxc PW_MOD).
mf.xc = "LDA_X,LDA_C_PW_MOD"
mf.grids.level = 7
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-8
mf.max_cycle = 100
energy = mf.kernel()
if not mf.converged:
    raise RuntimeError("H2 reference SCF did not converge")
result = {
    "pyscf_version": pyscf.__version__,
    "system": "H2",
    "bond_bohr": 1.4,
    "boundary": "isolated molecule",
    "hamiltonian": "nonrelativistic all-electron point nuclei",
    "xc": mf.xc,
    "basis": mol.basis,
    "grid_level": mf.grids.level,
    "energy_hartree": energy,
    "electron_count": mol.nelectron,
    "converged": mf.converged,
}
with open(sys.argv[1], "w") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
