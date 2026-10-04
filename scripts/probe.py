import time

import numpy as np
import pennylane as qml
from pennylane import qchem

for name, symbols, coords, kw in [
    ("H2", ["H","H"], np.array([0,0,0, 0,0,0.735]), {}),
    ("LiH", ["Li","H"], np.array([0,0,0, 0,0,1.6]), dict(active_electrons=2, active_orbitals=4)),
    ("H4", ["H"]*4, np.array([0,0,0, 0,0,1.0, 0,0,2.0, 0,0,3.0]), dict(active_electrons=4, active_orbitals=4)),
]:
    t=time.time()
    mol = qchem.Molecule(symbols, coords, unit="angstrom")
    H, n = qchem.molecular_hamiltonian(mol, **kw)
    M = qml.matrix(H, wire_order=range(n)).real
    e = np.linalg.eigvalsh(M)[0]
    ne = kw.get("active_electrons", mol.n_electrons)
    s, d = qchem.excitations(ne, n)
    print(name, n, "qubits", "E0", e, "singles", len(s), "doubles", len(d), f"{time.time()-t:.1f}s")
