"""Molecular problems built with PennyLane qchem (STO-3G, active spaces)."""
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
import scipy.sparse as sp
from pennylane import qchem

CHEM_ACC = 1.6e-3  # Hartree

# CNOT counts of PennyLane 0.45 decompositions (qml.transforms.decompose to CNOT/RY/H), checked in tests
CNOT_SINGLE = 2
CNOT_DOUBLE = 14

GEOMS = {
    "H2": lambda r: (["H", "H"], [0, 0, 0, 0, 0, r], {}),
    "LiH": lambda r: (["Li", "H"], [0, 0, 0, 0, 0, r], dict(active_electrons=2, active_orbitals=4)),
    "H4": lambda r: (["H"] * 4, [0, 0, 0, 0, 0, r, 0, 0, 2 * r, 0, 0, 3 * r],
                     dict(active_electrons=4, active_orbitals=4)),
    "H6": lambda r: (["H"] * 6, [c for i in range(6) for c in (0, 0, i * r)],
                     dict(active_electrons=6, active_orbitals=6)),
}


@dataclass
class Problem:
    name: str
    bond: float
    n_qubits: int
    n_electrons: int
    H: sp.csr_matrix
    e_exact: float
    pool: list = field(default_factory=list)  # list of wire tuples (len 2 or 4)

    @property
    def tag(self):
        return f"{self.name}@{self.bond:.2f}"


def build(name: str, bond: float) -> Problem:
    symbols, coords, kw = GEOMS[name](bond)
    mol = qchem.Molecule(symbols, np.array(coords, dtype=float), unit="angstrom")
    H, n = qchem.molecular_hamiltonian(mol, **kw)
    Hs = qml.matrix(H, wire_order=range(n)) if n <= 8 else H.sparse_matrix(wire_order=range(n))
    Hs = sp.csr_matrix(np.real(Hs) if not sp.issparse(Hs) else Hs.real)
    ne = kw.get("active_electrons", mol.n_electrons)
    e0 = sp.linalg.eigsh(Hs, k=1, which="SA")[0][0] if n > 8 else np.linalg.eigvalsh(Hs.toarray())[0]
    singles, doubles = qchem.excitations(ne, n)
    pool = [tuple(w) for w in doubles] + [tuple(w) for w in singles]
    return Problem(name, bond, n, ne, Hs, float(e0), pool)


def cnot_cost(wires) -> int:
    return CNOT_DOUBLE if len(wires) == 4 else CNOT_SINGLE
