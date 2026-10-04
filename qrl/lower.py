"""Lower an excitation-gate ansatz to a CNOT-level Qiskit circuit and the Hamiltonian to a SparsePauliOp.

PennyLane wire w maps to Qiskit qubit w; Pauli labels are reversed because Qiskit labels are little-endian.
"""
import numpy as np
import pennylane as qml
from pennylane import qchem
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .chem import GEOMS

_GATESET = {"CNOT", "RY", "Hadamard"}


def to_qiskit(problem, gates, thetas) -> QuantumCircuit:
    n = problem.n_qubits
    qc = QuantumCircuit(n)
    for w in range(problem.n_electrons):
        qc.x(w)
    for g, t in zip(gates, thetas):
        w = list(problem.pool[g])
        op = (qml.DoubleExcitation if len(w) == 4 else qml.SingleExcitation)(float(t), wires=w)
        (tape,), _ = qml.transforms.decompose(qml.tape.QuantumScript([op]), gate_set=_GATESET)
        for o in tape.operations:
            q = [int(x) for x in o.wires]
            if o.name == "CNOT":
                qc.cx(*q)
            elif o.name == "RY":
                qc.ry(float(o.parameters[0]), q[0])
            elif o.name == "Hadamard":
                qc.h(q[0])
            else:
                raise ValueError(o.name)
    return qc


def hamiltonian_spo(problem) -> SparsePauliOp:
    symbols, coords, kw = GEOMS[problem.name](problem.bond)
    mol = qchem.Molecule(symbols, np.array(coords, dtype=float), unit="angstrom")
    H, n = qchem.molecular_hamiltonian(mol, **kw)
    coeffs, ops = H.terms()
    labels, cs = [], []
    for c, op in zip(coeffs, ops):
        s = qml.pauli.pauli_word_to_string(op, wire_map={w: w for w in range(n)})
        labels.append(s[::-1])
        cs.append(float(np.real(c)))
    return SparsePauliOp(labels, cs).simplify()
