"""Benchmark circuit families for cut placement (CNOT-level Qiskit circuits)."""
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit


def modular(n, rng, k=2, intra=6, inter=1, layers=3):
    """k hidden clusters with dense intra-cluster and sparse inter-cluster CNOTs; qubit labels shuffled."""
    perm = rng.permutation(n)
    groups = np.array_split(perm, k)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, np.pi)), q)
        for g in groups:
            for _ in range(intra * len(g) // 4):
                a, b = rng.choice(g, 2, replace=False)
                qc.cx(int(a), int(b))
        for _ in range(inter):
            gi, gj = rng.choice(k, 2, replace=False)
            qc.cx(int(rng.choice(groups[gi])), int(rng.choice(groups[gj])))
    return qc


def qaoa_regular(n, rng, p=2, d=3):
    """QAOA MaxCut on a random d-regular graph, RZZ lowered to CX-RZ-CX."""
    g = nx.random_regular_graph(d, n, seed=int(rng.integers(1 << 30)))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        for a, b in g.edges:
            qc.cx(a, b)
            qc.rz(float(rng.uniform(0, np.pi)), b)
            qc.cx(a, b)
        for q in range(n):
            qc.rx(float(rng.uniform(0, np.pi)), q)
    return qc


def interaction_graph(qc) -> np.ndarray:
    n = qc.num_qubits
    W = np.zeros((n, n))
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            W[a, b] += 1
            W[b, a] += 1
    return W
