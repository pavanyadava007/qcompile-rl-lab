import numpy as np
import pennylane as qml
import pytest

from qrl.chem import build
from qrl.sim import Circuit


@pytest.fixture(scope="module")
def lih():
    return build("LiH", 1.6)


def pennylane_energy(p, gates, thetas):
    dev = qml.device("default.qubit", wires=p.n_qubits)
    Hm = p.H.toarray()

    @qml.qnode(dev)
    def circ():
        qml.BasisState(np.array([1] * p.n_electrons + [0] * (p.n_qubits - p.n_electrons)), wires=range(p.n_qubits))
        for g, t in zip(gates, thetas):
            w = p.pool[g]
            (qml.DoubleExcitation if len(w) == 4 else qml.SingleExcitation)(t, wires=list(w))
        return qml.state()

    psi = np.real(circ())
    return float(psi @ Hm @ psi)


def test_energy_matches_pennylane(lih):
    rng = np.random.default_rng(0)
    c = Circuit(lih.n_qubits, lih.n_electrons, lih.H, lih.pool)
    gates = list(rng.integers(0, len(lih.pool), 8))
    thetas = rng.normal(size=8)
    assert abs(c.energy(gates, thetas) - pennylane_energy(lih, gates, thetas)) < 1e-10


def test_adjoint_gradient_matches_finite_difference(lih):
    rng = np.random.default_rng(1)
    c = Circuit(lih.n_qubits, lih.n_electrons, lih.H, lih.pool)
    gates = list(rng.integers(0, len(lih.pool), 6))
    thetas = rng.normal(size=6)
    _, g = c.energy_grad(gates, thetas)
    eps = 1e-6
    fd = [(c.energy(gates, thetas + eps * e) - c.energy(gates, thetas - eps * e)) / (2 * eps) for e in np.eye(6)]
    assert np.allclose(g, fd, atol=1e-7)


def test_pool_gradient_matches_append(lih):
    rng = np.random.default_rng(2)
    c = Circuit(lih.n_qubits, lih.n_electrons, lih.H, lih.pool)
    gates, thetas = [0, 3], rng.normal(size=2)
    pg = c.pool_gradients(gates, thetas)
    for k in (0, 5, len(lih.pool) - 1):
        _, g = c.energy_grad(gates + [k], np.append(thetas, 0.0))
        assert abs(pg[k] - g[-1]) < 1e-10


def test_cnot_model_matches_pennylane_decomposition():
    from collections import Counter

    from qrl.chem import CNOT_DOUBLE, CNOT_SINGLE
    gs = {"CNOT", "RY", "Hadamard"}
    for op, want in [(qml.SingleExcitation(0.3, wires=[0, 1]), CNOT_SINGLE),
                     (qml.DoubleExcitation(0.3, wires=[0, 1, 2, 3]), CNOT_DOUBLE)]:
        (t,), _ = qml.transforms.decompose(qml.tape.QuantumScript([op]), gate_set=gs)
        assert Counter(o.name for o in t.operations)["CNOT"] == want
