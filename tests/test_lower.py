import numpy as np
from qiskit.quantum_info import Statevector

from qrl.chem import build
from qrl.lower import hamiltonian_spo, to_qiskit
from qrl.sim import Circuit


def test_qiskit_energy_matches_simulator():
    p = build("H4", 1.3)
    c = Circuit(p.n_qubits, p.n_electrons, p.H, p.pool)
    rng = np.random.default_rng(3)
    gates = [int(g) for g in rng.integers(0, len(p.pool), 5)]
    th = rng.normal(size=5)
    qc = to_qiskit(p, gates, th)
    e_q = float(np.real(Statevector(qc).expectation_value(hamiltonian_spo(p))))
    assert abs(e_q - c.energy(gates, th)) < 1e-9
    assert qc.count_ops().get("cx", 0) == sum(14 if len(p.pool[g]) == 4 else 2 for g in gates)


def test_uncut_shot_estimate_converges():
    from qrl.cutrun import exact_energy, uncut_energy
    p = build("LiH", 1.6)
    qc = to_qiskit(p, [0, 3], [0.1, -0.2])
    H = hamiltonian_spo(p)
    e0 = exact_energy(qc, H)
    assert abs(uncut_energy(qc, H, 400_000, 0) - e0) < 5e-3
