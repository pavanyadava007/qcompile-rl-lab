"""Run a lowered ansatz as two cut subcircuits with qiskit-addon-cutting on Aer and reconstruct the energy."""
import numpy as np
from qiskit.primitives import PrimitiveResult
from qiskit.quantum_info import Statevector
from qiskit_addon_cutting import generate_cutting_experiments, partition_problem, reconstruct_expectation_values
from qiskit_aer.primitives import SamplerV2


def cut_energy(qc, H, side, shots, seed):
    """side: bool array, True = subcircuit A. Returns (energy, sampling overhead, #subexperiments)."""
    labels = "".join("A" if s else "B" for s in side)
    pp = partition_problem(circuit=qc, partition_labels=labels, observables=H.paulis)
    overhead = float(np.prod([b.overhead for b in pp.bases])) if pp.bases else 1.0
    subexp, coeffs = generate_cutting_experiments(circuits=pp.subcircuits, observables=pp.subobservables,
                                                  num_samples=np.inf)
    # independent seed per subexperiment: one batch-wide seed correlates the QPD terms and the noise stops averaging out
    rng = np.random.default_rng(seed)
    results = {}
    for lab, circs in subexp.items():
        pubs = [SamplerV2(seed=int(rng.integers(1 << 31))).run([c], shots=shots).result()[0] for c in circs]
        results[lab] = PrimitiveResult(pubs)
    vals = reconstruct_expectation_values(results, coeffs, pp.subobservables)
    n_sub = sum(len(v) for v in subexp.values())
    return float(np.real(np.dot(H.coeffs, vals))), overhead, n_sub


def uncut_energy(qc, H, total_shots, seed):
    """Shot-based estimate on the full circuit: one measurement circuit per qubit-wise-commuting group,
    total shot budget split evenly across groups (what a device without cutting would run)."""
    groups = H.group_commuting(qubit_wise=True)
    shots = max(1, total_shots // len(groups))
    circs = []
    for g in groups:
        c = qc.copy()
        for q in range(qc.num_qubits):
            ops = {str(p[q]) for p in g.paulis} - {"I"}
            if ops == {"X"}:
                c.h(q)
            elif ops == {"Y"}:
                c.sdg(q)
                c.h(q)
        c.measure_all()
        circs.append(c)
    rng = np.random.default_rng(seed)
    res = [SamplerV2(seed=int(rng.integers(1 << 31))).run([c], shots=shots).result()[0] for c in circs]
    e = 0.0
    for g, r in zip(groups, res):
        counts = r.data.meas.get_counts()
        n = sum(counts.values())
        for pauli, coeff in zip(g.paulis, g.coeffs):
            mask = [q for q in range(qc.num_qubits) if str(pauli[q]) != "I"]
            ev = sum(v * (-1) ** sum(int(b[::-1][q]) for q in mask) for b, v in counts.items()) / n
            e += float(np.real(coeff)) * ev
    return e


def exact_energy(qc, H):
    return float(np.real(Statevector(qc).expectation_value(H)))
