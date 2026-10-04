"""Cut-and-run with qiskit-addon-cutting on Aer: (1) LiH ansatz split into two 4-qubit halves,
(2) 12-qubit modular circuit with real gate cuts placed by GNN + KL. Writes results/cut/execute.{json,md}."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import time

import numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp

from qrl.chem import build
from qrl.circuits import interaction_graph, modular
from qrl.cutrun import cut_energy, exact_energy, uncut_energy
from qrl.lower import hamiltonian_spo, to_qiskit
from qrl.partition import PartitionGNN, exact, gnn_partition, kernighan_lin
from qrl.sim import Circuit
from qrl.vqe import optimise

SEEDS = range(10)


def ising(n, edges):
    terms = [("".join("Z" if q in e else "I" for q in range(n))[::-1], 1.0) for e in edges]
    terms += [("".join("X" if q == i else "I" for q in range(n))[::-1], 0.5) for i in range(n)]
    return SparsePauliOp.from_list(terms)


def sweep(qc, H, side, shots_list):
    e0 = exact_energy(qc, H)
    rows = []
    for shots in shots_list:
        ec, eu, n_sub, ov, t = [], [], 0, 1.0, time.perf_counter()
        for s in SEEDS:
            e, ov, n_sub = cut_energy(qc, H, side, shots, s)
            ec.append(e - e0)
            eu.append(uncut_energy(qc, H, shots * max(n_sub, 1), s) - e0)  # same total shot budget
        rows.append(dict(shots_per_subexperiment=shots, n_subexperiments=n_sub, overhead=ov,
                         cut_abs_err=float(np.mean(np.abs(ec))), cut_err_sd=float(np.std(ec)),
                         uncut_abs_err=float(np.mean(np.abs(eu))), seconds=time.perf_counter() - t))
        print(rows[-1], flush=True)
    return e0, rows


def main():
    out, md = {}, []
    # 1) LiH @ 3.0 A: compact ansatz (2 singles + 1 double, 18 CNOTs) that reaches chemical accuracy
    p = build("LiH", 3.0)
    c = Circuit(p.n_qubits, p.n_electrons, p.H, p.pool)
    gates = [p.pool.index((0, 2)), p.pool.index((0, 1, 2, 3)), p.pool.index((0, 2))]
    e, th = optimise(c, gates)
    qc = to_qiskit(p, gates, th)
    H = hamiltonian_spo(p)
    W = interaction_graph(qc)
    best, side = exact(W)
    e0, rows = sweep(qc, H, side, [1000, 10000])
    out["lih"] = dict(ansatz_error_mha=(e - p.e_exact) * 1e3, cx=int(qc.count_ops()["cx"]), min_cuts=best,
                      side=side.astype(int).tolist(), pauli_terms=len(H), exact_energy=e0, rows=rows)
    md += [(f"LiH @ 3.0 A, 3-gate ansatz ({out['lih']['cx']} CNOTs, {out['lih']['ansatz_error_mha']:.2f} mHa from FCI), "
           f"{len(H)} Pauli terms. Exact bisection finds a split into two 4-qubit halves with {best:.0f} cut CNOTs "
           f"(sides {''.join(map(str, out['lih']['side']))}), so the halves run independently (overhead 1)."), "",
           "| shots per subexperiment | subexperiments | sampling overhead | mean abs error, cut (mHa) | mean abs error, uncut, same total shots (mHa) |",
           "|---|---|---|---|---|"]
    md += [f"| {r['shots_per_subexperiment']} | {r['n_subexperiments']} | {r['overhead']:.0f} | {r['cut_abs_err'] * 1e3:.2f} | {r['uncut_abs_err'] * 1e3:.2f} |" for r in rows]
    # 2) 12-qubit modular circuit, real gate cuts placed by GNN + KL
    rng = np.random.default_rng(7)
    qc2 = modular(12, rng, k=2, intra=6, inter=1, layers=2)
    W2 = interaction_graph(qc2)
    model = PartitionGNN()
    model.load_state_dict(torch.load("results/cut/gnn.pt", map_location="cpu"))
    model.eval()
    g, s = gnn_partition(model, W2)
    c2, side2 = kernighan_lin(W2, 1, 0, s)
    opt2, _ = exact(W2)
    edges = [tuple(sorted(e)) for e in zip(*np.nonzero(np.triu(W2)))]
    H2 = ising(12, edges)
    e02, rows2 = sweep(qc2, H2, side2, [1000, 10000, 100000])
    out["modular12"] = dict(cx=int(qc2.count_ops()["cx"]), gnn_cuts=g, gnn_kl_cuts=c2, exact_cuts=opt2,
                            pauli_terms=len(H2), exact_energy=e02, rows=rows2)
    md += ["", (f"12-qubit modular circuit ({out['modular12']['cx']} CNOTs, hidden 6+6 clusters), Ising-type observable with "
           f"{len(H2)} terms. Cut placement: GNN {g:.0f} cuts, GNN + one KL pass {c2:.0f}, exact optimum {opt2:.0f}."), "",
           "| shots per subexperiment | subexperiments | sampling overhead | mean abs error, cut | mean abs error, uncut, same total shots |",
           "|---|---|---|---|---|"]
    md += [f"| {r['shots_per_subexperiment']} | {r['n_subexperiments']} | {r['overhead']:.0f} | {r['cut_abs_err']:.4f} | {r['uncut_abs_err']:.4f} |" for r in rows2]
    md += ["", f"Errors are means over {len(SEEDS)} sampler seeds against the exact statevector value."]
    json.dump(out, open("results/cut/execute.json", "w"), indent=1, default=float)
    open("results/cut/execute.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
