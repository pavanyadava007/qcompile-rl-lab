# qcompile-rl-lab

Reinforcement learning and graph neural networks for two quantum compilation problems:

1. **Ansatz design for VQE.** An RL agent builds a variational quantum eigensolver circuit for a molecule one
   excitation gate at a time and tries to reach chemical accuracy (1.6 mHa) with as few CNOTs as possible.
   Baselines: UCCSD, ADAPT-VQE, random gate order and a beam search reference.
2. **Cut placement for circuit cutting.** Split a circuit across two devices of half the size with as few gate cuts
   as possible (sampling overhead 9 per cut CNOT). Methods: exact enumeration, Kernighan-Lin, spectral bisection, a
   GNN, a GNN with one Kernighan-Lin pass, and `find_cuts` from qiskit-addon-cutting. Selected cuts are then run with
   qiskit-addon-cutting on Qiskit Aer and the expectation value is reconstructed from the subexperiments.

Everything runs on CPU plus one NVIDIA L4 (AWS g6). All numbers below come from `results/` and are rebuilt by
`scripts/make_results.py`. Full tables: [results/RESULTS.md](results/RESULTS.md).

## Main results

**VQE ansatz design** (LiH and H4 in STO-3G, 8-qubit active spaces from PennyLane qchem; trained on 14 bond lengths,
tested on 14 other ones, 3 training seeds per policy):

| method | runs at chemical accuracy | mean CNOTs |
|---|---|---|
| ADAPT-VQE | 14/14 | 86.6 |
| beam search (reference) | 14/14 | 81.7 |
| RL, GNN policy | 42/42 | 83.3 |
| RL, policy without message passing | 42/42 | 80.0 |

- On stretched LiH (2.8 and 3.2 A) the policies find an 18-CNOT circuit where ADAPT-VQE uses 44-46 CNOTs.
- On stretched H4 (1.6 and 1.8 A) some runs need 146-150 CNOTs, fewer than both ADAPT-VQE and the beam search (162).
- **Negative result:** message passing did not help. The GNN policy was not better than the same network without
  message passing (both keep a global mean-pool readout).
- Zero-shot H6 (12 qubits, 117-gate pool): mixed. At 1.0 A, 4 of 6 runs used 662 CNOTs against 686 for ADAPT-VQE.
  At 1.2 A no method, ADAPT-VQE included, reached chemical accuracy within 60 gates.

**Cut placement** (200 circuits: modular circuits with hidden clusters and QAOA MaxCut circuits, 16-64 qubits):

- The GNN found the exact optimum on 69 of 80 circuits small enough to solve exactly (86 %); spectral bisection 60/80,
  GNN + one Kernighan-Lin pass 73/80, Kernighan-Lin with 10 restarts 80/80.
- On QAOA circuits the GNN needed 33-40 % fewer cuts than `find_cuts` (16 qubits: 18.0 vs 28.4; 32 qubits: 29.2 vs 49.0)
  and took milliseconds instead of 6-12 s. `find_cuts` may split into more than two subcircuits (a looser constraint)
  and still needed more cuts there. On modular circuits it used that freedom and needed fewer cuts than any bisection.
- Kernighan-Lin with restarts is the strongest simple baseline; the learned partitioner has not beaten it on accuracy.

**Cut-and-run on Aer:** see the tables in [results/RESULTS.md](results/RESULTS.md), section C. A compact LiH ansatz
splits into two 4-qubit halves with zero cuts. A 12-qubit modular circuit needs 2 gate cuts (sampling overhead 81),
and its reconstruction error is several times larger than uncut sampling with the same total shots.

## What is checked

- `tests/test_sim.py`: the statevector simulator matches PennyLane's `SingleExcitation`/`DoubleExcitation` energies
  to 1e-10 Ha; adjoint gradients match finite differences; CNOT costs (2 and 14) match PennyLane's decompositions.
- `tests/test_lower.py`: the CNOT-level Qiskit circuit gives the same energy as the simulator; the shot-based
  estimate converges to the exact value.
- `tests/test_partition.py`: exact bisection is a lower bound and recovers planted cuts.

## Limitations

- Simulation only, noiseless. No hardware runs, no noise models.
- Small molecules (up to 12 qubits) and a fixed fermionic excitation pool.
- The RL reward uses a CNOT cost model taken from PennyLane's decompositions, not from a hardware-native transpile.
- Quantum error correction (the third topic in the Fraunhofer IIS ad) is not covered.

## Run

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
export PYTHONPATH=.
.venv/bin/python -m pytest -q tests
scripts/run_training.sh                   # 3 seeds x (GNN, no message passing), about 20 min on 32 vCPU
.venv/bin/python scripts/eval_vqe_rl.py
.venv/bin/python scripts/train_cut_gnn.py # about 2 min on an L4
.venv/bin/python scripts/cut_benchmark.py
.venv/bin/python scripts/cut_execute.py
.venv/bin/python scripts/make_results.py
```

## References

- Ostaszewski et al., RL for optimization of variational quantum circuit architectures, NeurIPS 2021, arXiv:2103.16089
- Grimsley et al., ADAPT-VQE, Nat. Commun. 10, 3007 (2019); Tang et al., qubit-ADAPT-VQE, arXiv:1911.10205
- Mattick et al., Optimizing quantum circuits via ZX diagrams using RL and GNNs, arXiv:2504.03429
- Froehler et al., Scalable circuit cutting with combined gate and wire cuts using gate groups, arXiv:2608.05287
- qiskit-addon-cutting, https://github.com/Qiskit/qiskit-addon-cutting
