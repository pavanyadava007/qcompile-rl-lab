# Results

All numbers produced by the scripts in this repo on one AWS g6 instance (NVIDIA L4 GPU, 32 vCPU). Chemical accuracy = 1.6 mHa. CNOT counts use PennyLane 0.45 decompositions (single excitation 2, double excitation 14). 'x' = did not reach chemical accuracy within the gate budget (final error in brackets).

## A. RL ansatz construction for VQE

### Held-out bond lengths (trained on other LiH and H4 geometries)

| problem | HF error (mHa) | UCCSD CNOTs (error mHa) | ADAPT-VQE CNOTs | beam search CNOTs | random order CNOTs (reach) | RL-GNN CNOTs per seed | RL-MLP CNOTs per seed |
|---|---|---|---|---|---|---|---|
| LiH @ 3.2 A | 30.4 | 138 (0.00) | 46 | 18 | 118 (80%) | 18, 46, 18 | 18, 18, 18 |
| LiH @ 2.8 A | 9.0 | 138 (0.00) | 44 | 18 | 118 (80%) | 18, 44, 18 | 18, 18, 18 |
| LiH @ 1.6 A | 1.8 | 138 (0.00) | 14 | 14 | 74 (100%) | 14, 14, 14 | 14, 14, 14 |
| LiH @ 1.2 A | 2.1 | 138 (0.00) | 14 | 14 | 115 (100%) | 14, 14, 14 | 14, 14, 14 |
| LiH @ 2.0 A | 2.0 | 138 (-0.00) | 14 | 14 | 74 (100%) | 14, 14, 14 | 14, 14, 14 |
| LiH @ 2.4 A | 3.2 | 138 (0.00) | 28 | 16 | 100 (80%) | 28, 28, 28 | 16, 16, 16 |
| LiH @ 3.6 A | 203.3 | 138 (0.00) | 46 | 44 | 173 (20%) | 34, 46, 46 | 46, 46, 46 |
| H4 @ 0.8 A | 46.2 | 268 (0.01) | 126 | 126 | 208 (20%) | 126, 126, 126 | 126, 126, 126 |
| H4 @ 1.0 A | 67.8 | 268 (0.07) | 126 | 126 | 208 (20%) | 126, 126, 126 | 126, 126, 126 |
| H4 @ 1.4 A | 141.3 | 268 (0.84) | 140 | 140 | 208 (0%) | 140, 140, 140 | 140, 140, 140 |
| H4 @ 1.8 A | 257.6 | 268 (3.05) | 162 | 162 | 208 (0%) | 146, 162, 150 | 150, 150, 146 |
| H4 @ 1.6 A | 195.4 | 268 (1.81) | 162 | 162 | 208 (0%) | 146, 162, 150 | 150, 150, 150 |
| H4 @ 1.2 A | 98.7 | 268 (0.30) | 140 | 140 | 208 (0%) | 140, 140, 140 | 140, 140, 140 |
| H4 @ 2.0 A | 322.2 | 268 (4.18) | 150 | 150 | 208 (0%) | 152, 188, 150 | 148, 152, 150 |

### Zero-shot: H6 (12 qubits, 117-gate pool, never seen in training, 60-gate budget)

| problem | HF error (mHa) | UCCSD CNOTs (error mHa) | ADAPT-VQE CNOTs | beam search CNOTs | random order CNOTs (reach) | RL-GNN CNOTs per seed | RL-MLP CNOTs per seed |
|---|---|---|---|---|---|---|---|
| H6 @ 0.9 A | 83.8 | 1422 (0.68) | 630 | - | 706 (0%) | 630, 644, 630 | 630, 620, 618 |
| H6 @ 1.2 A | 145.0 | 1422 (3.08) | x (2.2 mHa) | - | 706 (0%) | x (2.2 mHa), x (3.1 mHa), x (3.2 mHa) | x (3.2 mHa), x (3.2 mHa), x (3.2 mHa) |
| H6 @ 1.0 A | 100.5 | 1422 (1.15) | 686 | - | 706 (0%) | 662, 686, 662 | 686, 664, 662 |

### Summary over the 14 held-out geometries

| method | reached chemical accuracy | mean CNOTs (when reached) | fewer CNOTs than ADAPT | equal | more |
|---|---|---|---|---|---|
| adapt | 14/14 | 86.6 | 0 | 14 | 0 |
| beam | 14/14 | 81.7 | 4 | 10 | 0 |
| gnn | 42/42 | 83.3 | 9 | 31 | 2 |
| mlp | 42/42 | 80.0 | 16 | 25 | 1 |

## B. Cut placement for two equal subcircuits (gate cuts, mean number of cut CNOTs, 20 circuits per row)

| family | qubits | CX gates | exact | KL (10 restarts) | spectral | GNN | GNN + KL (1 pass) | random | find_cuts (log9 overhead, gate+wire) | find_cuts timeouts |
|---|---|---|---|---|---|---|---|---|---|---|
| modular | 16 | 77.1 | 6.0 | 6.0 | 8.3 | 6.2 | 6.2 | 40.7 | 5.5 (n=20) | 0/20 |
| modular | 20 | 93.2 | 8.7 | 8.7 | 9.9 | 9.6 | 9.1 | 48.9 | 6.0 (n=20) | 0/20 |
| modular | 32 | 148.8 | - | 9.8 | 12.7 | 10.9 | 10.2 | 77.0 | 5.0 (n=20) | 0/20 |
| modular | 48 | 222.0 | - | 14.7 | 19.2 | 16.7 | 15.1 | 113.4 | - | not run |
| modular | 64 | 293.2 | - | 10.7 | 15.0 | 12.2 | 11.0 | 149.0 | - | not run |
| qaoa | 16 | 96.0 | 17.6 | 17.6 | 18.4 | 18.0 | 18.0 | 50.9 | 28.4 (n=20) | 0/20 |
| qaoa | 20 | 120.0 | 18.8 | 18.8 | 20.0 | 19.6 | 19.2 | 63.2 | 29.4 (n=20) | 0/20 |
| qaoa | 32 | 192.0 | - | 28.0 | 32.4 | 29.2 | 28.8 | 99.2 | 49.0 (n=20) | 0/20 |
| qaoa | 48 | 288.0 | - | 36.0 | 40.8 | 36.8 | 35.6 | 147.8 | - | not run |
| qaoa | 64 | 384.0 | - | 46.0 | 50.8 | 48.4 | 44.8 | 194.9 | - | not run |

Share of circuits where the method found the exact optimum (n = 16, 20):

- kl10: 80/80
- spectral: 60/80
- gnn: 69/80
- gnn_kl: 73/80
- find_cuts (gate + wire cuts allowed, compared in log9 overhead): at or below the gate-only optimum in 44/80

Runtime per circuit (seconds, CPU, 2 threads):

| family | qubits | exact s | KL10 s | spectral s | GNN s | GNN+KL s | find_cuts s |
|---|---|---|---|---|---|---|---|
| modular | 16 | 0.004 | 0.002 | 0.000 | 0.002 | 0.002 | 6.136 |
| modular | 20 | 0.067 | 0.003 | 0.000 | 0.002 | 0.002 | 8.610 |
| modular | 32 | - | 0.005 | 0.001 | 0.003 | 0.003 | 11.568 |
| modular | 48 | - | 0.008 | 0.000 | 0.003 | 0.004 | - |
| modular | 64 | - | 0.011 | 0.000 | 0.005 | 0.005 | - |
| qaoa | 16 | 0.004 | 0.002 | 0.000 | 0.002 | 0.002 | 9.771 |
| qaoa | 20 | 0.056 | 0.002 | 0.000 | 0.010 | 0.010 | 12.357 |
| qaoa | 32 | - | 0.003 | 0.000 | 0.003 | 0.003 | 10.599 |
| qaoa | 48 | - | 0.005 | 0.000 | 0.003 | 0.004 | - |
| qaoa | 64 | - | 0.006 | 0.000 | 0.004 | 0.005 | - |

## C. Cut-and-run on Aer (qiskit-addon-cutting)

LiH @ 3.0 A, 3-gate ansatz (18 CNOTs, 1.31 mHa from FCI), 105 Pauli terms. Exact bisection finds a split into two 4-qubit halves with 0 cut CNOTs (sides 11110000), so the halves run independently (overhead 1).

| shots per subexperiment | subexperiments | sampling overhead | mean abs error, cut (mHa) | mean abs error, uncut, same total shots (mHa) |
|---|---|---|---|---|
| 1000 | 18 | 1 | 3.57 | 6.88 |
| 10000 | 18 | 1 | 0.75 | 1.51 |

12-qubit modular circuit (38 CNOTs, hidden 6+6 clusters), Ising-type observable with 34 terms. Cut placement: GNN 2 cuts, GNN + one KL pass 2, exact optimum 2.

| shots per subexperiment | subexperiments | sampling overhead | mean abs error, cut | mean abs error, uncut, same total shots |
|---|---|---|---|---|
| 1000 | 144 | 81 | 0.1298 | 0.0161 |
| 10000 | 144 | 81 | 0.0471 | 0.0041 |
| 100000 | 144 | 81 | 0.0080 | 0.0013 |

Errors are means over 10 sampler seeds against the exact statevector value.
