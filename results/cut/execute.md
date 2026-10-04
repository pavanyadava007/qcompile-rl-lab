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
