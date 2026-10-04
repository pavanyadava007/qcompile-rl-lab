"""Ansatz-construction MDP: append one pool gate per step, re-optimise all angles (warm start)."""
import numpy as np

from .chem import CHEM_ACC, CNOT_DOUBLE, cnot_cost
from .sim import Circuit
from .vqe import optimise

N_FEAT = 9


class AnsatzEnv:
    def __init__(self, problem, max_gates=20, lam=0.1):
        self.p, self.max_gates, self.lam = problem, max_gates, lam
        self.circ = Circuit(problem.n_qubits, problem.n_electrons, problem.H, problem.pool)
        self.is_double = np.array([len(w) == 4 for w in problem.pool], float)
        self.cost = np.array([cnot_cost(w) for w in problem.pool], float)
        # gate graph: edge if two pool gates share a qubit
        S = [set(w) for w in problem.pool]
        self.adj = np.array([[1.0 if a is not b and S[a] & S[b] else 0.0 for b in range(len(S))]
                             for a in range(len(S))])

    def reset(self):
        self.gates, self.th = [], np.zeros(0)
        self.err = self.circ.energy([], []) - self.p.e_exact
        return self.features()

    def _one_d_gain(self):
        """Exact energy drop from appending gate g alone with its best angle: E(t) = A + B cos t + C sin t."""
        psi = self.circ.state(self.gates, self.th)
        e0 = float(psi @ (self.p.H @ psi))
        gains = np.empty(len(self.p.pool))
        for g in range(len(self.p.pool)):
            ep = self._e_after(psi, g, np.pi)
            eh = self._e_after(psi, g, np.pi / 2)
            A, B = (e0 + ep) / 2, (e0 - ep) / 2
            C = eh - A
            gains[g] = e0 - (A - np.hypot(B, C))
        return gains

    def _e_after(self, psi, g, t):
        phi = self.circ._apply(psi, g, t)
        return float(phi @ (self.p.H @ phi))

    def features(self):
        grad = np.abs(self.circ.pool_gradients(self.gates, self.th))
        gain = np.clip(self._one_d_gain(), 0, None)
        used = np.zeros(len(self.p.pool))
        for g in self.gates:
            used[g] += 1
        last = np.zeros(len(self.p.pool))
        if self.gates:
            last[self.gates[-1]] = 1
        le = np.log10(max(self.err, 1e-12))
        x = np.stack([
            grad / (grad.max() + 1e-12),
            np.log10(grad + 1e-12) / 6,
            np.minimum(gain / max(self.err, 1e-12), 1.0),
            np.log10(gain + 1e-12) / 6,
            self.is_double,
            np.minimum(used, 3) / 3,
            last,
            np.full(len(used), le / 4),
            np.full(len(used), len(self.gates) / self.max_gates),
        ], 1)
        mask = last == 0
        return x.astype(np.float32), mask

    def step(self, g):
        prev = self.err
        self.gates.append(int(g))
        e, self.th = optimise(self.circ, self.gates, np.append(self.th, 0.0))
        self.err = e - self.p.e_exact
        progress = np.log10(max(prev, CHEM_ACC)) - np.log10(max(self.err, CHEM_ACC))
        r = progress - self.lam * self.cost[g] / CNOT_DOUBLE
        done = self.err < CHEM_ACC or len(self.gates) >= self.max_gates
        if self.err < CHEM_ACC:
            r += 1.0
        return r, done
