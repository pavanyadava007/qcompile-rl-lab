"""Real statevector simulator for particle-conserving Givens/excitation gates.

Gate conventions follow PennyLane's qml.SingleExcitation / qml.DoubleExcitation
(wire 0 is the most significant bit). Each gate rotates basis-state pairs (i, j):
    psi[i] <- c psi[i] - s psi[j],  psi[j] <- s psi[i] + c psi[j],  c, s = cos, sin(theta/2)
Gradients use the adjoint method, so one energy+gradient call costs about 3 circuit passes.
"""
import numpy as np


def _bit(idx, w, n):
    return (idx >> (n - 1 - w)) & 1


def pair_indices(wires, n):
    idx = np.arange(2 ** n)
    if len(wires) == 2:
        a, b = wires
        i = idx[(_bit(idx, a, n) == 0) & (_bit(idx, b, n) == 1)]
        j = i ^ ((1 << (n - 1 - a)) | (1 << (n - 1 - b)))
    else:
        a, b, c, d = wires
        m = (_bit(idx, a, n) == 0) & (_bit(idx, b, n) == 0) & (_bit(idx, c, n) == 1) & (_bit(idx, d, n) == 1)
        i = idx[m]
        j = i ^ sum(1 << (n - 1 - w) for w in wires)
    return i, j


def hf_state(n, ne):
    psi = np.zeros(2 ** n)
    psi[int("1" * ne + "0" * (n - ne), 2)] = 1.0
    return psi


class Circuit:
    def __init__(self, n, ne, H, pool):
        self.n, self.H = n, H
        self.psi0 = hf_state(n, ne)
        self.pairs = [pair_indices(w, n) for w in pool]

    def _apply(self, psi, g, theta, deriv=False):
        i, j = self.pairs[g]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        out = psi.copy()
        if deriv:  # dU/dtheta applied to psi; zero outside the rotated subspace
            out[:] = 0.0
            c, s = -s / 2, c / 2
        pi, pj = psi[i], psi[j]
        out[i] = c * pi - s * pj
        out[j] = s * pi + c * pj
        return out

    def state(self, gates, thetas):
        psi = self.psi0
        for g, t in zip(gates, thetas):
            psi = self._apply(psi, g, t)
        return psi

    def energy(self, gates, thetas):
        psi = self.state(gates, thetas)
        return float(psi @ (self.H @ psi))

    def energy_grad(self, gates, thetas):
        psi = self.state(gates, thetas)
        lam = self.H @ psi
        e = float(psi @ lam)
        grad = np.zeros(len(gates))
        phi = psi
        for k in range(len(gates) - 1, -1, -1):
            g, t = gates[k], thetas[k]
            phi = self._apply(phi, g, -t)  # state before gate k
            grad[k] = 2.0 * lam @ self._apply(phi, g, t, deriv=True)
            lam = self._apply(lam, g, -t)
        return e, grad

    def pool_gradients(self, gates, thetas):
        """dE/dtheta for appending each pool gate at theta=0 (ADAPT-VQE criterion)."""
        psi = self.state(gates, thetas)
        hpsi = self.H @ psi
        out = np.empty(len(self.pairs))
        for g, (i, j) in enumerate(self.pairs):
            # dU/dtheta at 0 maps psi[i] -> -psi[j]/2, psi[j] -> psi[i]/2
            out[g] = 2.0 * (hpsi[i] @ (-psi[j] / 2) + hpsi[j] @ (psi[i] / 2))
        return out
