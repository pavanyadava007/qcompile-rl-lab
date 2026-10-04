"""Balanced qubit bisection for gate cutting: classical baselines and an unsupervised GNN.

Cost of a partition = number of CNOTs whose qubits sit in different halves (each is one CX gate cut,
sampling overhead 9 per cut in qiskit-addon-cutting), so overhead = 9 ** cost.
"""
import itertools

import networkx as nx
import numpy as np
import torch
from torch import nn


def cut_cost(W, side):
    side = np.asarray(side, bool)
    return float(W[np.ix_(side, ~side)].sum())


def exact(W):
    """Exhaustive balanced bisection (node 0 fixed on side A), vectorised over all subsets."""
    n = len(W)
    combs = np.array(list(itertools.combinations(range(1, n), n // 2 - 1)), dtype=np.int64)
    S = np.zeros((len(combs), n), bool)
    S[:, 0] = True
    S[np.arange(len(combs))[:, None], combs] = True
    iu, ju = np.nonzero(np.triu(W))
    cost = ((S[:, iu] != S[:, ju]) * W[iu, ju]).sum(1)
    k = int(np.argmin(cost))
    return float(cost[k]), S[k]


def _graph(W):
    G = nx.Graph()
    G.add_nodes_from(range(len(W)))
    for i, j in zip(*np.nonzero(np.triu(W))):
        G.add_edge(int(i), int(j), weight=float(W[i, j]))
    return G


def kernighan_lin(W, restarts=10, seed=0, init=None):
    G, best, arg = _graph(W), np.inf, None
    n = len(W)
    for r in range(restarts):
        part = None
        if init is not None and r == 0:
            part = (set(np.flatnonzero(init).tolist()), set(np.flatnonzero(~init).tolist()))
        a, _ = nx.algorithms.community.kernighan_lin_bisection(G, partition=part, weight="weight", seed=seed + r)
        s = np.zeros(n, bool)
        s[list(a)] = True
        c = cut_cost(W, s)
        if c < best:
            best, arg = c, s
    return best, arg


def spectral(W):
    L = np.diag(W.sum(1)) - W
    _, v = np.linalg.eigh(L)
    s = np.zeros(len(W), bool)
    s[np.argsort(v[:, 1])[: len(W) // 2]] = True
    return cut_cost(W, s), s


def random_balanced(W, rng, trials=1):
    n = len(W)
    cs = []
    for _ in range(trials):
        s = np.zeros(n, bool)
        s[rng.permutation(n)[: n // 2]] = True
        cs.append(cut_cost(W, s))
    return float(np.mean(cs))


N_EIG = 4


def node_features(W, rng=None):
    """Weighted degree, degree and the first non-trivial Laplacian eigenvectors (random sign flips in training)."""
    n = len(W)
    wdeg = W.sum(1)
    L = np.diag(wdeg) - W
    _, v = np.linalg.eigh(L)
    eig = v[:, 1:1 + N_EIG] * np.sqrt(n)
    if eig.shape[1] < N_EIG:
        eig = np.pad(eig, ((0, 0), (0, N_EIG - eig.shape[1])))
    if rng is not None:
        eig = eig * rng.choice([-1.0, 1.0], N_EIG)
    return np.concatenate([wdeg[:, None] / max(wdeg.max(), 1), (W > 0).sum(1, keepdims=True) / n, eig], 1)


class PartitionGNN(nn.Module):
    """Message passing on the weighted interaction graph; outputs one side-logit per qubit."""

    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.inp = nn.Linear(2 + N_EIG, hidden)
        self.ls = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.ln = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, W, X):
        A = W / W.sum(-1, keepdim=True).clamp(min=1)
        h = torch.relu(self.inp(X))
        for ls, ln in zip(self.ls, self.ln):
            h = torch.relu(ls(h) + ln(A @ h)) + h
        g = h.mean(1, keepdim=True).expand_as(h)
        return self.out(torch.cat([h, g], -1)).squeeze(-1)


def flip_invariant_bce(logits, y):
    bce = nn.functional.binary_cross_entropy_with_logits
    a = bce(logits, y, reduction="none").mean(1)
    b = bce(logits, 1 - y, reduction="none").mean(1)
    return torch.minimum(a, b).mean()


@torch.no_grad()
def gnn_partition(model, W, samples=8, seed=0, device="cpu"):
    """Score qubits, take the top half; several eigenvector sign patterns, keep the cheapest."""
    n = len(W)
    rng = np.random.default_rng(seed)
    X = np.stack([node_features(W, rng if k else None) for k in range(samples)])
    Wt = torch.tensor(W, dtype=torch.float32, device=device).expand(samples, n, n)
    logits = model(Wt, torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    best, arg = np.inf, None
    for z in logits:
        s = np.zeros(n, bool)
        s[np.argsort(-z)[: n // 2]] = True
        c = cut_cost(W, s)
        if c < best:
            best, arg = c, s
    return best, arg
