import numpy as np

from qrl.circuits import interaction_graph, modular
from qrl.partition import exact, kernighan_lin, spectral


def test_exact_is_lower_bound_and_finds_planted_cut():
    rng = np.random.default_rng(0)
    qc = modular(12, rng, inter=1, layers=3)
    W = interaction_graph(qc)
    c, s = exact(W)
    assert s.sum() == 6
    assert c <= kernighan_lin(W)[0] and c <= spectral(W)[0]
    assert c <= 3  # three planted inter-cluster CNOTs
