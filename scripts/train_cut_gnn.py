"""Supervised training of the partition GNN on exact min-bisection labels (n = 10..20)."""
import json
import os
import time

import numpy as np
import torch

from qrl.circuits import interaction_graph, modular, qaoa_regular
from qrl.partition import PartitionGNN, exact, flip_invariant_bce, node_features


def sample_graph(n, rng):
    if rng.random() < 0.5:
        return interaction_graph(modular(n, rng, k=int(rng.integers(2, 5)), inter=int(rng.integers(1, 4))))
    return interaction_graph(qaoa_regular(n, rng, d=int(rng.choice([3, 4]))))


def main(steps=4000, batch=32, seed=0, out="results/cut/gnn.pt", per_size=300):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sizes = list(range(10, 22, 2))
    t0 = time.time()
    data = {n: [] for n in sizes}
    for n in sizes:
        for _ in range(per_size):
            W = sample_graph(n, rng)
            data[n].append((W, exact(W)[1].astype(np.float32)))
    print(f"labels built in {time.time() - t0:.0f}s", flush=True)
    model = PartitionGNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = []
    for it in range(steps):
        n = int(rng.choice(sizes))
        items = [data[n][i] for i in rng.integers(0, per_size, batch)]
        W = torch.tensor(np.stack([w for w, _ in items]), dtype=torch.float32, device=dev)
        X = torch.tensor(np.stack([node_features(w, rng) for w, _ in items]), dtype=torch.float32, device=dev)
        y = torch.tensor(np.stack([y for _, y in items]), device=dev)
        loss = flip_invariant_bce(model(W, X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            log.append(dict(it=it, loss=float(loss), t=time.time() - t0))
            print(log[-1], flush=True)
    torch.save(model.state_dict(), out)
    json.dump(log, open(out.replace(".pt", "_log.json"), "w"))


if __name__ == "__main__":
    os.makedirs("results/cut", exist_ok=True)
    main()
