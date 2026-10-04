"""Cut-placement benchmark: exact / KL / spectral / GNN / GNN+KL / qiskit-addon-cutting find_cuts."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import math
import signal
import time

import numpy as np
import torch
from qiskit_addon_cutting.automated_cut_finding import DeviceConstraints, OptimizationParameters, find_cuts

from qrl.circuits import interaction_graph, modular, qaoa_regular
from qrl.partition import PartitionGNN, exact, gnn_partition, kernighan_lin, random_balanced, spectral

SIZES = [16, 20, 32, 48, 64]
PER = 20
FIND_CUTS_MAX_N = 32
FIND_CUTS_TIMEOUT = 60


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


def main():
    torch.set_num_threads(2)
    model = PartitionGNN()
    model.load_state_dict(torch.load("results/cut/gnn.pt", map_location="cpu"))
    model.eval()
    rng = np.random.default_rng(2026)
    rows = []
    signal.signal(signal.SIGALRM, _alarm)
    for fam in ("modular", "qaoa"):
        for n in SIZES:
            for i in range(PER):
                qc = modular(n, rng, k=int(rng.integers(2, 5)), inter=2) if fam == "modular" else qaoa_regular(n, rng)
                W = interaction_graph(qc)
                row = dict(family=fam, n=n, idx=i, cx=int(qc.count_ops().get("cx", 0)))
                if n <= 20:
                    (c, _), t = timed(exact, W)
                    row |= dict(exact=c, t_exact=t)
                (c, _), t = timed(kernighan_lin, W, 10)
                row |= dict(kl10=c, t_kl10=t)
                (c, _), t = timed(spectral, W)
                row |= dict(spectral=c, t_spectral=t)
                (g, s), t = timed(gnn_partition, model, W)
                (c2, _), t2 = timed(kernighan_lin, W, 1, 0, s)
                row |= dict(gnn=g, t_gnn=t, gnn_kl=c2, t_gnn_kl=t + t2, random=random_balanced(W, rng, 50))
                if n <= FIND_CUTS_MAX_N:
                    signal.alarm(FIND_CUTS_TIMEOUT)
                    try:
                        (_cc, md), t = timed(find_cuts, qc, OptimizationParameters(seed=1), DeviceConstraints(n // 2))
                        row |= dict(fc_log9=math.log(md["sampling_overhead"], 9), fc_min=bool(md["minimum_reached"]),
                                    fc_cuts=len(md["cuts"]), fc_wire=sum(c[0] == "Wire Cut" for c in md["cuts"]), t_fc=t)
                    except Timeout:
                        row |= dict(fc_log9=None, t_fc=FIND_CUTS_TIMEOUT)
                    finally:
                        signal.alarm(0)
                rows.append(row)
            print(fam, n, "done", flush=True)
            json.dump(rows, open("results/cut/benchmark.json", "w"), indent=1)


if __name__ == "__main__":
    main()
