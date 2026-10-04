"""Evaluate trained gate-selection policies against UCCSD, ADAPT-VQE, random order and beam search."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from qrl.chem import CHEM_ACC, build
from qrl.env import N_FEAT, AnsatzEnv
from qrl.policy import GatePolicy
from qrl.sim import Circuit
from qrl.vqe import adapt, beam_search, cnots, random_order, uccsd

TEST = ([("LiH", r) for r in (1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6)]
        + [("H4", r) for r in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)])
ZERO_SHOT = [("H6", r) for r in (0.9, 1.0, 1.2)]


def run_policy(path, key, max_gates):
    torch.set_num_threads(1)
    pol = GatePolicy(N_FEAT, message_passing="gnn_" in os.path.basename(path))
    pol.load_state_dict(torch.load(path))
    env = AnsatzEnv(build(*key), max_gates=max_gates)
    adj = torch.tensor(env.adj, dtype=torch.float32)
    x, m = env.reset()
    done = False
    t = time.perf_counter()
    while not done:
        with torch.no_grad():
            a = int(torch.argmax(pol(torch.tensor(x), adj, torch.tensor(m))))
        _, done = env.step(a)
        if not done:
            x, m = env.features()
    return dict(reached=bool(env.err < CHEM_ACC), n_gates=len(env.gates), cnots=cnots(env.p, env.gates),
                err=float(env.err), gates=env.gates, t=time.perf_counter() - t)


def baselines(key, max_gates, do_beam):
    p = build(*key)
    c = Circuit(p.n_qubits, p.n_electrons, p.H, p.pool)
    out = dict(key=list(key), e_exact=p.e_exact, hf_err=c.energy([], []) - p.e_exact, pool=len(p.pool))
    out["uccsd"] = uccsd(p, c)
    t = time.perf_counter()
    out["adapt"] = adapt(p, c, max_gates=max_gates) | dict(t=time.perf_counter() - t)
    rs = [random_order(p, c, np.random.default_rng(s), max_gates=max_gates) for s in range(5)]
    out["random"] = dict(reached=float(np.mean([r["reached"] for r in rs])),
                         cnots=float(np.mean([r["cnots"] for r in rs])))
    if do_beam:
        t = time.perf_counter()
        out["beam"] = beam_search(p, c, upper=out["adapt"]) | dict(t=time.perf_counter() - t)
    return out


def _job(args):
    kind, a = args
    if kind == "base":
        return ("base", tuple(a[0]), baselines(*a))
    path, key, mg = a
    return ("pol", (os.path.basename(path), tuple(key)), run_policy(path, key, mg))


def main():
    pols = sorted(glob.glob("results/vqe_rl/*_s[0-9].pt"))
    jobs = [("base", (k, 20, True)) for k in TEST] + [("base", (k, 60, False)) for k in ZERO_SHOT]
    jobs += [("pol", (p, k, 20)) for p in pols for k in TEST] + [("pol", (p, k, 60)) for p in pols for k in ZERO_SHOT]
    res = {"base": {}, "pol": {}}
    with mp.get_context("fork").Pool(30) as pool:
        for kind, key, r in pool.imap_unordered(_job, jobs):
            res[kind]["|".join(map(str, key))] = r
            print(kind, key, {k: v for k, v in r.items() if k in ("reached", "cnots")}, flush=True)
    json.dump(res, open("results/vqe_rl/eval.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
