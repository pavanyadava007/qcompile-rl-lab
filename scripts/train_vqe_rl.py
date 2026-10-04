"""Train the gate-selection policy with REINFORCE (group baseline per problem) on LiH + H4 bond lengths."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from qrl.chem import build
from qrl.env import N_FEAT, AnsatzEnv
from qrl.policy import GatePolicy

TRAIN = [("LiH", r) for r in (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4)] + [("H4", r) for r in (0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9)]
_ENVS = {}


def _env(key):
    if key not in _ENVS:
        _ENVS[key] = AnsatzEnv(build(*key))
    return _ENVS[key]


def rollout(args):
    key, state_dict, mp_flag, seed, greedy = args
    torch.set_num_threads(1)
    env = _env(key)
    pol = GatePolicy(N_FEAT, message_passing=mp_flag)
    pol.load_state_dict(state_dict)
    rng = np.random.default_rng(seed)
    adj = torch.tensor(env.adj, dtype=torch.float32)
    x, mask = env.reset()
    traj, rewards, done = [], [], False
    while not done:
        with torch.no_grad():
            logits = pol(torch.tensor(x), adj, torch.tensor(mask))
        p = torch.softmax(logits, 0).numpy().astype(np.float64)
        p /= p.sum()
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        traj.append((x, mask, a))
        r, done = env.step(a)
        rewards.append(r)
        if not done:
            x, mask = env.features()
    from qrl.vqe import cnots
    return dict(key=key, traj=traj, rewards=rewards, reached=bool(env.err < 1.6e-3),
                cnots=cnots(env.p, env.gates), n_gates=len(env.gates), err=float(env.err), gates=env.gates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--problems", type=int, default=8)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mp", action="store_true")
    ap.add_argument("--out", default="results/vqe_rl")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    pol = GatePolicy(N_FEAT, message_passing=not a.no_mp)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    adjs = {}
    tag = f"{'mlp' if a.no_mp else 'gnn'}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    log = []
    for k in TRAIN:  # build before fork so workers inherit the Hamiltonians
        _env(k)
        adjs[k] = torch.tensor(_ENVS[k].adj, dtype=torch.float32)
    t0 = time.time()
    with mp.get_context("fork").Pool(30) as pool:
        for it in range(a.iters):
            sd = {k: v.clone() for k, v in pol.state_dict().items()}
            keys = [TRAIN[i] for i in rng.choice(len(TRAIN), a.problems, replace=False)]
            jobs = [(k, sd, not a.no_mp, int(rng.integers(1 << 30)), False) for k in keys for _ in range(a.group)]
            outs = pool.map(rollout, jobs)
            loss, n = 0.0, 0
            for k in keys:
                grp = [o for o in outs if o["key"] == k]
                rets = np.array([sum(o["rewards"]) for o in grp])
                adv = (rets - rets.mean()) / (rets.std() + 1e-6)
                for o, A in zip(grp, adv):
                    for x, m, act in o["traj"]:
                        logits = pol(torch.tensor(x), adjs[k], torch.tensor(m))
                        lp = torch.log_softmax(logits, 0)
                        ent = -(lp.exp() * lp.clamp(min=-30)).sum()
                        loss = loss - float(A) * lp[act] - 0.01 * ent
                        n += 1
            opt.zero_grad()
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            row = dict(it=it, reach=float(np.mean([o["reached"] for o in outs])),
                       cnots=float(np.mean([o["cnots"] for o in outs if o["reached"]] or [np.nan])),
                       ret=float(np.mean([sum(o["rewards"]) for o in outs])), t=time.time() - t0)
            log.append(row)
            if it % 5 == 0 or it < 3:
                print(json.dumps(row), flush=True)
            if it % 25 == 0 or it == a.iters - 1:
                torch.save(pol.state_dict(), f"{a.out}/{tag}.pt")
    json.dump(log, open(f"{a.out}/{tag}_log.json", "w"))


if __name__ == "__main__":
    main()
