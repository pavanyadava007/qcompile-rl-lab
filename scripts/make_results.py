"""Build results/RESULTS.md from results/vqe_rl/eval.json and results/cut/benchmark.json."""
import json
from collections import defaultdict

import numpy as np


def vqe_tables(ev):
    import ast
    base = ev["base"]
    pol = {}
    for k, v in ev["pol"].items():  # "gnn_s0.pt|('H4', 1.2)" -> "gnn_s0|H4|1.2"
        f, key = k.split("|", 1)
        name, r = ast.literal_eval(key)
        pol[f"{f.removesuffix('.pt')}|{name}|{r}"] = v
    seeds = sorted({k.split("|")[0] for k in pol})
    kinds = {"gnn": [s for s in seeds if s.startswith("gnn")], "mlp": [s for s in seeds if s.startswith("mlp")]}
    out = []
    summ = {m: defaultdict(list) for m in ("adapt", "beam", "uccsd", "gnn", "mlp", "random")}
    for group, title in (("test", "Held-out bond lengths (trained on other LiH and H4 geometries)"),
                         ("zs", "Zero-shot: H6 (12 qubits, 117-gate pool, never seen in training, 60-gate budget)")):
        out += [f"### {title}", "",
                "| problem | HF error (mHa) | UCCSD CNOTs (error mHa) | ADAPT-VQE CNOTs | beam search CNOTs | random order CNOTs (reach) | RL-GNN CNOTs per seed | RL-MLP CNOTs per seed |",
                "|---|---|---|---|---|---|---|---|"]
        for key, b in base.items():
            is_zs = key.startswith("H6")
            if (group == "zs") != is_zs:
                continue
            cell = lambda r: (f"{r['cnots']}" if r.get("reached") else f"x ({r.get('err', float('nan')) * 1e3:.1f} mHa)")
            u, a = b["uccsd"], b["adapt"]
            bm = b.get("beam")
            row = [key.replace("|", " @ ") + " A", f"{b['hf_err'] * 1e3:.1f}",
                   f"{u['cnots']} ({u['err'] * 1e3:.2f})", cell(a), cell(bm) if bm else "-",
                   f"{b['random']['cnots']:.0f} ({b['random']['reached'] * 100:.0f}%)"]
            for m in ("gnn", "mlp"):
                rs = [pol[f"{s}|{key}"] for s in kinds[m] if f"{s}|{key}" in pol]
                row.append(", ".join(cell(r) for r in rs))
                summ[m]["grp_" + group].append(rs)
            summ["adapt"]["grp_" + group].append(a)
            summ["beam"]["grp_" + group].append(bm)
            summ["uccsd"]["grp_" + group].append(u)
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    # summary over held-out test problems
    out += ["### Summary over the 14 held-out geometries", "",
            "| method | reached chemical accuracy | mean CNOTs (when reached) | fewer CNOTs than ADAPT | equal | more |", "|---|---|---|---|---|---|"]
    A = summ["adapt"]["grp_test"]
    for m in ("adapt", "beam", "gnn", "mlp"):
        rows = summ[m]["grp_test"]
        if m in ("gnn", "mlp"):
            flat = [(r, a) for rs, a in zip(rows, A) for r in rs]
        else:
            flat = list(zip(rows, A))
        reached = [r for r, _ in flat if r and r.get("reached")]
        better = sum(1 for r, a in flat if r and r.get("reached") and (not a["reached"] or r["cnots"] < a["cnots"]))
        equal = sum(1 for r, a in flat if r and r.get("reached") and a["reached"] and r["cnots"] == a["cnots"])
        worse = sum(1 for r, a in flat if r and (not r.get("reached") or (a["reached"] and r["cnots"] > a["cnots"])))
        out.append(f"| {m} | {len(reached)}/{len(flat)} | {np.mean([r['cnots'] for r in reached]):.1f} | {better} | {equal} | {worse} |")
    out.append("")
    return out


def cut_tables(rows):
    out = ["| family | qubits | CX gates | exact | KL (10 restarts) | spectral | GNN | GNN + KL (1 pass) | random | find_cuts (log9 overhead, gate+wire) | find_cuts timeouts |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    tout = ["| family | qubits | exact s | KL10 s | spectral s | GNN s | GNN+KL s | find_cuts s |", "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["family"], r["n"])].append(r)
    for (fam, n), rs in sorted(by.items()):
        mean = lambda k, rs=rs: (f"{np.mean([r[k] for r in rs]):.1f}" if all(r.get(k) is not None for r in rs) else "-")
        fc = [r.get("fc_log9") for r in rs if "fc_log9" in r]
        fc_ok = [x for x in fc if x is not None]
        fcs = f"{np.mean(fc_ok):.1f} (n={len(fc_ok)})" if fc_ok else "-"
        to = f"{sum(x is None for x in fc)}/{len(fc)}" if fc else "not run"
        out.append(f"| {fam} | {n} | {mean('cx')} | {mean('exact')} | {mean('kl10')} | {mean('spectral')} | {mean('gnn')} | {mean('gnn_kl')} | {mean('random')} | {fcs} | {to} |")
        tm = lambda k, rs=rs: (f"{np.mean([r[k] for r in rs if r.get(k) is not None]):.3f}" if any(r.get(k) is not None for r in rs) else "-")
        tout.append(f"| {fam} | {n} | {tm('t_exact')} | {tm('t_kl10')} | {tm('t_spectral')} | {tm('t_gnn')} | {tm('t_gnn_kl')} | {tm('t_fc')} |")
    # optimality gap at n <= 20
    opt = ["", "Share of circuits where the method found the exact optimum (n = 16, 20):", ""]
    small = [r for r in rows if r.get("exact") is not None]
    for m in ("kl10", "spectral", "gnn", "gnn_kl"):
        opt.append(f"- {m}: {sum(r[m] == r['exact'] for r in small)}/{len(small)}")
    fc_small = [r for r in small if r.get("fc_log9") is not None]
    opt.append(f"- find_cuts (gate + wire cuts allowed, compared in log9 overhead): at or below the gate-only optimum in "
               f"{sum(r['fc_log9'] <= r['exact'] + 1e-9 for r in fc_small)}/{len(fc_small)}")
    return out, tout, opt


def main():
    md = ["# Results", "", ("All numbers produced by the scripts in this repo on one AWS g6 instance (NVIDIA L4 GPU, 32 vCPU). "
          "Chemical accuracy = 1.6 mHa. CNOT counts use PennyLane 0.45 decompositions (single excitation 2, double excitation 14). "
          "'x' = did not reach chemical accuracy within the gate budget (final error in brackets)."), ""]
    try:
        ev = json.load(open("results/vqe_rl/eval.json"))
        md += ["## A. RL ansatz construction for VQE", ""] + vqe_tables(ev)
    except FileNotFoundError:
        md += ["## A. (eval.json missing)", ""]
    try:
        rows = json.load(open("results/cut/benchmark.json"))
        t1, t2, opt = cut_tables(rows)
        md += ["## B. Cut placement for two equal subcircuits (gate cuts, mean number of cut CNOTs, 20 circuits per row)", ""] + t1 + opt + ["", "Runtime per circuit (seconds, CPU, 2 threads):", ""] + t2 + [""]
    except FileNotFoundError:
        md += ["## B. (benchmark.json missing)", ""]
    try:
        md += ["## C. Cut-and-run on Aer (qiskit-addon-cutting)", ""] + open("results/cut/execute.md").read().splitlines() + [""]
    except FileNotFoundError:
        pass
    open("results/RESULTS.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
