"""VQE optimisation and classical ansatz-construction baselines."""
import numpy as np
from scipy.optimize import minimize

from .chem import CHEM_ACC, cnot_cost
from .sim import Circuit


def optimise(circ: Circuit, gates, init=None):
    if not gates:
        return circ.energy([], []), np.zeros(0)
    x0 = np.zeros(len(gates)) if init is None else np.asarray(init, float)
    res = minimize(lambda x: circ.energy_grad(gates, x), x0, jac=True, method="BFGS",
                   options=dict(gtol=1e-7, maxiter=400))
    return float(res.fun), res.x


def cnots(problem, gates):
    return int(sum(cnot_cost(problem.pool[g]) for g in gates))


def trace(problem, circ, gates):
    """Error after each prefix of a gate sequence (re-optimised, warm-started)."""
    errs, th = [], np.zeros(0)
    for k in range(1, len(gates) + 1):
        e, th = optimise(circ, gates[:k], np.append(th, 0.0))
        errs.append(e - problem.e_exact)
    return errs


def summarise(problem, gates, errs):
    """Gates/CNOTs needed until the first prefix below chemical accuracy (None if never)."""
    for k, err in enumerate(errs, 1):
        if err < CHEM_ACC:
            return dict(reached=True, n_gates=k, cnots=cnots(problem, gates[:k]), err=err)
    return dict(reached=False, n_gates=len(gates), cnots=cnots(problem, gates), err=errs[-1] if errs else None)


def uccsd(problem, circ):
    gates = list(range(len(problem.pool)))  # doubles first, then singles, each once
    e, _ = optimise(circ, gates)
    err = e - problem.e_exact
    return dict(reached=err < CHEM_ACC, n_gates=len(gates), cnots=cnots(problem, gates), err=err)


def adapt(problem, circ, max_gates=40):
    gates, th, errs = [], np.zeros(0), []
    e = circ.energy([], [])
    while len(gates) < max_gates and e - problem.e_exact >= CHEM_ACC:
        g = int(np.argmax(np.abs(circ.pool_gradients(gates, th))))
        gates.append(g)
        e, th = optimise(circ, gates, np.append(th, 0.0))
        errs.append(e - problem.e_exact)
    return summarise(problem, gates, errs) | dict(gates=gates)


def random_order(problem, circ, rng, max_gates=40):
    gates, th, errs = [], np.zeros(0), []
    e = circ.energy([], [])
    while len(gates) < max_gates and e - problem.e_exact >= CHEM_ACC:
        gates.append(int(rng.integers(len(problem.pool))))
        e, th = optimise(circ, gates, np.append(th, 0.0))
        errs.append(e - problem.e_exact)
    return summarise(problem, gates, errs)


def beam_search(problem, circ, width=16, max_gates=24, expand=10, upper=None):
    """Reference: beam search over sequences, ranked by error, expanding the top-|grad| pool gates
    plus all cheap singles. Returns the cheapest (CNOT) sequence found that reaches chemical accuracy."""
    beam = [([], np.zeros(0), circ.energy([], []) - problem.e_exact)]
    best = dict(upper) if upper and upper.get("reached") else None
    singles = [g for g, w in enumerate(problem.pool) if len(w) == 2]
    for _ in range(max_gates):
        cands = []
        for gates, th, _err in beam:
            pg = np.abs(circ.pool_gradients(gates, th))
            top = set(np.argsort(-pg)[:expand].tolist()) | set(singles)
            for g in top:
                if gates and g == gates[-1]:
                    continue
                e, th2 = optimise(circ, gates + [g], np.append(th, 0.0))
                cands.append((gates + [g], th2, e - problem.e_exact))
        done = [c for c in cands if c[2] < CHEM_ACC]
        for gates, _, err in done:
            c = cnots(problem, gates)
            if best is None or c < best["cnots"]:
                best = dict(reached=True, n_gates=len(gates), cnots=c, err=err, gates=gates)
        alive = [c for c in cands if c[2] >= CHEM_ACC
                 and (best is None or cnots(problem, c[0]) < best["cnots"])]
        if not alive:
            break
        # rank by log-error per CNOT spent so cheap progress survives
        alive.sort(key=lambda c: np.log10(c[2]) + 0.01 * cnots(problem, c[0]))
        beam = alive[:width]
    return best or dict(reached=False)
