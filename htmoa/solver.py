"""Exact reference solver (MILP) and frozen instance sets.

`solve_reference` returns the optimal plan, so the normalized score of a plan is
its fraction of the optimal profit. The simulated-annealing helpers at the end
are only used by the offline mock backend.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path

from .env import Instance, generate, verify


def solve_reference(inst: Instance) -> tuple[dict, int]:
    """Optimal plan and profit, by mixed-integer programming (HiGHS via SciPy).

    Variables per project p and quarter q: seniors s, juniors j, and d = 1 once p is
    complete by the end of q; per crisis c and quarter q before it strikes: m = 1 if
    c is mitigated in q. The result is re-checked with the verifier.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    T, P, C = inst.n_quarters, inst.projects, inst.crises
    Q = range(1, T + 1)
    idx: dict[tuple, int] = {}
    ub: list[int] = []
    for p in P:
        for q in Q:
            for name, hi in (("s", inst.senior_cap), ("j", inst.junior_cap), ("d", 1)):
                idx[(name, p.id, q)] = len(idx)
                ub.append(hi)
    for c in C:
        for q in range(1, c.quarter):
            idx[("m", c.id, q)] = len(idx)
            ub.append(1)
    n = len(idx)
    rows, lows, highs = [], [], []

    def con(coefs: dict, lb=-np.inf, ub_=np.inf):
        row = np.zeros(n)
        for k, v in coefs.items():
            row[idx[k]] += v
        rows.append(row)
        lows.append(lb)
        highs.append(ub_)

    def pay(p, q):
        return max(0, p.revenue - p.late_penalty * max(0, q - p.deadline))

    def cash(q: int, revenue_upto: int) -> tuple[dict, float]:
        """Cash after paying the costs of quarter q, counting the revenue of quarters
        1..revenue_upto, as (coefficients, constant)."""
        co: dict = {}
        const = inst.initial_cash - inst.overhead * q
        for p in P:
            for t in range(1, q + 1):
                co[("s", p.id, t)] = -inst.senior_cost
                co[("j", p.id, t)] = -inst.junior_cost
            for t in range(1, revenue_upto + 1):  # sum_t pay(t) * (d_t - d_{t-1})
                co[("d", p.id, t)] = pay(p, t) - (pay(p, t + 1) if t < revenue_upto else 0)
        for c in C:
            struck = c.quarter <= q
            const -= c.damage if struck else 0
            for t in range(1, c.quarter):
                paid = -c.mitigation_cost if t <= q else 0
                co[("m", c.id, t)] = paid + (c.damage if struck else 0)
        return co, const

    for p in P:
        for q in Q:
            con({("j", p.id, q): 1, ("s", p.id, q): -inst.supervision_ratio}, ub_=0)
            work = {("d", p.id, q): p.work}          # d = 1 only if enough work was done
            for t in range(1, q + 1):
                work[("s", p.id, t)] = -inst.senior_output
                work[("j", p.id, t)] = -inst.junior_output
            con(work, ub_=0)
            if q > 1:
                con({("d", p.id, q): 1, ("d", p.id, q - 1): -1}, lb=0)
            for dep in p.deps:                        # staffed only after deps completed
                for role, cap in (("s", inst.senior_cap), ("j", inst.junior_cap)):
                    if q == 1:
                        con({(role, p.id, q): 1}, ub_=0)
                    else:
                        con({(role, p.id, q): 1, ("d", dep, q - 1): -cap}, ub_=0)
    for c in C:
        con({("m", c.id, t): 1 for t in range(1, c.quarter)}, ub_=1)
    for q in Q:
        sen = {("s", p.id, q): 1 for p in P}
        cap = inst.senior_cap
        for c in C:
            if q < c.quarter:
                sen[("m", c.id, q)] = c.mitigation_seniors
            if c.quarter == q:
                cap -= c.senior_loss
                for t in range(1, q):
                    sen[("m", c.id, t)] = -c.senior_loss
        con(sen, ub_=cap)
        con({("j", p.id, q): 1 for p in P}, ub_=inst.junior_cap)
        co, const = cash(q, q - 1)
        con(co, lb=-const)                            # no bankruptcy

    co, const = cash(T, T)
    obj = np.zeros(n)
    for k, v in co.items():
        obj[idx[k]] = -v
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lows, highs),
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.array(ub)),
               options={"mip_rel_gap": 0.0})
    if res.status == 2:  # infeasible: even doing nothing goes bankrupt
        return {"quarters": []}, -10**9
    if res.x is None:
        raise RuntimeError(f"MILP failed on {inst.id}: {res.message}")
    x = np.rint(res.x).astype(int)
    plan = {"quarters": [
        {"assignments": [{"project": p.id, "senior": int(x[idx[("s", p.id, q)]]),
                          "junior": int(x[idx[("j", p.id, q)]])} for p in P
                         if x[idx[("s", p.id, q)]] or x[idx[("j", p.id, q)]]],
         "mitigations": [c.id for c in C if q < c.quarter and x[idx[("m", c.id, q)]]]}
        for q in Q]}
    ev = verify(inst, plan)
    bound = round(const - res.fun) - inst.initial_cash
    if not ev.valid or ev.profit != bound:
        raise RuntimeError(f"MILP/verifier mismatch on {inst.id}: verifier {ev.profit}, "
                           f"MILP {bound}, {ev.violations}")
    return plan, ev.profit


MIN_OPTIMUM = 100


def build_instances(n: int, difficulty: str = "medium", first_seed: int = 0) -> list[Instance]:
    """n instances with their optimal profit attached. Draws whose optimum is below
    MIN_OPTIMUM are rejected so that the normalized score is well conditioned."""
    out = []
    for seed in range(first_seed, first_seed + n):
        for attempt in range(50):
            inst = generate(seed, difficulty, attempt)
            _, ref = solve_reference(inst)
            if ref >= MIN_OPTIMUM:
                out.append(dataclasses.replace(inst, ref_profit=ref))
                break
        else:
            raise RuntimeError(f"no solvable instance for seed {seed}")
    return out


def load_or_build(path: Path, n: int, difficulty: str, first_seed: int = 0) -> list[Instance]:
    """Instances are frozen to disk so every method sees exactly the same set."""
    if path.exists():
        insts = [Instance.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
        if any(i.difficulty != difficulty for i in insts):
            raise SystemExit(f"{path} holds '{insts[0].difficulty}' instances, not "
                             f"'{difficulty}'; use a new --out directory.")
        if len(insts) >= n:
            return insts[:n]
        # Generation is deterministic per seed, so a smaller saved set is a prefix of
        # the larger one: keep it and add the missing seeds.
        insts += build_instances(n - len(insts), difficulty, first_seed + len(insts))
    else:
        insts = build_instances(n, difficulty, first_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([i.to_dict() for i in insts], indent=1), encoding="utf-8")
    return insts


# --------------------------------------------------------------------------- #
# Simulated annealing — used only by the mock backend to fabricate noisy plans
# --------------------------------------------------------------------------- #

def empty_plan(inst: Instance) -> dict:
    return {"quarters": [{"assignments": {}, "mitigations": []} for _ in range(inst.n_quarters)]}


def _copy(plan: dict) -> dict:
    return {"quarters": [
        {"assignments": {k: dict(v) for k, v in q["assignments"].items()},
         "mitigations": list(q["mitigations"])} for q in plan["quarters"]]}


def _objective(inst: Instance, plan: dict) -> tuple[float, bool, int]:
    ev = verify(inst, plan)
    return ev.profit - 500 * len(ev.violations), ev.valid, ev.profit


def mutate(inst: Instance, plan: dict, rng: random.Random) -> dict:
    new = _copy(plan)
    qs = new["quarters"]
    T = inst.n_quarters
    r = rng.random()
    pid = rng.choice(inst.projects).id
    q = rng.randrange(T)
    if r < 0.60:
        cell = qs[q]["assignments"].setdefault(pid, {"senior": 0, "junior": 0})
        role = rng.choice(("senior", "junior"))
        cell[role] = max(0, cell[role] + rng.choice((-2, -1, 1, 2)))
    elif r < 0.75 and inst.crises:
        c = rng.choice(inst.crises)
        had = any(c.id in x["mitigations"] for x in qs)
        for x in qs:
            if c.id in x["mitigations"]:
                x["mitigations"].remove(c.id)
        if not had or rng.random() < 0.5:
            qs[rng.randrange(c.quarter - 1)]["mitigations"].append(c.id)
    elif r < 0.90:
        q2 = rng.randrange(T)
        a, b = qs[q]["assignments"].pop(pid, None), qs[q2]["assignments"].pop(pid, None)
        if b:
            qs[q]["assignments"][pid] = b
        if a:
            qs[q2]["assignments"][pid] = a
    else:
        qs[q]["assignments"].pop(pid, None)
    return new


def to_canonical(plan: dict) -> dict:
    """Convert the annealer's dict-form plan to the list form LLMs are asked for."""
    return {"quarters": [
        {"assignments": [{"project": k, **v} for k, v in sorted(q["assignments"].items())
                         if v["senior"] or v["junior"]],
         "mitigations": sorted(q["mitigations"])} for q in plan["quarters"]]}


def anneal(inst: Instance, iters: int, rng: random.Random, temp: float = 40.0) -> tuple[dict, int]:
    """Returns (best valid plan found, its profit). The empty plan is the fallback."""
    cur = empty_plan(inst)
    cur_obj, valid, profit = _objective(inst, cur)
    best, best_profit = (cur, profit) if valid else (empty_plan(inst), -10**9)
    for it in range(iters):
        t = temp * (1 - it / iters) + 1e-6
        cand = mutate(inst, cur, rng)
        obj, valid, profit = _objective(inst, cand)
        if obj >= cur_obj or rng.random() < math.exp((obj - cur_obj) / t):
            cur, cur_obj = cand, obj
            if valid and profit > best_profit:
                best, best_profit = cand, profit
    return best, best_profit
