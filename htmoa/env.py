"""Deterministic enterprise planning environment (Section 4.2 of the draft).

A COO allocates senior/junior engineers over a multi-quarter horizon to deliver
interdependent projects under cash-flow, capacity, supervision and dependency
constraints, with scheduled crises that can be mitigated proactively.

`verify(instance, plan)` is the deterministic verifier: no LLM is involved in
scoring. All money is in $k.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Project:
    id: str
    work: int            # work points required
    revenue: int         # paid at the end of the completion quarter
    deadline: int        # quarter (1-indexed)
    late_penalty: int    # deducted from revenue per quarter late
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Crisis:
    id: str
    quarter: int             # strikes at the start of this quarter
    damage: int              # cash lost if not mitigated
    senior_loss: int         # seniors unavailable that quarter if not mitigated
    mitigation_cost: int
    mitigation_seniors: int  # seniors consumed in the quarter the mitigation is run


@dataclass(frozen=True)
class Instance:
    id: str
    seed: int
    difficulty: str
    n_quarters: int
    initial_cash: int
    overhead: int
    senior_cap: int
    junior_cap: int
    senior_cost: int
    junior_cost: int
    senior_output: int
    junior_output: int
    supervision_ratio: int
    projects: tuple[Project, ...]
    crises: tuple[Crisis, ...]
    ref_profit: int = 0  # optimal profit, from the exact reference solver

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        d = dict(d)
        d["projects"] = tuple(Project(**{**p, "deps": tuple(p["deps"])}) for p in d["projects"])
        d["crises"] = tuple(Crisis(**c) for c in d["crises"])
        return Instance(**d)


@dataclass
class Evaluation:
    valid: bool
    profit: int
    violations: list[dict] = field(default_factory=list)   # {kind, quarter, detail}
    warnings: list[str] = field(default_factory=list)
    completed: dict[str, int] = field(default_factory=dict)  # project id -> quarter
    cash_by_quarter: list[int] = field(default_factory=list)

    @property
    def violation_kinds(self) -> list[str]:
        return sorted({v["kind"] for v in self.violations})


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

def _as_count(x) -> int | None:
    """Non-negative integer or None. Rejects bools, negatives, non-integral floats."""
    if isinstance(x, bool):
        return None
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, int) and x >= 0:
        return x
    return None


def _normalize(inst: Instance, plan) -> tuple[list[tuple[dict, list]], list[dict]]:
    """Parse a plan into per-quarter (assignments, mitigations). Schema problems
    are returned as violations; the offending entry is dropped."""
    viol: list[dict] = []

    def bad(q, detail):
        viol.append({"kind": "schema", "quarter": q, "detail": detail})

    if not isinstance(plan, dict) or not isinstance(plan.get("quarters"), list):
        bad(0, "plan must be an object with a 'quarters' list")
        return [({}, []) for _ in range(inst.n_quarters)], viol

    raw = plan["quarters"]
    if len(raw) > inst.n_quarters:
        bad(0, f"plan has {len(raw)} quarters, horizon is {inst.n_quarters}")
        raw = raw[: inst.n_quarters]

    pids = {p.id for p in inst.projects}
    cids = {c.id for c in inst.crises}
    out = []
    for qi in range(inst.n_quarters):
        q = qi + 1
        entry = raw[qi] if qi < len(raw) else {}
        assign: dict[str, list[int]] = {}
        mits: list[str] = []
        if not isinstance(entry, dict):
            bad(q, "quarter entry must be an object")
            out.append((assign, mits))
            continue

        items = entry.get("assignments") or []
        if isinstance(items, dict):  # lenient: {"P1": {"senior": 1, "junior": 0}}
            items = [{"project": k, **(v if isinstance(v, dict) else {})} for k, v in items.items()]
        if not isinstance(items, list):
            bad(q, "'assignments' must be a list")
            items = []
        for it in items:
            if not isinstance(it, dict) or it.get("project") not in pids:
                bad(q, f"unknown or malformed assignment: {it!r}"[:160])
                continue
            s, j = _as_count(it.get("senior", 0)), _as_count(it.get("junior", 0))
            if s is None or j is None:
                bad(q, f"senior/junior must be non-negative integers: {it!r}"[:160])
                continue
            cur = assign.setdefault(it["project"], [0, 0])
            cur[0] += s
            cur[1] += j

        m = entry.get("mitigations") or []
        if not isinstance(m, list):
            bad(q, "'mitigations' must be a list")
            m = []
        for cid in m:
            if cid not in cids:
                bad(q, f"unknown crisis id: {cid!r}"[:160])
            elif cid not in mits:
                mits.append(cid)
        out.append((assign, mits))
    return out, viol


def verify(inst: Instance, plan) -> Evaluation:
    """Deterministically simulate `plan`. A plan is valid iff it has no violations.

    Order of events in each quarter:
      1. crises scheduled for the quarter strike unless mitigated in an EARLIER quarter
      2. overhead + staffing + mitigation costs are paid (cash < 0 => bankruptcy)
      3. work is performed
      4. completed projects pay max(0, revenue - late_penalty * quarters_late)
    """
    quarters, violations = _normalize(inst, plan)
    warnings: list[str] = []
    projects = {p.id: p for p in inst.projects}
    crises = {c.id: c for c in inst.crises}
    progress = {p.id: 0 for p in inst.projects}
    completed: dict[str, int] = {}
    mitigated: set[str] = set()
    cash = inst.initial_cash
    cash_by_quarter = []

    def add(kind, q, detail):
        violations.append({"kind": kind, "quarter": q, "detail": detail})

    for qi, (assign, mits) in enumerate(quarters):
        q = qi + 1
        senior_cap = inst.senior_cap

        # 1. crises
        for c in inst.crises:
            if c.quarter == q and c.id not in mitigated:
                cash -= c.damage
                senior_cap -= c.senior_loss

        # mitigations run this quarter (effective from the next quarter)
        cost = inst.overhead
        seniors_used = 0
        for cid in mits:
            c = crises[cid]
            cost += c.mitigation_cost
            seniors_used += c.mitigation_seniors
            if cid in mitigated:
                warnings.append(f"Q{q}: {cid} mitigated twice (cost wasted)")
            elif c.quarter <= q:
                warnings.append(f"Q{q}: {cid} mitigated too late (cost wasted)")

        # staffing
        juniors_used = 0
        for pid, (s, j) in assign.items():
            if s == 0 and j == 0:
                continue
            seniors_used += s
            juniors_used += j
            cost += s * inst.senior_cost + j * inst.junior_cost
            if j > inst.supervision_ratio * s:
                add("supervision", q, f"{pid}: {j} juniors with {s} seniors")
            missing = [d for d in projects[pid].deps if d not in completed]
            if missing:
                add("dependency", q, f"{pid} staffed before {','.join(missing)} completed")
            if pid in completed:
                warnings.append(f"Q{q}: {pid} already completed (staffing wasted)")
        if seniors_used > senior_cap:
            add("capacity", q, f"{seniors_used} seniors used, {senior_cap} available")
        if juniors_used > inst.junior_cap:
            add("capacity", q, f"{juniors_used} juniors used, {inst.junior_cap} available")

        # 2. pay
        cash -= cost
        if cash < 0:
            add("bankruptcy", q, f"cash {cash} after paying costs")

        # 3-4. work and completions (dependency-violating work does not count)
        newly = []
        for pid, (s, j) in assign.items():
            p = projects[pid]
            if pid in completed or any(d not in completed for d in p.deps):
                continue
            progress[pid] += s * inst.senior_output + j * inst.junior_output
            if progress[pid] >= p.work:
                newly.append(pid)
        for pid in newly:
            p = projects[pid]
            completed[pid] = q
            cash += max(0, p.revenue - p.late_penalty * max(0, q - p.deadline))

        mitigated.update(mits)
        cash_by_quarter.append(cash)

    return Evaluation(
        valid=not violations,
        profit=cash - inst.initial_cash,
        violations=violations,
        warnings=warnings,
        completed=completed,
        cash_by_quarter=cash_by_quarter,
    )


def score(inst: Instance, ev: Evaluation) -> float:
    """Normalized score: profit / optimal profit for valid plans, 0 otherwise."""
    if not ev.valid or inst.ref_profit <= 0:
        return 0.0
    return max(0.0, ev.profit) / inst.ref_profit


# --------------------------------------------------------------------------- #
# Instance generation
# --------------------------------------------------------------------------- #

DIFFICULTY = {
    "easy":   dict(quarters=3, projects=4, crises=1, cash_frac=0.90),
    "medium": dict(quarters=3, projects=5, crises=2, cash_frac=0.70),
    "hard":   dict(quarters=4, projects=7, crises=3, cash_frac=0.55),
    "expert": dict(quarters=5, projects=10, crises=4, cash_frac=0.45),
}


def _r5(x: float) -> int:
    return int(round(x / 5.0)) * 5


def generate(seed: int, difficulty: str = "medium", attempt: int = 0) -> Instance:
    """Sample an instance. Deterministic in (seed, difficulty, attempt)."""
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(f"htmoa-{difficulty}-{seed}-{attempt}")
    T = cfg["quarters"]
    senior_cap, junior_cap = rng.randint(3, 5), rng.randint(3, 6)
    senior_cost, junior_cost = 30, 12
    overhead = rng.choice([15, 20, 25])
    full_q1 = senior_cap * senior_cost + junior_cap * junior_cost + overhead

    projects = []
    for i in range(cfg["projects"]):
        work = rng.randint(4, 12)
        revenue = _r5(work * rng.uniform(22, 42))
        deadline = rng.randint(1, T)
        deps: tuple[str, ...] = ()
        if i > 0 and rng.random() < 0.45:
            dep = rng.choice(projects)
            deps = (dep.id,)
            deadline = min(T, max(deadline, dep.deadline + 1))
        projects.append(Project(
            id=f"P{i + 1}", work=work, revenue=revenue, deadline=deadline,
            late_penalty=max(5, _r5(revenue * rng.uniform(0.10, 0.30))), deps=deps,
        ))

    crises = []
    for i in range(cfg["crises"]):
        crises.append(Crisis(
            id=f"C{i + 1}", quarter=rng.randint(2, T), damage=rng.randint(8, 18) * 5,
            senior_loss=rng.randint(0, min(2, senior_cap - 1)),
            mitigation_cost=rng.randint(2, 6) * 5, mitigation_seniors=rng.randint(0, 1),
        ))

    return Instance(
        id=f"{difficulty}-{seed:03d}", seed=seed, difficulty=difficulty, n_quarters=T,
        initial_cash=_r5(cfg["cash_frac"] * full_q1), overhead=overhead,
        senior_cap=senior_cap, junior_cap=junior_cap,
        senior_cost=senior_cost, junior_cost=junior_cost,
        senior_output=2, junior_output=1, supervision_ratio=2,
        projects=tuple(projects), crises=tuple(crises),
    )


# --------------------------------------------------------------------------- #
# Natural-language problem statement (the `x` of the paper)
# --------------------------------------------------------------------------- #

def describe(inst: Instance) -> str:
    T = inst.n_quarters
    qs = ", ".join(f"Q{q}" for q in range(1, T + 1))
    proj = "\n".join(
        f"| {p.id} | {p.work} | {p.revenue} | Q{p.deadline} | {p.late_penalty} | "
        f"{', '.join(p.deps) or '-'} |" for p in inst.projects)
    cris = "\n".join(
        f"| {c.id} | Q{c.quarter} | {c.damage} | {c.senior_loss} | {c.mitigation_cost} | "
        f"{c.mitigation_seniors} |" for c in inst.crises)
    example = json.dumps({"quarters": [
        {"assignments": [{"project": "P1", "senior": 1, "junior": 2}], "mitigations": ["C1"]},
        {"assignments": [], "mitigations": []}]})
    return f"""## Planning problem

You must produce a staffing plan for {T} quarters ({qs}). All money is in $k.

### Resources (per quarter)
- Up to {inst.senior_cap} senior engineers. Each assigned senior costs {inst.senior_cost} per quarter and produces {inst.senior_output} work points.
- Up to {inst.junior_cap} junior engineers. Each assigned junior costs {inst.junior_cost} per quarter and produces {inst.junior_output} work point(s).
- Unassigned engineers cost nothing. Engineers are whole numbers.
- Supervision rule: on each project, in each quarter, juniors <= {inst.supervision_ratio} x seniors assigned to that same project.
- Fixed overhead of {inst.overhead} is paid every quarter, whatever you do.

### Cash
- Starting cash: {inst.initial_cash}.
- Order of events inside each quarter:
  1. Crises scheduled for this quarter strike, unless they were mitigated in an EARLIER quarter.
  2. You pay overhead + staffing + mitigation costs. Cash must not be below 0 after this payment (otherwise: bankruptcy, the plan is invalid).
  3. Work is performed.
  4. Projects whose cumulative work reaches the required amount are completed and pay their revenue at the END of the quarter, so that money is only usable from the next quarter.

### Projects (all optional)
| id | work points required | revenue | deadline | late penalty per quarter | depends on |
|---|---|---|---|---|---|
{proj}
- Dependencies: a project may only be staffed in a quarter if all projects it depends on were completed in an earlier quarter.
- A project completed after its deadline pays max(0, revenue - late penalty x quarters late).
- A project not completed by Q{T} pays nothing; money spent on it is lost. Work beyond the requirement is wasted.

### Crises
| id | strikes at start of | cash damage | seniors unavailable that quarter | mitigation cost | seniors needed to mitigate |
|---|---|---|---|---|---|
{cris}
- To mitigate a crisis, list it under "mitigations" in a quarter strictly before it strikes. You pay the mitigation cost in that quarter, and the seniors needed count against that quarter's senior capacity (they cost nothing extra).
- An unmitigated crisis removes the cash damage and reduces that quarter's senior capacity.

### Objective
Maximize profit = final cash after Q{T} minus starting cash, with zero rule violations. An invalid plan is worth nothing.

### Plan format
A JSON object with one entry per quarter, in order. Omit projects you do not staff. Example of the format only:
{example}"""
