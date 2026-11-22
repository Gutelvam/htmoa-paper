"""Inference architectures: Single-Agent CoT, Flat-MoA and HT-MoA (Section 3).

Every node's plan is scored with the deterministic verifier for analysis. Those
scores are never shown to any agent.

HT-MoA options (all ablatable from the method spec):
  delta    0 = aggregators pass up only their plan
           1 = plan + free-text dissent notes (the draft's Delta protocol)
           2 = plan + a complete runner-up plan ("minority report") + notes
  topdown  1 = expand-then-reduce: every level generates, children refine the parent draft
           0 = bottom-up only: N^K independent leaves, then level-by-level aggregation
  split    0 = fixed personas at every level
           1 = below level 1, each parent writes N mutually exclusive commitments and
               each child must honor one (the tree partitions the decision space)
  personas 0 = no persona text at all
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from .env import Instance, describe, score, verify

SYSTEM = ("You are the Chief Operating Officer of a software company. You produce staffing "
          "plans that strictly obey the stated rules, checking cash and capacity quarter by "
          "quarter before committing to a plan.")

# Cognitive perturbations phi_i (Section 3.2). The first four are those named in the draft.
PERSONAS = [
    ("financial conservation", "Protect cash above all: keep a safety margin every quarter and "
     "prefer projects that pay back early."),
    ("proactive risk mitigation", "Treat the scheduled crises as the main threat: decide first "
     "how each one is handled, then plan delivery around that."),
    ("resource efficiency", "Maximize value per unit of cost: use the cheapest staffing mix the "
     "supervision rule allows and waste no work points."),
    ("adversarial stress-testing", "Assume the obvious plan hides a rule violation: hunt for "
     "capacity, dependency and cash-flow failures and build a plan that survives them."),
    ("deadline-driven delivery", "Prioritize finishing projects on or before their deadlines to "
     "avoid late penalties."),
    ("revenue growth", "Aim for the highest total revenue: fit in as many high-value projects "
     "as the rules allow."),
]
NEUTRAL = ("none", "")

# Character caps that bound the size of a state passed between nodes (L_max in Prop. 1).
CAP_RATIONALE, CAP_PLAN, CAP_DISSENT, CAP_BRANCH = 900, 3000, 900, 300

_RATIONALE = "<rationale>\nAt most 120 words: {what}.\n</rationale>"
_PLAN = "<plan>\nthe JSON plan\n</plan>"
_BRANCHES = ("<branches>\nExactly {n} lines, one commitment per line. Each is a concrete strategic "
             "decision that a planner refining your draft will be forced to take (for example which "
             "projects to pursue or drop, when to mitigate which crisis, how to sequence cash). The "
             "{n} commitments must be mutually exclusive and together cover the options you find "
             "most promising.\n</branches>")
_ALTERNATIVE = ("<alternative>\nA second complete JSON plan: the strongest plan that differs "
                "materially from your main plan (a different project portfolio or crisis handling). "
                "It must also obey every rule.\n</alternative>")
_DISSENT = ("<dissent>\nUp to 5 short bullets for the reviewer above you: unresolved trade-offs, "
            "failure modes, and promising ideas from the candidates that you did not adopt.\n</dissent>")


def gen_format(n_branches: int = 0) -> str:
    blocks = [_RATIONALE.format(what="the key decisions and the main risk of this plan"), _PLAN]
    if n_branches:
        blocks.append(_BRANCHES.format(n=n_branches))
    return ("Work through the problem, then end your answer with exactly these blocks:\n"
            + "\n".join(blocks))


def agg_format(delta: int) -> str:
    blocks = [_RATIONALE.format(what="what you kept from which candidate and why"), _PLAN]
    if delta == 2:
        blocks.append(_ALTERNATIVE)
    if delta >= 1:
        blocks.append(_DISSENT)
    return "End your answer with exactly these blocks:\n" + "\n".join(blocks)


@dataclass
class NodeState:
    """The state s (or s-hat) of a node: what is passed along tree edges."""
    rationale: str = ""
    plan_text: str = ""
    dissent: str = ""
    plan: dict | None = None
    alt_text: str = ""
    alt_plan: dict | None = None
    branches: list[str] = field(default_factory=list)


@dataclass
class Node:
    node_id: str
    phase: str          # "gen" (top-down expansion) or "agg" (bottom-up aggregation)
    level: int
    index: int          # 1-based index j within the level
    persona: str
    state: NodeState
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str
    valid: bool
    profit: int
    score: float
    violations: list[str]
    alt_valid: bool | None = None   # runner-up plan (delta=2 aggregators only)
    alt_score: float | None = None

    def record(self, with_text: bool = False) -> dict:
        d = {k: getattr(self, k) for k in (
            "node_id", "phase", "level", "index", "persona", "input_tokens", "output_tokens",
            "latency_s", "stop_reason", "valid", "profit", "score", "violations",
            "alt_valid", "alt_score")}
        d["parsed"] = self.state.plan is not None
        d["n_branches"] = len(self.state.branches)
        if with_text:
            d["text"] = self.text
        return d


@dataclass
class RunResult:
    final: Node
    nodes: list[Node] = field(default_factory=list)
    wall_s: float = 0.0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _tag(text: str, name: str) -> str:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text, flags=re.S | re.I)
    return m[-1].strip() if m else ""


def extract_plan(text: str) -> dict | None:
    """Last JSON object in `text` that has a 'quarters' key."""
    dec = json.JSONDecoder()
    found, i = None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "quarters" in obj:
            found = obj
        i = text.find("{", end)
    return found


def parse_state(text: str) -> NodeState:
    plan_text, alt_text = _tag(text, "plan"), _tag(text, "alternative")
    # The main plan is read from its own block; without one, from the text before any alternative.
    fallback = re.split(r"<alternative>", text, flags=re.I)[0]
    branches = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip()
                for ln in _tag(text, "branches").splitlines()]
    return NodeState(
        rationale=_tag(text, "rationale"), plan_text=plan_text, dissent=_tag(text, "dissent"),
        plan=extract_plan(plan_text or fallback), alt_text=alt_text,
        alt_plan=extract_plan(alt_text) if alt_text else None,
        branches=[b[:CAP_BRANCH] for b in branches if b])


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def _render(state: NodeState, delta: int) -> str:
    parts = [f"Rationale: {state.rationale[:CAP_RATIONALE] or '(none given)'}",
             f"Plan: {state.plan_text[:CAP_PLAN] or '(no plan block was produced)'}"]
    if delta == 2 and state.alt_text:
        parts.append(f"Runner-up plan kept from below: {state.alt_text[:CAP_PLAN]}")
    if delta >= 1 and state.dissent:
        parts.append(f"Unresolved notes passed up from below:\n{state.dissent[:CAP_DISSENT]}")
    return "\n".join(parts)


def gen_prompt(inst: Instance, persona: tuple[str, str], parent: NodeState | None,
               directive: str | None = None, n_branches: int = 0) -> str:
    out = [describe(inst)]
    if directive:
        out.append(f"## Your assigned commitment\n{directive}\nYour plan must honor this "
                   "commitment while obeying every rule and pursuing the objective. If the "
                   "commitment cannot be honored without breaking a rule, stay as close to it as "
                   "the rules allow.")
    elif persona[1]:
        out.append(f"## Your lens: {persona[0]}\n{persona[1]} The plan must still obey every "
                   f"rule and pursue the objective.")
    if parent is not None:
        out.append("## Draft from the planner above you\n" + _render(parent, 0) +
                   "\n\nThis draft may contain rule violations or miss better options. Produce "
                   "your own complete plan: fix, refine, or depart from it.")
    out.append("## Output\n" + gen_format(n_branches))
    return "\n\n".join(out)


def agg_prompt(inst: Instance, children: list[NodeState], delta: int) -> str:
    cands = "\n\n".join(f"### Candidate {i + 1}\n{_render(c, delta)}"
                        for i, c in enumerate(children))
    task = (f"## Your task\nBelow are {len(children)} candidate plans from planners who each "
            "explored a different direction. Any of them may violate rules. Check each one by "
            "simulating cash, capacity, supervision and dependencies quarter by quarter. Then "
            "write the single best valid plan: adopt the strongest candidate, repair it, or "
            "combine ideas from several.")
    return "\n\n".join([describe(inst), task, "## Candidates\n" + cands,
                        "## Output\n" + agg_format(delta)])


# --------------------------------------------------------------------------- #
# Node execution
# --------------------------------------------------------------------------- #

async def _call(llm, inst: Instance, run_key: str, *, phase: str, level: int, index: int,
                persona: str, prompt: str, meta: dict) -> Node:
    node_id = f"{phase}-L{level}-{index}"
    comp = await llm.complete(SYSTEM, prompt, phase=phase,
                              meta={"inst": inst, "key": f"{run_key}/{node_id}", **meta})
    state = parse_state(comp.text)
    ev = verify(inst, state.plan)
    alt = verify(inst, state.alt_plan) if state.alt_text else None
    return Node(node_id=node_id, phase=phase, level=level, index=index, persona=persona,
                state=state, text=comp.text, input_tokens=comp.input_tokens,
                output_tokens=comp.output_tokens, latency_s=comp.latency_s,
                stop_reason=comp.stop_reason, valid=ev.valid, profit=ev.profit,
                score=score(inst, ev), violations=ev.violation_kinds,
                alt_valid=alt.valid if alt else None,
                alt_score=score(inst, alt) if alt else None)


async def _generate(llm, inst, run_key, level, index, persona, parent: Node | None,
                    directive: str | None = None, n_branches: int = 0) -> Node:
    pstate = parent.state if parent else None
    return await _call(
        llm, inst, run_key, phase="gen", level=level, index=index,
        persona="branch" if directive else persona[0],
        prompt=gen_prompt(inst, persona, pstate, directive, n_branches),
        meta={"parent": pstate, "n_branches": n_branches})


async def _aggregate(llm, inst, run_key, level, index, children: list[Node], delta: int) -> Node:
    states = [c.state for c in children]
    return await _call(llm, inst, run_key, phase="agg", level=level, index=index, persona="",
                       prompt=agg_prompt(inst, states, delta),
                       meta={"children": states, "delta": delta})


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #

def htmoa_calls(n: int, k: int, topdown: bool = True) -> tuple[int, int]:
    """(generation calls, aggregation calls). With the top-down phase:
    (sum_{i=1..K} N^i, sum_{i=0..K-1} N^i); bottom-up only: (N^K, sum_{i=0..K-1} N^i)."""
    gen = sum(n ** i for i in range(1, k + 1)) if topdown else n ** k
    return gen, sum(n ** i for i in range(k))


async def run_single(llm, inst: Instance, run_key: str) -> RunResult:
    t0 = time.perf_counter()
    node = await _generate(llm, inst, run_key, 0, 1, NEUTRAL, None)
    return RunResult(final=node, nodes=[node], wall_s=time.perf_counter() - t0)


async def run_flat(llm, inst: Instance, run_key: str, M: int, personas: bool = True) -> RunResult:
    """M parallel workers + 1 global aggregator that reads all M outputs."""
    t0 = time.perf_counter()
    workers = await asyncio.gather(*[
        _generate(llm, inst, run_key, 1, j,
                  PERSONAS[(j - 1) % len(PERSONAS)] if personas else NEUTRAL, None)
        for j in range(1, M + 1)])
    root = await _aggregate(llm, inst, run_key, 0, 1, list(workers), delta=0)
    return RunResult(final=root, nodes=[*workers, root], wall_s=time.perf_counter() - t0)


async def run_htmoa(llm, inst: Instance, run_key: str, N: int, K: int, delta: int = 1,
                    personas: bool = True, topdown: bool = True, split: bool = False) -> RunResult:
    """Top-down expansion (Eq. 2) followed by bottom-up level-by-level aggregation (Eq. 3)."""
    if N > len(PERSONAS) and personas:
        raise ValueError(f"N={N} needs more than the {len(PERSONAS)} defined personas")
    t0 = time.perf_counter()
    nodes: list[Node] = []

    def persona(j: int):
        return PERSONAS[(j - 1) % N] if personas else NEUTRAL

    def directive(parent: Node | None, j: int) -> str | None:
        # Child i of a parent takes the parent's i-th commitment; fall back to the persona
        # when the parent did not produce N of them.
        if not split or parent is None or len(parent.state.branches) < N:
            return None
        return parent.state.branches[(j - 1) % N]

    # Top-down: node (k, j) is conditioned on its parent (k-1, ceil(j/N)) and persona
    # phi_i with i = ((j-1) mod N) + 1. The root holds the empty state s_{0,1}.
    level: list[Node | None] = [None]
    for k in range(1, K + 1) if topdown else [K]:
        level = list(await asyncio.gather(*[
            _generate(llm, inst, run_key, k, j, persona(j),
                      level[(j - 1) // N] if topdown else None,
                      directive(level[(j - 1) // N] if topdown else None, j),
                      N if split and k < K else 0)
            for j in range(1, N ** k + 1)]))
        nodes.extend(level)

    # Bottom-up: s-hat_{K,j} = s_{K,j}; aggregator (k, j) reads only its N direct children.
    agg: list[Node] = level  # type: ignore[assignment]
    for k in range(K - 1, -1, -1):
        agg = list(await asyncio.gather(*[
            _aggregate(llm, inst, run_key, k, j, agg[(j - 1) * N: j * N], delta)
            for j in range(1, N ** k + 1)]))
        nodes.extend(agg)

    assert len(nodes) == sum(htmoa_calls(N, K, topdown))
    return RunResult(final=agg[0], nodes=nodes, wall_s=time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Method specs: "single", "flat:M=8", "htmoa:N=2,K=2", "htmoa:N=2,K=2,delta=2,split=1"
# --------------------------------------------------------------------------- #

HTMOA_DEFAULTS = {"delta": 1, "personas": 1, "topdown": 1, "split": 0}


def parse_spec(spec: str) -> tuple[str, dict]:
    kind, _, rest = spec.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        params[k.strip()] = int(v)
    if kind == "single" and not params:
        return kind, params
    if kind == "flat" and "M" in params and set(params) <= {"M", "personas"}:
        return kind, params
    if (kind == "htmoa" and {"N", "K"} <= set(params) <= {"N", "K", *HTMOA_DEFAULTS}
            and params["N"] >= 2 and params["K"] >= 1 and params.get("delta", 1) in (0, 1, 2)
            and not (params.get("split") and not params.get("topdown", 1))):
        return kind, params
    raise ValueError(f"bad method spec: {spec!r}")


def base_spec(spec: str) -> str | None:
    """For an HT-MoA ablation, the default configuration with the same N and K."""
    kind, p = parse_spec(spec)
    if kind != "htmoa" or set(p) == {"N", "K"}:
        return None
    return f"htmoa:N={p['N']},K={p['K']}"


def n_calls(spec: str) -> int:
    kind, p = parse_spec(spec)
    if kind == "single":
        return 1
    if kind == "flat":
        return p["M"] + 1
    return sum(htmoa_calls(p["N"], p["K"], bool(p.get("topdown", 1))))


async def run_method(llm, inst: Instance, spec: str, run_key: str) -> RunResult:
    kind, p = parse_spec(spec)
    if kind == "single":
        return await run_single(llm, inst, run_key)
    if kind == "flat":
        return await run_flat(llm, inst, run_key, p["M"], bool(p.get("personas", 1)))
    o = {**HTMOA_DEFAULTS, **p}
    return await run_htmoa(llm, inst, run_key, o["N"], o["K"], o["delta"], bool(o["personas"]),
                           bool(o["topdown"]), bool(o["split"]))
