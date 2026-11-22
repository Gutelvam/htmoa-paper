import asyncio
import dataclasses

import pytest

from htmoa.env import Crisis, Instance, Project, describe, generate, verify
from htmoa.llm import MockLLM
from htmoa.methods import htmoa_calls, n_calls, parse_spec, parse_state, run_method
import random

from htmoa.solver import anneal, build_instances, solve_reference

INST = Instance(
    id="t", seed=0, difficulty="medium", n_quarters=3, initial_cash=100, overhead=10,
    senior_cap=2, junior_cap=2, senior_cost=30, junior_cost=12, senior_output=2,
    junior_output=1, supervision_ratio=2,
    projects=(Project("P1", work=4, revenue=150, deadline=1, late_penalty=20),
              Project("P2", work=2, revenue=100, deadline=2, late_penalty=10, deps=("P1",))),
    crises=(Crisis("C1", quarter=2, damage=60, senior_loss=1, mitigation_cost=10,
                   mitigation_seniors=0),),
    ref_profit=100)


def plan(*quarters):
    return {"quarters": [{"assignments": [{"project": p, "senior": s, "junior": j}
                                          for p, s, j in a], "mitigations": m}
                         for a, m in quarters]}


def kinds(p):
    return verify(INST, p).violation_kinds


def test_valid_plan_profit():
    ev = verify(INST, plan(([("P1", 2, 0)], ["C1"]), ([("P2", 1, 0)], []), ([], [])))
    assert ev.valid and ev.completed == {"P1": 1, "P2": 2}
    # 100 -10 -60 -10 +150 = 170 ; -10 -30 +100 = 230 ; -10 = 220
    assert ev.cash_by_quarter == [170, 230, 220] and ev.profit == 120


def test_empty_plan_takes_crisis_damage():
    ev = verify(INST, plan())
    assert ev.valid and ev.profit == -30 - 60


def test_capacity_and_supervision():
    assert kinds(plan(([("P1", 3, 0)], []))) == ["capacity"]
    assert kinds(plan(([("P1", 0, 1)], ["C1"]))) == ["supervision"]


def test_unmitigated_crisis_reduces_capacity():
    p = plan(([("P1", 2, 0)], []), ([("P2", 2, 0)], []))
    assert kinds(p) == ["capacity"]


def test_dependency_needs_earlier_quarter():
    assert "dependency" not in kinds(plan(([("P1", 2, 0), ("P2", 0, 0)], []), ([], [])))
    assert kinds(plan(([("P1", 1, 0), ("P2", 1, 0)], ["C1"]))) == ["dependency"]


def test_bankruptcy():
    tight = dataclasses.replace(INST, initial_cash=60)
    assert verify(tight, plan(([("P1", 2, 0)], []))).violation_kinds == ["bankruptcy"]


def test_late_penalty():
    ev = verify(INST, plan(([("P1", 1, 0)], ["C1"]), ([("P1", 1, 0)], [])))
    assert ev.completed == {"P1": 2}
    assert ev.cash_by_quarter[1] == (100 - 50) - 40 + (150 - 20)


def test_schema_violations():
    assert kinds(None) == ["schema"]
    assert kinds({"quarters": [{"assignments": [{"project": "PX", "senior": 1}]}]}) == ["schema"]
    assert kinds({"quarters": [{"assignments": [{"project": "P1", "senior": -1}]}]}) == ["schema"]
    assert kinds({"quarters": [{}] * 4}) == ["schema"]


def test_generation_is_deterministic_and_solvable():
    assert generate(3, "medium") == generate(3, "medium")
    inst = build_instances(1, "medium")[0]
    ref_plan, ref = solve_reference(inst)
    assert ref == inst.ref_profit > 0
    ev = verify(inst, ref_plan)
    assert ev.valid and ev.profit == ref
    for r in range(3):  # a heuristic search must never beat the proven optimum
        assert anneal(inst, 8000, random.Random(r))[1] <= ref
    assert "Plan format" in describe(inst)


@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (3, 2), (2, 3), (4, 1)])
def test_call_count_closed_form(n, k):
    gen, agg = htmoa_calls(n, k)
    assert gen + agg == (n + 1) * (n ** k - 1) // (n - 1)
    assert n_calls(f"htmoa:N={n},K={k}") == gen + agg


def test_parse_state():
    s = parse_state('thinking {"x": 1}\n<rationale>r</rationale>\n<plan>\n```json\n'
                    '{"quarters": []}\n```\n</plan><dissent>- d</dissent>')
    assert s.plan == {"quarters": []} and s.rationale == "r" and s.dissent == "- d"
    assert parse_state("no plan here").plan is None
    with pytest.raises(ValueError):
        parse_spec("htmoa:N=1,K=2")


def test_ablation_options():
    inst = build_instances(1, "easy")[0]
    run = lambda spec: asyncio.run(run_method(MockLLM(), inst, spec, "k"))
    # bottom-up only: N^K leaves, no internal generation
    r = run("htmoa:N=2,K=3,topdown=0")
    assert len(r.nodes) == n_calls("htmoa:N=2,K=3,topdown=0") == 8 + 7
    assert {n.level for n in r.nodes if n.phase == "gen"} == {3}
    # minority report: aggregators carry a scored runner-up plan, leaves do not
    r = run("htmoa:N=2,K=2,delta=2")
    assert all((n.alt_score is not None) == (n.phase == "agg") for n in r.nodes)
    # self-generated splits: below level 1 children take a parent commitment
    r = run("htmoa:N=2,K=2,split=1")
    assert {n.persona for n in r.nodes if n.phase == "gen" and n.level == 2} == {"branch"}
    assert all(n.persona != "branch" for n in r.nodes if n.level == 1)
    for bad in ("htmoa:N=2,K=2,delta=3", "htmoa:N=2,K=2,split=1,topdown=0"):
        with pytest.raises(ValueError):
            parse_spec(bad)
    s = parse_state("<plan>{\"quarters\": []}</plan><alternative>{\"quarters\": [{}]}"
                    "</alternative><branches>\n1. a\n- b\n</branches>")
    assert s.plan == {"quarters": []} and s.alt_plan == {"quarters": [{}]} and s.branches == ["a", "b"]


@pytest.mark.parametrize("spec", ["single", "flat:M=5", "htmoa:N=2,K=3", "htmoa:N=3,K=2,delta=0"])
def test_methods_end_to_end_with_mock(spec):
    inst = build_instances(1, "easy")[0]
    res = asyncio.run(run_method(MockLLM(), inst, spec, "k"))
    assert len(res.nodes) == n_calls(spec)
    assert res.final is res.nodes[-1] and res.final.level == 0
    if spec.startswith("htmoa"):
        n = parse_spec(spec)[1]["N"]
        # Proposition 1: every aggregator reads exactly N children, whatever the depth.
        aggs = [x for x in res.nodes if x.phase == "agg"]
        assert max(x.input_tokens for x in aggs) < 1.5 * min(x.input_tokens for x in aggs)
        assert {x.persona for x in res.nodes if x.phase == "gen"} <= {
            p for p, _ in __import__("htmoa.methods", fromlist=["PERSONAS"]).PERSONAS[:n]}
