"""Run the HT-MoA evaluation protocol (Section 4 of the draft).

Examples
    python run_experiment.py --backend mock --instances 5 --out results/smoke   # offline plumbing test
    python run_experiment.py --dry-run                                          # show the plan, no calls
    python run_experiment.py --instances 20 --out results/main                  # real run (needs API key)

Runs are appended to <out>/results.jsonl as they finish; re-running the same
command resumes and skips what is already done. Progress, spend and problems
are logged to the console and to <out>/run.log.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from htmoa.llm import AnthropicLLM, FatalLLMError, MockLLM, cost_usd
from htmoa.methods import n_calls, parse_spec, run_method
from htmoa.solver import load_or_build

log = logging.getLogger("htmoa")

# Each HT-MoA configuration is paired with a Flat-MoA using the same number of LLM
# calls (M = C_total - 1). Note htmoa:N=4,K=1 and flat:M=4 share the same topology.
DEFAULT_METHODS = [
    "single",
    "htmoa:N=4,K=1", "flat:M=4",
    "htmoa:N=2,K=2", "flat:M=8",
    "htmoa:N=3,K=2", "flat:M=15",
    "htmoa:N=2,K=3", "flat:M=20",
]


# Ablations of the HT-MoA mechanisms, each compared with the default configuration of
# the same N and K (see the option list at the top of htmoa/methods.py).
ABLATION_METHODS = [f"htmoa:N={n},K={k}{extra}" for n, k in ((3, 2), (2, 3))
                    for extra in ("", ",topdown=0", ",delta=0", ",delta=2", ",split=1",
                                  ",delta=2,split=1", ",personas=0")]
# Depth sweep at fixed width for RQ3: K = 1..4 with N = 2 (3, 9, 21, 45 calls), each
# with its call-matched Flat-MoA.
DEPTH_METHODS = ["single"] + [m for k, flat in ((1, 2), (2, 8), (3, 20), (4, 44))
                              for m in (f"htmoa:N=2,K={k}", f"flat:M={flat}")]
SUITES = {"main": DEFAULT_METHODS, "ablation": ABLATION_METHODS, "depth": DEPTH_METHODS,
          "all": DEFAULT_METHODS + [m for m in ABLATION_METHODS if m not in DEFAULT_METHODS]}
# Everything the paper needs, each method once.
SUITES["paper"] = list(dict.fromkeys(DEFAULT_METHODS + DEPTH_METHODS + ABLATION_METHODS))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["anthropic", "mock"], default="anthropic")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--suite", choices=list(SUITES), default="main",
                    help="main: baselines and HT-MoA shapes; ablation: HT-MoA mechanisms; all: both")
    ap.add_argument("--methods", nargs="+", default=None,
                    help='overrides --suite, e.g. single flat:M=8 htmoa:N=2,K=2,delta=2')
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], default="medium")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per (instance, method)")
    ap.add_argument("--out", type=Path, default=Path("results/main"))
    ap.add_argument("--concurrency", type=int, default=8, help="simultaneous API calls")
    ap.add_argument("--parallel-runs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8000, help="per call, thinking included")
    ap.add_argument("--effort", default="medium",
                    help="output_config.effort; use 'none' for models without it")
    ap.add_argument("--temp-exp", type=float, default=None,
                    help="tau_exp; only for models that accept temperature")
    ap.add_argument("--temp-agg", type=float, default=None, help="tau_agg; same caveat")
    ap.add_argument("--thinking", choices=["adaptive", "off"], default="adaptive",
                    help="off: no hidden reasoning, every node reasons in visible text only")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop starting new runs once this directory has cost this much")
    ap.add_argument("--verbose", action="store_true", help="log every LLM call to run.log")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


class BudgetStop(Exception):
    """Raised by a run that was not started because the spending cap was reached."""


def setup_logging(out: Path, verbose: bool) -> None:
    """Console at INFO; <out>/run.log keeps everything, per-call lines with --verbose."""
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    file = logging.FileHandler(out / "run.log", encoding="utf-8")
    file.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    for h in (file, console):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.DEBUG)


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


async def main() -> int:
    args = parse_args()
    args.methods = args.methods or SUITES[args.suite]
    for m in args.methods:
        parse_spec(m)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out, args.verbose)
    results_path, traces_path = args.out / "results.jsonl", args.out / "traces.jsonl"

    log.info("command: %s", " ".join(sys.argv))
    log.info("building/loading %d '%s' instances", args.instances, args.difficulty)
    instances = load_or_build(args.out / "instances.json", args.instances, args.difficulty)

    done, prior_cost = set(), 0.0
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["instance_id"], r["method"], r["rep"]))
            prior_cost += r.get("cost_usd") or 0.0
    todo = [(inst, m, rep) for inst in instances for m in args.methods
            for rep in range(args.reps) if (inst.id, m, rep) not in done]
    calls = sum(n_calls(m) for _, m, _ in todo)
    log.info("%d runs to do (%d already done, $%.2f already spent), %d LLM calls, backend=%s, "
             "model=%s", len(todo), len(done), prior_cost, calls, args.backend,
             args.model if args.backend == "anthropic" else "mock")
    for m in args.methods:
        log.info("  %-32s %3d calls/run", m, n_calls(m))
    if args.budget_usd is not None:
        log.info("spending cap for this directory: $%.2f", args.budget_usd)
    if args.dry_run or not todo:
        return 0

    try:
        llm = (MockLLM() if args.backend == "mock" else AnthropicLLM(
            model=args.model, max_tokens=args.max_tokens,
            effort=None if args.effort == "none" else args.effort,
            concurrency=args.concurrency, temp_exp=args.temp_exp, temp_agg=args.temp_agg,
            thinking=args.thinking))
    except FatalLLMError as e:
        log.error("ABORTED: %s", e)
        return 2
    (args.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1),
        encoding="utf-8")

    sem = asyncio.Semaphore(args.parallel_runs)

    def spent() -> float:
        return prior_cost + llm.spent_usd

    async def one(inst, spec, rep):
        async with sem:
            # Checked when the run is about to start, so runs already in flight finish.
            if args.budget_usd is not None and spent() >= args.budget_usd:
                raise BudgetStop
            res = await run_method(llm, inst, spec, f"{inst.id}/{spec}/{rep}")
        kind, p = parse_spec(spec)
        f, nodes = res.final, res.nodes
        tin, tout = sum(n.input_tokens for n in nodes), sum(n.output_tokens for n in nodes)
        row = {
            "instance_id": inst.id, "method": spec, "kind": kind, "params": p, "rep": rep,
            "model": llm.model, "valid": f.valid, "profit": f.profit, "score": f.score,
            "ref_profit": inst.ref_profit, "violations": f.violations,
            "parsed": f.state.plan is not None,
            "calls": len(nodes), "input_tokens": tin, "output_tokens": tout,
            "total_tokens": tin + tout,
            "max_agg_input_tokens": max((n.input_tokens for n in nodes if n.phase == "agg"), default=0),
            "wall_s": res.wall_s, "cost_usd": cost_usd(llm.model, tin, tout),
            "nodes": [n.record() for n in nodes],
        }
        trace = {"instance_id": inst.id, "method": spec, "rep": rep,
                 "nodes": [n.record(with_text=True) for n in nodes]}
        return row, trace

    tasks = [asyncio.create_task(one(*t)) for t in todo]
    n_done = failed = skipped = unparsed = 0
    t0 = time.perf_counter()
    try:
        with results_path.open("a", encoding="utf-8") as fr, traces_path.open("a", encoding="utf-8") as ft:
            for fut in asyncio.as_completed(tasks):
                try:
                    row, trace = await fut
                except FatalLLMError:
                    raise
                except BudgetStop:
                    skipped += 1
                    continue
                except Exception as e:  # transient failure: leave the run for a later resume
                    failed += 1
                    log.error("run failed (%s: %s)", type(e).__name__, e)
                    continue
                fr.write(json.dumps(row) + "\n"); fr.flush()
                ft.write(json.dumps(trace) + "\n"); ft.flush()
                n_done += 1
                unparsed += not row["parsed"]
                elapsed = time.perf_counter() - t0
                eta = elapsed / n_done * (len(todo) - n_done - failed - skipped)
                log.info("[%d/%d] %s %-30s valid=%-5s score=%.2f tokens=%d | spent $%.2f | "
                         "elapsed %s, eta %s", n_done, len(todo), row["instance_id"],
                         row["method"], row["valid"], row["score"], row["total_tokens"],
                         spent(), _hms(elapsed), _hms(eta))
    except FatalLLMError as e:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.error("ABORTED: %s", e)
        return 2

    log.info("finished in %s: %d runs written, %d failed, %d not started (spending cap), "
             "%d final plans unparseable", _hms(time.perf_counter() - t0), n_done, failed,
             skipped, unparsed)
    log.info("calls this session: %d, stop reasons: %s, spent in this directory: $%.2f",
             llm.calls, dict(llm.stop_reasons), spent())
    if failed or skipped:
        log.info("re-run the same command to do the remaining runs"
                 + (" (raise --budget-usd first)" if skipped else ""))
    log.info("next: python analyze.py %s", args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
