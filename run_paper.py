"""Reproduce the experiment of the paper: every method on every difficulty tier.

    python run_paper.py                       # 30 instances per tier, as in the paper
    python run_paper.py --instances 2         # small pilot first (recommended)
    python run_paper.py --backend mock        # offline plumbing test, no API calls

Each tier is written to results/<name>_<tier>/. The script is resumable: run it again
and it only does what is missing. It makes several passes so that runs which failed on
a network error are retried, then analyzes every tier and the pooled results.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

TIERS = ["easy", "medium", "hard", "expert"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--tiers", nargs="+", default=TIERS, choices=TIERS)
    ap.add_argument("--backend", choices=["anthropic", "mock"], default="anthropic")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--name", default="haiku55", help="prefix of the results directories")
    ap.add_argument("--budget-per-tier", type=float, default=45.0, help="spending cap in USD")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="simultaneous API calls; lower it if you hit rate limits")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    dirs = [f"results/{args.name}_{t}" for t in args.tiers]
    for _ in range(args.passes):
        for tier, out in zip(args.tiers, dirs):
            code = subprocess.call([
                sys.executable, "run_experiment.py", "--suite", "paper", "--difficulty", tier,
                "--instances", str(args.instances), "--backend", args.backend,
                "--model", args.model, "--thinking", "off", "--effort", "low",
                "--max-tokens", "8000", "--concurrency", str(args.concurrency),
                "--parallel-runs", "10", "--budget-usd", str(args.budget_per_tier), "--out", out])
            if code == 2:   # configuration error (credentials, model name): do not go on
                return 2
    for out in dirs:
        subprocess.call([sys.executable, "analyze.py", out], stdout=subprocess.DEVNULL)
    return subprocess.call([sys.executable, "pooled_analysis.py", *dirs, "--out", "analysis"],
                           stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    sys.exit(main())
