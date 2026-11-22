"""Pick the difficulty tier before spending on the full experiment.

Runs the single-agent baseline on a few instances of every tier and reports its
validity and score. A tier only separates the architectures if the single agent
is neither near the ceiling nor at the floor.

    python calibrate.py                       # 6 instances per tier, 24 LLM calls
    python calibrate.py --model claude-sonnet-5-5 --instances 8
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from htmoa.env import DIFFICULTY
from htmoa.llm import AnthropicLLM, FatalLLMError, MockLLM
from htmoa.methods import run_method
from htmoa.solver import build_instances


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["anthropic", "mock"], default="anthropic")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=32000, help="per call, thinking included")
    ap.add_argument("--thinking", choices=["adaptive", "off"], default="adaptive")
    ap.add_argument("--tiers", nargs="+", default=list(DIFFICULTY), choices=list(DIFFICULTY))
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=1000,
                    help="kept apart from the seeds of the main experiment")
    args = ap.parse_args()
    try:
        llm = MockLLM() if args.backend == "mock" else AnthropicLLM(
            model=args.model, max_tokens=args.max_tokens, thinking=args.thinking,
            effort=None if args.effort == "none" else args.effort)
        rows = []
        for tier in args.tiers:
            insts = build_instances(args.instances, tier, args.first_seed)
            res = await asyncio.gather(*[run_method(llm, i, "single", f"cal/{i.id}") for i in insts])
            valid = sum(r.final.valid for r in res) / len(res)
            score = sum(r.final.score for r in res) / len(res)
            tokens = sum(r.final.input_tokens + r.final.output_tokens for r in res) / len(res)
            out_max = max(r.final.output_tokens for r in res)
            secs = sum(r.final.latency_s for r in res) / len(res)
            rows.append((tier, valid, score, tokens))
            print(f"{tier:7s} valid={valid:5.0%}  score={score:.2f}  tokens/call={tokens:,.0f}  "
                  f"max output={out_max:,}  {secs:.0f}s/call", flush=True)
    except FatalLLMError as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return 2
    print(f"calls: {llm.calls}, stop reasons: {dict(llm.stop_reasons)}, "
          f"cost: ${llm.spent_usd:.3f}")
    # Headroom is what matters: the tier whose single-agent score is closest to 0.5.
    best = min(rows, key=lambda r: abs(r[2] - 0.5))
    print(f"\nSuggested tier: --difficulty {best[0]} (single-agent score {best[2]:.2f}).")
    if not 0.2 <= best[2] <= 0.8:
        print("No tier puts the single agent in the 0.2-0.8 band; the architectures may be hard "
              "to separate with this model. Try a different model or effort level.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
