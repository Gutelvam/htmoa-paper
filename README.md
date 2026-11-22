# HT-MoA: hierarchical vs. flat aggregation in Mixture-of-Agents

Code, benchmark and results for the paper *Selection, Not Synthesis: Why Hierarchical
Aggregation Keeps Scaling Mixture-of-Agents Where Flat Aggregation Saturates*.

The repository contains:

- a **planning benchmark with a known optimum**: an instance generator, an exact
  mixed-integer solver and a deterministic verifier that scores any plan;
- three **inference architectures** built on one LLM: a single agent, Flat-MoA
  (M workers + 1 aggregator) and HT-MoA (a tree of width N and depth K, expanded
  top-down and aggregated level by level);
- the **experiment runner and analysis** that produce every table and figure of the paper;
- the **saved results** of the paper's experiment: 3,000 runs, 49,440 model calls,
  including the full text of every call.

## Main result

Scores are profit as a fraction of the optimal profit (0 for an invalid plan), averaged
over 120 instances in four difficulty tiers, with `claude-haiku-5-5`.

| LLM calls | Flat-MoA | HT-MoA (N=2) | Difference [95% CI] |
|---|---|---|---|
| 1 (single agent) | 0.28 | | |
| 3 | 0.39 | 0.38 (K=1) | -0.01 [-0.07, +0.05] |
| 9 | 0.45 | 0.47 (K=2) | +0.02 [-0.04, +0.07] |
| 21 | 0.49 | 0.54 (K=3) | +0.05 [+0.01, +0.10] |
| 45 | 0.47 | 0.57 (K=4) | +0.10 [+0.05, +0.16] |

Aggregators return an exact copy of one of their inputs in 86% of 12,360 aggregation
steps. That copy is a best input in 82% of steps with 2 candidates and in 28% with 44,
which is why adding flat workers stops helping and a tree of small aggregations does not.

## Requirements

- Python 3.10 or newer
- `pip install -r requirements.txt`
- An Anthropic API key, only if you want to run new experiments. Everything else
  (tests, the benchmark, re-analysis of the saved results, the offline mock backend)
  works without a key.

### Providing your API key

Use either of the following. No key is stored in this repository.

- Environment variable:
  - Linux/macOS: `export ANTHROPIC_API_KEY=...`
  - Windows PowerShell: `$env:ANTHROPIC_API_KEY = "..."`
- Or a local file: copy `.env.example` to `.env` and put your key in it. `.env` is
  read automatically and is ignored by git.

## Quick start (no API key needed)

```
pip install -r requirements.txt
python -m pytest tests -q

# re-create every table and figure of the paper from the saved results
python pooled_analysis.py results/haiku55_easy results/haiku55_medium results/haiku55_hard results/haiku55_expert --out analysis

# run the whole pipeline offline with a mock model (numbers are meaningless)
python run_experiment.py --backend mock --instances 3 --out results/smoke
python analyze.py results/smoke
```

`analysis/pooled_summary.md` then holds all pooled tables; `analysis/tab_*.tex` are the
LaTeX tables and `analysis/fig_*.png` the figures used in the paper.

## Reproducing the experiment (needs an API key)

```
python calibrate.py --thinking off --effort low --max-tokens 8000   # single agent on every tier, 24 calls
python run_paper.py --instances 2                                    # pilot: about 3,300 calls
python run_paper.py                                                  # full experiment: 49,440 calls
```

`run_paper.py` runs the 25 methods of the paper on 30 instances of each of the four
tiers, with the configuration used in the paper (`claude-haiku-5-5`, hidden reasoning
off, effort `low`, at most 8,000 output tokens per call). It is resumable, retries runs
that failed on a network error, stops starting new runs when a tier reaches the spending
cap (`--budget-per-tier`, default 45 USD), and finishes by running the analysis. Our run
cost 44 USD in total and took about four and a half hours at 20 concurrent calls.

Model outputs are sampled, so a new run will not reproduce the saved numbers exactly.
If you hit rate limits, lower `--concurrency`.

To run a subset yourself:

```
python run_experiment.py --dry-run --suite paper                       # list methods and call counts
python run_experiment.py --suite main --difficulty hard --instances 10 --thinking off --effort low --out results/mine
python run_experiment.py --methods single flat:M=8 htmoa:N=2,K=3 --difficulty medium --thinking off --effort low --out results/mine
python analyze.py results/mine
```

## Methods

Method specs: `single`, `flat:M=8`, `htmoa:N=2,K=3`. Each HT-MoA configuration is compared
with the Flat-MoA that makes the same number of calls, `M = C(N,K) - 1`, where
`C(N,K) = (N+1)(N^K - 1)/(N - 1)`.

HT-MoA options (ablations), added to the spec as in `htmoa:N=2,K=3,delta=0`:

| Option | Meaning |
|---|---|
| `delta=0` / `1` / `2` | aggregators pass up only a plan / plan + dissent notes (default) / plan + a complete runner-up plan + notes |
| `topdown=0` | bottom-up only: N^K independent leaves, no parent-conditioned expansion |
| `split=1` | below level 1, each parent writes N mutually exclusive commitments and each child must honor one |
| `personas=0` | no persona text |

Suites (`--suite`): `main` (baselines and tree shapes), `depth` (N=2, K=1..4 with matched
flat baselines), `ablation`, `paper` (all of them, 25 methods).

## The benchmark

A plan staffs software projects with senior and junior engineers over several quarters,
under capacity, supervision, precedence and solvency constraints, with scheduled crises
that can be mitigated in advance. See `htmoa/env.py` for the generator, the verifier and
the problem statement shown to the model, and `htmoa/solver.py` for the exact solver.

```python
from htmoa.env import describe, verify, score
from htmoa.solver import build_instances, solve_reference

inst = build_instances(1, "medium")[0]     # deterministic in (tier, seed)
print(describe(inst))                      # the text given to the model
plan, optimum = solve_reference(inst)      # exact optimum
result = verify(inst, plan)                # .valid, .profit, .violations
print(score(inst, result))                 # 1.0
```

Tiers: `easy`, `medium`, `hard`, `expert` (3 to 5 quarters, 4 to 10 projects, 1 to 4 crises).

## Repository layout

| Path | Role |
|---|---|
| `htmoa/env.py` | Benchmark: instance generator, verifier, problem statement |
| `htmoa/solver.py` | Exact MILP reference solver, frozen instance sets |
| `htmoa/methods.py` | Single agent, Flat-MoA, HT-MoA, prompts, personas, parsing |
| `htmoa/llm.py` | Anthropic API backend (with retries and cost tracking) and offline mock backend |
| `run_experiment.py` | Runs methods x instances; resumable; logs progress and spend |
| `run_paper.py` | The full experiment of the paper on all tiers, then the analysis |
| `calibrate.py` | Single-agent score on every tier, to check headroom before a run |
| `analyze.py` | Tables and figures for one results directory |
| `pooled_analysis.py` | Pooled tables and figures of the paper, from several results directories |
| `tests/` | Verifier, solver, call-count and end-to-end tests |
| `results/haiku55_<tier>/` | Saved results of the paper's experiment |

Each results directory holds `instances.json` (the instances with their optimal profit),
`results.jsonl` (one row per run, with the score and token counts of every node),
`traces.jsonl.gz` (the full text of every model call), `config.json`, and the per-tier
`summary.md`, `summary.csv` and figures.

## Metrics

- **Score**: profit / optimal profit for a valid plan, 0 otherwise.
- **Validity**: no schema, capacity, supervision, precedence or bankruptcy violation.
- **Aggregation outcome**: each aggregator output is compared with the best plan it read:
  synthesis (better), selection (equal), loss (worse); exact copies, corruption of
  validity and regret are also reported, by fan-in and by tree level.
- **Cost**: calls, input and output tokens, estimated USD, largest aggregator prompt.

## License

Code and saved results are released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@misc{fernandes2026htmoa,
  author       = {Fernandes, Gutelvam Rodrigues and Ferreira, Lu{'i}s Gonzaga Martins},
  title        = {{Selection, Not Synthesis: Why Hierarchical Aggregation Keeps Scaling
                  Mixture-of-Agents Where Flat Aggregation Saturates}},
  year         = {2026},
  howpublished = {https://github.com/Gutelvam/htmoa-paper},
  note         = {Preprint}
}
```
