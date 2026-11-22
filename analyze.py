"""Analyze a results directory: tables for RQ1-RQ3 and three figures.

    python analyze.py results/main

Writes summary.md, summary.csv and fig_*.png into the same directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from htmoa.methods import base_spec, n_calls

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY_COLOR = {"single": "#2a78d6", "flat": "#eb6834", "htmoa": "#1baf7a"}
FAMILY_LABEL = {"single": "Single-agent CoT", "flat": "Flat-MoA", "htmoa": "HT-MoA"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [json.loads(l) for l in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    nodes = [{"instance_id": r["instance_id"], "method": r["method"], "kind": r["kind"],
              "rep": r["rep"], **r["params"], **n} for r in rows for n in r["nodes"]]
    return pd.DataFrame(rows).drop(columns=["nodes"]), pd.DataFrame(nodes)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, g in df.groupby("method", sort=False):
        per_inst = g.groupby("instance_id")["score"].mean().to_numpy()
        lo, hi = boot_ci(per_inst)
        out.append({
            "method": m, "calls": n_calls(m), "runs": len(g), "instances": len(per_inst),
            "valid_%": 100 * g["valid"].mean(),
            "score": per_inst.mean(), "score_lo": lo, "score_hi": hi,
            "profit_valid": g.loc[g["valid"], "profit"].mean(),
            "tokens": g["total_tokens"].mean(), "out_tokens": g["output_tokens"].mean(),
            "max_agg_ctx": g["max_agg_input_tokens"].max(),
            "wall_s": g["wall_s"].mean(),
            "cost_usd": g["cost_usd"].mean() if g["cost_usd"].notna().all() else float("nan"),
        })
    return pd.DataFrame(out).sort_values(["calls", "method"]).reset_index(drop=True)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """HT-MoA vs single and vs the call-matched Flat-MoA, paired by instance."""
    piv = df.groupby(["instance_id", "method"])["score"].mean().unstack()
    out = []
    for m in [c for c in piv.columns if c.startswith("htmoa")]:
        # an ablation is compared with its default configuration; a default
        # configuration with the single agent and the call-matched Flat-MoA
        bases = [base_spec(m)] if base_spec(m) else ["single", f"flat:M={n_calls(m) - 1}"]
        for base in bases:
            if base not in piv.columns:
                continue
            d = (piv[m] - piv[base]).dropna().to_numpy()
            if len(d) == 0:
                continue
            lo, hi = boot_ci(d)
            p = stats.wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else float("nan")
            out.append({"method": m, "baseline": base, "n": len(d), "mean_diff": d.mean(),
                        "ci_lo": lo, "ci_hi": hi, "wins": int((d > 0).sum()),
                        "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                        "wilcoxon_p": p})
    return pd.DataFrame(out)


def level_profile(nodes: pd.DataFrame) -> pd.DataFrame:
    """Mean verifier score of the states at each tree level (RQ3). 'agg' rows are the
    synthesized states s-hat; 'gen' rows are the top-down states s."""
    h = nodes[nodes["kind"] == "htmoa"]
    if h.empty:
        return pd.DataFrame()
    return (h.groupby(["method", "phase", "level"])
             .agg(nodes=("score", "size"), score=("score", "mean"), valid=("valid", "mean"),
                  input_tokens=("input_tokens", "mean"))
             .reset_index())


def aggregation_gain(nodes: pd.DataFrame) -> pd.DataFrame:
    """What each aggregation step does to the states it reads, per tree level.

    Every aggregator output is classed against the best plan among its children:
    synthesis (strictly better), selection (equal) or loss (worse). `gain` is the
    per-level amplification factor: mean aggregator score / mean child score; a tree
    only benefits from more depth while this stays above 1. `repair` = a valid plan
    out of all-invalid children; `corruption` = an invalid plan although a child was
    valid. `alt_better_%` = how often the runner-up plan (delta=2) outscored the main one.
    """
    out = []
    for (m, kind), g in nodes[nodes["kind"] != "single"].groupby(["method", "kind"], sort=False):
        recs = []  # (level, node, [children])
        for _, run in g.groupby(["instance_id", "rep"]):
            by = {(r.phase, r.level, r.index): r for r in run.itertuples()}
            if kind == "flat":
                recs.append((0, by[("agg", 0, 1)], [v for k, v in by.items() if k[0] == "gen"]))
                continue
            N, K = int(run["N"].iloc[0]), int(run["K"].iloc[0])
            for (ph, lvl, j), node in by.items():
                if ph != "agg":
                    continue
                child_phase = "gen" if lvl + 1 == K else "agg"
                recs.append((lvl, node, [by[(child_phase, lvl + 1, c)]
                                         for c in range((j - 1) * N + 1, j * N + 1)]))
        for lvl in sorted({r[0] for r in recs}):
            sel = [r for r in recs if r[0] == lvl]
            s = np.array([r[1].score for r in sel])
            best = np.array([max(c.score for c in r[2]) for r in sel])
            mean = np.array([np.mean([c.score for c in r[2]]) for r in sel])
            any_valid = np.array([any(c.valid for c in r[2]) for r in sel])
            valid = np.array([bool(r[1].valid) for r in sel])
            alts = [(r[1].alt_score, r[1].score) for r in sel
                    if r[1].alt_score is not None and not np.isnan(r[1].alt_score)]
            out.append({
                "method": m, "agg_level": lvl, "n": len(sel),
                "agg_score": s.mean(), "best_child": best.mean(), "mean_child": mean.mean(),
                "gain": s.mean() / mean.mean() if mean.mean() > 0 else float("nan"),
                "synthesis_%": 100 * (s > best + 1e-9).mean(),
                "selection_%": 100 * (np.abs(s - best) <= 1e-9).mean(),
                "loss_%": 100 * (s < best - 1e-9).mean(),
                "repair_%": 100 * (valid & ~any_valid).mean(),
                "corruption_%": 100 * (~valid & any_valid).mean(),
                "alt_better_%": (100 * np.mean([a > b + 1e-9 for a, b in alts])
                                 if alts else float("nan")),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)


def fig_score_vs_tokens(summ: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for fam in ("single", "flat", "htmoa"):
        g = summ[summ["method"].str.startswith(fam)]
        if g.empty:
            continue
        err = np.nan_to_num(np.array([g["score"] - g["score_lo"], g["score_hi"] - g["score"]]))
        ax.errorbar(g["tokens"], g["score"], yerr=np.clip(err, 0, None), fmt="o", ms=8,
                    color=FAMILY_COLOR[fam], ecolor=FAMILY_COLOR[fam], elinewidth=1.5,
                    mec=SURFACE, mew=2, label=FAMILY_LABEL[fam], zorder=3)
        for r in g.itertuples():
            ax.annotate(r.method.partition(":")[2] or "CoT", (r.tokens, r.score), xytext=(7, 6),
                        textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    _style(ax, "Mean total tokens per run (log scale)", "Normalized score (0 = invalid plan)",
           "Solution quality against compute spent (bars: 95% bootstrap CI)")
    _legend(ax, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=1.5)
    ax.set_title(ax.get_title(loc="left"), color=INK, fontsize=12, loc="left", pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_level_profile(lv: pd.DataFrame, path: Path):
    if lv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    # default configurations only; ablations are in the tables
    methods = [m for m in dict.fromkeys(lv["method"]) if base_spec(m) is None][: len(SERIES)]
    for i, m in enumerate(methods):
        g = lv[lv["method"] == m]
        K = int(g["level"].max())
        pts = pd.concat([g[(g["phase"] == "gen") & (g["level"] == K)], g[g["phase"] == "agg"]])
        pts = pts.sort_values("level", ascending=False)
        ax.plot(K - pts["level"], pts["score"], "-o", lw=2, ms=8, color=SERIES[i], mec=SURFACE,
                mew=2, label=m.partition(":")[2], zorder=3)
    ax.set_xticks(range(int(lv["level"].max()) + 1))
    _style(ax, "Aggregation steps above the leaves (0 = leaf states, rightmost point = root)",
           "Mean normalized score of the states", "State quality as it is aggregated up the tree")
    _legend(ax, loc="best", title="HT-MoA")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_context(summ: pd.DataFrame, path: Path):
    g = summ[summ["max_agg_ctx"] > 0].sort_values(["calls", "method"])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(g) + 1.6), facecolor=SURFACE)
    fams = [m.split(":")[0] for m in g["method"]]
    ax.barh(g["method"], g["max_agg_ctx"], height=0.6, color=[FAMILY_COLOR[f] for f in fams],
            zorder=3)
    for y, v in enumerate(g["max_agg_ctx"]):
        ax.text(v, y, f"  {int(v):,}", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    _style(ax, "Largest aggregator prompt observed (input tokens)", "",
           "Aggregator context load by architecture")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in ("flat", "htmoa")
               if f in fams]
    leg = ax.legend(handles, [FAMILY_LABEL[f] for f in ("flat", "htmoa") if f in fams],
                    frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_no data_"
    cols = list(df.columns)
    fmt = lambda v: (format(v, floatfmt) if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(path: Path):
    df, nodes = load(path)
    summ, tests = summary_table(df), paired_tests(df)
    lv, gain = level_profile(nodes), aggregation_gain(nodes)
    summ.to_csv(path / "summary.csv", index=False)
    fig_score_vs_tokens(summ, path / "fig_score_vs_tokens.png")
    fig_level_profile(lv, path / "fig_level_profile.png")
    fig_context(summ, path / "fig_context_load.png")

    models = ", ".join(sorted(df["model"].unique()))
    warn = ("\n> **Mock backend: these numbers only show that the pipeline runs. "
            "They are not evidence about HT-MoA.**\n" if "mock" in models else "")
    stops = nodes["stop_reason"].value_counts().to_dict()
    text = f"""# HT-MoA results — {path.name}
{warn}
Model: {models}. {df['instance_id'].nunique()} instances, {len(df)} runs, {len(nodes)} LLM calls.
Stop reasons over all calls: {stops}. Final plans that could not be parsed: {int((~df['parsed']).sum())}.

`score` = profit / optimal profit for a valid plan, 0 for an invalid one, averaged over
instances; `score_lo/hi` is a 95% bootstrap interval over instances. `profit_valid` averages
valid plans only. `max_agg_ctx` is the largest aggregator prompt (input tokens) seen.

## RQ1 / RQ2 — quality against compute
{md(summ)}

## Paired comparisons (HT-MoA minus baseline, per instance)
A default HT-MoA configuration is compared with the single agent and with the Flat-MoA
that makes the same number of LLM calls; an ablation with its default configuration.
{md(tests)}

## RQ3 — state quality by tree level
Level 0 is the root; `gen` = top-down states, `agg` = synthesized states.
{md(lv)}

## What each aggregation step does (synthesis / selection / loss)
Each aggregator output is classed against the best plan among the children it read.
`gain` = mean aggregator score / mean child score at that level; depth only pays while it
stays above 1. `repair` = valid plan from all-invalid children; `corruption` = invalid plan
although a child was valid; `alt_better_%` = runner-up plan outscored the main plan (delta=2).
{md(gain)}
"""
    (path / "summary.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles and pipes
    print(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
