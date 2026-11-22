"""Pooled analysis over several result directories (one per difficulty tier).

    python pooled_analysis.py results/haiku55_easy results/haiku55_medium \
        results/haiku55_hard results/haiku55_expert --out analysis

Writes pooled_summary.md, LaTeX tables (tab_*.tex), the worked example and prompts of the
appendix, and two figures into --out. Every number quoted in the paper comes from this script.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze as A
from htmoa.env import Instance, _normalize
from htmoa.methods import base_spec, n_calls, parse_spec, parse_state

FLAT = ["flat:M=2", "flat:M=4", "flat:M=8", "flat:M=15", "flat:M=20", "flat:M=44"]
DEEP = ["htmoa:N=2,K=1", "htmoa:N=2,K=2", "htmoa:N=2,K=3", "htmoa:N=2,K=4"]
WIDE = ["htmoa:N=4,K=1", "htmoa:N=3,K=2"]
MATCHED = [("htmoa:N=2,K=1", "flat:M=2"), ("htmoa:N=4,K=1", "flat:M=4"),
           ("htmoa:N=2,K=2", "flat:M=8"), ("htmoa:N=3,K=2", "flat:M=15"),
           ("htmoa:N=2,K=3", "flat:M=20"), ("htmoa:N=2,K=4", "flat:M=44")]
ABLATIONS = ["topdown=0", "delta=0", "delta=2", "split=1", "delta=2,split=1", "personas=0"]


def label(m: str, short: bool = False) -> str:
    kind, _, rest = m.partition(":")
    if kind == "single":
        return "Single agent"
    p = dict(kv.split("=") for kv in rest.split(","))
    if kind == "flat":
        return f"$M={p['M']}$" if short else f"Flat-MoA, $M={p['M']}$"
    extra = ", ".join(f"{k}={v}" for k, v in p.items() if k not in "NK")
    core = f"$N={p['N']}, K={p['K']}$"
    return (core if short else "HT-MoA, " + core) + (f" ({extra})" if extra else "")


def tier_of(index: pd.Index) -> np.ndarray:
    return index.str.rsplit("-", n=1).str[0].to_numpy()


def boot(s: pd.Series, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap of the mean, resampling instances within each tier."""
    s = s.dropna()
    tiers = tier_of(s.index)
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for t in np.unique(tiers):
        x = s.to_numpy(float)[tiers == t]
        total += x[rng.integers(0, len(x), (n, len(x)))].sum(axis=1)
    means = total / len(s)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired(piv: pd.DataFrame, a: str, b: str) -> dict:
    d = (piv[a].astype(float) - piv[b].astype(float)).dropna()
    lo, hi = boot(d)
    x = d.to_numpy()
    p = stats.wilcoxon(x).pvalue if np.any(x != 0) else float("nan")
    return {"a": a, "b": b, "n": len(x), "diff": x.mean(), "lo": lo, "hi": hi, "p": p,
            "wins": int((x > 0).sum()), "ties": int((x == 0).sum()), "losses": int((x < 0).sum())}


def aggregation_steps(nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per aggregator call: its score against the children it read."""
    rows = []
    for (m, kind), grp in nodes[nodes["kind"] != "single"].groupby(["method", "kind"]):
        for (tier, iid, rep), run in grp.groupby(["tier", "instance_id", "rep"]):
            by = {(x.phase, x.level, x.index): x for x in run.itertuples()}
            if kind == "flat":
                steps = [(0, by[("agg", 0, 1)], [v for k, v in by.items() if k[0] == "gen"])]
            else:
                N, K = int(run["N"].iloc[0]), int(run["K"].iloc[0])
                steps = [(lvl, a, [by[("gen" if lvl + 1 == K else "agg", lvl + 1, c)]
                                   for c in range((j - 1) * N + 1, j * N + 1)])
                         for (ph, lvl, j), a in by.items() if ph == "agg"]
            for lvl, a, kids in steps:
                best = max(c.score for c in kids)
                rows.append({
                    "method": m, "kind": kind, "tier": tier, "level": lvl, "fan_in": len(kids),
                    "score": a.score, "best_child": best,
                    "mean_child": float(np.mean([c.score for c in kids])),
                    "synthesis": a.score > best + 1e-9, "loss": a.score < best - 1e-9,
                    "corruption": (not a.valid) and any(c.valid for c in kids),
                    "repair": bool(a.valid) and not any(c.valid for c in kids)})
    return pd.DataFrame(rows)


def canonical(inst: Instance, plan) -> tuple | None:
    """Order-independent form of a plan, for exact comparison of two plans."""
    if plan is None:
        return None
    quarters, _ = _normalize(inst, plan)
    return tuple((tuple(sorted((p, s, j) for p, (s, j) in a.items() if s or j)), tuple(sorted(m)))
                 for a, m in quarters)


def copy_steps(dirs: list[Path]) -> pd.DataFrame:
    """One row per aggregator call, from the saved texts: is its plan an exact copy of
    one of the plans it read, and where did that plan sit in the prompt?"""
    rows = []
    for d in dirs:
        insts = {i["id"]: Instance.from_dict(i)
                 for i in json.loads((d / "instances.json").read_text(encoding="utf-8"))}
        plain = d / "traces.jsonl"   # the repository ships the traces gzip-compressed
        with (plain.open(encoding="utf-8") if plain.exists()
              else gzip.open(d / "traces.jsonl.gz", "rt", encoding="utf-8")) as f:
            for line in f:
                r = json.loads(line)
                kind, p = parse_spec(r["method"])
                if kind == "single":
                    continue
                inst = insts[r["instance_id"]]
                by = {(n["phase"], n["level"], n["index"]): n for n in r["nodes"]}
                for n in r["nodes"]:
                    n["canon"] = canonical(inst, parse_state(n["text"]).plan)
                if kind == "flat":
                    kids = sorted((n for n in r["nodes"] if n["phase"] == "gen"),
                                  key=lambda n: n["index"])
                    steps = [(by[("agg", 0, 1)], kids)]
                else:
                    N, K = p["N"], p["K"]
                    steps = [(a, [by[("gen" if lvl + 1 == K else "agg", lvl + 1, c)]
                                  for c in range((j - 1) * N + 1, j * N + 1)])
                             for (ph, lvl, j), a in by.items() if ph == "agg"]
                for a, kids in steps:
                    best = max(k["score"] for k in kids)
                    best_idx = [i for i, k in enumerate(kids) if k["score"] >= best - 1e-9]
                    match = [i for i, k in enumerate(kids)
                             if a["canon"] is not None and k["canon"] == a["canon"]]
                    rows.append({
                        "method": r["method"], "kind": kind, "fan_in": len(kids),
                        "copied": bool(match), "copy_idx": match[0] if match else -1,
                        "copied_best": any(i in best_idx for i in match),
                        "best_idx": best_idx[0], "n_best": len(best_idx), "best": best,
                        "loss": a["score"] < best - 1e-9, "better": a["score"] > best + 1e-9,
                        "distinct": len({k["canon"] for k in kids})})
    return pd.DataFrame(rows)


GEN = dict(fc="#cde2fb", ec="#2a78d6")    # a generation call
AGG = dict(fc="#fbd9cc", ec="#eb6834")    # an aggregation call


def _canvas(ax, title: str, xlim: float, ylim: float) -> None:
    ax.set_xlim(0, xlim)
    ax.set_ylim(0.45, ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, color=A.INK, loc="center", pad=4, linespacing=1.4)


def _arrow(ax, a, b, shrink=0.36) -> None:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(
        arrowstyle="-|>", color=A.INK2, lw=1.1, shrinkA=shrink * 72 / 2.6, shrinkB=shrink * 72 / 2.6,
        mutation_scale=9))


def _call(ax, xy, kind: str, label: str = "") -> None:
    from matplotlib.patches import Circle, FancyBboxPatch
    x, y = xy
    if kind == "gen":
        ax.add_patch(Circle(xy, 0.34, lw=1.4, zorder=3, **GEN))
    elif kind == "agg":
        ax.add_patch(FancyBboxPatch((x - 0.31, y - 0.31), 0.62, 0.62, zorder=3, lw=1.4,
                                    boxstyle="round,pad=0.02,rounding_size=0.08", **AGG))
    else:   # the empty root state
        ax.add_patch(Circle(xy, 0.34, lw=1.2, zorder=3, fc=A.SURFACE, ec=A.MUTED, ls=(0, (3, 2))))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color=A.INK, zorder=4)


def fig_architecture(path: Path) -> None:
    """Flat-MoA and HT-MoA at the same number of calls (9): the schematic of Section 3."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=A.SURFACE,
                             gridspec_kw={"wspace": 0.12})
    ax = axes[0]
    _canvas(ax, "Flat-MoA\n8 generations, then 1 aggregation of fan-in 8", 8.4, 4.75)
    workers = [(0.6 + i * 1.03, 1.0) for i in range(8)]
    for i, w in enumerate(workers):
        _call(ax, w, "gen", f"$s_{{{i + 1}}}$")
        _arrow(ax, w, (4.2, 3.9))
    _call(ax, (4.2, 3.9), "agg", "$y$")

    levels = {1: [(2.2, 2.6), (6.2, 2.6)], 2: [(1.2, 1.0), (3.2, 1.0), (5.2, 1.0), (7.2, 1.0)]}
    ax = axes[1]
    _canvas(ax, "HT-MoA, step 1: top-down expansion\n6 generations, each refines its parent's draft",
            8.4, 4.75)
    root = (4.2, 4.2)
    _call(ax, root, "root", r"$\emptyset$")
    for j, p in enumerate(levels[1]):
        _arrow(ax, root, p)
        _call(ax, p, "gen", f"$s_{{1,{j + 1}}}$")
        for c in range(2):
            leaf = levels[2][2 * j + c]
            _arrow(ax, p, leaf)
            _call(ax, leaf, "gen", f"$s_{{2,{2 * j + c + 1}}}$")

    ax = axes[2]
    _canvas(ax, "HT-MoA, step 2: bottom-up aggregation\n3 aggregations of fan-in 2", 8.4, 4.75)
    for j, p in enumerate(levels[1]):
        for c in range(2):
            leaf = levels[2][2 * j + c]
            _call(ax, leaf, "gen", f"$s_{{2,{2 * j + c + 1}}}$")
            _arrow(ax, leaf, p)
        _call(ax, p, "agg", rf"$\hat{{s}}_{{1,{j + 1}}}$")
        _arrow(ax, p, root)
    _call(ax, root, "agg", "$y$")

    handles = [Line2D([], [], marker="o", ls="", ms=11, mfc=GEN["fc"], mec=GEN["ec"], mew=1.4),
               Line2D([], [], marker="s", ls="", ms=10, mfc=AGG["fc"], mec=AGG["ec"], mew=1.4)]
    leg = fig.legend(handles, ["generation call", "aggregation call"], loc="lower center", ncol=2,
                     frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color(A.INK2)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=A.SURFACE)
    plt.close(fig)


def fig_study_flow(path: Path, stages: list[tuple[str, str, str]]) -> None:
    """Flow of the study with the number of units at each stage. `stages` holds
    (stage name, main text, side note or '')."""
    from matplotlib.patches import FancyBboxPatch
    h, gap, w, side_w = 1.25, 0.55, 6.6, 4.6
    total = len(stages) * h + (len(stages) - 1) * gap
    fig, ax = plt.subplots(figsize=(9.4, 0.62 * total + 0.3), facecolor=A.SURFACE)
    ax.set_xlim(0, w + side_w + 1.0)
    ax.set_ylim(-0.08, total + 0.08)
    ax.axis("off")
    for i, (name, text, note) in enumerate(stages):
        y = total - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.05, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    lw=1.3, fc="#eaf2fd", ec="#2a78d6"))
        ax.text(0.05 + w / 2, y + h - 0.30, name, ha="center", va="center", fontsize=9.5,
                color=A.INK, fontweight="bold")
        ax.text(0.05 + w / 2, y + h / 2 - 0.20, text, ha="center", va="center", fontsize=8.6,
                color=A.INK, linespacing=1.35)
        if i < len(stages) - 1:
            ax.annotate("", xy=(0.05 + w / 2, y - gap + 0.04), xytext=(0.05 + w / 2, y - 0.04),
                        arrowprops=dict(arrowstyle="-|>", color=A.INK2, lw=1.2, mutation_scale=11))
        if note:
            ax.add_patch(FancyBboxPatch((w + 0.9, y + 0.14), side_w, h - 0.28, lw=1.0,
                                        boxstyle="round,pad=0.02,rounding_size=0.10",
                                        fc=A.SURFACE, ec=A.MUTED))
            ax.text(w + 0.9 + side_w / 2, y + h / 2, note, ha="center", va="center", fontsize=8.2,
                    color=A.INK2, linespacing=1.35)
            ax.annotate("", xy=(w + 0.88, y + h / 2), xytext=(w + 0.07, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=A.MUTED, lw=1.0, mutation_scale=9))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=A.SURFACE)
    plt.close(fig)


def tex_table(path: Path, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == ["--"] else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows if r != ["--"]]
    for a, b in (("$", ""), ("\\%", "%"), ("\\texttt{", "`"), ("\\text{", ""), ("}", ""), ("{", ""),
                 ("\\Delta", "Δ"), ("\\ge", "≥")):
        out = [x.replace(a, b) for x in out]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dfs, nds, tiers = [], [], []
    for d in args.dirs:
        df, nodes = A.load(d)
        tier = df["instance_id"].iloc[0].rsplit("-", 1)[0]
        tiers.append(tier)
        dfs.append(df.assign(tier=tier))
        nds.append(nodes.assign(tier=tier))
    df, nodes = pd.concat(dfs), pd.concat(nds)
    df["near"] = df["score"] >= 0.9
    per_inst = df.groupby(["tier", "instance_id", "method"])["score"].mean().unstack()
    piv = per_inst.droplevel("tier")
    piv_valid = df.pivot(index="instance_id", columns="method", values="valid").astype(float)
    piv_near = df.pivot(index="instance_id", columns="method", values="near").astype(float)
    valid = df.groupby("method")["valid"].mean()
    steps = aggregation_steps(nodes)
    copies = copy_steps(args.dirs)
    md = [f"# Pooled results\n\nModel: {', '.join(sorted(df['model'].unique()))}. Tiers: "
          f"{', '.join(tiers)}. {len(piv)} instances, {len(df)} runs, {int(df['calls'].sum())} LLM "
          f"calls ({len(steps)} aggregations), ${df['cost_usd'].sum():.2f}. Stop reasons: "
          f"{nodes['stop_reason'].value_counts().to_dict()}. Unparseable final plans: "
          f"{int((~df['parsed']).sum())}.\n\nPooled intervals are 95% bootstrap intervals "
          f"stratified by tier; p-values are two-sided Wilcoxon signed-rank tests.\n"]

    def section(title, header, rows, name, align, note=""):
        tex_table(args.out / f"{name}.tex", header, rows, align)
        md.extend([f"\n## {title}", note, md_table(header, rows)] if note else
                  [f"\n## {title}", md_table(header, rows)])

    def est(r, digits=3):
        return [f"{r['diff']:+.{digits}f}", f"[{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}]"]

    def wtl(r):
        return f"{r['wins']}/{r['ties']}/{r['losses']}"

    # ---- score by tier and pooled ------------------------------------------
    header = ["Method", "Calls"] + [t.capitalize() for t in tiers] + ["Pooled [95\\% CI]", "Valid \\%"]
    rows = []
    for block in (["single"], FLAT, DEEP, WIDE):
        for m in block:
            lo, hi = boot(piv[m])
            rows.append([label(m), str(n_calls(m))]
                        + [f"{per_inst.loc[t, m].mean():.2f}" for t in tiers]
                        + [f"{piv[m].mean():.2f} [{lo:.2f}, {hi:.2f}]", f"{100 * valid[m]:.0f}"])
        rows.append(["--"])
    section("Score by tier and pooled", header, rows[:-1], "tab_main", "lr" + "c" * len(tiers) + "cr")

    # ---- HT-MoA vs call-matched Flat-MoA -----------------------------------
    res = [paired(piv, a, b) for a, b in MATCHED]
    adj = holm([r["p"] for r in res])
    header = ["HT-MoA", "Flat-MoA", "Calls", "$\\Delta$ score", "95\\% CI", "$p$",
              "$p_{\\text{Holm}}$", "W/T/L"]
    rows = [[label(r["a"], True), label(r["b"], True), str(n_calls(r["a"]))] + est(r)
            + [f"{r['p']:.4f}", f"{q:.4f}", wtl(r)] for r, q in zip(res, adj)]
    section("HT-MoA minus the call-matched Flat-MoA (pooled, paired by instance)", header, rows,
            "tab_matched", "llrccccc")

    header = ["HT-MoA", "Flat-MoA"] + [t.capitalize() for t in tiers]
    rows = []
    for a, b in MATCHED:
        row = [label(a, True), label(b, True)]
        for t in tiers:
            r = paired(per_inst.loc[t], a, b)
            row.append(f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
        rows.append(row)
    section("The same comparison within each tier (30 instances each)", header, rows,
            "tab_matched_tier", "ll" + "c" * len(tiers))

    header = ["HT-MoA", "Flat-MoA", "$\\Delta$ valid plans (pp)", "95\\% CI",
              "$\\Delta$ plans with score $\\ge 0.9$ (pp)", "95\\% CI"]
    rows = []
    for a, b in MATCHED:
        v, n9 = paired(piv_valid, a, b), paired(piv_near, a, b)
        rows.append([label(a, True), label(b, True), f"{100 * v['diff']:+.1f}",
                     f"[{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]", f"{100 * n9['diff']:+.1f}",
                     f"[{100 * n9['lo']:+.1f}, {100 * n9['hi']:+.1f}]"])
    section("Matched comparison under two other outcome measures", header, rows, "tab_robust",
            "llcccc")

    # ---- depth and width steps ---------------------------------------------
    header = ["Step", "$\\Delta$ score", "95\\% CI", "$p$", "W/T/L"]
    rows = []
    for block in (list(zip(DEEP[1:], DEEP[:-1])), list(zip(FLAT[1:], FLAT[:-1]))):
        for a, b in block:
            r = paired(piv, a, b)
            rows.append([f"{label(a)} vs. {label(b, True)}"] + est(r) + [f"{r['p']:.4f}", wtl(r)])
        rows.append(["--"])
    section("Marginal effect of one more level (HT-MoA) or more workers (Flat-MoA)", header,
            rows[:-1], "tab_steps", "lcccc")

    # ---- aggregation against fan-in -----------------------------------------
    root = steps[(steps["level"] == 0) & steps["method"].map(lambda m: base_spec(m) is None)]
    header = ["Method", "Fan-in", "Root score", "Best child", "Loss \\%", "Synthesis \\%",
              "Corruption \\%"]
    rows = []
    for block in (FLAT, DEEP, WIDE):
        for m in block:
            g = root[root["method"] == m]
            rows.append([label(m), str(int(g["fan_in"].iloc[0])), f"{g['score'].mean():.2f}",
                         f"{g['best_child'].mean():.2f}", f"{100 * g['loss'].mean():.0f}",
                         f"{100 * g['synthesis'].mean():.0f}", f"{100 * g['corruption'].mean():.0f}"])
        rows.append(["--"])
    section("Root aggregator against the best state it read", header, rows[:-1], "tab_fanin",
            "lrccccc", "Loss = scored below its best child; synthesis = above it; corruption = "
            "invalid output although a child was valid.")

    by_fan = steps.groupby("fan_in").agg(
        steps=("score", "size"), loss=("loss", "mean"), synthesis=("synthesis", "mean"),
        corruption=("corruption", "mean"), best=("best_child", "mean"), score=("score", "mean"))
    header = ["Fan-in", "Steps", "Loss \\%", "Synthesis \\%", "Selection \\%", "Corruption \\%",
              "Mean regret"]
    rows = [[str(f), str(int(r.steps)), f"{100 * r.loss:.1f}", f"{100 * r.synthesis:.1f}",
             f"{100 * (1 - r.loss - r.synthesis):.1f}", f"{100 * r.corruption:.1f}",
             f"{r.best - r.score:.3f}"] for f, r in by_fan.iterrows()]
    section("Every aggregation step of every method, grouped by fan-in", header, rows,
            "tab_fanin_all", "rrccccc")

    flat_root = root[root["kind"] == "flat"]
    header = ["Fan-in"] + [t.capitalize() for t in tiers]
    rows = [[str(f)] + [f"{100 * g[g['tier'] == t]['loss'].mean():.0f}" for t in tiers]
            for f, g in flat_root.groupby("fan_in")]
    section("Flat-MoA: share of runs (%) in which the aggregator scored below its best input, "
            "by tier", header, rows, "tab_fanin_tier", "r" + "c" * len(tiers))

    # ---- exact copies and position ------------------------------------------
    header = ["Fan-in", "Steps", "Exact copy \\%", "Copy of a best input \\%",
              "New plan, better \\%", "New plan, not better \\%", "Distinct inputs"]
    rows = []
    for f, g in copies.groupby("fan_in"):
        new = ~g["copied"]
        rows.append([str(f), str(len(g)), f"{100 * g['copied'].mean():.1f}",
                     f"{100 * g['copied_best'].mean():.1f}", f"{100 * (new & g['better']).mean():.1f}",
                     f"{100 * (new & ~g['better']).mean():.1f}", f"{g['distinct'].mean():.1f}"])
    section("Is the aggregator's plan an exact copy of a plan it read?", header, rows, "tab_copy",
            "rrccccc", "Plans are compared after normalization (order of entries ignored). "
            "Distinct inputs = mean number of different plans among the candidates.")

    wide = copies[(copies["kind"] == "flat") & (copies["fan_in"] >= 8)].copy()
    third = lambda idx, fan: np.minimum(2, (3 * idx / fan).astype(int))
    wide["copy_third"] = third(wide["copy_idx"], wide["fan_in"])
    wide["best_third"] = third(wide["best_idx"], wide["fan_in"])
    cp = wide[wide["copied"]]
    uniq = wide[(wide["n_best"] == 1) & (wide["best"] > 0)]
    header = ["Position in the prompt", "Share of copied plans \\%",
              "Loss \\% when the single best input is there", "Runs"]
    rows = [[name, f"{100 * (cp['copy_third'] == k).mean():.0f}",
             f"{100 * uniq[uniq['best_third'] == k]['loss'].mean():.0f}",
             str(int((uniq['best_third'] == k).sum()))]
            for k, name in enumerate(["First third", "Middle third", "Last third"])]
    section("Flat-MoA with 8 or more candidates: does position matter?", header, rows,
            "tab_position", "lccc", f"{len(cp)} copied plans; {len(uniq)} runs with a single "
            f"best input of positive score.")

    # ---- ablations -----------------------------------------------------------
    res = [paired(piv, f"{base},{x}", base) for base in ("htmoa:N=3,K=2", "htmoa:N=2,K=3")
           for x in ABLATIONS]
    adj = holm([r["p"] for r in res])
    header = ["Variant", "Base", "$\\Delta$ score", "95\\% CI", "$p$", "$p_{\\text{Holm}}$", "W/T/L"]
    rows = []
    for i, (r, q) in enumerate(zip(res, adj)):
        if i == len(ABLATIONS):
            rows.append(["--"])
        rows.append([f"\\texttt{{{r['a'].split(',', 2)[2]}}}", label(r["b"], True)] + est(r)
                    + [f"{r['p']:.4f}", f"{q:.4f}", wtl(r)])
    section("Ablations (variant minus default configuration, pooled)", header, rows,
            "tab_ablation", "llccccc")
    extra = paired(piv, "htmoa:N=2,K=3,topdown=0", "flat:M=8")
    md.append(f"\nSame 8 independent leaves, tree aggregation (N=2, K=3, topdown=0) minus flat "
              f"aggregation (M=8): {extra['diff']:+.3f} [{extra['lo']:+.3f}, {extra['hi']:+.3f}], "
              f"p={extra['p']:.4f}, W/T/L {wtl(extra)}.")
    alt = nodes[nodes["alt_score"].notna()]
    md.append(f"Runner-up plan (delta=2) scored above the main plan in "
              f"{100 * (alt['alt_score'] > alt['score'] + 1e-9).mean():.1f}% of {len(alt)} aggregations.")

    # ---- state quality by level, deepest tree --------------------------------
    deep = nodes[nodes["method"] == DEEP[-1]]
    K = int(deep["level"].max())
    lv = deep.groupby(["phase", "level"])["score"].mean()
    header = ["State"] + [f"Level {k}" for k in range(K + 1)]
    rows = [["Top-down $s_{k}$"] + ["--" if ("gen", k) not in lv else f"{lv[('gen', k)]:.2f}"
                                    for k in range(K + 1)],
            ["Bottom-up $\\hat{s}_{k}$"] + [f"{lv[('agg', k)]:.2f}" if ("agg", k) in lv
                                            else f"{lv[('gen', k)]:.2f}" for k in range(K + 1)]]
    section(f"Mean state score by level, {label(DEEP[-1])} (level 0 = root)", header, rows,
            "tab_levels", "l" + "c" * (K + 1))
    workers = nodes[(nodes["kind"] == "flat") & (nodes["phase"] == "gen")]["score"].mean()
    md.append(f"\nIndependent Flat-MoA workers average {workers:.3f}; the single agent "
              f"{piv['single'].mean():.3f}.")

    ctx = df.groupby("method")["max_agg_input_tokens"].max()
    tok = df.groupby("method")["total_tokens"].mean()
    header = ["Method", "Calls", "Largest aggregator prompt (tokens)", "Mean tokens per run"]
    rows = [[label(m), str(n_calls(m)), f"{int(ctx[m]):,}", f"{int(tok[m]):,}"]
            for m in FLAT + DEEP + WIDE]
    section("Aggregator context load and cost", header, rows, "tab_context", "lrrr")

    # ---- the benchmark: tier statistics, a worked example, the prompts ----------
    from htmoa.env import DIFFICULTY, describe, verify
    from htmoa.methods import PERSONAS, SYSTEM, agg_format, gen_format
    from htmoa.solver import solve_reference
    insts = {t: [Instance.from_dict(i) for i in json.loads(
        (d / "instances.json").read_text(encoding="utf-8"))] for t, d in zip(tiers, args.dirs)}
    header = ["Tier", "Quarters", "Projects", "Crises", "Cash ratio", "Integer variables",
              "Optimal profit", "Profit of the empty plan", "Single-agent score"]
    rows = []
    for t in tiers:
        cfg, ii = DIFFICULTY[t], insts[t]
        nvar = np.mean([3 * len(i.projects) * i.n_quarters + sum(c.quarter - 1 for c in i.crises)
                        for i in ii])
        rows.append([t.capitalize(), str(cfg["quarters"]), str(cfg["projects"]), str(cfg["crises"]),
                     f"{cfg['cash_frac']:.2f}", f"{nvar:.0f}",
                     f"{np.mean([i.ref_profit for i in ii]):.0f}",
                     f"{np.mean([verify(i, {'quarters': []}).profit for i in ii]):.0f}",
                     f"{per_inst.loc[t, 'single'].mean():.2f}"])
    section("Difficulty tiers (means over the 30 instances of each tier; money in $k)", header,
            rows, "tab_tiers", "lcccccccc")

    # ---- schematic of the architectures and flow of the study --------------------
    from htmoa.env import generate
    from htmoa.solver import MIN_OPTIMUM
    fig_architecture(args.out / "fig_architecture.png")
    rejected = 0   # generator draws discarded before each accepted instance
    for t in tiers:
        for i in insts[t]:
            attempt = 0
            while solve_reference(generate(i.seed, t, attempt))[1] < MIN_OPTIMUM:
                attempt += 1
            rejected += attempt
    n_inst, n_methods = len(piv), df["method"].nunique()
    kinds = df.drop_duplicates("method")
    n_abl = int(kinds["method"].map(lambda m: base_spec(m) is not None).sum())
    n_flat = int((kinds["kind"] == "flat").sum())
    n_tree = int((kinds["kind"] == "htmoa").sum()) - n_abl
    n_calls_total, n_agg = int(df["calls"].sum()), len(steps)
    ended = int((nodes["stop_reason"] == "end_turn").sum())
    fig_study_flow(args.out / "fig_study_flow.png", [
        ("Benchmark instances",
         f"{len(tiers)} difficulty tiers x {n_inst // len(tiers)} seeds\n"
         f"instances used: n = {n_inst}, each solved to optimality",
         f"generator draws rejected: n = {rejected}\n(optimal profit below {MIN_OPTIMUM}, or infeasible)"),
        ("Methods",
         f"{n_methods} methods: 1 single agent, {n_flat} Flat-MoA,\n"
         f"{n_tree} HT-MoA, {n_abl} HT-MoA ablations", ""),
        ("Runs",
         f"{n_inst} instances x {n_methods} methods = {len(df):,} runs\n"
         f"completed: n = {len(df):,}", f"missing runs: n = {n_inst * n_methods - len(df)}"),
        ("Model calls",
         f"{n_calls_total:,} calls: {n_calls_total - n_agg:,} generations,\n{n_agg:,} aggregations",
         f"calls not ended normally: n = {n_calls_total - ended}\n"
         f"final plans not parseable: n = {int((~df['parsed']).sum())}"),
        ("Scoring",
         f"every state scored by the exact verifier:\nn = {len(nodes):,} plans, against the optimum", ""),
        ("Analyses",
         f"RQ1: {len(MATCHED)} matched comparisons   RQ2: {n_agg:,} aggregations\n"
         f"RQ3: state quality by tree level   RQ4: {n_abl} ablations", ""),
    ])
    md.append(f"\nStudy flow: {rejected} generator draws rejected; {n_calls_total - n_agg} generation "
              f"calls and {n_agg} aggregation calls; {len(nodes)} scored states.")

    ex = insts[tiers[0]][0]
    plan, profit = solve_reference(ex)
    ev = verify(ex, plan)
    def wrap(text: str, width: int = 92) -> str:
        import textwrap
        return "\n".join(textwrap.fill(line, width, subsequent_indent="  ") if line else ""
                         for line in text.splitlines())
    plan_text = "{\"quarters\": [\n" + ",\n".join(
        "  " + json.dumps(q) for q in plan["quarters"]) + "\n]}"
    (args.out / "app_example.tex").write_text(
        "\\begin{verbatim}\n" + wrap(describe(ex)) + "\n\\end{verbatim}\n\n"
        f"\\noindent The optimal plan, found by the exact solver, has profit {profit} "
        f"(cash after each quarter: {', '.join(map(str, ev.cash_by_quarter))}; projects completed: "
        + ", ".join(f"{p} in Q{q}" for p, q in sorted(ev.completed.items())) + "):\n"
        "\\begin{verbatim}\n" + wrap(plan_text) + "\n\\end{verbatim}\n", encoding="utf-8")
    personas = "\n".join(f"{i + 1}. {name}: {text}" for i, (name, text) in enumerate(PERSONAS))
    (args.out / "app_prompts.tex").write_text(
        "\\noindent System prompt (all calls):\n\\begin{verbatim}\n" + wrap(SYSTEM)
        + "\n\\end{verbatim}\n\\noindent Personas (a tree with branching factor $N$ uses the first "
        "$N$; flat workers cycle through all six):\n\\begin{verbatim}\n" + wrap(personas)
        + "\n\\end{verbatim}\n\\noindent Output instruction of a generation call:\n\\begin{verbatim}\n"
        + wrap(gen_format()) + "\n\\end{verbatim}\n\\noindent Output instruction of an aggregation "
        "call with dissent notes (the default):\n\\begin{verbatim}\n" + wrap(agg_format(1))
        + "\n\\end{verbatim}\n", encoding="utf-8")

    # ---- figures ---------------------------------------------------------------
    def series(ms):
        cis = [boot(piv[m]) for m in ms]
        y = np.array([piv[m].mean() for m in ms])
        return ([n_calls(m) for m in ms], y, y - [c[0] for c in cis], [c[1] for c in cis] - y)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=A.SURFACE)
    for ms, fam, name in ((FLAT, "flat", "Flat-MoA (wider)"), (DEEP, "htmoa", "HT-MoA, N=2 (deeper)")):
        x, y, lo, hi = series(ms)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="-o", lw=2, ms=8, color=A.FAMILY_COLOR[fam],
                    mec=A.SURFACE, mew=2, elinewidth=1.2, label=name, zorder=3)
    x, y, lo, hi = series(["single"])
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=8, color=A.FAMILY_COLOR["single"],
                mec=A.SURFACE, mew=2, elinewidth=1.2, label="Single agent", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 9, 16, 21, 45], labels=["1", "3", "5", "9", "16", "21", "45"])
    ax.minorticks_off()
    A._style(ax, "LLM calls per run (log scale)", "Normalized score, pooled over tiers",
             "More workers saturate; more levels keep paying")
    A._legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "fig_depth_vs_width.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=A.SURFACE)
    fl = flat_root.groupby("fan_in").agg(loss=("loss", "mean"), synthesis=("synthesis", "mean"))
    ax.plot(fl.index, 100 * fl["loss"], "-o", lw=2, ms=8, color=A.FAMILY_COLOR["flat"],
            mec=A.SURFACE, mew=2, label="Aggregator scored below its best input", zorder=3)
    ax.plot(fl.index, 100 * fl["synthesis"], "-o", lw=2, ms=8, color=A.SERIES[6],
            mec=A.SURFACE, mew=2, label="Aggregator scored above its best input", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(list(fl.index), labels=[str(i) for i in fl.index])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    A._style(ax, "Number of candidates the aggregator reads (log scale)",
             "Share of Flat-MoA aggregations (%)",
             "The more candidates an aggregator reads, the more often it loses the best one")
    A._legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "fig_fanin_loss.png", dpi=200)
    plt.close(fig)

    (args.out / "pooled_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles and pipes
    print("\n".join(md))


if __name__ == "__main__":
    main()
