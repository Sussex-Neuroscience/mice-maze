#!/usr/bin/env python3
"""
Experiment 2: Thesis Figures
============================

    fig_exp2_session  the session-level preference for the grammar heard in the
                      enriched cage (EE) over the one heard in the standard cage
                      (SC): (a) time on the EE and SC grammar arms per session,
                      day 1; (b) the EE-SC index of time at each level of
                      predictability, day 1, coloured by counterbalancing group;
                      (c) the index on day 1 and day 2
    fig_exp2_model    the value model of Chapter 5 as fitted by the earlier
                      analysis (model_validation/, read only, from its saved
                      results.json and grain_results.json): (a) recovery of a
                      known semantic weight; (b) the semantic weight at three
                      grains of observation; (c) the predicted and observed share
                      of time in each arm type (per-mouse cell means)

Usage:
    python exp2_figures.py [GRAMMAR_DIR] --thesis-figures DIR

    run exp2_lmm.py first (it writes BATCH_ANALYSIS/exp2/arm_blocks.csv).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EE, SC = "#1565C0", "#B71C1C"
GROUP = {1: "#1565C0", 2: "#E65100"}
TIERS = ["dominant", "secondary", "rare"]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def index(a, b):
    return (a - b) / (a + b) if a + b > 0 else np.nan


def per_mouse_index(d, tier=None):
    g = d[d["cls"] == "grammar"]
    if tier is not None:
        g = g[g["tier"] == tier]
    t = g.groupby(["mouse", "environment"])["time_spent_s"].sum().unstack(fill_value=0)
    return t.apply(lambda r: index(r["EE"], r["SC"]), axis=1).dropna()


def fig_session(d, path):
    d1, d2 = d[d["day"] == "day_1"], d[d["day"] == "day_2"]
    groups = d.groupby("mouse")["group"].first()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [0.8, 1.4, 0.9]})
    rng = np.random.default_rng(0)

    ax = axes[0]
    t = d1[d1["cls"] == "grammar"].groupby(["mouse", "environment"])["time_spent_s"].sum().unstack(fill_value=0)
    t = t[t.sum(axis=1) > 0] / 60
    for _, r in t.iterrows():
        ax.plot([0, 1], [r["EE"], r["SC"]], color="#BDBDBD", lw=0.5, zorder=1)
    ax.scatter(np.zeros(len(t)), t["EE"], s=5, color=EE, zorder=2, lw=0)
    ax.scatter(np.ones(len(t)), t["SC"], s=5, color=SC, zorder=2, lw=0)
    for x, c in ((0, "EE"), (1, "SC")):
        ax.plot([x - 0.2, x + 0.2], [t[c].median()] * 2, color="black", lw=1.6, zorder=3)
    p = wilcoxon(t["EE"], t["SC"]).pvalue
    ax.text(0.5, t.values.max() * 1.02, stars(p), ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["EE\ngrammar", "SC\ngrammar"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Time in the grammar arms\nper session (min)")
    ax.set_title("(a) day 1", loc="left")

    ax = axes[1]
    pis = {tier: per_mouse_index(d1, tier) for tier in TIERS}
    ph = multipletests([wilcoxon(pis[t_]).pvalue for t_ in TIERS], method="holm")[1]
    for i, tier in enumerate(TIERS):
        x = pis[tier]
        cols = [GROUP[int(groups[m])] for m in x.index]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=5, c=cols, lw=0, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [x.median()] * 2, color="black", lw=1.6, zorder=3)
        ax.text(i, 0.93, stars(ph[i]), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(TIERS)
    ax.set_ylim(-1.0, 1.05)
    ax.set_ylabel("EE-SC index of time")
    ax.set_title("(b) day 1, by level of predictability", loc="left")
    for g_, c in GROUP.items():
        ax.scatter([], [], s=8, color=c, label=f"group {g_} (EE = grammar {'A' if g_ == 1 else 'B'})")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")

    ax = axes[2]
    a, b = per_mouse_index(d1), per_mouse_index(d2)
    both = pd.concat([a.rename("d1"), b.rename("d2")], axis=1)
    for _, r in both.dropna().iterrows():
        ax.plot([0, 1], [r["d1"], r["d2"]], color="#BDBDBD", lw=0.5, zorder=1)
    for x, v in ((0, a), (1, b)):
        ax.scatter(np.full(len(v), x), v, s=5, color="#424242", lw=0, zorder=2)
        ax.plot([x - 0.2, x + 0.2], [v.median()] * 2, color="black", lw=1.6, zorder=3)
        ax.text(x, 0.55, stars(wilcoxon(v).pvalue), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["day 1", "day 2"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.35, 0.65)
    ax.set_ylabel("EE-SC index of time")
    ax.set_title("(c) both test days", loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_model(mv, path):
    res = json.load(open(os.path.join(mv, "results.json")))
    grain = json.load(open(os.path.join(mv, "grain_results.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), gridspec_kw={"width_ratios": [0.9, 1.0, 1.5]})

    ax = axes[0]
    grid = res["recovery"]["parameter_recovery"]["grid"]
    tr = [g["wS_true"] for g in grid]
    mean = [g["wS_recovered_mean"] for g in grid]
    lo = [g["wS_recovered_mean"] - g["ci"][0] for g in grid]
    hi = [g["ci"][1] - g["wS_recovered_mean"] for g in grid]
    ax.plot([-0.2, 2.2], [-0.2, 2.2], color="grey", lw=0.6, ls=(0, (4, 2)))
    ax.errorbar(tr, mean, [lo, hi], fmt="o", ms=3, color="#2E7D32", capsize=2)
    ax.set_xlabel("True semantic weight")
    ax.set_ylabel("Recovered (95% interval)")
    ax.set_title("(a) recovery", loc="left")

    ax = axes[1]
    wt = grain["weight_table"]
    rows = [("per block", wt["A_per_block"]["wS"]), ("per mouse,\ncell means", wt["B_cell_mean"]["wS"]),
            ("per level,\nEE-SC index", wt["C_tier_PI"]["wS"])]
    for i, (lab, w) in enumerate(rows):
        ax.errorbar(w["mean"], i, xerr=[[w["mean"] - w["hdi"][0]], [w["hdi"][1] - w["mean"]]],
                    fmt="o", ms=3.5, color="#2E7D32", capsize=2)
    ax.axvline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Semantic weight $w_S$ (95% HDI)")
    ax.set_title("(b) by grain", loc="left")

    ax = axes[2]
    cells = ["EE\ndom.", "EE\nsec.", "EE\nrare", "SC\ndom.", "SC\nsec.", "SC\nrare", "silent"]
    obs, pred = grain["grainB_observed_mean"], grain["grainB_ppc_predicted"]
    x = np.arange(7)
    colours = [EE] * 3 + [SC] * 3 + ["#9E9E9E"]
    ax.bar(x - 0.2, obs, 0.38, color=colours, alpha=0.9, label="observed")
    ax.bar(x + 0.2, pred, 0.38, color="white", edgecolor=colours, lw=0.9, label="model")
    ax.axhline(1 / 7, color="grey", lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=5.5)
    ax.set_ylabel("Share of time (mean over mice)")
    ax.set_title("(c) observed and predicted", loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--thesis-figures", metavar="DIR", required=True)
    args = parser.parse_args()
    d = pd.read_csv(os.path.join(args.root, "BATCH_ANALYSIS", "exp2", "arm_blocks.csv"))
    d["mouse"] = d["mouse"].astype(str)
    with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42}):
        fig_session(d, os.path.join(args.thesis_figures, "fig_exp2_session"))
        fig_model(os.path.join(args.root, "model_validation_output"),
                  os.path.join(args.thesis_figures, "fig_exp2_model"))
    print("written to", args.thesis_figures)


if __name__ == "__main__":
    main()
