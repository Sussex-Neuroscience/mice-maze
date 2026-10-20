#!/usr/bin/env python3
"""
Experiment 1: Individual Mice
=============================

How the averages of Experiment 1 divide among the animals behind them.

1. Each stimulus against the silent arm, as in Figure 4.10, with the mixed model
   of each set beside the indices of the mice: log mean visit length on the
   stimulus, the silent arm the reference, the mouse, session and arm as random
   effects (exp1_individuals.R). The effect is drawn as the index it corresponds
   to, tanh(log ratio / 2), with its 95% interval; filled where it survives a Holm
   correction within the set.
2. Every mouse's index for every stimulus on every day, as a grid, and one mouse
   whose session departs from the group, in close-up.
3. The vocalisation against the silent arm, day by day, one line per mouse.
4. How the variance of the per-session index divides between the day, the
   stimulus, the mouse and the session (a mouse on one day), with bootstrap
   intervals, and again with the silent arm's own visits taken out, since every
   index of a session shares that one arm; and the share of variance between mice
   of the vocalisation, the synthetic and the vocalisation against synthetic index.
5. A principal component analysis of what each session did, to see whether
   sessions group by mouse or by day, with each mouse's scores day by day.

A session's index of a stimulus is the thesis index on mean visit length over the
session's four sound blocks, (a - b) / (a + b), with the silent arm of the same
blocks as b. A mouse that ran a set twice is averaged where one value per mouse is
drawn, and kept as two sessions where sessions are the unit.

Usage:
    python exp1_individuals.py [DATA_DIR] [--thesis-figures DIR]

Reads DATA_DIR/BATCH_ANALYSIS/lmm/arm_blocks.csv (exp1_lmm.py) and
stimulus_pi/stimulus_pi.csv (exp1_stimulus_pi.py). Outputs go to
DATA_DIR/BATCH_ANALYSIS/individuals/; with --thesis-figures the figures are also
written as fig_exp1_stimulus_pi_lmm, fig_exp1_individuals, fig_exp1_voc_by_day,
fig_exp1_variance and fig_exp1_pca, under new names, for review before any goes
into the thesis.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp1_lmm import rscript
from exp1_stimulus_pi import COLOURS, label, order_key, star

SET_ORDER = ["TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"]
CMAP = plt.get_cmap("RdBu_r")
NORM = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
INK = "#1F3864"
GREY = "#9E9E9E"


def pi(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else np.nan


def mean_visit(g):
    v = g["visits"].sum()
    return g["time_s"].sum() / v if v > 0 else np.nan


# -- per session --------------------------------------------------------------
def session_tables(d):
    """Each stimulus against the silent arm per session, the class indices, and the
    behaviour of each session that the principal component analysis uses."""
    stim_rows, cls_rows = [], []
    for (mouse, set_name, session), g in d.groupby(["mouse", "set", "session"]):
        silent = mean_visit(g[g["family"] == "silent"])
        if np.isfinite(silent):
            for (stim, fam), h in g[g["family"] != "silent"].groupby(["stimulus", "family"]):
                stim_rows.append({"mouse": mouse, "set": set_name, "session": session,
                                  "stimulus": stim, "family": fam,
                                  "pi": pi(mean_visit(h), silent), "log_silent": np.log(silent)})
        voc, syn = mean_visit(g[g["cls"] == "vocalisation"]), mean_visit(g[g["cls"] == "synthetic"])
        blocks = g["block"].nunique()
        t_arm = g.groupby("arm")["time_s"].sum()
        p = t_arm / t_arm.sum() if t_arm.sum() > 0 else t_arm * np.nan
        evenness = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(8)) if t_arm.sum() > 0 else np.nan
        cls_rows.append({
            "mouse": mouse, "set": set_name, "session": session,
            "voc_vs_silent": pi(voc, silent), "synth_vs_silent": pi(syn, silent),
            "voc_vs_synth": pi(voc, syn),
            "entries_per_block": g["visits"].sum() / blocks,
            "log_visit_s": np.log(mean_visit(g)) if np.isfinite(mean_visit(g)) else np.nan,
            "time_per_block_s": g["time_s"].sum() / blocks,
            "evenness": evenness})
    return pd.DataFrame(stim_rows), pd.DataFrame(cls_rows)


# -- figures -------------------------------------------------------------------
def fig_stimulus_lmm(st, lmm, path):
    """Figure 4.10 with the mixed model of each set beside the mice."""
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.8))
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for k, (ax, set_name) in enumerate(zip(axes, SET_ORDER)):
        d = st[st["set"] == set_name].dropna(subset=["pi"])
        names = sorted({(n, f) for n, f in zip(d["stimulus"], d["family"])},
                       key=lambda nf: order_key(*nf))
        names = [nf for nf in names if (d["stimulus"] == nf[0]).sum() >= 5]
        m = lmm[lmm["set"] == set_name].set_index("stimulus")
        for i, (name, fam) in enumerate(names):
            colour = COLOURS.get(fam, "#424242")
            if name == "vocalisation":
                ax.axvspan(i - 0.48, i + 0.48, color="#E8F5E9", zorder=0, lw=0)
            vals = d.loc[d["stimulus"] == name, "pi"].to_numpy()
            ax.scatter(i - 0.12 + rng.uniform(-0.12, 0.12, len(vals)), vals, s=4, color=colour,
                       alpha=0.5, lw=0, zorder=2)
            ax.plot([i - 0.32, i + 0.08], [np.median(vals)] * 2, color=colour, lw=2, zorder=3)
            hp = d.loc[d["stimulus"] == name, "holm"]
            mark = star(hp.iloc[0]) if hp.notna().any() else ""
            if mark:
                ax.text(i - 0.12, 0.93, mark, ha="center", va="bottom", fontsize=6.5)
            if name in m.index:
                r = m.loc[name]
                ax.errorbar(i + 0.27, r["pi"], yerr=[[r["pi"] - r["pi_low"]], [r["pi_high"] - r["pi"]]],
                            fmt="D", ms=3.2, color=colour, mfc=colour if r["p.holm"] < 0.05 else "white",
                            mew=0.9, lw=0.9, capsize=1.5, zorder=4)
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([label(n) for n, _ in names], rotation=60, ha="right", fontsize=5.5)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(f"({'abcde'[k]}) {set_name}", loc="left")
        ax.set_ylim(-0.9, 1.05)
        if k % 3:
            ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # the sixth panel holds the key
    key = axes[5]
    key.axis("off")
    key.scatter([0.08] * 3, [0.78, 0.81, 0.75], s=4, color=INK, alpha=0.5, lw=0)
    key.plot([0.03, 0.13], [0.78, 0.78], color=INK, lw=2)
    key.text(0.2, 0.78, "mice, and their median\n(stars, Wilcoxon, Holm in the set)",
             va="center", fontsize=6)
    key.errorbar(0.08, 0.52, yerr=0.06, fmt="D", ms=3.2, color=INK, mfc=INK, capsize=1.5, lw=0.9)
    key.text(0.2, 0.52, "mixed model, the index its\neffect corresponds to, 95% interval\n"
                        "(filled when Holm p < 0.05)", va="center", fontsize=6)
    key.add_patch(plt.Rectangle((0.03, 0.22), 0.1, 0.1, color="#E8F5E9", lw=0))
    key.text(0.2, 0.27, "the vocalisation arm", va="center", fontsize=6)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Preference index\nagainst the silent arm")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_individuals(st, d, cls, example, path):
    """Every mouse's index for every stimulus of every set, and one mouse in close-up."""
    per_mouse, voc_mean = per_mouse_table(st)
    mice = list(voc_mean.sort_values(ascending=False).index)
    mice += sorted(set(per_mouse["mouse"]) - set(mice))

    fig = plt.figure(figsize=(6.3, 6.0))
    widths = []
    cols = {}
    for s in SET_ORDER:
        x = per_mouse[per_mouse["set"] == s]
        names = sorted({(n, f) for n, f in zip(x["stimulus"], x["family"])},
                       key=lambda nf: order_key(*nf))
        cols[s] = names
        widths.append(len(names))
    gs = fig.add_gridspec(2, len(SET_ORDER) + 1, width_ratios=widths + [0.35],
                          height_ratios=[3.2, 1.35], hspace=0.55, wspace=0.12)
    for k, s in enumerate(SET_ORDER):
        ax = fig.add_subplot(gs[0, k])
        x = per_mouse[per_mouse["set"] == s].pivot_table(index="mouse", columns="stimulus",
                                                         values="pi")
        grid = x.reindex(index=mice, columns=[n for n, _ in cols[s]])
        ax.set_facecolor("#ECEFF1")
        ax.imshow(np.ma.masked_invalid(grid.to_numpy(float)), cmap=CMAP, norm=NORM,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(cols[s])))
        ax.set_xticklabels([label(n) for n, _ in cols[s]], rotation=70, ha="right", fontsize=5)
        for t, (n, f) in zip(ax.get_xticklabels(), cols[s]):
            t.set_color(COLOURS.get(f, "#424242"))
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(f"{len(mice)} mice, ordered by their\nmean vocalisation index", labelpad=16)
            r = mice.index(example)
            ax.annotate("", xy=(-0.6, r), xytext=(-2.2, r), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
        ax.set_title(f"({'abcde'[k]}) {s}", loc="left", fontsize=6.5)
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax)
    cb.set_label("index against the silent arm", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    close_up(fig.add_subplot(gs[1, :4]), d, cls, example,
             f"(f) one mouse in close-up, {example.replace('mouse', 'mouse ')}, day by day",
             rotation=0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_mouse_table(st):
    """One index per mouse, set and stimulus (a set run twice is averaged), and each
    mouse's mean vocalisation index over the days."""
    per_mouse = st.groupby(["mouse", "set", "stimulus", "family"], as_index=False)["pi"].mean()
    voc_mean = per_mouse[per_mouse["family"] == "vocalisation"].groupby("mouse")["pi"].mean()
    return per_mouse, voc_mean


def pick_example(d, cls):
    """The mouse for the close-up: of the mice that ran at least four days with at
    least ten visits to the vocalisation and ten to the silent arm on each, the one
    with the vocalisation above the silent arm on the most days, then on average."""
    visits = d.groupby(["mouse", "set", "cls"])["visits"].sum().unstack()
    pm = cls.groupby(["mouse", "set"])["voc_vs_silent"].mean().to_frame().join(visits)
    pm = pm.dropna(subset=["voc_vs_silent"])
    s = pm.groupby("mouse").agg(days=("voc_vs_silent", "size"),
                                above=("voc_vs_silent", lambda x: int((x > 0).sum())),
                                mean=("voc_vs_silent", "mean"),
                                voc=("vocalisation", "min"), silent=("silent", "min"))
    s = s[(s["days"] >= 4) & (s["voc"] >= 10) & (s["silent"] >= 10)]
    return s.sort_values(["above", "mean"], ascending=False).index[0]


def close_up(ax, d, cls, example, title, rotation, right=False):
    """The close-up mouse day by day: mean visit on the silent arm, the synthetic
    sounds and the vocalisation, with its vocalisation index against the silent arm
    above each day."""
    g = d[d["mouse"] == example]
    sets = [s for s in SET_ORDER if s in set(g["set"])]
    kinds = [("silent", GREY, "silent arm"), ("synthetic", "#455A64", "synthetic sounds"),
             ("vocalisation", "#2E7D32", "vocalisation")]
    top = 0.0
    for i, s in enumerate(sets):
        h = g[g["set"] == s]
        heights = []
        for j, (c, colour, name) in enumerate(kinds):
            mv = mean_visit(h[h["cls"] == c])
            if np.isfinite(mv):
                ax.bar(i + (j - 1) * 0.27, mv, width=0.26, color=colour,
                       label=name if i == 0 else None)
                heights.append(mv)
        idx = cls.loc[(cls["mouse"] == example) & (cls["set"] == s), "voc_vs_silent"].mean()
        ax.text(i, max(heights) * 1.03, f"{idx:+.2f}", ha="center", va="bottom", fontsize=5.5,
                color="#2E7D32")
        top = max(top, max(heights))
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, fontsize=5.5, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel("Mean visit (s)")
    ax.set_title(title, loc="left", fontsize=7)
    ax.set_ylim(0, top * 1.45)
    ax.legend(fontsize=5, frameon=False, loc="upper left", handlelength=1, borderaxespad=0.1)
    ax.spines["top"].set_visible(False)
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["right"].set_visible(False)


LINE_CMAP = plt.get_cmap("coolwarm")
LINE_NORM = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)


def fig_individual_lines(st, d, cls, example, path):
    """Figure 4.10 drawn mouse by mouse: each mouse a line through its index for each
    stimulus of the day, coloured by its mean vocalisation index over the days, the
    close-up mouse in dark blue; and that mouse day by day."""
    per_mouse, voc_mean = per_mouse_table(st)
    fig = plt.figure(figsize=(6.3, 4.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.75, wspace=0.18)
    for k, s in enumerate(SET_ORDER):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        x = per_mouse[per_mouse["set"] == s]
        names = sorted({(n, f) for n, f in zip(x["stimulus"], x["family"])},
                       key=lambda nf: order_key(*nf))
        names = [nf for nf in names if (x["stimulus"] == nf[0]).sum() >= 5]
        grid = x.pivot_table(index="mouse", columns="stimulus", values="pi")
        grid = grid.reindex(columns=[n for n, _ in names])
        xs = np.arange(len(names))
        if "vocalisation" in grid.columns:
            ax.axvspan(list(grid.columns).index("vocalisation") - 0.4,
                       list(grid.columns).index("vocalisation") + 0.4,
                       color="#E8F5E9", zorder=0, lw=0)
        for mouse, row in grid.iterrows():
            y = row.to_numpy(float)
            ok = np.isfinite(y)
            if mouse == example:
                continue
            colour = LINE_CMAP(LINE_NORM(voc_mean.get(mouse, 0.0)))
            ax.plot(xs[ok], y[ok], color=colour, lw=0.6, alpha=0.75, zorder=2)
            ax.scatter(xs[ok], y[ok], color=colour, s=3, lw=0, alpha=0.9, zorder=2)
        if example in grid.index:
            y = grid.loc[example].to_numpy(float)
            ok = np.isfinite(y)
            ax.plot(xs[ok], y[ok], color=INK, lw=1.3, marker="o", ms=2.2, zorder=4)
            if s == "TEM":
                ax.text(xs[ok][-1] + 0.15, y[ok][-1], example.replace("mouse", ""), color=INK,
                        fontsize=5.5, va="center", zorder=5)
        med = grid.median()
        for i, (n, f) in enumerate(names):
            ax.plot([i - 0.22, i + 0.22], [med[n]] * 2, color="black", lw=2.2, zorder=3,
                    solid_capstyle="butt")
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([label(n) for n, _ in names], rotation=60, ha="right", fontsize=5.5)
        for t, (n, f) in zip(ax.get_xticklabels(), names):
            t.set_color(COLOURS.get(f, "#424242"))
        ax.set_xlim(-0.5, len(names) - 0.5 + (0.6 if s == "TEM" else 0))
        ax.set_ylim(-1.0, 1.0)
        if k % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Preference index\nagainst the silent arm")
        ax.set_title(f"({'abcde'[k]}) {s}, {len(grid)} mice", loc="left", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LINE_NORM, cmap=LINE_CMAP), cax=cax,
                      extend="both")
    cb.set_label("the mouse's mean\nvocalisation index", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    close_up(fig.add_subplot(gs[1, 2]), d, cls, example,
             f"(f) {example.replace('mouse', 'mouse ')}, day by day", rotation=45, right=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_voc_by_day(cls, icc, path):
    """The vocalisation against the silent arm, day by day, one line per mouse."""
    pm = cls.groupby(["mouse", "set"])["voc_vs_silent"].mean().unstack().reindex(columns=SET_ORDER)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.arange(len(SET_ORDER))
    for mouse, row in pm.iterrows():
        y = row.to_numpy(float)
        colour = CMAP(NORM(np.nanmean(y)))
        ok = np.isfinite(y)
        ax.plot(xs[ok], y[ok], color=colour, lw=0.7, alpha=0.75, zorder=2)
        ax.scatter(xs[ok], y[ok], color=colour, s=6, zorder=3, lw=0)
    med = pm.median()
    q1, q3 = pm.quantile(0.25), pm.quantile(0.75)
    ax.errorbar(xs + 0.18, med, yerr=[med - q1, q3 - med], fmt="o", color="black", ms=3.5,
                capsize=2, lw=1.2, zorder=4)
    ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels(SET_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Vocalisation against\nthe silent arm")
    r = icc.set_index("index").loc["voc_vs_silent"]
    ax.text(0.02, 0.97, f"between mice {r['icc'] * 100:.0f}% of the variance "
                        f"(95% interval {r['low'] * 100:.0f} to {r['high'] * 100:.0f}%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK)
    ax.set_ylim(-1.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


FEATURES = {"synth_vs_silent": "synthetic against silent", "voc_vs_silent": "vocalisation against silent",
            "voc_vs_synth": "vocalisation against synthetic", "entries_per_block": "entries a block",
            "log_visit_s": "mean visit (log)", "time_per_block_s": "time in the arms",
            "evenness": "spread over the arms"}
SET_COLOUR = {"TEM": "#00838F", "Intervals 1": "#1565C0", "Intervals 2": "#7E57C2",
              "Sequences": INK}


def pca(cls):
    """Principal components of what each session did, on the four days with synthetic
    sounds. The first is turned to grow with activity and the second with the
    vocalisation, so that their signs read the same way as the features."""
    x = cls[cls["set"].isin(SET_ORDER[:4])].dropna(subset=list(FEATURES)).copy()
    z = (x[list(FEATURES)] - x[list(FEATURES)].mean()) / x[list(FEATURES)].std(ddof=1)
    u, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    scores = u * s
    sign = np.array([np.sign(vt[0, list(FEATURES).index("entries_per_block")]),
                     np.sign(vt[1, list(FEATURES).index("voc_vs_silent")])])
    x["pc1"], x["pc2"] = scores[:, 0] * sign[0], scores[:, 1] * sign[1]
    explained = s ** 2 / (s ** 2).sum()
    loadings = pd.DataFrame(vt[:2].T * sign, index=list(FEATURES), columns=["pc1", "pc2"])
    return x, explained, loadings


def fig_variance(var, var_adj, icc, path):
    """How the variance of the per-session index divides, with and without the silent
    arm's own visits, and the share between mice of three indices."""
    fig = plt.figure(figsize=(6.3, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.75)

    ax = fig.add_subplot(gs[0, 0])
    names = {"set": "the day (stimulus set)", "stim_in_set": "the stimulus, within its set",
             "mouse": "the mouse", "session": "the session\n(a mouse on one day)",
             "silent_visit": "the silent arm's\nown visits",
             "Residual": "left over (a stimulus\nwithin a session)"}
    colours = [GREY, GREY, "#1565C0", "#E65100", "#6D4C41", "#90A4AE"]
    v = var.set_index("component").reindex(list(names))
    a = var_adj.set_index("component").reindex(list(names))
    ys = np.arange(len(names))[::-1].astype(float)
    ax.barh(ys + 0.19, v["share"].fillna(0) * 100, color=colours, height=0.36)
    ax.errorbar(v["share"] * 100, ys + 0.19, xerr=[(v["share"] - v["low"]) * 100,
                                                  (v["high"] - v["share"]) * 100],
                fmt="none", ecolor=INK, lw=0.7, capsize=1.3)
    ax.barh(ys - 0.19, a["share"].fillna(0) * 100, color=colours, height=0.36, alpha=0.45,
            hatch="//////", edgecolor="white", lw=0)
    for y_, val in zip(ys, a["share"]):
        if np.isfinite(val):
            ax.text(val * 100 + 1, y_ - 0.19, f"{val * 100:.0f}%", va="center", fontsize=5.3, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(list(names.values()), fontsize=6)
    ax.set_xlabel("Share of the variance (%)")
    ax.set_title("(a) the index of each stimulus in each session", loc="left", fontsize=7)
    ax.legend(handles=[Patch(color=GREY, label="as measured (95% interval)"),
                       Patch(facecolor=GREY, alpha=0.45, hatch="//////", edgecolor="white",
                             label="the silent arm's visits taken out")],
              fontsize=5.5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    rows = {"voc_vs_silent": "vocalisation\nagainst silent", "synth_vs_silent": "synthetic\nagainst silent",
            "voc_vs_synth": "vocalisation\nagainst synthetic"}
    c = icc.set_index("index").reindex(list(rows))
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, c["icc"] * 100, color=["#2E7D32", "#455A64", "#81C784"], height=0.55)
    ax.errorbar(c["icc"] * 100, ys, xerr=[(c["icc"] - c["low"]) * 100, (c["high"] - c["icc"]) * 100],
                fmt="none", ecolor=INK, lw=0.8, capsize=1.5)
    for y_, val in zip(ys, c["icc"]):
        ax.text(1, y_ + 0.36, f"{val * 100:.0f}%", va="center", fontsize=5.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(list(rows.values()), fontsize=6)
    ax.set_xlim(0, 70)
    ax.set_xlabel("Share of the variance between mice (%)")
    ax.set_title("(b) how much of an index is the mouse", loc="left", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pca(x, explained, loadings, icc, path):
    """The sessions in the space of their first two components, what the components
    weigh, and each mouse's scores from day to day."""
    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.6, wspace=1.0)

    ax = fig.add_subplot(gs[0, 0])
    for mouse, g in x.groupby("mouse"):
        if len(g) > 1:
            g = g.assign(o=g["set"].map(SET_ORDER.index)).sort_values("o")
            ax.plot(g["pc1"], g["pc2"], color="#CFD8DC", lw=0.6, zorder=1)
    for s in SET_ORDER[:4]:
        g = x[x["set"] == s]
        ax.scatter(g["pc1"], g["pc2"], s=9, color=SET_COLOUR[s], label=s, zorder=2, lw=0)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4)
    ax.set_xlabel(f"PC1, activity ({explained[0] * 100:.0f}% of the variance)")
    ax.set_ylabel(f"PC2, the vocalisation ({explained[1] * 100:.0f}%)")
    ax.legend(fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none", loc="lower left",
              handletextpad=0.2, borderaxespad=0.2)
    ax.set_title("(a) sessions, joined by mouse", loc="left", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(loadings))[::-1]
    ax.barh(ys + 0.18, loadings["pc1"], height=0.34, color="#455A64", label="PC1")
    ax.barh(ys - 0.18, loadings["pc2"], height=0.34, color="#81C784", label="PC2")
    ax.set_yticks(ys)
    ax.set_yticklabels([FEATURES[f] for f in loadings.index], fontsize=5.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Loading")
    ax.set_xlim(-0.8, 0.8)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("(b) what the components weigh", loc="left", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    c = icc.set_index("index")
    xs = np.arange(4)
    for k, (pc, name) in enumerate((("pc1", "PC1, activity"), ("pc2", "PC2, the vocalisation"))):
        ax = fig.add_subplot(gs[1, k])
        pm = x.groupby(["mouse", "set"])[pc].mean().unstack().reindex(columns=SET_ORDER[:4])
        for mouse, row in pm.iterrows():
            y = row.to_numpy(float)
            ok = np.isfinite(y)
            ax.plot(xs[ok], y[ok], color="#90A4AE", lw=0.6, alpha=0.7, zorder=1)
            ax.scatter(xs[ok], y[ok], color=[SET_COLOUR[s] for s, o in zip(SET_ORDER[:4], ok) if o],
                       s=5, lw=0, zorder=2)
        med = pm.median()
        ax.plot(xs + 0.15, med, "o", color="black", ms=3, zorder=3)
        ax.axhline(0, color="black", lw=0.5, ls=(0, (4, 2)))
        ax.set_xticks(xs)
        ax.set_xticklabels(SET_ORDER[:4], rotation=20, ha="right")
        ax.set_ylabel(f"{name} score")
        r = c.loc[pc]
        ax.set_title(f"({'cd'[k]}) {name}, mouse by mouse\nbetween mice {r['icc'] * 100:.0f}% "
                     f"(95% interval {r['low'] * 100:.0f} to {r['high'] * 100:.0f}%)",
                     loc="left", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    batch = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(batch, "individuals")
    os.makedirs(out, exist_ok=True)

    d = pd.read_csv(os.path.join(batch, "lmm", "arm_blocks.csv"))
    d = d[d["set"].isin(SET_ORDER)].copy()
    d.to_csv(os.path.join(out, "arm_blocks.csv"), index=False)
    st_session, cls = session_tables(d)
    x, explained, loadings = pca(cls)
    cls = cls.merge(x[["session", "pc1", "pc2"]], on="session", how="left")
    st_session.to_csv(os.path.join(out, "session_pi.csv"), index=False)
    cls.to_csv(os.path.join(out, "session_cls.csv"), index=False)
    loadings.to_csv(os.path.join(out, "pca_loadings.csv"))
    st = pd.read_csv(os.path.join(batch, "stimulus_pi", "stimulus_pi.csv"))

    r = subprocess.run([rscript(), os.path.join(HERE, "exp1_individuals.R"), out])
    if r.returncode:
        sys.exit(r.returncode)
    lmm = pd.read_csv(os.path.join(out, "stimulus_lmm.csv"))
    var = pd.read_csv(os.path.join(out, "variance_components.csv"))
    var_adj = pd.read_csv(os.path.join(out, "variance_components_adjusted.csv"))
    icc = pd.read_csv(os.path.join(out, "icc.csv"))

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("Rank of the vocalisation among the stimuli of each set, by median index over mice:")
    for s in SET_ORDER[:4]:
        med = st[st["set"] == s].groupby("stimulus")["pi"].median().sort_values(ascending=False)
        say(f"  {s:12s} {list(med.index).index('vocalisation') + 1} of {len(med)} "
            f"(vocalisation {med['vocalisation']:+.3f}, best other "
            f"{med.drop('vocalisation').iloc[0]:+.3f})")
    v = st[st["set"] == "Vocalisations"].groupby("stimulus")["pi"].median()
    say(f"  Vocalisations: {int((v > 0).sum())} of {len(v)} recordings with a median above zero")
    m = lmm[lmm["set"] == "Vocalisations"]
    say(f"  mixed model, vocalisation day: {int((m['pi'] > 0).sum())} of {len(m)} recordings "
        f"above the silent arm, {int((m['p.holm'] < 0.05).sum())} with Holm p < 0.05")
    sil = st_session.groupby("session").agg(mean_pi=("pi", "mean"), log_silent=("log_silent", "first"))
    rho = sil["mean_pi"].corr(sil["log_silent"], method="spearman")
    say(f"\nA session's mean index against the log of its silent arm's mean visit: "
        f"Spearman rho {rho:.2f}, r squared {sil['mean_pi'].corr(sil['log_silent']) ** 2:.2f}, "
        f"{len(sil)} sessions")
    say("\nPCA of sessions (TEM, intervals 1 and 2, sequences): "
        + ", ".join(f"PC{i + 1} {e * 100:.0f}%" for i, e in enumerate(explained[:3])))
    say(loadings.round(2).to_string())
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42, "hatch.linewidth": 0.6}):
        targets = [out] + ([args.thesis_figures] if args.thesis_figures else [])
        example = pick_example(d, cls)
        for t in targets:
            fig_stimulus_lmm(st, lmm, os.path.join(t, "fig_exp1_stimulus_pi_lmm"))
            fig_individuals(st_session, d, cls, example, os.path.join(t, "fig_exp1_individuals"))
            fig_individual_lines(st_session, d, cls, example,
                                 os.path.join(t, "fig_exp1_individuals_lines"))
            fig_voc_by_day(cls, icc, os.path.join(t, "fig_exp1_voc_by_day"))
            fig_variance(var, var_adj, icc, os.path.join(t, "fig_exp1_variance"))
            fig_pca(x, explained, loadings, icc, os.path.join(t, "fig_exp1_pca"))
    print(f"\nclose-up: {example}")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
