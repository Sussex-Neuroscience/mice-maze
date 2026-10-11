#!/usr/bin/env python3
"""
Experiment 1: Entering Against Staying
======================================

The preference index measures staying: how long a visit lasts once the mouse
has entered an arm. This script measures entering, how often the mouse enters
an arm at all, and sets the two against each other.

The unit is the arm in one sound block (the table exp1_lmm.py writes). The
number of entries is the number of visits. In sessions logged only as totals
per arm, an arm never entered was left blank rather than written as zero, so
a blank count is a zero there; a session that logged no visit anywhere, in its
habituation, its silent blocks or its sound blocks, is a failed recording and
is left out.

An arm plays its sound only once entered, so a first entry cannot be guided by
the sound; what the sound can change is whether the mouse comes back. But the
first sound block of a session always opened on the same layout, the order in
which the stimuli were listed, with the silent arm and the vocalisation in two
of the arms the mice favoured anyway. Only the later blocks were reshuffled.
Every measure is therefore given over all four sound blocks, over the shuffled
blocks 2 to 4 alone and over block 1 alone.

Indices, per mouse, from its arm-blocks pooled over its sessions (the silent arm
and the vocalisation are one arm each, often not entered in a block, and an
index of a single session would swing to -1 or +1 on one entry):

    entering:  (entries per arm with a - entries per arm with b) / (the sum)
    staying:   (mean visit length with a - mean visit length with b) / (the sum)

for the vocalisation against synthetic sounds, synthetic sounds against the
silent arm and the vocalisation against the silent arm: over the four stimulus
sets that offer all three (each mouse's sessions pooled), and per set. Tests:
Wilcoxon signed-rank against zero across mice, and Spearman correlations of
entering with staying across mice.

The confirmatory test is a Poisson mixed model of the entries (exp1_approach.R,
lme4), with an observation-level random effect for overdispersion and the
fixed and random effects of the model of visit length, over all blocks and
over the shuffled blocks.

Block 1 is not only place. The favoured arms 1, 2, 7 and 8 held silence or the
vocalisation in block 1 on some days and a synthetic sound on others, so each
can be compared with itself: its share of the entries and of the time in
block 1 over its share in the silent habituation block of the same session
(log2, with one percentage point added to both so that an empty arm counts),
when it held silence or the vocalisation, against when it held a synthetic
sound. Same arm, same block, so place is held fixed; the day is not.

Usage:
    python exp1_approach.py [DATA_DIR] [--thesis-figures DIR]

    run exp1_lmm.py and exp1_order_place.py first.

Outputs (DATA_DIR/BATCH_ANALYSIS/approach/):
    entries.csv, indices_per_mouse.csv, stats.txt, glmm_results.txt,
    glmm_contrasts.csv, fig_approach.png/.pdf
    and with --thesis-figures, fig_exp1_approach
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp1_lmm import rscript

SET_ORDER = ["TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"]
FOUR = ["TEM", "Intervals 1", "Intervals 2", "Sequences"]
CLASSES = ["silent", "synthetic", "vocalisation"]
CONTRASTS = [("voc_vs_synth", "vocalisation", "synthetic", "Vocalisation against\nsynthetic sounds"),
             ("synth_vs_silent", "synthetic", "silent", "Synthetic sounds against\nthe silent arm"),
             ("voc_vs_silent", "vocalisation", "silent", "Vocalisation against\nthe silent arm")]
BLOCKS = {"all": "all four sound blocks", "shuffled": "shuffled blocks 2 to 4",
          "first": "block 1 alone"}
FAVOURED = ["ROI1", "ROI2", "ROI7", "ROI8"]
ENTER_COLOUR, STAY_COLOUR = "#6A9BC3", "#D08C4B"


def logged_sessions(data_dir, sessions):
    """Sessions with at least one visit logged anywhere: habituation, silent or sound blocks."""
    keep = set()
    for s in sessions:
        files = glob.glob(os.path.join(data_dir, "*", s, "trials_*.csv"))
        dv = glob.glob(os.path.join(data_dir, "*", s, "*detailed_visits*.csv"))
        n = 0
        if files:
            n += pd.read_csv(files[0])["visitation_count"].fillna(0).sum()
        if dv:
            n += len(pd.read_csv(dv[0]))
        if n > 0:
            keep.add(s)
    return keep


def subset(e, blocks):
    return {"all": e, "shuffled": e[e["block"] > 1], "first": e[e["block"] == 1]}[blocks]


def pooled_indices(e, by):
    """Entering and staying indices from the arm-blocks of each group (by) pooled."""
    ent = e.groupby(by + ["cls"])["visits"].mean().unstack().reindex(columns=CLASSES)
    t = e.groupby(by + ["cls"])[["time_s", "visits"]].sum()
    stay = (t["time_s"] / t["visits"].where(t["visits"] > 0)).unstack().reindex(columns=CLASSES)
    out = {}
    for col, a, b, _ in CONTRASTS:
        out[f"enter_{col}"] = (ent[a] - ent[b]) / (ent[a] + ent[b])
        out[f"stay_{col}"] = (stay[a] - stay[b]) / (stay[a] + stay[b])
    return pd.DataFrame(out).reset_index()


def per_mouse(e):
    """Indices per mouse over the four sets with all three classes, all blocks and shuffled."""
    four = e[e["set"].isin(FOUR)]
    return pd.concat([pooled_indices(subset(four, b), ["mouse"]).assign(blocks=b) for b in BLOCKS],
                     ignore_index=True)


def first_block_layout(t):
    """Share of sessions in which each stimulus sat in its usual arm, in block 1 and later."""
    rows = []
    for (set_name, block), g in t.groupby(["set", "block"]):
        c = pd.crosstab(g["stimulus"], g["arm"])
        share = c.max(axis=1) / c.sum(axis=1)
        usual = c.idxmax(axis=1)
        rows.append({"set": set_name, "block": block, "min_share": share.min(),
                     "max_share": share.max(),
                     "silent_arm": usual.get("silent", ""),
                     "vocalisation_arm": usual.get("vocalisation", "")})
    return pd.DataFrame(rows)


def same_arm_block1(t, at):
    """Block 1, each favoured arm against its own habituation share, by what it held."""
    hab = at[at["trial"] == 1].set_index(["session", "arm"])
    b1 = at[(at["trial"] == 2) & at["arm"].isin(FAVOURED)].merge(
        t.loc[t["block"] == 1, ["session", "arm", "cls"]], on=["session", "arm"])
    b1 = b1[b1["cls"].isin(CLASSES)].copy()
    key = pd.MultiIndex.from_frame(b1[["session", "arm"]])
    for m in ("n", "t"):
        expected = hab[f"{m}_share"].reindex(key).to_numpy()
        b1[f"{m}_lr"] = np.log2((b1[f"{m}_share"] + 0.01) / (expected + 0.01))
    return b1


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def draw(ind, path, thesis=False):
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.5) if thesis else (14, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    rng = np.random.default_rng(0)
    columns = [("enter", "all"), ("enter", "shuffled"), ("stay", "all"), ("stay", "shuffled")]
    for k, (ax, (col, _, _, name)) in enumerate(zip(axes[:3], CONTRASTS)):
        for i, (measure, blocks) in enumerate(columns):
            x = ind.loc[ind["blocks"] == blocks, f"{measure}_{col}"].dropna()
            colour = ENTER_COLOUR if measure == "enter" else STAY_COLOUR
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=4 if thesis else 8,
                       color=colour, lw=0, zorder=2)
            ax.plot([i - 0.25, i + 0.25], [x.median()] * 2, color="#1F3864", lw=2, zorder=3)
            ax.text(i, 0.97, stars(wilcoxon(x).pvalue), ha="center", va="bottom",
                    fontsize=7)
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.axvline(1.5, color="grey", lw=0.4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["all", "2-4", "all", "2-4"])
        for x0, label in ((0.5, "entering"), (2.5, "staying")):
            ax.text(x0, -0.2, label, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=7 if thesis else 9)
        ax.set_ylim(-1.05, 1.12)
        ax.set_title(f"({'abc'[k]}) {name[0].lower() + name[1:]}" if thesis else name,
                     loc="left" if thesis else "center", fontsize=7 if thesis else 9)
        if k == 0:
            ax.set_ylabel("Index")
        else:
            ax.set_yticklabels([])
    # (d) entering against staying, vocalisation against synthetic, shuffled blocks
    ax = axes[3]
    m = ind.loc[ind["blocks"] == "shuffled", ["enter_voc_vs_synth", "stay_voc_vs_synth"]].dropna()
    ax.scatter(m["enter_voc_vs_synth"], m["stay_voc_vs_synth"], s=8 if thesis else 16,
               color="#2E7D32", lw=0)
    ax.axhline(0, color="grey", lw=0.5, ls=(0, (4, 2)))
    ax.axvline(0, color="grey", lw=0.5, ls=(0, (4, 2)))
    rho, p = spearmanr(m["enter_voc_vs_synth"], m["stay_voc_vs_synth"])
    ax.set_xlabel("Entering index")
    ax.set_ylabel("Staying index")
    ax.set_title(f"(d) vocalisation against\nsynthetic, blocks 2-4,\nrho = {rho:+.2f}, p = {p:.2f}"
                 if thesis else f"Entering against staying, vocalisation against synthetic,\n"
                                f"blocks 2-4: rho = {rho:+.2f}, p = {p:.2f}",
                 loc="left" if thesis else "center", fontsize=7 if thesis else 9)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    import preference_analysis_config as cfg
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "approach")
    os.makedirs(out, exist_ok=True)

    t = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    keep = logged_sessions(args.data_dir, t["session"].unique())
    dropped = sorted(set(t["session"]) - keep)
    e = t[t["session"].isin(keep) & t["cls"].isin(CLASSES)].copy()
    e["visits"] = e["visits"].fillna(0)                    # a blank count is no entry
    e.to_csv(os.path.join(out, "entries.csv"), index=False)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say(f"{len(dropped)} sessions logged no visit anywhere and were left out as failed "
        f"recordings; {e['session'].nunique()} sessions, {e['mouse'].nunique()} mice, "
        f"{len(e)} arm-blocks")

    lay = first_block_layout(t[t["session"].isin(keep)])
    say("\nlayout: share of sessions in which each stimulus sat in its usual arm "
        "(min and max over the stimuli), and the usual arm of the silent arm and the vocalisation")
    say(lay.round(2).to_string(index=False))

    for blocks, label in BLOCKS.items():
        say(f"\nentries per arm and sound block, {label}, mean over arm-blocks:")
        say(subset(e, blocks).groupby(["set", "cls"])["visits"].mean().unstack().round(2).to_string())

    ind = per_mouse(e)
    ind.to_csv(os.path.join(out, "indices_per_mouse.csv"), index=False)
    for col, _, _, name in CONTRASTS:
        say(f"\n{name.replace(chr(10), ' ')}: median [quartiles], mice above 0, Wilcoxon against 0")
        for blocks, label in BLOCKS.items():
            g = ind[ind["blocks"] == blocks]
            say(f"  {label}, four sets pooled per mouse:")
            for measure in ("enter", "stay"):
                x = g[f"{measure}_{col}"].dropna()
                q1, q3 = np.percentile(x, [25, 75])
                say(f"    {measure:5s} n={len(x):2d}  {x.median():+.3f} [{q1:+.3f}, {q3:+.3f}]  "
                    f"{int((x > 0).sum()):2d} above 0  p = {wilcoxon(x).pvalue:.4f}")
            both = g[[f"enter_{col}", f"stay_{col}"]].dropna()
            rho, p = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
            say(f"    entering against staying, across mice: rho = {rho:+.2f}, p = {p:.3f} "
                f"({len(both)} mice)")
            per_set = pooled_indices(subset(e, blocks), ["mouse", "set"])
            for s in SET_ORDER:
                x = per_set.loc[per_set["set"] == s, f"enter_{col}"].dropna()
                y = per_set.loc[per_set["set"] == s, f"stay_{col}"].dropna()
                if len(x) >= 5 and len(y) >= 5:
                    say(f"      {s:13s} enter {x.median():+.3f} (n={len(x)}, p = {wilcoxon(x).pvalue:.4f})"
                        f"   stay {y.median():+.3f} (n={len(y)}, p = {wilcoxon(y).pvalue:.4f})")

    trials_csv = os.path.join(base, "order_place", "arm_trials.csv")
    if os.path.exists(trials_csv):
        b1 = same_arm_block1(t, pd.read_csv(trials_csv))
        b1.to_csv(os.path.join(out, "block1_same_arm.csv"), index=False)
        say("\nBlock 1, the same favoured arm (1, 2, 7, 8) on different days: its share over its "
            "habituation share,\nwhen it held a or b; per mouse (mean over its arms), ratio of "
            "medians, mice above 1, Wilcoxon")
        say("  which arm held what in block 1 (most sessions):")
        held = t[(t["block"] == 1) & t["arm"].isin(FAVOURED)].groupby(["arm", "set"])["cls"].agg(
            lambda c: c.value_counts().index[0]).unstack()
        say(held.reindex(columns=[s for s in SET_ORDER + ["Intervals 3"] if s in held.columns])
            .to_string())
        for m, label in (("n", "entering"), ("t", "time in arm")):
            g = b1.groupby(["mouse", "arm", "cls"])[f"{m}_lr"].mean().unstack()
            for a, b in (("silent", "synthetic"), ("vocalisation", "synthetic"),
                         ("vocalisation", "silent")):
                d = (g[a] - g[b]).dropna()
                dm = d.groupby(level="mouse").mean()
                say(f"  {label:11s} {a} against {b}: {len(d)} mouse-arm pairs, {len(dm)} mice, "
                    f"x{2 ** dm.median():.2f}, {int((dm > 0).sum())} of {len(dm)} above, "
                    f"p = {wilcoxon(dm).pvalue:.4f}")
    else:
        say("\n(run exp1_order_place.py for the block 1 comparison of each arm with itself)")
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    draw(ind, os.path.join(out, "fig_approach"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            draw(ind, os.path.join(args.thesis_figures, "fig_exp1_approach"), thesis=True)

    print("\nPoisson mixed model of the entries (lme4):")
    r = subprocess.run([rscript(), os.path.join(HERE, "exp1_approach.R"),
                        os.path.join(out, "entries.csv"), out])
    if r.returncode:
        sys.exit(r.returncode)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
