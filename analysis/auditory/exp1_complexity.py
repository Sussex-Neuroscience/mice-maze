#!/usr/bin/env python3
"""
Experiment 1: Does Preference Follow Stimulus Complexity?
=========================================================

Places every synthetic stimulus of Experiment 1 on two measures of
complexity (stimulus_complexity.describe_exp1):

    spectral   the normalised spectral entropy of its waveform
    temporal   log2 of the cycles of its lowest frequency before the
               waveform repeats, one scale for tones, intervals,
               amplitude-modulated tones and sequences

and asks whether the preference index of each stimulus against the silent
arm (from exp1_stimulus_pi.py) follows either measure, in a straight line or
as an inverted U. Mixed models, mouse as a random intercept, stimulus set as
a fixed effect (avoidance was strongest on the first set and complexity
differs between sets, so without it the two would be confounded):

    index ~ set                                  baseline
    index ~ set + c                              linear
    index ~ set + c + c^2                        inverted U if c^2 < 0
                                                 with its peak inside the range

compared by likelihood-ratio tests, for c the temporal and the spectral
measure in turn (each standardised). Vocalisations do not repeat and have
no temporal value; they are left out.

Usage:
    python exp1_complexity.py [DATA_DIR] [--thesis-figures DIR] [--shuffled-only]

    run exp1_stimulus_pi.py first (with --shuffled-only for the shuffled
    blocks 2 to 4 alone, read from stimulus_pi_shuffled/).

Outputs (DATA_DIR/BATCH_ANALYSIS/complexity/ or complexity_shuffled/):
    complexity.csv, stats.txt, fig_complexity.png/.pdf
    and with --thesis-figures, fig_exp1_complexity
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimulus_complexity as sc
from exp1_stimulus_pi import COLOURS, label

SET_MARKERS = {"TEM": "o", "Intervals 1": "s", "Intervals 2": "D", "Sequences": "^"}
MEASURES = [("temporal_bits", "Temporal complexity (bits)"),
            ("spectral", "Spectral complexity")]


def fit(d, formula):
    return smf.mixedlm(formula, d, groups=d["mouse"]).fit(reml=False)


def run_stats(d, comp, say):
    say(f"{len(d)} indices, {d['mouse'].nunique()} mice, {d['stimulus'].nunique()} synthetic "
        "stimuli in 4 sets; vocalisations left out")
    for col, name in MEASURES:
        z = (d[col] - d[col].mean()) / d[col].std()
        x = d.assign(c=z, c2=z ** 2)
        m0, m1, m2 = fit(x, "pi ~ C(set)"), fit(x, "pi ~ C(set) + c"), fit(x, "pi ~ C(set) + c + c2")
        lr1, lr2 = 2 * (m1.llf - m0.llf), 2 * (m2.llf - m1.llf)
        say(f"\n{name} (standardised)")
        say(f"  linear:    slope {m1.params['c']:+.4f} (SE {m1.bse['c']:.4f}), "
            f"LRT chi2(1) = {lr1:.2f}, p = {chi2.sf(lr1, 1):.3f}")
        say(f"  quadratic: c2 {m2.params['c2']:+.4f} (SE {m2.bse['c2']:.4f}), "
            f"LRT chi2(1) = {lr2:.2f}, p = {chi2.sf(lr2, 1):.3f}")
        if m2.params["c2"] < 0:
            peak = -m2.params["c"] / (2 * m2.params["c2"]) * d[col].std() + d[col].mean()
            say(f"  the fitted curve peaks at {peak:.2f} (range {d[col].min():.2f} "
                f"to {d[col].max():.2f})")
        else:
            say("  the fitted curve is U-shaped or flat, not an inverted U")
        pooled = fit(x, "pi ~ c + c2")
        say(f"  without set: slope {pooled.params['c']:+.4f} (p = {pooled.pvalues['c']:.3f}), "
            f"c2 {pooled.params['c2']:+.4f} (p = {pooled.pvalues['c2']:.3f})")
        say("  within each set, Spearman of the stimulus medians against the measure:")
        med = d.groupby(["set", "stimulus"], as_index=False).agg(pi=("pi", "median"),
                                                                 value=(col, "first"))
        for s, g in med.groupby("set"):
            if g["value"].nunique() > 2:
                rho, p = spearmanr(g["value"], g["pi"])
                say(f"    {s:12s} {len(g)} stimuli  rho = {rho:+.2f}, p = {p:.2f}")


def draw(d, comp, path, thesis=False):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3) if thesis else (12, 3.8),
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    # (a) every synthetic stimulus in the two measures
    ax = axes[0]
    for _, r in comp.iterrows():
        ax.scatter(r["temporal_bits"], r["spectral"], color=COLOURS.get(r["family"], "#424242"),
                   marker=SET_MARKERS[r["set"]], s=14 if thesis else 26, zorder=3,
                   edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Temporal complexity (bits)")
    ax.set_ylabel("Spectral complexity")
    ax.set_title("(a) the synthetic stimuli" if thesis else "The synthetic stimuli",
                 loc="left" if thesis else "center")
    handles = [plt.Line2D([], [], marker=m, ls="", color="#616161", label=s,
                          markersize=3.5 if thesis else 6) for s, m in SET_MARKERS.items()]
    ax.legend(handles=handles, fontsize=5.5 if thesis else 7, frameon=False, loc="upper left")

    # (b), (c) preference against each measure, centred on each set's mean
    d = d.copy()
    d["pi_c"] = d["pi"] - d.groupby("set")["pi"].transform("mean")
    for k, (ax, (col, name)) in enumerate(zip(axes[1:], MEASURES)):
        g = d.groupby(["set", "stimulus", "family"], as_index=False).agg(
            med=("pi_c", "median"), q1=("pi_c", lambda v: v.quantile(0.25)),
            q3=("pi_c", lambda v: v.quantile(0.75)), value=(col, "first"))
        for _, r in g.iterrows():
            colour = COLOURS.get(r["family"], "#424242")
            ax.plot([r["value"]] * 2, [r["q1"], r["q3"]], color=colour, lw=0.7, alpha=0.6)
            ax.scatter(r["value"], r["med"], color=colour, marker=SET_MARKERS[r["set"]],
                       s=14 if thesis else 26, zorder=3, edgecolor="white", linewidth=0.3)
        # the quadratic fit, set effects removed
        mu, sd = d[col].mean(), d[col].std()
        x = d.assign(c=(d[col] - mu) / sd, c2=((d[col] - mu) / sd) ** 2)
        m = fit(x.assign(pi=x["pi_c"]), "pi ~ c + c2")
        grid = np.linspace(d[col].min(), d[col].max(), 100)
        zg = (grid - mu) / sd
        ax.plot(grid, m.params["Intercept"] + m.params["c"] * zg + m.params["c2"] * zg ** 2,
                color="black", lw=1, ls=(0, (3, 1.5)))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel(name)
        letter = "bc"[k]
        ax.set_title(f"({letter}) preference against it" if thesis else f"Preference against {name.lower()}",
                     loc="left" if thesis else "center")
        if k == 0:
            ax.set_ylabel("Preference index against silence,\nfrom its set's mean")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    import preference_analysis_config as cfg
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR")
    parser.add_argument("--shuffled-only", action="store_true",
                        help="use the indices of the shuffled blocks 2 to 4 alone")
    args = parser.parse_args()
    suffix = "_shuffled" if args.shuffled_only else ""
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "complexity" + suffix)
    os.makedirs(out, exist_ok=True)

    st = pd.read_csv(os.path.join(base, "stimulus_pi" + suffix, "stimulus_pi.csv"))
    st = st[(st["family"] != "vocalisation") & st["set"].isin(SET_MARKERS)].dropna(subset=["pi"])
    # stimuli heard by fewer than five mice (the early sessions' intervals) are left out
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    comp = st[["set", "stimulus", "family"]].drop_duplicates().reset_index(drop=True)
    vals = pd.DataFrame([sc.describe_exp1(n, f) for n, f in zip(comp["stimulus"], comp["family"])])
    comp = pd.concat([comp, vals[["spectral", "temporal_bits"]]], axis=1)
    comp.to_csv(os.path.join(out, "complexity.csv"), index=False)
    d = st.merge(comp, on=["set", "stimulus", "family"])

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say(comp.assign(name=comp["stimulus"].map(label)).round(3)
        .to_string(index=False, columns=["set", "name", "family", "spectral", "temporal_bits"]))
    say()
    run_stats(d, comp, say)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    draw(d, comp, os.path.join(out, "fig_complexity"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            draw(d, comp, os.path.join(args.thesis_figures, f"fig_exp1_complexity{suffix}"), thesis=True)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
