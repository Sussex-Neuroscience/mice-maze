#!/usr/bin/env python3
"""
The Preference Index and Its Fisher z
=====================================

The figure of Section 2.6 of the thesis. The index (a - b) / (a + b) of mean
visit lengths a and b is tanh of half the log of their ratio, so its Fisher z,
artanh(PI), is half the log ratio itself:

    (a) the two scales: each doubling of the ratio is one equal step of z, while
        on the index the steps shrink towards the bounds at -1 and +1
    (b) both scales against the ratio: they agree near indifference and part
        towards the bounds; shaded, the range of 90% of the indices of the mice
        of Experiment 1 (one per mouse, stimulus and day, stimuli heard by five
        mice or more), and two results of Chapter 4, the vocalisation against
        the synthetic sounds (x1.17) and the synthetic sounds against the
        silent arm (x0.84)
    (c, d) why the models of the thesis use log visit length: the mean visit
        of every arm in every sound block of Experiment 1, in seconds and on a
        log scale, with skewness and the normal curve of the same mean and
        standard deviation

Set in Computer Modern (cmr10, shipped with matplotlib) to match the thesis.

Usage:
    python pi_fisher_figure.py [DATA_DIR] [--thesis-figures DIR]

Reads DATA_DIR/BATCH_ANALYSIS/stimulus_pi/stimulus_pi.csv (exp1_stimulus_pi.py)
and lmm/arm_blocks.csv (exp1_lmm.py); writes fig_pi_fisher.png/.pdf and
stats.txt to DATA_DIR/BATCH_ANALYSIS/pi_fisher/, and with --thesis-figures the
figure as fig_pi_fisher in DIR.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, skew

BLUE = "#1F4E8C"      # the index
RED = "#B03A2E"       # Fisher z
GREY = "#8C8C8C"
LIGHT = "#E9ECEF"
RATIOS = 2.0 ** np.arange(-4, 5)    # 1/16 to 16
RESULTS = (1.17, 0.84)              # Section 4.4: vocalisation against synthetic, synthetic against silent

STYLE = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
}
RATIO_LABEL = r"$\bar{t}_{\mathrm{sound}}\,/\,\bar{t}_{\mathrm{silent}}$"


def num(x, digits=2):
    """A signed number as mathtext, so that its minus is a true minus sign."""
    return f"${x:.{digits}f}$" if x < 0 else f"${x:.{digits}f}$".replace("$0.00$", "$0$")


def ratio_tick(r):
    return "1" if r == 1 else (f"{r:g}" if r > 1 else f"1/{1 / r:g}")


def letter(ax, text, x=-0.02, y=1.02):
    # cmr10 has no bold weight, so the letter is set in bold maths
    ax.text(x, y, rf"$\mathbf{{{text}}}$", transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom")


def panel_scales(ax):
    """(a) The two scales, joined at the same ratios."""
    pi = np.tanh(0.5 * np.log(RATIOS))
    z = 0.5 * np.log(RATIOS)
    zmax = z.max() * 1.1
    for p, q in zip(pi, z):
        ax.plot([p, q / zmax], [1, 0], color="#C9CED3", lw=0.6, zorder=1)
    ax.plot([-1, 1], [1, 1], color=BLUE, lw=1.2, zorder=2)
    ax.plot([-1, 1], [0, 0], color=RED, lw=1.2, zorder=2)
    ax.scatter(pi, np.ones_like(pi), color=BLUE, s=14, zorder=3, lw=0)
    ax.scatter(z / zmax, np.zeros_like(z), color=RED, s=14, zorder=3, lw=0)
    for x, t in ((-1, "$-1$"), (1, "$+1$")):
        ax.plot([x, x], [0.9, 1.1], color=BLUE, lw=1.0)
        ax.text(x, 1.18, t, ha="center", va="bottom", fontsize=7)
    for p, r in zip(pi, RATIOS):
        if 1 <= r < 16:
            ax.text(p, 1.18, num(p), ha="center", va="bottom", fontsize=6.5, color=BLUE)
    for q, r in zip(z, RATIOS):
        ax.text(q / zmax, -0.16, ratio_tick(r), ha="center", va="top", fontsize=7)
        if r >= 1:
            ax.text(q / zmax, -0.42, num(q), ha="center", va="top", fontsize=6.5, color=RED)
    ax.text(-1.1, 1.0, r"$\mathrm{PI}$", color=BLUE, ha="right", va="center", fontsize=9)
    ax.text(-1.1, 0.0, r"$z$", color=RED, ha="right", va="center", fontsize=9)
    ax.text(1.0, -0.82, RATIO_LABEL, ha="right", va="top", fontsize=8)
    ax.set_xlim(-1.22, 1.06)
    ax.set_ylim(-1.05, 1.5)
    ax.axis("off")


def panel_agree(ax, lo, hi):
    """(b) Both scales against the ratio, with the range of the mice."""
    r = np.exp(np.linspace(np.log(1 / 20), np.log(20), 400))
    ax.axhspan(lo, hi, color=LIGHT, lw=0, zorder=0)
    for y in (-1, 1):
        ax.axhline(y, color=GREY, lw=0.5, ls=(0, (4, 3)))
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(1, color=GREY, lw=0.4)
    ax.plot(r, 0.5 * np.log(r), color=RED, lw=1.2, ls=(0, (5, 2.5)), zorder=2, label=r"$z$")
    ax.plot(r, (r - 1) / (r + 1), color=BLUE, lw=1.4, zorder=3, label=r"$\mathrm{PI}$")
    for ratio in RESULTS:
        ax.plot([ratio], [(ratio - 1) / (ratio + 1)], "o", ms=3.2, color="black", zorder=4)
    ax.set_xscale("log")
    ticks = [1 / 16, 1 / 4, 1, 4, 16]
    ax.set_xticks(ticks)
    ax.set_xticklabels([ratio_tick(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(RATIO_LABEL)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.2), handlelength=2.2)


def panel_visits(ax_raw, ax_log, mean_s):
    """(c, d) Mean visit per arm and block, in seconds and on a log scale."""
    out = {}
    for ax, vals, label, colour in ((ax_raw, mean_s, "mean visit (s)", GREY),
                                    (ax_log, np.log(mean_s), r"$\ln$ mean visit", RED)):
        if ax is ax_raw:
            bins = np.linspace(0, np.quantile(vals, 0.995), 45)
        else:
            bins = np.linspace(vals.min(), vals.max(), 45)
        ax.hist(vals, bins=bins, density=True, color=colour, alpha=0.55, lw=0)
        xs = np.linspace(bins[0], bins[-1], 300)
        ax.plot(xs, norm.pdf(xs, vals.mean(), vals.std()), color="black", lw=0.8)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        s = skew(vals)
        ax.text(0.97 if ax is ax_raw else 0.03, 0.97, f"skewness {num(s, 1)}",
                transform=ax.transAxes, ha="right" if ax is ax_raw else "left", va="top",
                fontsize=7)
        out[label] = s
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "pi_fisher")
    os.makedirs(out, exist_ok=True)

    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    lo, hi = np.quantile(st["pi"], [0.05, 0.95])
    arms = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    arms = arms[(arms["visits"] > 0) & (arms["mean_s"] > 0)]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(6.3, 4.3))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.15], width_ratios=[1.45, 1, 1],
                              hspace=0.42, wspace=0.42)
        ax = fig.add_subplot(gs[0, :])
        panel_scales(ax)
        letter(ax, "a", x=0.0, y=0.98)
        ax_b = fig.add_subplot(gs[1, 0])
        panel_agree(ax_b, lo, hi)
        letter(ax_b, "b", x=-0.12)
        ax_c, ax_d = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
        sk = panel_visits(ax_c, ax_d, arms["mean_s"].to_numpy())
        letter(ax_c, "c", x=0.0)
        letter(ax_d, "d", x=0.0)
        targets = [out] + ([args.thesis_figures] if args.thesis_figures else [])
        for t in targets:
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(t, f"fig_pi_fisher.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    lines = [
        f"Indices of Experiment 1, one per mouse, stimulus and day (stimuli heard by five mice "
        f"or more): {len(st)}; 90% between {lo:+.2f} and {hi:+.2f}",
        "Results marked in (b): " + ", ".join(
            f"x{r:g} is an index of {(r - 1) / (r + 1):+.3f}" for r in RESULTS),
        f"Mean visit per arm and sound block, Experiment 1: {len(arms)} arm blocks; skewness "
        + ", ".join(f"{k} {v:+.2f}" for k, v in sk.items()),
    ]
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
