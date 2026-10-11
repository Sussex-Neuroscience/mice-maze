#!/usr/bin/env python3
"""
Experiment 1: Does Exploratory Drive Go With Engagement With Sounds?
====================================================================

Exploratory drive is measured in the silent habituation block that opens
every session (trial 1), before any sound is played:

    entries per minute into the arms                       (primary)
    roaming entropy: how evenly the time in the arms is
    spread over the eight of them, 0 to 1

and engagement with sounds in the sound blocks of the same session:

    the preference index of all sounds together against
    the silent arm (Section on the preference index)       (primary)
    entries per minute into the arms that played a sound

Entries into sound arms are counted in the same way as habituation entries,
so the two share any general difference in activity between mice; the
preference index, a ratio of visit lengths, does not. Entries into the
silent arm of the sound blocks, scaled by seven to match the seven sound
arms, are the control: what drive predicts about them it predicts about
activity, not about sound. A habituation logged at more than the upper
quartile plus three interquartile ranges of entries a minute is taken as a
tracking fault and left out.

Tests, on every day with a silent arm (all but the third interval set):
between mice, Spearman correlation of each mouse's mean over its days; within
mice, a mixed model, engagement ~ drive within + drive between + (1 | mouse),
where 'within' is a day's departure from the mouse's mean and 'between' the
mouse's mean.

Usage:
    python exp1_exploration.py [DATA_DIR] [--thesis-figures DIR]

Outputs (DATA_DIR/BATCH_ANALYSIS/exploration/):
    exploration.csv, stats.txt, fig_exploration.png/.pdf
    and with --thesis-figures, fig_exp1_exploration
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
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROIS = [f"ROI{i}" for i in range(1, 9)]
SOUND_TRIALS = {2, 4, 6, 8}
DRIVES = [("hab_entries_per_min", "Habituation entries per minute"),
          ("roaming_entropy", "Habituation roaming entropy")]
ENGAGEMENTS = [("pi_all_sounds", "Preference index, all sounds\nagainst the silent arm"),
               ("sound_entries_per_min", "Sound-arm entries per minute"),
               ("silent_entries_per_min", "Silent-arm entries per minute")]


def trial_minutes(df, trial):
    """A trial's length from whatever the session logged; 15 min for habituation by default."""
    rows = df[df["trial_ID"] == trial]
    if "total_time_trial_s" in rows and rows["total_time_trial_s"].notna().any():
        return float(rows["total_time_trial_s"].dropna().iloc[0]) / 60
    if {"trial_start_time", "end_trial_time"}.issubset(rows.columns) and len(rows):
        return float(rows["end_trial_time"].iloc[0] - rows["trial_start_time"].iloc[0]) / 60
    return 15.0 if trial == 1 else 10.0


def session_measures(df, day, classify):
    df = df[df["ROIs"].isin(ROIS)].copy()
    hab = df[df["trial_ID"] == 1]
    t = np.array([hab.loc[hab["ROIs"] == r, "time_spent"].sum() for r in ROIS], float)
    p = t / t.sum() if t.sum() > 0 else np.full(8, np.nan)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum() / np.log2(8)) if t.sum() > 0 else np.nan

    snd = df[df["trial_ID"].isin(SOUND_TRIALS)].copy()
    snd["label"] = [classify(r, day) for _, r in snd.iterrows()]
    silent = snd[snd["label"] == "silent"]
    sound = snd[~snd["label"].isin(["silent", "unknown", "control"])]
    mean = lambda g: g["time_spent"].sum() / g["visitation_count"].sum() \
        if g["visitation_count"].sum() > 0 else np.nan
    a, b = mean(sound), mean(silent)
    sound_minutes = sum(trial_minutes(df, tr) for tr in SOUND_TRIALS
                        if (df["trial_ID"] == tr).any())
    return {"hab_entries_per_min": hab["visitation_count"].sum() / trial_minutes(df, 1),
            "roaming_entropy": entropy,
            "pi_all_sounds": (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) else np.nan,
            "sound_entries_per_min": sound["visitation_count"].sum() / sound_minutes
            if sound_minutes else np.nan,
            # one arm against seven: the control for activity, not for sound
            "silent_entries_per_min": silent["visitation_count"].sum() * 7 / sound_minutes
            if sound_minutes else np.nan}


def stats(t, say):
    for xcol, xname in DRIVES:
        for ycol, yname in ENGAGEMENTS:
            d = t.dropna(subset=[xcol, ycol])
            m = d.groupby("mouse")[[xcol, ycol]].mean()
            rho, p = spearmanr(m[xcol], m[ycol])
            d = d.assign(x_between=d.groupby("mouse")[xcol].transform("mean"))
            d = d.assign(x_within=d[xcol] - d["x_between"])
            fit = smf.mixedlm(f"{ycol} ~ x_within + x_between", d, groups=d["mouse"]).fit()
            say(f"\n{xname} against {yname.replace(chr(10), ' ')}")
            say(f"  between mice: Spearman rho = {rho:+.2f}, p = {p:.3f} ({len(m)} mice)")
            say(f"  within mice:  slope {fit.params['x_within']:+.4f} "
                f"(SE {fit.bse['x_within']:.4f}), p = {fit.pvalues['x_within']:.3f}; "
                f"between-mouse slope {fit.params['x_between']:+.4f}, "
                f"p = {fit.pvalues['x_between']:.3f} ({len(d)} sessions)")


def draw(t, path, thesis=False):
    pairs = [(DRIVES[0], ENGAGEMENTS[0]), (DRIVES[0], ENGAGEMENTS[1]),
             (DRIVES[0], ENGAGEMENTS[2])]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2) if thesis else (11, 3.4))
    for k, (ax, ((xc, xn), (yc, yn))) in enumerate(zip(axes, pairs)):
        # each mouse's mean over the days that have both measures, as in stats()
        d = t.dropna(subset=[xc, yc]).groupby("mouse")[[xc, yc]].mean()
        ax.scatter(d[xc], d[yc], s=10 if thesis else 20, color="#1F3864", alpha=0.75, lw=0)
        rho, p = spearmanr(d[xc], d[yc])
        ptext = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        ax.text(0.97, 0.03, f"rho = {rho:+.2f}, {ptext}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6 if thesis else 8)
        if yc == "pi_all_sounds":
            ax.axhline(0, color="grey", lw=0.5, ls=(0, (4, 2)))
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        if thesis:
            ax.set_title(f"({'abc'[k]})", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    import preference_analysis_config as cfg
    from run_batch_preference import classify_stimulus
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    cfg.INDIVIDUAL_VISIT_CLIP_MS = 0
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "exploration")
    os.makedirs(out, exist_ok=True)

    rows = []
    for sess in cfg.discover_sessions():
        if sess.day == "w1_d4":
            continue
        df, _ = cfg.load_session_visits(sess)
        if df is None:
            continue
        rows.append({"mouse": sess.mouse_id, "day": sess.day,
                     **session_measures(df, sess.day, classify_stimulus)})
    t = pd.DataFrame(rows).groupby(["mouse", "day"], as_index=False).mean(numeric_only=True)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    # habituation logged at an impossible rate is a tracking fault, not exploration
    q1, q3 = t["hab_entries_per_min"].quantile([0.25, 0.75])
    limit = q3 + 3 * (q3 - q1)
    bad = t["hab_entries_per_min"] > limit
    for _, r in t[bad].iterrows():
        say(f"excluded habituation of {r['mouse']} {r['day']}: {r['hab_entries_per_min']:.1f} "
            f"entries a minute, above {limit:.1f} (upper quartile + 3 interquartile ranges)")
    t.loc[bad, ["hab_entries_per_min", "roaming_entropy"]] = np.nan
    t.to_csv(os.path.join(out, "exploration.csv"), index=False)

    say(f"{len(t)} sessions, {t['mouse'].nunique()} mice; days with a silent arm")
    say(t[[c for c, _ in DRIVES + ENGAGEMENTS]].describe().round(3).to_string())
    stats(t, say)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    draw(t, os.path.join(out, "fig_exploration"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            draw(t, os.path.join(args.thesis_figures, "fig_exp1_exploration"), thesis=True)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
