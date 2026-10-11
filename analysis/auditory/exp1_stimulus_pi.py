#!/usr/bin/env python3
"""
Experiment 1: Each Stimulus Against Silence, and Consonance and Roughness
=========================================================================

Two analyses of the sound blocks (trials 2, 4, 6, 8) of Experiment 1, each on
the thesis preference index, (a - b) / (a + b) of mean visit length, with
both arms taken from the same blocks and every visit uncapped:

1. Each stimulus against the silent arm, per mouse, for every stimulus set
   that has a silent arm (all but the third interval set). Within a set the
   stimuli are compared with a mixed model, index ~ stimulus + (1 | mouse),
   by a likelihood-ratio test against index ~ 1 + (1 | mouse); and each
   stimulus is tested against zero (Wilcoxon signed-rank), Holm-corrected
   within its set.

2. Direct contrasts within the same blocks, where the silent arm plays no
   part:
     consonant against dissonant intervals, pooled, on all three interval
       sets (the third has four of each and nothing else)
     on temporal envelope modulation, rough against smooth, changing-rate
       against smooth and changing-rate against rough, each pooled over
       10 and 20 kHz
     the amplitude-modulated unison against the plain unison, on the first
       two interval sets
   each tested against zero per set and, over sets, on each mouse's mean.

A mouse that ran a set twice contributes the mean of its two indices.

The first sound block of a session always opened on the same layout, each
stimulus in the arm its place in the list gave it, and only blocks 2 to 4 were
reshuffled. With --shuffled-only the first block is left out, and the outputs
go to stimulus_pi_shuffled/.

Usage:
    python exp1_stimulus_pi.py [DATA_DIR] [--thesis-figures DIR] [--shuffled-only]

Outputs (DATA_DIR/BATCH_ANALYSIS/stimulus_pi/ or stimulus_pi_shuffled/):
    stimulus_pi.csv, contrasts.csv, stats.txt
    fig_stimulus_pi.png/.pdf, fig_contrasts.png/.pdf
    and with --thesis-figures, fig_exp1_stimulus_pi and fig_exp1_contrasts
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
from scipy.stats import chi2, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimulus_complexity as sc

SOUND_TRIALS = {2, 4, 6, 8}
SHUFFLED_TRIALS = {4, 6, 8}                  # the sound blocks after the first
ROIS = {f"ROI{i}" for i in range(1, 9)}
SETS = {"w1_d1": "TEM", "w1_d2": "Intervals 1", "w1_d3": "Intervals 2",
        "w1_d4": "Intervals 3", "w2_sequences": "Sequences",
        "w2_vocalisations": "Vocalisations"}
SEQ_ORDER = ["AAAAA", "AoAo", "ABAB", "BABA", "ABBA", "ABCABC"]
TEM_KINDS = ["smooth", "rough", "rough_complex"]
PRETTY = {"perf_5": "5th", "perf_4": "4th", "maj_6": "maj 6th", "min_6": "min 6th",
          "maj_3": "maj 3rd", "min_3": "min 3rd", "octave": "octave", "maj_2": "maj 2nd",
          "min_2": "min 2nd", "maj_7": "maj 7th", "min_7": "min 7th", "tritone": "tritone",
          "unison": "unison", "AM unison": "AM unison", "vocalisation": "voc",
          "dyadic_social_interaction": "dyadic", "male_male_interaction": "M+M",
          "male_w_female_oestrus": "M+F", "run1_day2_male_w_female_oestrus": "M+F r1",
          "run3_day2_male_w_female_oestrus": "M+F r3",
          "run3_day2-2_male_w_female_oestrus": "M+F r3b",
          "mousetube3BB89473_socialcomparison": "mouseTube"}
COLOURS = {"consonant": "#1565C0", "dissonant": "#E65100", "unison": "#9E9E9E",
           "vocalisation": "#2E7D32", "smooth": "#00838F", "rough": "#6A1B9A",
           "rough_complex": "#AD1457", "sequence": "#1F3864"}


def pi(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else np.nan


def stimulus(row, day, classify, modes):
    """Name and family of the stimulus an arm played."""
    base = classify(row, day)
    if base in ("silent", "unknown", "control"):
        return base, base
    if base == "vocalisation":
        return base, "vocalisation"
    mode = modes[day]
    if mode == "temporal_envelope_modulation":
        try:
            return f"{base} {float(row['frequency']) / 1000:g} kHz", base
        except (TypeError, ValueError):
            return base, base
    if mode == "complex_intervals":
        if base in ("smooth", "rough"):
            return ("unison" if base == "smooth" else "AM unison"), "unison"
        return str(row["interval_name"]), base
    if mode == "sequences":
        return base, "sequence"
    return base, "vocalisation"                      # a recording on the vocalisation day


def order_key(name, family):
    fam = ["unison", "smooth", "rough", "rough_complex", "consonant", "dissonant",
           "sequence", "vocalisation"].index(family) if family in COLOURS else 9
    if family in ("consonant", "dissonant") and name in sc.RATIOS:
        within = sc.interval_periodicity_bits(name)
    elif family == "sequence" and name in SEQ_ORDER:
        within = SEQ_ORDER.index(name)
    elif name.endswith(" kHz"):
        within = float(name.split()[-2])
    else:
        within = 0
    return (0 if name == "vocalisation" and family == "vocalisation" else fam, within, name)


def label(name):
    if name.endswith(" kHz"):
        kind, f = name[:-4].rsplit(" ", 1)
        return f"{kind.replace('rough_complex', 'complex')} {f}"
    return PRETTY.get(name, name)


# -- per session -------------------------------------------------------

def session_table(df, day, classify, modes, trials=SOUND_TRIALS):
    s = df[df["trial_ID"].isin(trials) & df["ROIs"].isin(ROIS)].copy()
    named = [stimulus(r, day, classify, modes) for _, r in s.iterrows()]
    s["stimulus"] = [n for n, _ in named]
    s["family"] = [f for _, f in named]
    return s


def mean_visit(s, mask):
    g = s[mask]
    v = g["visitation_count"].sum()
    return g["time_spent"].sum() / v if v > 0 else np.nan


def per_session(s):
    """Each stimulus against silence, and the direct contrasts, for one session."""
    silent = mean_visit(s, s["family"] == "silent")
    stim = []
    for (name, fam), _ in s[~s["family"].isin(["silent", "unknown", "control"])].groupby(
            ["stimulus", "family"]):
        stim.append({"stimulus": name, "family": fam,
                     "pi": pi(mean_visit(s, s["stimulus"] == name), silent)})
    fam = {f: mean_visit(s, s["family"] == f) for f in
           ("consonant", "dissonant", "smooth", "rough", "rough_complex")}
    contrasts = {
        "consonant vs dissonant": pi(fam["consonant"], fam["dissonant"]),
        "rough vs smooth": pi(fam["rough"], fam["smooth"]),
        "changing-rate vs smooth": pi(fam["rough_complex"], fam["smooth"]),
        "changing-rate vs rough": pi(fam["rough_complex"], fam["rough"]),
        "AM unison vs unison": pi(mean_visit(s, s["stimulus"] == "AM unison"),
                                  mean_visit(s, s["stimulus"] == "unison")),
    }
    return stim, contrasts


# -- statistics --------------------------------------------------------

def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def run_stats(st, ct, say):
    say("1. Each stimulus against the silent arm")
    for set_name in SETS.values():
        d = st[st["set"] == set_name].dropna(subset=["pi"])
        if d.empty:
            continue
        keep = d.groupby("stimulus")["pi"].transform("size") >= 5
        d = d[keep]
        full = smf.mixedlm("pi ~ C(stimulus)", d, groups=d["mouse"]).fit(reml=False)
        null = smf.mixedlm("pi ~ 1", d, groups=d["mouse"]).fit(reml=False)
        lr = 2 * (full.llf - null.llf)
        dof = d["stimulus"].nunique() - 1
        say(f"\n  {set_name}: do the stimuli differ? LRT chi2({dof}) = {lr:.2f}, "
            f"p = {chi2.sf(lr, dof):.4f}  ({d['mouse'].nunique()} mice)")
        rows = []
        for (name, fam), g in d.groupby(["stimulus", "family"]):
            rows.append((name, fam, len(g), g["pi"].median(), int((g["pi"] > 0).sum()),
                         wilcoxon(g["pi"]).pvalue))
        rows.sort(key=lambda r: order_key(r[0], r[1]))
        p_holm = multipletests([r[5] for r in rows], method="holm")[1]
        for (name, fam, n, med, above, p), ph in zip(rows, p_holm):
            say(f"    {label(name):12s} {fam:13s} n={n:2d}  median {med:+.3f}  "
                f"{above:2d} above 0  p = {p:.4f}  Holm p = {ph:.4f} {star(ph)}")
        st.loc[st["set"] == set_name, "holm"] = st.loc[st["set"] == set_name, "stimulus"].map(
            dict(zip([r[0] for r in rows], p_holm)))

    say("\n2. Direct contrasts in the same blocks: median, mice above 0, Wilcoxon against 0")
    for contrast in ct["contrast"].unique():
        d = ct[ct["contrast"] == contrast].dropna(subset=["pi"])
        say(f"\n  {contrast}")
        for set_name in SETS.values():
            x = d.loc[d["set"] == set_name, "pi"]
            if len(x) >= 5:
                say(f"    {set_name:13s} n={len(x):2d}  median {x.median():+.3f}  "
                    f"{int((x > 0).sum()):2d} above 0  p = {wilcoxon(x).pvalue:.4f}")
        per_mouse = d.groupby("mouse")["pi"].mean()
        if d["set"].nunique() > 1 and len(per_mouse) >= 5:
            say(f"    {'all sets':13s} n={len(per_mouse):2d}  median {per_mouse.median():+.3f}  "
                f"{int((per_mouse > 0).sum()):2d} above 0  p = {wilcoxon(per_mouse).pvalue:.4f}"
                "  (each mouse's mean)")


# -- figures -----------------------------------------------------------

def strip(ax, values, x, colour, rng, size):
    ax.scatter(x + rng.uniform(-0.15, 0.15, len(values)), values, s=size, color=colour,
               alpha=0.55, lw=0, zorder=2)
    ax.plot([x - 0.27, x + 0.27], [np.median(values)] * 2, color=colour, lw=2, zorder=3)


def fig_stimuli(st, path, thesis=False):
    sets = [s for s in SETS.values() if s in set(st["set"])]
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.6) if thesis else (13, 7), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for k, (ax, set_name) in enumerate(zip(axes, sets)):
        d = st[st["set"] == set_name].dropna(subset=["pi"])
        names = sorted({(n, f) for n, f in zip(d["stimulus"], d["family"])},
                       key=lambda nf: order_key(*nf))
        names = [nf for nf in names if (d["stimulus"] == nf[0]).sum() >= 5]
        for i, (name, fam) in enumerate(names):
            g = d[d["stimulus"] == name]
            strip(ax, g["pi"].to_numpy(), i, COLOURS.get(fam, "#424242"), rng,
                  4 if thesis else 8)
            mark = star(g["holm"].iloc[0]) if "holm" in g and g["holm"].notna().any() else ""
            if mark:
                ax.text(i, 1.02, mark, ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([label(n) for n, _ in names], rotation=60, ha="right",
                           fontsize=5.5 if thesis else 7)
        ax.set_title(f"({'abcde'[k]}) {set_name}" if thesis else set_name,
                     loc="left" if thesis else "center")
        ax.set_ylim(-1.05, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(sets):]:
        ax.set_visible(False)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Preference index\nagainst the silent arm")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_contrasts(ct, path, thesis=False):
    panels = [("(a) consonant against\ndissonant", ["consonant vs dissonant"],
               ["Intervals 1", "Intervals 2", "Intervals 3"]),
              ("(b) roughness, temporal\nenvelope modulation",
               ["rough vs smooth", "changing-rate vs smooth", "changing-rate vs rough"], ["TEM"]),
              ("(c) AM unison\nagainst unison", ["AM unison vs unison"],
               ["Intervals 1", "Intervals 2"])]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5) if thesis else (11, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    rng = np.random.default_rng(1)
    for ax, (title, contrasts, sets) in zip(axes, panels):
        ticks = []
        for contrast in contrasts:
            for set_name in sets:
                x = ct.loc[(ct["contrast"] == contrast) & (ct["set"] == set_name), "pi"].dropna()
                if len(x) < 5:
                    continue
                i = len(ticks)
                strip(ax, x.to_numpy(), i, "#424242", rng, 4 if thesis else 8)
                mark = star(wilcoxon(x).pvalue)
                if mark:
                    ax.text(i, 1.02, mark, ha="center", va="bottom", fontsize=7)
                ticks.append(set_name if len(contrasts) == 1 else
                             contrast.replace("changing-rate", "changing")
                                     .replace(" vs ", "\nagainst\n"))
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=30 if len(contrasts) == 1 else 0,
                           ha="right" if len(contrasts) == 1 else "center",
                           fontsize=6 if thesis else 7)
        ax.set_title(title if thesis else title[4:].replace("\n", " "),
                     loc="left" if thesis else "center", fontsize=7 if thesis else 9)
        ax.set_ylim(-1.05, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Preference index")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


# -- main --------------------------------------------------------------

def main():
    import preference_analysis_config as cfg
    from run_batch_preference import classify_stimulus

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR")
    parser.add_argument("--shuffled-only", action="store_true",
                        help="leave out the first sound block, whose layout was fixed")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    cfg.INDIVIDUAL_VISIT_CLIP_MS = 0              # uncapped: see exp1_vocalisation_pi.py
    trials = SHUFFLED_TRIALS if args.shuffled_only else SOUND_TRIALS
    suffix = "_shuffled" if args.shuffled_only else ""
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "stimulus_pi" + suffix)
    os.makedirs(out, exist_ok=True)
    modes = {d: cfg.EXPERIMENT_DAYS[d]["mode"] for d in cfg.EXPERIMENT_DAYS}

    stim_rows, contrast_rows = [], []
    for sess in cfg.discover_sessions():
        df, _ = cfg.load_session_visits(sess)
        if df is None:
            continue
        s = session_table(df, sess.day, classify_stimulus, modes, trials)
        stim, contrasts = per_session(s)
        for r in stim:
            stim_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day], **r})
        for name, value in contrasts.items():
            contrast_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day],
                                  "contrast": name, "pi": value})
    # a mouse that ran a set twice: the mean of its two values
    st = (pd.DataFrame(stim_rows).groupby(["mouse", "set", "stimulus", "family"], as_index=False)
          ["pi"].mean())
    st = st[st["set"] != "Intervals 3"]           # no silent arm on that set
    ct = (pd.DataFrame(contrast_rows).groupby(["mouse", "set", "contrast"], as_index=False)
          ["pi"].mean())

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    run_stats(st, ct, say)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    st.to_csv(os.path.join(out, "stimulus_pi.csv"), index=False)
    ct.to_csv(os.path.join(out, "contrasts.csv"), index=False)

    fig_stimuli(st, os.path.join(out, "fig_stimulus_pi"))
    fig_contrasts(ct, os.path.join(out, "fig_contrasts"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            fig_stimuli(st, os.path.join(args.thesis_figures, f"fig_exp1_stimulus_pi{suffix}"), True)
            fig_contrasts(ct, os.path.join(args.thesis_figures, f"fig_exp1_contrasts{suffix}"), True)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
