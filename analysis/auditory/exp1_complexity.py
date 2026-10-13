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

A second part asks the same question within the seven recordings of the
vocalisation day. A recording never repeats, so the temporal measure above
would only give its length; it is measured instead by its temporal entropy,
how evenly its energy is spread over time (stimulus_complexity.
temporal_entropy, after Sueur et al. 2008), a scale of its own that the
synthetic stimuli are not on. Each recording is read from the recordings
folder as the maze played it (mono, resampled to 192 kHz) and measured:

    spectral             the whole recording, 0 to 96 kHz, as for the
                         synthetic stimuli
    spectral_audible     the whole recording from 2 kHz up, where mice hear;
                         one recording (the dyadic interaction) holds nearly
                         all its power in a hum below 2 kHz, which sets its
                         value on the full band
    spectral_opening     its opening OPENING_S seconds, the part heard most,
                         since playback restarted at every entry
    spectral_ultrasonic  the whole recording from 35 kHz up, the calls
                         without the broadband noise several recordings carry
                         below 25 kHz
    temporal             temporal entropy of the whole recording from 2 kHz up
    temporal_opening     the same over its opening OPENING_S seconds
    temporal_ultrasonic  the same from 35 kHz up

and, all seven being from one set, modelled as index ~ c (+ c^2) + (1 | mouse)
against index ~ 1 + (1 | mouse). The recordings were not corrected for the
speaker, whose response is not known above 60 kHz, so the measures describe
the signal sent to the speaker. Skipped if the recordings are not found.

Usage:
    python exp1_complexity.py [DATA_DIR] [--thesis-figures DIR] [--shuffled-only]

    run exp1_stimulus_pi.py first (with --shuffled-only for the shuffled
    blocks 2 to 4 alone, read from stimulus_pi_shuffled/).

Outputs (DATA_DIR/BATCH_ANALYSIS/complexity/ or complexity_shuffled/):
    complexity.csv, stats.txt, fig_complexity.png/.pdf
    vocalisation_complexity.csv, vocalisation_stats.txt,
    fig_vocalisation_complexity.png/.pdf
    and with --thesis-figures, fig_exp1_complexity and
    fig_exp1_vocalisation_complexity
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

# Where the day 6 recordings live differs between computers; the first that
# exists is used. Add a line for a new computer rather than replacing one.
VOCALISATION_FOLDERS = [
    r"D:\AuditoryExperiments\vocalisationzzzzzz\trimmed_vocalisations",
    r"C:\Users\shahd\Box\Awake Project\Maze data\Auditory experiments"
    r"\vocalisationzzzzzz\trimmed_vocalisations",
]
OPENING_S = 10.0
AUDIBLE_HZ = (2000.0, sc.FS / 2)
ULTRASONIC_HZ = (35000.0, sc.FS / 2)
VOC_MEASURES = [
    ("spectral", "Spectral complexity, whole recording, 0 to 96 kHz"),
    ("spectral_audible", "Spectral complexity, whole recording, 2 to 96 kHz"),
    ("spectral_opening", f"Spectral complexity, opening {OPENING_S:g} s, 0 to 96 kHz"),
    ("spectral_ultrasonic", "Spectral complexity, whole recording, 35 to 96 kHz"),
    ("temporal", "Temporal complexity, whole recording, 2 to 96 kHz"),
    ("temporal_opening", f"Temporal complexity, opening {OPENING_S:g} s, 2 to 96 kHz"),
    ("temporal_ultrasonic", "Temporal complexity, whole recording, 35 to 96 kHz"),
]
# where each recording's label sits beside its point in the figure, in points,
# for the spectral panel and the temporal one
LABEL_OFFSETS = {
    "spectral_audible": {"run3_day2-2_male_w_female_oestrus": (-4, 3, "right"),
                         "male_male_interaction": (4, -6, "left"),
                         "run1_day2_male_w_female_oestrus": (4, 2, "left")},
    "temporal": {"run3_day2-2_male_w_female_oestrus": (-4, 3, "right"),
                 "run1_day2_male_w_female_oestrus": (4, 3, "left"),
                 "male_male_interaction": (4, -6, "left")},
}


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


# -- within the vocalisations ---------------------------------------------------
def recording_as_played(path):
    """A recording as the maze played it: mono, resampled to 192 kHz."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wave, fs = sf.read(path, always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if fs != sc.FS:
        g = gcd(int(fs), sc.FS)
        wave = resample_poly(wave, sc.FS // g, int(fs) // g)
    return wave


def vocalisation_measures(folder, stems):
    """Spectral and temporal complexity of each day 6 recording (see the docstring)."""
    files = {os.path.splitext(f)[0]: os.path.join(folder, f)
             for f in os.listdir(folder) if f.lower().endswith(".wav")}
    rows = []
    for stem in stems:
        if stem not in files:
            print(f"  no recording for {stem!r} in {folder}")
            continue
        wave = recording_as_played(files[stem])
        opening = wave[:int(OPENING_S * sc.FS)]
        rows.append({"stimulus": stem, "seconds": len(wave) / sc.FS,
                     "spectral": sc.spectral_entropy(wave),
                     "spectral_audible": sc.spectral_entropy(wave, band=AUDIBLE_HZ),
                     "spectral_opening": sc.spectral_entropy(opening),
                     "spectral_ultrasonic": sc.spectral_entropy(wave, band=ULTRASONIC_HZ),
                     "temporal": sc.temporal_entropy(wave, AUDIBLE_HZ[0]),
                     "temporal_opening": sc.temporal_entropy(opening, AUDIBLE_HZ[0]),
                     "temporal_ultrasonic": sc.temporal_entropy(wave, ULTRASONIC_HZ[0])})
    return pd.DataFrame(rows)


def run_vocalisation_stats(v, say):
    say(f"{len(v)} indices, {v['mouse'].nunique()} mice, {v['stimulus'].nunique()} recordings "
        "of the vocalisation day")
    for col, name in VOC_MEASURES:
        z = (v[col] - v[col].mean()) / v[col].std()
        x = v.assign(c=z, c2=z ** 2)
        m0, m1, m2 = fit(x, "pi ~ 1"), fit(x, "pi ~ c"), fit(x, "pi ~ c + c2")
        lr1, lr2 = 2 * (m1.llf - m0.llf), 2 * (m2.llf - m1.llf)
        say(f"\n{name} (standardised; range {v[col].min():.3f} to {v[col].max():.3f})")
        say(f"  linear:    slope {m1.params['c']:+.4f} (SE {m1.bse['c']:.4f}), "
            f"LRT chi2(1) = {lr1:.2f}, p = {chi2.sf(lr1, 1):.3f}")
        say(f"  quadratic: c2 {m2.params['c2']:+.4f} (SE {m2.bse['c2']:.4f}), "
            f"LRT chi2(1) = {lr2:.2f}, p = {chi2.sf(lr2, 1):.3f}")
        med = v.groupby("stimulus", as_index=False).agg(pi=("pi", "median"), value=(col, "first"))
        rho, p = spearmanr(med["value"], med["pi"])
        say(f"  Spearman of the {len(med)} recording medians: rho = {rho:+.2f}, p = {p:.2f}")
    per = v.drop_duplicates("stimulus")
    rho, p = spearmanr(per["spectral_audible"], per["temporal"])
    say(f"\nThe two measures across the recordings (2 to 96 kHz): Spearman rho = {rho:+.2f}, "
        f"p = {p:.2f}")


def preference_panel(ax, v, measure, xlabel, thesis, size):
    """The index of each recording against one measure, medians and quartiles."""
    g = v.groupby("stimulus", as_index=False).agg(
        med=("pi", "median"), q1=("pi", lambda s: s.quantile(0.25)),
        q3=("pi", lambda s: s.quantile(0.75)), value=(measure, "first"))
    colour = COLOURS["vocalisation"]
    offsets = LABEL_OFFSETS.get(measure, {})
    for _, r in g.iterrows():
        ax.plot([r["value"]] * 2, [r["q1"], r["q3"]], color=colour, lw=0.7, alpha=0.6)
        ax.scatter(r["value"], r["med"], color=colour, marker="v", s=size, zorder=3,
                   edgecolor="white", linewidth=0.3)
        dx, dy, ha = offsets.get(r["stimulus"], (3, 2, "left"))
        ax.annotate(label(r["stimulus"]), (r["value"], r["med"]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=5.5 if thesis else 8, color="#424242")
    # the straight-line fit: with seven recordings a squared term would draw a
    # curve that one or two of them decide
    mu, sd = v[measure].mean(), v[measure].std()
    m = fit(v.assign(c=(v[measure] - mu) / sd), "pi ~ c")
    grid = np.linspace(v[measure].min(), v[measure].max(), 100)
    ax.plot(grid, m.params["Intercept"] + m.params["c"] * (grid - mu) / sd,
            color="black", lw=1, ls=(0, (3, 1.5)))
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(xlabel)


def draw_vocalisations(v, comp, voc, path, thesis=False, measure="spectral_audible"):
    """The recordings drawn on measure, 2 to 96 kHz by default, and on temporal entropy.

    For the synthetic sounds, which hold nothing below 8 kHz, the 2 to 96 kHz
    band and the full band give the same value to within 0.3%, so panel (a)
    can set the two side by side. The synthetic sounds have no place on the
    temporal entropy of panel (c), a scale of the recordings alone.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3) if thesis else (14, 3.8),
                             gridspec_kw={"width_ratios": [0.9, 1, 1]})
    size = 14 if thesis else 26
    # (a) where the recordings sit against the synthetic stimuli
    ax = axes[0]
    order = list(SET_MARKERS) + ["Vocalisations"]
    rng = np.random.default_rng(0)
    for k, s in enumerate(order):
        g = voc if s == "Vocalisations" else comp[comp["set"] == s]
        col = measure if s == "Vocalisations" else "spectral"
        for _, r in g.iterrows():
            fam = "vocalisation" if s == "Vocalisations" else r["family"]
            ax.scatter(k + rng.uniform(-0.15, 0.15), r[col], s=size, zorder=3,
                       color=COLOURS.get(fam, "#424242"), marker=SET_MARKERS.get(s, "v"),
                       edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace("Intervals ", "Int. ").replace("Vocalisations", "Voc.")
                        .replace("Sequences", "Seq.") for s in order])
    ax.set_ylabel("Spectral complexity")
    ax.set_title("(a) every stimulus" if thesis else "Spectral complexity by set",
                 loc="left" if thesis else "center")

    # (b), (c) preference against each, within the vocalisations
    preference_panel(axes[1], v, measure, "Spectral complexity", thesis, size)
    axes[1].set_ylabel("Preference index against silence")
    axes[1].set_title("(b) the recordings, spectral" if thesis else "Preference and spectral complexity",
                      loc="left" if thesis else "center")
    preference_panel(axes[2], v, "temporal", "Temporal complexity (entropy)", thesis, size)
    axes[2].set_title("(c) the recordings, temporal" if thesis else "Preference and temporal complexity",
                      loc="left" if thesis else "center")
    axes[2].sharey(axes[1])
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def vocalisations(st_path, comp, out, suffix, thesis_dir):
    """The second part: spectral and temporal complexity within the vocalisation day."""
    folder = next((p for p in VOCALISATION_FOLDERS if os.path.isdir(p)), None)
    if folder is None:
        print("\nVocalisations: recordings not found in VOCALISATION_FOLDERS, part skipped")
        return
    st = pd.read_csv(st_path)
    v = st[(st["set"] == "Vocalisations") & (st["family"] == "vocalisation")].dropna(subset=["pi"])
    voc = vocalisation_measures(folder, sorted(v["stimulus"].unique()))
    voc.to_csv(os.path.join(out, "vocalisation_complexity.csv"), index=False)
    v = v.merge(voc, on="stimulus")

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say(voc.assign(name=voc["stimulus"].map(label)).round(3).to_string(
        index=False, columns=["name", "seconds"] + [c for c, _ in VOC_MEASURES]))
    say()
    run_vocalisation_stats(v, say)
    with open(os.path.join(out, "vocalisation_stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    draw_vocalisations(v, comp, voc, os.path.join(out, "fig_vocalisation_complexity"))
    if thesis_dir:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            draw_vocalisations(v, comp, voc, os.path.join(
                thesis_dir, f"fig_exp1_vocalisation_complexity{suffix}"), thesis=True)


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

    vocalisations(os.path.join(base, "stimulus_pi" + suffix, "stimulus_pi.csv"), comp, out,
                  suffix, args.thesis_figures)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
