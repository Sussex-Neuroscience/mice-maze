#!/usr/bin/env python3
"""The four-arm pilot: what the animals did, before the eight-arm design.

    python analysis/auditory/pilot_four_arm.py
    python analysis/auditory/pilot_four_arm.py --thesis-figures <thesis>/figures

Reads the table written by survey_standalone_sessions.py (blocks.csv: one row
per arm per block) and describes the three four-arm cohorts:

    62xx   2 mice, August 2024              pure tones
    67xx   5 mice, September-November 2024  sequences, then intervals
    80xx   4 mice, November-December 2024   pure tones, then intervals

Nothing here is a test. With two to five animals per cohort the pilot can show
what to change in the design, not what mice prefer, so every figure shows each
mouse and every summary is a mean over mice: a mouse that ran more sessions
does not count for more.

The preference index is the one defined in the thesis: the mean visit length in
an arm against the mean visit length in the silent arm of the same blocks,

    PI = (t_stimulus - t_silent) / (t_stimulus + t_silent)

from -1 (only silence) to +1 (only the stimulus). It needs a silent arm in the
blocks being compared, which in the pilot only the interval sessions had, so
the indices come from those sessions alone. The same formula with a second
stimulus in place of the silent arm compares two stimuli directly: every
interval session put one consonant and one dissonant interval in the same
blocks.

Complexity comes from stimulus_complexity.py: spectral (how spread a sound's
energy is across frequency) and temporal (how long it runs before repeating).

Exclusions, printed when it runs:

    blocks that log more time in the arms than the block lasted, which the
    early code could do when a visit crossed a block boundary
    the opening 10-minute silent block, wherever silent blocks are compared
    with sound blocks, because it is the animal's first exploration that day

Figures and tables go in pilot/ beside the survey's tables.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stimulus_complexity as sc

SURVEY = Path(r"D:\AuditoryExperiments_survey")
COHORTS = ["62xx", "67xx", "80xx"]
MARKER = {"62xx": "s", "67xx": "o", "80xx": "^"}

BLUE = "#1565C0"
ORANGE = "#E65100"
GREY = "#9E9E9E"
GREEN = "#2E7D32"
INK = "#1F3864"
FAMILY_COLOUR = {"unison": GREY, "consonant": BLUE, "dissonant": ORANGE,
                 "tone": GREEN, "sequence": INK}

PRETTY = {"unison": "unison", "min_2": "minor 2nd", "maj_2": "major 2nd",
          "maj_3": "major 3rd", "perf_4": "perfect 4th", "tritone": "tritone",
          "perf_5": "perfect 5th", "maj_6": "major 6th", "maj_7": "major 7th"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# -- the data -------------------------------------------------------------------
def load(survey: Path) -> pd.DataFrame:
    b = pd.read_csv(survey / "blocks.csv", keep_default_na=False,
                    na_values={"time_s": [""], "visits": [""], "block_min": [""]})
    b["mouse"] = b["mouse"].astype(str)
    b["cohort"] = b["mouse"].str[:2] + "xx"
    b = b[b["cohort"].isin(COHORTS)].copy()

    per_block = (b.groupby(["mouse", "date", "block"], as_index=False)
                   .agg(t=("time_s", "sum"), m=("block_min", "first")))
    impossible = per_block[per_block["t"] > per_block["m"] * 60]
    for r in impossible.itertuples():
        print(f"  excluded: mouse {r.mouse} {r.date} block {r.block}, "
              f"{r.t:.0f} s in the arms of a {r.m:g}-minute block")
    drop = set(zip(impossible["mouse"], impossible["date"], impossible["block"]))
    keep = [k not in drop for k in zip(b["mouse"], b["date"], b["block"])]
    return b[keep].copy()


def pi(a: float, z: float) -> float:
    return (a - z) / (a + z) if (a + z) > 0 else np.nan


def mean_visit(rows: pd.DataFrame) -> float:
    v = rows["visits"].sum()
    return rows["time_s"].sum() / v if v else np.nan


# -- preference indices -----------------------------------------------------------
def interval_sound_blocks(b: pd.DataFrame) -> pd.DataFrame:
    return b[(b["kind"] == "intervals") & (b["block_type"] == "sound")]


def stimulus_indices(b: pd.DataFrame) -> pd.DataFrame:
    """Each stimulus against the silent arm, per mouse.

    Pooled over the sessions in which that stimulus played, and compared with
    the silent arm of those same sessions only.
    """
    s = interval_sound_blocks(b)
    rows = []
    for (cohort, mouse), m in s.groupby(["cohort", "mouse"]):
        for stim in sorted(set(m["stimulus"]) - {"silent"}):
            dates = m.loc[m["stimulus"] == stim, "date"].unique()
            x = m[m["date"].isin(dates)]
            value = pi(mean_visit(x[x["stimulus"] == stim]),
                       mean_visit(x[x["stimulus"] == "silent"]))
            if not np.isnan(value):
                rows.append({"cohort": cohort, "mouse": mouse, "stimulus": stim,
                             "pi": value})
        value = pi(mean_visit(m[m["stimulus"] != "silent"]),
                   mean_visit(m[m["stimulus"] == "silent"]))
        rows.append({"cohort": cohort, "mouse": mouse, "stimulus": "all sounds",
                     "pi": value})
    return pd.DataFrame(rows)


def pair_indices(b: pd.DataFrame) -> pd.DataFrame:
    """The consonant against the dissonant interval of the same blocks.

    Positive means the consonant interval held the longer visits.
    """
    s = interval_sound_blocks(b)
    rows = []
    for (cohort, mouse, date), m in s.groupby(["cohort", "mouse", "date"]):
        present = set(m["stimulus"]) - {"silent", "unison"}
        cons = [x for x in present if x in sc.CONSONANT]
        diss = [x for x in present if x in sc.DISSONANT]
        if len(cons) != 1 or len(diss) != 1:
            continue
        rows.append({"cohort": cohort, "mouse": mouse, "date": date,
                     "pair": f"{PRETTY[cons[0]]} vs {PRETTY[diss[0]]}",
                     "consonant": cons[0], "dissonant": diss[0],
                     "pi": pi(mean_visit(m[m["stimulus"] == cons[0]]),
                              mean_visit(m[m["stimulus"] == diss[0]]))})
    p = pd.DataFrame(rows).dropna(subset=["pi"])
    # one value per mouse per pair, however many sessions it took
    return (p.groupby(["cohort", "mouse", "pair", "consonant", "dissonant"],
                      as_index=False)["pi"].mean())


# -- complexity -------------------------------------------------------------------
def complexity_table(b: pd.DataFrame) -> pd.DataFrame:
    """Both complexities for every stimulus the pilot played."""
    rows, seen = [], set()
    for r in b[b["stimulus"] != "silent"].itertuples():
        key = (r.kind, r.stimulus, r.tones if r.kind == "sequences" else "")
        if key in seen:
            continue
        seen.add(key)
        if r.kind == "sequences":
            tone_of = {k: float(v) for k, v in
                       (part.split("=") for part in str(r.tones).split(";") if "=" in part)}
            d = sc.describe("sequences", r.stimulus, seq_tones=tone_of or None)
            d["tones"] = r.tones
        else:
            d = sc.describe(r.kind, r.stimulus)
        rows.append(d)
    t = pd.DataFrame(rows)
    # a sequence played with two tone sets is one stimulus: average its spectra
    return (t.groupby(["kind", "stimulus", "family"], as_index=False)
              .agg(spectral=("spectral", "mean"), temporal_bits=("temporal_bits", "first"),
                   ratio=("ratio", "first") if "ratio" in t else ("stimulus", "first")))


# -- figures --------------------------------------------------------------------
def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _family(stim: str) -> str:
    if stim == "unison":
        return "unison"
    return "consonant" if stim in sc.CONSONANT else "dissonant"


def fig_pi_stimuli(idx: pd.DataFrame, out: Path):
    """Each stimulus against silence, ordered by temporal complexity."""
    stims = [s for s in idx["stimulus"].unique() if s != "all sounds"]
    stims.sort(key=lambda s: (sc.interval_periodicity_bits(s), s))
    order = ["all sounds"] + stims
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for row, stim in enumerate(order):
        d = idx[idx["stimulus"] == stim]
        colour = INK if stim == "all sounds" else FAMILY_COLOUR[_family(stim)]
        for cohort, g in d.groupby("cohort"):
            jitter = np.linspace(-0.13, 0.13, len(g)) if len(g) > 1 else [0]
            ax.scatter(g["pi"], row + np.asarray(jitter), marker=MARKER[cohort],
                       s=14, facecolor="white", edgecolor=colour, linewidth=0.8,
                       zorder=3)
        m = d["pi"].mean()
        ax.plot([m, m], [row - 0.3, row + 0.3], color=colour, lw=2.2, zorder=4)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.axhline(0.5, color=GREY, lw=0.5)
    labels = ["all sounds"] + [
        f"{PRETTY[s]}  {sc.RATIOS[s]}   ({sc.interval_periodicity_bits(s):.1f} bits)"
        for s in stims]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    lim = max(0.6, np.nanmax(np.abs(idx["pi"])) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Preference index against the silent arm")
    ax.text(-lim, -1.0, "prefers silence", ha="left", va="bottom", fontsize=6.5,
            color=GREY)
    ax.text(lim, -1.0, "prefers the sound", ha="right", va="bottom", fontsize=6.5,
            color=GREY)
    handles = [plt.Line2D([], [], marker=MARKER[c], ls="", markerfacecolor="white",
                          markeredgecolor=INK, label=c) for c in ("67xx", "80xx")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "pilot_pi_stimuli")


def fig_pi_pairs(pairs: pd.DataFrame, out: Path):
    """The consonant against the dissonant interval of the same blocks."""
    pairs = pairs.copy()
    pairs["c_bits"] = pairs["consonant"].map(sc.interval_periodicity_bits)
    order = (pairs.drop_duplicates(["cohort", "pair"])
                  .sort_values(["cohort", "c_bits", "pair"])[["cohort", "pair"]]
                  .itertuples(index=False))
    order = list(order)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ylabels = []
    for row, (cohort, pair) in enumerate(order):
        d = pairs[(pairs["cohort"] == cohort) & (pairs["pair"] == pair)]
        jitter = np.linspace(-0.13, 0.13, len(d)) if len(d) > 1 else [0]
        ax.scatter(d["pi"], row + np.asarray(jitter), marker=MARKER[cohort], s=14,
                   facecolor="white", edgecolor=BLUE, linewidth=0.8, zorder=3)
        m = d["pi"].mean()
        ax.plot([m, m], [row - 0.3, row + 0.3], color=BLUE, lw=2.2, zorder=4)
        ylabels.append(f"{pair}  ({cohort})")
    per_mouse = pairs.groupby(["cohort", "mouse"])["pi"].mean().reset_index()
    row = len(order) + 0.6
    for cohort, g in per_mouse.groupby("cohort"):
        jitter = np.linspace(-0.13, 0.13, len(g)) if len(g) > 1 else [0]
        ax.scatter(g["pi"], row + np.asarray(jitter), marker=MARKER[cohort], s=14,
                   facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.plot([per_mouse["pi"].mean()] * 2, [row - 0.3, row + 0.3], color=INK, lw=2.2)
    ax.axhline(len(order) - 0.2, color=GREY, lw=0.5)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.set_yticks(list(range(len(order))) + [row])
    ax.set_yticklabels(ylabels + ["each mouse, all pairs"])
    ax.invert_yaxis()
    lim = max(0.6, np.nanmax(np.abs(pairs["pi"])) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Preference index, consonant against dissonant")
    ax.text(-lim, -0.9, "prefers the dissonant", ha="left", va="bottom",
            fontsize=6.5, color=GREY)
    ax.text(lim, -0.9, "prefers the consonant", ha="right", va="bottom",
            fontsize=6.5, color=GREY)
    fig.tight_layout()
    save(fig, out, "pilot_pi_pairs")


def fig_complexity(table: pd.DataFrame, idx: pd.DataFrame, out: Path):
    """The pilot's stimuli in complexity space, and preference against it."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) where every pilot stimulus sits. A unison is one tone played twice,
    # so it lands on the pure tones: draw those as rings around it.
    tones = table[table["family"] == "tone"]
    a.scatter(tones["temporal_bits"], tones["spectral"], s=55, facecolor="none",
              edgecolor=GREEN, linewidth=1.0, label="pure tones", zorder=3)
    iv = table[table["kind"] == "intervals"]
    for fam in ("unison", "consonant", "dissonant"):
        d = iv[iv["family"] == fam]
        a.scatter(d["temporal_bits"], d["spectral"], color=FAMILY_COLOUR[fam], s=16,
                  label=fam, zorder=4)
    sq = table[table["family"] == "sequence"]
    a.scatter(sq["temporal_bits"], sq["spectral"], color=INK, marker="D", s=16,
              label="sequences", zorder=3)
    for r in sq.itertuples():
        a.annotate(r.stimulus, (r.temporal_bits, r.spectral), xytext=(3, 3),
                   textcoords="offset points", fontsize=6, color=INK)
    two = iv[iv["family"] != "unison"]
    a.annotate("two-tone intervals", (two["temporal_bits"].median(), two["spectral"].mean()),
               xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
               fontsize=6, color=GREY)
    a.set_xlabel("Temporal complexity (bits)")
    a.set_ylabel("Spectral complexity")
    a.text(0.02, 0.98, "(a)", transform=a.transAxes, ha="left", va="top",
           fontweight="bold")
    a.legend(frameon=False, fontsize=6, loc="lower right", ncol=2, columnspacing=0.8)

    # (b) preference against temporal complexity, intervals only. Two
    # intervals with the same complexity are set slightly apart.
    d = idx[idx["stimulus"] != "all sounds"].copy()
    d["bits"] = d["stimulus"].map(sc.interval_periodicity_bits)
    d["fam"] = d["stimulus"].map(_family)
    stims = sorted(d["stimulus"].unique(), key=lambda s: (sc.interval_periodicity_bits(s), s))
    shift = {}
    for bits in sorted({sc.interval_periodicity_bits(s) for s in stims}):
        same = [s for s in stims if sc.interval_periodicity_bits(s) == bits]
        for i, s in enumerate(same):
            shift[s] = (i - (len(same) - 1) / 2) * 0.22
    d["x"] = d["bits"] + d["stimulus"].map(shift)
    for r in d.itertuples():
        b.scatter(r.x, r.pi, marker=MARKER[r.cohort], s=9, facecolor="white",
                  edgecolor=FAMILY_COLOUR[r.fam], linewidth=0.6, alpha=0.8, zorder=2)
    means = d.groupby(["stimulus", "x", "fam"])["pi"].agg(["mean", "sem"]).reset_index()
    for r in means.itertuples():
        b.errorbar(r.x, r.mean, yerr=r.sem, fmt="o", color=FAMILY_COLOUR[r.fam],
                   ms=4.5, lw=1.2, capsize=2, zorder=4)
    b.axhline(0, color=INK, lw=0.7, ls=(0, (4, 2)))
    b.set_xlabel("Temporal complexity of the interval (bits)")
    b.set_ylabel("Preference index\nagainst the silent arm")
    # name every interval along the top, at its own position
    top = b.secondary_xaxis("top")
    top.set_xticks([sc.interval_periodicity_bits(s) + shift[s] for s in stims])
    top.set_xticklabels([PRETTY[s] for s in stims], rotation=60, ha="left", fontsize=6)
    top.tick_params(length=2, pad=1)
    for lab, s in zip(top.get_xticklabels(), stims):
        lab.set_color(FAMILY_COLOUR[_family(s)])
    b.text(0.02, 0.98, "(b)", transform=b.transAxes, ha="left", va="top",
           fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "pilot_complexity")


def fig_place_bias(b: pd.DataFrame, out: Path):
    """Each arm's share of the time spent in arms, in silence and with sound."""
    d = b[~((b["block_type"] == "silent") & (b["block"] == 1))]
    arm = d.groupby(["cohort", "mouse", "block_type", "arm"])["time_s"].sum().reset_index()
    arm["share"] = arm["time_s"] / arm.groupby(["cohort", "mouse", "block_type"])[
        "time_s"].transform("sum")
    arms = sorted(arm["arm"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        c = arm[arm["cohort"] == cohort]
        for kind, colour, dx in (("silent", GREY, -0.18), ("sound", ORANGE, 0.18)):
            ck = c[c["block_type"] == kind]
            means = [ck[ck["arm"] == a]["share"].mean() * 100 for a in arms]
            ax.bar(np.arange(len(arms)) + dx, means, width=0.34, color=colour,
                   alpha=0.35 if kind == "silent" else 0.5, label=kind)
            for i, a in enumerate(arms):
                ys = ck[ck["arm"] == a]["share"] * 100
                ax.scatter(np.full(len(ys), i + dx), ys, s=7, color=colour,
                           edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(25, color=INK, lw=0.7, ls=(0, (4, 2)))
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("ROI", "arm ") for a in arms])
        ax.set_title(f"{cohort}: {c['mouse'].nunique()} mice", loc="left")
    axes[0].set_ylabel("Share of time in arms (%)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].text(0.98, 25, "chance", transform=axes[-1].get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=6.5, color=INK)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "pilot_place_bias")


def fig_silent_arm(b: pd.DataFrame, out: Path):
    """In the interval sessions one arm of every sound block was silent."""
    s = interval_sound_blocks(b)
    s = s.assign(is_silent=s["stimulus"] == "silent")
    g = (s.groupby(["cohort", "mouse", "is_silent"])["time_s"].sum()
           .unstack(fill_value=0.0))
    g["silent_share"] = g[True] / (g[True] + g[False])
    g = g.reset_index()

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for i, cohort in enumerate([c for c in COHORTS if c in set(g["cohort"])]):
        ys = g[g["cohort"] == cohort]["silent_share"] * 100
        jitter = np.linspace(-0.12, 0.12, len(ys)) if len(ys) > 1 else [0]
        ax.scatter(i + np.asarray(jitter), ys, color=GREY, s=14, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [ys.mean()] * 2, color=INK, lw=1.8)
        ax.text(i, -0.12, f"{cohort}\n{len(ys)} mice",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=INK)
    ax.axhline(25, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(set(g["cohort"])) - 0.4)
    ax.set_ylim(0, max(50, g["silent_share"].max() * 110))
    ax.set_ylabel("Silent arm, share of\ntime in arms (%)")
    ax.text(0.98, 25, "chance", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6.5, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, out, "pilot_silent_arm")


# -- the sound-block comparison, kept as numbers only --------------------------------
def entries_summary(b: pd.DataFrame) -> pd.DataFrame:
    """Entries per minute and seconds per visit, silent against sound blocks.

    Reported in the text with its caveat (the blocks differ in length), not
    drawn: the thesis's own index compares arms within the same blocks.
    """
    t = (b.groupby(["cohort", "mouse", "date", "block", "block_type"], as_index=False)
           .agg(time_s=("time_s", "sum"), visits=("visits", "sum"),
                block_min=("block_min", "first")))
    t = t[~((t["block_type"] == "silent") & (t["block"] == 1))]
    g = (t.groupby(["cohort", "mouse", "block_type"])
           .agg(t=("time_s", "sum"), v=("visits", "sum"), mins=("block_min", "sum"))
           .reset_index())
    g["entries_per_min"] = g["v"] / g["mins"]
    g["s_per_visit"] = g["t"] / g["v"]
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--survey", type=Path, default=SURVEY,
                    help="folder holding blocks.csv from the survey")
    ap.add_argument("--thesis-figures", type=Path, default=None,
                    help="also copy the figures into this folder, under the "
                         "names the thesis includes")
    args = ap.parse_args(argv)
    if not (args.survey / "blocks.csv").exists():
        print(f"no blocks.csv in {args.survey}: run survey_standalone_sessions.py first")
        return 1
    out = args.survey / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    b = load(args.survey)
    print(f"{b[['mouse', 'date']].drop_duplicates().shape[0]} sessions, "
          f"{b['mouse'].nunique()} mice")

    idx = stimulus_indices(b)
    pairs = pair_indices(b)
    table = complexity_table(b)
    idx.to_csv(out / "pilot_pi_stimuli.csv", index=False)
    pairs.to_csv(out / "pilot_pi_pairs.csv", index=False)
    table.to_csv(out / "pilot_complexity.csv", index=False)

    fig_pi_stimuli(idx, out)
    fig_pi_pairs(pairs, out)
    fig_complexity(table, idx, out)
    fig_place_bias(b, out)
    fig_silent_arm(b, out)

    if args.thesis_figures:
        names = {"pilot_pi_stimuli": "fig_pilot_pi_stimuli",
                 "pilot_pi_pairs": "fig_pilot_pi_pairs",
                 "pilot_complexity": "fig_pilot_complexity",
                 "pilot_place_bias": "fig_pilot_place_bias",
                 "pilot_silent_arm": "fig_pilot_silent_arm"}
        args.thesis_figures.mkdir(parents=True, exist_ok=True)
        for src, dst in names.items():
            for ext in (".pdf", ".png"):
                shutil.copyfile(out / f"{src}{ext}", args.thesis_figures / f"{dst}{ext}")
        print(f"  copied to {args.thesis_figures}")

    print("\npreference index against silence, mean over mice:")
    print(idx.groupby("stimulus")["pi"].agg(["mean", "count"]).round(2).to_string())
    print("\nconsonant against dissonant, per mouse:")
    print(pairs.groupby(["cohort", "mouse"])["pi"].mean().round(2).to_string())
    g = entries_summary(b)
    print("\nentries per minute and seconds per visit (opening block out):")
    print(g.groupby(["cohort", "block_type"])[["entries_per_min", "s_per_visit"]]
            .mean().round(2).unstack().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
