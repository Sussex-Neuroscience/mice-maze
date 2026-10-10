"""
Fiber Photometry: Dopamine While the Animal Is in Each Arm
==========================================================

Reads what fibpho_alignment.py saved for every session (events.csv,
blocks.csv, dff_downsampled.npz) and asks what the dopamine level is while
the animal stays in an arm, stimulus by stimulus, against the silent arm of
the same blocks and against arm visits in the silent blocks.

delta F/F is z-scored within each block, so a level reads as standard
deviations above or below that block's own mean, and blocks with
different baselines can be pooled. The level of a visit is the mean over
the whole visit, from entry to exit.

The dopamine level of each stimulus is then set against the behavioural
preference for it, the preference index of the thesis computed from the
same visits: (mean visit with the stimulus - mean visit in the silent arm)
/ (their sum), within the sound blocks.

Usage:
    python fibpho_arms.py [FIBPHO_ANALYSIS]

Outputs (FIBPHO_ANALYSIS/arms/):
    visits.csv                   every visit with its dopamine level
    stimuli.csv                  per session and stimulus: visits, mean visit
                                 length, preference index, dopamine level
    fig_arm_level.png/.pdf       dopamine level per visit, by stimulus
    fig_arm_timecourse.png/.pdf  the level through a visit, while the
                                 animal is still in the arm
    fig_arm_vs_preference.png/.pdf dopamine level against preference index
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibpho_alignment as fa

COURSE_S = 15.0            # follow a visit for at most this long
MIN_VISITS = 3             # a stimulus needs this many visits to be plotted
CONTROLS = ["silent block", "silent arm"]
SOUND_GROUP = "all sounds"

DAYS = ["temporal_envelope_modulation", "intervals", "sequences", "vocalisations"]
DAY_TITLE = {"temporal_envelope_modulation": "Envelope modulation",
             "intervals": "Intervals", "sequences": "Sequences",
             "vocalisations": "Vocalisations"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# -- per session -------------------------------------------------------

def zscore_by_block(time_s, dff, blocks):
    """delta F/F in standard deviations of its own block."""
    z = np.full_like(dff, np.nan)
    for _, b in blocks.iterrows():
        m = (time_s >= b["start_tdt"]) & (time_s < b["end_tdt"]) & np.isfinite(dff)
        if m.sum() > 10:
            z[m] = (dff[m] - dff[m].mean()) / dff[m].std()
    return z


def visit_levels(events, z, fs):
    """Mean z from entry to exit for every visit after the habituation block."""
    rows = []
    for _, ev in events[events["trial_ID"] > 1].iterrows():
        i0 = int(round(ev["tdt_time_s"] * fs))
        i1 = int(round((ev["tdt_time_s"] + ev["time_spent_seconds"]) * fs))
        seg = z[i0:max(i1, i0 + 1)] if 0 <= i0 < len(z) else np.array([])
        seg = seg[np.isfinite(seg)]
        rows.append({"trial_ID": int(ev["trial_ID"]), "ROI": ev["ROI_visited"],
                     "label": ev["label"], "tdt_time_s": ev["tdt_time_s"],
                     "visit_s": ev["time_spent_seconds"],
                     "level_z": seg.mean() if len(seg) else np.nan})
    v = pd.DataFrame(rows)
    v["group"] = np.where(v["label"].isin(CONTROLS), v["label"], SOUND_GROUP)
    return v


def visit_course(visits, z, fs):
    """z through each visit, from entry, NaN once the animal has left."""
    n = int(COURSE_S * fs)
    out = np.full((len(visits), n), np.nan)
    for k, (_, v) in enumerate(visits.iterrows()):
        i0 = int(round(v["tdt_time_s"] * fs))
        stay = min(n, int(round(v["visit_s"] * fs)))
        if i0 < 0 or i0 + stay > len(z):
            continue
        out[k, :stay] = z[i0:i0 + stay]
    return out, np.arange(n) / fs


def stimulus_table(visits):
    """Behaviour and dopamine per stimulus, from the sound-block visits."""
    sb = visits[visits["trial_ID"] % 2 == 0]
    silent = sb[sb["label"] == "silent arm"]
    t_silent = silent["visit_s"].mean() if len(silent) else np.nan
    rows = []
    for label, g in sb.groupby("label"):
        t = g["visit_s"].mean()
        lv = g["level_z"].dropna()
        rows.append({"stimulus": label, "visits": len(g), "mean_visit_s": t,
                     "pi_vs_silent": (t - t_silent) / (t + t_silent)
                     if label != "silent arm" and t + t_silent > 0 else np.nan,
                     "level_z": lv.mean() if len(lv) else np.nan,
                     "level_sem": lv.std() / np.sqrt(len(lv)) if len(lv) > 1 else np.nan})
    return pd.DataFrame(rows)


# -- figures -----------------------------------------------------------

def grid(mice, title, sharex=False):
    fig, axes = plt.subplots(len(mice), len(DAYS), figsize=(3.6 * len(DAYS), 3.0 * len(mice)),
                             squeeze=False, sharex=sharex)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(DAY_TITLE[day])
    for r, mouse in enumerate(mice):
        axes[r, 0].annotate(mouse, (-0.42, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=11)
    return fig, axes


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=200, bbox_inches="tight")
    fa.safe_savefig(fig, os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def stimulus_order(labels):
    """Controls first, then the sounds alphabetically."""
    sounds = sorted(l for l in labels if l not in CONTROLS)
    return [c for c in CONTROLS if c in labels] + sounds


def fig_level(sessions, mice, out):
    fig, axes = grid(mice, "Dopamine level during each visit, by what the arm played "
                           "(z within block; bars: mean ± SEM of visits)")
    rng = np.random.default_rng(0)
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        v = s["visits"].dropna(subset=["level_z"])
        counts = v["label"].value_counts()
        order = [l for l in stimulus_order(set(v["label"])) if counts.get(l, 0) >= MIN_VISITS]
        colours = fa.label_colours(order)
        for i, label in enumerate(order):
            y = v.loc[v["label"] == label, "level_z"]
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(y)), y, s=5, lw=0, alpha=0.6,
                       color=colours[label], zorder=2)
            ax.errorbar(i, y.mean(), yerr=y.std() / np.sqrt(len(y)), fmt="_", color="black",
                        ms=12, mew=1.5, capsize=2, zorder=3)
            ax.annotate(str(len(y)), (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=5.5, color="#555555")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([fa_short(l) for l in order], rotation=60, ha="right", fontsize=6)
        ax.set_ylim(-2.5, 2.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Level during visit (z)")
    save(fig, out, "fig_arm_level")


def fa_short(label):
    return label.replace("_", " ") if len(label) <= 18 else label[:16] + "..."


def fig_timecourse(sessions, mice, out):
    fig, axes = grid(mice, "Dopamine through a visit, while the animal is still in the arm "
                           "(z within block; mean ± SEM of the visits still in the arm)",
                     sharex=True)
    style = {SOUND_GROUP: "#1565C0", "silent arm": "#424242", "silent block": "#9E9E9E"}
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        course, t = s["course"]
        for group, colour in style.items():
            rows = (s["visits"]["group"] == group).to_numpy()
            c = course[rows]
            n = np.isfinite(c).sum(axis=0)
            keep = n >= MIN_VISITS
            if not keep.any():
                continue
            mean = np.nanmean(np.where(keep, c, np.nan), axis=0)
            sem = np.nanstd(c, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.plot(t[keep], mean[keep], color=colour, lw=1.1,
                    label=f"{group} ({rows.sum()} visits)")
            ax.fill_between(t[keep], (mean - sem)[keep], (mean + sem)[keep], color=colour,
                            alpha=0.15, lw=0)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time since entering the arm (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Level (z)")
    save(fig, out, "fig_arm_timecourse")


def fig_vs_preference(stimuli, mice, out):
    fig, axes = grid(mice, "Dopamine level in an arm against the preference index for its "
                           "stimulus (each point one stimulus; the silent arm at 0)")
    for (day, mouse), st in stimuli.groupby(["day", "mouse"]):
        ax = axes[mice.index(mouse), DAYS.index(day)]
        st = st[st["visits"] >= MIN_VISITS]
        x = st["pi_vs_silent"].fillna(0.0)          # the silent arm itself sits at 0
        colours = fa.label_colours(st["stimulus"])
        for (_, row), xi in zip(st.iterrows(), x):
            ax.errorbar(xi, row["level_z"], yerr=row["level_sem"], fmt="o", ms=4,
                        color=colours[row["stimulus"]], capsize=2)
            ax.annotate(fa_short(row["stimulus"]), (xi, row["level_z"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=5.5,
                        color=colours[row["stimulus"]])
        sounds = st["stimulus"] != "silent arm"
        if sounds.sum() >= 4:
            rho, p = spearmanr(st.loc[sounds, "pi_vs_silent"], st.loc[sounds, "level_z"])
            ax.text(0.02, 0.02, f"sounds: rho = {rho:.2f}, p = {p:.2f}",
                    transform=ax.transAxes, fontsize=6.5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Preference index against the silent arm")
    for ax in axes[:, 0]:
        ax.set_ylabel("Level during visits (z)")
    save(fig, out, "fig_arm_vs_preference")


# -- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?",
                        default=os.path.join(fa.DEFAULT_ROOT, "FIBPHO_ANALYSIS"),
                        help="the folder fibpho_alignment.py wrote")
    args = parser.parse_args()
    out = os.path.join(args.root, "arms")
    os.makedirs(out, exist_ok=True)

    sessions, all_visits, all_stimuli, tests = {}, [], [], []
    for day in DAYS:
        day_dir = os.path.join(args.root, day)
        if not os.path.isdir(day_dir):
            continue
        for mouse in sorted(os.listdir(day_dir)):
            folder = os.path.join(day_dir, mouse)
            if not os.path.exists(os.path.join(folder, "events.csv")):
                continue
            events = pd.read_csv(os.path.join(folder, "events.csv"))
            blocks = pd.read_csv(os.path.join(folder, "blocks.csv"))
            npz = np.load(os.path.join(folder, "dff_downsampled.npz"))
            time_s, dff, fs = npz["time_s"].astype(float), npz["dff"].astype(float), float(npz["fs"])
            z = zscore_by_block(time_s, dff, blocks)
            visits = visit_levels(events, z, fs)
            sessions[(day, mouse)] = {"visits": visits, "course": visit_course(visits, z, fs)}
            all_visits.append(visits.assign(day=day, mouse=mouse))
            all_stimuli.append(stimulus_table(visits).assign(day=day, mouse=mouse))

            sb = visits[(visits["trial_ID"] % 2 == 0)].dropna(subset=["level_z"])
            snd = sb.loc[sb["group"] == SOUND_GROUP, "level_z"]
            sil = sb.loc[sb["label"] == "silent arm", "level_z"]
            p = mannwhitneyu(snd, sil).pvalue if len(snd) > 2 and len(sil) > 2 else np.nan
            tests.append({"day": day, "mouse": mouse, "sound_visits": len(snd),
                          "sound_level_z": snd.mean(), "silent_arm_visits": len(sil),
                          "silent_arm_level_z": sil.mean() if len(sil) else np.nan,
                          "mann_whitney_p": p})
    mice = sorted({m for _, m in sessions})

    pd.concat(all_visits, ignore_index=True).to_csv(os.path.join(out, "visits.csv"), index=False)
    stimuli = pd.concat(all_stimuli, ignore_index=True)
    stimuli.to_csv(os.path.join(out, "stimuli.csv"), index=False)

    fig_level(sessions, mice, out)
    fig_timecourse(sessions, mice, out)
    fig_vs_preference(stimuli, mice, out)

    pd.set_option("display.width", 200)
    print("\nlevel during visits in the sound blocks, sounds against the silent arm (z):")
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    s = stimuli[(stimuli["stimulus"] != "silent arm") & (stimuli["visits"] >= MIN_VISITS)]
    rho, p = spearmanr(s["pi_vs_silent"], s["level_z"], nan_policy="omit")
    print(f"\ndopamine level against preference index, every stimulus with >= {MIN_VISITS} "
          f"visits in every session: rho = {rho:.2f}, p = {p:.3f}, n = {len(s)}")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
