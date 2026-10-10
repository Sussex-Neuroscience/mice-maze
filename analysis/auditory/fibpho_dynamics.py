"""
Fiber Photometry: How Responses Change Within Blocks and Across Arms
====================================================================

Reads what fibpho_alignment.py saved for every session (events.csv,
blocks.csv, dff_downsampled.npz) and asks how the dopamine signal changes
from the beginning to the end of each sound block, across the blocks of a
session, and from arm to arm.

The response to an entry is the change it brings: mean delta F/F over the
5 s after the entry (the response builds slowly and peaks 4 to 6 s in)
minus the mean over the 2 s before it, so each entry is measured against
its own baseline. Entries whose windows reach outside the blocks are left
out.

Because the stimuli move between arms from block to block, an arm pooled
over the whole session stands for a place, and an arm within one block for
one stimulus in one place.

Usage:
    python fibpho_dynamics.py [FIBPHO_ANALYSIS]

    FIBPHO_ANALYSIS defaults to the folder fibpho_alignment.py writes,
    DATA_ROOT/FIBPHO_ANALYSIS.

Outputs (FIBPHO_ANALYSIS/dynamics/):
    entry_responses.csv              every entry with its response and
                                     its place in time and in the maze
    block_drift.csv                  delta F/F in the first and last minute
                                     of every block
    fig_tonic_within_block.png/.pdf  delta F/F through each sound block
    fig_entry_response_time.png/.pdf entry responses against time in block
    fig_early_late.png/.pdf          peri-event traces, first against
                                     second half of the blocks
    fig_roi_block.png/.pdf           mean response per arm in each block,
                                     with the stimulus the arm played
    fig_roi_session.png/.pdf         every entry per arm, pooled over the
                                     session
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibpho_alignment as fa

BASE_WIN = (-2.0, 0.0)     # s before entry, the entry's own baseline
RESP_WIN = (0.0, 5.0)      # s after entry
TONIC_BIN_S = 30           # bin for delta F/F through the block
RESPONSE_BIN_MIN = 2       # bin for the mean entry response through the block

DAYS = ["temporal_envelope_modulation", "intervals", "sequences", "vocalisations"]
DAY_TITLE = {"temporal_envelope_modulation": "Envelope modulation",
             "intervals": "Intervals", "sequences": "Sequences",
             "vocalisations": "Vocalisations"}
ROIS = [f"ROI{i}" for i in range(1, 9)]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# -- per session -------------------------------------------------------

def load_session(folder):
    events = pd.read_csv(os.path.join(folder, "events.csv"))
    blocks = pd.read_csv(os.path.join(folder, "blocks.csv"))
    npz = np.load(os.path.join(folder, "dff_downsampled.npz"))
    return events, blocks, npz["time_s"].astype(float), npz["dff"].astype(float), float(npz["fs"])


def window_mean(dff, fs, t, win):
    i0, i1 = int(round((t + win[0]) * fs)), int(round((t + win[1]) * fs))
    if i0 < 0 or i1 > len(dff):
        return np.nan
    seg = dff[i0:i1]
    return float(seg.mean()) if np.isfinite(seg).all() else np.nan


def entry_responses(events, blocks, dff, fs):
    """Every entry of the sound blocks with its response and place in time."""
    sound_blocks = blocks[~blocks["is_silent"]].set_index("trial_ID")
    rows = []
    for _, ev in events[events["trial_ID"].isin(sound_blocks.index)].iterrows():
        tb = sound_blocks.loc[ev["trial_ID"]]
        t = ev["tdt_time_s"]
        base = window_mean(dff, fs, t, BASE_WIN)
        resp = window_mean(dff, fs, t, RESP_WIN)
        length_min = (tb["end_tdt"] - tb["start_tdt"]) / 60
        minute = (t - tb["start_tdt"]) / 60
        rows.append({"trial_ID": int(ev["trial_ID"]), "ROI": ev["ROI_visited"],
                     "label": ev["label"], "tdt_time_s": t,
                     "minute_in_block": minute,
                     "half": "early" if minute < length_min / 2 else "late",
                     "response": resp - base, "baseline": base,
                     "time_in_arm_s": ev["time_spent_seconds"]})
    return pd.DataFrame(rows)


def block_drift(blocks, time_s, dff):
    """Mean delta F/F in the first and the last minute of every block."""
    out = []
    for _, tb in blocks.iterrows():
        first = dff[(time_s >= tb["start_tdt"]) & (time_s < tb["start_tdt"] + 60)]
        last = dff[(time_s >= tb["end_tdt"] - 60) & (time_s < tb["end_tdt"])]
        kind = ("habituation" if tb["trial_ID"] == 1 else
                "silent" if tb["is_silent"] else "sound")
        out.append({"trial_ID": int(tb["trial_ID"]), "kind": kind,
                    "first_min": np.nanmean(first), "last_min": np.nanmean(last)})
    d = pd.DataFrame(out)
    d["fall"] = d["first_min"] - d["last_min"]
    return d


def short_label(label):
    """A few characters for a stimulus, to fit in a heatmap cell."""
    fixed = {"silent arm": "sil", "vocalisation": "voc", "silent block": "", "habituation": ""}
    if label in fixed:
        return fixed[label]
    if label.endswith(" kHz"):                  # 'rough_complex 20 kHz' -> 'rc20'
        kind, freq = label[:-4].rsplit(" ", 1)
        return "".join(w[0] for w in kind.split("_")) + freq
    if len(label) > 7:                          # vocalisation files, by initials
        return "".join(w[0] for w in label.replace("-", "_").split("_") if w)[:6]
    return label


def tonic_course(blocks, time_s, dff):
    """Mean delta F/F in TONIC_BIN_S bins through each sound block."""
    out = []
    for _, tb in blocks[~blocks["is_silent"]].iterrows():
        edges = np.arange(tb["start_tdt"], tb["end_tdt"] + 1e-9, TONIC_BIN_S)
        for a, b in zip(edges[:-1], edges[1:]):
            seg = dff[(time_s >= a) & (time_s < b)]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                out.append({"trial_ID": int(tb["trial_ID"]),
                            "minute_in_block": (a - tb["start_tdt"] + TONIC_BIN_S / 2) / 60,
                            "dff": float(seg.mean())})
    return pd.DataFrame(out)


# -- figures -----------------------------------------------------------

def grid(mice, title):
    fig, axes = plt.subplots(len(mice), len(DAYS), figsize=(3.4 * len(DAYS), 2.8 * len(mice)),
                             squeeze=False, sharex=True)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(DAY_TITLE[day])
    for r, mouse in enumerate(mice):
        axes[r, 0].annotate(mouse, (-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=11)
    return fig, axes


def block_colours(trial_ids):
    cmap = plt.get_cmap("viridis")
    ids = sorted(trial_ids)
    return {t: cmap(0.1 + 0.75 * i / max(1, len(ids) - 1)) for i, t in enumerate(ids)}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=200, bbox_inches="tight")
    fa.safe_savefig(fig, os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def fig_tonic(sessions, mice, out):
    fig, axes = grid(mice, "ΔF/F through each sound block (30 s bins)")
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        tonic = s["tonic"]
        colours = block_colours(tonic["trial_ID"].unique())
        for tid, g in tonic.groupby("trial_ID"):
            ax.plot(g["minute_in_block"], g["dff"], color=colours[tid], lw=1,
                    label=f"block T{tid}")
        ax.legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Minutes into the sound block")
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔF/F")
    save(fig, out, "fig_tonic_within_block")


def fig_response_time(sessions, mice, out):
    fig, axes = grid(mice, "Response to each entry against its time in the block "
                           f"(ΔF/F {RESP_WIN[0]:g} to {RESP_WIN[1]:g} s after entry, "
                           f"minus {BASE_WIN[0]:g} to {BASE_WIN[1]:g} s)")
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        r = s["responses"].dropna(subset=["response"])
        colours = block_colours(r["trial_ID"].unique())
        silent = r[r["label"] == "silent arm"]
        sound = r[r["label"] != "silent arm"]
        ax.scatter(silent["minute_in_block"], silent["response"], s=8, marker="x",
                   color="#9E9E9E", lw=0.6, label="silent arm", zorder=2)
        for tid, g in sound.groupby("trial_ID"):
            ax.scatter(g["minute_in_block"], g["response"], s=7, color=colours[tid],
                       alpha=0.7, lw=0, label=f"T{tid}", zorder=2)
        bins = np.arange(0, sound["minute_in_block"].max() + RESPONSE_BIN_MIN, RESPONSE_BIN_MIN)
        b = pd.cut(sound["minute_in_block"], bins, right=False)
        m = sound.groupby(b, observed=True)["response"].agg(["mean", "sem", "size"])
        centres = [iv.mid for iv in m.index]
        ax.errorbar(centres, m["mean"], yerr=m["sem"], color="black", lw=1.2,
                    capsize=2, zorder=3, label=f"sounds, {RESPONSE_BIN_MIN}-min mean")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=5.5, frameon=False, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Minutes into the sound block")
    for ax in axes[:, 0]:
        ax.set_ylabel("Entry response (ΔF/F)")
    save(fig, out, "fig_entry_response_time")


def fig_early_late(sessions, mice, out):
    fig, axes = grid(mice, "Peri-entry ΔF/F, first against second half of the sound blocks "
                           "(each entry minus its own 2 s baseline)")
    style = {"early": ("#1565C0", "first half"), "late": ("#E65100", "second half")}
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        r = s["responses"]
        sound = r[r["label"] != "silent arm"]
        for half, (colour, name) in style.items():
            times = sound.loc[sound["half"] == half, "tdt_time_s"]
            traces, t = fa.peri_event_traces(s["dff"], s["fs"], times)
            if len(traces) < fa.MIN_EVENTS_PETH:
                continue
            base = traces[:, (t >= BASE_WIN[0]) & (t < BASE_WIN[1])].mean(axis=1, keepdims=True)
            traces = traces - base
            mean, sem = traces.mean(axis=0), traces.std(axis=0) / np.sqrt(len(traces))
            ax.plot(t, mean, color=colour, lw=1.2, label=f"{name} (n={len(traces)})")
            ax.fill_between(t, mean - sem, mean + sem, color=colour, alpha=0.15, lw=0)
        ax.axvline(0, color="black", ls="--", lw=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time from arm entry (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔF/F from baseline")
    save(fig, out, "fig_early_late")


def fig_roi_block(sessions, mice, out):
    """Arms down, sound blocks across; each cell one stimulus in one arm."""
    cells = {}
    for key, s in sessions.items():
        r = s["responses"]
        tids = sorted(s["blocks"].loc[~s["blocks"]["is_silent"], "trial_ID"])
        mean = np.full((len(ROIS), len(tids)), np.nan)
        text = {}
        for i, roi in enumerate(ROIS):
            for j, tid in enumerate(tids):
                cell = r[(r["ROI"] == roi) & (r["trial_ID"] == tid)]
                if len(cell):
                    valid = cell["response"].dropna()
                    if len(valid):
                        mean[i, j] = valid.mean()
                    text[(i, j)] = f"{short_label(cell['label'].mode()[0])}\n{len(valid)}"
        cells[key] = (tids, mean, text)
    lim = np.nanpercentile(np.abs(np.concatenate([m.ravel() for _, m, _ in cells.values()])), 95)

    fig, axes = plt.subplots(len(mice), len(DAYS), figsize=(3.3 * len(DAYS), 3.6 * len(mice)),
                             squeeze=False)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(DAY_TITLE[day])
    for (day, mouse), (tids, mean, text) in cells.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        im = ax.imshow(mean, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), t in text.items():
            ax.text(j, i, t, ha="center", va="center", fontsize=5, linespacing=1.1)
        ax.set_xticks(range(len(tids)))
        ax.set_xticklabels([f"T{t}" for t in tids], fontsize=7)
        ax.set_yticks(range(len(ROIS)))
        ax.set_yticklabels(ROIS, fontsize=6)
        if DAYS.index(day) == 0:
            ax.annotate(mouse, (-0.32, 0.5), xycoords="axes fraction", rotation=90,
                        va="center", fontsize=10, fontweight="bold")
    fig.suptitle("Mean entry response per arm in each sound block; each cell names "
                 "the stimulus the arm played and the number of entries", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6,
                 label=f"Entry response (ΔF/F, 0 to {RESP_WIN[1]:g} s minus "
                       f"{BASE_WIN[0]:g} to 0 s)")
    fig.savefig(os.path.join(out, "fig_roi_block.png"), dpi=200, bbox_inches="tight")
    fa.safe_savefig(fig, os.path.join(out, "fig_roi_block.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_roi_block.png / .pdf")


def fig_roi_session(sessions, mice, out):
    """Every entry per arm, pooled over the session: the arm as a place."""
    fig, axes = grid(mice, "Entry responses per arm over the whole session "
                           "(bars: mean ± SEM of all entries; x: silent arm)")
    rng = np.random.default_rng(0)
    for (day, mouse), s in sessions.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        r = s["responses"].dropna(subset=["response"])
        colours = block_colours(r["trial_ID"].unique())
        for i, roi in enumerate(ROIS):
            arm = r[r["ROI"] == roi]
            if arm.empty:
                continue
            x = i + rng.uniform(-0.18, 0.18, len(arm))
            silent = (arm["label"] == "silent arm").to_numpy()
            ax.scatter(x[~silent], arm.loc[~silent, "response"], s=6, lw=0, alpha=0.7,
                       color=[colours[t] for t in arm.loc[~silent, "trial_ID"]], zorder=2)
            ax.scatter(x[silent], arm.loc[silent, "response"], s=8, marker="x", lw=0.6,
                       color="#9E9E9E", zorder=2)
            m = arm["response"].mean()
            sem = arm["response"].std() / np.sqrt(len(arm)) if len(arm) > 1 else 0
            ax.errorbar(i, m, yerr=sem, fmt="_", color="black", ms=12, mew=1.5,
                        capsize=2, zorder=3)
            ax.annotate(str(len(arm)), (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=5.5, color="#555555")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(len(ROIS)))
        ax.set_xticklabels([r_[3:] for r_ in ROIS], fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Arm (ROI)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Entry response (ΔF/F)")
    save(fig, out, "fig_roi_session")


# -- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?",
                        default=os.path.join(fa.DEFAULT_ROOT, "FIBPHO_ANALYSIS"),
                        help="the folder fibpho_alignment.py wrote")
    args = parser.parse_args()
    out = os.path.join(args.root, "dynamics")
    os.makedirs(out, exist_ok=True)

    sessions = {}
    for day in DAYS:
        day_dir = os.path.join(args.root, day)
        if not os.path.isdir(day_dir):
            continue
        for mouse in sorted(os.listdir(day_dir)):
            folder = os.path.join(day_dir, mouse)
            if not os.path.exists(os.path.join(folder, "events.csv")):
                continue
            events, blocks, time_s, dff, fs = load_session(folder)
            responses = entry_responses(events, blocks, dff, fs)
            sessions[(day, mouse)] = {"blocks": blocks, "dff": dff, "fs": fs,
                                      "responses": responses,
                                      "tonic": tonic_course(blocks, time_s, dff),
                                      "drift": block_drift(blocks, time_s, dff)}
            print(f"{day} / {mouse}: {responses['response'].notna().sum()} entries "
                  f"with a response")
    mice = sorted({m for _, m in sessions})

    table = pd.concat([s["responses"].assign(day=d, mouse=m)
                       for (d, m), s in sessions.items()], ignore_index=True)
    table.to_csv(os.path.join(out, "entry_responses.csv"), index=False)
    drift = pd.concat([s["drift"].assign(day=d, mouse=m)
                       for (d, m), s in sessions.items()], ignore_index=True)
    drift.to_csv(os.path.join(out, "block_drift.csv"), index=False)

    fig_tonic(sessions, mice, out)
    fig_response_time(sessions, mice, out)
    fig_early_late(sessions, mice, out)
    fig_roi_block(sessions, mice, out)
    fig_roi_session(sessions, mice, out)

    print("\ndelta F/F, first minute minus last minute of each block:")
    print(drift.groupby("kind")["fall"].agg(
        mean="mean", blocks="size", falling=lambda x: int((x > 0).sum())).round(4).to_string())

    sound = table[(table["label"] != "silent arm")].dropna(subset=["response"])
    print("\nmean entry response (sounds), first against second half of the blocks:")
    print(sound.pivot_table(index=["mouse", "day"], columns="half", values="response",
                            aggfunc=["mean", "size"]).round(4).to_string())
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
