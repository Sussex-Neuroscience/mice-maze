"""
Fiber Photometry: Dopamine Before, During and After Each Exposure
=================================================================

Every arm visit is an exposure to whatever that arm played. This script
measures dopamine (delta F/F, z-scored within its block) in three windows
of each visit, and relates them to how long the mouse stayed and to what
it heard:

    before   the 5 s before entering the arm
    during   the whole time in the arm, entry to exit
    after    the 5 s after leaving it

Visits are classed by what the arm played: a vocalisation recording, a
synthetic sound, the silent arm of a sound block, or any arm during a
silent block (no sound anywhere). A window that reaches outside the block
the visit began in is left out, as is a visit with any window missing.

Statistics, at the level of visits, with session as a random intercept
(linear mixed models, statsmodels):

    entry change  (during - before) ~ class + log2(visit length)
    exit change   (after - during)  ~ class + log2(visit length)
    stay          log2(visit length) ~ before + class

and, per session, Wilcoxon signed-rank tests of during against before.
Long visits (at least LONG_S) are also tested for a trend within the
visit: the first 2 s against the last 2 s.

With two mice, every test is within these two animals; nothing here
generalises to mice.

Usage:
    python fibpho_dwell.py [FIBPHO_ANALYSIS]

Outputs (FIBPHO_ANALYSIS/dwell/):
    visits.csv                       every visit, its windows and class
    stats.txt                        every model and test
    fig_dwell_sorted.png/.pdf        every visit of at least 1 s as a row,
                                     longest at the top, aligned at entry
    fig_long_visits.png/.pdf         each visit of at least LONG_S on its own
    fig_before_during_after.png/.pdf the three windows by class and length
    fig_psth.png/.pdf                the time course before, during and after a
                                     visit, for every stimulus of every day, with
                                     each stay stretched or squeezed to one width
    stats_psth.txt                   during against before, per stimulus and day;
                                     the complex tones of the envelope day against
                                     the smooth and rough ones; the longest visits
    fig_complex.png/.pdf             entries, time and dopamine for those tones

With --thesis-figures DIR, also fig_fp_long_visits, fig_fp_before_during_after
and fig_fp_psth in DIR.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibpho_alignment as fa
import stimulus_complexity as sc
from fibpho_behaviour import thesis_label

WINDOW_S = 5.0             # before and after
LONG_S = 10.0              # a long visit
EDGE_S = 2.0               # first and last part of a long visit
HEATMAP_MAX_S = 40.0       # x range of the sorted heatmap after entry
WARP_N = 50                # samples for each of before, during and after
LENGTH_BINS = [0, 1, 3, 10, np.inf]
LENGTH_NAMES = ["< 1 s", "1 to 3 s", "3 to 10 s", "10 s or more"]

CLASSES = ["vocalisation", "synthetic sound", "silent arm", "silent block"]
CLASS_COLOURS = {"vocalisation": "#2E7D32", "synthetic sound": "#1565C0",
                 "silent arm": "#424242", "silent block": "#BDBDBD"}
DAY_TITLE = {"temporal_envelope_modulation": "envelope modulation",
             "intervals": "intervals", "sequences": "sequences",
             "vocalisations": "vocalisations"}
# the day each stimulus set was played, as the thesis names it
THESIS_DAY = {"temporal_envelope_modulation": "envelope day", "intervals": "interval day",
              "sequences": "sequence day", "vocalisations": "vocalisation day"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def visit_class(label, day):
    if label in ("silent arm", "silent block"):
        return label
    if label == "vocalisation" or day == "vocalisations":
        return "vocalisation"
    return "synthetic sound"


# -- per session -------------------------------------------------------

def load(fa_root, day, mouse):
    folder = os.path.join(fa_root, day, mouse)
    events = pd.read_csv(os.path.join(folder, "events.csv"))
    blocks = pd.read_csv(os.path.join(folder, "blocks.csv"))
    npz = np.load(os.path.join(folder, "dff_downsampled.npz"))
    t, dff, fs = npz["time_s"].astype(float), npz["dff"].astype(float), float(npz["fs"])
    z = np.full_like(dff, np.nan)
    block = np.zeros(len(t), int)
    for _, b in blocks.iterrows():
        m = (t >= b["start_tdt"]) & (t < b["end_tdt"])
        block[m] = int(b["trial_ID"])
        ok = m & np.isfinite(dff)
        if ok.sum() > 10:
            z[ok] = (dff[ok] - dff[ok].mean()) / dff[ok].std()
    return events, z, block, fs


def window(z, block, fs, t0, t1, trial):
    """Mean z between t0 and t1, or NaN if any of it is outside the visit's block."""
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    i1 = max(i1, i0 + 1)
    if i0 < 0 or i1 > len(z):
        return np.nan
    seg, bl = z[i0:i1], block[i0:i1]
    return float(seg.mean()) if np.isfinite(seg).all() and (bl == trial).all() else np.nan


def measure(events, z, block, fs, day, mouse):
    rows = []
    for _, ev in events[events["trial_ID"] > 1].iterrows():
        t0, dur, tr = ev["tdt_time_s"], ev["time_spent_seconds"], int(ev["trial_ID"])
        t1 = t0 + dur
        row = {"day": day, "mouse": mouse, "session": f"{mouse} {day}", "trial_ID": tr,
               "ROI": ev["ROI_visited"], "stimulus": ev["label"],
               "cls": visit_class(ev["label"], day), "visit_s": dur, "t_entry": t0,
               "before": window(z, block, fs, t0 - WINDOW_S, t0, tr),
               "during": window(z, block, fs, t0, t1, tr),
               "after": window(z, block, fs, t1, t1 + WINDOW_S, tr)}
        if dur >= LONG_S:
            row["first_2s"] = window(z, block, fs, t0, t0 + EDGE_S, tr)
            row["last_2s"] = window(z, block, fs, t1 - EDGE_S, t1, tr)
        rows.append(row)
    return pd.DataFrame(rows)


def warped(z, block, fs, t0, dur, trial):
    """z before, during and after a visit on one time base: the WINDOW_S either
    side in real time, and the stay stretched or squeezed to the same number of
    samples, so that visits of any length line up from entry to exit. A part that
    reaches outside the visit's block is left empty."""
    def part(a, b):
        i0, i1 = int(np.floor(a * fs)), int(np.ceil(b * fs))
        if i0 < 0 or i1 > len(z) or i1 <= i0:
            return np.full(WARP_N, np.nan)
        seg = z[i0:i1]
        if not ((block[i0:i1] == trial).all() and np.isfinite(seg).all()):
            return np.full(WARP_N, np.nan)
        grid = a + (np.arange(WARP_N) + 0.5) / WARP_N * (b - a)
        return np.interp(grid, np.arange(i0, i1) / fs, seg)
    return np.concatenate([part(t0 - WINDOW_S, t0), part(t0, t0 + dur),
                           part(t0 + dur, t0 + dur + WINDOW_S)])


def aligned(z, fs, t_entry, visit_s, pre, post_max):
    """z from pre s before entry to WINDOW_S after exit, at 10 Hz, NaN beyond."""
    step = max(1, int(round(fs / 10)))
    n_pre, n_post = int(pre * 10), int(post_max * 10)
    i = int(round(t_entry * fs))
    out = np.full(n_pre + n_post, np.nan)
    for k in range(-n_pre, n_post):
        if k / 10 > visit_s + WINDOW_S:
            break
        j = i + k * step
        if 0 <= j < len(z) - step:
            out[k + n_pre] = np.nanmean(z[j:j + step]) if np.isfinite(z[j:j + step]).any() else np.nan
    return out


# -- statistics --------------------------------------------------------

def preference_indices(v, say):
    """The thesis preference index for these mice, and visit length by class.

    Per session, from the sound-block visits: vocalisation and synthetic sound
    each against the silent arm, and vocalisation against synthetic directly,
    which needs both in the same blocks (every day but the vocalisation day).
    """
    def pi(a, b):
        return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) else np.nan

    sb = v[v["trial_ID"] % 2 == 0]
    rows = []
    for (day, mouse), g in sb.groupby(["day", "mouse"]):
        m = g.groupby("cls")["visit_s"].mean()
        voc, syn, sil = (m.get(c, np.nan) for c in
                         ("vocalisation", "synthetic sound", "silent arm"))
        rows.append({"session": f"{mouse} {day}", "voc_vs_silent": pi(voc, sil),
                     "synth_vs_silent": pi(syn, sil), "voc_vs_synth": pi(voc, syn)})
    say("\n--- preference indices (mean visit length, (a - b) / (a + b)), per session")
    say(pd.DataFrame(rows).round(3).to_string(index=False))

    both = sb[(sb["day"] != "vocalisations") & sb["cls"].isin(["vocalisation", "synthetic sound"])]
    both = both.assign(log2_len=np.log2(both["visit_s"]))
    m = smf.mixedlm("log2_len ~ C(cls, Treatment('synthetic sound'))", both,
                    groups=both["session"]).fit()
    say("\n--- visit length, vocalisation against synthetic sound in the same blocks "
        "(days other than the vocalisation day): log2(length) ~ class + (1 | session)")
    say(m.summary().tables[1].to_string())


def run_stats(v, out_path):
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    ok = v.dropna(subset=["before", "during", "after"]).copy()
    ok["entry_change"] = ok["during"] - ok["before"]
    ok["exit_change"] = ok["after"] - ok["during"]
    ok["log2_len"] = np.log2(ok["visit_s"])
    ok["cls"] = pd.Categorical(ok["cls"], categories=CLASSES)
    say(f"{len(ok)} visits with all three windows, of {len(v)}; "
        f"{ok['session'].nunique()} sessions, {ok['mouse'].nunique()} mice")
    say("\nmean z by class and window (visits):")
    say(ok.groupby("cls", observed=True)[["before", "during", "after"]]
          .agg(["mean", "count"]).round(3).to_string())

    for outcome, label in (("entry_change", "entry change, during - before"),
                           ("exit_change", "exit change, after - during")):
        say(f"\n--- {label} ~ class + log2(visit length) + (1 | session)")
        say("reference class: silent arm")
        m = smf.mixedlm(f"{outcome} ~ C(cls, Treatment('silent arm')) + log2_len",
                        ok, groups=ok["session"]).fit(reml=True)
        say(m.summary().tables[1].to_string())
        # the change itself, class by class, in one model without an intercept
        m0 = smf.mixedlm(f"{outcome} ~ 0 + C(cls) + log2_len", ok, groups=ok["session"]).fit()
        say("change within each class (at a 1 s visit, log2 length 0):")
        say(m0.summary().tables[1].to_string())

    say("\n--- log2(visit length) ~ before + class + (1 | session): does dopamine "
        "before entry predict how long the mouse stays?")
    m = smf.mixedlm("log2_len ~ before + C(cls, Treatment('silent arm'))", ok,
                    groups=ok["session"]).fit()
    say(m.summary().tables[1].to_string())

    say("\n--- per session: during against before (Wilcoxon signed-rank), sound visits")
    for s, g in ok[ok["cls"].isin(["vocalisation", "synthetic sound"])].groupby("session"):
        if len(g) >= 6:
            p = wilcoxon(g["during"], g["before"]).pvalue
            say(f"  {s:42s} n={len(g):3d}  before {g['before'].mean():+.3f}  "
                f"during {g['during'].mean():+.3f}  p={p:.3f}")

    say("\n--- the same entry change with synthetic sound as the reference: vocalisation "
        "against synthetic directly")
    m = smf.mixedlm("entry_change ~ C(cls, Treatment('synthetic sound')) + log2_len",
                    ok, groups=ok["session"]).fit(reml=True)
    say(m.summary().tables[1].to_string())

    preference_indices(v, say)

    lv = v.dropna(subset=["first_2s", "last_2s"]) if "first_2s" in v else pd.DataFrame()
    if len(lv):
        say(f"\n--- visits of {LONG_S:g} s or more: first {EDGE_S:g} s against last "
            f"{EDGE_S:g} s of the visit")
        say(lv[["session", "stimulus", "visit_s", "before", "first_2s", "last_2s", "after"]]
            .sort_values("visit_s", ascending=False).round(2).to_string(index=False))
        if len(lv) >= 6:
            p = wilcoxon(lv["first_2s"], lv["last_2s"]).pvalue
            say(f"  Wilcoxon, n={len(lv)}: first {lv['first_2s'].mean():+.3f}, "
                f"last {lv['last_2s'].mean():+.3f}, p={p:.3f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ok


# -- figures -----------------------------------------------------------

def save(fig, path):
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight")
    fa.safe_savefig(fig, path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}.png / .pdf")


def fig_sorted(v, traces, out):
    mice = sorted(v["mouse"].unique())
    pre = WINDOW_S
    fig, axes = plt.subplots(1, len(mice), figsize=(6.2 * len(mice), 8), squeeze=False)
    lim = 2.5
    for ax, mouse in zip(axes[0], mice):
        idx = v.index[(v["mouse"] == mouse) & (v["visit_s"] >= 1)]
        idx = sorted(idx, key=lambda i: v.loc[i, "visit_s"])[::-1]
        mat = np.vstack([traces[i] for i in idx])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[-pre, HEATMAP_MAX_S, len(idx), 0], interpolation="nearest")
        ax.axvline(0, color="black", lw=0.6)
        ax.plot(np.minimum(v.loc[idx, "visit_s"], HEATMAP_MAX_S), np.arange(len(idx)) + 0.5,
                color="black", lw=0.8)
        for r, i in enumerate(idx):
            ax.add_patch(mpatches.Rectangle((-pre - 1.2, r), 0.9, 1, clip_on=False,
                                            color=CLASS_COLOURS[v.loc[i, "cls"]], lw=0))
        ax.set_xlim(-pre - 1.3, HEATMAP_MAX_S)
        ax.set_xlabel("Time from entering the arm (s)")
        ax.set_ylabel("Visits of 1 s or more, longest at the top")
        ax.set_title(f"{mouse}: {len(idx)} visits, all four days")
    fig.colorbar(im, ax=axes, shrink=0.5, label="Dopamine (z within block)")
    handles = [mpatches.Patch(color=c, label=k) for k, c in CLASS_COLOURS.items()]
    handles.append(plt.Line2D([], [], color="black", lw=0.8, label="exit"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.45, 1.0))
    fig.suptitle("Dopamine through every visit, sorted by how long the mouse stayed "
                 f"(shown up to {WINDOW_S:g} s after it left)", fontsize=11, y=1.04)
    save(fig, os.path.join(out, "fig_dwell_sorted"))


def fig_long(v, long_traces, path, thesis=False):
    lv = v[v["visit_s"] >= LONG_S].sort_values("visit_s", ascending=False)
    n = len(lv)
    if not n:
        return
    cols = 4
    rows = int(np.ceil(n / cols))
    size = (6.3, 1.6 * rows) if thesis else (3.6 * cols, 2.3 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, sharey=True)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, (i, r) in zip(axes.ravel(), lv.iterrows()):
        t, y = long_traces[i]
        ax.axvspan(0, r["visit_s"], color=CLASS_COLOURS[r["cls"]], alpha=0.15, lw=0)
        ax.plot(t, y, color="#1a1a2e", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(r["visit_s"], color="black", lw=0.6, ls="--")
        ax.axhline(0, color="grey", lw=0.4)
        name = r["stimulus"].replace("_", " ")
        if thesis:
            letter = "abcdefghijklmnop"[list(lv.index).index(i)]
            # what the arm played, and on which day
            ax.set_title(f"({letter}) {r['mouse'].replace('_', '.')}, {r['visit_s']:.1f} s\n"
                         f"{thesis_label(r['stimulus'])}, {THESIS_DAY[r['day']]}",
                         fontsize=6.5, loc="left")
            ax.tick_params(labelsize=5.5)
        else:
            ax.set_title(f"{r['mouse']}, {DAY_TITLE[r['day']]}, T{r['trial_ID']}\n"
                         f"{name[:34]}, {r['visit_s']:.1f} s", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time from entry (s)", fontsize=6.5 if thesis else None)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dopamine (z)", fontsize=6.5 if thesis else None)
    if not thesis:
        fig.suptitle(f"Every visit of {LONG_S:g} s or more: shaded while in the arm, "
                     "solid line entry, dashed line exit", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_bda(ok, path, thesis=False):
    ok = ok.copy()
    ok["length"] = pd.cut(ok["visit_s"], LENGTH_BINS, labels=LENGTH_NAMES, right=False)
    size = (6.3, 2.2) if thesis else (3.4 * len(CLASSES), 3.2)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=size, sharey=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(LENGTH_NAMES)))
    for ax, cls in zip(axes, CLASSES):
        g = ok[ok["cls"] == cls]
        for c, name in zip(colours, LENGTH_NAMES):
            h = g[g["length"] == name]
            if len(h) < 3:
                continue
            m = h[["before", "during", "after"]].mean()
            e = h[["before", "during", "after"]].sem()
            ax.errorbar([0, 1, 2], m, yerr=e, color=c, marker="o", ms=4, capsize=2, lw=1.2,
                        label=f"{name} (n={len(h)})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["before", "during", "after"] if thesis
                           else ["5 s before", "during", "5 s after"])
        ax.set_title(f"({'abcd'[CLASSES.index(cls)]}) {cls}" if thesis else cls,
                     loc="left" if thesis else "center")
        ax.legend(fontsize=5 if thesis else 6, frameon=False)
    axes[0].set_ylabel("Dopamine (z within block)")
    if not thesis:
        fig.suptitle("Dopamine before, during and after a visit, by what the arm played and "
                     "how long the mouse stayed (mean ± SEM of visits, both mice, all days)",
                     fontsize=10)
    fig.tight_layout()
    save(fig, path)


PSTH_DAYS = ["temporal_envelope_modulation", "intervals", "sequences", "vocalisations"]
TEM_ORDER = {"smooth": 0, "rough": 1, "rough_complex": 2}
SEQUENCE_ORDER = ["AAAAA", "AoAo", "ABAB", "BABA", "ABBA", "ABCABC"]


def psth_columns(v, day):
    """The stimuli of one day in the order of Figure 4.10, after the two silent
    references: any arm in a silent block, and the silent arm of the sound blocks."""
    d = v[(v["day"] == day) & v["cls"].isin(["vocalisation", "synthetic sound"])]
    names = set(d["stimulus"]) - {"vocalisation"}
    if day == "temporal_envelope_modulation":
        key = lambda s: (TEM_ORDER[s[:-4].rsplit(" ", 1)[0]], s)
    elif day == "intervals":
        key = lambda s: (sc.interval_periodicity_bits(s), s)
    elif day == "sequences":
        key = lambda s: (SEQUENCE_ORDER.index(s) if s in SEQUENCE_ORDER else 99, s)
    else:
        key = lambda s: s
    voc = ["vocalisation"] if "vocalisation" in set(d["stimulus"]) else []
    return ["silent block", "silent arm"] + voc + sorted(names, key=key)


def psth_tests(v, say):
    """During against before, visit by visit, for each stimulus of each day; Holm
    within the day. The visits of two mice, so these describe the two animals."""
    found = {}
    say("\n--- dopamine during a visit against the 5 s before, per stimulus and day "
        "(Wilcoxon signed rank over visits, Holm within the day)")
    for day in PSTH_DAYS:
        rows = []
        for stim in psth_columns(v, day):
            g = v[(v["day"] == day) & (v["stimulus"] == stim)].dropna(subset=["before", "during"])
            p = wilcoxon(g["during"], g["before"]).pvalue if len(g) >= 6 else np.nan
            rows.append((stim, len(g), g["before"].mean(), g["during"].mean(),
                         g["after"].mean(), p))
        ps = [r[5] for r in rows]
        ok = [i for i, p in enumerate(ps) if np.isfinite(p)]
        adj = dict(zip(ok, holm([ps[i] for i in ok])))
        say(f"  {THESIS_DAY[day]}")
        for i, (stim, n, b, d, a, p) in enumerate(rows):
            h = adj.get(i, np.nan)
            found[(day, stim)] = h
            say(f"    {thesis_label(stim):12s} n={n:3d}  before {b:+.2f}  during {d:+.2f}  "
                f"after {a:+.2f}  p={p:.3f}  Holm p={h:.3f}")
    return found


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


def stimulus_colour(stim, cls):
    """The colour of a stimulus's kind, as in Figure 4.10: consonant blue, dissonant
    orange, the envelopes in their own colours, sequences dark blue, vocalisations
    green; the silent block light and the silent arm dark grey."""
    from exp1_stimulus_pi import COLOURS
    if cls == "silent block":
        return "#9E9E9E"
    if cls in ("silent arm", "vocalisation"):
        return CLASS_COLOURS[cls]
    if stim.endswith(" kHz"):
        return COLOURS[stim[:-4].rsplit(" ", 1)[0]]
    if stim in sc.DISSONANT:
        return COLOURS["dissonant"]
    if stim == "unison":
        return COLOURS["unison"]
    if stim in sc.CONSONANT:
        return COLOURS["consonant"]
    return COLOURS["sequence"]


def fig_psth(v, warps, holm, path, thesis=False):
    """Dopamine before, during and after a visit, for every stimulus of every day:
    the stay of each visit stretched or squeezed to one width, between the 5 s
    before entry and the 5 s after exit in real time. Mean over the visits of both
    mice, with its standard error."""
    cols = max(len(psth_columns(v, d)) for d in PSTH_DAYS)
    size = (6.3, 5.0) if thesis else (2.0 * cols, 2.3 * len(PSTH_DAYS))
    fig, axes = plt.subplots(len(PSTH_DAYS), cols, figsize=size, sharex=True, sharey=True,
                             squeeze=False)
    step = WINDOW_S / WARP_N
    x = np.concatenate([-WINDOW_S + (np.arange(WARP_N) + 0.5) * step,
                        (np.arange(WARP_N) + 0.5) * step,
                        WINDOW_S + (np.arange(WARP_N) + 0.5) * step])
    for r, day in enumerate(PSTH_DAYS):
        names = psth_columns(v, day)
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(names):
                ax.set_visible(False)
                continue
            stim = names[c]
            idx = v.index[(v["day"] == day) & (v["stimulus"] == stim)]
            cls = v.loc[idx[0], "cls"] if len(idx) else "silent block"
            colour = stimulus_colour(stim, cls)
            mat = np.vstack([warps[i] for i in idx]) if len(idx) else np.full((1, 3 * WARP_N), np.nan)
            n_t = np.isfinite(mat).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.nanmean(mat, axis=0)
                e = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_t)
            ax.axvspan(0, WINDOW_S, color=colour, alpha=0.10, lw=0)
            ax.axhline(0, color="grey", lw=0.4)
            ax.axvline(0, color="black", lw=0.5)
            ax.axvline(WINDOW_S, color="black", lw=0.5, ls=(0, (3, 2)))
            ax.fill_between(x, m - e, m + e, color=colour, alpha=0.25, lw=0)
            ax.plot(x, m, color=colour, lw=0.9)
            mark = "*" if np.isfinite(holm.get((day, stim), np.nan)) and holm[(day, stim)] < 0.05 else ""
            name = "silent block" if stim == "silent block" else thesis_label(stim)
            ax.set_title(f"{name}{mark}\nn = {len(idx)}", fontsize=6 if thesis else 8, pad=2)
            ax.set_xticks([-WINDOW_S, 0, WINDOW_S, 2 * WINDOW_S])
            # the ends are 5 s before entry and 5 s after exit
            ax.set_xticklabels(["", "in", "out", ""])
            ax.tick_params(labelsize=5 if thesis else 7, length=2, pad=1)
            if c == 0:
                ax.set_ylabel(f"{THESIS_DAY[day]}\nDopamine (z)", fontsize=6.5 if thesis else 8)
    axes[0, 0].set_ylim(-1.5, 1.5)
    axes[0, 0].set_xlim(-WINDOW_S, 2 * WINDOW_S)
    if not thesis:
        fig.suptitle("Dopamine before, during (stretched to one width) and after a visit, "
                     "per stimulus and day; mean ± SEM of visits, both mice", fontsize=10)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, path)


def longest_visits(v, say):
    """Which sounds held the longest visits. The visits of LONG_S or more by class;
    and, on the days with one vocalisation arm among seven sound arms, whether the
    single longest visit to a sound in each session was to the vocalisation, set
    against the chance that it would be if visit length did not depend on what the
    arm played (each session's share of sound visits to the vocalisation; the
    count of sessions is Poisson binomial, the sessions taken as independent)."""
    sv = v[v["trial_ID"] > 1]
    long = sv[sv["visit_s"] >= LONG_S]
    say(f"\n--- visits of {LONG_S:g} s or more: {len(long)}, by class "
        + ", ".join(f"{c} {n}" for c, n in long["cls"].value_counts().items()))
    lv = long[long["cls"] == "vocalisation"]
    say(f"  to vocalisations {len(lv)}, of which on the vocalisation day {int((lv['day'] == 'vocalisations').sum())}")
    sound = sv[sv["cls"].isin(["vocalisation", "synthetic sound"]) & (sv["day"] != "vocalisations")]
    shares, hits = [], 0
    say("  days with one vocalisation arm: the longest visit to a sound in each session")
    for (day, mouse), g in sound.groupby(["day", "mouse"]):
        top = g.sort_values("visit_s", ascending=False).iloc[0]
        q = float((g["cls"] == "vocalisation").mean())
        shares.append(q)
        hits += int(top["cls"] == "vocalisation")
        say(f"    {mouse} {THESIS_DAY[day]:17s} longest {top['visit_s']:5.1f} s to {thesis_label(top['stimulus']):10s} "
            f"vocalisation share of sound visits {q:.2f}")
    dist = np.array([1.0])
    for q in shares:
        dist = np.convolve(dist, [1 - q, q])
    say(f"  the vocalisation held the longest visit in {hits} of {len(shares)} sessions, "
        f"{sum(shares):.2f} expected, p = {dist[hits:].sum():.3f}")


COMPLEX = "#AD1457"
OTHER_TEM = "#546E7A"


def complex_tones(v, say):
    """With temporal envelope modulation the cohort entered the two complex tones
    (changing in rate) more often than the four smooth and rough ones, and so spent
    more time on them (Chapter 4). The same comparison in the two photometry mice,
    and dopamine around those visits: the windows of each visit, complex against
    smooth or rough (Mann-Whitney over visits)."""
    day = "temporal_envelope_modulation"
    s = v[(v["day"] == day) & (v["cls"] == "synthetic sound")].copy()
    s["kind"] = np.where(s["stimulus"].str.startswith("rough_complex"), "complex", "smooth or rough")
    blocks = v[(v["day"] == day) & (v["trial_ID"] % 2 == 0)].groupby("mouse")["trial_ID"].nunique()
    rows = []
    for (mouse, kind), h in s.groupby(["mouse", "kind"]):
        arms = 2 if kind == "complex" else 4
        rows.append({"mouse": mouse, "kind": kind, "entries": len(h) / arms / blocks[mouse],
                     "time_s": h["visit_s"].sum() / arms / blocks[mouse],
                     "mean_visit_s": h["visit_s"].mean(), "visits": len(h)})
    beh = pd.DataFrame(rows)
    say("\n--- complex tones against smooth and rough, envelope day, per mouse "
        "(per arm in a sound block)")
    say(beh.round(2).to_string(index=False))
    ok = s.dropna(subset=["before", "during", "after"]).copy()
    ok["entry_change"] = ok["during"] - ok["before"]
    ok["exit_change"] = ok["after"] - ok["during"]
    say("dopamine, complex against smooth or rough (visits of both mice, Mann-Whitney)")
    for col in ("before", "during", "after", "entry_change", "exit_change"):
        a, b = ok.loc[ok["kind"] == "complex", col], ok.loc[ok["kind"] != "complex", col]
        say(f"  {col:12s} complex {a.mean():+.2f} (n={len(a)})  smooth or rough {b.mean():+.2f} "
            f"(n={len(b)})  p = {mannwhitneyu(a, b).pvalue:.3f}")
    return s, beh


def fig_complex(s, beh, warps, path, thesis=False):
    """(a, b) Entries and time per arm in a block, complex against smooth or rough,
    for each photometry mouse; (c) dopamine through those visits, each stay stretched
    to one width as in fig_psth."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2) if thesis else (11, 3.4),
                             gridspec_kw={"width_ratios": [0.7, 0.7, 1.6]})
    for ax, col, label in ((axes[0], "entries", "Entries per arm\nin a block"),
                           (axes[1], "time_s", "Time per arm\nin a block (s)")):
        for mouse, g in beh.groupby("mouse"):
            g = g.set_index("kind")
            y = [g.loc["smooth or rough", col], g.loc["complex", col]]
            ax.plot([0, 1], y, color="#9E9E9E", lw=0.8, zorder=1)
            ax.scatter([0, 1], y, color=[OTHER_TEM, COMPLEX], s=16, zorder=2)
            ax.text(1.08, y[1], mouse.replace("_", "."), va="center", fontsize=5.5 if thesis else 7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["smooth\nor rough", "complex"])
        ax.set_xlim(-0.4, 1.6)
        ax.set_ylim(0, None)
        ax.set_ylabel(label)
    axes[0].set_title("(a)" if thesis else "(a) entries", loc="left")
    axes[1].set_title("(b)" if thesis else "(b) time", loc="left")
    ax = axes[2]
    step = WINDOW_S / WARP_N
    x = np.concatenate([-WINDOW_S + (np.arange(WARP_N) + 0.5) * step, (np.arange(WARP_N) + 0.5) * step,
                        WINDOW_S + (np.arange(WARP_N) + 0.5) * step])
    ax.axvspan(0, WINDOW_S, color="#ECEFF1", lw=0)
    for kind, colour in (("smooth or rough", OTHER_TEM), ("complex", COMPLEX)):
        mat = np.vstack([warps[i] for i in s.index[s["kind"] == kind]])
        n_t = np.isfinite(mat).sum(axis=0)
        m = np.nanmean(mat, axis=0)
        e = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_t)
        ax.fill_between(x, m - e, m + e, color=colour, alpha=0.2, lw=0)
        ax.plot(x, m, color=colour, lw=1.0, label=f"{kind} (n = {len(mat)})")
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(WINDOW_S, color="black", lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks([-WINDOW_S, 0, WINDOW_S, 2 * WINDOW_S])
    ax.set_xticklabels(["−5 s", "entry", "exit", "+5 s"])
    ax.set_xlim(-WINDOW_S, 2 * WINDOW_S)
    ax.set_ylabel("Dopamine (z)")
    ax.legend(fontsize=5.5 if thesis else 7, frameon=False, loc="lower right")
    ax.set_title("(c)" if thesis else "(c) dopamine through the visit", loc="left")
    fig.tight_layout()
    save(fig, path)


# -- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?",
                        default=os.path.join(fa.DEFAULT_ROOT, "FIBPHO_ANALYSIS"),
                        help="the folder fibpho_alignment.py wrote")
    parser.add_argument("--thesis-figures", metavar="DIR",
                        help="also write the thesis versions (no titles, thesis fonts) "
                             "as fig_fp_long_visits and fig_fp_before_during_after")
    args = parser.parse_args()
    out = os.path.join(args.root, "dwell")
    os.makedirs(out, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.root, "alignment_summary.csv"))
    frames, traces, long_traces = [], {}, {}
    for _, s in summary.iterrows():
        events, z, block, fs = load(args.root, s["day"], s["mouse"])
        v = measure(events, z, block, fs, s["day"], s["mouse"])
        v["_z"] = [aligned(z, fs, t0, d, WINDOW_S, HEATMAP_MAX_S)
                   for t0, d in zip(v["t_entry"], v["visit_s"])]
        v["_w"] = [warped(z, block, fs, t0, d, tr)
                   for t0, d, tr in zip(v["t_entry"], v["visit_s"], v["trial_ID"])]
        # long visits: 10 s either side, in full
        v["_long"] = [None] * len(v)
        for k in v.index[v["visit_s"] >= LONG_S]:
            t0, d = v.loc[k, "t_entry"], v.loc[k, "visit_s"]
            i0, i1 = int((t0 - 10) * fs), int((t0 + d + 10) * fs)
            v.at[k, "_long"] = (np.arange(i0, i1) / fs - t0, z[i0:i1])
        frames.append(v)
    v = pd.concat(frames, ignore_index=True)
    traces = dict(zip(v.index, v.pop("_z")))
    warps = dict(zip(v.index, v.pop("_w")))
    long_traces = {k: x for k, x in zip(v.index, v.pop("_long")) if x is not None}
    v.to_csv(os.path.join(out, "visits.csv"), index=False)

    ok = run_stats(v, os.path.join(out, "stats.txt"))
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    holm_p = psth_tests(v, say)
    cx, cx_beh = complex_tones(v, say)
    longest_visits(v, say)
    with open(os.path.join(out, "stats_psth.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nfigures:")
    fig_sorted(v, traces, out)
    fig_long(v, long_traces, os.path.join(out, "fig_long_visits"))
    fig_bda(ok, os.path.join(out, "fig_before_during_after"))
    fig_psth(v, warps, holm_p, os.path.join(out, "fig_psth"))
    fig_complex(cx, cx_beh, warps, os.path.join(out, "fig_complex"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            fig_long(v, long_traces, os.path.join(args.thesis_figures, "fig_fp_long_visits"),
                     thesis=True)
            fig_bda(ok, os.path.join(args.thesis_figures, "fig_fp_before_during_after"),
                    thesis=True)
            fig_psth(v, warps, holm_p, os.path.join(args.thesis_figures, "fig_fp_psth"),
                     thesis=True)
            fig_complex(cx, cx_beh, warps, os.path.join(args.thesis_figures, "fig_fp_complex"),
                        thesis=True)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
