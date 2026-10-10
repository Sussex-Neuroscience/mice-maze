"""
Fiber Photometry <-> Auditory Maze Alignment & Plotting
=======================================================

Aligns every Tucker-Davis Technologies (TDT) fiber photometry recording
with the auditory maze session it was made during, using the TTL pulses
the maze sent on every sound, and computes delta F/F for each.

Expected layout (DATA_ROOT is the "dopamine recordings" folder):

    DATA_ROOT/
        dopaminergic recordings/      one TDT block (tank) per recording
            MickeyMouse-251202-132525/
            ...
        temporal_envelope_modulation/ one folder per experiment day,
        intervals/                    holding the maze sessions
            time_2025-12-03_12_08_38mouseK30_5/
                trials_time_*.csv
                *_detailed_visits.csv
        ...

For each maze session the script:
  1. Reads the TTL epocs of every TDT block recorded that day
  2. Finds the clock offset by a vote: every difference between a logged
     sound onset and a TTL onset is a candidate, and the true offset is
     the one most of them share. Stray pulses (tests sent before the
     session, a restarted session) do not move it.
  3. Takes the block with the most matched pulses as the session's own
  4. Reads its 405nm isosbestic and 465nm GRAB-DA2m streams
  5. Calculates delta F/F: each channel divided by a double exponential
     fitted over the session to remove its bleaching, then motion-
     corrected via isosbestic, fitted separately within each block
  6. Plots delta F/F with every entry marked by stimulus, and the
     peri-event average for each stimulus

Every arm entry in a block is an event, labelled by its stimulus: the
sounds, the silent arm of the sound blocks, and the entries made in the
silent blocks. A sound event is timed by its TTL where one matched, and
by its logged onset converted to the TDT clock otherwise.

Usage:
    python fibpho_alignment.py [DATA_ROOT] [--out OUT_DIR]

    DATA_ROOT defaults to $MAZE_PHOTOMETRY_DIR, then to the Box copy.
    OUT_DIR defaults to DATA_ROOT/FIBPHO_ANALYSIS.

Outputs:
    OUT_DIR/alignment_summary.csv        one row per session
    OUT_DIR/<day>/<mouse>/
        fibpho_aligned_overview.png/.pdf full-session delta F/F with entries
        fibpho_trial_panels.png/.pdf     per-sound-block zoomed panels
        fibpho_peri_event.png/.pdf       peri-event average per stimulus
        fibpho_peri_event_pooled.png/.pdf
        alignment_report.csv             every TTL and the event it matched
        events.csv                       every entry, in both clocks
        blocks.csv                       block boundaries, in both clocks
        dff_downsampled.npz              time and delta F/F at ~100 Hz
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def safe_savefig(fig, path, **kwargs):
    """Save figure, catching PermissionError for locked files (e.g. open PDFs)."""
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: Could not write {os.path.basename(path)} (file locked)")


# -- configuration -----------------------------------------------------

DEFAULT_ROOT = os.environ.get("MAZE_PHOTOMETRY_DIR", os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "dopamine recordings"))
TANK_FOLDER = "dopaminergic recordings"

# TDT store names
STREAM_465 = "_465A"   # dopamine sensor (GRAB-DA2m)
STREAM_405 = "_405A"   # Isosbestic reference
EPOC_TTL = "MTL_"      # TTL epoc store

# Alignment
VOTE_BIN_S = 0.2       # width of the offset histogram bins
MATCH_TOL_S = 0.25     # a TTL and a logged onset this close are one event

# Signal processing
DOWNSAMPLE = 10        # downsample factor for plotting (1017 Hz -> ~102 Hz)
SMOOTH_WIN = 51        # Savitzky-Golay window for dF/F smoothing
BASELINE_PERCENTILE = 10  # percentile for F0 baseline
PERI_EVENT_WINDOW = (-3, 7)  # seconds around each event for PETH
MIN_EVENTS_PETH = 3    # a stimulus needs this many events for a PETH line

FIXED_COLOURS = {
    "silent arm": "#9E9E9E",
    "silent block": "#C8C8C8",
    "habituation": "#E0E0E0",
    "vocalisation": "#2E7D32",
}


# -- helper functions --------------------------------------------------

def stimulus_label(stimulus, trial_id):
    """A short name for what an arm entry played.

    The stimulus field is 'key:value | key:value ...', with different keys
    on different experiment days, or a path to a .wav on the vocalisation
    day. Odd trials are silent (1 = habituation, 3, 5, 7, 9 between the
    sound blocks), whatever the field says.
    """
    if int(trial_id) == 1:
        return "habituation"
    if int(trial_id) % 2 == 1:
        return "silent block"
    fields = {}
    for part in str(stimulus).split("|"):
        if ":" in part:
            key, value = part.split(":", 1)
            fields[key.strip()] = value.strip()
    freq = fields.get("frequency", "")
    if freq.lower().endswith(".wav") or "/" in freq or "\\" in freq:
        return os.path.splitext(os.path.basename(freq))[0]
    if "interval_type" in fields:          # intervals and vocalisation days
        kind = fields["interval_type"]
        if kind in ("silent", "control"):
            return "silent arm"
        if kind == "vocalisation":
            return "vocalisation"
        return fields.get("interval_name") or kind
    if "sound_type" in fields:             # temporal envelope modulation
        kind = fields["sound_type"]
        if kind == "control":
            return "vocalisation" if freq == "vocalisation" else "silent arm"
        try:
            return f"{kind} {float(freq) / 1000:g} kHz"
        except ValueError:
            return kind
    if "pattern" in fields:                # sequences
        pattern = fields["pattern"]
        return "silent arm" if pattern in ("silence", "0") else pattern
    return str(stimulus)[:30]


def label_colours(labels):
    """One colour per label, fixed for the controls and cycled for the rest."""
    others = sorted(l for l in set(labels) if l not in FIXED_COLOURS)
    # tab10 without its green and grey, which the controls use
    cycle = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i not in (2, 7)]
    cycle += list(plt.get_cmap("Dark2").colors)
    colours = dict(FIXED_COLOURS)
    colours.update({l: matplotlib.colors.to_hex(cycle[i % len(cycle)])
                    for i, l in enumerate(others)})
    return colours


def tank_start(tank_dir):
    """Wall-clock start of a TDT block, from its folder name."""
    m = re.search(r"-(\d{6})-(\d{6})$", os.path.basename(tank_dir))
    return datetime.strptime("".join(m.groups()), "%y%m%d%H%M%S") if m else None


def session_info(session_dir):
    """Date and mouse of a maze session, from its folder name."""
    m = re.search(r"time_(\d{4}-\d{2}-\d{2})_(\d{2}_\d{2}_\d{2})mouse(.+)$",
                  os.path.basename(session_dir))
    if not m:
        return None
    start = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H_%M_%S")
    return start, m.group(3)


def vote_offset(event_times, ttl_onsets):
    """The clock offset (event clock - TDT clock) most TTLs agree on."""
    if len(event_times) == 0 or len(ttl_onsets) == 0:
        return np.nan
    diffs = (np.asarray(event_times)[:, None] - np.asarray(ttl_onsets)[None, :]).ravel()
    bins = np.floor(diffs / VOTE_BIN_S)
    values, counts = np.unique(bins, return_counts=True)
    best = values[np.argmax(counts)]
    near = diffs[np.abs(bins - best) <= 1]   # the winning bin and its neighbours
    return float(np.median(near))


def match_ttls(event_times, ttl_onsets, offset):
    """Pair each TTL with the nearest logged onset, one to one.

    Returns (ttl index, event index, residual in s) for every pair closer
    than MATCH_TOL_S, taking the closest pairs first.
    """
    ev_tdt = np.asarray(event_times) - offset
    pairs = []
    for i, t in enumerate(ttl_onsets):
        j = int(np.argmin(np.abs(ev_tdt - t)))
        r = ev_tdt[j] - t
        if abs(r) <= MATCH_TOL_S:
            pairs.append((abs(r), i, j, r))
    used_ttl, used_ev, out = set(), set(), []
    for _, i, j, r in sorted(pairs):
        if i not in used_ttl and j not in used_ev:
            used_ttl.add(i)
            used_ev.add(j)
            out.append((i, j, r))
    return sorted(out)


def bleaching_curve(x, time, span):
    """A double exponential fitted to one channel over the session.

    Photobleaching makes each channel decay smoothly over the recording,
    and the two decay at different rates. The curve is fitted to 1 s means
    between span[0] and span[1] (the session), with a fast component of
    30 s to 50 min and a slow one of 5 min upwards. A monotone curve over
    the whole session cannot follow anything that repeats block by block.
    Falls back to one exponential, then a straight line, if the fit fails.
    Returns the curve at every sample and its name.
    """
    sel = np.flatnonzero((time >= span[0]) & (time < span[1]))
    step = max(1, int(round(1 / (time[1] - time[0]))))
    m = len(sel) // step
    tb = time[sel[:m * step]].reshape(m, step).mean(axis=1)
    xb = x[sel[:m * step]].reshape(m, step).mean(axis=1)
    t0, drop = tb[0], max(xb[0] - xb[-1], 1e-9)

    def double(u, a, ta, b, tb_, c):
        return a * np.exp(-u / ta) + b * np.exp(-u / tb_) + c

    def single(u, a, ta, c):
        return a * np.exp(-u / ta) + c

    try:
        p, _ = curve_fit(double, tb - t0, xb,
                         p0=[drop / 2, 300, drop / 2, 3000, max(xb[-1], 0)],
                         bounds=([0, 30, 0, 300, 0], [np.inf, 3000, np.inf, 1e6, np.inf]),
                         maxfev=20000)
        return double(time - t0, *p), "double exponential"
    except (RuntimeError, ValueError):
        pass
    try:
        p, _ = curve_fit(single, tb - t0, xb, p0=[drop, 1000, max(xb[-1], 0)],
                         bounds=([0, 30, 0], [np.inf, 1e6, np.inf]), maxfev=20000)
        return single(time - t0, *p), "single exponential"
    except (RuntimeError, ValueError):
        return np.polyval(np.polyfit(tb, xb, 1), time), "line"


def compute_delta_f_over_f(signal_465, signal_405, fs, blocks,
                           smooth_win=SMOOTH_WIN,
                           baseline_pct=BASELINE_PERCENTILE):
    """Compute motion-corrected delta F/F, detrended, then fitted block by block.

    Each channel is first divided by its own bleaching curve over the
    session (bleaching_curve), since the two bleach at different rates.
    Then the isosbestic (405nm) channel is used to remove motion
    artefacts via linear regression, made separately within each
    block of the session.

    Parameters
    ----------
    signal_465 : array - dopamine sensor (GRAB-DA) signal
    signal_405 : array - isosbestic reference signal
    fs : float - sampling rate (Hz)
    blocks : list of (start_s, end_s) - block boundaries in recording
             time; samples outside every block are left as NaN
    smooth_win : int - Savitzky-Golay smoothing window
    baseline_pct : float - percentile for F0 estimation, per block

    Returns
    -------
    dff : array - delta F/F (dimensionless), NaN outside the blocks
    time : array - time vector (seconds)
    bleaching : dict - for each channel, the fitted curve's form and its
                fall over the session in percent
    """
    # ensure same length
    n = min(len(signal_465), len(signal_405))
    sig = signal_465[:n].astype(np.float64)
    ref = signal_405[:n].astype(np.float64)
    time = np.arange(n) / fs

    # 1. smooth both channels
    if smooth_win > 3 and smooth_win < len(sig):
        win = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
        sig_s = savgol_filter(sig, win, polyorder=3)
        ref_s = savgol_filter(ref, win, polyorder=3)
    else:
        sig_s = sig
        ref_s = ref

    # 2. remove each channel's bleaching over the session
    span = (min(b[0] for b in blocks), max(b[1] for b in blocks))
    in_span = (time >= span[0]) & (time < span[1])
    bleaching = {}
    for name, x in (("465", sig_s), ("405", ref_s)):
        curve, form = bleaching_curve(x, time, span)
        c = curve[in_span]
        bleaching[name] = (form, 100 * (1 - c[-1] / c[0]))
        x /= curve      # in place: sig_s and ref_s are now detrended

    dff = np.full(n, np.nan)
    for start_s, end_s in blocks:
        in_block = (time >= start_s) & (time < end_s)
        if in_block.sum() < 2:
            continue
        # 3. linear regression within the block: sig = a * ref + b
        coeffs = np.polyfit(ref_s[in_block], sig_s[in_block], deg=1)
        predicted_465 = np.polyval(coeffs, ref_s[in_block])

        # 4. delta F / F = (signal - predicted) / predicted
        block_dff = (sig_s[in_block] - predicted_465) / predicted_465

        # 5. baseline: the block's own low percentile
        dff[in_block] = block_dff - np.percentile(block_dff, baseline_pct)

    return dff, time, bleaching


def peri_event_traces(dff_ds, fs_ds, times):
    """Windows of dF/F around each time; windows reaching outside the blocks are skipped."""
    pre_s, post_s = PERI_EVENT_WINDOW
    pre, post = int(abs(pre_s) * fs_ds), int(post_s * fs_ds)
    traces = []
    for t in times:
        idx = int(round(t * fs_ds))
        if idx - pre < 0 or idx + post > len(dff_ds):
            continue
        trace = dff_ds[idx - pre:idx + post]
        if np.isfinite(trace).all():
            traces.append(trace)
    return np.array(traces), np.linspace(pre_s, post_s, pre + post)


# -- discovery ---------------------------------------------------------

def find_sessions(root):
    """Every maze session under the experiment-day folders."""
    sessions = []
    for day_dir in sorted(glob.glob(os.path.join(root, "*"))):
        day = os.path.basename(day_dir)
        if not os.path.isdir(day_dir) or day in (TANK_FOLDER, "FIBPHO_ANALYSIS"):
            continue
        for s in sorted(glob.glob(os.path.join(day_dir, "time_*"))):
            info = session_info(s)
            visits = glob.glob(os.path.join(s, "*_detailed_visits.csv"))
            trials = glob.glob(os.path.join(s, "trials_time_*.csv"))
            if info and visits and trials:
                sessions.append({"day": day, "dir": s, "start": info[0],
                                 "mouse": info[1], "visits": visits[0],
                                 "trials": trials[0]})
    return sessions


def load_events(session):
    """Every arm entry of the session, labelled, with block boundaries."""
    v = pd.read_csv(session["visits"])
    v = v[~v["stimulus"].astype(str).str.contains("Unknown", na=False)].copy()
    v["label"] = [stimulus_label(s, t) for s, t in zip(v["stimulus"], v["trial_ID"])]
    v = v.sort_values("sound_on_time").reset_index(drop=True)
    t = pd.read_csv(session["trials"])
    blocks = (t.groupby("trial_ID", as_index=False)
                .agg(start=("trial_start_time", "first"), end=("end_trial_time", "first")))
    return v, blocks


# -- one session -------------------------------------------------------

def process_session(session, tanks, tank_epocs, out_root, tdt):
    day, mouse = session["day"], session["mouse"]
    print(f"\n{'=' * 60}\n{day} / {mouse}\n{'=' * 60}")
    events, blocks = load_events(session)
    sound = events[events["trial_ID"] % 2 == 0]       # entries in sound blocks
    sound_times = sound["sound_on_time"].to_numpy()

    # 1-3. pick the block recorded during this session, by matched TTLs
    best = None
    for tank in tanks:
        start = tank_start(tank)
        if start is None or start.date() != session["start"].date():
            continue
        onsets = tank_epocs[tank][0]
        offset = vote_offset(sound_times, onsets)
        if np.isnan(offset):
            continue
        n = len(match_ttls(sound_times, onsets, offset))
        print(f"  candidate {os.path.basename(tank)}: {len(onsets)} TTLs, {n} matched")
        if best is None or n > best[2]:
            best = (tank, offset, n)
    if best is None or best[2] < 5:
        print("  no recording matches this session; skipped")
        return None
    tank, offset, _ = best
    onsets, offs = tank_epocs[tank]
    pairs = match_ttls(sound_times, onsets, offset)
    # refine the offset on the matched pairs alone
    offset += float(np.median([r for _, _, r in pairs]))
    pairs = match_ttls(sound_times, onsets, offset)
    residuals = np.array([r for _, _, r in pairs])
    print(f"  using {os.path.basename(tank)}: {len(pairs)} of {len(onsets)} TTLs "
          f"matched {len(sound)} sound-block entries; residual sd "
          f"{residuals.std() * 1000:.1f} ms, max {np.abs(residuals).max() * 1000:.1f} ms")

    out_dir = os.path.join(out_root, day, mouse)
    os.makedirs(out_dir, exist_ok=True)

    # every entry in the TDT clock: its TTL where one matched
    events["tdt_time_s"] = events["sound_on_time"] - offset
    events["has_ttl"] = False
    events["ttl_duration_s"] = np.nan
    sound_idx = sound.index.to_numpy()
    report = []
    for i, j, r in pairs:
        k = sound_idx[j]
        events.loc[k, ["tdt_time_s", "has_ttl", "ttl_duration_s"]] = \
            [onsets[i], True, offs[i] - onsets[i]]
        report.append({"ttl_idx": i, "tdt_time_s": onsets[i],
                       "ttl_duration_s": offs[i] - onsets[i],
                       "csv_sound_on": events.loc[k, "sound_on_time"],
                       "residual_s": r, "trial_ID": events.loc[k, "trial_ID"],
                       "ROI_visited": events.loc[k, "ROI_visited"],
                       "label": events.loc[k, "label"]})
    matched = {i for i, _, _ in pairs}
    for i in range(len(onsets)):
        if i not in matched:
            report.append({"ttl_idx": i, "tdt_time_s": onsets[i],
                           "ttl_duration_s": offs[i] - onsets[i], "label": "unmatched"})
    pd.DataFrame(report).sort_values("ttl_idx").to_csv(
        os.path.join(out_dir, "alignment_report.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)

    # 4-5. signals and delta F/F within each block
    blocks["start_tdt"] = blocks["start"] - offset
    blocks["end_tdt"] = blocks["end"] - offset
    blocks["is_silent"] = blocks["trial_ID"] % 2 == 1
    blocks.to_csv(os.path.join(out_dir, "blocks.csv"), index=False)
    # stores are named without the underscore tdt adds for attribute access
    data = tdt.read_block(tank, store=[STREAM_465.lstrip("_"), STREAM_405.lstrip("_")])
    fs = data.streams[STREAM_465].fs
    dff, time_dff, bleaching = compute_delta_f_over_f(
        data.streams[STREAM_465].data, data.streams[STREAM_405].data, fs,
        list(zip(blocks["start_tdt"], blocks["end_tdt"])))
    for ch, (form, fall) in bleaching.items():
        print(f"  {ch} nm bleaching over the session: {fall:.1f}% ({form})")
    dff_ds, time_ds = dff[::DOWNSAMPLE], time_dff[::DOWNSAMPLE]
    fs_ds = fs / DOWNSAMPLE
    np.savez_compressed(os.path.join(out_dir, "dff_downsampled.npz"),
                        time_s=time_ds.astype(np.float32),
                        dff=dff_ds.astype(np.float32), fs=fs_ds)

    # 6. figures
    colours = label_colours(events["label"])
    title = f"Mouse {mouse}, {day.replace('_', ' ')}"
    plot_overview(events, blocks, dff_ds, time_ds, colours, title, out_dir)
    plot_trial_panels(events, blocks, dff_ds, time_ds, colours, title, out_dir)
    n_peth = plot_peri_event(events, dff_ds, fs_ds, colours, title, out_dir)

    return {"day": day, "mouse": mouse, "session": os.path.basename(session["dir"]),
            "tank": os.path.basename(tank), "recording_min": round(time_dff[-1] / 60, 1),
            "ttls": len(onsets), "sound_block_entries": len(sound),
            "matched": len(pairs), "unmatched_ttls": len(onsets) - len(pairs),
            "residual_sd_ms": round(residuals.std() * 1000, 1),
            "residual_max_ms": round(np.abs(residuals).max() * 1000, 1),
            "offset_s": round(offset, 4), "events_in_peth": n_peth,
            "bleach_465_pct": round(bleaching["465"][1], 1),
            "bleach_405_pct": round(bleaching["405"][1], 1)}


# -- figures -----------------------------------------------------------

def _clip_ylim(ax, time_ds, dff_ds):
    # Clip y-axis to exclude the initial ramp-up outlier: use 0.5th-99.5th
    # percentile of the signal AFTER the first 60 seconds
    stable = (time_ds > 60) & np.isfinite(dff_ds)
    if stable.any():
        lo, hi = np.percentile(dff_ds[stable], [0.5, 99.5])
        margin = (hi - lo) * 0.05
        ax.set_ylim(lo - margin, hi + margin)


def plot_overview(events, blocks, dff_ds, time_ds, colours, title, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                             gridspec_kw={"height_ratios": [4, 0.5]})
    ax1 = axes[0]
    # entry bars BEHIND the trace; sound entries span their TTL
    for _, ev in events[events["trial_ID"] > 1].iterrows():
        dur = ev["ttl_duration_s"] if ev["has_ttl"] else ev["time_spent_seconds"]
        ax1.axvspan(ev["tdt_time_s"], ev["tdt_time_s"] + dur, alpha=0.35,
                    color=colours[ev["label"]], zorder=1, linewidth=0)
    ax1.plot(time_ds, dff_ds, color="#1a1a2e", linewidth=0.3, alpha=0.9, zorder=2)
    _clip_ylim(ax1, time_ds, dff_ds)
    ax1.set_ylabel("ΔF/F", fontsize=12)
    ax1.set_title(f"{title}: GRAB-DA (465nm), motion-corrected via 405nm isosbestic\n"
                  "Coloured bars = arm entries, by stimulus", fontsize=13)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    for _, tb in blocks.iterrows():
        colour = "#2CA02C" if not tb["is_silent"] else "#D62728"
        ax2.barh(0, tb["end_tdt"] - tb["start_tdt"], left=tb["start_tdt"],
                 height=0.8, color=colour, alpha=0.6, edgecolor="white", linewidth=0.5)
        ax2.text((tb["start_tdt"] + tb["end_tdt"]) / 2, 0, f"T{int(tb['trial_ID'])}",
                 ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_ylabel("Trial", fontsize=10)
    ax2.set_xlabel("Time (seconds from TDT recording start)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    shown = [l for l in colours if l in set(events["label"]) and l != "habituation"]
    patches = [mpatches.Patch(color=colours[l], label=l.replace("_", " ")) for l in shown]
    patches += [mpatches.Patch(color="#2CA02C", alpha=0.6, label="Sound block"),
                mpatches.Patch(color="#D62728", alpha=0.6, label="Silent block")]
    ax1.legend(handles=patches, loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fibpho_aligned_overview.png"), dpi=200,
                bbox_inches="tight")
    safe_savefig(fig, os.path.join(out_dir, "fibpho_aligned_overview.pdf"),
                 bbox_inches="tight")
    plt.close(fig)


def plot_trial_panels(events, blocks, dff_ds, time_ds, colours, title, out_dir):
    active = blocks[~blocks["is_silent"]]
    if active.empty:
        return
    fig, axes = plt.subplots(len(active), 1, figsize=(18, 4 * len(active)), squeeze=False)
    for ax, (_, tb) in zip(axes[:, 0], active.iterrows()):
        tid = int(tb["trial_ID"])
        t0, t1 = max(tb["start_tdt"] - 30, 0), min(tb["end_tdt"] + 30, time_ds[-1])
        mask = (time_ds >= t0) & (time_ds <= t1)
        ax.plot(time_ds[mask], dff_ds[mask], color="#1a1a2e", linewidth=0.5,
                alpha=0.9, zorder=2)
        ax.axvline(tb["start_tdt"], color="#888888", linestyle="--", alpha=0.4, lw=0.8)
        ax.axvline(tb["end_tdt"], color="#888888", linestyle="--", alpha=0.4, lw=0.8)
        trial_events = events[events["trial_ID"] == tid]
        y_top = ax.get_ylim()[1]
        for _, ev in trial_events.iterrows():
            dur = ev["ttl_duration_s"] if ev["has_ttl"] else ev["time_spent_seconds"]
            ax.axvspan(ev["tdt_time_s"], ev["tdt_time_s"] + dur, alpha=0.30,
                       color=colours[ev["label"]], zorder=1, linewidth=0)
            ax.text(ev["tdt_time_s"], y_top * 0.95, ev["ROI_visited"], fontsize=5,
                    rotation=90, ha="right", va="top", color=colours[ev["label"]],
                    alpha=0.7, zorder=3)
        ax.set_ylabel("ΔF/F", fontsize=10)
        ax.set_title(f"Trial {tid} (sound block) - {len(trial_events)} entries",
                     fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        patches = [mpatches.Patch(color=colours[l], label=l.replace("_", " "))
                   for l in sorted(trial_events["label"].unique())]
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (seconds from TDT recording start)", fontsize=11)
    plt.suptitle(f"{title}: ΔF/F in each sound block", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fibpho_trial_panels.png"), dpi=200,
                bbox_inches="tight")
    safe_savefig(fig, os.path.join(out_dir, "fibpho_trial_panels.pdf"),
                 bbox_inches="tight")
    plt.close(fig)


def plot_peri_event(events, dff_ds, fs_ds, colours, title, out_dir):
    """Peri-event average per stimulus, and pooled over every sound."""
    pre_s, post_s = PERI_EVENT_WINDOW
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in [l for l in colours if l in set(events["label"]) and l != "habituation"]:
        traces, t = peri_event_traces(dff_ds, fs_ds,
                                      events.loc[events["label"] == label, "tdt_time_s"])
        if len(traces) < MIN_EVENTS_PETH:
            continue
        mean, sem = traces.mean(axis=0), traces.std(axis=0) / np.sqrt(len(traces))
        ax.plot(t, mean, color=colours[label], linewidth=1.5,
                label=f"{label.replace('_', ' ')} (n={len(traces)})")
        ax.fill_between(t, mean - sem, mean + sem, color=colours[label], alpha=0.15)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1, label="Entry")
    ax.axhline(0, color="grey", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time relative to arm entry (s)", fontsize=12)
    ax.set_ylabel("ΔF/F (mean ± SEM)", fontsize=12)
    ax.set_title(f"{title}: peri-event ΔF/F by stimulus", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(pre_s, post_s)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fibpho_peri_event.png"), dpi=200,
                bbox_inches="tight")
    safe_savefig(fig, os.path.join(out_dir, "fibpho_peri_event.pdf"), bbox_inches="tight")
    plt.close(fig)

    # pooled over every sound (sound blocks, silent arm excluded)
    sounds = events[(events["trial_ID"] % 2 == 0) & (events["label"] != "silent arm")]
    traces, t = peri_event_traces(dff_ds, fs_ds, sounds["tdt_time_s"])
    if len(traces) == 0:
        return 0
    mean, sem = traces.mean(axis=0), traces.std(axis=0) / np.sqrt(len(traces))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean, color="#1a1a2e", linewidth=2)
    ax.fill_between(t, mean - sem, mean + sem, color="#1a1a2e", alpha=0.15)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Sound onset")
    ax.axhline(0, color="grey", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time relative to sound onset (s)", fontsize=12)
    ax.set_ylabel("ΔF/F (mean ± SEM)", fontsize=12)
    ax.set_title(f"{title}: pooled peri-event ΔF/F (n={len(traces)} sounds)", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(pre_s, post_s)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fibpho_peri_event_pooled.png"), dpi=200,
                bbox_inches="tight")
    safe_savefig(fig, os.path.join(out_dir, "fibpho_peri_event_pooled.pdf"),
                 bbox_inches="tight")
    plt.close(fig)
    return len(traces)


# -- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                        help="the 'dopamine recordings' folder")
    parser.add_argument("--out", help="output folder (default ROOT/FIBPHO_ANALYSIS)")
    args = parser.parse_args()
    try:
        import tdt
    except ImportError:
        print("ERROR: tdt package not installed. Install with: pip install tdt")
        sys.exit(1)

    root = args.root
    out_root = args.out or os.path.join(root, "FIBPHO_ANALYSIS")
    sessions = find_sessions(root)
    tanks = sorted(d for d in glob.glob(os.path.join(root, TANK_FOLDER, "*"))
                   if os.path.isdir(d))
    print(f"{len(sessions)} maze sessions, {len(tanks)} TDT blocks under {root}")

    # TTL epocs of every block, read once
    tank_epocs = {}
    for tank in tanks:
        ep = tdt.read_block(tank, evtype=["epocs"]).epocs
        if EPOC_TTL in ep.keys():
            tank_epocs[tank] = (np.asarray(ep[EPOC_TTL].onset),
                                np.asarray(ep[EPOC_TTL].offset))
    tanks = [t for t in tanks if t in tank_epocs]

    rows = [r for s in sessions
            if (r := process_session(s, tanks, tank_epocs, out_root, tdt))]
    summary = pd.DataFrame(rows)
    os.makedirs(out_root, exist_ok=True)
    summary.to_csv(os.path.join(out_root, "alignment_summary.csv"), index=False)
    print(f"\n{'=' * 60}\nAlignment summary\n{'=' * 60}")
    print(summary.drop(columns=["session", "offset_s"]).to_string(index=False))
    print(f"\nAll outputs saved to:\n  {out_root}")


if __name__ == "__main__":
    main()
