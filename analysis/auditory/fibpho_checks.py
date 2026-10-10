"""
Fiber Photometry: Checks on the Signal Around Arm Entries
=========================================================

Three checks that decide how far the photometry can be read, using what
fibpho_alignment.py saved and the TDT recordings themselves:

  1. Entries against chance. The response to a sound entry (mean z over
     the 5 s after it minus the 2 s before) is compared with the same
     measure at random moments in the same sound blocks: a permutation
     test of the mean, per session.
  2. Each channel on its own. The 465 nm (dopamine) and 405 nm channels
     are each detrended for bleaching and z-scored within block, without
     either correcting the other, and averaged around sound entries. A
     change in the 465 nm channel alone cannot have been made by the
     correction.
  3. The reference channel. Within each sound block, the correlation
     between the two detrended channels and the slope of the fit. Motion
     moves both channels the same way, so a negative correlation means the
     405 nm channel is not a clean motion reference for this sensor.

Usage:
    python fibpho_checks.py [DATA_ROOT]

    DATA_ROOT is the "dopamine recordings" folder; results go to
    DATA_ROOT/FIBPHO_ANALYSIS/checks/.

Outputs:
    diagnostics.csv              per session: block spread, entry and random
                                 responses, permutation p, channel correlation
    channels_at_entry.csv        each channel's mean z before, at and after entry
    fig_entry_vs_random.png      sound entries, silent-arm entries and random
                                 moments, in z
    fig_channels_at_entry.png    each channel on its own around sound entries
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibpho_alignment as fa

BASE, RESP = (-2.0, 0.0), (0.0, 5.0)
WIDE = (-10.0, 15.0)
N_RANDOM = 4000
N_PERMUTATIONS = 5000
DAYS = ["temporal_envelope_modulation", "intervals", "sequences", "vocalisations"]


def zscore_blocks(x, t, blocks, sound_only=False):
    z = np.full_like(x, np.nan)
    for _, b in blocks.iterrows():
        if sound_only and b["is_silent"]:
            continue
        m = (t >= b["start_tdt"]) & (t < b["end_tdt"]) & np.isfinite(x)
        if m.sum() > 10:
            z[m] = (x[m] - x[m].mean()) / x[m].std()
    return z


def responses(z, fs, times):
    out = []
    for tt in times:
        i0, i1 = int(round((tt + BASE[0]) * fs)), int(round((tt + BASE[1]) * fs))
        j0, j1 = int(round((tt + RESP[0]) * fs)), int(round((tt + RESP[1]) * fs))
        if i0 < 0 or j1 > len(z):
            continue
        a, b = z[i0:i1], z[j0:j1]
        if np.isfinite(a).all() and np.isfinite(b).all():
            out.append(b.mean() - a.mean())
    return np.array(out)


def traces(z, fs, times):
    pre, post = int(-WIDE[0] * fs), int(WIDE[1] * fs)
    out = []
    for tt in times:
        i = int(round(tt * fs))
        if i - pre >= 0 and i + post <= len(z):
            w = z[i - pre:i + post]
            if np.isfinite(w).all():
                out.append(w)
    return np.array(out), np.linspace(WIDE[0], WIDE[1], pre + post)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default=fa.DEFAULT_ROOT,
                        help="the 'dopamine recordings' folder")
    args = parser.parse_args()
    import tdt
    fa_root = os.path.join(args.root, "FIBPHO_ANALYSIS")
    out = os.path.join(fa_root, "checks")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)

    summary = pd.read_csv(os.path.join(fa_root, "alignment_summary.csv"))
    mice = sorted(summary["mouse"].unique())
    fig1, ax1 = plt.subplots(len(mice), 4, figsize=(15, 6), sharex=True, squeeze=False)
    fig2, ax2 = plt.subplots(len(mice), 4, figsize=(15, 6), sharex=True, squeeze=False)
    rows, chan_rows = [], []
    for _, s in summary.iterrows():
        folder = os.path.join(fa_root, s["day"], s["mouse"])
        ev = pd.read_csv(os.path.join(folder, "events.csv"))
        bl = pd.read_csv(os.path.join(folder, "blocks.csv"))
        npz = np.load(os.path.join(folder, "dff_downsampled.npz"))
        t, dff, fs = npz["time_s"].astype(float), npz["dff"].astype(float), float(npz["fs"])
        z = zscore_blocks(dff, t, bl)
        sb = bl[~bl["is_silent"]]
        in_sb = ev["trial_ID"].isin(sb["trial_ID"])
        snd = ev.loc[in_sb & (ev["label"] != "silent arm"), "tdt_time_s"]
        sil = ev.loc[in_sb & (ev["label"] == "silent arm"), "tdt_time_s"]

        # 1. entries against random moments in the same sound blocks
        lens = (sb["end_tdt"] - sb["start_tdt"]).to_numpy()
        pick = rng.choice(len(sb), N_RANDOM, p=lens / lens.sum())
        rand_t = sb["start_tdt"].to_numpy()[pick] + rng.uniform(3, lens[pick] - 6)
        r_snd, r_sil, r_rand = responses(z, fs, snd), responses(z, fs, sil), responses(z, fs, rand_t)
        null = np.array([rng.choice(r_rand, len(r_snd), replace=False).mean()
                         for _ in range(N_PERMUTATIONS)])
        p = (np.sum(np.abs(null - r_rand.mean()) >= abs(r_snd.mean() - r_rand.mean())) + 1) \
            / (N_PERMUTATIONS + 1)

        # 2 and 3. the raw channels, detrended but not corrected by each other
        d = tdt.read_block(os.path.join(args.root, fa.TANK_FOLDER, s["tank"]),
                           store=[fa.STREAM_465.lstrip("_"), fa.STREAM_405.lstrip("_")])
        rate = d.streams[fa.STREAM_465].fs
        g = d.streams[fa.STREAM_465].data.astype(float)
        r = d.streams[fa.STREAM_405].data.astype(float)
        n = min(len(g), len(r))
        tr = np.arange(n) / rate
        span = (bl["start_tdt"].min(), bl["end_tdt"].max())
        chans = {}
        for name, x in (("465", g[:n]), ("405", r[:n])):
            x = savgol_filter(x, fa.SMOOTH_WIN, 3)
            x = x / fa.bleaching_curve(x, tr, span)[0] - 1
            chans[name] = x[::fa.DOWNSAMPLE]
        td = tr[::fa.DOWNSAMPLE]
        rs, slopes = [], []
        for _, b in sb.iterrows():
            m = (td >= b["start_tdt"]) & (td < b["end_tdt"])
            rs.append(np.corrcoef(chans["465"][m], chans["405"][m])[0, 1])
            slopes.append(np.polyfit(chans["405"][m], chans["465"][m], 1)[0])

        rows.append({"day": s["day"], "mouse": s["mouse"],
                     "block_sd_dff": np.nanmean([np.nanstd(dff[(t >= b.start_tdt) & (t < b.end_tdt)])
                                                 for _, b in sb.iterrows()]),
                     "sound_entries": len(r_snd), "sound_response_z": r_snd.mean(),
                     "silent_arm_entries": len(r_sil),
                     "silent_arm_response_z": r_sil.mean() if len(r_sil) else np.nan,
                     "random_response_z": r_rand.mean(), "random_response_sd_z": r_rand.std(),
                     "sound_vs_random_p": p,
                     "r_465_405": np.mean(rs), "fit_slope_465_on_405": np.mean(slopes)})

        # figures
        col = DAYS.index(s["day"])
        ax = ax1[mice.index(s["mouse"]), col]
        for name, times, colour in (("sound", snd, "#1565C0"), ("silent arm", sil, "#757575"),
                                    ("random", rand_t[:400], "#E65100")):
            tr_, tt = traces(z, fs, times)
            if len(tr_) < 3:
                continue
            mean, sem = tr_.mean(0), tr_.std(0) / np.sqrt(len(tr_))
            ax.plot(tt, mean, color=colour, lw=1.1, label=f"{name} (n={len(tr_)})")
            ax.fill_between(tt, mean - sem, mean + sem, color=colour, alpha=0.15, lw=0)
        ax = ax2[mice.index(s["mouse"]), col]
        fsd = rate / fa.DOWNSAMPLE
        for name, colour in (("465", "#2E7D32"), ("405", "#6A1B9A")):
            zc = zscore_blocks(chans[name], td, bl, sound_only=True)
            tr_, tt = traces(zc, fsd, snd)
            mean, sem = tr_.mean(0), tr_.std(0) / np.sqrt(len(tr_))
            ax.plot(tt, mean, color=colour, lw=1.1, label=f"{name} nm alone (n={len(tr_)})")
            ax.fill_between(tt, mean - sem, mean + sem, color=colour, alpha=0.15, lw=0)
            chan_rows.append({"day": s["day"], "mouse": s["mouse"], "channel": name,
                              "z_8_to_5_s_before": mean[(tt >= -8) & (tt < -5)].mean(),
                              "z_at_entry": mean[(tt >= -1) & (tt < 1)].mean(),
                              "z_6_to_10_s_after": mean[(tt >= 6) & (tt < 10)].mean()})
        for a in (ax1[mice.index(s["mouse"]), col], ax):
            a.axvline(0, color="k", ls="--", lw=0.7)
            a.axhline(0, color="grey", lw=0.5)
            a.set_title(f"{s['mouse']}, {s['day'].replace('_', ' ')}", fontsize=9)
            a.legend(fontsize=6, frameon=False)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

    for fig, axes, ylabel, title, name in (
            (fig1, ax1, "ΔF/F (z, within block)",
             "Arm entries against random moments in the same sound blocks, in z units",
             "fig_entry_vs_random"),
            (fig2, ax2, "Channel level (z, within block)",
             "Each channel around sound entries, detrended for bleaching but not "
             "corrected by the other", "fig_channels_at_entry")):
        for a in axes[-1]:
            a.set_xlabel("Time from arm entry (s)")
        for a in axes[:, 0]:
            a.set_ylabel(ylabel)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=170)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
    chan = pd.DataFrame(chan_rows)
    chan.to_csv(os.path.join(out, "channels_at_entry.csv"), index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print()
    print(chan.pivot_table(index=["mouse", "day"], columns="channel",
                           values="z_at_entry").round(2).to_string())
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
