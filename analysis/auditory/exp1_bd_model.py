#!/usr/bin/env python3
"""
Experiment 1: the Model of Brielmann and Dayan, Fitted
======================================================

Does the model of Brielmann and Dayan (2022), with the stimuli described only by
their measured acoustics, account for how the mice of Experiment 1 treated each
stimulus, and does it single out the vocalisation?

The listener's system state is a point mu in a space of two features, spectral
and temporal complexity (stimulus_complexity.py), standardised over the stimuli
of the experiment; its covariance is fixed and equal in every direction, so it
drops out of every comparison and is absorbed into the weights. Each exposure to
a stimulus s gives

    fluency   r(s)  = -|s - mu|^2
    learning  dV(s) = -(|mu' - mu_T|^2 - |mu - mu_T|^2),  mu' = mu + alpha (s - mu)
    value     A(s)  = w_r r(s) + w_V dV(s)

and then moves the state to mu'. mu_T is the mean of the environment the listener
expects in the long run. The state starts there, so that before the experiment
the listener's model matches its world. The days run in the order the mice heard
them (temporal envelope modulation, the three interval sets, the sequences, the
vocalisations), and within a block each stimulus is met as often as the mice
visited its arm, on average over the sessions of that set.

A recording never repeats within a visit (playback restarted at every entry), so
its temporal complexity is log2 of its length in cycles of 2 kHz, the floor of
the band its spectral complexity is measured over; the vocalisation arm of days
1 to 5 takes the mean of the three recordings it played (Table 4.2). The silent
arm has no features: the index of a stimulus is modelled as A(s) - c, with c the
value of the silent arm, constant over days.

The data are the mean index of every stimulus against the silent arm on every
day (stimuli heard by at least five mice, as in Figure 4.10). Nested models are
fitted by least squares from many starting points:

    intercept   c                                   1 parameter
    voc only    c, w_voc                            2
    fluency     c, w_r, alpha, mu_T (2)             5
    B and D     c, w_r, w_V, alpha, mu_T (2)        6
    B and D + voc   the same and w_voc              7
    B and D, centred   B and D with mu_T fixed at the centre of the stimuli,
                    0 on both standardised features   4

where w_voc is an extra value for a vocalisation that its acoustics do not give.
They are compared by AIC and by leaving out one day at a time: each model is
fitted to the other four days and predicts the left out one.

Usage:
    python exp1_bd_model.py [DATA_DIR] [--thesis-figures DIR]

Reads stimulus_pi/stimulus_pi.csv, complexity/complexity.csv,
complexity/vocalisation_complexity.csv and lmm/arm_blocks.csv from
DATA_DIR/BATCH_ANALYSIS, and writes to DATA_DIR/BATCH_ANALYSIS/bd_model/.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimulus_complexity as sc
from exp1_stimulus_pi import COLOURS, label

DAYS = ["TEM", "Intervals 1", "Intervals 2", "Intervals 3", "Sequences", "Vocalisations"]
OBSERVED = ["TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"]
BLOCKS = 4
VOC_DAYS_1_TO_5 = ["run1_day2_male_w_female_oestrus", "run3_day2_male_w_female_oestrus",
                   "run3_day2-2_male_w_female_oestrus"]
FULL_RECORDING_S = 300.0   # run1 day2 was played in full, five minutes, until January 2026
FLOOR_HZ = 2000.0
N_STARTS = 40
CV_STARTS = 12
SEED = 1


# -- the stimuli and the data ---------------------------------------------------
def features(base):
    """Spectral and temporal complexity of every stimulus, keyed by (set, stimulus)."""
    rows = []
    arms = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    voc = pd.read_csv(os.path.join(base, "complexity", "vocalisation_complexity.csv")).set_index("stimulus")
    voc_bits = np.log2(voc["seconds"] * FLOOR_HZ)
    played = voc.loc[VOC_DAYS_1_TO_5].copy()
    played_bits = voc_bits.loc[VOC_DAYS_1_TO_5].copy()
    played_bits.loc["run1_day2_male_w_female_oestrus"] = np.log2(FULL_RECORDING_S * FLOOR_HZ)
    for (set_name, stim, fam), _ in arms[arms["family"] != "silent"].groupby(["set", "stimulus", "family"]):
        if fam == "vocalisation" and set_name != "Vocalisations":
            spec, bits = played["spectral_audible"].mean(), played_bits.mean()
        elif set_name == "Vocalisations":
            spec, bits = voc.loc[stim, "spectral_audible"], voc_bits.loc[stim]
        else:
            d = sc.describe_exp1(stim, fam)
            spec, bits = d["spectral"], d["temporal_bits"]
        rows.append({"set": set_name, "stimulus": stim, "family": fam,
                     "spectral": spec, "temporal_bits": bits,
                     "voc": float(fam == "vocalisation" or set_name == "Vocalisations")})
    f = pd.DataFrame(rows).dropna(subset=["spectral", "temporal_bits"])
    # standardised over the stimuli of the experiment
    for col in ("spectral", "temporal_bits"):
        f[col + "_z"] = (f[col] - f[col].mean()) / f[col].std(ddof=0)
    return f


def exposures(base, f):
    """Visits to each stimulus in a block, on average over the sessions of its set."""
    arms = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    sessions = arms.groupby("set")["session"].nunique()
    v = arms[arms["family"] != "silent"].groupby(["set", "stimulus"])["visits"].sum()
    out = {}
    for s in DAYS:
        if s not in sessions:
            continue
        per = (v.loc[s] / sessions[s] / BLOCKS).round().clip(lower=1).astype(int)
        out[s] = per
    return out


def observations(base, f):
    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    st = st[st["set"].isin(OBSERVED)]
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    obs = st.groupby(["set", "stimulus"], as_index=False).agg(
        pi=("pi", "mean"), se=("pi", lambda x: x.std(ddof=1) / np.sqrt(len(x))), n=("pi", "size"))
    return obs.merge(f, on=["set", "stimulus"])


# -- the model ------------------------------------------------------------------
MODELS = {
    "intercept": ["c"],
    "voc only": ["c", "w_voc"],
    "fluency": ["c", "w_r", "alpha", "muT_s", "muT_t"],
    "B and D": ["c", "w_r", "w_V", "alpha", "muT_s", "muT_t"],
    "B and D, centred": ["c", "w_r", "w_V", "alpha"],
    "B and D + voc": ["c", "w_r", "w_V", "alpha", "muT_s", "muT_t", "w_voc"],
}
NAMES = {"intercept": "intercept only", "voc only": "vocalisation only", "fluency": "fluency",
         "B and D": "Brielmann and Dayan", "B and D, centred": "B and D, world at centre",
         "B and D + voc": "B and D and vocalisation"}
BOUNDS = {"c": (-1.0, 1.0), "w_r": (0.0, 5.0), "w_V": (0.0, 200.0), "alpha": (1e-4, 0.3),
          "muT_s": (-3.0, 6.0), "muT_t": (-3.0, 6.0), "w_voc": (-1.0, 1.0)}


def plan(f, expo):
    """The order of the exposures within a block of each day, built once: each
    stimulus met as often as the mice visited it, its visits spread through the
    block in turn with the others."""
    vec = {(r.set, r.stimulus): (r.spectral_z, r.temporal_bits_z) for r in f.itertuples()}
    out = []
    for day in DAYS:
        if day not in expo:
            continue
        counts = {s: int(n) for s, n in expo[day].items() if (day, s) in vec}
        order = [s for k in range(max(counts.values())) for s in sorted(counts) if counts[s] > k]
        out.append((day, [(s,) + vec[(day, s)] for s in order]))
    return out


def simulate(p, steps, record=False):
    """Mean value of each stimulus on each day, and the path of the state."""
    ts, tt = p.get("muT_s", 0.0), p.get("muT_t", 0.0)
    ms, mt = ts, tt
    alpha, w_r, w_V = p.get("alpha", 0.0), p.get("w_r", 0.0), p.get("w_V", 0.0)
    values, path = {}, [(ms, mt)]
    for day, order in steps:
        acc = {}
        for _ in range(BLOCKS):
            for stim, ss, st in order:
                ns, nt = ms + alpha * (ss - ms), mt + alpha * (st - mt)
                r = -((ss - ms) ** 2 + (st - mt) ** 2)
                dV = -(((ns - ts) ** 2 + (nt - tt) ** 2) - ((ms - ts) ** 2 + (mt - tt) ** 2))
                acc.setdefault(stim, []).append(w_r * r + w_V * dV)
                ms, mt = ns, nt
            path.append((ms, mt))
        for stim, a in acc.items():
            values[(day, stim)] = sum(a) / len(a)
    return (values, np.array(path)) if record else values


def predict(p, obs, steps):
    values = simulate(p, steps) if any(k in p for k in ("w_r", "w_V")) else {}
    out = []
    for r in obs.itertuples():
        a = values.get((r.set, r.stimulus), 0.0)
        out.append(a - p["c"] + p.get("w_voc", 0.0) * r.voc)
    return np.array(out)


def fit(model, obs, steps, rng, starts=None):
    names = MODELS[model]
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])

    def resid(x):
        return predict(dict(zip(names, x)), obs, steps) - obs["pi"].to_numpy()

    best = None
    starts = 1 if model in ("intercept", "voc only") else (starts or N_STARTS)
    for k in range(starts):
        x0 = lo + rng.uniform(0.05, 0.95, len(names)) * (hi - lo)
        if names[0] == "c":
            x0[0] = 0.0
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return dict(zip(names, best.x)), 2 * best.cost


def aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


# -- figure -----------------------------------------------------------------------
def fig_fit(obs, preds, path_traj, f, comparison, path):
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.6, wspace=0.45)
    order = obs.assign(o=obs["set"].map(OBSERVED.index)).sort_values(["o", "voc", "stimulus"])
    x = np.arange(len(order))
    ax = fig.add_subplot(gs[0, :])
    ax.errorbar(x, order["pi"], yerr=order["se"], fmt="o", ms=3, color="black", lw=0.8,
                capsize=1.5, label="mice, mean and standard error", zorder=3)
    for name, colour, ls in (("B and D", "#1565C0", "-"), ("B and D + voc", "#2E7D32", (0, (4, 2)))):
        ax.plot(x, preds[name].loc[order.index], color=colour, lw=1.1, ls=ls, marker="D", ms=2.5,
                label=f"{name.replace('B and D', 'Brielmann and Dayan')}")
    for k, s in enumerate(OBSERVED):
        idx = np.where(order["set"].to_numpy() == s)[0]
        if k:
            ax.axvline(idx[0] - 0.5, color="#BDBDBD", lw=0.6)
        ax.text(idx.mean(), 0.33, s, ha="center", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels([label(s) for s in order["stimulus"]], rotation=70, ha="right", fontsize=5)
    for t, v in zip(ax.get_xticklabels(), order["voc"]):
        if v:
            t.set_color("#2E7D32")
    ax.set_ylabel("Index against the silent arm")
    ax.set_ylim(-0.4, 0.38)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right", ncol=3)
    ax.set_title("(a) every stimulus, observed and fitted", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    names = list(comparison.index)
    ax.barh(range(len(names)), comparison["cv_rmse"], color="#90A4AE", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([NAMES.get(n, n) for n in names], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Error predicting a\nleft out day (RMSE)")
    ax.set_title("(b) prediction", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.barh(range(len(names)), comparison["aic"] - comparison["aic"].min(), color="#B0BEC5", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("AIC above the best")
    ax.set_title("(c) fit", loc="left")

    ax = fig.add_subplot(gs[1, 2])
    for r in f.itertuples():
        colour = "#2E7D32" if r.voc else COLOURS.get(r.family, "#424242")
        ax.scatter(r.temporal_bits_z, r.spectral_z, s=8, color=colour, lw=0, alpha=0.8)
    ax.plot(path_traj[:, 1], path_traj[:, 0], color="black", lw=0.8)
    ax.scatter(path_traj[0, 1], path_traj[0, 0], marker="*", s=40, color="#E65100", zorder=4,
               label="the expected world, where the state starts")
    ax.scatter(path_traj[-1, 1], path_traj[-1, 0], marker="o", s=14, color="black", zorder=4,
               label="the state at the end")
    ax.set_xlabel("Temporal complexity (z)")
    ax.set_ylabel("Spectral complexity (z)")
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax.set_title("(d) the state, Brielmann and Dayan", loc="left")
    for a in fig.axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "bd_model")
    os.makedirs(out, exist_ok=True)

    f = features(base)
    expo = exposures(base, f)
    obs = observations(base, f)
    steps = plan(f, expo)
    rng = np.random.default_rng(SEED)
    lines = []

    def say(t=""):
        lines.append(t)
        print(t)

    say(f"{len(obs)} stimulus indices over {obs['set'].nunique()} days, "
        f"{int(obs['voc'].sum())} of them vocalisations")
    fits, preds, rows = {}, {}, []
    n = len(obs)
    for model in MODELS:
        p, rss = fit(model, obs, steps, rng)
        fits[model] = p
        preds[model] = pd.Series(predict(p, obs, steps), index=obs.index)
        # leave one day out
        cv = []
        for day in OBSERVED:
            train, test = obs[obs["set"] != day], obs[obs["set"] == day]
            pt, _ = fit(model, train, steps, rng, starts=CV_STARTS)
            cv += list(predict(pt, test, steps) - test["pi"].to_numpy())
        rows.append({"model": model, "k": len(MODELS[model]), "rss": rss,
                     "rmse": np.sqrt(rss / n), "aic": aic(rss, n, len(MODELS[model])),
                     "cv_rmse": np.sqrt(np.mean(np.square(cv)))})
        say(f"\n{model}: " + ", ".join(f"{k} = {v:.4g}" for k, v in p.items()))
        say(f"  RMSE {np.sqrt(rss / n):.4f}, AIC {aic(rss, n, len(MODELS[model])):.1f}, "
            f"left out day RMSE {rows[-1]['cv_rmse']:.4f}")
    comparison = pd.DataFrame(rows).set_index("model")
    say("\n" + comparison.round(4).to_string())

    # where the vocalisation sits for the fitted B and D model
    bd = fits["B and D"]
    say("\nB and D, fitted against observed, the vocalisation on each day:")
    for r in obs[obs["voc"] == 1].itertuples():
        say(f"  {r.set:13s} {label(r.stimulus):12s} observed {r.pi:+.3f}  "
            f"B and D {preds['B and D'][r.Index]:+.3f}  B and D + voc {preds['B and D + voc'][r.Index]:+.3f}")
    resid = obs.assign(res=obs["pi"] - preds["B and D"])
    say("\nB and D, mean residual: vocalisations "
        f"{resid.loc[resid['voc'] == 1, 'res'].mean():+.3f}, synthetic sounds "
        f"{resid.loc[resid['voc'] == 0, 'res'].mean():+.3f}")
    say("mean residual by day: " + ", ".join(
        f"{d} {resid.loc[resid['set'] == d, 'res'].mean():+.3f}" for d in OBSERVED))

    _, path_traj = simulate(bd, steps, record=True)
    obs.assign(**{f"pred_{k}": v for k, v in preds.items()}).to_csv(
        os.path.join(out, "predictions.csv"), index=False)
    comparison.to_csv(os.path.join(out, "comparison.csv"))
    pd.DataFrame(fits).T.to_csv(os.path.join(out, "parameters.csv"))
    f.to_csv(os.path.join(out, "features.csv"), index=False)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42}):
        targets = [out] + ([args.thesis_figures] if args.thesis_figures else [])
        for t in targets:
            fig_fit(obs, preds, path_traj, f, comparison, os.path.join(t, "fig_exp1_bd_model"))
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
