#!/usr/bin/env python3
"""
Experiment 2: the Value Model, With Place Controlled
====================================================

Two analyses of the model of Chapter 5 on the grammar test days (day 1). The
earlier model analysis (model_validation/ and its output) is used read-only and
left as it was.

1. Can place alone produce the earlier positive semantic weight? The earlier
   model scores each arm-block from the melodies of its (group, environment,
   level) cell, so the arm a cell sat in never enters it; in block 1 the EE
   grammar always sat in arms 1 to 3, two of them favoured. Synthetic mice that
   care only for arms are built from each real mouse's own arm preference (its
   share of the time in each arm over the shuffled blocks 2 to 4, where the
   sounds moved), laid onto the real layouts of every block, and fitted with the
   earlier model unchanged (its features, tier-restricted emissions, and its
   least-squares fit, model_validation/recovery.py). A positive semantic weight
   from these mice over all four blocks, and none over blocks 2 to 4, would mean
   the earlier weight could be place.

2. The ladder of Chapter 5, fitted to the real data over the shuffled blocks 2
   to 4 (primary) and over all four blocks:
     original   one context, Gaussian state over tone frequency (fluency and
                learning as in Brielmann and Dayan 2022)
     sequential one context, transitions under one grammar learned from both
                (the mean of the two matrices); learning on the transition
                estimate
     full       two contexts, each with its own grammar (the full matrices the
                mice heard in training), belief by the bigram filter; fluency,
                learning on each context's transition estimate, and semantic
                value
     full, tier-restricted emissions (the earlier analysis's choice), for
                comparison
   Every model has an offset for the silent arm, so that fluency does not have
   to carry silence against sound; the vocalisation is held out, as before.
   Each arm's predicted share of a block's time is a softmax of its value,
   fitted by least squares on the observed shares. Models are compared by
   leave-one-mouse-out cross-validated error, against a model with the silent
   offset alone, and each weight gets a 95% interval from 500 bootstrap
   resamples of mice.

Learning on transitions: the listener's estimate Q of each row of the
transition matrix moves towards each observed transition at rate alpha,
Q_i <- Q_i + alpha * b * (e_o - Q_i) (b = 1 with one context, the belief with
two), and the state's value is minus the mean over rows of KL(Q_i || T_i), T
the transitions of the listener's environment; Delta V is its change. Each
melody is scored from the trained state (Q = T, belief uniform), and features
are means over the melodies of each cell (as in the earlier analysis).

Usage:
    python exp2_model_check.py [GRAMMAR_DIR] [--boot 500] [--sims 200]

Outputs (GRAMMAR_DIR/BATCH_ANALYSIS/exp2_model/):
    cell_features.csv, ladder.csv, place_only.csv, stats.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_validation.config import validated_config, symbol_to_index
from model_validation.data_loading import load_arm_blocks, load_day
from model_validation import latent_regressors as lr
from model_validation import recovery as rec

POOL_CAP = 3000
TIERS = ["dominant", "secondary", "rare"]


# -- features -----------------------------------------------------------

def gaussian_features(seq, cfg):
    """Brielmann-Dayan, one context, Gaussian state over log2 tone frequency."""
    o = cfg.tone_logfreq[seq]
    mu, s2, muT, s2T, a = cfg.p_T_mu, cfg.sigma2, cfg.p_T_mu, cfg.p_T_sigma2, cfg.alpha

    def value(m):
        return -(np.log(np.sqrt(s2T / s2)) + (s2 + (m - muT) ** 2) / (2 * s2T) - 0.5)

    r, dv = [], []
    v_prev = value(mu)
    for x in o:
        r.append(-0.5 * (x - mu) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2))
        mu = mu + a * (x - mu)
        v = value(mu)
        dv.append(v - v_prev)
        v_prev = v
    return np.mean(r), np.mean(dv), 0.0


def kl_rows(Q, T):
    return np.mean(np.sum(Q * np.log(Q / T), axis=1))


def sequential_features(seq, T, cfg):
    """One context: transitions under a single learned grammar T."""
    Q = T.copy()
    r, dv = [], []
    v_prev = -kl_rows(Q, T)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        r.append(np.log(T[a, b]))
        e = np.zeros(T.shape[1]); e[b] = 1.0
        Q[a] = Q[a] + cfg.alpha * (e - Q[a])
        v = -kl_rows(Q, T)
        dv.append(v - v_prev)
        v_prev = v
    return np.mean(r), np.mean(dv), 0.0


def full_features(seq, M_EE, M_SC, cfg, T_EE=None, T_SC=None):
    """Two contexts: bigram filter (fluency, semantic value) and belief-weighted
    learning on each context's transition estimate."""
    T_EE = M_EE if T_EE is None else T_EE
    T_SC = M_SC if T_SC is None else T_SC
    tr = lr.hmm_filter(seq, M_EE, M_SC, cfg)
    Q = [T_EE.copy(), T_SC.copy()]
    T = [T_EE, T_SC]

    def value(b):
        return -sum(b[k] * kl_rows(Q[k], T[k]) for k in range(2))

    dv = []
    v_prev = value(cfg.b0)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        bk = tr.b_post[i]
        for k in range(2):
            e = np.zeros(6); e[b] = 1.0
            Q[k][a] = Q[k][a] + cfg.alpha * bk[k] * (e - Q[k][a])
        v = value(bk)
        dv.append(v - v_prev)
        v_prev = v
    return np.mean(tr.r_hmm), np.mean(dv), np.mean(tr.S)


def cell_features(day1, cfg):
    s2i = symbol_to_index(cfg)
    pooled_T = (cfg.grammar_A + cfg.grammar_B) / 2
    rows = []
    gram = day1[day1["arm_type"] == "grammar"]
    for (group, env, tier), sub in gram.groupby(["group", "environment", "tier"]):
        mels = [m for ms in sub["melodies"] for m in ms]
        if len(mels) > POOL_CAP:
            idx = np.linspace(0, len(mels) - 1, POOL_CAP).astype(int)
            mels = [mels[i] for i in idx]
        seqs = [lr.parse_symbols(m, s2i) for m in mels]
        M_EE, M_SC = cfg.emission_matrices(int(group))
        R_EE = lr.tier_restricted_emissions(M_EE, tier, cfg)
        R_SC = lr.tier_restricted_emissions(M_SC, tier, cfg)
        feats = {
            "original": np.mean([gaussian_features(s, cfg) for s in seqs], axis=0),
            "sequential": np.mean([sequential_features(s, pooled_T, cfg) for s in seqs], axis=0),
            "full": np.mean([full_features(s, M_EE, M_SC, cfg) for s in seqs], axis=0),
            "full_tier": np.mean([full_features(s, R_EE, R_SC, cfg, M_EE, M_SC) for s in seqs], axis=0),
        }
        row = {"group": group, "environment": env, "tier": tier, "n_melodies": len(seqs)}
        for name, (r, dv, s) in feats.items():
            row.update({f"{name}_r": r, f"{name}_dV": dv, f"{name}_S": s})
        rows.append(row)
    return pd.DataFrame(rows)


# -- design and fit -------------------------------------------------------

def blocks_for(day1, cells, model, which):
    """Per (mouse, block): feature rows of the grammar and silent arms, and observed shares."""
    use = day1[day1["arm_type"].isin(["grammar", "silent"])]
    if which == "shuffled":
        use = use[use["block"] > 1]
    use = use.merge(cells, on=["group", "environment", "tier"], how="left")
    g = use[use["arm_type"] == "grammar"]
    scale = {f: (g[f"{model}_{f}"].std(ddof=1) or 1.0) for f in ("r", "dV", "S")}
    scale = {k: (v if v > 1e-12 else 1.0) for k, v in scale.items()}
    out = []
    for (mouse, blk), sub in use.groupby(["mouse", "block"]):
        tot = sub["time_spent_s"].sum()
        if tot <= 0 or len(sub) < 2:
            continue
        X = np.column_stack([
            (sub["arm_type"] == "silent").astype(float).to_numpy(),
            np.nan_to_num(sub[f"{model}_r"].to_numpy()) / scale["r"],
            np.nan_to_num(sub[f"{model}_dV"].to_numpy()) / scale["dV"],
            np.nan_to_num(sub[f"{model}_S"].to_numpy()) / scale["S"],
        ])
        out.append({"mouse": mouse, "X": X, "y": (sub["time_spent_s"] / tot).to_numpy(),
                    "arm": sub["roi"].astype(str).str.extract(r"(\d)")[0].astype(int).to_numpy()})
    return out


PARAMS = {"silent offset only": [0], "original": [0, 1, 2], "sequential": [0, 1, 2],
          "full": [0, 1, 2, 3], "full_tier": [0, 1, 2, 3]}
NAMES = ["w_silent", "wr", "wV", "wS"]


def sse(w, blocks, cols):
    tot = 0.0
    for b in blocks:
        a = b["X"][:, cols] @ w
        p = np.exp(a - a.max()); p /= p.sum()
        tot += np.sum((b["y"] - p) ** 2)
    return tot


def fit(blocks, cols):
    res = minimize(sse, np.zeros(len(cols)), args=(blocks, cols), method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x


def lomo(blocks, cols):
    mice = sorted({b["mouse"] for b in blocks})
    err = 0.0
    for m in mice:
        train = [b for b in blocks if b["mouse"] != m]
        test = [b for b in blocks if b["mouse"] == m]
        err += sse(fit(train, cols), test, cols)
    return err


def bootstrap(blocks, cols, n, rng):
    mice = sorted({b["mouse"] for b in blocks})
    by = {m: [b for b in blocks if b["mouse"] == m] for m in mice}
    est = []
    for _ in range(n):
        pick = rng.choice(mice, len(mice), replace=True)
        est.append(fit([b for m in pick for b in by[m]], cols))
    return np.array(est)


# -- place-only mice --------------------------------------------------------

def place_preferences(day1):
    """Each mouse's share of time per arm over the shuffled blocks (sounds moving)."""
    x = day1[(day1["block"] > 1) & day1["arm_type"].isin(["grammar", "silent", "vocalisation"])].copy()
    x["arm"] = x["roi"].astype(str).str.extract(r"(\d)")[0].astype(int)
    t = x.groupby(["mouse", "arm"])["time_spent_s"].sum().unstack(fill_value=0)
    return (t.T / t.sum(axis=1)).T


def place_only_designs(design, feat_df, prefs, kappa, rng):
    """The earlier design with each block's shares replaced by place alone."""
    use = feat_df[feat_df["arm_type"].isin(["grammar", "silent"])]
    groups = [(m, b, sub) for (m, b), sub in use.groupby(["mouse", "block"])
              if sub["time_spent_s"].sum() > 0 and len(sub) >= 2]
    assert len(groups) == len(design.blocks)
    new = []
    for blk, (mouse, b, sub) in zip(design.blocks, groups):
        arms = sub["roi"].astype(str).str.extract(r"(\d)")[0].astype(int).to_numpy()
        p = prefs.loc[mouse, arms].to_numpy() + 1e-6
        p = p / p.sum()
        y = p if kappa is None else rng.dirichlet(np.clip(kappa * p, 1e-3, None))
        new.append(rec.Block(blk.r, blk.dV, blk.S, blk.g, y, blk.mouse))
    return rec.Design(blocks=new, scales=design.scales), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()
    out = os.path.join(args.root, "BATCH_ANALYSIS", "exp2_model")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    cfg = validated_config()

    lines = []

    def say(text=""):
        lines.append(text)
        print(text, flush=True)

    data = load_arm_blocks(args.root).arm_blocks
    day1 = load_day(data, "primary")
    day1 = day1[day1.groupby("mouse")["time_spent_s"].transform("sum") > 0]   # 13672 never entered
    say(f"day 1: {day1['mouse'].nunique()} mice")

    # -- 1. place-only mice under the earlier model
    say("\n== 1. The earlier model on mice that care only for place ==")
    feat = lr.build_features(day1, cfg, emission="tier_restricted")
    prefs = place_preferences(day1)
    rows = []
    for which, label in (("all", "all four blocks"), ("shuffled", "blocks 2 to 4")):
        f = feat if which == "all" else feat[feat["block"] > 1]
        design = rec.build_design(f)
        w_real = rec.fit(design.blocks, "full")
        clean, _ = place_only_designs(design, f, prefs, None, rng)
        w_clean = rec.fit(clean.blocks, "full")
        sims = []
        for _ in range(args.sims):
            noisy, _ = place_only_designs(design, f, prefs, 30.0, rng)
            sims.append(rec.fit(noisy.blocks, "full")["wS"])
        sims = np.array(sims)
        say(f"  {label}: real mice wS = {w_real['wS']:+.3f}; place-only mice wS = {w_clean['wS']:+.3f} "
            f"without noise, {np.mean(sims):+.3f} [{np.percentile(sims, 2.5):+.3f}, "
            f"{np.percentile(sims, 97.5):+.3f}] over {args.sims} noisy sets, "
            f"{np.mean(sims > 0):.0%} above 0")
        rows.append({"blocks": which, "wS_real": w_real["wS"], "wS_place_clean": w_clean["wS"],
                     "wS_place_mean": sims.mean(), "wS_place_lo": np.percentile(sims, 2.5),
                     "wS_place_hi": np.percentile(sims, 97.5)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "place_only.csv"), index=False)

    # -- 2. the ladder
    say("\n== 2. The ladder of Chapter 5 (silent offset in every model; vocalisation held out) ==")
    cells = cell_features(day1, cfg)
    cells.to_csv(os.path.join(out, "cell_features.csv"), index=False)
    say("cell features (group, environment, level):")
    show = cells[["group", "environment", "tier"] + [c for c in cells.columns if c.endswith(("_r", "_dV", "_S"))]]
    say(show.round(4).to_string(index=False))
    for name in ("original", "sequential", "full", "full_tier"):
        sub = cells[[f"{name}_r", f"{name}_dV", f"{name}_S"]]
        say(f"  {name}: between-cell sd r {sub.iloc[:, 0].std():.4f}, dV {sub.iloc[:, 1].std():.2e}, "
            f"S {sub.iloc[:, 2].std():.4f}; corr(r, S) "
            f"{np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 2])[0, 1] if sub.iloc[:, 2].std() > 0 else float('nan'):+.2f}")

    rows = []
    for which, label in (("shuffled", "blocks 2 to 4"), ("all", "all four blocks")):
        say(f"\n  {label}:")
        base = None
        for model, cols in PARAMS.items():
            feats = "full" if model == "silent offset only" else model
            blocks = blocks_for(day1, cells, feats, which)
            w = fit(blocks, cols)
            cv = lomo(blocks, cols)
            base = cv if base is None else base
            boot = bootstrap(blocks, cols, args.boot, rng)
            ci = {NAMES[c]: (np.percentile(boot[:, j], 2.5), np.percentile(boot[:, j], 97.5))
                  for j, c in enumerate(cols)}
            say(f"    {model:19s} CV error {cv:8.4f} ({cv - base:+.4f} against the silent offset alone); "
                + "; ".join(f"{NAMES[c]} {w[j]:+.3f} [{ci[NAMES[c]][0]:+.3f}, {ci[NAMES[c]][1]:+.3f}]"
                            for j, c in enumerate(cols)))
            row = {"blocks": which, "model": model, "cv_error": cv, "n_blocks": len(blocks)}
            for j, c in enumerate(cols):
                row.update({NAMES[c]: w[j], f"{NAMES[c]}_lo": ci[NAMES[c]][0], f"{NAMES[c]}_hi": ci[NAMES[c]][1]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "ladder.csv"), index=False)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
