#!/usr/bin/env python3
"""
Does Fisher z Change Any Result? A Check
========================================

The models of visit length in this thesis are fitted on a log scale, the Fisher z
scale of the preference index (Section 2.6). A few analyses model the index
itself, as it is. This script refits each of them with the index replaced by its
Fisher z, artanh(PI), and sets the two answers side by side:

    Section 4.5   do the stimuli of a day differ? (mixed model, likelihood ratio)
    Section 4.8   the index against temporal and spectral complexity, straight and
                  squared terms with the set as a fixed effect; and within the
                  vocalisation day
    Chapter 3     the pilot's index against temporal complexity
    Section 4.9   the index of all sounds against exploratory drive
    Section 4.10  the indices block by block
    Section 4.6   the division of variance and the share between mice
                  (fisher_z_check.R)
    Chapter 5     the fit of the model of Brielmann and Dayan, with the mean z of
                  each stimulus in place of its mean index

No index in these tables is exactly -1 or +1, so every one has a finite z.

Usage:
    python fisher_z_check.py [DATA_DIR] [--survey DIR] [--skip-model]

Reads the tables the scripts named above write, and writes
DATA_DIR/BATCH_ANALYSIS/fisher_z_check/stats.txt and comparison.csv.
"""

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimulus_complexity as sc
from exp1_lmm import rscript

SCALES = {"index": lambda x: x, "Fisher z": np.arctanh}


def lrt(d, y, full, null, groups="mouse", dof=None):
    f = smf.mixedlm(f"{y} ~ {full}", d, groups=d[groups]).fit(reml=False)
    n = smf.mixedlm(f"{y} ~ {null}", d, groups=d[groups]).fit(reml=False)
    stat = 2 * (f.llf - n.llf)
    dof = dof or (len(f.params) - len(n.params))
    return chi2.sf(stat, dof), stat, dof


def check_stimuli(base, rows):
    """Section 4.5: per set, pi ~ stimulus against pi ~ 1, mouse as random intercept."""
    for name, folder in (("all blocks", "stimulus_pi"), ("blocks 2 to 4", "stimulus_pi_shuffled")):
        path = os.path.join(base, folder, "stimulus_pi.csv")
        if not os.path.exists(path):
            continue
        st = pd.read_csv(path).dropna(subset=["pi"])
        for s, d in st.groupby("set"):
            d = d[d.groupby("stimulus")["pi"].transform("size") >= 5]
            if d["stimulus"].nunique() < 2:
                continue
            for scale, f in SCALES.items():
                p, stat, dof = lrt(d.assign(y=f(d["pi"])), "y", "C(stimulus)", "1",
                                   dof=d["stimulus"].nunique() - 1)
                rows.append(("4.5", f"stimuli differ, {s}, {name}", scale, stat, p))


def check_complexity(base, rows):
    """Section 4.8, as exp1_complexity.py fits it."""
    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    comp = pd.read_csv(os.path.join(base, "complexity", "complexity.csv"))
    st = st[(st["family"] != "vocalisation") & st["set"].isin(comp["set"].unique())]
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    d = st.merge(comp, on=["set", "stimulus", "family"])
    for col, cname in (("temporal_bits", "temporal"), ("spectral", "spectral")):
        z = (d[col] - d[col].mean()) / d[col].std()
        x = d.assign(c=z, c2=z ** 2)
        for scale, f in SCALES.items():
            x = x.assign(y=f(x["pi"]))
            p1, s1, _ = lrt(x, "y", "C(set) + c", "C(set)")
            p2, s2, _ = lrt(x, "y", "C(set) + c + c2", "C(set) + c")
            rows.append(("4.8", f"{cname} complexity, straight term", scale, s1, p1))
            rows.append(("4.8", f"{cname} complexity, squared term", scale, s2, p2))
    # within the vocalisation day
    voc = pd.read_csv(os.path.join(base, "complexity", "vocalisation_complexity.csv"))
    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    v = st[st["set"] == "Vocalisations"].merge(voc, on="stimulus")
    for col, cname in (("spectral_audible", "spectral"), ("temporal", "temporal")):
        z = (v[col] - v[col].mean()) / v[col].std()
        x = v.assign(c=z, c2=z ** 2)
        for scale, f in SCALES.items():
            x = x.assign(y=f(x["pi"]))
            p1, s1, _ = lrt(x, "y", "c", "1")
            p2, s2, _ = lrt(x, "y", "c + c2", "c")
            rows.append(("4.8", f"vocalisation day, {cname}, straight term", scale, s1, p1))
            rows.append(("4.8", f"vocalisation day, {cname}, squared term", scale, s2, p2))


def check_pilot(survey, rows):
    """Chapter 3, as pilot_four_arm.py fits it (REML, Wald tests of the terms)."""
    idx = pd.read_csv(Path(survey) / "pilot" / "pilot_pi_stimuli.csv")
    d = idx[idx["stimulus"] != "all sounds"].copy()
    d["bits"] = d["stimulus"].map(sc.interval_periodicity_bits)
    for scale, f in SCALES.items():
        x = d.assign(y=f(d["pi"]))
        m1 = smf.mixedlm("y ~ bits", x, groups=x["mouse"]).fit(reml=True)
        m2 = smf.mixedlm("y ~ bits + I(bits ** 2)", x, groups=x["mouse"]).fit(reml=True)
        rows.append(("3", "pilot, temporal complexity, straight term", scale,
                     m1.params["bits"], m1.pvalues["bits"]))
        rows.append(("3", "pilot, temporal complexity, squared term", scale,
                     m2.params["I(bits ** 2)"], m2.pvalues["I(bits ** 2)"]))


def check_exploration(base, rows):
    """Section 4.9, as exp1_exploration.py fits it, for the index of all sounds."""
    t = pd.read_csv(os.path.join(base, "exploration", "exploration.csv"))
    from exp1_exploration import DRIVES
    for xcol, _ in DRIVES:
        d = t.dropna(subset=[xcol, "pi_all_sounds"])
        d = d.assign(x_between=d.groupby("mouse")[xcol].transform("mean"))
        d = d.assign(x_within=d[xcol] - d["x_between"])
        for scale, f in SCALES.items():
            x = d.assign(y=f(d["pi_all_sounds"]))
            m = smf.mixedlm("y ~ x_within + x_between", x, groups=x["mouse"]).fit()
            rows.append(("4.9", f"index of all sounds against {xcol}, within mice", scale,
                         m.params["x_within"], m.pvalues["x_within"]))


def check_blocks(base, rows):
    """Section 4.10, as exp1_order_place.py fits it: index ~ block, mouse intercept."""
    b = pd.read_csv(os.path.join(base, "order_place", "blocks.csv"))
    for col in ("synth_vs_silent", "voc_vs_synth", "voc_vs_silent"):
        for s, d in b.groupby("set"):
            d = d.dropna(subset=[col])
            if d["mouse"].nunique() < 8:
                continue
            for scale, f in SCALES.items():
                x = d.assign(y=f(d[col]))
                m = smf.mixedlm("y ~ block", x, groups=x["mouse"]).fit()
                rows.append(("4.10", f"{col} by block, {s}", scale, m.params["block"],
                             m.pvalues["block"]))


def check_variance(base):
    """Section 4.6, through fisher_z_check.R."""
    src = os.path.join(base, "individuals")
    out = {}
    for scale, f in SCALES.items():
        tmp = os.path.join(base, "fisher_z_check", "variance_" + ("z" if scale != "index" else "pi"))
        os.makedirs(tmp, exist_ok=True)
        sp = pd.read_csv(os.path.join(src, "session_pi.csv")).dropna(subset=["pi"])
        sp.assign(y=f(sp["pi"])).to_csv(os.path.join(tmp, "session_pi.csv"), index=False)
        cls = pd.read_csv(os.path.join(src, "session_cls.csv"))
        for col in ("voc_vs_silent", "synth_vs_silent", "voc_vs_synth"):
            cls[col] = f(cls[col])
        cls.to_csv(os.path.join(tmp, "session_cls.csv"), index=False)
        r = subprocess.run([rscript(), os.path.join(HERE, "fisher_z_check.R"), tmp],
                           capture_output=True, text=True)
        out[scale] = pd.read_csv(io.StringIO(r.stdout)).set_index("measure")["value"]
    return pd.DataFrame(out)


def check_model(base):
    """Chapter 5: the fit of exp1_bd_model.py with the mean z of each stimulus."""
    import exp1_bd_model as bd
    f = bd.features(base)
    expo = bd.exposures(base, f)
    steps = bd.plan(f, expo)
    obs = bd.observations(base, f)
    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    zmean = st.assign(z=np.arctanh(st["pi"])).groupby(["set", "stimulus"])["z"].mean()
    out = {}
    for scale in SCALES:
        o = obs.copy()
        if scale != "index":
            o["pi"] = [zmean[(s, n)] for s, n in zip(o["set"], o["stimulus"])]
        rng = np.random.default_rng(bd.SEED)
        res = {}
        for model in bd.MODELS:
            p, rss = bd.fit(model, o, steps, rng)
            cv = []
            for day in bd.OBSERVED:
                train, test = o[o["set"] != day], o[o["set"] == day]
                pt, _ = bd.fit(model, train, steps, rng, starts=bd.CV_STARTS)
                cv += list(bd.predict(pt, test, steps) - test["pi"].to_numpy())
            res[model] = {"aic": bd.aic(rss, len(o), len(bd.MODELS[model])),
                          "cv_rmse": float(np.sqrt(np.mean(np.square(cv))))}
        t = pd.DataFrame(res).T
        t["aic_above_best"] = t["aic"] - t["aic"].min()
        out[scale] = t
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--survey", default="D:/AuditoryExperiments_survey")
    parser.add_argument("--skip-model", action="store_true",
                        help="leave out the refit of the model of Brielmann and Dayan (slow)")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "fisher_z_check")
    os.makedirs(out, exist_ok=True)

    rows = []
    check_stimuli(base, rows)
    check_complexity(base, rows)
    check_pilot(args.survey, rows)
    check_exploration(base, rows)
    check_blocks(base, rows)
    t = pd.DataFrame(rows, columns=["section", "test", "scale", "statistic", "p"])
    wide = t.pivot_table(index=["section", "test"], columns="scale", values="p", sort=False)
    wide["same conclusion"] = (wide["index"] < 0.05) == (wide["Fisher z"] < 0.05)
    t.to_csv(os.path.join(out, "comparison.csv"), index=False)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("p values with the index as it is and with its Fisher z (significance at 0.05)")
    say(wide.round(4).to_string())
    say(f"\n{int(wide['same conclusion'].sum())} of {len(wide)} tests reach the same conclusion")
    say("\nSection 4.6, division of variance and share between mice, on each scale:")
    say(check_variance(base).round(3).to_string())
    if not args.skip_model:
        say("\nChapter 5, the model of Brielmann and Dayan fitted to the mean index and to the "
            "mean z of each stimulus:")
        for scale, tab in check_model(base).items():
            say(f"  {scale}")
            say(tab[["aic_above_best", "cv_rmse"]].round(4).to_string())
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
