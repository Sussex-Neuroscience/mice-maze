#!/usr/bin/env python3
"""
Experiment 1: Mixed Models of Visit Length (lme4)
=================================================

Builds one row per arm, sound block and session of Experiment 1: the mean
visit length in that arm in that block, the number of visits it rests on,
what the arm played (class, family and stimulus, as exp1_stimulus_pi.py names
them), the stimulus set, the block (1 to 4) and the mouse. Visits are
uncapped (see exp1_vocalisation_pi.py).

It then runs exp1_lmm.R, which fits linear mixed models of the log mean visit
length with lme4: class against class (silent arm, synthetic sounds,
vocalisation) by stimulus set, consonant against dissonant intervals,
smooth against rough envelopes, and the modulated against the plain unison.
A contrast is a ratio of visit lengths, and tanh(log ratio / 2) is the
preference index it corresponds to.

R with lme4, lmerTest and emmeans is needed; Rscript is looked for on the
PATH, then in the newest R under C:/Program Files/R.

With --shuffled-only every model is fitted to the shuffled blocks 2 to 4
alone (the first sound block of a session always opened on the same layout),
and the results go to lmm_shuffled/; the table is still written in full.

Usage:
    python exp1_lmm.py [DATA_DIR] [--shuffled-only]

Outputs (DATA_DIR/BATCH_ANALYSIS/lmm/, results in lmm_shuffled/ with --shuffled-only):
    arm_blocks.csv       the table the models are fitted to
    lmm_results.txt      every model, its random effects and its contrasts
    lmm_contrasts.csv    the contrasts in one table
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp1_stimulus_pi import SETS, stimulus

ROIS = {f"ROI{i}" for i in range(1, 9)}
SOUND_TRIALS = [2, 4, 6, 8]


def rscript():
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    if not candidates:
        sys.exit("Rscript not found: install R with lme4, lmerTest and emmeans")
    return candidates[-1]


def build_table(cfg, classify):
    modes = {d: cfg.EXPERIMENT_DAYS[d]["mode"] for d in cfg.EXPERIMENT_DAYS}
    rows = []
    for sess in cfg.discover_sessions():
        df, _ = cfg.load_session_visits(sess)
        if df is None:
            continue
        s = df[df["trial_ID"].isin(SOUND_TRIALS) & df["ROIs"].isin(ROIS)]
        for _, r in s.iterrows():
            name, family = stimulus(r, sess.day, classify, modes)
            cls = ("silent" if family == "silent" else
                   "vocalisation" if family == "vocalisation" else
                   "other" if family in ("unknown", "control") else "synthetic")
            visits = r["visitation_count"]
            rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day],
                         "session": os.path.basename(sess.session_folder),
                         "block": SOUND_TRIALS.index(int(r["trial_ID"])) + 1,
                         "arm": r["ROIs"], "cls": cls, "family": family, "stimulus": name,
                         "visits": visits, "time_s": r["time_spent"] / 1000,
                         "mean_s": r["time_spent"] / 1000 / visits if visits > 0 else None})
    return pd.DataFrame(rows)


def main():
    import preference_analysis_config as cfg
    from run_batch_preference import classify_stimulus
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--shuffled-only", action="store_true",
                        help="fit the models to the shuffled blocks 2 to 4 alone")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    cfg.INDIVIDUAL_VISIT_CLIP_MS = 0
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "lmm")
    os.makedirs(out, exist_ok=True)

    table = build_table(cfg, classify_stimulus)
    path = os.path.join(out, "arm_blocks.csv")
    table.to_csv(path, index=False)
    print(f"{len(table)} arm-blocks, {int((table['visits'] > 0).sum())} with visits")
    results = out
    if args.shuffled_only:
        results = os.path.join(args.data_dir, "BATCH_ANALYSIS", "lmm_shuffled")
        os.makedirs(results, exist_ok=True)
    r = subprocess.run([rscript(), os.path.join(HERE, "exp1_lmm.R"), path, results]
                       + (["shuffled"] if args.shuffled_only else []))
    if r.returncode:
        sys.exit(r.returncode)
    print(f"\nAll outputs saved to:\n  {results}")


if __name__ == "__main__":
    main()
