import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import supfun_sequences as sf

def load_rois(rois_csv: Path):
    df = pd.read_csv(rois_csv, index_col=0)
    rois = {}
    for col in df.columns:
        x = int(df.loc["xstart", col])
        y = int(df.loc["ystart", col])
        w = int(df.loc["xlen", col])
        h = int(df.loc["ylen", col])
        rois[col] = {"xstart": x, "ystart": y, "xlen": w, "ylen": h}
    return rois

def grab_cut(frame, xstart, ystart, xlen, ylen):
    return frame[ystart:ystart+ylen, xstart:xstart+xlen]

def compute_thresholds(cap, rois, num_frames=10, thresh_value=160):
    thresholds = {name: 0.0 for name in rois}
    counts = 0
    pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
        for name, r in rois.items():
            area = grab_cut(bw, r["xstart"], r["ystart"], r["xlen"], r["ylen"])
            thresholds[name] += float(np.sum(area))
        counts += 1
    cap.set(cv.CAP_PROP_POS_FRAMES, pos)
    if counts == 0:
        raise RuntimeError("Could not read any frames to compute thresholds.")
    for k in thresholds:
        thresholds[k] /= counts
    return thresholds

def detect_inside_stream(cap, rois, thresholds, trials_meta, fps=None, thresh_value=160, manual_offset_secs=None):
    if fps is None:
        fps = cap.get(cv.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
    dt_ms = 1000.0 / fps
    first_start = trials_meta["trial_start_time"].min()
    if manual_offset_secs is None:
        offset = 0.0
    else:
        offset = float(manual_offset_secs)
    trial_windows = {}
    for tid, row in trials_meta.iterrows():
        start_rel = float(row["trial_start_time"] - first_start - offset)
        end_rel = float(row["end_trial_time"] - first_start - offset)
        trial_windows[int(tid)] = (start_rel, end_rel)

    ent1_hist = [False, False]
    ent2_hist = [False, False]
    hasLeft1 = False
    hasLeft2 = False
    e2Aftere1 = False
    e1Aftere2 = False
    enteredMaze = False
    inside_ms = {int(tid): 0.0 for tid in trials_meta.index.unique()}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)

        mouse_present = {}
        for name, r in rois.items():
            area = grab_cut(bw, r["xstart"], r["ystart"], r["xlen"], r["ylen"])
            mouse_present[name] = (np.sum(area) < thresholds[name] * 0.5)

        ent1_hist.insert(0, mouse_present.get("entrance1", False)); ent1_hist.pop()
        ent2_hist.insert(0, mouse_present.get("entrance2", False)); ent2_hist.pop()

        if (not ent1_hist[0]) and ent1_hist[1]:
            hasLeft1 = True
            if hasLeft2:
                e1Aftere2 = True; e2Aftere1 = False
        if (not ent2_hist[0]) and ent2_hist[1]:
            hasLeft2 = True
            if hasLeft1:
                e1Aftere2 = False; e2Aftere1 = True
        if e2Aftere1 and not enteredMaze:
            enteredMaze = True
        if e1Aftere2 and enteredMaze:
            enteredMaze = False

        t_rel = frame_idx / fps
        if enteredMaze:
            for tid, (s_rel, e_rel) in trial_windows.items():
                if (t_rel >= s_rel) and (t_rel < e_rel):
                    inside_ms[int(tid)] += dt_ms
                    break
        frame_idx += 1
    return inside_ms

def update_trials_csv(trials_csv: Path, inside_ms: dict):
    df = pd.read_csv(trials_csv)
    if "time_in_maze_ms" not in df.columns:
        df["time_in_maze_ms"] = np.nan
    for tid, ms in inside_ms.items():
        df.loc[df["trial_ID"] == int(tid), "time_in_maze_ms"] = float(ms)
    df.to_csv(trials_csv, index=False)
    return df

def auto_find(path: Path, patterns):
    files = []
    for pat in patterns:
        files.extend(list(path.glob(pat)))
    return files[0] if files else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, required=True)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None)
    ap.add_argument("--rois", type=str, default="rois1.csv")
    ap.add_argument("--manual-offset-secs", type=float, default=None)
    ap.add_argument("--probe-frames", type=int, default=10)
    ap.add_argument("--thresh-value", type=int, default=160)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--rois-number", type=int, default=8)
    args = ap.parse_args()

    session = Path(args.session)
    if not session.exists():
        raise SystemExit(f"Session path not found: {session}")
    video = Path(args.video) if args.video else auto_find(session, ["*.mp4", "*.avi", "*.mov", "*.mkv"])
    trials = Path(args.trials) if args.trials else auto_find(session, ["trials_time_*.csv", "*trials*.csv"])
    rois_csv = Path(args.rois)
    if not rois_csv.is_file():
        rois_csv = session / args.rois
    if video is None: raise SystemExit("No video file found.")
    if trials is None: raise SystemExit("No trials CSV found.")

    print(f"[INFO] Video : {video}")
    print(f"[INFO] Trials: {trials}")
    print(f"[INFO] ROIs  : {rois_csv}")

    if not rois_csv.exists():
        print(f"[INFO] ROI file not found at {rois_csv}, launching ROI selection...")
        
        roiNames = ["entrance1", "entrance2"] + sf.get_rois_list(args.rois_number)
        sf.define_rois(videoInput=str(video), roiNames=roiNames, outputName=str(rois_csv))
        if not Path(rois_csv).exists():
            raise SystemExit("ROI selection cancelled or failed; no file created.")

    trials_df = pd.read_csv(trials)
    required_cols = {"trial_ID","trial_start_time","end_trial_time"}
    missing = required_cols - set(trials_df.columns)
    if missing:
        raise SystemExit(f"Trials CSV missing: {missing}")
    trials_meta = (trials_df.groupby("trial_ID")[["trial_start_time","end_trial_time"]].first()
                   .sort_index())

    rois = load_rois(rois_csv)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    thresholds = compute_thresholds(cap, rois, num_frames=args.probe_frames, thresh_value=args.thresh_value)
    print("[INFO] Baseline thresholds computed.")

    inside_ms = detect_inside_stream(
        cap, rois, thresholds, trials_meta,
        fps=args.fps, thresh_value=args.thresh_value,
        manual_offset_secs=args.manual_offset_secs
    )
    cap.release()
    update_trials_csv(trials, inside_ms)
    print("[DONE] Updated CSV with 'time_in_maze_ms'.")

if __name__ == "__main__":
    main()
