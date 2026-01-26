import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.signal import savgol_filter
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import analysisfunct as af


# FILE PATHS
base_path = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/"
session_path = base_path + r"2024-08-28_11_58_146357session3.6/"

TRIAL_INFO_PATH = session_path + r"trials_corrected_final_frames.csv"
DLC_DATA_PATH = base_path + r"deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
VIDEO_PATH = session_path + r'6357_2024-08-28_11_58_14s3.6.mp4'

FPS = 30
PX_PER_CM = 7.5
BODYPART = 'mid'
LIKELIHOOD_THRESH = 0.5
OUTPUT_DIR = session_path + r"trial_analysis_gaussian"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Likelihood Filter
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 2. Boundary
boundary_file = session_path + 'maze_boundary.csv'
if not os.path.exists(boundary_file):
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(boundary_file, index=False)
else:
    boundary_pts = pd.read_csv(boundary_file).values.tolist()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = af.filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics
tracking = af.process_kinematics(tracking, FPS, PX_PER_CM)

# 4. ROIs
rois_file = session_path + "rois1.csv"
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if not os.path.exists(rois_file):
    rois = af.define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv(rois_file)
else:
    df_r = pd.read_csv(rois_file, index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}

# DATA COLLECTION
data_points = []

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        is_hit = (row['hit'] == 1)
        status = "Hit" if is_hit else "Miss"
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if np.isnan(pos['x_smooth']) or np.isnan(pos['y_smooth']): continue
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    if len(p1) > 0:
        data_points.append({'Phase': 'P1', 'Status': status, 'MeanSpeed': p1['speed_cm_s'].mean()})
    if len(p2) > 0:
        data_points.append({'Phase': 'P2', 'Status': status, 'MeanSpeed': p2['speed_cm_s'].mean()})

df_res = pd.DataFrame(data_points)
df_res.to_csv(f"{OUTPUT_DIR}/gaussian_source_data.csv", index=False)




print("Generating Combined Gaussian Plot (Shared Y)...")
af.plot_combined_gaussian(df_res, f'{OUTPUT_DIR}/summary_gaussian_combined_sharedY.html')

print(f"Done! Open {OUTPUT_DIR}/summary_gaussian_combined_sharedY.html")