import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os

# ==========================================
# CONFIGURATION
# ==========================================
TRIAL_INFO_PATH = "C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/trials_corrected_final_frames.csv"

DLC_DATA_PATH = "C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"

VIDEO_PATH = 'C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/6357_2024-08-28_11_58_14s3.6.mp4' # Replace with your full video path
FPS = 30  # Adjust to your video's frame rate
BODYPART = 'mid'  # Bodypart used for tracking
DRAW_ROI = True   # Set to True to draw ROIs manually
GRID_SIZE = 20    # For Spatial Entropy
OUTPUT_DIR = 'C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/trial_analysis_plots'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# ROI DEFINITION LOGIC (from your supFun.py)
# ==========================================
def define_rois_from_video(video_path, roi_names):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Could not read video for ROI selection.")
        return None
    
    rois = {}
    temp_display = frame.copy()
    
    for name in roi_names:
        print(f"Select ROI for: {name}. Press SPACE or ENTER to confirm.")
        roi = cv.selectROI("Select ROIs", temp_display, fromCenter=False, showCrosshair=True)
        rois[name] = roi # (x, y, w, h)
        # Draw on temp display to show progress
        x, y, w, h = roi
        cv.rectangle(temp_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(temp_display, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    cv.destroyAllWindows()
    cap.release()
    return rois

def is_in_roi(x, y, roi_coords):
    """Checks if point x,y is inside ROI (x_start, y_start, width, height)"""
    rx, ry, rw, rh = roi_coords
    return rx <= x <= (rx + rw) and ry <= y <= (ry + rh)

# ==========================================
# CALCULATION HELPERS
# ==========================================
def calculate_spatial_entropy(x, y, bins=GRID_SIZE):
    if len(x) == 0: return 0
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    probs = hist.flatten() / np.sum(hist)
    probs = probs[probs > 0]
    return entropy(probs)

# ==========================================
# MAIN PIPELINE
# ==========================================

# 1. Load Data
df_trials = pd.read_csv(TRIAL_INFO_PATH)
# DLC has 3 header rows
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 2. Draw ROIs if flag is set
roi_list = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROI:
    rois = define_rois_from_video(VIDEO_PATH, roi_list)
else:
    # Logic to load from existing rois1.csv if needed
    rois_df = pd.read_csv("rois1.csv", index_col=0)
    rois = {col: (rois_df[col]['xstart'], rois_df[col]['ystart'], rois_df[col]['xlen'], rois_df[col]['ylen']) for col in rois_df.columns}

# 3. Process Trials
all_results = []

for idx, row in df_trials.iterrows():
    # Identify key timepoints (ms to frames)
    # Note: Using absolute time mapping. Adjust if your CSV uses relative offsets.
    start_frame = int(row['mouse_enter_time'] * FPS / 1000)
    end_frame = int(row['end_trial_time'] * FPS / 1000)
    first_roi_name = row['first_reward_area_visited']
    
    if first_roi_name not in rois:
        print(f"Skipping trial {idx}: first ROI '{first_roi_name}' not defined.")
        continue

    trial_track = tracking.iloc[start_frame:end_frame].copy()
    
    # Find exact frame of first ROI entry
    first_roi_coords = rois[first_roi_name]
    entry_idx = None
    for i, (f_idx, pos) in enumerate(trial_track.iterrows()):
        if is_in_roi(pos['x'], pos['y'], first_roi_coords):
            entry_idx = f_idx
            break
    
    if entry_idx is None:
        print(f"Warning: Mouse never detected in {first_roi_name} for trial {idx} (using mid-point split)")
        entry_idx = start_frame + (end_frame - start_frame) // 2

    # Split phases
    phase1 = tracking.iloc[start_frame:entry_idx].copy()
    phase2 = tracking.iloc[entry_idx:end_frame].copy()

    trial_metrics = {'trial': idx, 'first_roi': first_roi_name}

    # Calculate metrics for both phases
    for p_name, p_data in [('P1_EntryToROI', phase1), ('P2_ROIToExit', phase2)]:
        if len(p_data) < 2:
            continue
            
        # Kinematics
        dist = np.sqrt(p_data['x'].diff()**2 + p_data['y'].diff()**2)
        speed = dist * FPS
        accel = speed.diff() * FPS
        
        trial_metrics[f'{p_name}_time'] = len(p_data) / FPS
        trial_metrics[f'{p_name}_avg_speed'] = speed.mean()
        trial_metrics[f'{p_name}_avg_accel'] = accel.mean()
        trial_metrics[f'{p_name}_entropy'] = calculate_spatial_entropy(p_data['x'], p_data['y'])
        
        # Plot Speed Over Time for this trial
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(p_data)/FPS, len(p_data)), speed, label=p_name)
        plt.title(f"Trial {idx} - Speed Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (px/s)")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_{p_name}_speed.png")
        plt.close()

    all_results.append(trial_metrics)

# 4. Create Final Dataframe
results_df = pd.DataFrame(all_results)
results_df.to_csv("comprehensive_trial_analysis.csv", index=False)

# 5. Summary Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df, x='first_roi', y='P1_EntryToROI_avg_speed')
plt.title("Average Speed to First ROI by Target Location")
plt.savefig("summary_speed_by_roi.png")

plt.figure(figsize=(8, 6))
sns.violinplot(data=results_df.melt(value_vars=['P1_EntryToROI_entropy', 'P2_ROIToExit_entropy']), 
               x='variable', y='value')
plt.title("Spatial Entropy Comparison Between Phases")
plt.savefig("summary_entropy_comparison.png")

print("Analysis complete. Results saved to comprehensive_trial_analysis.csv")