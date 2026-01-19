import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.signal import savgol_filter
import os
import matplotlib.patches as patches

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
LIKELIHOOD_THRESH = 0.8
OUTPUT_DIR = 'C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/trial_analysis_plots'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. ROI SELECTION
# ==========================================
def define_rois_full_frame(video_path, roi_names):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading video: {video_path}")
        return {}

    print("\n--- ROI SELECTION ---")
    print("Draw the box for each ROI and press SPACE or ENTER to confirm.")
    
    rois = {}
    display_frame = frame.copy()

    for name in roi_names:
        print(f"--> Select ROI for: {name}")
        roi = cv.selectROI("Select ROIs", display_frame, fromCenter=False, showCrosshair=True)
        rois[name] = roi
        
        # Draw on display frame
        x, y, w, h = roi
        cv.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(display_frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.destroyAllWindows()
    cap.release()
    
    # Save to CSV
    df_rois = pd.DataFrame(rois).T
    df_rois.columns = ['x', 'y', 'w', 'h']
    df_rois.to_csv('defined_rois.csv')
    
    return rois

def is_in_roi(x, y, roi_coords):
    rx, ry, rw, rh = roi_coords
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

# ==========================================
# 2. ANALYSIS HELPERS
# ==========================================
def process_kinematics(df, fps):
    # Smooth coordinates
    df['x_smooth'] = savgol_filter(df['x'], window_length=15, polyorder=3)
    df['y_smooth'] = savgol_filter(df['y'], window_length=15, polyorder=3)
    
    # Speed & Accel
    dx = df['x_smooth'].diff()
    dy = df['y_smooth'].diff()
    dist = np.sqrt(dx**2 + dy**2)
    
    df['speed_px_s'] = dist * fps
    df['accel_px_s2'] = df['speed_px_s'].diff() * fps
    return df

def get_spatial_entropy(x, y, grid_size=20):
    if len(x) < 2: return 0
    hist, _, _ = np.histogram2d(x, y, bins=grid_size, density=True)
    probs = hist.flatten()
    probs = probs[probs > 0]
    return entropy(probs)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

# --- A. Load Data ---
print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH) #

df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# --- B. Likelihood Filtering ---
print(f"Filtering points with likelihood < {LIKELIHOOD_THRESH}...")
low_conf_mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[low_conf_mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate(method='linear', limit=10).ffill().bfill()

# --- C. Kinematics ---
print("Calculating Kinematics...")
tracking = process_kinematics(tracking, FPS)

# --- D. ROIs ---
roi_list = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROI:
    rois = define_rois_full_frame(VIDEO_PATH, roi_list)
elif os.path.exists('defined_rois.csv'):
    print("Loading saved ROIs...")
    df_r = pd.read_csv('defined_rois.csv', index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
else:
    print("No ROI file. Set DRAW_ROI=True.")
    exit()

# --- E. Iterate Trials ---
results = []
print("Processing trials...")

for idx, row in df_trials.iterrows():
    # 1. Get Frames
    try:
        f_start = int(row['start_frame'])
        f_end = int(row['end_frame'])
    except (ValueError, TypeError):
        continue
        
    target_roi_name = row['first_reward_area_visited']
    
    # 2. Extract Data
    if f_start >= f_end or target_roi_name not in rois: continue
    
    try:
        trial_data = tracking.iloc[f_start:f_end].copy()
    except IndexError:
        continue

    if len(trial_data) < 5: continue
    
    # 3. Split Phases
    target_coords = rois[target_roi_name]
    split_idx = -1
    
    for i, (f_idx, pos) in enumerate(trial_data.iterrows()):
        if is_in_roi(pos['x_smooth'], pos['y_smooth'], target_coords):
            split_idx = i
            break
    
    if split_idx == -1:
        print(f"Trial {idx}: Never entered {target_roi_name}")
        continue

    phase1 = trial_data.iloc[:split_idx]
    phase2 = trial_data.iloc[split_idx:]
    
    # 4. Calculate Metrics
    ent1 = get_spatial_entropy(phase1['x_smooth'], phase1['y_smooth'])
    ent2 = get_spatial_entropy(phase2['x_smooth'], phase2['y_smooth'])
    
    trial_res = {
        'trial_id': idx, 
        'target_roi': target_roi_name,
        'P1_dur': len(phase1)/FPS, 'P1_speed': phase1['speed_px_s'].mean(), 'P1_entropy': ent1,
        'P2_dur': len(phase2)/FPS, 'P2_speed': phase2['speed_px_s'].mean(), 'P2_entropy': ent2
    }
    results.append(trial_res)

    # ==========================================
    # VISUALIZATION 1: Speed over Time
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Phase 1 Speed
    if len(phase1) > 0:
        axes[0].plot(np.arange(len(phase1))/FPS, phase1['speed_px_s'], color='blue')
        axes[0].set_title(f"P1 (Entry->ROI): Speed\nEntropy: {ent1:.2f}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Speed (px/s)")
        
    # Phase 2 Speed
    if len(phase2) > 0:
        axes[1].plot(np.arange(len(phase2))/FPS, phase2['speed_px_s'], color='red')
        axes[1].set_title(f"P2 (ROI->Exit): Speed\nEntropy: {ent2:.2f}")
        axes[1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_speed.png")
    plt.close()

    # ==========================================
    # VISUALIZATION 2: Trajectory Map (Path)
    # ==========================================
    # This visualizes "Spatial Entropy" effectively
    plt.figure(figsize=(6, 6))
    
    # Plot Path P1
    plt.plot(phase1['x_smooth'], phase1['y_smooth'], color='blue', alpha=0.7, label='P1: Entry->ROI')
    # Plot Path P2
    plt.plot(phase2['x_smooth'], phase2['y_smooth'], color='red', alpha=0.7, label='P2: ROI->Exit')
    
    # Draw Target ROI Box
    rx, ry, rw, rh = target_coords
    rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='green', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(rx, ry-10, target_roi_name, color='green', fontsize=9)
    
    # Flip Y axis (images usually have (0,0) at top-left)
    plt.gca().invert_yaxis()
    plt.title(f"Trial {idx} Path\nEnt1: {ent1:.2f} | Ent2: {ent2:.2f}")
    plt.legend()
    plt.axis('equal')
    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_path.png")
    plt.close()

# --- F. Summary Plots ---
df_results = pd.DataFrame(results)
output_csv = os.path.join(OUTPUT_DIR, 'final_analysis_results.csv')
df_results.to_csv(output_csv, index=False)
print(f"\nAnalysis complete. Results saved to {output_csv}")

if not df_results.empty:
    # 1. Speed Summary
    plt.figure(figsize=(8, 6))
    melted_speed = df_results.melt(value_vars=['P1_speed', 'P2_speed'], var_name='Phase', value_name='Speed')
    sns.boxplot(data=melted_speed, x='Phase', y='Speed', palette=['blue', 'red'])
    plt.title("Average Speed Comparison")
    plt.ylabel("Speed (px/s)")
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_speed_boxplot.png"))
    plt.close()

    # 2. ENTROPY Summary
    plt.figure(figsize=(8, 6))
    melted_ent = df_results.melt(value_vars=['P1_entropy', 'P2_entropy'], var_name='Phase', value_name='Entropy')
    sns.boxplot(data=melted_ent, x='Phase', y='Entropy', palette=['blue', 'red'])
    plt.title("Spatial Entropy Comparison")
    plt.ylabel("Shannon Entropy")
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_entropy_boxplot.png"))
    plt.close()