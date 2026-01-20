import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.stats import entropy
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


# CONFIGURATION

base_path = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/"

TRIAL_INFO_PATH = base_path + r"2024-08-28_11_58_146357session3.6/trials_corrected_final_frames.csv"

DLC_DATA_PATH = base_path + r"deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"

VIDEO_PATH = base_path + r'2024-08-28_11_58_146357session3.6/6357_2024-08-28_11_58_14s3.6.mp4' # Replace with your full video path
FPS = 30  # Adjust to your video's frame rate
BODYPART = 'mid'  # Bodypart used for tracking
DRAW_ROIS = True   # Set to True to draw ROIs manually
DRAW_BOUNDARIES = True
# GRID_SIZE = 20    # For Spatial Entropy
LIKELIHOOD_THRESH = 0.8
OUTPUT_DIR = base_path + r'2024-08-28_11_58_146357session3.6/trial_analysis_plots'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# UI HELPERS (OPENCV)
# ==========================================
def resize_for_display(img, max_height=800):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv.resize(img, (int(w*scale), int(h*scale))), scale
    return img.copy(), 1.0

def select_maze_boundary(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return []

    display_frame, scale = resize_for_display(frame)
    points = []
    
    def click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv.circle(display_frame, (x, y), 4, (0, 0, 255), -1)
            if len(points) > 1:
                cv.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
            cv.imshow("Draw Boundary", display_frame)

    cv.namedWindow("Draw Boundary", cv.WINDOW_NORMAL)
    cv.imshow("Draw Boundary", display_frame)
    cv.setMouseCallback("Draw Boundary", click)
    
    print("Draw Boundary: Click corners -> 'c' to close -> Space to confirm")
    while True:
        k = cv.waitKey(1) & 0xFF
        if k == ord('c') and len(points) > 2:
            cv.line(display_frame, points[-1], points[0], (0, 255, 0), 2)
            cv.imshow("Draw Boundary", display_frame)
        if k == 32 or k == 13: break
            
    cv.destroyAllWindows()
    # Scale back
    return [(int(p[0]/scale), int(p[1]/scale)) for p in points]

def define_rois(video_path, roi_names):
    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()
    
    display_frame, scale = resize_for_display(frame)
    cv.namedWindow("Select ROIs", cv.WINDOW_NORMAL)
    rois = {}
    
    print("Draw ROIs: Select box -> Space to confirm")
    for name in roi_names:
        print(f"Select: {name}")
        r = cv.selectROI("Select ROIs", display_frame, fromCenter=False)
        # Scale back
        real_r = (int(r[0]/scale), int(r[1]/scale), int(r[2]/scale), int(r[3]/scale))
        rois[name] = real_r
        
        cv.rectangle(display_frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0,255,0), 2)
        cv.putText(display_frame, name, (r[0], r[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv.imshow("Select ROIs", display_frame)
        
    cv.destroyAllWindows()
    return rois

# ==========================================
# PROCESSING HELPERS
# ==========================================
def filter_by_boundary(df, boundary_points):
    poly = Path(boundary_points)
    scorer = df.columns[0][0]
    bp = df.columns[0][1]
    coords = df[scorer][bp][['x', 'y']].fillna(-100).values
    is_inside = poly.contains_points(coords)
    df.loc[~is_inside, (scorer, bp, 'x')] = np.nan
    df.loc[~is_inside, (scorer, bp, 'y')] = np.nan
    return df

def process_kinematics(df, fps):
    # Savitzky-Golay Smoothing
    df['x_smooth'] = savgol_filter(df['x'], 15, 3)
    df['y_smooth'] = savgol_filter(df['y'], 15, 3)
    
    # Speed (px/s)
    dist = np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2)
    df['speed'] = dist * fps
    return df

def get_entropy(x, y, grid_size=20):
    if len(x) < 2: return 0
    hist, _, _ = np.histogram2d(x, y, bins=grid_size, density=True)
    return entropy(hist.flatten()[hist.flatten() > 0])

# ==========================================
# MAIN EXECUTION
# ==========================================
print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Likelihood Filter
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 2. Boundary Filter
if DRAW_BOUNDARIES:
    boundary_pts = select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv('maze_boundary.csv', index=False)
elif os.path.exists('maze_boundary.csv'):
    boundary_pts = pd.read_csv('maze_boundary.csv').values.tolist()
else:
    print("Set DRAW_BOUNDARIES=True"); exit()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics
tracking = process_kinematics(tracking, FPS)

# 4. ROIs
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROIS:
    rois = define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv('defined_rois.csv')
elif os.path.exists('defined_rois.csv'):
    df_r = pd.read_csv('defined_rois.csv', index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
else:
    print("Set DRAW_ROIS=True"); exit()

# --- ANALYSIS LOOP ---
results = []
p1_traces = [] # Store data for Plotly Summary
p2_traces = []

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    # Identify Phase Split
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    ent1 = get_entropy(p1['x_smooth'], p1['y_smooth'])
    ent2 = get_entropy(p2['x_smooth'], p2['y_smooth'])
    
    results.append({
        'trial': idx, 'target': target,
        'P1_speed': p1['speed'].mean(), 'P1_entropy': ent1,
        'P2_speed': p2['speed'].mean(), 'P2_entropy': ent2
    })
    
    # --- STORE DATA FOR PLOTLY SUMMARY ---
    # We store the time series (Time vs Speed) and Trial ID
    if len(p1) > 0:
        time_p1 = np.arange(len(p1)) / FPS
        p1_traces.append(pd.DataFrame({'Time': time_p1, 'Speed': p1['speed'].values, 'Trial': f"Trial {idx}"}))
    if len(p2) > 0:
        time_p2 = np.arange(len(p2)) / FPS
        p2_traces.append(pd.DataFrame({'Time': time_p2, 'Speed': p2['speed'].values, 'Trial': f"Trial {idx}"}))

    # --- SAVE STATIC HEATMAP (MATPLOTLIB) ---
    # Heatmaps are best saved as static images for browsing
    plt.figure(figsize=(6, 6))
    plt.hist2d(trial_data['x_smooth'], trial_data['y_smooth'], bins=30, cmap='inferno', norm=LogNorm())
    
    # Overlay Boundary
    px, py = zip(*boundary_pts); px = list(px)+[px[0]]; py = list(py)+[py[0]]
    plt.plot(px, py, 'w-', lw=2)
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title(f"Trial {idx} Heatmap")
    plt.colorbar(label='Density')
    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_heatmap.png")
    plt.close()

# --- SAVE RESULTS ---
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

# ==========================================
# PLOTLY SUMMARY PLOTS
# ==========================================
print("Generating Plotly Summaries...")

def plot_summary_plotly(trace_data, title, filename):
    if not trace_data: return
    
    # Combine all traces into one DataFrame
    df_plot = pd.concat(trace_data)
    
    # Create interactive line plot
    fig = px.line(df_plot, x='Time', y='Speed', color='Trial', 
                  title=title, labels={'Time': 'Time (s)', 'Speed': 'Speed (px/s)'})
    
    # Improve styling
    fig.update_layout(hovermode="x unified", template="plotly_white")
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

plot_summary_plotly(p1_traces, "Phase 1 Speed (Entry -> First ROI)", "summary_speed_P1.html")
plot_summary_plotly(p2_traces, "Phase 2 Speed (First ROI -> Exit)", "summary_speed_P2.html")

print(f"Analysis Complete! Open {OUTPUT_DIR} to see:")
print("1. summary_speed_P1.html (Interactive)")
print("2. summary_speed_P2.html (Interactive)")
print("3. Individual trial_XX_heatmap.png files")
print("4. results.csv")