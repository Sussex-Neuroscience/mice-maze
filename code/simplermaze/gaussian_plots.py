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

# ==========================================
# CONFIGURATION
# ==========================================
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

# ==========================================
# HELPERS
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
            if len(points) > 1: cv.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
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
        real_r = (int(r[0]/scale), int(r[1]/scale), int(r[2]/scale), int(r[3]/scale))
        rois[name] = real_r
    cv.destroyAllWindows()
    return rois

def filter_by_boundary(df, boundary_points):
    poly = Path(boundary_points)
    scorer = df.columns[0][0]
    bp = df.columns[0][1]
    coords = df[scorer][bp][['x', 'y']].fillna(-100).values
    is_inside = poly.contains_points(coords)
    df.loc[~is_inside, (scorer, bp, 'x')] = np.nan
    df.loc[~is_inside, (scorer, bp, 'y')] = np.nan
    return df

def process_kinematics(df, fps, px_per_cm):
    df['x_smooth'] = savgol_filter(df['x'], 15, 3)
    df['y_smooth'] = savgol_filter(df['y'], 15, 3)
    dist_px = np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2)
    df['speed_cm_s'] = (dist_px / px_per_cm) * fps
    return df

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

# 2. Boundary
boundary_file = session_path + 'maze_boundary.csv'
if not os.path.exists(boundary_file):
    boundary_pts = select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(boundary_file, index=False)
else:
    boundary_pts = pd.read_csv(boundary_file).values.tolist()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics
tracking = process_kinematics(tracking, FPS, PX_PER_CM)

# 4. ROIs
rois_file = session_path + "rois1.csv"
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if not os.path.exists(rois_file):
    rois = define_rois(VIDEO_PATH, roi_names)
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

# ==========================================
# COMBINED GAUSSIAN PLOTTING FUNCTION
# ==========================================
def plot_combined_gaussian(df, filename):
    if df.empty: return

    # Create subplots: 1 Row, 2 Columns
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Phase 1: Entry -> ROI", "Phase 2: ROI -> Exit"),
        horizontal_spacing=0.15 # Gap between plots
    )
    
    # Styles
    # Line color (Opaque) | Fill color (Transparent)
    styles = {
        'Hit':  {'line': 'rgba(0, 128, 0, 1)',   'fill': 'rgba(0, 128, 0, 0.1)'},
        'Miss': {'line': 'rgba(255, 0, 0, 1)',   'fill': 'rgba(255, 0, 0, 0.1)'}
    }
    
    phases = ['P1', 'P2']
    
    # Global Max Speed for shared X-axis
    global_max_speed = df['MeanSpeed'].max() * 1.3
    x_range = np.linspace(0, global_max_speed, 1000)

    for i, phase in enumerate(phases):
        col_idx = i + 1 # Plotly uses 1-based indexing
        subset = df[df['Phase'] == phase]
        
        if subset.empty: continue
        
        stats = subset.groupby('Status')['MeanSpeed'].agg(['mean', 'std']).reset_index()
        
        for idx, row in stats.iterrows():
            status = row['Status']
            mu = row['mean']
            sigma = row['std']
            
            if pd.isna(sigma) or sigma == 0: sigma = 0.1
            
            # PDF Calculation
            y = norm.pdf(x_range, mu, sigma)
            
            # Add Trace
            fig.add_trace(go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f"{status} (μ={mu:.1f})",
                line=dict(color=styles[status]['line'], width=4), # Thick Line
                fill='tozeroy', 
                fillcolor=styles[status]['fill'], # Transparent Fill
                legendgroup=status, # Click one legend to hide both P1/P2 traces
                showlegend=(col_idx==1) # Only show legend once
            ), row=1, col=col_idx)
            
            # Add Vertical Line for Mean
            fig.add_vline(x=mu, line_width=1, line_dash="dash", 
                          line_color=styles[status]['line'], row=1, col=col_idx)

    # Add a visual separator line (Draw a shape on the layout)
    fig.add_shape(type="line",
        x0=0.5, y0=0, x1=0.5, y1=1,
        xref="paper", yref="paper",
        line=dict(color="Black", width=2)
    )

    fig.update_layout(
        title="Speed Distributions: Hits vs Misses (Gaussian Fit)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        width=1200
    )
    
    # Set shared X-axis labels
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

print("Generating Combined Gaussian Plot...")
plot_combined_gaussian(df_res, 'summary_gaussian_combined.html')

print(f"Done! Open {OUTPUT_DIR}/summary_gaussian_combined.html")