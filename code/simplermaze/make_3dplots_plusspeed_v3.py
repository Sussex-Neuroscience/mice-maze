"""
Batch-process DeepLabCut data to generate per-trial 3D (x, y, time)
interactive trajectory plots using Plotly.

MAJOR UPDATE:
- Matches trials by extracting the Trial Index (e.g., "001") from both files.
- Trials Info File: Extracts "trial_001" ([-13:-4]) -> parses to "001".
- Segments File: Extracts "001" directly using [-7:-4].

Example Usage:
----------------
python make_3dplots_plusspeed_v3.py ^
    --dlc "C:/Path/To/DLC_coords.csv" ^
    --segments "C:/Path/To/segments_manifest.csv" ^
    --trials-types "C:/Path/To/mouse6357_session3.6_trial_info.csv" ^
    --keypoint "nose" ^
    --maze-geometry "C:/Path/To/maze_structure.json" ^
    --output-dir "./3d_trajectory_plots"

python make_3dplots_plusspeed_v3.py --dlc "C:/Users/shahd/OneDrive - University of Sussex/DLC_MOSEQ/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv" --segments "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_segments_manifest.csv" --trials-types "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/mouse6357_session3.6_trial_info.csv" --keypoint back_mid --output-dir "./trajectories_maze_video_back_mid" --min-likelihood 0.8 --fps 30



"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import medfilt
import cv2  # For pointPolygonTest

# ----------------------------------------------------------------------------
# 1. UTILITY FUNCTIONS
# ----------------------------------------------------------------------------

def read_dlc_csv_multiindex(path: str) -> pd.DataFrame:
    """Loads a standard 3-header DLC CSV file."""
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    except Exception as e:
        print(f"Error reading DLC file: {e}")
        print("Trying with no index column...")
        df = pd.read_csv(path, header=[0, 1, 2])
        
    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) != 3:
        raise ValueError("DLC CSV does not appear to have a 3-level header.")
    return df

def get_scorer_and_bodyparts(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """Extracts the scorer and bodyparts from the multi-index columns."""
    level0 = sorted(set([lvl0 for (lvl0, _, _) in df.columns]))
    scorer_candidates = [s for s in level0 if s.lower() != "scorer"]
    if not scorer_candidates:
        raise ValueError("Could not find DLC 'scorer' level.")
    scorer = scorer_candidates[0]
    
    bodyparts = sorted(set([bp for (_, bp, _) in df.columns if bp.lower() != "bodyparts"]))
    return scorer, bodyparts

def extract_frame_series(df: pd.DataFrame) -> pd.Series:
    """Gets the frame index, whether it's from the index or a column."""
    if ('scorer', 'bodyparts', 'coords') in df.columns:
        frames = df[('scorer', 'bodyparts', 'coords')].astype(int)
        frames.name = 'frame'
        return frames
    return pd.Series(np.arange(len(df)), name='frame') # Fallback to row number

def pick_keypoint(bodyparts: List[str], desired: Optional[str]) -> str:
    """Selects the desired keypoint or a sensible default."""
    if desired and desired in bodyparts:
        return desired
    for candidate in ["nose", "mid", "center", "centroid", "tailbase"]:
        if candidate in bodyparts:
            print(f"Warning: Keypoint '{desired}' not found. Defaulting to '{candidate}'.")
            return candidate
    print(f"Warning: Keypoint '{desired}' not found. Defaulting to first part: '{bodyparts[0]}'.")
    return bodyparts[0]

def load_maze_geometry(path: str) -> Optional[Dict]:
    """Loads the maze structure JSON file."""
    if not path or not os.path.exists(path):
        print("Info: No maze geometry file provided or found.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            maze = json.load(f)
        # Pre-compile contours for cv2
        for comp in maze.get('compartments', []):
            pts = np.array(comp['contour']).reshape(-1, 1, 2).astype(np.int32)
            comp['_cv2_contour'] = pts
        print(f"✓ Loaded maze structure with {len(maze.get('compartments', []))} compartments")
        return maze
    except Exception as e:
        print(f"Error loading maze geometry: {e}")
        return None

def get_compartment(x: float, y: float, maze_structure: Dict) -> str:
    """Determine which compartment a point is in using cv2.pointPolygonTest."""
    for comp in maze_structure.get('compartments', []):
        contour = comp.get('_cv2_contour')
        if contour is None:
            continue
        result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
        if result >= 0:
            return comp.get('id', 'Unknown')
    return 'Outside'  # Not in any compartment

def decode_state_sequence(value: Any) -> str:
    """
    Maps state codes to labels.
    1: Explore, 2: Exploit, 3: Nest, 4: Groom
    """
    if pd.isna(value):
        return "Unknown"
    
    mapping = {'1': 'Explore', '2': 'Exploit', '3': 'Nest', '4': 'Groom'}
    
    # Convert input to string and clean
    s_val = str(value).replace('"', '').replace("'", "")
    
    # Split by comma
    parts = [p.strip() for p in s_val.split(',')]
    
    # Decode
    decoded_parts = []
    for p in parts:
        if p.replace('.','',1).isdigit():
            p = str(int(float(p)))
        decoded_parts.append(mapping.get(p, p))
        
    return "-".join(decoded_parts)

def extract_trial_id_from_path(path_str: Any) -> str:
    """
    Extracts the trial ID (e.g. 'trial_001') from the video path.
    Implementation uses the user's logic: slice [-13:-4].
    Expects format ending like: ..._trial_001.mp4
    """
    if pd.isna(path_str):
        return ""
    path_str = str(path_str).strip()
    if len(path_str) < 13:
        return ""
    
    # Extracts "trial_001"
    extracted = path_str[-13:-4]
    return extracted

def process_trial_data(df_full: pd.DataFrame, frames: pd.Series, 
                       start_frame: int, end_frame: int, trial_id: str,
                       scorer: str, keypoint: str, min_likelihood: float, 
                       fps: float, smooth_window: int, maze_structure: Optional[Dict]) -> Optional[pd.DataFrame]:
    """
    Slices, filters, and processes all data for a single trial.
    """
    # 1. Slice the full DataFrame by frame index
    mask = (frames >= start_frame) & (frames <= end_frame)
    if mask.sum() == 0:
        print(f"Warning: No frames found for {trial_id} (frames {start_frame}-{end_frame}). Skipping.")
        return None
        
    df_trial = df_full.loc[mask].copy()
    
    # 2. Extract columns
    xcol = (scorer, keypoint, 'x')
    ycol = (scorer, keypoint, 'y')
    lkcol = (scorer, keypoint, 'likelihood')
    
    if lkcol not in df_trial.columns:
        print(f"Warning: Likelihood column not found for {keypoint}. Filtering disabled.")
        min_likelihood = 0.0
    
    # 3. Filter by likelihood
    if min_likelihood > 0:
        low_like_mask = df_trial[lkcol] < min_likelihood
        df_trial.loc[low_like_mask, [xcol, ycol]] = np.nan
        
    # 4. Create new DataFrame with clean columns
    df_out = pd.DataFrame({
        'frame': df_trial.index if isinstance(df_trial.index, pd.RangeIndex) else frames[mask],
        'x': df_trial[xcol],
        'y': df_trial[ycol],
        'likelihood': df_trial[lkcol] if min_likelihood > 0 else 1.0
    })
    
    # 5. Interpolate missing (low-likelihood) points
    df_out['x'] = df_out['x'].interpolate(method='linear').bfill().ffill()
    df_out['y'] = df_out['y'].interpolate(method='linear').bfill().ffill()
    
    if df_out.isnull().values.any():
        print(f"Warning: Trial {trial_id} contains persistent NaN values. Skipping.")
        return None

    # 6. Smooth trajectory
    if smooth_window > 1 and len(df_out) > smooth_window:
        df_out['x'] = medfilt(df_out['x'], kernel_size=smooth_window)
        df_out['y'] = medfilt(df_out['y'], kernel_size=smooth_window)
        
    # 7. Calculate time and speed
    df_out['time_seconds'] = (df_out['frame'] - start_frame) / float(fps)
    speed = np.sqrt(np.diff(df_out['x'])**2 + np.diff(df_out['y'])**2)
    df_out['speed_px_per_frame'] = np.concatenate([[0], speed])
    
    # 8. Compartment analysis
    if maze_structure:
        df_out['compartment'] = df_out.apply(lambda row: get_compartment(row['x'], row['y'], maze_structure), axis=1)
        
    return df_out

# ----------------------------------------------------------------------------
# 2. PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------

def plot_trial_3d(df_trial: pd.DataFrame, trial_id: str, 
                  maze_structure: Optional[Dict], plot_type: str = 'time',
                  subtitle: str = "") -> go.Figure:
    
    fig = go.Figure()
    
    # --- Determine coloring ---
    if plot_type == 'speed' and 'speed_px_per_frame' in df_trial.columns:
        color_data = df_trial['speed_px_per_frame']
        colorscale = 'Jet'
        colorbar_title = "Speed (px/frame)"
        hover_text = "Speed: %{line.color:.2f}<br>"
    elif plot_type == 'compartment' and 'compartment' in df_trial.columns:
        unique_comps = df_trial['compartment'].unique()
        color_map = {comp: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, comp in enumerate(unique_comps)}
        df_trial['color'] = df_trial['compartment'].map(color_map)
        
        for comp_name, comp_data in df_trial.groupby('compartment'):
            fig.add_trace(go.Scatter3d(
                x=comp_data['x'],
                y=comp_data['y'],
                z=comp_data['time_seconds'],
                mode='lines+markers',
                line=dict(color=color_map[comp_name], width=4),
                marker=dict(size=2),
                name=comp_name,
                hovertemplate=f"<b>{comp_name}</b><br>" +
                              "X: %{x:.1f}<br>Y: %{y:.1f}<br>Time: %{z:.2f}s<br><extra></extra>"
            ))
        
        title_text = f'{trial_id} - {subtitle}' if subtitle else f'{trial_id}'
        fig.update_layout(title=title_text)
        return fig
        
    else: # Default to 'time'
        color_data = df_trial['time_seconds']
        colorscale = 'Viridis'
        colorbar_title = "Time (s)"
        hover_text = ""

    # --- Add Main Trajectory Line ---
    fig.add_trace(go.Scatter3d(
        x=df_trial['x'],
        y=df_trial['y'],
        z=df_trial['time_seconds'],
        mode='lines',
        line=dict(
            color=color_data,
            colorscale=colorscale,
            width=4,
            colorbar=dict(title=colorbar_title, x=1.1)
        ),
        name='Trajectory',
        hovertemplate='<b>Position</b><br>' +
                      'X: %{x:.1f}<br>Y: %{y:.1f}<br>Time: %{z:.2f}s<br>' +
                      hover_text + '<extra></extra>'
    ))

    # --- Add Start/End Markers ---
    fig.add_trace(go.Scatter3d(
        x=[df_trial['x'].iloc[0]], y=[df_trial['y'].iloc[0]], z=[df_trial['time_seconds'].iloc[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[df_trial['x'].iloc[-1]], y=[df_trial['y'].iloc[-1]], z=[df_trial['time_seconds'].iloc[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='End'
    ))

    # --- Add Maze Structure ---
    if maze_structure is not None:
        for comp in maze_structure.get('compartments', []):
            contour = np.array(comp['contour'])
            contour_closed = np.vstack([contour, contour[0]]) 
            fig.add_trace(go.Scatter3d(
                x=contour_closed[:, 0],
                y=contour_closed[:, 1],
                z=np.zeros(len(contour_closed)),
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # --- Add Vertical Projection Lines ---
    sample_rate = max(1, len(df_trial) // 50)
    for i in range(0, len(df_trial), sample_rate):
        row = df_trial.iloc[i]
        fig.add_trace(go.Scatter3d(
            x=[row['x'], row['x']], y=[row['y'], row['y']], z=[0, row['time_seconds']],
            mode='lines',
            line=dict(color='lightblue', width=1, dash='dot'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    title_text = f'{trial_id} - {subtitle}' if subtitle else f'{trial_id}'
    fig.update_layout(title=f'{title_text} (Colored by {plot_type.capitalize()})')
    return fig

def set_common_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (px)', showbackground=True, backgroundcolor="rgb(230,230,230)"),
            yaxis=dict(title='Y (px)', showbackground=True, backgroundcolor="rgb(230,230,230)", autorange='reversed'),
            zaxis=dict(title='Time (s)', showbackground=True, backgroundcolor="rgb(230,230,230)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=1000, height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest'
    )
    return fig

# ----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch-process DLC data into 3D interactive plots.")
    ap.add_argument("--dlc", required=True, help="Path to DLC CSV.")
    ap.add_argument("--segments", required=True, help="Path to segments_manifest CSV.")
    ap.add_argument("--trials-types", required=True, help="Path to CSV containing trial info (state_sequence, video_path).")
    ap.add_argument("--keypoint", required=True, help="Bodypart to plot.")
    ap.add_argument("--output-dir", required=True, help="Directory to save output.")
    ap.add_argument("--maze-geometry", default=None, help="Optional: JSON file with maze structure.")
    ap.add_argument("--min-likelihood", type=float, default=0.9)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--smoothing-window", type=int, default=5)
    
    args = ap.parse_args()

    # --- Setup ---
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.smoothing_window % 2 == 0: args.smoothing_window += 1
        
    print("Starting batch analysis (Matched by Filename ID)...")
    
    # --- Load Files ---
    try:
        df_full = read_dlc_csv_multiindex(args.dlc)
        seg_df = pd.read_csv(args.segments)
        types_df = pd.read_csv(args.trials_types)
        print(f"✓ Loaded Trials Info ({len(types_df)} rows) and Segments ({len(seg_df)} rows)")
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    scorer, bodyparts = get_scorer_and_bodyparts(df_full)
    frames = extract_frame_series(df_full)
    keypoint = pick_keypoint(bodyparts, args.keypoint)
    maze = load_maze_geometry(args.maze_geometry)

    # Clean column names for easier access
    seg_df.columns = seg_df.columns.str.lower().str.strip()
    
    # --- Prepare Segments Lookup Table ---
    # We want to quickly find start/end frames based on the trial ID (e.g., "001")
    print("Building segment lookup table...")
    
    seg_path_col = None
    possible_names = ['name', 'video', 'file', 'path', 'segment_path', 'video_segment_path']
    for c in seg_df.columns:
        if any(x in c for x in possible_names):
            seg_path_col = c
            break
    if not seg_path_col:
        seg_path_col = seg_df.columns[0]
        print(f"Warning: Could not identify path column in segments file. Using first column: '{seg_path_col}'")

    # Helper to clean paths for matching
    segments_lookup = {}
    for _, row in seg_df.iterrows():
        path_val = row[seg_path_col]
        
        # USER SPECIFIED LOGIC FOR SEGMENTS: [-7, -4]
        # Example: ".../mouse_trial_001.mp4" -> "001"
        if pd.notna(path_val):
            s_path = str(path_val).strip()
            if len(s_path) >= 7:
                # Extracts the number directly
                seg_id = s_path[-7:-4] 
                segments_lookup[seg_id] = (row['start_frame'], row['end_frame'])

    print(f"✓ Indexed {len(segments_lookup)} segments.")
    
    calculated_metrics = {} 

    # --- Process Each Trial in TRIALS TYPES ---
    print(f"\nProcessing {len(types_df)} trials from Trials Info file...")
    
    for idx, row in types_df.iterrows():
        # 1. Extract Full ID from trials_types file
        # USER SPECIFIED LOGIC FOR TRIALS: [-13, -4]
        # Example: ".../mouse_trial_001.mp4" -> "trial_001"
        vid_path = row.get('video_segment_path', '')
        full_trial_id = extract_trial_id_from_path(vid_path)
        
        if not full_trial_id:
            print(f"Skipping row {idx}: Could not extract ID from '{vid_path}'")
            continue
            
        # 2. Extract Index to match with Segments
        # "trial_001" -> "001"
        if "_" in full_trial_id:
            trial_index = full_trial_id.split('_')[-1]
        else:
            trial_index = full_trial_id # Fallback if no underscore

        # 3. Find matching frames in Segments Lookup
        if trial_index not in segments_lookup:
            print(f"  {full_trial_id} (Idx: {trial_index}) -> Not found in Segments manifest. Skipping.")
            continue
            
        print(f"  Processing {full_trial_id}...", end=" ")
        
        start_f, end_f = segments_lookup[trial_index]
        
        # 4. Decode State
        state_label = decode_state_sequence(row.get('state_sequence', ''))
        
        # 5. Process Data
        df_trial = process_trial_data(df_full, frames, int(start_f), int(end_f), full_trial_id,
                                      scorer, keypoint, args.min_likelihood, args.fps, 
                                      args.smoothing_window, maze)
                                      
        if df_trial is None or df_trial.empty:
            print(f"No valid data.")
            continue
            
        # 6. Calculate Stats
        avg_speed = df_trial['speed_px_per_frame'].mean()
        max_speed = df_trial['speed_px_per_frame'].max()
        min_speed = df_trial['speed_px_per_frame'].min()
        
        calculated_metrics[idx] = {
            'matched_trial_id': full_trial_id, 
            'avg_speed_px_per_frame': avg_speed,
            'max_speed_px_per_frame': max_speed,
            'min_speed_px_per_frame': min_speed
        }
        
        # 7. Generate Plots
        plot_types = ['time', 'speed']
        if 'compartment' in df_trial.columns:
            plot_types.append('compartment')
            
        for plot_type in plot_types:
            fig = plot_trial_3d(df_trial, full_trial_id, maze, plot_type=plot_type, subtitle=state_label)
            fig = set_common_layout(fig)
            sanitized_state = state_label.replace(" ", "_").replace(",", "-")
            plot_path = outdir / f"{full_trial_id}_{sanitized_state}_{plot_type}.html"
            fig.write_html(str(plot_path))
            
        # Save CSV
        data_path = outdir / f"{full_trial_id}_data.csv"
        df_trial.to_csv(data_path, index=False)
        
        print(f"✓ Done ({state_label})")

    # --- Export Summary ---
    print("\nGenerating summary CSV...")
    metrics_df = pd.DataFrame.from_dict(calculated_metrics, orient='index')
    final_df = types_df.merge(metrics_df, left_index=True, right_index=True, how='left')
    
    summary_path = outdir / "trials_with_metrics_matched.csv"
    final_df.to_csv(summary_path, index=False)
    print(f"✓ Saved matched summary to: {summary_path}")

if __name__ == "__main__":
    main()