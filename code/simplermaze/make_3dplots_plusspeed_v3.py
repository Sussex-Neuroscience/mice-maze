"""
Batch-process DeepLabCut data to generate per-trial 3D (x, y, time)
interactive trajectory plots using Plotly.

Modified to include trial type information (state sequence) and speed statistics.

Features:
- Reads a single main DLC .csv file and a segments_manifest .csv.
- Reads an optional 'trials_types' .csv containing 'state_sequence' (1=explore, 2=exploit, etc.).
- Loops through each trial.
- Filters by likelihood.
- Performs optional smoothing (median filter).
- Calculates speed and (optional) maze compartment per frame.
- Saves one interactive 3D .html plot for each trial (with state info in title/filename).
- Saves one processed .csv data file for each trial.
- Outputs an updated trials_types .csv with added average/max/min speed columns.

Example Usage:
----------------
python make_3d_dlc_plot_trajectories.py ^
    --dlc "path/to/dlc.csv" ^
    --segments "path/to/segments.csv" ^
    --trials-types "path/to/mouse6357_session3.6_trial_info.csv" ^
    --keypoint "nose" ^
    --maze-geometry "path/to/maze.json" ^
    --output-dir "./3d_trajectory_plots"
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import medfilt
import cv2  # For pointPolygonTest

# ----------------------------------------------------------------------------
# 1. UTILITY FUNCTIONS (Loading, Parsing, Analysis)
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

def parse_state_sequence(seq_val) -> str:
    """
    Parses the state_sequence column value (e.g., '1,4,1') into a readable string
    (e.g., 'explore-groom-explore').
    Mapping: 1=explore, 2=exploit, 3=nest, 4=groom
    """
    if pd.isna(seq_val) or seq_val == "":
        return "Unknown"
    
    # State mapping dictionary
    mapping = {'1': 'explore', '2': 'exploit', '3': 'nest', '4': 'groom'}
    
    # Convert to string and split by comma
    parts = str(seq_val).split(',')
    
    labels = []
    for p in parts:
        p_clean = p.strip()
        # Handle cases like '1.0' if reading as float string
        if p_clean.endswith('.0'):
            p_clean = p_clean[:-2]
            
        labels.append(mapping.get(p_clean, p_clean))
        
    return "-".join(labels)

def process_trial_data(df_full: pd.DataFrame, frames: pd.Series, trial: pd.Series, 
                         scorer: str, keypoint: str, min_likelihood: float, 
                         fps: float, smooth_window: int, maze_structure: Optional[Dict]) -> Optional[pd.DataFrame]:
    """
    Slices, filters, and processes all data for a single trial.
    """
    try:
        start_frame = int(trial['start_frame'])
        end_frame = int(trial['end_frame'])
    except KeyError:
        print(f"Error: Segments file missing 'start_frame' or 'end_frame'. Found: {list(trial.index)}")
        return None
    except ValueError:
        print(f"Error: Could not read start/end frame for trial {trial.get('trial_index', 'N/A')}")
        return None

    # 1. Slice the full DataFrame by frame index
    mask = (frames >= start_frame) & (frames <= end_frame)
    if mask.sum() == 0:
        print(f"Warning: No frames found for trial {trial.get('trial_index')} (frames {start_frame}-{end_frame}). Skipping.")
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
        print(f"Warning: Trial {trial.get('trial_index')} contains persistent NaN values after interpolation. Skipping.")
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
                  state_label: str = "") -> go.Figure:
    """
    Creates the interactive 3D Plotly figure for a single trial.
    """
    
    fig = go.Figure()
    
    # --- Determine coloring ---
    if plot_type == 'speed' and 'speed_px_per_frame' in df_trial.columns:
        color_data = df_trial['speed_px_per_frame']
        colorscale = 'Jet'
        colorbar_title = "Speed (px/frame)"
        hover_text = "Speed: %{line.color:.2f}<br>"
    elif plot_type == 'compartment' and 'compartment' in df_trial.columns:
        # Use Plotly Express to easily get categorical colors
        unique_comps = df_trial['compartment'].unique()
        color_map = {comp: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, comp in enumerate(unique_comps)}
        df_trial['color'] = df_trial['compartment'].map(color_map)
        
        # Add a trace for each compartment
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
        
        # Build Title
        title_text = f'Trial {trial_id}'
        if state_label:
            title_text += f' - {state_label}'
        title_text += ' - 3D Trajectory by Compartment'
        
        fig.update_layout(title=title_text)
        return fig
        
    else: # Default to 'time'
        color_data = df_trial['time_seconds']
        colorscale = 'Viridis'
        colorbar_title = "Time (s)"
        hover_text = ""

    # --- Add Main Trajectory Line (for 'time' and 'speed' plots) ---
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
                      'X: %{x:.1f}<br>' +
                      'Y: %{y:.1f}<br>' +
                      'Time: %{z:.2f}s<br>' +
                      hover_text +
                      '<extra></extra>'
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

    # --- Add Maze Structure at Base (Z=0) ---
    if maze_structure is not None:
        for comp in maze_structure.get('compartments', []):
            contour = np.array(comp['contour'])
            contour_closed = np.vstack([contour, contour[0]]) # Close the loop
            
            fig.add_trace(go.Scatter3d(
                x=contour_closed[:, 0],
                y=contour_closed[:, 1],
                z=np.zeros(len(contour_closed)),
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # --- Add Vertical Projection Lines (Sampled) ---
    sample_rate = max(1, len(df_trial) // 50)  # Show ~50 lines
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

    # Build Title
    title_text = f'Trial {trial_id}'
    if state_label:
        title_text += f' - {state_label}'
    title_text += f' - 3D Trajectory (Colored by {plot_type.capitalize()})'

    fig.update_layout(title=title_text)
    return fig

def set_common_layout(fig: go.Figure) -> go.Figure:
    """Applies the common 3D layout to a Plotly figure."""
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X Position (pixels)', showbackground=True, backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Y Position (pixels)', showbackground=True, backgroundcolor="rgb(230, 230,230)", autorange='reversed'), # Invert Y
            zaxis=dict(title='Time (seconds)', showbackground=True, backgroundcolor="rgb(230, 230,230)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5) # Z-axis is compressed
        ),
        width=1000,
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest'
    )
    return fig

# ----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch-process DLC data into 3D interactive plots.")
    ap.add_argument("--dlc", required=True, help="Path to DLC CSV with 3-row header.")
    ap.add_argument("--segments", required=True, help="Path to segments_manifest CSV with trial_index, start_frame, end_frame.")
    ap.add_argument("--trials-types", default=None, help="Optional: Path to trial info CSV with 'state_sequence' for naming and stats.")
    ap.add_argument("--keypoint", required=True, help="Bodypart to plot (e.g., 'nose').")
    ap.add_argument("--output-dir", required=True, help="Directory to save HTML plots and trial CSVs.")
    ap.add_argument("--maze-geometry", default=None, help="Optional: JSON file with maze compartment definitions.")
    ap.add_argument("--min-likelihood", type=float, default=0.9, help="Minimum DLC likelihood to include a point.")
    ap.add_argument("--fps", type=float, default=30.0, help="Video frame rate for time/speed calculations.")
    ap.add_argument("--smoothing-window", type=int, default=5, help="Kernel size for median filter (odd number). 1 to disable.")
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N trials (for testing).")
    
    args = ap.parse_args()

    # --- 1. Setup Environment ---
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Ensure smoothing window is odd
    if args.smoothing_window % 2 == 0:
        args.smoothing_window += 1
        
    print("Starting batch 3D trajectory analysis...")
    print(f"  DLC File: {args.dlc}")
    print(f"  Segments File: {args.segments}")
    print(f"  Output Dir: {outdir.resolve()}")
    
    # --- 2. Load Main Data Files ---
    try:
        df_full = read_dlc_csv_multiindex(args.dlc)
        seg_df = pd.read_csv(args.segments)
    except Exception as e:
        print(f"Fatal Error: Could not load input files. {e}")
        return

    # --- 3. Load Trials Types (If provided) ---
    df_types = None
    if args.trials_types:
        try:
            df_types = pd.read_csv(args.trials_types)
            # Initialize stat columns with NaN
            df_types['avg_speed'] = np.nan
            df_types['max_speed'] = np.nan
            df_types['min_speed'] = np.nan
            print(f"✓ Loaded trials types: {args.trials_types}")
        except Exception as e:
            print(f"Error loading trials types file: {e}")
            return

    scorer, bodyparts = get_scorer_and_bodyparts(df_full)
    frames = extract_frame_series(df_full)
    keypoint = pick_keypoint(bodyparts, args.keypoint)
    maze = load_maze_geometry(args.maze_geometry)
    
    print(f"✓ Using scorer='{scorer}', keypoint='{keypoint}'")
    
    # Normalize segment column names
    seg_df.columns = seg_df.columns.str.lower().str.strip()
    if 'trial_index' not in seg_df.columns:
        print("Warning: 'trial_index' not found. Using row number as trial ID.")
        seg_df['trial_index'] = seg_df.index
        
    if args.limit:
        seg_df = seg_df.head(args.limit)
        print(f"--- Limiting analysis to first {args.limit} trials ---")

    # --- 4. Process Each Trial ---
    print(f"\nProcessing {len(seg_df)} trials...")
    
    # Using enumerate to get the integer index corresponding to rows in df_types
    for idx, trial_row in seg_df.iterrows():
        # idx is the original index of seg_df. 
        # We assume seg_df and df_types are aligned by row index.
        
        trial_id = trial_row['trial_index']
        print(f"  Processing Trial {trial_id}...")
        
        # --- Retrieve State Label ---
        state_label = ""
        # Check if we have valid trial types data for this index
        if df_types is not None:
            if idx in df_types.index:
                raw_state = df_types.loc[idx, 'state_sequence']
                state_label = parse_state_sequence(raw_state)
            else:
                state_label = "Unknown-Index"

        # --- Process Data ---
        df_trial = process_trial_data(df_full, frames, trial_row, scorer, keypoint,
                                      args.min_likelihood, args.fps, 
                                      args.smoothing_window, maze)
        
        if df_trial is None or df_trial.empty:
            print(f"    ...Skipped Trial {trial_id} (no valid data).")
            continue
            
        # --- Calculate Statistics and Update df_types ---
        if df_types is not None and idx in df_types.index:
            # We ignore the first value (0) for diff calculation usually, 
            # or include it. df_trial['speed_px_per_frame'] has 0 at start.
            # Using mean of all frames including start 0 might slightly lower avg.
            # Let's exclude the very first 0 frame if desired, but here we take simple stats.
            speeds = df_trial['speed_px_per_frame']
            
            df_types.loc[idx, 'avg_speed'] = speeds.mean()
            df_types.loc[idx, 'max_speed'] = speeds.max()
            df_types.loc[idx, 'min_speed'] = speeds.min()

        # --- Save Trial Data CSV ---
        # Filename sanity check (remove invalid chars from state_label)
        safe_label = re.sub(r'[^\w\-]', '_', state_label) if state_label else ""
        label_suffix = f"_{safe_label}" if safe_label else ""
        
        data_path = outdir / f"trial_{trial_id}{label_suffix}_processed_data.csv"
        df_trial.to_csv(data_path, index=False)
        
        # --- Generate and Save Plots ---
        plot_types = ['time', 'speed']
        if 'compartment' in df_trial.columns:
            plot_types.append('compartment')
            
        for plot_type in plot_types:
            fig = plot_trial_3d(df_trial, trial_id, maze, plot_type=plot_type, state_label=state_label)
            fig = set_common_layout(fig)
            
            # Save plot as HTML
            plot_path = outdir / f"trial_{trial_id}{label_suffix}_3d_plot_{plot_type}.html"
            fig.write_html(str(plot_path))
            
        print(f"    ✓ Saved data and plots for Trial {trial_id} ({state_label})")

    # --- 5. Save Updated Trial Info CSV ---
    if df_types is not None:
        stats_path = outdir / "trial_stats_with_types.csv"
        df_types.to_csv(stats_path, index=False)
        print(f"\n✓ Saved updated trial stats with speeds to: {stats_path}")

    print("\nBatch analysis complete!")


if __name__ == "__main__":
    main()