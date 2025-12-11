"""
Batch-process DeepLabCut data to generate per-trial 3D (x, y, time)
interactive trajectory plots using Plotly, integrating Trial Type metadata.

Features:
- Reads DLC .csv, segments .csv, and a TRIAL INFO .csv.
- Decodes 'state_sequence' (1=Explore, 2=Exploit, 3=Nest, 4=Groom) for plot titles.
- Calculates Avg/Max/Min speed per trial.
- Saves interactive 3D .html plots.
- Exports a summary CSV containing the original trial info + calculated speed metrics.

Example Usage:
----------------
python make_3d_dlc_plot_trajectories_v2.py ^
    --dlc "C:/Path/To/DLC_coords.csv" ^
    --segments "C:/Path/To/segments_manifest.csv" ^
    --trials-types "C:/Path/To/mouse6357_session3.6_trial_info.csv" ^
    --keypoint "nose" ^
    --maze-geometry "C:/Path/To/maze_structure.json" ^
    --output-dir "./3d_trajectory_plots" ^
    --min-likelihood 0.9 ^
    --fps 30



python make_3d_dlc_plot_trajectories.py --dlc "C:/Users/shahd/OneDrive - University of Sussex/DLC_MOSEQ/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv" --segments "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_segments_manifest.csv" --trials-types "C:/Path/To/mouse6357_session3.6_trial_info.csv" --keypoint nose --output-dir "./trajectories_maze_video_nose" --min-likelihood 0.8 --fps 30"


"""

import argparse
import json
import os
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
    Handles single ints (3) or strings ("1,4,1").
    """
    if pd.isna(value):
        return "Unknown"
    
    mapping = {
        '1': 'Explore',
        '2': 'Exploit',
        '3': 'Nest',
        '4': 'Groom'
    }
    
    # Convert input to string and remove quotes if pandas loaded them weirdly
    s_val = str(value).replace('"', '').replace("'", "")
    
    # Split by comma if multiple states exist
    parts = [p.strip() for p in s_val.split(',')]
    
    # Decode each part
    decoded_parts = []
    for p in parts:
        # If float-like string "1.0", convert to "1"
        if p.replace('.','',1).isdigit():
            p = str(int(float(p)))
        decoded_parts.append(mapping.get(p, p)) # Default to original if not in map
        
    return "-".join(decoded_parts)

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
                  subtitle: str = "") -> go.Figure:
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
        
        # This is a special case, so we return the fig early
        title_text = f'Trial {trial_id} - {subtitle}' if subtitle else f'Trial {trial_id}'
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

    title_text = f'Trial {trial_id} - {subtitle}' if subtitle else f'Trial {trial_id}'
    fig.update_layout(title=f'{title_text} (Colored by {plot_type.capitalize()})')
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
    ap.add_argument("--trials-types", default=None, help="Path to CSV containing trial info (state_sequence, etc).")
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
    
    if args.smoothing_window % 2 == 0:
        args.smoothing_window += 1
        
    print("Starting batch 3D trajectory analysis...")
    print(f"  DLC File: {args.dlc}")
    print(f"  Segments File: {args.segments}")
    if args.trials_types:
        print(f"  Trials Info File: {args.trials_types}")
    print(f"  Output Dir: {outdir.resolve()}")
    
    # --- 2. Load Data Files ---
    try:
        df_full = read_dlc_csv_multiindex(args.dlc)
        seg_df = pd.read_csv(args.segments)
        
        # Load Trial Types if provided
        types_df = None
        if args.trials_types:
            types_df = pd.read_csv(args.trials_types)
            print(f"✓ Loaded Trial Types ({len(types_df)} rows)")
            
    except Exception as e:
        print(f"Fatal Error: Could not load input files. {e}")
        return

    scorer, bodyparts = get_scorer_and_bodyparts(df_full)
    frames = extract_frame_series(df_full)
    keypoint = pick_keypoint(bodyparts, args.keypoint)
    maze = load_maze_geometry(args.maze_geometry)
    
    print(f"✓ Using scorer='{scorer}', keypoint='{keypoint}'")
    
    # Normalize segment column names
    seg_df.columns = seg_df.columns.str.lower().str.strip()
    if 'trial_index' not in seg_df.columns:
        seg_df['trial_index'] = seg_df.index
        
    if args.limit:
        seg_df = seg_df.head(args.limit)
        print(f"--- Limiting analysis to first {args.limit} trials ---")

    # Store calculated metrics for the summary CSV
    # We use a dictionary keyed by original index to ensure alignment later
    calculated_metrics = {} 

    # --- 3. Process Each Trial ---
    print(f"\nProcessing {len(seg_df)} trials...")
    for idx, trial_row in seg_df.iterrows():
        trial_id = trial_row['trial_index']
        print(f"  Processing Trial {trial_id}...", end=" ")
        
        # Get Trial Type Info (State Sequence)
        state_label = "Unknown"
        if types_df is not None:
            # Assume row alignment between segments and types_df
            # If indices don't match exactly 0-to-0, ensure your CSVs are aligned beforehand.
            if idx < len(types_df):
                raw_state = types_df.iloc[idx].get('state_sequence', np.nan)
                state_label = decode_state_sequence(raw_state)
            else:
                print("(Warning: Index out of bounds in trials-types CSV)", end=" ")

        # Process the data
        df_trial = process_trial_data(df_full, frames, trial_row, scorer, keypoint,
                                      args.min_likelihood, args.fps, 
                                      args.smoothing_window, maze)
        
        if df_trial is None or df_trial.empty:
            print(f"Skipped (no data).")
            continue
            
        # Calculate Stats
        avg_speed = df_trial['speed_px_per_frame'].mean()
        max_speed = df_trial['speed_px_per_frame'].max()
        min_speed = df_trial['speed_px_per_frame'].min()
        
        # Store metrics (keyed by the index of the segment dataframe)
        calculated_metrics[idx] = {
            'avg_speed_px_per_frame': avg_speed,
            'max_speed_px_per_frame': max_speed,
            'min_speed_px_per_frame': min_speed
        }

        # --- Generate Plots ---
        plot_types = ['time', 'speed']
        if 'compartment' in df_trial.columns:
            plot_types.append('compartment')
            
        for plot_type in plot_types:
            fig = plot_trial_3d(df_trial, trial_id, maze, plot_type=plot_type, subtitle=state_label)
            fig = set_common_layout(fig)
            
            # Save plot as HTML
            # Filename includes state label for easy searching
            sanitized_state = state_label.replace(" ", "_").replace(",", "-")
            plot_path = outdir / f"trial_{trial_id}_{sanitized_state}_{plot_type}.html"
            fig.write_html(str(plot_path))
            
        # Save per-trial raw data
        data_path = outdir / f"trial_{trial_id}_processed_data.csv"
        df_trial.to_csv(data_path, index=False)
        
        print(f"✓ Done ({state_label})")

    # --- 4. Export Summary CSV ---
    if types_df is not None:
        print("\nGenering summary CSV with speed metrics...")
        
        # Create a DataFrame from the calculated metrics
        metrics_df = pd.DataFrame.from_dict(calculated_metrics, orient='index')
        
        # Merge with the original types_df
        # We merge on index to ensure rows align
        final_df = types_df.merge(metrics_df, left_index=True, right_index=True, how='left')
        
        summary_path = outdir / "updated_trial_info_with_metrics.csv"
        final_df.to_csv(summary_path, index=False)
        print(f"✓ Saved updated trial info to: {summary_path}")

    print("\nBatch analysis complete!")


if __name__ == "__main__":
    main()