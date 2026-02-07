import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, mannwhitneyu
from analysisfunc_config import Paths

#  1. CONFIGURATION 
MOUSE_BASE_PATH = Paths.base_path 
SESSIONS = [
    "2024-08-28_11_58_146357session3.6",
    "2024-08-29_10_23_026357session3.7",
    "2024-08-30_10_07_556357session3.8"
]

RUN_GAUSSIAN_SPEED    = True
RUN_GAUSSIAN_DURATION = True
RUN_STATISTICAL_TESTS = True
SAVE_POOLED_CSV       = True

OUTPUT_DIR = os.path.join(MOUSE_BASE_PATH, "MOUSE_6357_TOTAL_ANALYSIS")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

#  2. CONCATENATION & COLUMN UNIFICATION
all_data = []

print("Searching for master CSVs and unifying columns...")
for session_id in SESSIONS:
    # Look for both naming conventions to ensure backward compatibility
    search_paths = [
        # os.path.join(MOUSE_BASE_PATH, session_id, "total_analysis_output", "master_behavioral_data.csv"),
        os.path.join(MOUSE_BASE_PATH, session_id, "total_analysis_output", "master_behavioural_data.csv")
    ]
    
    found = False
    for path in search_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # FIX: Unify older 'cm_s' columns into the current standard
            rename_map = {
                'p1_mean_speed_cm_s': 'p1_mean_speed',
                'p2_mean_speed_cm_s': 'p2_mean_speed'
            }
            df = df.rename(columns=rename_map)
            
            # Ensure session tracking
            df['session_id'] = session_id[-3:] 
            all_data.append(df)
            print(f"  [FOUND] {session_id}")
            found = True
            break
    
    if not found:
        print(f"  [MISSING] {session_id}")

if not all_data:
    print("Error: No data found. Ensure individual sessions are processed first.")
    exit()

# Combine all sessions into one master dataframe
df_total = pd.concat(all_data, ignore_index=True)

if SAVE_POOLED_CSV:
    df_total.to_csv(os.path.join(OUTPUT_DIR, "mouse_6357_massive_pooled_data.csv"), index=False)
    print(f"Concatenated CSV saved: {len(df_total)} total trials.")

#  3. GLOBAL GAUSSIAN PLOTTING 
def plot_pooled_gaussians(df, metric_suffix, title, unit, filename):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"P1 {title}", f"P2 {title}"), shared_yaxes=True)
    styles = {
        'Hit':  {'line': 'rgba(0, 128, 0, 1)',   'fill': 'rgba(0, 128, 0, 0.1)'},
        'Miss': {'line': 'rgba(255, 0, 0, 1)',   'fill': 'rgba(255, 0, 0, 0.1)'}
    }

    for i, phase in enumerate(['p1', 'p2']):
        col = f"{phase}_{metric_suffix}"
        if col not in df.columns: 
            print(f"Warning: Column {col} not found for Gaussian plot.")
            continue
        
        valid_subset = df[col].dropna()
        if valid_subset.empty: continue
        
        # Calculate X range based on the data spread
        x_range = np.linspace(0, valid_subset.max() * 1.3, 1000)
        
        for status in ['Hit', 'Miss']:
            subset = df[df['status'] == status][col].dropna()
            if len(subset) < 2: continue
            
            mu, sigma = subset.mean(), subset.std()
            if sigma == 0 or np.isnan(sigma): sigma = 0.1
            
            y = norm.pdf(x_range, mu, sigma)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y,
                name=f"{status} (μ={mu:.2f} {unit})",
                line=dict(color=styles[status]['line'], width=4),
                fill='tozeroy', 
                fillcolor=styles[status]['fill'],
                legendgroup=status,
                showlegend=(i == 0)
            ), row=1, col=i+1)
            
    fig.update_layout(title=f"Global {title} Distributions: Mouse 6357 (Pooled)", template="plotly_white")
    fig.write_html(os.path.join(OUTPUT_DIR, filename))

if RUN_GAUSSIAN_SPEED:
    plot_pooled_gaussians(df_total, 'mean_speed', 'Speed', 'cm/s', 'global_speed_gaussians.html')

if RUN_GAUSSIAN_DURATION:
    plot_pooled_gaussians(df_total, 'duration_s', 'Duration', 's', 'global_duration_gaussians.html')

#  4. STATISTICAL TESTS 
if RUN_STATISTICAL_TESTS:
    print("Running Global Statistics including Speed...")
    
    # These names must match the Unified names in the rename_map above
    metrics = ['p1_duration_s', 'p1_mean_speed', 'p1_entropy', 
               'p2_duration_s', 'p2_mean_speed', 'p2_entropy']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    stats_summary = []
    
    for i, m in enumerate(metrics):
        if m not in df_total.columns: 
            print(f"Warning: Metric '{m}' not in data.")
            continue
        
        # Plotting distributions with jittered raw points
        sns.boxplot(data=df_total, x='status', y=m, ax=axes[i], 
                    palette={'Hit': 'green', 'Miss': 'red'}, hue='status', legend=False)
        sns.stripplot(data=df_total, x='status', y=m, ax=axes[i], color='black', alpha=0.3)
        
        # Statistical Comparison (Mann-Whitney U)
        h = df_total[df_total['status'] == 'Hit'][m].dropna()
        ms = df_total[df_total['status'] == 'Miss'][m].dropna()
        
        if not h.empty and not ms.empty:
            stat, p = mannwhitneyu(h, ms)
            # Define significance level stars
            sig = "NS" if p >= 0.05 else ("*" if p < 0.05 else "**" if p < 0.01 else "***")
            if p < 0.001: sig = "***"
            
            axes[i].set_title(f"{m}\np={p:.4f} ({sig})")
            stats_summary.append({'Metric': m, 'p-value': p, 'significance': sig})

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "global_pooled_stats.png"))
    
    # Save a specific report for the p-values
    pd.DataFrame(stats_summary).to_csv(os.path.join(OUTPUT_DIR, "global_stats_report.csv"), index=False)

print(f"Done! All pooled files saved in: {OUTPUT_DIR}")