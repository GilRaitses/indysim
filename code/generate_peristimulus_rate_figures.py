#!/usr/bin/env python3
"""
Generate peristimulus turn rate and reversal rate figures.

Creates two figure panels for each behavioral event type:
1. 4 side-by-side subplots (one per experimental condition)
2. 1 overlaid plot with all conditions (using distinguishable colors)

Nomenclature follows neuroscience conventions:
- "Turn Rate" (not "hazard rate")
- "Reversal Rate" 
- "Peristimulus" (aligned to LED onset)

Usage:
    python scripts/generate_peristimulus_rate_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
from pathlib import Path
import json

# Distinguishable color palette (colorblind-friendly, includes blue)
CONDITION_COLORS = {
    '0-250PWM_Const': '#e41a1c',      # Red
    '0-250PWM_Ramp': '#377eb8',       # Blue
    '50-250PWM_Const': '#4daf4a',     # Green
    '50-250PWM_Ramp': '#ff7f00',      # Orange
}

# Fallback colors if condition names don't match
FALLBACK_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']


def load_trajectories(data_dir: Path) -> pd.DataFrame:
    """Load trajectory data from consolidated H5 or parquet files."""
    import h5py
    
    # Try consolidated H5 first
    h5_path = data_dir / "consolidated_dataset.h5"
    if h5_path.exists():
        print(f"Loading from consolidated H5: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            # Load from events table (has is_reorientation_start)
            events = f['events']
            cols_needed = ['experiment_id', 'track_id', 'time', 'time_since_stimulus',
                          'is_reorientation_start', 'is_reverse_crawl_start', 
                          'is_turn_start', 'is_pause_start', 'led1Val', 'stimulus_on']
            data = {}
            for col in cols_needed:
                if col in events:
                    arr = events[col][:]
                    if arr.dtype.kind == 'S':  # bytes
                        arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
                    data[col] = arr
            df = pd.DataFrame(data)
        
        # Parse condition from experiment_id
        def parse_condition(expt_id):
            if '50to250' in expt_id:
                intensity = '50-250PWM'
            elif '0to250' in expt_id:
                intensity = '0-250PWM'
            else:
                intensity = 'Unknown'
            if '#C_Bl' in expt_id:
                mode = 'Const'
            elif '#T_Bl' in expt_id or '#T_Re_Sq_5to15' in expt_id:
                mode = 'Ramp'
            else:
                mode = 'Unknown'
            return f'{intensity}_{mode}'
        
        df['condition'] = df['experiment_id'].apply(parse_condition)
        print(f"Loaded {len(df):,} rows from H5 events table")
        return df
    
    # Fall back to parquet files
    traj_files = sorted(data_dir.glob("*_trajectories.parquet"))
    
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files or consolidated H5 in {data_dir}")
    
    print(f"Loading {len(traj_files)} trajectory files...")
    
    all_dfs = []
    for f in traj_files:
        df = pd.read_parquet(f)
        expt_id = f.stem.replace("_trajectories", "")
        df['experiment_id'] = expt_id
        
        # Parse condition from experiment ID
        if '50to250' in expt_id:
            intensity = '50-250PWM'
        elif '0to250' in expt_id:
            intensity = '0-250PWM'
        else:
            intensity = 'Unknown'
        
        if '#C_Bl' in expt_id:
            mode = 'Const'
        elif '#T_Bl' in expt_id or '#T_Re_Sq_5to15' in expt_id:
            mode = 'Ramp'
        else:
            mode = 'Unknown'
        
        df['condition'] = f'{intensity}_{mode}'
        
        if df['condition'].iloc[0] == 'Unknown_Unknown':
            print(f"  WARNING: Could not parse condition from: {expt_id}")
        
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} frames from {len(traj_files)} experiments")
    print(f"Conditions: {combined['condition'].value_counts().to_dict()}")
    
    return combined


def compute_peristimulus_rate(df: pd.DataFrame, event_col: str, 
                               bin_width: float = 0.5, 
                               time_range: tuple = (-5, 20)) -> tuple:
    """
    Compute peristimulus event rate.
    
    Args:
        df: DataFrame with 'time_since_stimulus' and event column
        event_col: Column name for event starts (e.g., 'is_reorientation_start')
        bin_width: Time bin width in seconds
        time_range: (min, max) time relative to stimulus
    
    Returns:
        (bin_centers, rates, sem) - rate in events/second
    """
    if 'time_since_stimulus' not in df.columns:
        raise ValueError("time_since_stimulus column required")
    
    if event_col not in df.columns:
        raise ValueError(f"{event_col} column not found")
    
    # Create time bins
    bins = np.arange(time_range[0], time_range[1] + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # TIME-WRAPPING: Map 25-30s to -5-0s for pre-stimulus bins
    # The stimulus cycle is 30s (10s ON, 20s OFF), so times 25-30s
    # represent the 5 seconds before the next stimulus onset
    df = df.copy()
    df['time_wrapped'] = df['time_since_stimulus'].copy()
    mask_wrap = df['time_since_stimulus'] >= 25
    df.loc[mask_wrap, 'time_wrapped'] = df.loc[mask_wrap, 'time_since_stimulus'] - 30
    
    # Bin data by wrapped time
    df['time_bin'] = pd.cut(df['time_wrapped'], bins=bins)
    
    # Group by time bin, compute rate (events per second)
    # Aggregate by experiment and time bin for SEM calculation
    if 'experiment_id' in df.columns:
        grouped = df.groupby(['experiment_id', 'time_bin'], observed=True).agg({
            event_col: 'sum',
            'time': 'count'  # frame count
        }).reset_index()
        
        # Compute rate per experiment (events / time in bin)
        # Assuming ~20 fps, convert frames to seconds
        fps = 20.0  # approximate
        grouped['time_seconds'] = grouped['time'] / fps
        grouped['rate'] = np.where(
            grouped['time_seconds'] > 0,
            grouped[event_col] / grouped['time_seconds'],
            0
        )
        
        # Aggregate across experiments: mean and SEM
        rate_by_bin = grouped.groupby('time_bin', observed=True)['rate'].agg(['mean', 'std', 'count'])
        rate_by_bin['sem'] = rate_by_bin['std'] / np.sqrt(rate_by_bin['count'])
        
        rates = rate_by_bin['mean'].values
        sems = rate_by_bin['sem'].fillna(0).values
    else:
        # Simple binning without experiment structure
        binned = df.groupby('time_bin', observed=True)[event_col].agg(['sum', 'count'])
        binned['rate'] = binned['sum'] / (binned['count'] / 20.0)  # events/second
        rates = binned['rate'].values
        sems = np.zeros_like(rates)
    
    return bin_centers, rates, sems


def plot_4panel_rate(df: pd.DataFrame, event_col: str, event_name: str,
                     output_path: Path, bin_width: float = 0.5):
    """
    Create 4-panel plot with one subplot per condition.
    """
    conditions = sorted(df['condition'].unique())
    conditions = [c for c in conditions if c != 'Unknown']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, condition in enumerate(conditions[:4]):
        ax = axes[i]
        cond_df = df[df['condition'] == condition]
        
        color = CONDITION_COLORS.get(condition, FALLBACK_COLORS[i])
        
        try:
            bin_centers, rates, sems = compute_peristimulus_rate(
                cond_df, event_col, bin_width=bin_width
            )
            
            # Plot rate with shaded SEM
            ax.plot(bin_centers, rates, color=color, linewidth=2, label=condition)
            ax.fill_between(bin_centers, rates - sems, rates + sems, 
                           color=color, alpha=0.3)
            
        except Exception as e:
            print(f"Warning: Could not compute rate for {condition}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha='center')
        
        # Stimulus onset marker
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvspan(0, 10, color='red', alpha=0.1)  # LED ON period (10s)
        
        # Labels
        ax.set_title(condition.replace('_', ' '), fontsize=12, fontweight='ultralight')
        ax.set_xlabel('Time Since LED Onset (s)', fontsize=10)
        ax.set_ylabel(f'{event_name} Rate (events/s)', fontsize=10)
        ax.set_xlim(-5, 20)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Peristimulus {event_name} Rate by Condition', fontsize=14, fontweight='ultralight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_overlaid_rate(df: pd.DataFrame, event_col: str, event_name: str,
                       output_path: Path, bin_width: float = 0.5):
    """
    Create single plot with all conditions overlaid using distinguishable colors.
    """
    conditions = sorted(df['condition'].unique())
    conditions = [c for c in conditions if c != 'Unknown']
    
    fig, ax = plt.subplots(figsize=(6, 5))  # Match aspect ratio of individual 4-panel plots
    
    for i, condition in enumerate(conditions[:4]):
        cond_df = df[df['condition'] == condition]
        
        color = CONDITION_COLORS.get(condition, FALLBACK_COLORS[i])
        
        try:
            bin_centers, rates, sems = compute_peristimulus_rate(
                cond_df, event_col, bin_width=bin_width
            )
            
            # Plot with distinct line styles for extra distinguishability
            line_styles = ['-', '--', '-.', ':']
            ax.plot(bin_centers, rates, color=color, linewidth=2.5, 
                   linestyle=line_styles[i % 4],
                   label=condition.replace('_', ' '))
            ax.fill_between(bin_centers, rates - sems, rates + sems, 
                           color=color, alpha=0.15)
            
        except Exception as e:
            print(f"Warning: Could not compute rate for {condition}: {e}")
    
    # Stimulus onset marker
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='LED onset')
    ax.axvspan(0, 10, color='red', alpha=0.08)  # LED ON period (10s)
    
    # Labels
    ax.set_xlabel('Time Since LED Onset (s)', fontsize=12)
    ax.set_ylabel(f'{event_name} Rate (events/s)', fontsize=12)
    ax.set_title(f'Peristimulus {event_name} Rate', fontsize=14, fontweight='ultralight')
    ax.set_xlim(-5, 20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Peristimulus Rate Figures")
    print("=" * 60)
    
    # Find data
    data_dir = Path("data/processed_with_reversals")
    if not data_dir.exists():
        data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1
    
    # Load trajectories
    df = load_trajectories(data_dir)
    
    # Output directory
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    
    # =================================================================
    # TURN RATE FIGURES
    # =================================================================
    print("\n--- Turn Rate Figures ---")
    
    # Determine turn event column - need to derive onset from frame flags
    turn_col = None
    for col in ['is_reorientation_start', 'is_turn_start']:
        if col in df.columns and df[col].sum() > 0:
            turn_col = col
            break
    
    # If no onset column, derive it from is_reorientation
    if turn_col is None and 'is_reorientation' in df.columns:
        print("Deriving turn onsets from is_reorientation frame flags...")
        # Onset is where is_reorientation goes from False to True
        df = df.sort_values(['experiment_id', 'frame']).copy()
        df['is_reorientation_start'] = (
            df['is_reorientation'] & 
            ~df.groupby('experiment_id')['is_reorientation'].shift(1, fill_value=False)
        )
        turn_col = 'is_reorientation_start'
    
    if turn_col:
        print(f"Using turn column: {turn_col} ({df[turn_col].sum()} events)")
        
        # 4-panel figure
        plot_4panel_rate(
            df, turn_col, 'Turn',
            fig_dir / "peristimulus_turn_rate_4panel.png"
        )
        
        # Overlaid figure
        plot_overlaid_rate(
            df, turn_col, 'Turn',
            fig_dir / "peristimulus_turn_rate_overlaid.png"
        )
    else:
        print("Warning: No turn event column found")
    
    # =================================================================
    # REVERSAL RATE FIGURES
    # =================================================================
    print("\n--- Reversal Rate Figures ---")
    
    reversal_col = None
    for col in ['is_reverse_crawl_start', 'is_reverse_crawl']:
        if col in df.columns and df[col].sum() > 0:
            reversal_col = col
            break
    
    if reversal_col:
        print(f"Using reversal column: {reversal_col} ({df[reversal_col].sum()} events)")
        
        # 4-panel figure
        plot_4panel_rate(
            df, reversal_col, 'Reversal',
            fig_dir / "peristimulus_reversal_rate_4panel.png"
        )
        
        # Overlaid figure
        plot_overlaid_rate(
            df, reversal_col, 'Reversal',
            fig_dir / "peristimulus_reversal_rate_overlaid.png"
        )
    else:
        print("Warning: No reversal event column found")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())





