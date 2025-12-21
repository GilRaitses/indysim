#!/usr/bin/env python3
"""
Generate fractional behavior by pulse figure for supplement.

Creates a 2x2 grid showing behavioral state fractions (Run, Pause, Turn, Reverse Crawl)
across pulses for each experimental condition.

Usage:
    python scripts/generate_fractional_behavior_figure.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Set font
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'

# Behavioral state colors
STATE_COLORS = {
    'CONTINUATION (Run)': '#808080',  # Gray
    'PAUSE': '#1f77b4',               # Blue
    'TURN': '#2ca02c',                # Green
    'REVERSE CRAWL': '#ff7f0e'        # Orange
}

# Condition patterns - using ASCII "to" instead of Unicode arrow
CONDITION_LABELS = {
    '0-250PWM_Const': '0-to-250 | Constant',
    '0-250PWM_Ramp': '0-to-250 | Cycling',
    '50-250PWM_Const': '50-to-250 | Constant',
    '50-250PWM_Ramp': '50-to-250 | Cycling',
}


def parse_condition(experiment_id: str) -> str:
    """Parse condition from experiment ID."""
    if '50to250' in experiment_id:
        intensity = '50-250PWM'
    elif '0to250' in experiment_id:
        intensity = '0-250PWM'
    else:
        return 'Unknown'
    
    if '#C_Bl' in experiment_id:
        mode = 'Const'
    elif '#T_Bl' in experiment_id or '#T_Re_Sq' in experiment_id:
        mode = 'Ramp'
    else:
        mode = 'Unknown'
    
    return f'{intensity}_{mode}'


def load_behavioral_data(h5_path: Path) -> pd.DataFrame:
    """Load behavioral state data from consolidated H5 trajectories table."""
    with h5py.File(h5_path, 'r') as f:
        # Use trajectories table (has proper behavioral state flags)
        traj = f['trajectories']
        
        # Load needed columns
        # NOTE: is_turn = forward crawling/continuation (confusing name from MAGAT)
        #       is_turn_simple = actual turning behavior (broader than is_reorientation)
        #       is_reorientation = turn onset frames only (too strict)
        cols = ['experiment_id', 'track_id', 'time', 'time_since_stimulus',
                'is_turn', 'is_pause', 'is_turn_simple', 'is_reverse_crawl']
        
        data = {}
        for col in cols:
            if col in traj:
                arr = traj[col][:]
                if arr.dtype.kind == 'S':
                    arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
                data[col] = arr
        
        df = pd.DataFrame(data)
    
    # Rename is_turn to is_continuation for clarity
    if 'is_turn' in df.columns:
        df['is_continuation'] = df['is_turn']
        df = df.drop(columns=['is_turn'])
    
    # Parse condition
    df['condition'] = df['experiment_id'].apply(parse_condition)
    
    # Compute pulse number (30s cycles: 10s ON, 20s OFF)
    df['pulse_number'] = (df['time'] // 30).astype(int)
    
    return df


def compute_fractional_behavior(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Compute fractional behavior by pulse for a condition."""
    cond_df = df[df['condition'] == condition].copy()
    
    if len(cond_df) == 0:
        return pd.DataFrame()
    
    # Group by pulse and compute fractions
    results = []
    for pulse in sorted(cond_df['pulse_number'].unique()):
        pulse_df = cond_df[cond_df['pulse_number'] == pulse]
        total_frames = len(pulse_df)
        
        if total_frames == 0:
            continue
        
        # Compute fractions using proper behavioral state columns
        # is_continuation = forward crawling (formerly is_turn in MAGAT)
        # is_turn_simple = actual turning behavior (14% of time)
        cont_frac = pulse_df['is_continuation'].sum() / total_frames if 'is_continuation' in pulse_df else 0
        pause_frac = pulse_df['is_pause'].sum() / total_frames if 'is_pause' in pulse_df else 0
        turn_frac = pulse_df['is_turn_simple'].sum() / total_frames if 'is_turn_simple' in pulse_df else 0
        rev_frac = pulse_df['is_reverse_crawl'].sum() / total_frames if 'is_reverse_crawl' in pulse_df else 0
        
        # Normalize to sum to 1 (handle any missing states)
        total = cont_frac + pause_frac + turn_frac + rev_frac
        if total > 0:
            cont_frac /= total
            pause_frac /= total
            turn_frac /= total
            rev_frac /= total
        
        results.append({
            'pulse': pulse,
            'CONTINUATION (Run)': cont_frac,
            'PAUSE': pause_frac,
            'TURN': turn_frac,
            'REVERSE CRAWL': rev_frac
        })
    
    return pd.DataFrame(results)


def plot_fractional_behavior(df: pd.DataFrame, output_path: Path):
    """Create 2x2 fractional behavior plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    conditions = ['0-250PWM_Const', '0-250PWM_Ramp', '50-250PWM_Const', '50-250PWM_Ramp']
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        frac_df = compute_fractional_behavior(df, condition)
        
        if len(frac_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Limit to first 18 pulses (for visibility)
        frac_df = frac_df[frac_df['pulse'] <= 17]
        
        pulses = frac_df['pulse'].values
        
        # Stack bar chart
        bottom = np.zeros(len(pulses))
        for state in ['CONTINUATION (Run)', 'PAUSE', 'TURN', 'REVERSE CRAWL']:
            values = frac_df[state].values
            ax.bar(pulses, values, bottom=bottom, color=STATE_COLORS[state], 
                   label=state if i == 0 else '', width=0.8)
            bottom += values
        
        # Labels
        title = CONDITION_LABELS.get(condition, condition)
        ax.set_title(title, fontsize=12, fontweight='ultralight')
        ax.set_xlabel('Pulse number', fontsize=10)
        ax.set_ylabel('Fractional behavior usage', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 17.5)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
    
    # Add legend to first subplot
    axes[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.3), ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Generating fractional behavior by pulse figure...")
    
    # Find consolidated dataset
    data_dir = Path("data/processed")
    h5_path = data_dir / "consolidated_dataset.h5"
    
    if not h5_path.exists():
        data_dir = Path("data/processed_with_reversals")
        h5_path = data_dir / "consolidated_dataset.h5"
    
    if not h5_path.exists():
        print(f"ERROR: Consolidated dataset not found")
        return 1
    
    # Load data
    df = load_behavioral_data(h5_path)
    print(f"Loaded {len(df):,} frames from {df['experiment_id'].nunique()} experiments")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    
    # Generate figure
    output_path = Path('figures') / 'fractional_behavior_by_pulse.png'
    output_path.parent.mkdir(exist_ok=True)
    plot_fractional_behavior(df, output_path)
    
    return 0


if __name__ == '__main__':
    exit(main())

