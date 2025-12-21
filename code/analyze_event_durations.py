#!/usr/bin/env python3
"""
Analyze event durations for phenotype identification.

This is a downstream analysis script that creates visualizations and 
statistical summaries of behavioral event durations across conditions.

Data Sources:
    - klein_run_table.py outputs: Run durations (column 'runT')
    - detect_events.py outputs: Pause and turn durations
    - trajectory data: Reverse crawl segments (is_reverse_crawl flag)

Dependencies:
    - Consolidated HDF5 file must exist at: data/processed/consolidated_dataset.h5
    - OR individual parquet files in: data/data/processed_with_reversals/

Outputs:
    - figures/event_durations.png (4-panel boxplot figure)
    - data/model/event_duration_summary.json (summary statistics)

Statistical Methods:
    - Kruskal-Wallis H-test for non-parametric comparison across conditions
    - Dunn's post-hoc tests (Mann-Whitney U with Bonferroni correction) when omnibus is significant
    - Reports: median, IQR, n_events per condition, pairwise significance

Usage:
    python scripts/analyze_event_durations.py

Author: INDYsim pipeline
Date: 2025-12-13
"""

from typing import Dict, List, Tuple, Optional, Union

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
from pathlib import Path
import h5py
from scipy import stats

# Condition colors
COLORS = {
    '0→250 | Constant': '#e41a1c',
    '0→250 | Cycling': '#377eb8',
    '50→250 | Constant': '#4daf4a',
    '50→250 | Cycling': '#ff7f00'
}


def parse_condition(experiment_id: Union[str, bytes]) -> str:
    """Parse experiment ID to condition label.
    
    Args:
        experiment_id: Experiment identifier string (may be bytes from HDF5)
        
    Returns:
        Condition label in format 'intensity | mode', e.g. '0→250 | Constant'
        Returns 'Unknown' if pattern not recognized.
    """
    if isinstance(experiment_id, bytes):
        experiment_id = experiment_id.decode('utf-8')
    
    # Determine intensity range
    if '50to250' in experiment_id or '50to250PMW' in experiment_id:
        intensity = '50→250'
    elif '0to250' in experiment_id:
        intensity = '0→250'
    else:
        return 'Unknown'
    
    # Determine LED2 mode
    if '#C_Bl' in experiment_id:
        mode = 'Constant'
    elif '#T_Bl_Sq' in experiment_id or '#T_Re_Sq' in experiment_id:
        mode = 'Cycling'
    else:
        return 'Unknown'
    
    return f'{intensity} | {mode}'


def load_data_from_h5(h5_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from consolidated HDF5 file.
    
    Args:
        h5_path: Path to consolidated_dataset.h5 file
        
    Returns:
        Tuple of (klein_df, traj_df) DataFrames containing:
        - klein_df: Klein run table with run durations
        - traj_df: Trajectory data with pause/turn/reverse crawl flags
    """
    print(f"Loading data from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # Load Klein run table
        klein_cols = [c.decode() if isinstance(c, bytes) else c for c in f['klein_run_table'].attrs['columns']]
        klein_data = {}
        for col in klein_cols:
            data = f['klein_run_table'][col][:]
            if data.dtype.kind == 'S':  # bytes
                data = np.array([x.decode() if isinstance(x, bytes) else x for x in data])
            klein_data[col] = data
        klein_df = pd.DataFrame(klein_data)
        print(f"  Klein run table: {len(klein_df):,} runs")
        
        # Load trajectories (for pause/turn durations)
        traj_cols = [c.decode() if isinstance(c, bytes) else c for c in f['trajectories'].attrs['columns']]
        
        # Only load columns we need
        needed_cols = ['experiment_id', 'pause_duration', 'turn_duration', 
                       'is_reverse_crawl', 'is_pause', 'is_turn', 'time']
        traj_data = {}
        for col in needed_cols:
            if col in traj_cols:
                data = f['trajectories'][col][:]
                if data.dtype.kind == 'S':
                    data = np.array([x.decode() if isinstance(x, bytes) else x for x in data])
                traj_data[col] = data
        traj_df = pd.DataFrame(traj_data)
        print(f"  Trajectories: {len(traj_df):,} frames")
    
    return klein_df, traj_df


def compute_reverse_crawl_durations(traj_df):
    """Compute duration of each reverse crawl segment."""
    if 'is_reverse_crawl' not in traj_df.columns:
        return []
    
    # Group by experiment and find contiguous reverse crawl segments
    durations = []
    
    for exp_id, group in traj_df.groupby('experiment_id'):
        is_rc = group['is_reverse_crawl'].values.astype(bool)
        time = group['time'].values
        
        # Find segment boundaries
        diff = np.diff(is_rc.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if is_rc[0]:
            starts = np.concatenate([[0], starts])
        if is_rc[-1]:
            ends = np.concatenate([ends, [len(is_rc)]])
        
        for s, e in zip(starts, ends):
            if e > s:
                duration = time[e-1] - time[s]
                if duration >= 3.0:  # Minimum 3s for reverse crawl
                    durations.append({
                        'experiment_id': exp_id,
                        'duration': duration,
                        'start_time': time[s]
                    })
    
    return durations


def main():
    print("=" * 60)
    print("Event Duration Analysis")
    print("=" * 60)
    
    # Load data
    h5_path = Path('data/processed/consolidated_dataset.h5')
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return 1
    
    klein_df, traj_df = load_data_from_h5(h5_path)
    
    # Add condition labels
    klein_df['condition'] = klein_df['experiment_id'].apply(parse_condition)
    traj_df['condition'] = traj_df['experiment_id'].apply(parse_condition)
    
    # Filter to known conditions
    conditions = list(COLORS.keys())
    klein_df = klein_df[klein_df['condition'].isin(conditions)]
    traj_df = traj_df[traj_df['condition'].isin(conditions)]
    
    print(f"\nFiltered to {len(conditions)} conditions")
    print(f"  Klein runs: {len(klein_df):,}")
    print(f"  Trajectory frames: {len(traj_df):,}")
    
    # =================================================================
    # Extract duration statistics
    # =================================================================
    
    # 1. Run durations (from Klein table)
    run_durations = {}
    for cond in conditions:
        mask = klein_df['condition'] == cond
        run_durations[cond] = klein_df.loc[mask, 'runT'].values
    
    # 2. Turn durations (from trajectories - non-zero values only)
    turn_durations = {}
    for cond in conditions:
        mask = (traj_df['condition'] == cond) & (traj_df['turn_duration'] > 0)
        turn_durations[cond] = traj_df.loc[mask, 'turn_duration'].values
    
    # 3. Pause durations (from trajectories - non-zero values only)
    pause_durations = {}
    for cond in conditions:
        mask = (traj_df['condition'] == cond) & (traj_df['pause_duration'] > 0)
        pause_durations[cond] = traj_df.loc[mask, 'pause_duration'].values
    
    # 4. Reverse crawl durations
    print("\nComputing reverse crawl durations...")
    rc_events = compute_reverse_crawl_durations(traj_df)
    rc_df = pd.DataFrame(rc_events) if rc_events else pd.DataFrame(columns=['experiment_id', 'duration'])
    if len(rc_df) > 0:
        rc_df['condition'] = rc_df['experiment_id'].apply(parse_condition)
    
    rc_durations = {}
    for cond in conditions:
        if len(rc_df) > 0:
            mask = rc_df['condition'] == cond
            rc_durations[cond] = rc_df.loc[mask, 'duration'].values
        else:
            rc_durations[cond] = np.array([])
    
    # =================================================================
    # Print summary statistics
    # =================================================================
    print("\n" + "=" * 60)
    print("DURATION STATISTICS BY CONDITION")
    print("=" * 60)
    
    for event_type, data in [('Run', run_durations), ('Turn', turn_durations), 
                              ('Pause', pause_durations), ('Reverse Crawl', rc_durations)]:
        print(f"\n{event_type} Duration (seconds):")
        print("-" * 60)
        print(f"{'Condition':<25} {'N':<8} {'Mean':<10} {'Median':<10} {'Std':<10}")
        print("-" * 60)
        for cond in conditions:
            vals = data[cond]
            if len(vals) > 0:
                print(f"{cond:<25} {len(vals):<8} {np.mean(vals):.3f}     {np.median(vals):.3f}     {np.std(vals):.3f}")
            else:
                print(f"{cond:<25} {0:<8} --        --        --")
    
    # =================================================================
    # Statistical tests (Kruskal-Wallis + Dunn's post-hoc)
    # =================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    
    stat_results = {}
    
    for event_type, data in [('Run', run_durations), ('Turn', turn_durations), 
                              ('Pause', pause_durations)]:
        valid_conditions = [cond for cond in conditions if len(data[cond]) > 0]
        groups = [data[cond] for cond in valid_conditions]
        
        if len(groups) >= 2:
            # Kruskal-Wallis omnibus test
            stat, p = stats.kruskal(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"\n{event_type}: Kruskal-Wallis H = {stat:.2f}, p = {p:.2e} {sig}")
            
            stat_results[event_type] = {
                'kruskal_wallis': {'H': float(stat), 'p': float(p)},
                'pairwise': {}
            }
            
            # Dunn's post-hoc (pairwise Mann-Whitney U with Bonferroni correction)
            if p < 0.05:
                print(f"  Post-hoc pairwise comparisons (Mann-Whitney U, Bonferroni corrected):")
                n_comparisons = len(valid_conditions) * (len(valid_conditions) - 1) // 2
                bonferroni_alpha = 0.05 / n_comparisons
                
                for i in range(len(valid_conditions)):
                    for j in range(i + 1, len(valid_conditions)):
                        g1, g2 = groups[i], groups[j]
                        cond1, cond2 = valid_conditions[i], valid_conditions[j]
                        
                        if len(g1) > 0 and len(g2) > 0:
                            u_stat, p_pair = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            sig_pair = "*" if p_pair < bonferroni_alpha else "ns"
                            
                            # Short labels for display
                            short1 = cond1.replace(' | ', '/').replace('→', '>')
                            short2 = cond2.replace(' | ', '/').replace('→', '>')
                            print(f"    {short1} vs {short2}: U={u_stat:.0f}, p={p_pair:.3e} {sig_pair}")
                            
                            stat_results[event_type]['pairwise'][f"{cond1} vs {cond2}"] = {
                                'U': float(u_stat), 
                                'p': float(p_pair),
                                'significant': bool(p_pair < bonferroni_alpha)
                            }
    
    # Statistical results will be added to summary below
    
    # =================================================================
    # Create figure
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Run duration boxplot
    ax = axes[0, 0]
    box_data = [run_durations[c] for c in conditions]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[c.split(' | ')[0] + '\n' + c.split(' | ')[1] for c in conditions])
    for patch, color in zip(bp['boxes'], [COLORS[c] for c in conditions]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Duration (s)', fontsize=12)
    ax.set_title('A. Run Duration', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Turn duration boxplot
    ax = axes[0, 1]
    box_data = [turn_durations[c] for c in conditions]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[c.split(' | ')[0] + '\n' + c.split(' | ')[1] for c in conditions])
    for patch, color in zip(bp['boxes'], [COLORS[c] for c in conditions]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Duration (s)', fontsize=12)
    ax.set_title('B. Turn Duration', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Pause duration boxplot
    ax = axes[1, 0]
    box_data = [pause_durations[c] for c in conditions]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[c.split(' | ')[0] + '\n' + c.split(' | ')[1] for c in conditions])
    for patch, color in zip(bp['boxes'], [COLORS[c] for c in conditions]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Duration (s)', fontsize=12)
    ax.set_title('C. Pause Duration', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel D: Event counts comparison
    ax = axes[1, 1]
    x = np.arange(len(conditions))
    width = 0.2
    
    run_counts = [len(run_durations[c]) for c in conditions]
    turn_counts = [len(turn_durations[c]) for c in conditions]
    pause_counts = [len(pause_durations[c]) for c in conditions]
    rc_counts = [len(rc_durations[c]) for c in conditions]
    
    ax.bar(x - 1.5*width, run_counts, width, label='Runs', color='#1f77b4')
    ax.bar(x - 0.5*width, turn_counts, width, label='Turns', color='#ff7f0e')
    ax.bar(x + 0.5*width, pause_counts, width, label='Pauses', color='#2ca02c')
    ax.bar(x + 1.5*width, rc_counts, width, label='Rev. Crawls', color='#d62728')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.split(' | ')[0] + '\n' + c.split(' | ')[1] for c in conditions])
    ax.set_ylabel('Event Count', fontsize=12)
    ax.set_title('D. Event Counts by Condition', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('figures/event_durations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Save summary JSON
    summary = {
        'conditions': conditions,
        'run_duration': {
            cond: {
                'n': len(run_durations[cond]),
                'mean': float(np.mean(run_durations[cond])) if len(run_durations[cond]) > 0 else None,
                'median': float(np.median(run_durations[cond])) if len(run_durations[cond]) > 0 else None,
                'std': float(np.std(run_durations[cond])) if len(run_durations[cond]) > 0 else None
            } for cond in conditions
        },
        'turn_duration': {
            cond: {
                'n': len(turn_durations[cond]),
                'mean': float(np.mean(turn_durations[cond])) if len(turn_durations[cond]) > 0 else None,
                'median': float(np.median(turn_durations[cond])) if len(turn_durations[cond]) > 0 else None,
                'std': float(np.std(turn_durations[cond])) if len(turn_durations[cond]) > 0 else None
            } for cond in conditions
        },
        'pause_duration': {
            cond: {
                'n': len(pause_durations[cond]),
                'mean': float(np.mean(pause_durations[cond])) if len(pause_durations[cond]) > 0 else None,
                'median': float(np.median(pause_durations[cond])) if len(pause_durations[cond]) > 0 else None,
                'std': float(np.std(pause_durations[cond])) if len(pause_durations[cond]) > 0 else None
            } for cond in conditions
        },
        'reverse_crawl_duration': {
            cond: {
                'n': len(rc_durations[cond]),
                'mean': float(np.mean(rc_durations[cond])) if len(rc_durations[cond]) > 0 else None,
                'median': float(np.median(rc_durations[cond])) if len(rc_durations[cond]) > 0 else None,
                'std': float(np.std(rc_durations[cond])) if len(rc_durations[cond]) > 0 else None
            } for cond in conditions
        },
        'statistical_tests': stat_results
    }
    
    summary_path = Path('data/model/event_duration_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    plt.show()
    return 0


if __name__ == '__main__':
    exit(main())
