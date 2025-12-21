#!/usr/bin/env python3
"""
Compute Cliff's Delta for Event Durations

Loads individual event data from consolidated H5 and computes
non-parametric effect sizes (Cliff's delta) for pairwise condition comparisons.

Cliff's delta interpretation:
- |δ| < 0.147: negligible
- |δ| < 0.33: small
- |δ| < 0.474: medium
- |δ| >= 0.474: large

Output:
- data/model/cliffs_delta_durations.json

Usage:
    python scripts/compute_cliffs_delta.py
"""

import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta non-parametric effect size.
    
    δ = P(X > Y) - P(X < Y)
    
    Returns (delta, interpretation).
    """
    n_x = len(x)
    n_y = len(y)
    
    if n_x == 0 or n_y == 0:
        return 0.0, "undefined"
    
    # Count dominance pairs
    greater = 0
    less = 0
    
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    
    total = n_x * n_y
    delta = (greater - less) / total
    
    # Interpretation (Romano et al., 2006 thresholds)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interp = "negligible"
    elif abs_delta < 0.33:
        interp = "small"
    elif abs_delta < 0.474:
        interp = "medium"
    else:
        interp = "large"
    
    return delta, interp


def parse_condition(experiment_id: str) -> str:
    """Parse experiment ID to extract condition."""
    if '0to250PWM' in experiment_id and 'C_Bl' in experiment_id:
        return '0→250 | Constant'
    elif '0to250PWM' in experiment_id and 'T_Bl_Sq' in experiment_id:
        return '0→250 | Cycling'
    elif '50to250' in experiment_id and 'C_Bl' in experiment_id:
        return '50→250 | Constant'
    elif '50to250' in experiment_id and 'T_Bl_Sq' in experiment_id:
        return '50→250 | Cycling'
    return 'Unknown'


def load_events_from_h5(h5_path: Path) -> pd.DataFrame:
    """Load Klein run table events from consolidated H5."""
    print(f"Loading events from {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if 'klein_runs' not in f:
            print("  Warning: 'klein_runs' group not found")
            return pd.DataFrame()
        
        grp = f['klein_runs']
        columns = [c.decode('utf-8') if isinstance(c, bytes) else c 
                   for c in grp.attrs['columns']]
        
        data = {}
        for col in columns:
            if col in grp:
                vals = grp[col][:]
                # Decode bytes if needed
                if vals.dtype.kind == 'S':
                    vals = np.array([v.decode('utf-8') if isinstance(v, bytes) else v for v in vals])
                data[col] = vals
        
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} run events with columns: {columns[:5]}...")
    
    return df


def load_trajectories_for_pauses(h5_path: Path) -> pd.DataFrame:
    """Load trajectories to extract pause events."""
    print(f"Loading trajectories for pause extraction...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'trajectories' not in f:
            print("  Warning: 'trajectories' group not found")
            return pd.DataFrame()
        
        grp = f['trajectories']
        columns = [c.decode('utf-8') if isinstance(c, bytes) else c 
                   for c in grp.attrs['columns']]
        
        # Only load needed columns
        needed = ['experiment_id', 'time', 'is_pause', 'is_turn', 'is_reverse_crawl', 'speed']
        available = [c for c in needed if c in grp]
        
        data = {}
        for col in available:
            vals = grp[col][:]
            if vals.dtype.kind == 'S':
                vals = np.array([v.decode('utf-8') if isinstance(v, bytes) else v for v in vals])
            data[col] = vals
        
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} frames with columns: {available}")
    
    return df


def compute_pause_durations(traj_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Compute pause durations per condition from trajectory data."""
    if 'is_pause' not in traj_df.columns:
        return {}
    
    results = {}
    
    for exp_id, group in traj_df.groupby('experiment_id'):
        condition = parse_condition(exp_id)
        if condition == 'Unknown':
            continue
        
        if condition not in results:
            results[condition] = []
        
        is_pause = group['is_pause'].values.astype(bool)
        time = group['time'].values
        
        # Find pause boundaries
        diff = np.diff(is_pause.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if is_pause[0]:
            starts = np.concatenate([[0], starts])
        if is_pause[-1]:
            ends = np.concatenate([ends, [len(is_pause)]])
        
        for s, e in zip(starts, ends):
            if e > s:
                duration = time[e-1] - time[s]
                if duration >= 0.5:  # Minimum pause duration
                    results[condition].append(duration)
    
    return results


def compute_turn_durations(traj_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Compute turn durations per condition."""
    if 'is_turn' not in traj_df.columns:
        return {}
    
    results = {}
    
    for exp_id, group in traj_df.groupby('experiment_id'):
        condition = parse_condition(exp_id)
        if condition == 'Unknown':
            continue
        
        if condition not in results:
            results[condition] = []
        
        is_turn = group['is_turn'].values.astype(bool)
        time = group['time'].values
        
        # Find turn boundaries
        diff = np.diff(is_turn.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if is_turn[0]:
            starts = np.concatenate([[0], starts])
        if is_turn[-1]:
            ends = np.concatenate([ends, [len(is_turn)]])
        
        for s, e in zip(starts, ends):
            if e > s:
                duration = time[e-1] - time[s]
                if duration > 0.1:  # Minimum turn duration
                    results[condition].append(duration)
    
    return results


def compute_reverse_crawl_durations(traj_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Compute reverse crawl durations per condition."""
    if 'is_reverse_crawl' not in traj_df.columns:
        return {}
    
    results = {}
    
    for exp_id, group in traj_df.groupby('experiment_id'):
        condition = parse_condition(exp_id)
        if condition == 'Unknown':
            continue
        
        if condition not in results:
            results[condition] = []
        
        is_rc = group['is_reverse_crawl'].values.astype(bool)
        time = group['time'].values
        
        # Find reverse crawl boundaries
        diff = np.diff(is_rc.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if is_rc[0]:
            starts = np.concatenate([[0], starts])
        if is_rc[-1]:
            ends = np.concatenate([ends, [len(is_rc)]])
        
        for s, e in zip(starts, ends):
            if e > s:
                duration = time[e-1] - time[s]
                if duration >= 3.0:  # Minimum reverse crawl duration
                    results[condition].append(duration)
    
    return results


def compute_run_durations(events_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract run durations from Klein run table."""
    if 'run_duration' not in events_df.columns and 'duration' not in events_df.columns:
        return {}
    
    duration_col = 'run_duration' if 'run_duration' in events_df.columns else 'duration'
    exp_col = 'experiment_id' if 'experiment_id' in events_df.columns else None
    
    if exp_col is None:
        print("  Warning: No experiment_id column for condition parsing")
        return {}
    
    results = {}
    
    for exp_id, group in events_df.groupby(exp_col):
        condition = parse_condition(exp_id)
        if condition == 'Unknown':
            continue
        
        if condition not in results:
            results[condition] = []
        
        durations = group[duration_col].dropna().values
        durations = durations[durations > 0]
        results[condition].extend(durations.tolist())
    
    return results


def compute_pairwise_cliffs_delta(durations: Dict[str, List[float]]) -> Dict:
    """Compute pairwise Cliff's delta for all condition pairs."""
    conditions = list(durations.keys())
    results = {}
    
    for cond1, cond2 in combinations(conditions, 2):
        x = np.array(durations[cond1])
        y = np.array(durations[cond2])
        
        delta, interp = cliffs_delta(x, y)
        
        key = f"{cond1} vs {cond2}"
        results[key] = {
            'cliffs_delta': round(delta, 3),
            'interpretation': interp,
            'n_x': len(x),
            'n_y': len(y),
            'mean_x': round(np.mean(x), 2) if len(x) > 0 else None,
            'mean_y': round(np.mean(y), 2) if len(y) > 0 else None
        }
    
    return results


def main():
    print("=" * 70)
    print("COMPUTING CLIFF'S DELTA FOR EVENT DURATIONS")
    print("=" * 70)
    
    h5_path = Path('data/processed/consolidated_dataset.h5')
    
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return
    
    # Load trajectories for pause/turn/reverse analysis
    traj_df = load_trajectories_for_pauses(h5_path)
    
    all_results = {}
    
    # Compute durations for each event type
    if len(traj_df) > 0:
        print("\nComputing pause durations...")
        pause_durations = compute_pause_durations(traj_df)
        if pause_durations:
            for cond, durs in pause_durations.items():
                print(f"  {cond}: {len(durs)} pauses, mean = {np.mean(durs):.2f} s")
            pause_cliffs = compute_pairwise_cliffs_delta(pause_durations)
            all_results['pause_duration'] = {
                'pairwise_cliffs_delta': pause_cliffs,
                'n_per_condition': {k: len(v) for k, v in pause_durations.items()}
            }
        
        print("\nComputing turn durations...")
        turn_durations = compute_turn_durations(traj_df)
        if turn_durations:
            for cond, durs in turn_durations.items():
                print(f"  {cond}: {len(durs)} turns, mean = {np.mean(durs):.2f} s")
            turn_cliffs = compute_pairwise_cliffs_delta(turn_durations)
            all_results['turn_duration'] = {
                'pairwise_cliffs_delta': turn_cliffs,
                'n_per_condition': {k: len(v) for k, v in turn_durations.items()}
            }
        
        print("\nComputing reverse crawl durations...")
        rc_durations = compute_reverse_crawl_durations(traj_df)
        if rc_durations:
            for cond, durs in rc_durations.items():
                print(f"  {cond}: {len(durs)} reverse crawls, mean = {np.mean(durs):.2f} s")
            rc_cliffs = compute_pairwise_cliffs_delta(rc_durations)
            all_results['reverse_crawl_duration'] = {
                'pairwise_cliffs_delta': rc_cliffs,
                'n_per_condition': {k: len(v) for k, v in rc_durations.items()}
            }
    
    # Load Klein events for run durations
    events_df = load_events_from_h5(h5_path)
    if len(events_df) > 0:
        print("\nComputing run durations from Klein table...")
        run_durations = compute_run_durations(events_df)
        if run_durations:
            for cond, durs in run_durations.items():
                print(f"  {cond}: {len(durs)} runs, mean = {np.mean(durs):.2f} s")
            run_cliffs = compute_pairwise_cliffs_delta(run_durations)
            all_results['run_duration'] = {
                'pairwise_cliffs_delta': run_cliffs,
                'n_per_condition': {k: len(v) for k, v in run_durations.items()}
            }
    
    # Summary
    print("\n" + "=" * 70)
    print("CLIFF'S DELTA SUMMARY")
    print("=" * 70)
    
    for event_type, data in all_results.items():
        print(f"\n{event_type}:")
        for pair, stats in data['pairwise_cliffs_delta'].items():
            print(f"  {pair}: δ = {stats['cliffs_delta']:.3f} ({stats['interpretation']})")
    
    # Add methodology
    all_results['methodology'] = {
        'effect_size_metric': 'Cliff\'s delta (non-parametric)',
        'formula': 'δ = P(X > Y) - P(X < Y)',
        'thresholds': {
            'negligible': '|δ| < 0.147',
            'small': '|δ| < 0.33',
            'medium': '|δ| < 0.474',
            'large': '|δ| >= 0.474'
        },
        'reference': 'Romano et al. (2006)'
    }
    
    # Save results
    output_path = Path('data/model/cliffs_delta_durations.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nSaved to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()
