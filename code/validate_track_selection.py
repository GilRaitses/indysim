#!/usr/bin/env python3
"""
Validate Track Selection

Compares complete vs incomplete tracks on multiple behavioral metrics
to assess whether the 79 complete tracks (11%) are representative of
the full population.

Metrics compared:
- Mean speed
- Turn rate (turns per minute)
- Pause fraction
- Reverse crawl fraction
- Event count

Output:
- data/model/track_selection_validation.json

Usage:
    python scripts/validate_track_selection.py
"""

import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


def hedges_g(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Compute Hedges' g effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0, "undefined"
    
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx + ny - 2))
    
    if pooled_std == 0:
        return 0.0, "undefined"
    
    d = (np.mean(x) - np.mean(y)) / pooled_std
    
    # Hedges correction
    df = nx + ny - 2
    j = 1 - 3 / (4 * df - 1) if df > 0 else 1
    g = d * j
    
    abs_g = abs(g)
    if abs_g < 0.2:
        interp = "negligible"
    elif abs_g < 0.5:
        interp = "small"
    elif abs_g < 0.8:
        interp = "medium"
    else:
        interp = "large"
    
    return g, interp


def load_trajectories(h5_path: Path) -> pd.DataFrame:
    """Load event data from consolidated H5 (events group has track_id)."""
    print(f"Loading events from {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Use 'events' group which has track_id
        grp = f['events']
        columns = [c.decode('utf-8') if isinstance(c, bytes) else c 
                   for c in grp.attrs['columns']]
        
        # Load all needed columns
        needed = ['experiment_id', 'track_id', 'time', 'speed', 
                  'is_turn', 'is_turn_start', 'is_pause', 'is_reverse_crawl', 
                  'is_reorientation_start']
        available = [c for c in needed if c in columns]
        
        data = {}
        for col in available:
            vals = grp[col][:]
            if vals.dtype.kind == 'S':
                vals = np.array([v.decode('utf-8') if isinstance(v, bytes) else v for v in vals])
            data[col] = vals
        
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} frames with columns: {available}")
    
    return df


def compute_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track behavioral metrics."""
    print("Computing per-track metrics...")
    
    # Create unique track identifier
    df['unique_track_id'] = df['experiment_id'].astype(str) + '_' + df['track_id'].astype(str)
    
    metrics = []
    
    for track_id, group in df.groupby('unique_track_id'):
        time = group['time'].values
        duration = time.max() - time.min() if len(time) > 1 else 0
        n_frames = len(group)
        
        # Mean speed
        mean_speed = group['speed'].mean() if 'speed' in group.columns else np.nan
        
        # Turn count and rate
        if 'is_reorientation_start' in group.columns:
            n_turns = group['is_reorientation_start'].sum()
        elif 'is_turn' in group.columns:
            # Count turn onsets
            is_turn = group['is_turn'].values.astype(bool)
            n_turns = np.sum(np.diff(is_turn.astype(int)) == 1)
        else:
            n_turns = 0
        turn_rate = n_turns / (duration / 60) if duration > 0 else 0
        
        # Pause fraction
        if 'is_pause' in group.columns:
            pause_fraction = group['is_pause'].mean()
        else:
            pause_fraction = np.nan
        
        # Reverse crawl fraction
        if 'is_reverse_crawl' in group.columns:
            rc_fraction = group['is_reverse_crawl'].mean()
        else:
            rc_fraction = np.nan
        
        metrics.append({
            'unique_track_id': track_id,
            'experiment_id': group['experiment_id'].iloc[0],
            'duration': duration,
            'n_frames': n_frames,
            'mean_speed': mean_speed,
            'n_turns': n_turns,
            'turn_rate': turn_rate,
            'pause_fraction': pause_fraction,
            'rc_fraction': rc_fraction
        })
    
    metrics_df = pd.DataFrame(metrics)
    print(f"  Computed metrics for {len(metrics_df)} tracks")
    
    return metrics_df


def identify_complete_tracks(metrics_df: pd.DataFrame, min_duration: float = 1100.0) -> pd.DataFrame:
    """
    Identify complete tracks that span approximately the full experiment.
    
    Experiment is ~20 minutes (1200 s). Complete tracks span at least
    1100 s (~18 min) to allow for tracking dropout at start/end.
    """
    metrics_df['is_complete'] = metrics_df['duration'] >= min_duration
    
    n_complete = metrics_df['is_complete'].sum()
    n_total = len(metrics_df)
    
    print(f"\nTrack completeness:")
    print(f"  Complete tracks (>= {min_duration}s): {n_complete} ({100*n_complete/n_total:.1f}%)")
    print(f"  Incomplete tracks: {n_total - n_complete} ({100*(n_total-n_complete)/n_total:.1f}%)")
    
    return metrics_df


def compare_groups(complete: np.ndarray, incomplete: np.ndarray, metric_name: str) -> Dict:
    """Compare complete vs incomplete tracks on a metric."""
    # Remove NaN values
    complete = complete[~np.isnan(complete)]
    incomplete = incomplete[~np.isnan(incomplete)]
    
    if len(complete) < 2 or len(incomplete) < 2:
        return {'error': 'insufficient data'}
    
    # Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(complete, incomplete, alternative='two-sided')
    
    # Effect size
    g, interp = hedges_g(complete, incomplete)
    
    result = {
        'metric': metric_name,
        'complete': {
            'n': len(complete),
            'mean': round(float(np.mean(complete)), 4),
            'std': round(float(np.std(complete)), 4),
            'median': round(float(np.median(complete)), 4)
        },
        'incomplete': {
            'n': len(incomplete),
            'mean': round(float(np.mean(incomplete)), 4),
            'std': round(float(np.std(incomplete)), 4),
            'median': round(float(np.median(incomplete)), 4)
        },
        'test': {
            'mann_whitney_U': round(float(stat), 2),
            'p_value': round(float(p_value), 4),
            'significant': p_value < 0.05
        },
        'effect_size': {
            'hedges_g': round(g, 3),
            'interpretation': interp
        }
    }
    
    return result


def main():
    print("=" * 70)
    print("VALIDATE TRACK SELECTION: Complete vs Incomplete Tracks")
    print("=" * 70)
    
    h5_path = Path('data/processed/consolidated_dataset.h5')
    
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return
    
    # Load data
    df = load_trajectories(h5_path)
    
    # Compute per-track metrics
    metrics_df = compute_track_metrics(df)
    
    # Identify complete tracks
    metrics_df = identify_complete_tracks(metrics_df, min_duration=1100.0)
    
    complete = metrics_df[metrics_df['is_complete']]
    incomplete = metrics_df[~metrics_df['is_complete']]
    
    print(f"\nComplete track duration range: {complete['duration'].min():.0f} - {complete['duration'].max():.0f} s")
    print(f"Incomplete track duration range: {incomplete['duration'].min():.0f} - {incomplete['duration'].max():.0f} s")
    
    # Compare metrics
    results = {}
    
    comparisons = [
        ('mean_speed', 'Mean Speed (mm/s)'),
        ('turn_rate', 'Turn Rate (turns/min)'),
        ('pause_fraction', 'Pause Fraction'),
        ('rc_fraction', 'Reverse Crawl Fraction'),
        ('n_turns', 'Total Turn Count')
    ]
    
    print("\n" + "=" * 70)
    print("METRIC COMPARISONS: Complete vs Incomplete Tracks")
    print("=" * 70)
    
    for col, name in comparisons:
        if col not in metrics_df.columns:
            continue
        
        result = compare_groups(
            complete[col].values,
            incomplete[col].values,
            name
        )
        
        if 'error' not in result:
            print(f"\n{name}:")
            print(f"  Complete:   mean = {result['complete']['mean']:.4f} ± {result['complete']['std']:.4f} (n={result['complete']['n']})")
            print(f"  Incomplete: mean = {result['incomplete']['mean']:.4f} ± {result['incomplete']['std']:.4f} (n={result['incomplete']['n']})")
            print(f"  Mann-Whitney U p = {result['test']['p_value']:.4f} {'*' if result['test']['significant'] else ''}")
            print(f"  Effect size: Hedges' g = {result['effect_size']['hedges_g']:.3f} ({result['effect_size']['interpretation']})")
            
            results[col] = result
    
    # Overall assessment
    n_significant = sum(1 for r in results.values() if r['test']['significant'])
    n_large_effects = sum(1 for r in results.values() if abs(r['effect_size']['hedges_g']) >= 0.8)
    
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print(f"  Significant differences (p < 0.05): {n_significant} / {len(results)}")
    print(f"  Large effect sizes (|g| >= 0.8): {n_large_effects} / {len(results)}")
    
    if n_significant == 0 and n_large_effects == 0:
        assessment = "Complete tracks appear REPRESENTATIVE of the full population"
    elif n_significant <= 1 and n_large_effects == 0:
        assessment = "Complete tracks appear MOSTLY representative, with minor differences"
    else:
        assessment = "Complete tracks show SIGNIFICANT differences from incomplete tracks - potential selection bias"
    
    print(f"\n  Assessment: {assessment}")
    
    # Convert booleans for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj
    
    results = convert_for_json(results)
    
    # Save results
    output = {
        'summary': {
            'n_complete': int(complete.shape[0]),
            'n_incomplete': int(incomplete.shape[0]),
            'pct_complete': round(100 * len(complete) / len(metrics_df), 1),
            'min_duration_threshold': 1100.0,
            'n_significant_differences': n_significant,
            'n_large_effects': n_large_effects,
            'assessment': assessment
        },
        'comparisons': results,
        'methodology': {
            'test': 'Mann-Whitney U (two-sided)',
            'effect_size': 'Hedges\' g',
            'alpha': 0.05,
            'complete_threshold': '>= 1100 s duration'
        }
    }
    
    output_path = Path('data/model/track_selection_validation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return output


if __name__ == '__main__':
    main()
