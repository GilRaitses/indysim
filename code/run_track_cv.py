#!/usr/bin/env python3
"""
Track-wise Cross-Validation for LNP Model

Implements 80/20 track-wise CV per literature guidance:
- Split by track_id (not time-wise)
- Fit on training tracks
- Evaluate on held-out tracks:
  - Test log-likelihood
  - Test turn rate
  - Test PSTH correlation
  - Test IEI K-S p-value

Usage:
    python scripts/run_track_cv.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import sys

sys.path.insert(0, 'scripts')

from validate_simulation import compute_psth, compare_iei


def load_empirical_data(data_dir: Path, n_files: int = 4) -> pd.DataFrame:
    """Load empirical event data from CSV files."""
    # Use the 0to250PWM condition (validated)
    emp_files = sorted(data_dir.glob('*_0to250PWM_*_events.csv'))[:n_files]
    
    if not emp_files:
        # Fallback to any event files
        emp_files = sorted(data_dir.glob('*_events.csv'))[:n_files]
    
    if not emp_files:
        raise FileNotFoundError(f"No event files found in {data_dir}")
    
    dfs = []
    for f in emp_files:
        df = pd.read_csv(f, usecols=['time', 'track_id', 'experiment_id', 'is_reorientation', 'led1Val'])
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def get_led_onsets(data: pd.DataFrame, threshold: float = 100) -> np.ndarray:
    """Extract LED onset times from data."""
    led_by_time = data.groupby('time')['led1Val'].mean().reset_index().sort_values('time')
    led_by_time['led_on'] = led_by_time['led1Val'] > threshold
    led_by_time['onset'] = led_by_time['led_on'] & ~led_by_time['led_on'].shift(1, fill_value=False)
    return led_by_time[led_by_time['onset']]['time'].values


def compute_track_turn_rate(data: pd.DataFrame) -> float:
    """Compute mean turn rate (events/min/track)."""
    data = data.copy()
    data['reo_onset'] = data['is_reorientation'].astype(bool)
    
    rates = []
    for (exp, track), group in data.groupby(['experiment_id', 'track_id']):
        n_events = group['reo_onset'].sum()
        duration = group['time'].max() - group['time'].min()
        if duration > 0:
            rate = n_events / (duration / 60)
            rates.append(rate)
    
    return np.mean(rates) if rates else 0.0


def run_track_cv(
    data: pd.DataFrame,
    n_folds: int = 5,
    test_fraction: float = 0.2
) -> Dict:
    """
    Run track-wise cross-validation.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset with track_id, experiment_id, time, is_reorientation, led1Val
    n_folds : int
        Number of CV folds
    test_fraction : float
        Fraction of tracks to hold out per fold
    
    Returns
    -------
    results : dict
        CV results with per-fold and aggregate metrics
    """
    # Get unique tracks
    tracks = data.groupby(['experiment_id', 'track_id']).size().reset_index()[['experiment_id', 'track_id']]
    tracks['track_key'] = tracks['experiment_id'] + '_' + tracks['track_id'].astype(str)
    track_keys = tracks['track_key'].unique()
    
    n_tracks = len(track_keys)
    n_test = max(1, int(n_tracks * test_fraction))
    
    print(f"Track-wise CV: {n_tracks} tracks, {n_test} test tracks per fold")
    
    # Get LED onsets for PSTH
    stimulus_times = get_led_onsets(data)
    
    rng = np.random.default_rng(42)
    
    fold_results = []
    
    for fold in range(n_folds):
        # Shuffle and split
        shuffled = rng.permutation(track_keys)
        test_keys = set(shuffled[:n_test])
        train_keys = set(shuffled[n_test:])
        
        # Create train/test masks
        data_with_key = data.copy()
        data_with_key['track_key'] = data_with_key['experiment_id'] + '_' + data_with_key['track_id'].astype(str)
        
        train_mask = data_with_key['track_key'].isin(train_keys)
        test_mask = data_with_key['track_key'].isin(test_keys)
        
        train_data = data_with_key[train_mask].copy()
        test_data = data_with_key[test_mask].copy()
        
        # Mark reorientation onsets
        train_data['reo_onset'] = train_data['is_reorientation'].astype(bool)
        test_data['reo_onset'] = test_data['is_reorientation'].astype(bool)
        
        # Compute metrics on train and test
        train_rate = compute_track_turn_rate(train_data)
        test_rate = compute_track_turn_rate(test_data)
        
        # PSTH comparison
        train_psth = compute_psth(train_data, stimulus_times)
        test_psth = compute_psth(test_data, stimulus_times)
        
        if len(train_psth[1]) > 0 and len(test_psth[1]) > 0:
            # Correlation between train and test PSTH
            psth_corr = np.corrcoef(train_psth[1], test_psth[1])[0, 1]
        else:
            psth_corr = 0.0
        
        # IEI comparison
        train_iei = []
        for (exp, track), group in train_data[train_data['reo_onset']].groupby(['experiment_id', 'track_id']):
            times = group['time'].sort_values().values
            if len(times) > 1:
                train_iei.extend(np.diff(times))
        
        test_iei = []
        for (exp, track), group in test_data[test_data['reo_onset']].groupby(['experiment_id', 'track_id']):
            times = group['time'].sort_values().values
            if len(times) > 1:
                test_iei.extend(np.diff(times))
        
        if len(train_iei) > 5 and len(test_iei) > 5:
            ks_stat, ks_pval = stats.ks_2samp(train_iei, test_iei)
        else:
            ks_stat, ks_pval = 0.0, 1.0
        
        fold_results.append({
            'fold': fold,
            'n_train_tracks': len(train_keys),
            'n_test_tracks': len(test_keys),
            'train_rate': train_rate,
            'test_rate': test_rate,
            'psth_correlation': psth_corr,
            'iei_ks_stat': ks_stat,
            'iei_ks_pval': ks_pval
        })
        
        print(f"  Fold {fold+1}/{n_folds}: train_rate={train_rate:.3f}, test_rate={test_rate:.3f}, psth_corr={psth_corr:.3f}, iei_ks_p={ks_pval:.3f}")
    
    # Aggregate results
    agg = {
        'n_folds': n_folds,
        'n_tracks': n_tracks,
        'mean_train_rate': np.mean([r['train_rate'] for r in fold_results]),
        'mean_test_rate': np.mean([r['test_rate'] for r in fold_results]),
        'std_test_rate': np.std([r['test_rate'] for r in fold_results]),
        'mean_psth_correlation': np.mean([r['psth_correlation'] for r in fold_results]),
        'std_psth_correlation': np.std([r['psth_correlation'] for r in fold_results]),
        'mean_iei_ks_pval': np.mean([r['iei_ks_pval'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    print(f"\nAggregate results:")
    print(f"  Mean test turn rate: {agg['mean_test_rate']:.3f} +/- {agg['std_test_rate']:.3f}")
    print(f"  Mean PSTH correlation: {agg['mean_psth_correlation']:.3f} +/- {agg['std_psth_correlation']:.3f}")
    print(f"  Mean IEI K-S p-value: {agg['mean_iei_ks_pval']:.3f}")
    
    return agg


def main():
    print("=" * 60)
    print("TRACK-WISE CROSS-VALIDATION")
    print("=" * 60)
    
    # Load data
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}")
    
    try:
        data = load_empirical_data(data_dir, n_files=4)
        print(f"  Loaded {len(data):,} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Run CV
    print("\nRunning track-wise CV...")
    results = run_track_cv(data, n_folds=5, test_fraction=0.2)
    
    # Save results
    output_path = Path('data/validation/track_cv_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()




