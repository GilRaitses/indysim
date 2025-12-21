#!/usr/bin/env python3
"""
Create Factorial Design Matrix

Creates a pooled design matrix for the 2×2 factorial NB-GLMM:
- 12 experiments, 270 tracks, ~10,759 events
- Columns: events, I, T, IT, K_on, I_K_on, T_K_on, K_off, track, experiment, condition

Run with: source .venv-larvaworld/bin/activate && python scripts/create_factorial_design.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import gamma as gamma_dist

# Kernel parameters (from reference condition)
KERNEL_PARAMS = {
    'A': 0.456,
    'alpha1': 2.22,
    'beta1': 0.132,
    'B': 12.54,
    'alpha2': 4.38,
    'beta2': 0.869,
    'D': -0.114,
    'tau_off': 2.0
}

# LED timing (verified identical across all conditions)
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3

# Condition definitions
CONDITIONS = {
    '0→250 | Control': {'I': 0, 'T': 0, 'pattern': ('0to250PWM', '#C_Bl_7PWM')},
    '0→250 | Temp': {'I': 0, 'T': 1, 'pattern': ('0to250PWM', '#T_Bl_Sq_5to15PWM')},
    '50→250 | Control': {'I': 1, 'T': 0, 'pattern': ('50to250PWM', '#C_Bl_7PWM')},
    '50→250 | Temp': {'I': 1, 'T': 1, 'pattern': ('50to250PWM', '#T_Bl_Sq_5to15PWM')},
}

# Anomalous files to exclude (10-20x higher event counts)
ANOMALOUS_FILES = ['202510291652', '202510291713']


def gamma_pdf(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute gamma PDF."""
    result = np.zeros_like(t, dtype=float)
    valid = t > 0
    if valid.any():
        result[valid] = gamma_dist.pdf(t[valid], a=alpha, scale=beta)
    return result


def compute_K_on(t_since_onset: np.ndarray) -> np.ndarray:
    """
    Compute gamma-difference kernel K_on(t).
    
    K_on(t) = A * Gamma(t; alpha1, beta1) - B * Gamma(t; alpha2, beta2)
    """
    p = KERNEL_PARAMS
    fast = p['A'] * gamma_pdf(t_since_onset, p['alpha1'], p['beta1'])
    slow = p['B'] * gamma_pdf(t_since_onset, p['alpha2'], p['beta2'])
    return fast - slow


def compute_K_off(t_since_offset: np.ndarray) -> np.ndarray:
    """
    Compute exponential rebound kernel K_off(t).
    
    K_off(t) = D * exp(-t / tau_off)
    """
    p = KERNEL_PARAMS
    result = np.zeros_like(t_since_offset, dtype=float)
    valid = t_since_offset > 0
    result[valid] = p['D'] * np.exp(-t_since_offset[valid] / p['tau_off'])
    return result


def compute_time_since_led(times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time since LED onset and offset for each frame.
    
    Returns:
        t_since_onset: Time since most recent LED onset (0 if LED OFF)
        t_since_offset: Time since most recent LED offset (0 if LED ON)
    """
    t_since_onset = np.zeros(len(times))
    t_since_offset = np.zeros(len(times))
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            # Before first LED
            t_since_onset[i] = 0
            t_since_offset[i] = 0
        else:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                # LED is ON
                t_since_onset[i] = cycle_time
                t_since_offset[i] = 0
            else:
                # LED is OFF
                t_since_onset[i] = 0
                t_since_offset[i] = cycle_time - LED_ON_DURATION
    
    return t_since_onset, t_since_offset


def get_condition_files(data_dir: str = "data/engineered") -> Dict[str, List[Path]]:
    """Get files for each condition."""
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob('*_events.csv'))
    
    condition_files = {}
    for cond_name, cond_info in CONDITIONS.items():
        intensity_pattern, bg_pattern = cond_info['pattern']
        matching = [
            f for f in all_files
            if intensity_pattern in f.name and bg_pattern in f.name
            and not any(a in f.name for a in ANOMALOUS_FILES)
        ]
        condition_files[cond_name] = matching
    
    return condition_files


def load_and_process_condition(
    cond_name: str,
    files: List[Path],
    cond_info: Dict
) -> pd.DataFrame:
    """Load and process data for one condition."""
    I = cond_info['I']
    T = cond_info['T']
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract full experiment ID (date+time) from filename
        # Filename format: GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228_events.csv
        # The date is the second-to-last part before _events.csv
        parts = f.stem.replace('_events', '').split('_')
        experiment_id = parts[-1]  # e.g., '202510301228'
        
        # Compute time since LED onset/offset
        times = df['time'].values
        t_since_onset, t_since_offset = compute_time_since_led(times)
        
        # Compute kernel values
        K_on = compute_K_on(t_since_onset)
        K_off = compute_K_off(t_since_offset)
        
        # Create design columns
        processed = pd.DataFrame({
            'events': df['is_reorientation_start'].fillna(0).astype(int),
            'I': I,
            'T': T,
            'IT': I * T,
            'K_on': K_on,
            'I_K_on': I * K_on,
            'T_K_on': T * K_on,
            'K_off': K_off,
            'track': df['track_id'].astype(str) + '_' + experiment_id,
            'experiment': experiment_id,
            'condition': cond_name,
            'time': times
        })
        
        dfs.append(processed)
    
    return pd.concat(dfs, ignore_index=True)


def main():
    print("=" * 70)
    print("CREATING FACTORIAL DESIGN MATRIX")
    print("=" * 70)
    
    # Get files by condition
    condition_files = get_condition_files()
    
    print("\nCondition files (excluding anomalous):")
    for cond, files in condition_files.items():
        print(f"  {cond}: {len(files)} files")
    
    # Load and process each condition
    print("\nProcessing conditions...")
    all_data = []
    
    for cond_name, files in condition_files.items():
        if not files:
            print(f"  Skipping {cond_name}: no files")
            continue
        
        cond_info = CONDITIONS[cond_name]
        df = load_and_process_condition(cond_name, files, cond_info)
        all_data.append(df)
        
        n_events = df['events'].sum()
        n_tracks = df['track'].nunique()
        print(f"  {cond_name}: {len(files)} files, {n_tracks} tracks, {n_events} events, {len(df):,} frames")
    
    # Combine all conditions
    pooled = pd.concat(all_data, ignore_index=True)
    
    print("\n" + "=" * 70)
    print("POOLED DATASET SUMMARY")
    print("=" * 70)
    print(f"Total frames: {len(pooled):,}")
    print(f"Total events: {pooled['events'].sum():,}")
    print(f"Total tracks: {pooled['track'].nunique()}")
    print(f"Total experiments: {pooled['experiment'].nunique()}")
    
    # Verify factorial structure
    print("\nFactorial structure:")
    for cond in pooled['condition'].unique():
        subset = pooled[pooled['condition'] == cond]
        I_val = subset['I'].iloc[0]
        T_val = subset['T'].iloc[0]
        n_events = subset['events'].sum()
        n_tracks = subset['track'].nunique()
        print(f"  {cond}: I={I_val}, T={T_val}, {n_tracks} tracks, {n_events} events")
    
    # Check kernel values
    print("\nKernel value statistics:")
    print(f"  K_on range: [{pooled['K_on'].min():.3f}, {pooled['K_on'].max():.3f}]")
    print(f"  K_off range: [{pooled['K_off'].min():.3f}, {pooled['K_off'].max():.3f}]")
    print(f"  K_on non-zero: {(pooled['K_on'] != 0).sum():,} frames")
    print(f"  K_off non-zero: {(pooled['K_off'] != 0).sum():,} frames")
    
    # Save to parquet
    output_path = Path('data/processed/factorial_design_matrix.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Also save a summary
    summary = {
        'n_frames': len(pooled),
        'n_events': int(pooled['events'].sum()),
        'n_tracks': pooled['track'].nunique(),
        'n_experiments': pooled['experiment'].nunique(),
        'conditions': {
            cond: {
                'I': int(pooled[pooled['condition'] == cond]['I'].iloc[0]),
                'T': int(pooled[pooled['condition'] == cond]['T'].iloc[0]),
                'n_events': int(pooled[pooled['condition'] == cond]['events'].sum()),
                'n_tracks': pooled[pooled['condition'] == cond]['track'].nunique()
            }
            for cond in pooled['condition'].unique()
        },
        'kernel_params': KERNEL_PARAMS
    }
    
    summary_path = Path('data/processed/factorial_design_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    return pooled


if __name__ == '__main__':
    main()
