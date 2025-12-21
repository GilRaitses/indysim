#!/usr/bin/env python3
"""
Simulate Events Using Biphasic LNP Model

Generates synthetic reorientation events using the fitted biphasic kernel model.

Usage:
    python scripts/simulate_biphasic.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict


def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """Compute raised-cosine basis functions."""
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def find_led_onsets(led_pattern: Callable, t_scan: np.ndarray, threshold: float = 50) -> np.ndarray:
    """Find LED onset times."""
    led_vals = np.array([led_pattern(t) for t in t_scan])
    led_on = led_vals > threshold
    onsets = []
    for i in range(1, len(led_on)):
        if led_on[i] and not led_on[i-1]:
            onsets.append(t_scan[i])
    return np.array(onsets)


def find_led_offsets(led_pattern: Callable, t_scan: np.ndarray, threshold: float = 50) -> np.ndarray:
    """Find LED offset times."""
    led_vals = np.array([led_pattern(t) for t in t_scan])
    led_on = led_vals > threshold
    offsets = []
    for i in range(1, len(led_on)):
        if not led_on[i] and led_on[i-1]:
            offsets.append(t_scan[i])
    return np.array(offsets)


def make_biphasic_hazard(
    model_results: Dict,
    led_pattern: Callable,
    split_point: float = 1.5
) -> Callable:
    """
    Create hazard function from biphasic model.
    """
    coefs = model_results['coefficients']
    
    # Extract coefficients
    intercept = coefs.get('intercept', -6.7)
    beta_led = coefs.get('LED1_scaled', 0)
    beta_rebound = coefs.get('led_off_rebound', 0)
    
    # Early kernel coefficients (3 bases)
    early_coefs = [coefs.get(f'kernel_early_{i+1}', 0) for i in range(3)]
    early_centers = np.linspace(0, split_point, 3)
    early_width = split_point / 2 * 0.8
    
    # Late kernel coefficients (4 bases)
    late_coefs = [coefs.get(f'kernel_late_{i+1}', 0) for i in range(4)]
    late_centers = np.linspace(split_point, 6.0, 4)
    late_width = (6.0 - split_point) / 3 * 0.8
    
    # Pre-compute LED onsets and offsets
    t_scan = np.arange(0, 1500, 0.5)
    led_onsets = find_led_onsets(led_pattern, t_scan)
    led_offsets = find_led_offsets(led_pattern, t_scan)
    
    def hazard_func(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t)
        n = len(t)
        
        # Base linear predictor
        eta = np.full(n, intercept)
        
        # LED intensity
        led_vals = np.array([led_pattern(ti) for ti in t])
        eta += beta_led * (led_vals / 250.0)
        
        # Time since most recent LED onset
        time_since_onset = np.full(n, 999.0)
        for i, ti in enumerate(t):
            recent_onsets = led_onsets[led_onsets < ti]
            if len(recent_onsets) > 0:
                time_since_onset[i] = ti - recent_onsets[-1]
        
        # Early kernel
        early_basis = raised_cosine_basis(time_since_onset, early_centers, early_width)
        for j, phi in enumerate(early_coefs):
            eta += phi * early_basis[:, j]
        
        # Late kernel
        late_basis = raised_cosine_basis(time_since_onset, late_centers, late_width)
        for j, phi in enumerate(late_coefs):
            eta += phi * late_basis[:, j]
        
        # LED-off rebound
        time_since_offset = np.full(n, 999.0)
        for i, ti in enumerate(t):
            recent_offsets = led_offsets[led_offsets < ti]
            if len(recent_offsets) > 0:
                time_since_offset[i] = ti - recent_offsets[-1]
        
        rebound = np.where(time_since_offset < 999, np.exp(-time_since_offset / 1.5), 0)
        eta += beta_rebound * rebound
        
        # Hazard rate
        lam = np.exp(eta)
        return lam
    
    return hazard_func


def generate_events_inversion(
    hazard_func: Callable,
    t_start: float,
    t_end: float,
    dt: float = 0.05,
    frame_rate: float = 20.0,  # Model was fit at 20 Hz (50ms frames)
    rng: np.random.Generator = None
) -> np.ndarray:
    """Generate events using inversion method."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Precompute hazard on grid
    t_grid = np.arange(t_start, t_end, dt)
    lambda_grid = hazard_func(t_grid)
    lambda_grid = np.maximum(lambda_grid, 0)
    
    # Convert from per-frame rate to per-second rate
    # Model was fit on per-frame data, so lambda is probability per frame
    # Multiply by frame_rate to get rate per second
    lambda_grid = lambda_grid * frame_rate
    
    # Cumulative hazard
    H_grid = np.zeros(len(t_grid))
    H_grid[1:] = np.cumsum(0.5 * (lambda_grid[:-1] + lambda_grid[1:]) * dt)
    
    # Generate events
    events = []
    current_time = t_start
    
    while current_time < t_end:
        # Draw waiting time
        u = rng.random()
        target_H = -np.log(1 - u) if u < 1 else 1e10
        
        # Find current H
        idx = np.searchsorted(t_grid, current_time)
        if idx >= len(H_grid):
            break
        
        current_H = H_grid[idx]
        target_H_abs = current_H + target_H
        
        # Find time when H reaches target
        event_idx = np.searchsorted(H_grid, target_H_abs)
        
        if event_idx >= len(t_grid):
            break
        
        event_time = t_grid[event_idx]
        events.append(event_time)
        current_time = event_time + 0.1  # Small refractory
    
    return np.array(events)


def simulate_experiment(
    model_results: Dict,
    n_tracks: int = 50,
    duration: float = 1200.0,
    intensity: float = 250.0,
    pulse_duration: float = 10.0,
    gap_duration: float = 20.0,
    first_onset: float = 21.0,
    seed: int = None
) -> pd.DataFrame:
    """Generate synthetic experiment with biphasic model."""
    
    rng = np.random.default_rng(seed)
    cycle_period = pulse_duration + gap_duration
    
    def led_pattern(t):
        if t < first_onset:
            return 0.0
        time_in_cycles = t - first_onset
        cycle_position = time_in_cycles % cycle_period
        if cycle_position < pulse_duration:
            return intensity
        return 0.0
    
    # Create hazard function
    hazard_func = make_biphasic_hazard(model_results, led_pattern, split_point=1.5)
    
    # Generate events for each track
    all_events = []
    
    for track_id in range(n_tracks):
        event_times = generate_events_inversion(hazard_func, 0, duration, rng=rng)
        
        for t in event_times:
            all_events.append({
                'track_id': track_id,
                'time': t,
                'led1Val': led_pattern(t),
                'led2Val': 7.0,
                'is_reorientation': True
            })
    
    return pd.DataFrame(all_events)


def main():
    print("=" * 60)
    print("SIMULATE WITH BIPHASIC MODEL")
    print("=" * 60)
    
    # Load biphasic model
    model_path = Path('data/model/biphasic_model_results.json')
    print(f"\nLoading model from {model_path}...")
    
    with open(model_path, 'r') as f:
        model_results = json.load(f)
    
    print(f"  Split point: {model_results.get('split_point', 1.5)}s")
    print(f"  AIC: {model_results.get('aic', 'N/A')}")
    
    # Generate 2 experiments
    print("\nGenerating synthetic experiments...")
    
    all_dfs = []
    for exp_id in range(2):
        df = simulate_experiment(
            model_results,
            n_tracks=50,
            duration=1200.0,
            seed=42 + exp_id
        )
        df['experiment_id'] = f'biphasic_exp_{exp_id:03d}'
        all_dfs.append(df)
        print(f"  Experiment {exp_id+1}: {len(df)} events")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    output_path = Path('data/simulated/synthetic_events_biphasic.csv')
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} events to {output_path}")
    
    # Summary stats
    n_tracks = combined['track_id'].nunique()
    duration_min = 20.0  # 1200s / 60
    rate = len(combined) / (n_tracks * 2) / duration_min
    print(f"\nTurn rate: {rate:.2f} events/min/track")


if __name__ == '__main__':
    main()




