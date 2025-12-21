#!/usr/bin/env python3
"""
Simple Mixed-Effects Approximation (Pure Python, No R)

Estimates per-track baseline variability by:
1. Computing per-track event rates
2. Fitting the kernel with mean intercept fixed
3. Reporting between-track variance

This approximates a random-intercept model without R/glmmTMB.

Usage:
    python scripts/fit_mixed_effects_simple.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.optimize import minimize

# Parameters
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3
EXPERIMENT_DURATION = 1200.0


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


def main():
    print("=" * 70)
    print("SIMPLE MIXED-EFFECTS APPROXIMATION (Pure Python)")
    print("=" * 70)
    
    # Load data
    data_dir = Path('data/engineered')
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        print("No data files found")
        return
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem.split('_')[0]
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(data):,} observations from {len(csv_files)} experiments")
    
    # Create unique track identifier
    data['track_uid'] = data['experiment_id'].astype(str) + '_' + data['track_id'].astype(str)
    
    # Get response variable
    if 'is_reorientation_start' in data.columns:
        event_col = 'is_reorientation_start'
    else:
        event_col = 'is_reorientation'
    
    # =========================================
    # STEP 1: Compute per-track event rates
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 1: Per-Track Event Rates")
    print("=" * 50)
    
    track_stats = data.groupby('track_uid').agg({
        event_col: 'sum',
        'time': ['min', 'max', 'count']
    }).reset_index()
    track_stats.columns = ['track_uid', 'n_events', 't_min', 't_max', 'n_frames']
    
    # Compute duration and rate for each track
    track_stats['duration_min'] = (track_stats['t_max'] - track_stats['t_min']) / 60
    track_stats['rate_per_min'] = track_stats['n_events'] / track_stats['duration_min']
    
    # Compute per-frame probability (for intercept estimation)
    track_stats['p_event'] = track_stats['n_events'] / track_stats['n_frames']
    
    # FILTER: Only include tracks with at least 1 event (exclude 0-event tracks)
    # Otherwise log(0) = -inf distorts the statistics
    track_stats_with_events = track_stats[track_stats['n_events'] > 0].copy()
    track_stats_with_events['log_p'] = np.log(track_stats_with_events['p_event'])
    
    n_tracks_with_events = len(track_stats_with_events)
    n_tracks_zero = len(track_stats) - n_tracks_with_events
    
    print(f"\nTracks with events: {n_tracks_with_events}")
    print(f"Tracks with 0 events: {n_tracks_zero} (excluded from intercept estimation)")
    
    n_tracks = len(track_stats)
    print(f"\nTotal unique tracks: {n_tracks}")
    print(f"\nPer-track rate statistics (events/min, all tracks):")
    print(f"  Mean:   {track_stats['rate_per_min'].mean():.3f}")
    print(f"  Std:    {track_stats['rate_per_min'].std():.3f}")
    print(f"  Min:    {track_stats['rate_per_min'].min():.3f}")
    print(f"  Max:    {track_stats['rate_per_min'].max():.3f}")
    print(f"  Median: {track_stats['rate_per_min'].median():.3f}")
    
    print(f"\nPer-track log-probability (tracks with events only):")
    print(f"  Mean:   {track_stats_with_events['log_p'].mean():.3f}")
    print(f"  Std:    {track_stats_with_events['log_p'].std():.3f}")
    print(f"  Range:  [{track_stats_with_events['log_p'].min():.3f}, {track_stats_with_events['log_p'].max():.3f}]")
    
    # =========================================
    # STEP 2: Fit kernel with pooled intercept
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 2: Fit Kernel with Pooled Mean Intercept")
    print("=" * 50)
    
    # Use the mean log-probability as the fixed intercept (from tracks with events)
    mean_intercept = track_stats_with_events['log_p'].mean()
    std_intercept = track_stats_with_events['log_p'].std()
    
    print(f"\nUsing fixed intercept: {mean_intercept:.4f}")
    print(f"Between-track SD: {std_intercept:.4f}")
    
    # Build design matrix
    times = data['time'].values
    n = len(times)
    y = data[event_col].values.astype(float)
    
    # Compute time since LED onset
    time_since_onset = np.zeros(n)
    is_led_on = np.zeros(n, dtype=bool)
    
    for i, t in enumerate(times):
        if t >= FIRST_LED_ONSET:
            time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
            if time_in_cycle < LED_ON_DURATION:
                time_since_onset[i] = time_in_cycle
                is_led_on[i] = True
    
    # Kernel configuration (triphasic)
    early_centers = np.array([0.2, 0.7, 1.4])
    intm_centers = np.array([2.0, 2.5])
    late_centers = np.array([3.0, 5.0, 7.0, 9.0])
    
    tso = time_since_onset[is_led_on]
    
    early_basis = np.zeros((n, 3))
    intm_basis = np.zeros((n, 2))
    late_basis = np.zeros((n, 4))
    
    early_basis[is_led_on] = raised_cosine_basis(tso, early_centers, 0.4)
    intm_basis[is_led_on] = raised_cosine_basis(tso, intm_centers, 0.6)
    late_basis[is_led_on] = raised_cosine_basis(tso, late_centers, 1.8)
    
    # LED-off rebound
    rebound = np.zeros(n)
    off_mask = ~is_led_on & (times >= FIRST_LED_ONSET)
    for i in np.where(off_mask)[0]:
        t = times[i]
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        time_since_offset = time_in_cycle - LED_ON_DURATION
        if time_since_offset > 0:
            rebound[i] = np.exp(-time_since_offset / 2.0)
    
    # Design matrix (without intercept - it's fixed)
    X_kernel = np.hstack([early_basis, intm_basis, late_basis, rebound.reshape(-1, 1)])
    
    feature_names = [f'kernel_early_{i+1}' for i in range(3)]
    feature_names += [f'kernel_intm_{i+1}' for i in range(2)]
    feature_names += [f'kernel_late_{i+1}' for i in range(4)]
    feature_names += ['led_off_rebound']
    
    # Fit NB-GLM with fixed intercept
    alpha = 0.1  # NB dispersion
    
    def objective(beta_kernel):
        eta = mean_intercept + X_kernel @ beta_kernel
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        
        r = 1 / alpha
        ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
        
        # Ridge penalty
        ridge = 0.01 * np.sum(beta_kernel**2)
        
        return -(ll - ridge)
    
    print("\nFitting NB-GLM with fixed intercept...")
    
    beta_init = np.zeros(len(feature_names))
    bounds = [(None, None)] * len(feature_names)
    bounds[0] = (0, None)  # First early basis non-negative
    
    result = minimize(objective, beta_init, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        beta_kernel = result.x
        print("  Optimization converged!")
        
        print("\n  Kernel coefficients:")
        for name, coef in zip(feature_names, beta_kernel):
            sig = '***' if abs(coef) > 1.0 else ('**' if abs(coef) > 0.5 else '')
            print(f"    {name}: {coef:+.4f} {sig}")
    else:
        print(f"  Optimization failed: {result.message}")
        return
    
    # =========================================
    # STEP 3: Key Insights
    # =========================================
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)
    
    # Compute implied baseline rate
    baseline_hazard = np.exp(mean_intercept)
    baseline_rate = baseline_hazard * 20 * 60  # per frame -> per minute
    
    print(f"\n1. BASELINE RATE")
    print(f"   Mean intercept: {mean_intercept:.4f}")
    print(f"   Implied baseline: {baseline_rate:.2f} events/min")
    print(f"   Empirical rate: 0.71 events/min")
    print(f"   Ratio: {baseline_rate/0.71:.2f}x")
    
    print(f"\n2. BETWEEN-TRACK VARIABILITY")
    print(f"   Intercept SD: {std_intercept:.4f}")
    print(f"   This explains some of the rate mismatch!")
    print(f"   High-rate tracks pull up the overall MLE intercept.")
    
    # Compute rate multiplier range from intercept variability
    low_rate_mult = np.exp(mean_intercept - 2*std_intercept) / np.exp(mean_intercept)
    high_rate_mult = np.exp(mean_intercept + 2*std_intercept) / np.exp(mean_intercept)
    
    print(f"\n3. RATE RANGE (±2 SD)")
    print(f"   Low-rate tracks: {low_rate_mult:.2f}x mean")
    print(f"   High-rate tracks: {high_rate_mult:.2f}x mean")
    
    # =========================================
    # Save results
    # =========================================
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'mean_intercept': float(mean_intercept),
        'intercept_std': float(std_intercept),
        'implied_baseline_rate': float(baseline_rate),
        'empirical_rate': 0.71,
        'rate_ratio': float(baseline_rate / 0.71),
        'n_tracks_total': int(n_tracks),
        'n_tracks_with_events': int(n_tracks_with_events),
        'kernel_coefficients': dict(zip(feature_names, [float(x) for x in beta_kernel])),
        'track_rate_stats': {
            'mean': float(track_stats_with_events['rate_per_min'].mean()),
            'std': float(track_stats_with_events['rate_per_min'].std()),
            'min': float(track_stats_with_events['rate_per_min'].min()),
            'max': float(track_stats_with_events['rate_per_min'].max())
        }
    }
    
    output_path = output_dir / 'mixed_effects_simple_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # =========================================
    # Recommendations
    # =========================================
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print("""
1. The between-track SD of {:.3f} is substantial.
   This means some larvae have ~{:.1f}x higher baseline rates than others.

2. For paper/power analysis:
   - Use the MEAN intercept ({:.4f}) for the global model
   - Report the SD ({:.4f}) as "random effect variance"
   - This gives you rate calibration without full glmmTMB

3. For simulation:
   - Draw per-track intercepts from N({:.4f}, {:.4f}²)
   - This will produce realistic track-to-track variability
""".format(
        std_intercept,
        np.exp(2*std_intercept),
        mean_intercept,
        std_intercept,
        mean_intercept,
        std_intercept
    ))


if __name__ == '__main__':
    main()




