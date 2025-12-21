#!/usr/bin/env python3
"""
Time-Rescaling Test for Hazard Model Validation

Implements the time-rescaling theorem to validate the Poisson assumption:
1. Transform event times by cumulative hazard: τᵢ = Λ(tᵢ)
2. Compute rescaled IEIs: Δτᵢ = τᵢ - τᵢ₋₁
3. Test if Δτ ~ Exponential(1) via KS test and Q-Q plot

If the model is correct, rescaled IEIs should follow Exp(1).

References:
- Brown et al. (2002) "Time-rescaling methods for point process likelihoods"
- Ogata (1988) "Statistical models for earthquake occurrences"

Usage:
    python scripts/time_rescaling_test.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from load_fitting_data import (
    load_fitting_dataset,
    get_event_times,
    get_filtered_events,
    get_led_timing,
    get_track_intercepts,
    LED_ON_DURATION,
    LED_OFF_DURATION,
    LED_CYCLE,
    FIRST_LED_ONSET
)
from analytic_hazard import AnalyticHazardModel, KernelParams


def compute_cumulative_hazard(
    model: AnalyticHazardModel,
    times: np.ndarray,
    led_onsets: np.ndarray,
    led_offsets: np.ndarray,
    track_intercept: float = 0.0,
    dt: float = 0.05
) -> np.ndarray:
    """
    Compute cumulative hazard Λ(t) = ∫₀ᵗ λ(s) ds at each event time.
    
    Parameters
    ----------
    model : AnalyticHazardModel
        The fitted hazard model
    times : ndarray
        Event times to evaluate
    led_onsets : ndarray
        LED onset times
    led_offsets : ndarray
        LED offset times
    track_intercept : float
        Track-specific intercept adjustment
    dt : float
        Integration step size (seconds)
    
    Returns
    -------
    Lambda : ndarray
        Cumulative hazard at each event time
    """
    if len(times) == 0:
        return np.array([])
    
    max_time = times.max()
    t_grid = np.arange(0, max_time + dt, dt)
    
    # Compute hazard on grid
    hazard_grid = np.zeros(len(t_grid))
    
    for i, t in enumerate(t_grid):
        # Time since last LED onset
        past_onsets = led_onsets[led_onsets <= t]
        past_offsets = led_offsets[led_offsets <= t]
        
        t_since_onset = t - past_onsets[-1] if len(past_onsets) > 0 else -1
        t_since_offset = t - past_offsets[-1] if len(past_offsets) > 0 else -1
        
        # LED state
        led_on = False
        if len(past_onsets) > 0:
            if len(past_offsets) == 0:
                led_on = True
            else:
                led_on = past_onsets[-1] > past_offsets[-1]
        
        # Compute hazard (per frame, then convert to per second)
        hazard_grid[i] = model.compute_hazard(
            t_since_onset, t_since_offset, track_intercept, led_on,
            per_second=True
        )
    
    # Integrate to get cumulative hazard
    cumulative = np.cumsum(hazard_grid) * dt
    
    # Interpolate to event times
    Lambda = np.interp(times, t_grid, cumulative)
    
    return Lambda


def time_rescaling_test(
    event_times: np.ndarray,
    cumulative_hazard: np.ndarray
) -> Dict:
    """
    Perform time-rescaling test.
    
    If the model is correct:
    - Rescaled IEIs Δτ = Λ(tᵢ) - Λ(tᵢ₋₁) should be Exp(1)
    - Equivalently, 1 - exp(-Δτ) should be Uniform(0,1)
    
    Parameters
    ----------
    event_times : ndarray
        Original event times
    cumulative_hazard : ndarray
        Cumulative hazard Λ(tᵢ) at each event
    
    Returns
    -------
    results : dict
        Test statistics and p-values
    """
    # Compute rescaled IEIs
    rescaled_ieis = np.diff(cumulative_hazard)
    
    # Filter out any non-positive values
    rescaled_ieis = rescaled_ieis[rescaled_ieis > 0]
    
    if len(rescaled_ieis) < 10:
        return {
            'n_events': len(event_times),
            'n_rescaled_ieis': len(rescaled_ieis),
            'error': 'Too few rescaled IEIs for test'
        }
    
    # Transform to uniform via 1 - exp(-Δτ)
    uniform_vals = 1 - np.exp(-rescaled_ieis)
    
    # KS test against Exp(1)
    ks_exp = stats.kstest(rescaled_ieis, 'expon', args=(0, 1))
    
    # KS test against Uniform(0,1)
    ks_unif = stats.kstest(uniform_vals, 'uniform')
    
    # Compute summary statistics
    results = {
        'n_events': int(len(event_times)),
        'n_rescaled_ieis': int(len(rescaled_ieis)),
        'rescaled_iei_mean': float(np.mean(rescaled_ieis)),
        'rescaled_iei_std': float(np.std(rescaled_ieis)),
        'expected_mean': 1.0,  # Exp(1) has mean 1
        'ks_exp_stat': float(ks_exp.statistic),
        'ks_exp_pval': float(ks_exp.pvalue),
        'ks_unif_stat': float(ks_unif.statistic),
        'ks_unif_pval': float(ks_unif.pvalue),
        'pass_exp': bool(ks_exp.pvalue > 0.05),
        'pass_unif': bool(ks_unif.pvalue > 0.05),
    }
    
    return results


def plot_time_rescaling(
    rescaled_ieis: np.ndarray,
    results: Dict,
    output_path: Path
):
    """Create diagnostic plots for time-rescaling test."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Q-Q plot against Exp(1)
    ax = axes[0, 0]
    sorted_ieis = np.sort(rescaled_ieis)
    n = len(sorted_ieis)
    theoretical_quantiles = stats.expon.ppf(np.arange(1, n+1) / (n+1))
    
    ax.scatter(theoretical_quantiles, sorted_ieis, alpha=0.5, s=10)
    max_val = max(theoretical_quantiles.max(), sorted_ieis.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect fit')
    ax.set_xlabel('Theoretical Exp(1) quantiles', fontsize=12)
    ax.set_ylabel('Rescaled IEI quantiles', fontsize=12)
    ax.set_title('Q-Q Plot: Rescaled IEIs vs Exp(1)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram of rescaled IEIs
    ax = axes[0, 1]
    ax.hist(rescaled_ieis, bins=30, density=True, alpha=0.7, label='Observed')
    x = np.linspace(0, rescaled_ieis.max(), 100)
    ax.plot(x, stats.expon.pdf(x), 'r-', linewidth=2, label='Exp(1)')
    ax.set_xlabel('Rescaled IEI', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Rescaled IEI Distribution (KS p={results["ks_exp_pval"]:.3f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Uniform transform Q-Q
    ax = axes[1, 0]
    uniform_vals = 1 - np.exp(-rescaled_ieis)
    sorted_unif = np.sort(uniform_vals)
    theoretical_unif = np.arange(1, n+1) / (n+1)
    
    ax.scatter(theoretical_unif, sorted_unif, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect fit')
    ax.set_xlabel('Theoretical Uniform(0,1) quantiles', fontsize=12)
    ax.set_ylabel('Transformed quantiles', fontsize=12)
    ax.set_title('Q-Q Plot: Uniform Transform', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    pass_exp = "PASS" if results['pass_exp'] else "FAIL"
    pass_unif = "PASS" if results['pass_unif'] else "FAIL"
    
    summary = f"""
TIME-RESCALING TEST RESULTS
===========================

Events analyzed: {results['n_events']}
Rescaled IEIs: {results['n_rescaled_ieis']}

Rescaled IEI statistics:
  Mean: {results['rescaled_iei_mean']:.3f} (expected: 1.0)
  Std:  {results['rescaled_iei_std']:.3f} (expected: 1.0)

Kolmogorov-Smirnov Tests:
  vs Exp(1):      stat={results['ks_exp_stat']:.3f}, p={results['ks_exp_pval']:.4f} [{pass_exp}]
  vs Uniform(0,1): stat={results['ks_unif_stat']:.3f}, p={results['ks_unif_pval']:.4f} [{pass_unif}]

Interpretation:
  p > 0.05 suggests Poisson assumption holds
  p < 0.05 suggests temporal structure not captured by model
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def run_time_rescaling_by_track(
    data: pd.DataFrame,
    model: AnalyticHazardModel,
    led_onsets: np.ndarray,
    led_offsets: np.ndarray,
    track_intercepts: Dict[int, float],
    use_filtered: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Run time-rescaling test across all tracks.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset
    model : AnalyticHazardModel
        Fitted model
    led_onsets, led_offsets : ndarray
        LED timing
    track_intercepts : dict
        Per-track intercepts
    use_filtered : bool
        If True, use filtered events (duration > 0.1s)
    
    Returns
    -------
    all_rescaled_ieis : ndarray
        All rescaled IEIs across tracks
    results : dict
        Test results
    """
    if use_filtered:
        events = get_filtered_events(data, min_duration=0.1)
    else:
        events = get_event_times(data)
    
    all_rescaled_ieis = []
    
    for track_id in events['track_id'].unique():
        track_events = events[events['track_id'] == track_id]['time'].values
        track_events = np.sort(track_events)
        
        if len(track_events) < 2:
            continue
        
        # Get track intercept
        intercept = track_intercepts.get(track_id, 0.0)
        
        # Compute cumulative hazard
        Lambda = compute_cumulative_hazard(
            model, track_events, led_onsets, led_offsets,
            track_intercept=intercept, dt=0.05
        )
        
        # Compute rescaled IEIs
        rescaled = np.diff(Lambda)
        all_rescaled_ieis.extend(rescaled[rescaled > 0])
    
    all_rescaled_ieis = np.array(all_rescaled_ieis)
    
    # Run test
    results = time_rescaling_test(
        events['time'].values,
        np.cumsum(np.concatenate([[0], all_rescaled_ieis]))
    )
    
    # Add event type info
    results['event_type'] = 'filtered' if use_filtered else 'all'
    
    return all_rescaled_ieis, results


def main():
    print("=" * 70)
    print("TIME-RESCALING TEST")
    print("=" * 70)
    
    # Load data
    data, model_info = load_fitting_dataset()
    led_onsets, led_offsets = get_led_timing(data)
    track_intercepts = get_track_intercepts(model_info)
    
    # Create model with calibrated parameters
    model = AnalyticHazardModel()
    
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_all = {}
    
    # Test on all events
    print("\n" + "=" * 50)
    print("TEST ON ALL EVENTS (1,407)")
    print("=" * 50)
    
    rescaled_all, results_all['all_events'] = run_time_rescaling_by_track(
        data, model, led_onsets, led_offsets, track_intercepts, use_filtered=False
    )
    
    print(f"  Rescaled IEIs: {len(rescaled_all)}")
    print(f"  Mean: {results_all['all_events']['rescaled_iei_mean']:.3f} (expected: 1.0)")
    print(f"  KS vs Exp(1): stat={results_all['all_events']['ks_exp_stat']:.3f}, p={results_all['all_events']['ks_exp_pval']:.4f}")
    print(f"  Result: {'PASS' if results_all['all_events']['pass_exp'] else 'FAIL'}")
    
    # Test on filtered events
    print("\n" + "=" * 50)
    print("TEST ON FILTERED EVENTS (duration > 0.1s)")
    print("=" * 50)
    
    rescaled_filtered, results_all['filtered_events'] = run_time_rescaling_by_track(
        data, model, led_onsets, led_offsets, track_intercepts, use_filtered=True
    )
    
    print(f"  Rescaled IEIs: {len(rescaled_filtered)}")
    print(f"  Mean: {results_all['filtered_events']['rescaled_iei_mean']:.3f} (expected: 1.0)")
    print(f"  KS vs Exp(1): stat={results_all['filtered_events']['ks_exp_stat']:.3f}, p={results_all['filtered_events']['ks_exp_pval']:.4f}")
    print(f"  Result: {'PASS' if results_all['filtered_events']['pass_exp'] else 'FAIL'}")
    
    # Save results
    with open(output_dir / 'time_rescaling.json', 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved results to {output_dir / 'time_rescaling.json'}")
    
    # Plot for all events
    if len(rescaled_all) > 10:
        plot_time_rescaling(
            rescaled_all,
            results_all['all_events'],
            output_dir / 'time_rescaling_all.png'
        )
    
    # Plot for filtered events
    if len(rescaled_filtered) > 10:
        plot_time_rescaling(
            rescaled_filtered,
            results_all['filtered_events'],
            output_dir / 'time_rescaling_filtered.png'
        )
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if results_all['all_events']['pass_exp']:
        print("  All events: Poisson assumption SUPPORTED")
    else:
        print("  All events: Poisson assumption VIOLATED")
        print("    -> Consider adding refractory component")
    
    if results_all['filtered_events']['pass_exp']:
        print("  Filtered events: Poisson assumption SUPPORTED")
    else:
        print("  Filtered events: Poisson assumption VIOLATED")
    
    print("\nTime-rescaling test complete!")


if __name__ == '__main__':
    main()


