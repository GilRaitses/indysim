#!/usr/bin/env python3
"""
Extract Turn Distributions

Extracts empirical turn angle and duration distributions from the fitting dataset
for use in trajectory simulation.

Usage:
    python scripts/extract_turn_distributions.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from load_fitting_data import load_fitting_dataset, get_filtered_events


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def compute_turn_angles(data: pd.DataFrame, min_duration: float = 0.1) -> np.ndarray:
    """
    Compute turn angles (heading change) for filtered events.
    
    For each event with duration > min_duration:
    1. Find the heading at event start
    2. Find the heading at event end (start + turn_duration)
    3. Compute Δθ = heading_end - heading_start (wrapped to [-π, π])
    
    Parameters
    ----------
    data : DataFrame
        Frame-level data with columns: track_id, time, heading, 
        is_reorientation_start, turn_duration
    min_duration : float
        Minimum turn duration to include
    
    Returns
    -------
    turn_angles : ndarray
        Turn angles in radians
    """
    # Get filtered events
    events = data[
        (data['is_reorientation_start'] == True) & 
        (data['turn_duration'] > min_duration)
    ].copy()
    
    turn_angles = []
    
    for idx, event in events.iterrows():
        track_id = event['track_id']
        t_start = event['time']
        duration = event['turn_duration']
        t_end = t_start + duration
        
        # Get track data
        track_data = data[data['track_id'] == track_id].copy()
        track_data = track_data.sort_values('time')
        
        # Find heading at start and end
        # Use nearest frame within small tolerance
        dt = 0.05  # Frame interval
        
        start_mask = (track_data['time'] >= t_start - dt) & (track_data['time'] <= t_start + dt)
        end_mask = (track_data['time'] >= t_end - dt) & (track_data['time'] <= t_end + dt)
        
        if start_mask.sum() > 0 and end_mask.sum() > 0:
            heading_start = track_data[start_mask]['heading'].iloc[0]
            heading_end = track_data[end_mask]['heading'].iloc[0]
            
            # Compute wrapped angle difference
            delta_theta = wrap_angle(heading_end - heading_start)
            turn_angles.append(delta_theta)
    
    return np.array(turn_angles)


def fit_turn_angle_distribution(angles: np.ndarray) -> Dict:
    """
    Fit distribution to turn angles.
    
    Tries von Mises and wrapped Cauchy; selects best by log-likelihood.
    Also fits a simple Gaussian for comparison.
    """
    results = {}
    
    # Basic statistics
    results['n_samples'] = len(angles)
    results['mean'] = float(np.mean(angles))
    results['std'] = float(np.std(angles))
    results['median'] = float(np.median(angles))
    results['abs_mean'] = float(np.mean(np.abs(angles)))
    
    # Fit von Mises (circular normal)
    # vonmises.fit returns (kappa, loc, scale) but scale is always 1 for von Mises
    try:
        kappa, loc, _ = stats.vonmises.fit(angles, fscale=1)
        ll_vm = np.sum(stats.vonmises.logpdf(angles, kappa, loc=loc))
        results['vonmises'] = {
            'kappa': float(kappa),
            'loc': float(loc),
            'log_likelihood': float(ll_vm)
        }
    except Exception as e:
        results['vonmises'] = {'error': str(e)}
    
    # Fit normal (for comparison)
    mu, sigma = stats.norm.fit(angles)
    ll_norm = np.sum(stats.norm.logpdf(angles, mu, sigma))
    results['normal'] = {
        'mu': float(mu),
        'sigma': float(sigma),
        'log_likelihood': float(ll_norm)
    }
    
    # Determine best distribution
    if 'log_likelihood' in results.get('vonmises', {}):
        if results['vonmises']['log_likelihood'] > results['normal']['log_likelihood']:
            results['best_fit'] = 'vonmises'
        else:
            results['best_fit'] = 'normal'
    else:
        results['best_fit'] = 'normal'
    
    return results


def fit_turn_duration_distribution(durations: np.ndarray) -> Dict:
    """
    Fit distribution to turn durations.
    
    Tries exponential, gamma, and lognormal.
    """
    results = {}
    
    # Filter positive durations
    durations = durations[durations > 0]
    
    results['n_samples'] = len(durations)
    results['mean'] = float(np.mean(durations))
    results['std'] = float(np.std(durations))
    results['median'] = float(np.median(durations))
    results['min'] = float(np.min(durations))
    results['max'] = float(np.max(durations))
    
    # Fit exponential
    loc, scale = stats.expon.fit(durations, floc=0)
    ll_exp = np.sum(stats.expon.logpdf(durations, loc=loc, scale=scale))
    results['exponential'] = {
        'loc': float(loc),
        'scale': float(scale),
        'log_likelihood': float(ll_exp)
    }
    
    # Fit gamma
    a, loc, scale = stats.gamma.fit(durations, floc=0)
    ll_gamma = np.sum(stats.gamma.logpdf(durations, a, loc=loc, scale=scale))
    results['gamma'] = {
        'shape': float(a),
        'loc': float(loc),
        'scale': float(scale),
        'log_likelihood': float(ll_gamma)
    }
    
    # Fit lognormal
    s, loc, scale = stats.lognorm.fit(durations, floc=0)
    ll_ln = np.sum(stats.lognorm.logpdf(durations, s, loc=loc, scale=scale))
    results['lognormal'] = {
        's': float(s),
        'loc': float(loc),
        'scale': float(scale),
        'log_likelihood': float(ll_ln)
    }
    
    # Best fit
    lls = {
        'exponential': ll_exp,
        'gamma': ll_gamma,
        'lognormal': ll_ln
    }
    results['best_fit'] = max(lls, key=lls.get)
    
    return results


def plot_distributions(
    angles: np.ndarray,
    durations: np.ndarray,
    angle_fit: Dict,
    duration_fit: Dict,
    output_path: Path
):
    """Plot turn angle and duration distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Turn angle distribution
    ax = axes[0]
    ax.hist(angles, bins=30, density=True, alpha=0.7, label='Empirical')
    
    x = np.linspace(-np.pi, np.pi, 200)
    if 'vonmises' in angle_fit and 'kappa' in angle_fit['vonmises']:
        kappa = angle_fit['vonmises']['kappa']
        loc = angle_fit['vonmises']['loc']
        ax.plot(x, stats.vonmises.pdf(x, kappa, loc=loc), 'r-', linewidth=2, 
                label=f'von Mises (κ={kappa:.2f})')
    
    mu, sigma = angle_fit['normal']['mu'], angle_fit['normal']['sigma']
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'g--', linewidth=2,
            label=f'Normal (σ={sigma:.2f} rad)')
    
    ax.set_xlabel('Turn angle (rad)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Turn Angle Distribution (n={len(angles)})', fontsize=14)
    ax.legend()
    ax.set_xlim(-np.pi, np.pi)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Turn duration distribution
    ax = axes[1]
    ax.hist(durations, bins=30, density=True, alpha=0.7, label='Empirical')
    
    x = np.linspace(0, durations.max(), 200)
    
    # Plot best fit
    best = duration_fit['best_fit']
    if best == 'gamma':
        params = duration_fit['gamma']
        y = stats.gamma.pdf(x, params['shape'], loc=params['loc'], scale=params['scale'])
        ax.plot(x, y, 'r-', linewidth=2, 
                label=f'Gamma (α={params["shape"]:.2f}, β={params["scale"]:.2f})')
    elif best == 'lognormal':
        params = duration_fit['lognormal']
        y = stats.lognorm.pdf(x, params['s'], loc=params['loc'], scale=params['scale'])
        ax.plot(x, y, 'r-', linewidth=2,
                label=f'Lognormal (s={params["s"]:.2f})')
    else:
        params = duration_fit['exponential']
        y = stats.expon.pdf(x, loc=params['loc'], scale=params['scale'])
        ax.plot(x, y, 'r-', linewidth=2,
                label=f'Exponential (λ={1/params["scale"]:.2f})')
    
    ax.set_xlabel('Turn duration (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Turn Duration Distribution (n={len(durations)})', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    print("=" * 70)
    print("EXTRACT TURN DISTRIBUTIONS")
    print("=" * 70)
    
    # Load data
    data, model_info = load_fitting_dataset()
    
    # Get filtered events
    filtered = get_filtered_events(data, min_duration=0.1)
    print(f"\nFiltered events (duration > 0.1s): {len(filtered)}")
    
    # Extract turn angles
    print("\nComputing turn angles...")
    turn_angles = compute_turn_angles(data, min_duration=0.1)
    print(f"  Extracted {len(turn_angles)} turn angles")
    print(f"  Mean: {np.mean(turn_angles):.3f} rad ({np.degrees(np.mean(turn_angles)):.1f}°)")
    print(f"  Std: {np.std(turn_angles):.3f} rad ({np.degrees(np.std(turn_angles)):.1f}°)")
    print(f"  Mean |Δθ|: {np.mean(np.abs(turn_angles)):.3f} rad ({np.degrees(np.mean(np.abs(turn_angles))):.1f}°)")
    
    # Get turn durations
    turn_durations = filtered['turn_duration'].values
    print(f"\nTurn durations:")
    print(f"  Mean: {np.mean(turn_durations):.3f}s")
    print(f"  Median: {np.median(turn_durations):.3f}s")
    print(f"  Range: [{np.min(turn_durations):.3f}, {np.max(turn_durations):.3f}]s")
    
    # Fit distributions
    print("\nFitting turn angle distribution...")
    angle_fit = fit_turn_angle_distribution(turn_angles)
    print(f"  Best fit: {angle_fit['best_fit']}")
    
    print("\nFitting turn duration distribution...")
    duration_fit = fit_turn_duration_distribution(turn_durations)
    print(f"  Best fit: {duration_fit['best_fit']}")
    
    # Compile results
    results = {
        'turn_angle': {
            'n_samples': len(turn_angles),
            'mean_rad': float(np.mean(turn_angles)),
            'std_rad': float(np.std(turn_angles)),
            'mean_deg': float(np.degrees(np.mean(turn_angles))),
            'std_deg': float(np.degrees(np.std(turn_angles))),
            'abs_mean_rad': float(np.mean(np.abs(turn_angles))),
            'abs_mean_deg': float(np.degrees(np.mean(np.abs(turn_angles)))),
            'distribution': angle_fit
        },
        'turn_duration': {
            'n_samples': len(turn_durations),
            'mean': float(np.mean(turn_durations)),
            'std': float(np.std(turn_durations)),
            'median': float(np.median(turn_durations)),
            'min': float(np.min(turn_durations)),
            'max': float(np.max(turn_durations)),
            'distribution': duration_fit
        }
    }
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'turn_distributions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'turn_distributions.json'}")
    
    # Plot
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_distributions(
        turn_angles, turn_durations,
        angle_fit, duration_fit,
        figures_dir / 'turn_distributions.png'
    )
    
    # Summary for trajectory simulation
    print("\n" + "=" * 50)
    print("PARAMETERS FOR TRAJECTORY SIMULATION")
    print("=" * 50)
    
    print(f"\nTurn angle (use Normal distribution):")
    print(f"  mu = {angle_fit['normal']['mu']:.4f} rad")
    print(f"  sigma = {angle_fit['normal']['sigma']:.4f} rad")
    
    print(f"\nTurn duration (use {duration_fit['best_fit']} distribution):")
    if duration_fit['best_fit'] == 'gamma':
        p = duration_fit['gamma']
        print(f"  shape (α) = {p['shape']:.4f}")
        print(f"  scale (β) = {p['scale']:.4f}")
    elif duration_fit['best_fit'] == 'lognormal':
        p = duration_fit['lognormal']
        print(f"  s = {p['s']:.4f}")
        print(f"  scale = {p['scale']:.4f}")
    
    print("\nTurn distribution extraction complete!")


if __name__ == '__main__':
    main()


