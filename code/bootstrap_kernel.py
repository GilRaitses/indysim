#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Gamma-Difference Kernel

PROPER IMPLEMENTATION: Track-level resampling from raw data.
This respects temporal autocorrelation within tracks.

Usage:
    python scripts/bootstrap_kernel.py [--n_bootstrap 1000]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gamma as gamma_dist
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# Configuration
BO_OPTIMAL_CONFIG = {
    'early_centers': [0.2, 0.6333, 1.0667, 1.5],
    'early_width': 0.30,
    'intm_centers': [2.0, 2.5],
    'intm_width': 0.6,
    'late_centers': [3.0, 4.2, 5.4, 6.6, 7.8, 9.0],
    'late_width': 2.494,
}

LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


def gamma_diff_kernel(t: np.ndarray, A: float, alpha1: float, beta1: float,
                      B: float, alpha2: float, beta2: float) -> np.ndarray:
    """Gamma-difference kernel."""
    pdf1 = gamma_dist.pdf(t, alpha1, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha2, scale=beta2)
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def fit_gamma_diff(t: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Fit gamma-diff to kernel data. Returns (params, converged)."""
    p0 = [0.5, 2.2, 0.13, 12.0, 4.4, 0.87]
    bounds = ([0.01, 1.0, 0.05, 0.1, 2.0, 0.3], [5.0, 5.0, 0.5, 50.0, 8.0, 2.0])
    
    try:
        popt, _ = curve_fit(gamma_diff_kernel, t, K, p0=p0, bounds=bounds, maxfev=10000)
        return popt, True
    except:
        return np.array(p0), False


def compute_derived_quantities(params: np.ndarray) -> Dict:
    """Compute derived biological timescales from parameters."""
    A, alpha1, beta1, B, alpha2, beta2 = params
    
    # Peak time = (alpha - 1) * beta for alpha >= 1
    peak_fast = (alpha1 - 1) * beta1 if alpha1 > 1 else 0
    peak_slow = (alpha2 - 1) * beta2 if alpha2 > 1 else 0
    
    # Mean time (tau) = alpha * beta
    tau1 = alpha1 * beta1
    tau2 = alpha2 * beta2
    
    return {
        'peak_fast': peak_fast,
        'peak_slow': peak_slow,
        'tau1': tau1,
        'tau2': tau2,
        'mean_fast': tau1,
        'mean_slow': tau2,
        'std_fast': np.sqrt(alpha1 * beta1**2),
        'std_slow': np.sqrt(alpha2 * beta2**2),
        'amplitude_ratio': A / B if B > 0 else np.inf
    }


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


def compute_time_since_led_onset(times: np.ndarray) -> np.ndarray:
    """Compute time since last LED onset."""
    time_since_onset = np.full(len(times), -1.0)
    
    for i, t in enumerate(times):
        if t >= FIRST_LED_ONSET:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                time_since_onset[i] = cycle_time
    
    return time_since_onset


def build_design_matrix(data: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Build design matrix for GLM."""
    times = data['time'].values
    time_since_onset = compute_time_since_led_onset(times)
    
    early_basis = raised_cosine_basis(
        time_since_onset, 
        np.array(config['early_centers']), 
        config['early_width']
    )
    intm_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['intm_centers']),
        config['intm_width']
    )
    late_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['late_centers']),
        config['late_width']
    )
    
    X = np.column_stack([
        np.ones(len(data)),
        early_basis,
        intm_basis,
        late_basis
    ])
    
    y = data['is_reorientation_start'].fillna(0).values.astype(int)
    
    return X, y


def reconstruct_kernel(coeffs: np.ndarray, config: dict, t: np.ndarray) -> np.ndarray:
    """Reconstruct kernel from basis coefficients."""
    early_basis = raised_cosine_basis(t, np.array(config['early_centers']), config['early_width'])
    intm_basis = raised_cosine_basis(t, np.array(config['intm_centers']), config['intm_width'])
    late_basis = raised_cosine_basis(t, np.array(config['late_centers']), config['late_width'])
    
    basis = np.column_stack([early_basis, intm_basis, late_basis])
    return basis @ coeffs


def run_track_level_bootstrap(data: pd.DataFrame, config: dict, 
                              n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """
    PROPER track-level bootstrap.
    
    Resamples tracks (not frames) to respect temporal autocorrelation.
    For each bootstrap sample:
    1. Resample tracks with replacement
    2. Fit GLM to get kernel coefficients  
    3. Fit gamma-difference to reconstructed kernel
    4. Store all 6 parameters + derived quantities
    """
    np.random.seed(seed)
    
    t_grid = np.linspace(0.01, 10, 500)
    track_ids = data['track_id'].unique()
    n_tracks = len(track_ids)
    
    print(f"Track-level bootstrap: {n_bootstrap} iterations, {n_tracks} tracks")
    
    param_names = ['A', 'alpha1', 'beta1', 'B', 'alpha2', 'beta2']
    bootstrap_params = []
    bootstrap_derived = []
    n_failed = 0
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}")
        
        # Resample tracks with replacement
        sampled_tracks = np.random.choice(track_ids, size=n_tracks, replace=True)
        
        # Build bootstrap sample
        boot_dfs = [data[data['track_id'] == track].copy() for track in sampled_tracks]
        boot_data = pd.concat(boot_dfs, ignore_index=True)
        
        # Fit GLM
        X, y = build_design_matrix(boot_data, config)
        
        try:
            model = GLM(y, X, family=NegativeBinomial(alpha=1.0))
            result = model.fit(disp=False)
            coeffs = result.params[1:]  # Exclude intercept
        except:
            n_failed += 1
            continue
        
        # Reconstruct kernel and fit gamma
        K = reconstruct_kernel(coeffs, config, t_grid)
        params, converged = fit_gamma_diff(t_grid, K)
        
        if converged:
            bootstrap_params.append(params)
            bootstrap_derived.append(compute_derived_quantities(params))
        else:
            n_failed += 1
    
    if n_failed > 0:
        print(f"  Warning: {n_failed}/{n_bootstrap} iterations failed")
    
    bootstrap_params = np.array(bootstrap_params)
    
    # Compute statistics
    results = {
        'method': 'track_level_bootstrap',
        'n_bootstrap': n_bootstrap,
        'n_converged': len(bootstrap_params),
        'n_tracks': n_tracks,
        'parameters': {},
        'derived': {}
    }
    
    # Parameter CIs
    for i, name in enumerate(param_names):
        vals = bootstrap_params[:, i]
        results['parameters'][name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'ci_2.5': float(np.percentile(vals, 2.5)),
            'ci_97.5': float(np.percentile(vals, 97.5)),
            'median': float(np.median(vals))
        }
    
    # Derived quantity CIs (including tau1, tau2)
    derived_names = ['tau1', 'tau2', 'peak_fast', 'peak_slow', 'mean_fast', 'mean_slow']
    for name in derived_names:
        vals = [d[name] for d in bootstrap_derived if name in d]
        if vals:
            results['derived'][name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'ci_2.5': float(np.percentile(vals, 2.5)),
                'ci_97.5': float(np.percentile(vals, 97.5)),
                'median': float(np.median(vals))
            }
    
    return results, bootstrap_params


def plot_bootstrap_results(results: Dict, bootstrap_params: np.ndarray, 
                           output_path: Path):
    """Visualize bootstrap parameter distributions."""
    param_names = ['A', 'alpha1', 'beta1', 'B', 'alpha2', 'beta2']
    param_labels = ['A (fast amp)', 'α₁ (fast shape)', 'β₁ (fast scale)', 
                    'B (slow amp)', 'α₂ (slow shape)', 'β₂ (slow scale)']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, (name, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]
        vals = bootstrap_params[:, i]
        
        ax.hist(vals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        
        # Mark original value
        orig = results['original_params'][name]
        ax.axvline(orig, color='red', linestyle='-', linewidth=2, label=f'Original: {orig:.4f}')
        
        # Mark 95% CI
        ci_low = results['parameters'][name]['ci_2.5']
        ci_high = results['parameters'][name]['ci_97.5']
        ax.axvline(ci_low, color='orange', linestyle='--', linewidth=1.5, label=f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        ax.axvline(ci_high, color='orange', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name}', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Bootstrap Parameter Distributions (n={results["n_bootstrap"]})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bootstrap plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Bootstrap CIs for gamma-diff kernel')
    parser.add_argument('--n_bootstrap', type=int, default=100, 
                        help='Number of bootstrap samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR GAMMA-DIFF KERNEL")
    print("=" * 70)
    
    # Load dense kernel
    kernel_path = Path('data/model/kernel_dense.csv')
    if not kernel_path.exists():
        print(f"Kernel not found: {kernel_path}")
        return
    
    df = pd.read_csv(kernel_path)
    t = df['time'].values
    K = df['kernel_value'].values
    
    print(f"Loaded kernel: {len(t)} points")
    
    # Run bootstrap
    print(f"\nRunning {args.n_bootstrap} bootstrap iterations...")
    results, bootstrap_params = run_bootstrap(t, K, n_bootstrap=args.n_bootstrap, 
                                               seed=args.seed)
    
    # Print results
    print("\n" + "=" * 70)
    print("BOOTSTRAP RESULTS")
    print("=" * 70)
    
    print("\nParameter estimates with 95% CIs:")
    print(f"{'Parameter':<12} {'Original':>10} {'Mean':>10} {'95% CI':>25}")
    print("-" * 60)
    
    for name in ['A', 'alpha1', 'beta1', 'B', 'alpha2', 'beta2']:
        orig = results['original_params'][name]
        mean = results['parameters'][name]['mean']
        ci_low = results['parameters'][name]['ci_2.5']
        ci_high = results['parameters'][name]['ci_97.5']
        print(f"{name:<12} {orig:>10.4f} {mean:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    print("\nDerived quantities with 95% CIs:")
    print(f"{'Quantity':<20} {'Value':>10} {'95% CI':>25}")
    print("-" * 60)
    
    for name in ['peak_fast', 'peak_slow', 'mean_fast', 'mean_slow']:
        val = results['derived'][name]['mean']
        ci_low = results['derived'][name]['ci_2.5']
        ci_high = results['derived'][name]['ci_97.5']
        unit = 's'
        print(f"{name:<20} {val:>10.4f}{unit} [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'kernel_bootstrap_ci.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'kernel_bootstrap_ci.json'}")
    
    # Plot
    val_dir = Path('data/validation')
    val_dir.mkdir(parents=True, exist_ok=True)
    plot_bootstrap_results(results, bootstrap_params, val_dir / 'bootstrap_distributions.png')
    
    print("\nPhase 1 complete!")


if __name__ == '__main__':
    main()


