#!/usr/bin/env python3
"""
Time-Rescaling Test for Point Process Validation

Standard validation for point-process / LNP models.
Transforms inter-event times via the integrated intensity,
checks if the rescaled times are i.i.d. uniform(0,1).

Reference:
  Brown et al. (2002) "The time-rescaling theorem and its application
  to neural spike train data analysis"
  
Usage:
    python scripts/compute_time_rescaling.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple
from scipy.stats import kstest, uniform
import matplotlib.pyplot as plt

# Parameters matching empirical data
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


def make_hazard_function(model_results: dict, kernel_config: dict) -> Callable:
    """Create hazard function from model results."""
    coefficients = model_results['coefficients']
    intercept = coefficients.get('intercept', -7.0)
    
    # Get kernel coefficients
    early_coefs = [coefficients.get(f'kernel_early_{i+1}', 0) for i in range(3)]
    intm_coefs = [coefficients.get(f'kernel_intm_{i+1}', 0) for i in range(2)]
    late_coefs = [coefficients.get(f'kernel_late_{i+1}', 0) for i in range(4)]
    
    early_centers = np.array(kernel_config.get('early_centers', [0.2, 0.7, 1.4]))
    intm_centers = np.array(kernel_config.get('intm_centers', [2.0, 2.5]))
    late_centers = np.array(kernel_config.get('late_centers', [3.0, 5.0, 7.0, 9.0]))
    early_width = kernel_config.get('early_width', 0.4)
    intm_width = kernel_config.get('intm_width', 0.6)
    late_width = kernel_config.get('late_width', 1.8)
    
    rebound_coef = coefficients.get('led_off_rebound', 0.0)
    rebound_tau = kernel_config.get('rebound_tau', 2.0)
    
    def hazard_func(t: float) -> float:
        if t < FIRST_LED_ONSET:
            return np.exp(intercept)
        
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        
        if time_in_cycle < LED_ON_DURATION:
            time_since_onset = time_in_cycle
            tso = np.array([time_since_onset])
            
            early_basis = raised_cosine_basis(tso, early_centers, early_width)
            intm_basis = raised_cosine_basis(tso, intm_centers, intm_width)
            late_basis = raised_cosine_basis(tso, late_centers, late_width)
            
            kernel_contrib = 0.0
            for j, c in enumerate(early_coefs):
                kernel_contrib += c * early_basis[0, j]
            for j, c in enumerate(intm_coefs):
                kernel_contrib += c * intm_basis[0, j]
            for j, c in enumerate(late_coefs):
                kernel_contrib += c * late_basis[0, j]
            
            eta = intercept + kernel_contrib
            return np.exp(eta)
        else:
            time_since_offset = time_in_cycle - LED_ON_DURATION
            rebound = np.exp(-time_since_offset / rebound_tau)
            eta = intercept + rebound_coef * rebound
            return np.exp(eta)
    
    return hazard_func


def time_rescaling_test(
    event_times: np.ndarray,
    hazard_func: Callable,
    dt: float = 0.05
) -> Tuple[np.ndarray, float, float]:
    """
    Perform time-rescaling test on point process.
    
    For a correct model, the rescaled inter-event times:
        z_k = 1 - exp(-integral of lambda from t_{k-1} to t_k)
    should be i.i.d. uniform(0, 1).
    
    Parameters
    ----------
    event_times : ndarray
        Sorted event times
    hazard_func : callable
        Function returning hazard rate at time t
    dt : float
        Integration step size
    
    Returns
    -------
    z_values : ndarray
        Rescaled times (should be uniform if model is correct)
    ks_stat : float
        Kolmogorov-Smirnov statistic
    ks_pval : float
        K-S test p-value (p > 0.05 means PASS)
    """
    event_times = np.sort(event_times)
    n_events = len(event_times)
    
    if n_events < 2:
        return np.array([]), 0.0, 1.0
    
    z_values = []
    
    for k in range(1, n_events):
        t_prev = event_times[k - 1]
        t_curr = event_times[k]
        
        # Integrate hazard from t_prev to t_curr
        cumulative_hazard = 0.0
        t = t_prev
        while t < t_curr:
            h = hazard_func(t)
            cumulative_hazard += h * dt
            t += dt
        
        # Rescale: z_k = 1 - exp(-cumulative_hazard)
        z_k = 1 - np.exp(-cumulative_hazard)
        z_values.append(z_k)
    
    z_values = np.array(z_values)
    
    # K-S test against uniform(0, 1)
    ks_stat, ks_pval = kstest(z_values, 'uniform')
    
    return z_values, ks_stat, ks_pval


def plot_time_rescaling(z_values: np.ndarray, output_path: Path):
    """Create QQ-plot of rescaled times against uniform distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # QQ-plot
    z_sorted = np.sort(z_values)
    n = len(z_sorted)
    theoretical = (np.arange(1, n + 1) - 0.5) / n
    
    axes[0].plot(theoretical, z_sorted, 'b.', alpha=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', label='y = x')
    axes[0].set_xlabel('Theoretical (Uniform)')
    axes[0].set_ylabel('Empirical (Rescaled times)')
    axes[0].set_title('QQ-Plot: Time-Rescaling Test')
    axes[0].legend()
    axes[0].set_aspect('equal')
    
    # Histogram
    axes[1].hist(z_values, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1].axhline(1, color='r', linestyle='--', label='Uniform(0,1)')
    axes[1].set_xlabel('Rescaled Time')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Histogram of Rescaled Times')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved time-rescaling plot to {output_path}")


def main():
    print("=" * 70)
    print("TIME-RESCALING TEST")
    print("=" * 70)
    
    # Load model
    model_path = Path('data/model/extended_biphasic_model_results.json')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    with open(model_path) as f:
        model_results = json.load(f)
    
    kernel_config = model_results.get('kernel_config', {})
    hazard_func = make_hazard_function(model_results, kernel_config)
    
    # Load empirical events
    data_dir = Path('data/engineered')
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    all_events = []
    for f in csv_files:
        df = pd.read_csv(f)
        if 'is_reorientation_start' in df.columns:
            events = df.loc[df['is_reorientation_start'] == 1, 'time'].values
        elif 'is_reorientation' in df.columns:
            events = df.loc[df['is_reorientation'] == 1, 'time'].values
        else:
            continue
        all_events.extend(events)
    
    emp_events = np.array(all_events)
    print(f"\nEmpirical events: {len(emp_events)}")
    
    # Perform time-rescaling test
    print("\nPerforming time-rescaling test...")
    z_values, ks_stat, ks_pval = time_rescaling_test(emp_events, hazard_func, dt=0.05)
    
    print(f"\nResults:")
    print(f"  K-S statistic: {ks_stat:.4f}")
    print(f"  K-S p-value: {ks_pval:.4f}")
    print(f"  Status: {'PASS' if ks_pval > 0.05 else 'FAIL'} (p > 0.05)")
    
    # Save plot
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_time_rescaling(z_values, output_dir / 'time_rescaling_qq.png')
    
    # Save results
    results = {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'n_events': len(emp_events),
        'n_intervals': len(z_values),
        'pass': bool(ks_pval > 0.05)
    }
    
    results_path = output_dir / 'time_rescaling_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()




