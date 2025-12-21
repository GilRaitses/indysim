#!/usr/bin/env python3
"""
Compute Deviance Residuals for NB-GLM Validation

Deviance residuals help identify systematic bias in the model fit.
For a good fit, residuals should be randomly scattered around zero.

Usage:
    python scripts/compute_deviance_residuals.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Parameters
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


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


def compute_nb_deviance_residuals(y: np.ndarray, mu: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Compute deviance residuals for Negative Binomial model.
    
    d_i = sign(y_i - mu_i) * sqrt(2 * D_i)
    
    where D_i is the deviance contribution from observation i.
    """
    r = 1 / alpha  # NB dispersion parameter
    
    # Avoid log(0)
    y = np.maximum(y, 1e-10)
    mu = np.maximum(mu, 1e-10)
    
    # Deviance contribution
    # D_i = 2 * [y * log(y/mu) - (y + r) * log((y + r)/(mu + r))]
    d_contrib = 2 * (
        y * np.log(y / mu) - 
        (y + r) * np.log((y + r) / (mu + r))
    )
    
    # Handle y = 0 case
    zero_mask = y < 1e-9
    d_contrib[zero_mask] = 2 * r * np.log(r / (mu[zero_mask] + r))
    
    # Residual with sign
    sign = np.sign(y - mu)
    deviance_residuals = sign * np.sqrt(np.maximum(d_contrib, 0))
    
    return deviance_residuals


def compute_predictions(data: pd.DataFrame, model_results: dict, kernel_config: dict) -> np.ndarray:
    """Compute predicted values for each frame."""
    coefficients = model_results['coefficients']
    intercept = coefficients.get('intercept', -7.0)
    
    # Kernel coefficients
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
    
    n = len(data)
    mu = np.zeros(n)
    
    times = data['time'].values
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            eta = intercept
        else:
            time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
            
            if time_in_cycle < LED_ON_DURATION:
                tso = np.array([time_in_cycle])
                
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
            else:
                time_since_offset = time_in_cycle - LED_ON_DURATION
                rebound = np.exp(-time_since_offset / rebound_tau)
                eta = intercept + rebound_coef * rebound
        
        mu[i] = np.exp(eta)
    
    return mu


def plot_deviance_residuals(residuals: np.ndarray, mu: np.ndarray, led1Val_ton: np.ndarray, output_dir: Path):
    """Create diagnostic plots for deviance residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs fitted values
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(residuals), min(10000, len(residuals)), replace=False)
    ax.scatter(mu[sample_idx], residuals[sample_idx], alpha=0.1, s=1)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Fitted values (mu)')
    ax.set_ylabel('Deviance residuals')
    ax.set_title('Residuals vs Fitted')
    ax.set_xlim(0, np.percentile(mu, 99.9))
    
    # 2. Residuals vs time since stimulus
    ax = axes[0, 1]
    valid = led1Val_ton < 15
    sample_idx = np.random.choice(np.where(valid)[0], min(10000, valid.sum()), replace=False)
    ax.scatter(led1Val_ton[sample_idx], residuals[sample_idx], alpha=0.1, s=1)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Deviance residuals')
    ax.set_title('Residuals vs Time Since Stimulus')
    
    # 3. Mean residual by time bin
    ax = axes[1, 0]
    bins = np.arange(-5, 15, 0.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(led1Val_ton, bins) - 1
    
    mean_residuals = []
    for i in range(len(bins) - 1):
        mask = bin_idx == i
        if mask.sum() > 0:
            mean_residuals.append(residuals[mask].mean())
        else:
            mean_residuals.append(np.nan)
    
    ax.plot(bin_centers, mean_residuals, 'b-o')
    ax.axhline(0, color='r', linestyle='--')
    ax.axvline(0, color='g', linestyle=':', label='LED onset')
    ax.axvline(10, color='orange', linestyle=':', label='LED offset')
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Mean deviance residual')
    ax.set_title('Mean Residual by Time Bin')
    ax.legend()
    
    # 4. Histogram of residuals
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(-5, 5, 100)
    ax.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r-', label='N(0,1)')
    ax.set_xlabel('Deviance residual')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals')
    ax.legend()
    ax.set_xlim(-5, 5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deviance_residuals.png', dpi=150)
    plt.close()
    print(f"Saved deviance residuals plot to {output_dir / 'deviance_residuals.png'}")


def main():
    print("=" * 70)
    print("DEVIANCE RESIDUALS ANALYSIS")
    print("=" * 70)
    
    # Load model
    model_path = Path('data/model/extended_biphasic_model_results.json')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    with open(model_path) as f:
        model_results = json.load(f)
    
    kernel_config = model_results.get('kernel_config', {})
    
    # Load data
    data_dir = Path('data/engineered')
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        print("No data files found")
        return
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(data):,} observations")
    
    # Get response
    if 'is_reorientation_start' in data.columns:
        y = data['is_reorientation_start'].astype(float).values
    elif 'is_reorientation' in data.columns:
        y = data['is_reorientation'].astype(float).values
    else:
        print("No reorientation column found")
        return
    
    print(f"Events: {y.sum():.0f}")
    
    # Compute predictions
    print("\nComputing predictions...")
    mu = compute_predictions(data, model_results, kernel_config)
    
    # Compute deviance residuals
    print("Computing deviance residuals...")
    alpha = 0.1  # NB dispersion
    residuals = compute_nb_deviance_residuals(y, mu, alpha)
    
    # Summary statistics
    print(f"\nResidual Summary:")
    print(f"  Mean: {residuals.mean():.4f}")
    print(f"  Std: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.4f}")
    print(f"  Max: {residuals.max():.4f}")
    
    # Compute time since stimulus
    times = data['time'].values
    led1Val_ton = np.zeros(len(times))
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            led1Val_ton[i] = t - FIRST_LED_ONSET
        else:
            time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
            led1Val_ton[i] = time_in_cycle
    
    # Plot
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_deviance_residuals(residuals, mu, led1Val_ton, output_dir)
    
    # Save results
    results = {
        'mean_residual': float(residuals.mean()),
        'std_residual': float(residuals.std()),
        'min_residual': float(residuals.min()),
        'max_residual': float(residuals.max()),
        'n_obs': len(residuals)
    }
    
    results_path = output_dir / 'deviance_residuals_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()




