#!/usr/bin/env python3
"""
Factorial Model Diagnostics

Generate residual plots and diagnostic statistics for the factorial NB-GLM.

Usage:
    python scripts/factorial_diagnostics.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
except ImportError:
    print("Error: statsmodels required")
    exit(1)


from typing import Dict

def compute_residuals(df: pd.DataFrame, results: Dict) -> pd.DataFrame:
    """
    Compute various residual types for the fitted model.
    """
    # Extract coefficients
    coeffs = results['coefficients']
    
    # Get indicators
    I = df['I'].values
    T = df['T'].values
    IT = df['IT'].values
    K_on = df['K_on'].values
    I_K_on = df['I_K_on'].values
    T_K_on = df['T_K_on'].values
    K_off = df['K_off'].values
    y = df['events'].values
    
    # Compute linear predictor
    eta = (coeffs['beta_0']['mean'] + 
           coeffs['beta_I']['mean'] * I +
           coeffs['beta_T']['mean'] * T +
           coeffs['beta_IT']['mean'] * IT +
           coeffs['alpha']['mean'] * K_on +
           coeffs['alpha_I']['mean'] * I_K_on +
           coeffs['alpha_T']['mean'] * T_K_on +
           coeffs['gamma']['mean'] * K_off)
    
    # Fitted values
    mu = np.exp(eta)
    
    # Pearson residuals
    pearson = (y - mu) / np.sqrt(mu + 1e-10)
    
    # Deviance residuals
    # For NB, deviance residual is more complex; use simplified version
    with np.errstate(divide='ignore', invalid='ignore'):
        deviance = np.sign(y - mu) * np.sqrt(2 * np.abs(
            y * np.log((y + 1e-10) / (mu + 1e-10)) - (y - mu)
        ))
    deviance = np.nan_to_num(deviance, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Response residuals
    response = y - mu
    
    return pd.DataFrame({
        'fitted': mu,
        'observed': y,
        'pearson': pearson,
        'deviance': deviance,
        'response': response,
        'condition': df['condition'].values
    })


def plot_diagnostics(residuals: pd.DataFrame, output_dir: Path):
    """
    Generate diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Pearson residuals vs fitted
    ax = axes[0, 0]
    # Sample for plotting (too many points)
    sample_idx = np.random.choice(len(residuals), min(5000, len(residuals)), replace=False)
    ax.scatter(residuals.iloc[sample_idx]['fitted'], 
               residuals.iloc[sample_idx]['pearson'], 
               alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Pearson Residuals vs Fitted')
    ax.set_xlim(0, 0.01)  # Most fitted values are very small
    
    # 2. Deviance residuals vs fitted
    ax = axes[0, 1]
    ax.scatter(residuals.iloc[sample_idx]['fitted'], 
               residuals.iloc[sample_idx]['deviance'], 
               alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Deviance residuals')
    ax.set_title('Deviance Residuals vs Fitted')
    ax.set_xlim(0, 0.01)
    
    # 3. QQ plot of Pearson residuals
    ax = axes[1, 0]
    pearson_sorted = np.sort(residuals['pearson'].values)
    n = len(pearson_sorted)
    theoretical = np.linspace(-3, 3, n)
    # Use only a subset for QQ
    step = max(1, n // 1000)
    ax.scatter(theoretical[::step], pearson_sorted[::step], alpha=0.5, s=10)
    ax.plot([-3, 3], [-3, 3], 'r--', label='y=x')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('Q-Q Plot (Pearson Residuals)')
    ax.legend()
    
    # 4. Residuals by condition
    ax = axes[1, 1]
    conditions = residuals['condition'].unique()
    box_data = [residuals[residuals['condition'] == c]['pearson'].values for c in conditions]
    bp = ax.boxplot(box_data, labels=[c.replace(' | ', '\n') for c in conditions])
    ax.axhline(0, color='red', linestyle='--')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Residuals by Condition')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'factorial_diagnostics.png', dpi=150)
    plt.close()
    
    print(f"Saved diagnostics plot to {output_dir / 'factorial_diagnostics.png'}")


def compute_diagnostic_stats(residuals: pd.DataFrame) -> Dict:
    """
    Compute diagnostic statistics.
    """
    pearson = residuals['pearson'].values
    
    return {
        'pearson_mean': float(np.mean(pearson)),
        'pearson_std': float(np.std(pearson)),
        'pearson_skew': float(pd.Series(pearson).skew()),
        'pearson_kurtosis': float(pd.Series(pearson).kurtosis()),
        'pearson_min': float(np.min(pearson)),
        'pearson_max': float(np.max(pearson)),
        'n_large_residuals': int(np.sum(np.abs(pearson) > 3)),
        'pct_large_residuals': float(np.mean(np.abs(pearson) > 3) * 100)
    }


def compute_time_rescaling(df: pd.DataFrame, results: Dict) -> Dict:
    """
    Compute time-rescaling test for Poisson process assumption.
    
    For each event, compute the integrated hazard since the previous event.
    Under a correct model, these should be Exp(1) distributed.
    """
    coeffs = results['coefficients']
    
    # Get indicators and compute hazard
    I = df['I'].values
    T = df['T'].values
    IT = df['IT'].values
    K_on = df['K_on'].values
    I_K_on = df['I_K_on'].values
    T_K_on = df['T_K_on'].values
    K_off = df['K_off'].values
    y = df['events'].values
    
    # Compute log-hazard
    eta = (coeffs['beta_0']['mean'] + 
           coeffs['beta_I']['mean'] * I +
           coeffs['beta_T']['mean'] * T +
           coeffs['beta_IT']['mean'] * IT +
           coeffs['alpha']['mean'] * K_on +
           coeffs['alpha_I']['mean'] * I_K_on +
           coeffs['alpha_T']['mean'] * T_K_on +
           coeffs['gamma']['mean'] * K_off)
    
    hazard = np.exp(eta)
    
    # Find event indices
    event_idx = np.where(y == 1)[0]
    
    if len(event_idx) < 100:
        return {'error': 'Too few events for time-rescaling test'}
    
    # Compute integrated hazard between events
    rescaled_times = []
    for i in range(1, len(event_idx)):
        start = event_idx[i-1]
        end = event_idx[i]
        integrated = np.sum(hazard[start:end])
        rescaled_times.append(integrated)
    
    rescaled_times = np.array(rescaled_times)
    
    # Transform to uniform via 1 - exp(-z)
    uniform_times = 1 - np.exp(-rescaled_times)
    
    # KS test against Uniform(0,1)
    from scipy import stats
    ks_stat, ks_pval = stats.kstest(uniform_times, 'uniform')
    
    # Compute deviation from expected
    sorted_u = np.sort(uniform_times)
    n = len(sorted_u)
    expected = np.linspace(0, 1, n)
    max_deviation = np.max(np.abs(sorted_u - expected))
    mean_deviation = np.mean(np.abs(sorted_u - expected))
    
    return {
        'n_intervals': len(rescaled_times),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'max_deviation': float(max_deviation),
        'mean_deviation': float(mean_deviation),
        'pct_deviation': float(mean_deviation * 100),
        'sorted_uniform': sorted_u.tolist()[:1000]  # Save subset for plotting
    }


def plot_time_rescaling(tr_results: Dict, output_dir: Path):
    """
    Plot time-rescaling cumulative distribution.
    """
    if 'error' in tr_results:
        print(f"Skipping time-rescaling plot: {tr_results['error']}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sorted_u = np.array(tr_results['sorted_uniform'])
    n = len(sorted_u)
    empirical = np.arange(1, n+1) / n
    
    ax.plot(sorted_u, empirical, 'b-', linewidth=1.5, label='Empirical CDF')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Expected (Uniform)')
    
    # Add confidence bands (approximate 95%)
    alpha = 0.05
    c = np.sqrt(-0.5 * np.log(alpha / 2))
    upper = np.minimum(empirical + c / np.sqrt(n), 1)
    lower = np.maximum(empirical - c / np.sqrt(n), 0)
    ax.fill_between(sorted_u, lower, upper, alpha=0.2, color='gray', label='95% CI')
    
    ax.set_xlabel('Rescaled Time (Uniform)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Time-Rescaling Test (KS p = {tr_results["ks_pvalue"]:.4f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add deviation annotation
    ax.text(0.05, 0.95, f'Mean deviation: {tr_results["pct_deviation"]:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_rescaling.png', dpi=150)
    plt.close()
    
    print(f"Saved time-rescaling plot to {output_dir / 'time_rescaling.png'}")


def main():
    print("=" * 70)
    print("FACTORIAL MODEL DIAGNOSTICS")
    print("=" * 70)
    
    # Load design matrix and results
    dm_path = Path('data/processed/factorial_design_matrix.parquet')
    results_path = Path('data/model/factorial_model_results.json')
    
    if not dm_path.exists() or not results_path.exists():
        print("Error: Run fit_factorial_model.py first")
        return
    
    df = pd.read_parquet(dm_path)
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"\nLoaded {len(df):,} observations")
    
    # Compute residuals
    print("\nComputing residuals...")
    residuals = compute_residuals(df, results)
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    output_dir = Path('figures/factorial_diagnostics')
    plot_diagnostics(residuals, output_dir)
    
    # Compute statistics
    stats = compute_diagnostic_stats(residuals)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC STATISTICS")
    print("=" * 70)
    print(f"\nPearson residuals:")
    print(f"  Mean: {stats['pearson_mean']:.4f} (should be ~0)")
    print(f"  Std:  {stats['pearson_std']:.4f}")
    print(f"  Skew: {stats['pearson_skew']:.4f}")
    print(f"  Kurt: {stats['pearson_kurtosis']:.4f}")
    print(f"  Range: [{stats['pearson_min']:.2f}, {stats['pearson_max']:.2f}]")
    print(f"  |r| > 3: {stats['n_large_residuals']} ({stats['pct_large_residuals']:.2f}%)")
    
    # Time-rescaling test
    print("\n" + "=" * 70)
    print("TIME-RESCALING TEST")
    print("=" * 70)
    print("\nComputing time-rescaling (sampling 10% for speed)...")
    df_sample = df.sample(frac=0.1, random_state=42)
    tr_results = compute_time_rescaling(df_sample, results)
    
    if 'error' not in tr_results:
        print(f"\nTime-rescaling results:")
        print(f"  Intervals: {tr_results['n_intervals']}")
        print(f"  KS statistic: {tr_results['ks_statistic']:.4f}")
        print(f"  KS p-value: {tr_results['ks_pvalue']:.4f}")
        print(f"  Mean deviation: {tr_results['pct_deviation']:.1f}%")
        
        plot_time_rescaling(tr_results, output_dir)
        stats['time_rescaling'] = tr_results
    else:
        print(f"  Error: {tr_results['error']}")
    
    # Save statistics
    # Remove large arrays before saving
    stats_to_save = stats.copy()
    if 'time_rescaling' in stats_to_save:
        stats_to_save['time_rescaling'] = {
            k: v for k, v in stats_to_save['time_rescaling'].items() 
            if k != 'sorted_uniform'
        }
    
    with open(output_dir / 'diagnostic_stats.json', 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    print(f"\nSaved diagnostic stats to {output_dir / 'diagnostic_stats.json'}")
    
    return stats


if __name__ == '__main__':
    main()
