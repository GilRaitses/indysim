#!/usr/bin/env python3
"""
Inter-Event Interval (IEI) Validation with QQ-Plots

Compares empirical and simulated IEI distributions using:
- K-S test
- Moment comparison (mean, variance)
- QQ-plots

Usage:
    python scripts/compute_iei_validation.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, expon, pearsonr
import matplotlib.pyplot as plt


def compute_iei(event_times: np.ndarray) -> np.ndarray:
    """Compute inter-event intervals from sorted event times."""
    event_times = np.sort(event_times)
    return np.diff(event_times)


def plot_iei_validation(emp_iei: np.ndarray, sim_iei: np.ndarray, output_path: Path):
    """Create validation plots for IEI comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram comparison
    ax = axes[0, 0]
    bins = np.linspace(0, np.percentile(np.concatenate([emp_iei, sim_iei]), 95), 50)
    ax.hist(emp_iei, bins=bins, density=True, alpha=0.5, label='Empirical', edgecolor='blue')
    ax.hist(sim_iei, bins=bins, density=True, alpha=0.5, label='Simulated', edgecolor='red')
    ax.set_xlabel('Inter-Event Interval (s)')
    ax.set_ylabel('Density')
    ax.set_title('IEI Distribution Comparison')
    ax.legend()
    
    # 2. QQ-plot: Empirical vs Simulated
    ax = axes[0, 1]
    emp_sorted = np.sort(emp_iei)
    sim_sorted = np.sort(sim_iei)
    
    # Interpolate to same length
    n_points = min(len(emp_sorted), len(sim_sorted), 1000)
    emp_quantiles = np.percentile(emp_sorted, np.linspace(0, 100, n_points))
    sim_quantiles = np.percentile(sim_sorted, np.linspace(0, 100, n_points))
    
    ax.scatter(emp_quantiles, sim_quantiles, alpha=0.5, s=10)
    max_val = max(emp_quantiles.max(), sim_quantiles.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y = x')
    ax.set_xlabel('Empirical IEI Quantiles')
    ax.set_ylabel('Simulated IEI Quantiles')
    ax.set_title('QQ-Plot: Empirical vs Simulated')
    ax.legend()
    
    # 3. QQ-plot: Empirical vs Exponential
    ax = axes[1, 0]
    emp_mean = emp_iei.mean()
    theoretical = expon.ppf(np.linspace(0.01, 0.99, len(emp_sorted)), scale=emp_mean)
    ax.scatter(theoretical, emp_sorted, alpha=0.3, s=5)
    ax.plot([0, theoretical.max()], [0, theoretical.max()], 'r--', label='y = x')
    ax.set_xlabel('Theoretical (Exponential)')
    ax.set_ylabel('Empirical IEI')
    ax.set_title(f'QQ-Plot: Empirical vs Exponential(mean={emp_mean:.1f}s)')
    ax.legend()
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # K-S test between emp and sim (2-sample)
    ks_stat, ks_pval = ks_2samp(emp_iei, sim_iei)
    
    # Correlation between quantiles
    corr, _ = pearsonr(emp_quantiles, sim_quantiles)
    
    summary_text = f"""
    IEI Validation Summary
    ========================
    
    Empirical IEI:
      Mean: {emp_iei.mean():.2f} s
      Std:  {emp_iei.std():.2f} s
      CV:   {emp_iei.std()/emp_iei.mean():.2f}
      N:    {len(emp_iei)}
    
    Simulated IEI:
      Mean: {sim_iei.mean():.2f} s
      Std:  {sim_iei.std():.2f} s
      CV:   {sim_iei.std()/sim_iei.mean():.2f}
      N:    {len(sim_iei)}
    
    Comparison:
      Mean ratio:  {sim_iei.mean()/emp_iei.mean():.2f}
      Std ratio:   {sim_iei.std()/emp_iei.std():.2f}
      QQ corr:     {corr:.3f}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved IEI validation plot to {output_path}")


def main():
    print("=" * 70)
    print("IEI VALIDATION WITH QQ-PLOTS")
    print("=" * 70)
    
    # Load empirical events
    data_dir = Path('data/engineered')
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    emp_events = []
    for f in csv_files:
        df = pd.read_csv(f)
        if 'is_reorientation_start' in df.columns:
            events = df.loc[df['is_reorientation_start'] == 1, 'time'].values
        elif 'is_reorientation' in df.columns:
            events = df.loc[df['is_reorientation'] == 1, 'time'].values
        else:
            continue
        emp_events.extend(events)
    
    emp_events = np.array(emp_events)
    emp_iei = compute_iei(emp_events)
    print(f"\nEmpirical events: {len(emp_events)}")
    print(f"Empirical IEIs: {len(emp_iei)}")
    
    # Load simulated events
    sim_path = Path('data/simulated/extended_biphasic_events.csv')
    if not sim_path.exists():
        print(f"Simulated events not found: {sim_path}")
        return
    
    sim_df = pd.read_csv(sim_path)
    sim_events = sim_df['time'].values
    sim_iei = compute_iei(sim_events)
    print(f"\nSimulated events: {len(sim_events)}")
    print(f"Simulated IEIs: {len(sim_iei)}")
    
    # Summary statistics
    print(f"\n=== IEI Statistics ===")
    print(f"Empirical: mean={emp_iei.mean():.2f}s, std={emp_iei.std():.2f}s, CV={emp_iei.std()/emp_iei.mean():.2f}")
    print(f"Simulated: mean={sim_iei.mean():.2f}s, std={sim_iei.std():.2f}s, CV={sim_iei.std()/sim_iei.mean():.2f}")
    print(f"Ratio: mean={sim_iei.mean()/emp_iei.mean():.2f}, std={sim_iei.std()/emp_iei.std():.2f}")
    
    # Create plots
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_iei_validation(emp_iei, sim_iei, output_dir / 'iei_validation.png')
    
    # Save results
    results = {
        'empirical_mean': float(emp_iei.mean()),
        'empirical_std': float(emp_iei.std()),
        'empirical_cv': float(emp_iei.std() / emp_iei.mean()),
        'simulated_mean': float(sim_iei.mean()),
        'simulated_std': float(sim_iei.std()),
        'simulated_cv': float(sim_iei.std() / sim_iei.mean()),
        'mean_ratio': float(sim_iei.mean() / emp_iei.mean()),
        'std_ratio': float(sim_iei.std() / emp_iei.std()),
        'n_empirical': len(emp_iei),
        'n_simulated': len(sim_iei)
    }
    
    results_path = output_dir / 'iei_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()




