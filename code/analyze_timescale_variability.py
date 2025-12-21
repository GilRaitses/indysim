#!/usr/bin/env python3
"""
Analyze and visualize timescale variability across conditions.

This is a downstream visualization script that depends on outputs from
fit_gamma_per_condition.py. It does NOT perform any fitting; it only
reads pre-computed results and generates figures.

Dependencies:
    - fit_gamma_per_condition.py must be run first to generate:
      data/model/per_condition_timescales.json

Outputs:
    - figures/timescale_variability.png (4-panel figure)
    - data/model/timescale_variability_summary.json (summary statistics)

Creates a 4-panel figure showing:
    - Panel A: Forest plot of τ₁ (fast timescale) by condition
    - Panel B: Forest plot of τ₂ (slow timescale) by condition  
    - Panel C: τ₁ vs τ₂ scatter with condition colors
    - Panel D: Kernel overlay (all 4 conditions)

Usage:
    python scripts/analyze_timescale_variability.py

Author: INDYsim pipeline
Date: 2025-12-13
"""

from typing import List, Dict, Tuple

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
from pathlib import Path
from scipy.special import gamma as gamma_func

# Condition colors (consistent with other figures)
COLORS = {
    '0→250 | Constant': '#e41a1c',   # Red
    '0→250 | Cycling': '#377eb8',    # Blue
    '50→250 | Constant': '#4daf4a',  # Green
    '50→250 | Cycling': '#ff7f00'    # Orange
}

# Short labels for plots
SHORT_LABELS = {
    '0→250 | Constant': '0→250\nConst',
    '0→250 | Cycling': '0→250\nCycling',
    '50→250 | Constant': '50→250\nConst',
    '50→250 | Cycling': '50→250\nCycling'
}


def gamma_pdf(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute gamma probability density function.
    
    Args:
        t: Time array (seconds)
        alpha: Shape parameter (dimensionless)
        beta: Scale parameter (seconds)
        
    Returns:
        Gamma PDF values at each time point
    """
    t = np.maximum(t, 1e-10)  # Avoid division by zero
    return (t ** (alpha - 1) * np.exp(-t / beta)) / (beta ** alpha * gamma_func(alpha))


def gamma_difference_kernel(t: np.ndarray, A: float, alpha1: float, beta1: float, 
                            B: float, alpha2: float, beta2: float) -> np.ndarray:
    """Compute gamma-difference kernel K(t) = A·Gamma(t;α₁,β₁) - B·Gamma(t;α₂,β₂).
    
    Args:
        t: Time array (seconds)
        A: Amplitude of fast component (log-hazard units)
        alpha1: Shape of fast component
        beta1: Scale of fast component (seconds)
        B: Amplitude of slow component (log-hazard units)
        alpha2: Shape of slow component  
        beta2: Scale of slow component (seconds)
        
    Returns:
        Kernel values at each time point
    """
    return A * gamma_pdf(t, alpha1, beta1) - B * gamma_pdf(t, alpha2, beta2)


def main():
    # Load per-condition timescales
    data_path = Path('data/model/per_condition_timescales.json')
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return 1
    
    with open(data_path) as f:
        data = json.load(f)
    
    print("=" * 60)
    print("Timescale Variability Analysis")
    print("=" * 60)
    
    # Extract data for plotting
    conditions = list(data.keys())
    tau1_vals = []
    tau1_ci_low = []
    tau1_ci_high = []
    tau2_vals = []
    tau2_ci_low = []
    tau2_ci_high = []
    r2_vals = []
    
    for cond in conditions:
        d = data[cond]
        tau1_vals.append(d['tau1'])
        tau2_vals.append(d['tau2'])
        r2_vals.append(d['r_squared'])
        
        # Bootstrap CIs
        if 'bootstrap' in d and 'tau1' in d['bootstrap']:
            tau1_ci_low.append(d['bootstrap']['tau1']['ci_lower'])
            tau1_ci_high.append(d['bootstrap']['tau1']['ci_upper'])
            tau2_ci_low.append(d['bootstrap']['tau2']['ci_lower'])
            tau2_ci_high.append(d['bootstrap']['tau2']['ci_upper'])
        else:
            # No CIs available
            tau1_ci_low.append(d['tau1'])
            tau1_ci_high.append(d['tau1'])
            tau2_ci_low.append(d['tau2'])
            tau2_ci_high.append(d['tau2'])
    
    # Print summary table
    print("\nPer-Condition Timescales:")
    print("-" * 70)
    print(f"{'Condition':<25} {'τ₁ (s)':<12} {'τ₂ (s)':<12} {'R²':<8}")
    print("-" * 70)
    for i, cond in enumerate(conditions):
        print(f"{cond:<25} {tau1_vals[i]:.3f}        {tau2_vals[i]:.3f}        {r2_vals[i]:.3f}")
    print("-" * 70)
    
    # Key finding
    max_tau1_idx = np.argmax(tau1_vals)
    min_tau1_idx = np.argmin(tau1_vals)
    ratio = tau1_vals[max_tau1_idx] / tau1_vals[min_tau1_idx]
    print(f"\nKey finding: τ₁ ranges from {tau1_vals[min_tau1_idx]:.2f}s to {tau1_vals[max_tau1_idx]:.2f}s ({ratio:.1f}x difference)")
    print(f"  Slowest: {conditions[max_tau1_idx]}")
    print(f"  Fastest: {conditions[min_tau1_idx]}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # =================================================================
    # Panel A: Forest plot of τ₁
    # =================================================================
    ax = axes[0, 0]
    y_pos = np.arange(len(conditions))
    colors = [COLORS.get(c, 'gray') for c in conditions]
    
    # Error bars
    xerr_low = [tau1_vals[i] - tau1_ci_low[i] for i in range(len(conditions))]
    xerr_high = [tau1_ci_high[i] - tau1_vals[i] for i in range(len(conditions))]
    
    ax.errorbar(tau1_vals, y_pos, xerr=[xerr_low, xerr_high], 
                fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)
    ax.scatter(tau1_vals, y_pos, c=colors, s=150, zorder=10, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_xlabel('τ₁ (Fast Timescale, s)', fontsize=12)
    ax.set_title('A. Fast Timescale by Condition', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(tau1_vals), color='gray', linestyle='--', alpha=0.5, label='Mean')
    ax.set_xlim(0, max(tau1_ci_high) * 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    # =================================================================
    # Panel B: Forest plot of τ₂
    # =================================================================
    ax = axes[0, 1]
    
    xerr_low = [tau2_vals[i] - tau2_ci_low[i] for i in range(len(conditions))]
    xerr_high = [tau2_ci_high[i] - tau2_vals[i] for i in range(len(conditions))]
    
    ax.errorbar(tau2_vals, y_pos, xerr=[xerr_low, xerr_high], 
                fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)
    ax.scatter(tau2_vals, y_pos, c=colors, s=150, zorder=10, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_xlabel('τ₂ (Slow Timescale, s)', fontsize=12)
    ax.set_title('B. Slow Timescale by Condition', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(tau2_vals), color='gray', linestyle='--', alpha=0.5, label='Mean')
    ax.set_xlim(min(tau2_ci_low) * 0.8, max(tau2_ci_high) * 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    # =================================================================
    # Panel C: τ₁ vs τ₂ scatter
    # =================================================================
    ax = axes[1, 0]
    
    for i, cond in enumerate(conditions):
        ax.scatter(tau1_vals[i], tau2_vals[i], c=colors[i], s=200, 
                   edgecolor='black', linewidth=1.5, label=cond, zorder=10)
        
        # Error bars (cross)
        ax.errorbar(tau1_vals[i], tau2_vals[i], 
                    xerr=[[tau1_vals[i] - tau1_ci_low[i]], [tau1_ci_high[i] - tau1_vals[i]]],
                    yerr=[[tau2_vals[i] - tau2_ci_low[i]], [tau2_ci_high[i] - tau2_vals[i]]],
                    fmt='none', ecolor=colors[i], alpha=0.5, capsize=3)
    
    ax.set_xlabel('τ₁ (Fast Timescale, s)', fontsize=12)
    ax.set_ylabel('τ₂ (Slow Timescale, s)', fontsize=12)
    ax.set_title('C. Timescale Relationship', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # =================================================================
    # Panel D: Kernel overlay
    # =================================================================
    ax = axes[1, 1]
    t = np.linspace(0.01, 10, 500)
    
    for i, cond in enumerate(conditions):
        d = data[cond]
        kernel = gamma_difference_kernel(
            t, d['A'], d['alpha1'], d['beta1'], 
            d['B'], d['alpha2'], d['beta2']
        )
        ax.plot(t, kernel, color=colors[i], linewidth=2, label=SHORT_LABELS.get(cond, cond).replace('\n', ' '))
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Kernel Value', fontsize=12)
    ax.set_title('D. Kernel Shapes by Condition', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 10)
    ax.grid(alpha=0.3)
    
    # =================================================================
    # Finalize
    # =================================================================
    plt.tight_layout()
    
    # Save figure
    output_path = Path('figures/timescale_variability.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Also save summary JSON
    summary = {
        'conditions': conditions,
        'tau1': {
            'values': tau1_vals,
            'ci_lower': tau1_ci_low,
            'ci_upper': tau1_ci_high,
            'range_ratio': float(ratio),
            'slowest_condition': conditions[max_tau1_idx],
            'fastest_condition': conditions[min_tau1_idx]
        },
        'tau2': {
            'values': tau2_vals,
            'ci_lower': tau2_ci_low,
            'ci_upper': tau2_ci_high,
            'range': [min(tau2_vals), max(tau2_vals)]
        },
        'r_squared': r2_vals,
        'interpretation': (
            f"The fast timescale τ₁ shows striking condition-dependence, "
            f"ranging from {min(tau1_vals):.2f}s ({conditions[min_tau1_idx]}) "
            f"to {max(tau1_vals):.2f}s ({conditions[max_tau1_idx]}). "
            f"This {ratio:.1f}-fold difference suggests that baseline neural excitation "
            f"modulates the speed of sensory transduction. In contrast, the slow "
            f"timescale τ₂ remains relatively stable ({min(tau2_vals):.2f}--{max(tau2_vals):.2f}s) "
            f"across conditions, indicating that synaptic adaptation operates on an "
            f"intrinsic circuit timescale independent of stimulus parameters."
        )
    }
    
    summary_path = Path('data/model/timescale_variability_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    plt.show()
    return 0


if __name__ == '__main__':
    exit(main())
