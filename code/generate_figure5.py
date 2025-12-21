#!/usr/bin/env python3
"""
Generate Figure 5: Factorial Analysis Results

Panel A: 2x2 heatmap of suppression amplitude by condition
Panel B: Forest plot of factorial coefficients with 95% CIs

Usage:
    python scripts/generate_figure5.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict


def load_results() -> Dict:
    """Load factorial model results."""
    results_path = Path('data/model/factorial_model_results.json')
    
    if not results_path.exists():
        print("Error: Run fit_factorial_model.py first")
        exit(1)
    
    with open(results_path) as f:
        return json.load(f)


def create_figure5(results: Dict, output_path: Path):
    """
    Create the main factorial results figure.
    
    Panel A: 2x2 heatmap of suppression amplitude
    Panel B: Forest plot of key coefficients
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: Heatmap of suppression amplitude
    coeffs = results['coefficients']
    alpha = coeffs['alpha']['mean']
    alpha_I = coeffs['alpha_I']['mean']
    alpha_T = coeffs['alpha_T']['mean']
    
    # Compute amplitudes for each condition
    # Rows: Control (T=0), Temp (T=1)
    # Cols: 0-250 (I=0), 50-250 (I=1)
    amplitudes = np.array([
        [alpha, alpha + alpha_I],                     # Control: T=0
        [alpha + alpha_T, alpha + alpha_I + alpha_T]  # Temp: T=1
    ])
    
    # Heatmap
    im = ax1.imshow(amplitudes, cmap='RdBu_r', aspect='auto', 
                    vmin=0, vmax=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Suppression Amplitude (α)', shrink=0.8)
    
    # Labels
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['0→250\nPWM', '50→250\nPWM'], fontsize=11)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Constant\n(7 PWM)', 'Cycling\n(5-15 PWM)'], fontsize=11)
    ax1.set_xlabel('LED1 Intensity Step', fontsize=12)
    ax1.set_ylabel('LED2 Background Pattern', fontsize=12)
    ax1.set_title('A. Suppression Amplitude by Condition', fontsize=13, fontweight='ultralight', fontfamily='Avenir')
    
    # Add value annotations
    for i in range(2):
        for j in range(2):
            val = amplitudes[i, j]
            color = 'white' if val > 0.8 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                     fontsize=14, fontweight='bold', color=color)
    
    # Panel B: Forest plot
    # Parameters to show
    params = [
        ('beta_I', 'β_I (Intensity)', 'Baseline'),
        ('beta_T', 'β_C (Cycling)', 'Baseline'),
        ('beta_IT', 'β_I×C (Interaction)', 'Baseline'),
        ('alpha_I', 'α_I (Intensity mod.)', 'Kernel'),
        ('alpha_T', 'α_C (Cycling mod.)', 'Kernel'),
        ('gamma', 'γ (Rebound)', 'Kernel'),
    ]
    
    y_pos = np.arange(len(params))
    means = []
    ci_lows = []
    ci_highs = []
    labels = []
    categories = []
    
    for key, label, cat in params:
        c = coeffs[key]
        means.append(c['mean'])
        ci_lows.append(c['ci_low'])
        ci_highs.append(c['ci_high'])
        labels.append(label)
        categories.append(cat)
    
    # Convert to arrays
    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    
    # Error bar widths
    errors = np.array([means - ci_lows, ci_highs - means])
    
    # Colors by significance
    colors = []
    for key, _, _ in params:
        if coeffs[key]['significant']:
            colors.append('#2E86AB')  # Blue for significant
        else:
            colors.append('#888888')  # Gray for non-significant
    
    # Plot horizontal error bars
    for i, (m, lo, hi, c) in enumerate(zip(means, ci_lows, ci_highs, colors)):
        ax2.errorbar(m, i, xerr=[[m - lo], [hi - m]], 
                     fmt='o', color=c, markersize=8, capsize=5, 
                     linewidth=2, capthick=2)
    
    # Reference line at 0
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Y-axis labels
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    
    # X-axis
    ax2.set_xlabel('Coefficient Estimate (95% CI)', fontsize=12)
    ax2.set_title('B. Factorial Effect Estimates', fontsize=13, fontweight='ultralight', fontfamily='Avenir')
    
    # Add significance markers
    for i, (key, _, _) in enumerate(params):
        if coeffs[key]['significant']:
            ax2.annotate('*', xy=(ci_highs[i] + 0.1, i), fontsize=16, 
                        fontweight='bold', ha='left', va='center')
    
    # Legend for significance
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#2E86AB', linestyle='None', 
               markersize=8, label='p < 0.05'),
        Line2D([0], [0], marker='o', color='#888888', linestyle='None', 
               markersize=8, label='p ≥ 0.05'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # Grid
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Figure 5 to {output_path}")


def print_summary(results: Dict):
    """Print summary of factorial results."""
    coeffs = results['coefficients']
    
    print("\n" + "=" * 70)
    print("FACTORIAL MODEL SUMMARY FOR PAPER")
    print("=" * 70)
    
    # Suppression amplitudes
    alpha = coeffs['alpha']['mean']
    alpha_I = coeffs['alpha_I']['mean']
    alpha_T = coeffs['alpha_T']['mean']
    
    print("\nSuppression amplitudes:")
    print(f"  0→250 | Control:  {alpha:.3f}")
    print(f"  0→250 | Temp:     {alpha + alpha_T:.3f}")
    print(f"  50→250 | Control: {alpha + alpha_I:.3f}")
    print(f"  50→250 | Temp:    {alpha + alpha_I + alpha_T:.3f}")
    
    # Effects
    print("\nKey effects:")
    print(f"  Intensity effect (α_I): {alpha_I:.3f} ({100*alpha_I/alpha:.0f}% of baseline)")
    print(f"  Temperature effect (α_T): {alpha_T:.3f} ({100*alpha_T/alpha:.0f}% of baseline)")
    
    # Fold range
    amps = [alpha, alpha + alpha_T, alpha + alpha_I, alpha + alpha_I + alpha_T]
    fold_range = max(amps) / min(amps)
    print(f"  Fold range: {fold_range:.2f}x")
    
    # CV summary
    if 'validation' in results:
        val = results['validation']
        rrs = [v['rate_ratio'] for v in val.values()]
        print(f"\nValidation (per-condition):")
        print(f"  Mean rate ratio: {np.mean(rrs):.3f}")
        print(f"  Range: [{min(rrs):.3f}, {max(rrs):.3f}]")


def main():
    print("=" * 70)
    print("GENERATING FIGURE 5: FACTORIAL ANALYSIS")
    print("=" * 70)
    
    # Load results
    results = load_results()
    
    # Create figure
    output_path = Path('docs/paper/figure5_factorial.png')
    create_figure5(results, output_path)
    
    # Also save to figures directory
    create_figure5(results, Path('figures/figure5_factorial.png'))
    
    # Print summary
    print_summary(results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
