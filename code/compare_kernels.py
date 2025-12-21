#!/usr/bin/env python3
"""
Compare Fixed-Effects vs Mixed-Effects Kernels

Visualizes and compares the temporal kernels from:
1. Fixed-effects NB-GLM (current model)
2. Mixed-effects NB-GLM (with random intercepts)
3. BO-optimal configuration

Usage:
    python scripts/compare_kernels.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def evaluate_kernel(kernel_config: dict, coefficients: dict, t: np.ndarray) -> np.ndarray:
    """Evaluate the kernel on a time grid."""
    K = np.zeros_like(t)
    
    # Early bases
    early_centers = np.array(kernel_config.get('early_centers', [0.2, 0.7, 1.4]))
    early_width = kernel_config.get('early_width', 0.4)
    early_basis = raised_cosine_basis(t, early_centers, early_width)
    
    for j in range(len(early_centers)):
        key = f'kernel_early_{j+1}'
        if key in coefficients:
            K += coefficients[key] * early_basis[:, j]
    
    # Intermediate bases
    intm_centers = np.array(kernel_config.get('intm_centers', [2.0, 2.5]))
    intm_width = kernel_config.get('intm_width', 0.6)
    if len(intm_centers) > 0:
        intm_basis = raised_cosine_basis(t, intm_centers, intm_width)
        for j in range(len(intm_centers)):
            key = f'kernel_intm_{j+1}'
            if key in coefficients:
                K += coefficients[key] * intm_basis[:, j]
    
    # Late bases
    late_centers = np.array(kernel_config.get('late_centers', [3, 5, 7, 9]))
    late_width = kernel_config.get('late_width', 1.8)
    late_basis = raised_cosine_basis(t, late_centers, late_width)
    
    for j in range(len(late_centers)):
        key = f'kernel_late_{j+1}'
        if key in coefficients:
            K += coefficients[key] * late_basis[:, j]
    
    return K


def main():
    print("=" * 70)
    print("KERNEL COMPARISON: Fixed-Effects vs Mixed-Effects")
    print("=" * 70)
    
    # Load models
    fixed_path = Path('data/model/extended_biphasic_model_results.json')
    mixed_path = Path('data/model/mixed_effects_results.json')
    bo_path = Path('data/model/bayesian_opt_results.json')
    
    with open(fixed_path) as f:
        fixed_model = json.load(f)
    
    with open(mixed_path) as f:
        mixed_model = json.load(f)
    
    with open(bo_path) as f:
        bo_results = json.load(f)
    
    # Time grid
    t = np.linspace(0, 10, 500)
    
    # Evaluate kernels
    fixed_kernel = evaluate_kernel(
        fixed_model['kernel_config'], 
        fixed_model['coefficients'], 
        t
    )
    
    mixed_kernel = evaluate_kernel(
        fixed_model['kernel_config'],  # Same config, different coefficients
        mixed_model['coefficients'], 
        t
    )
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COEFFICIENT COMPARISON")
    print("=" * 70)
    print(f"{'Coefficient':<20} {'Fixed-Effects':>15} {'Mixed-Effects':>15} {'Difference':>12}")
    print("-" * 62)
    
    for key in fixed_model['coefficients']:
        fixed_val = fixed_model['coefficients'][key]
        mixed_val = mixed_model['coefficients'].get(key, 0)
        diff = mixed_val - fixed_val
        print(f"{key:<20} {fixed_val:>15.3f} {mixed_val:>15.3f} {diff:>+12.3f}")
    
    # Key differences
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    intercept_diff = mixed_model['coefficients']['intercept'] - fixed_model['coefficients']['intercept']
    print(f"\n1. INTERCEPT:")
    print(f"   Fixed:  {fixed_model['coefficients']['intercept']:.3f}")
    print(f"   Mixed:  {mixed_model['coefficients']['intercept']:.3f}")
    print(f"   Diff:   {intercept_diff:+.3f}")
    print(f"   → Mixed-effects has LOWER baseline (more negative intercept)")
    
    print(f"\n2. RANDOM INTERCEPT VARIABILITY:")
    print(f"   Between-track SD: {mixed_model['random_intercept_std']:.3f}")
    print(f"   → High individual variability in baseline rates")
    
    # Early kernel comparison
    fixed_early = sum(fixed_model['coefficients'].get(f'kernel_early_{i}', 0) for i in [1,2,3])
    mixed_early = sum(mixed_model['coefficients'].get(f'kernel_early_{i}', 0) for i in [1,2,3])
    print(f"\n3. EARLY KERNEL (0-1.5s):")
    print(f"   Fixed sum:  {fixed_early:+.3f}")
    print(f"   Mixed sum:  {mixed_early:+.3f}")
    print(f"   → Mixed-effects shows STRONGER early positive response")
    
    # Late kernel comparison
    fixed_late = sum(fixed_model['coefficients'].get(f'kernel_late_{i}', 0) for i in [1,2,3,4])
    mixed_late = sum(mixed_model['coefficients'].get(f'kernel_late_{i}', 0) for i in [1,2,3,4])
    print(f"\n4. LATE KERNEL (3-9s):")
    print(f"   Fixed sum:  {fixed_late:+.3f}")
    print(f"   Mixed sum:  {mixed_late:+.3f}")
    print(f"   → Mixed-effects shows WEAKER overall suppression, with recovery")
    
    # Rebound comparison
    print(f"\n5. LED-OFF REBOUND:")
    print(f"   Fixed:  {fixed_model['coefficients']['led_off_rebound']:+.3f}")
    print(f"   Mixed:  {mixed_model['coefficients']['led_off_rebound']:+.3f}")
    print(f"   → Mixed-effects shows STRONGER rebound")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Kernel comparison
    ax = axes[0, 0]
    ax.plot(t, fixed_kernel, 'b-', linewidth=2, label='Fixed-effects')
    ax.plot(t, mixed_kernel, 'r--', linewidth=2, label='Mixed-effects')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvspan(0, 10, alpha=0.1, color='yellow', label='LED ON')
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel value (log-hazard)')
    ax.set_title('Temporal Kernel Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # Plot 2: Hazard ratio (exp of kernel)
    ax = axes[0, 1]
    ax.plot(t, np.exp(fixed_kernel), 'b-', linewidth=2, label='Fixed-effects')
    ax.plot(t, np.exp(mixed_kernel), 'r--', linewidth=2, label='Mixed-effects')
    ax.axhline(1, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Hazard ratio (vs baseline)')
    ax.set_title('Hazard Ratio (exp of kernel)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    
    # Plot 3: Coefficient bar chart
    ax = axes[1, 0]
    coef_names = ['early_1', 'early_2', 'early_3', 'intm_1', 'intm_2', 
                  'late_1', 'late_2', 'late_3', 'late_4', 'rebound']
    fixed_vals = [fixed_model['coefficients'].get(f'kernel_{n}', 
                  fixed_model['coefficients'].get(f'led_off_{n}', 0)) 
                  for n in coef_names]
    mixed_vals = [mixed_model['coefficients'].get(f'kernel_{n}',
                  mixed_model['coefficients'].get(f'led_off_{n}', 0))
                  for n in coef_names]
    
    # Manually extract
    fixed_vals = [
        fixed_model['coefficients']['kernel_early_1'],
        fixed_model['coefficients']['kernel_early_2'],
        fixed_model['coefficients']['kernel_early_3'],
        fixed_model['coefficients']['kernel_intm_1'],
        fixed_model['coefficients']['kernel_intm_2'],
        fixed_model['coefficients']['kernel_late_1'],
        fixed_model['coefficients']['kernel_late_2'],
        fixed_model['coefficients']['kernel_late_3'],
        fixed_model['coefficients']['kernel_late_4'],
        fixed_model['coefficients']['led_off_rebound']
    ]
    mixed_vals = [
        mixed_model['coefficients']['kernel_early_1'],
        mixed_model['coefficients']['kernel_early_2'],
        mixed_model['coefficients']['kernel_early_3'],
        mixed_model['coefficients']['kernel_intm_1'],
        mixed_model['coefficients']['kernel_intm_2'],
        mixed_model['coefficients']['kernel_late_1'],
        mixed_model['coefficients']['kernel_late_2'],
        mixed_model['coefficients']['kernel_late_3'],
        mixed_model['coefficients']['kernel_late_4'],
        mixed_model['coefficients']['led_off_rebound']
    ]
    
    x = np.arange(len(coef_names))
    width = 0.35
    ax.bar(x - width/2, fixed_vals, width, label='Fixed-effects', color='blue', alpha=0.7)
    ax.bar(x + width/2, mixed_vals, width, label='Mixed-effects', color='red', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(coef_names, rotation=45, ha='right')
    ax.set_ylabel('Coefficient value')
    ax.set_title('Coefficient Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Kernel difference
    ax = axes[1, 1]
    diff = mixed_kernel - fixed_kernel
    ax.fill_between(t, diff, 0, where=diff > 0, alpha=0.5, color='green', label='Mixed > Fixed')
    ax.fill_between(t, diff, 0, where=diff < 0, alpha=0.5, color='purple', label='Fixed > Mixed')
    ax.plot(t, diff, 'k-', linewidth=1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel difference (Mixed - Fixed)')
    ax.set_title('Kernel Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    plt.tight_layout()
    
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'kernel_comparison.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved comparison plot to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Fixed-Effects Model:
    - Intercept fixed at -7.0 (empirical baseline)
    - Moderate early bump (+1.1 at 0.2s)
    - Strong suppression (-2.5 at 3s)
    - Weak recovery (near 0 at 7-9s)
    - Weak rebound (+0.56)
    
    Mixed-Effects Model:
    - Lower mean intercept (-7.82)
    - STRONGER early bump (+2.5 at 0.2s)
    - Moderate suppression (-1.6 at 3s)
    - CLEAR recovery (+0.9 to +1.1 at 7-9s)
    - STRONGER rebound (+1.95)
    
    Key Difference:
    The mixed-effects model captures more dynamic range by accounting
    for individual baseline variability. This allows the kernel to show
    stronger modulation (both positive and negative) around a per-track
    baseline, rather than a population average.
    """)


if __name__ == '__main__':
    main()




