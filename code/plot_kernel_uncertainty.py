#!/usr/bin/env python3
"""
Plot temporal kernel with 95% confidence bands.

Extracts kernel coefficients and standard errors from fitted model,
computes pointwise CIs via delta method, and creates publication-ready plot.

Usage:
    python scripts/plot_kernel_uncertainty.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple


def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Compute raised-cosine basis functions.
    
    B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / w))  if |t - c_j| < w
           = 0                                      otherwise
    """
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def extract_kernel_from_model(model_results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract kernel coefficients and standard errors from model results.
    
    Returns
    -------
    phi : ndarray
        Kernel coefficients
    phi_se : ndarray
        Standard errors
    """
    coefs = model_results['coefficients']
    ses = model_results['std_errors']
    
    # Find kernel coefficient names
    kernel_names = sorted([k for k in coefs.keys() if k.startswith('kernel')])
    
    phi = np.array([coefs[k] for k in kernel_names])
    phi_se = np.array([ses[k] for k in kernel_names])
    
    return phi, phi_se


def compute_kernel_with_ci(
    phi: np.ndarray,
    phi_se: np.ndarray,
    window: Tuple[float, float] = (0.0, 4.0),
    width: float = 0.6,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute kernel shape and confidence intervals.
    
    Parameters
    ----------
    phi : ndarray
        Kernel coefficients
    phi_se : ndarray
        Standard errors
    window : tuple
        Time window
    width : float
        Basis function width
    alpha : float
        Significance level (0.05 = 95% CI)
    
    Returns
    -------
    t_grid : ndarray
        Time points
    K : ndarray
        Log-rate modulation K(t) = sum_j(phi_j * B_j(t))
    RR : ndarray
        Rate ratio RR(t) = exp(K(t))
    RR_lower : ndarray
        Lower 95% CI
    RR_upper : ndarray
        Upper 95% CI
    """
    from scipy.stats import norm
    
    n_bases = len(phi)
    t_grid = np.linspace(window[0], window[1], 301)
    centers = np.linspace(window[0], window[1], n_bases)
    
    B = raised_cosine_basis(t_grid, centers, width)
    K = B @ phi
    
    # Variance of K(t) via delta method (assuming independent errors)
    # Var(K) = B @ diag(phi_se^2) @ B.T
    # For pointwise: var_K = sum_j(B_j^2 * se_j^2)
    var_K = np.sum(B**2 * phi_se**2, axis=1)
    se_K = np.sqrt(np.maximum(var_K, 0))
    
    z = norm.ppf(1 - alpha / 2)
    K_lower = K - z * se_K
    K_upper = K + z * se_K
    
    RR = np.exp(K)
    RR_lower = np.exp(K_lower)
    RR_upper = np.exp(K_upper)
    
    return t_grid, K, RR, RR_lower, RR_upper


def plot_kernel_with_ci(
    t_grid: np.ndarray,
    RR: np.ndarray,
    RR_lower: np.ndarray,
    RR_upper: np.ndarray,
    output_path: Path
):
    """Create publication-ready kernel plot with confidence bands."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Rate ratio with CI
    ax.fill_between(t_grid, RR_lower, RR_upper, alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(t_grid, RR, 'b-', linewidth=2, label='Rate ratio')
    
    # Reference line
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline (RR=1)')
    
    # Peak annotation
    peak_idx = np.argmax(RR)
    ax.scatter([t_grid[peak_idx]], [RR[peak_idx]], color='red', s=100, zorder=5)
    ax.annotate(
        f'Peak: {RR[peak_idx]:.2f}x at {t_grid[peak_idx]:.2f}s',
        xy=(t_grid[peak_idx], RR[peak_idx]),
        xytext=(t_grid[peak_idx] + 0.5, RR[peak_idx] + 1),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7)
    )
    
    ax.set_xlabel('Time since LED onset (s)', fontsize=12)
    ax.set_ylabel('Rate ratio (relative to baseline)', fontsize=12)
    ax.set_title('LNP Temporal Kernel with 95% CI', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y limits
    y_max = min(RR_upper.max() * 1.2, 20)
    y_min = max(RR_lower.min() * 0.8, 0)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved kernel plot to {output_path}")


def main():
    print("=" * 60)
    print("KERNEL UNCERTAINTY ANALYSIS")
    print("=" * 60)
    
    model_path = Path('data/model/model_results.json')
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    
    with open(model_path, 'r') as f:
        model_results = json.load(f)
    
    # Extract kernel coefficients and SEs
    phi, phi_se = extract_kernel_from_model(model_results)
    
    print(f"\nKernel coefficients and SEs:")
    for i, (c, se) in enumerate(zip(phi, phi_se)):
        z = c / se if se > 0 else 0
        sig = "*" if abs(z) > 1.96 else ""
        print(f"  kernel_{i+1}: {c:.4f} +/- {se:.4f} (z={z:.2f}) {sig}")
    
    # Compute kernel shape with CI
    t_grid, K, RR, RR_lower, RR_upper = compute_kernel_with_ci(phi, phi_se)
    
    # Find peak
    peak_idx = np.argmax(RR)
    print(f"\nPeak response:")
    print(f"  Time: {t_grid[peak_idx]:.2f} s")
    print(f"  Rate ratio: {RR[peak_idx]:.2f}x ({RR_lower[peak_idx]:.2f} - {RR_upper[peak_idx]:.2f})")
    
    # Save plot
    output_path = Path('data/model/kernel_with_ci.png')
    plot_kernel_with_ci(t_grid, RR, RR_lower, RR_upper, output_path)
    
    # Save kernel data
    kernel_data = {
        'phi': phi.tolist(),
        'phi_se': phi_se.tolist(),
        'peak_time': float(t_grid[peak_idx]),
        'peak_rr': float(RR[peak_idx]),
        'peak_rr_ci': [float(RR_lower[peak_idx]), float(RR_upper[peak_idx])]
    }
    
    kernel_path = Path('data/model/kernel_uncertainty.json')
    with open(kernel_path, 'w') as f:
        json.dump(kernel_data, f, indent=2)
    
    print(f"Saved kernel data to {kernel_path}")


if __name__ == '__main__':
    main()




