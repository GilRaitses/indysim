#!/usr/bin/env python3
"""
Fit Analytic Kernel to Learned Raised-Cosine Kernel

After Bayesian optimization finds the best raised-cosine configuration,
this script fits a simpler analytic form (difference of exponentials)
for interpretability.

Functional form (double-exp):
    K(t) = A * exp(-t/tau_fast) - B * exp(-t/tau_slow)

Where:
- tau_fast ~ 0.2-0.4s (captures early bump)
- tau_slow ~ 2-4s (captures suppression)
- A, B are amplitudes

Usage:
    python scripts/fit_analytic_kernel.py          # Dense eval + double-exp fit
    python scripts/fit_analytic_kernel.py --dense  # Only dense kernel evaluation
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """Compute raised-cosine basis functions."""
    n_times = len(t)
    n_bases = len(centers) if len(centers) > 0 else 0
    if n_bases == 0:
        return np.zeros((n_times, 0))
    
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def double_exp_kernel(t: np.ndarray, A: float, B: float, 
                      tau_fast: float, tau_slow: float) -> np.ndarray:
    """
    Difference-of-exponentials kernel.
    
    K(t) = A * exp(-t/tau_fast) - B * exp(-t/tau_slow)
    
    Parameters
    ----------
    t : ndarray
        Time values (seconds)
    A : float
        Fast component amplitude (positive = early bump)
    B : float
        Slow component amplitude (positive = suppression)
    tau_fast : float
        Fast time constant (seconds)
    tau_slow : float
        Slow time constant (seconds)
    
    Returns
    -------
    K : ndarray
        Kernel values
    """
    return A * np.exp(-t / tau_fast) - B * np.exp(-t / tau_slow)


def evaluate_kernel(kernel_config: dict, coefficients: dict, 
                    t: np.ndarray) -> np.ndarray:
    """Evaluate the learned raised-cosine kernel on a time grid.
    
    Supports two coefficient formats:
    1. kernel_early_1, kernel_intm_1, kernel_late_1, ... (old format)
    2. x1, x2, x3, ... (new format from hybrid model)
    """
    early_centers = kernel_config.get('early_centers', [])
    intm_centers = kernel_config.get('intm_centers', [])
    late_centers = kernel_config.get('late_centers', [])
    
    n_early = len(early_centers)
    n_intm = len(intm_centers)
    n_late = len(late_centers)
    
    # Try to detect coefficient format
    if 'x1' in coefficients:
        # New format: x1, x2, x3, ...
        # Order: early (n_early) + intm (n_intm) + late (n_late) + possibly rebound
        all_coefs = [coefficients.get(f'x{i+1}', 0) for i in range(n_early + n_intm + n_late)]
        early_coefs = all_coefs[:n_early]
        intm_coefs = all_coefs[n_early:n_early + n_intm]
        late_coefs = all_coefs[n_early + n_intm:n_early + n_intm + n_late]
    else:
        # Old format: kernel_early_1, kernel_intm_1, kernel_late_1, ...
        early_coefs = [coefficients.get(f'kernel_early_{i+1}', 0) for i in range(n_early)]
        intm_coefs = [coefficients.get(f'kernel_intm_{i+1}', 0) for i in range(n_intm)]
        late_coefs = [coefficients.get(f'kernel_late_{i+1}', 0) for i in range(n_late)]
    
    # Evaluate bases
    early_basis = raised_cosine_basis(t, np.array(early_centers), 
                                      kernel_config.get('early_width', 0.3))
    intm_basis = raised_cosine_basis(t, np.array(intm_centers), 
                                     kernel_config.get('intm_width', 0.6))
    late_basis = raised_cosine_basis(t, np.array(late_centers), 
                                     kernel_config.get('late_width', 2.0))
    
    # Sum contributions
    K = np.zeros_like(t)
    for j, c in enumerate(early_coefs):
        if early_basis.shape[1] > j:
            K += c * early_basis[:, j]
    for j, c in enumerate(intm_coefs):
        if intm_basis.shape[1] > j:
            K += c * intm_basis[:, j]
    for j, c in enumerate(late_coefs):
        if late_basis.shape[1] > j:
            K += c * late_basis[:, j]
    
    return K


def fit_double_exponential(t: np.ndarray, K: np.ndarray) -> dict:
    """
    Fit difference-of-exponentials to the learned kernel.
    
    Returns fitted parameters and goodness-of-fit metrics.
    """
    # Initial parameter guesses based on typical kernel shape
    # A ~ amplitude of early bump
    # B ~ amplitude of suppression
    # tau_fast ~ 0.3s (early bump decay)
    # tau_slow ~ 3s (suppression decay)
    
    p0 = [2.0, 1.5, 0.3, 3.0]  # [A, B, tau_fast, tau_slow]
    bounds = ([0, 0, 0.1, 1.0], [10, 10, 1.0, 10.0])
    
    try:
        popt, pcov = curve_fit(double_exp_kernel, t, K, p0=p0, bounds=bounds, maxfev=5000)
        
        A, B, tau_fast, tau_slow = popt
        
        # Compute fitted kernel
        K_fit = double_exp_kernel(t, *popt)
        
        # Goodness of fit
        residuals = K - K_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((K - np.mean(K))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'A': float(A),
            'B': float(B),
            'tau_fast': float(tau_fast),
            'tau_slow': float(tau_slow),
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'converged': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'converged': False
        }


def plot_kernel_comparison(t: np.ndarray, K_learned: np.ndarray, 
                           K_analytic: np.ndarray, params: dict,
                           output_path: Path):
    """Plot learned vs analytic kernel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Kernel comparison
    ax = axes[0]
    ax.plot(t, K_learned, 'b-', linewidth=2, label='Learned (raised-cosine)')
    ax.plot(t, K_analytic, 'r--', linewidth=2, label='Analytic (double-exp)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel value (log-hazard contribution)')
    ax.set_title('Learned vs Analytic Kernel')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1]
    residuals = K_learned - K_analytic
    ax.plot(t, residuals, 'g-', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(t, residuals, 0, alpha=0.3)
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Residual (learned - analytic)')
    ax.set_title(f'Residuals (R² = {params["r_squared"]:.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def evaluate_dense_kernel(model_path: Path = None) -> tuple:
    """
    Evaluate learned kernel on dense 1000-point grid.
    
    Returns
    -------
    t : ndarray
        Time grid [0, 10]s at 0.01s resolution
    K : ndarray
        Kernel values
    kernel_config : dict
        Kernel configuration
    coefficients : dict
        Model coefficients
    """
    if model_path is None:
        model_path = Path('data/model/hybrid_model_results.json')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path) as f:
        model_results = json.load(f)
    
    kernel_config = model_results.get('kernel_config', {})
    coefficients = model_results.get('coefficients', {})
    
    # Dense evaluation: 0 to 10s at 0.01s resolution (1001 points)
    t = np.arange(0, 10.01, 0.01)
    K = evaluate_kernel(kernel_config, coefficients, t)
    
    return t, K, kernel_config, coefficients


def save_dense_kernel(t: np.ndarray, K: np.ndarray, output_path: Path):
    """Save dense kernel evaluation to CSV."""
    df = pd.DataFrame({'time': t, 'kernel_value': K})
    df.to_csv(output_path, index=False)
    print(f"Saved dense kernel to {output_path}")


def plot_kernel_shape(t: np.ndarray, K: np.ndarray, output_path: Path, 
                      kernel_config: dict = None):
    """Plot kernel shape with annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full kernel
    ax = axes[0]
    ax.plot(t, K, 'b-', linewidth=2, label='Learned kernel')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)', fontsize=12)
    ax.set_ylabel('Kernel value (log-hazard contribution)', fontsize=12)
    ax.set_title('Learned Raised-Cosine Kernel (12 bases)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark key features
    max_idx = np.argmax(K)
    min_idx = np.argmin(K)
    ax.scatter([t[max_idx]], [K[max_idx]], color='green', s=100, zorder=5, 
               label=f'Peak: {K[max_idx]:.2f} at {t[max_idx]:.2f}s')
    ax.scatter([t[min_idx]], [K[min_idx]], color='red', s=100, zorder=5,
               label=f'Min: {K[min_idx]:.2f} at {t[min_idx]:.2f}s')
    
    # Find zero crossing
    sign_changes = np.where(np.diff(np.sign(K)))[0]
    if len(sign_changes) > 0:
        zc_idx = sign_changes[0]
        ax.axvline(t[zc_idx], color='orange', linestyle='--', alpha=0.7,
                   label=f'Zero crossing: {t[zc_idx]:.2f}s')
    
    ax.legend(loc='upper right')
    
    # Zoom on early portion (0-2s)
    ax = axes[1]
    mask = t <= 2.0
    ax.plot(t[mask], K[mask], 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)', fontsize=12)
    ax.set_ylabel('Kernel value', fontsize=12)
    ax.set_title('Early Kernel Response (0-2s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Show basis centers if provided
    if kernel_config:
        early_centers = kernel_config.get('early_centers', [])
        for c in early_centers:
            if c <= 2.0:
                ax.axvline(c, color='green', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved kernel shape plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fit analytic kernel to learned raised-cosine kernel')
    parser.add_argument('--dense', action='store_true', 
                        help='Only perform dense kernel evaluation (Phase 1)')
    parser.add_argument('--model', type=str, default='data/model/hybrid_model_results.json',
                        help='Path to model results JSON')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIT ANALYTIC KERNEL (Difference of Exponentials)")
    print("=" * 70)
    
    model_path = Path(args.model)
    
    # Phase 1: Dense kernel evaluation
    print("\n[Phase 1] Dense Kernel Evaluation")
    print("-" * 40)
    
    try:
        t, K_learned, kernel_config, coefficients = evaluate_dense_kernel(model_path)
    except FileNotFoundError as e:
        print(str(e))
        return
    
    print(f"Loaded model from: {model_path}")
    print(f"Kernel config:")
    print(f"  Early centers: {kernel_config.get('early_centers', [])}")
    print(f"  Intm centers: {kernel_config.get('intm_centers', [])}")
    print(f"  Late centers: {kernel_config.get('late_centers', [])}")
    print(f"  Total bases: {len(kernel_config.get('early_centers', [])) + len(kernel_config.get('intm_centers', [])) + len(kernel_config.get('late_centers', []))}")
    
    print(f"\nKernel statistics (1001 points, 0-10s):")
    print(f"  Max: {K_learned.max():.3f} at t={t[np.argmax(K_learned)]:.2f}s")
    print(f"  Min: {K_learned.min():.3f} at t={t[np.argmin(K_learned)]:.2f}s")
    print(f"  Mean: {K_learned.mean():.3f}")
    print(f"  Std: {K_learned.std():.3f}")
    
    # Save dense kernel
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dense_kernel(t, K_learned, output_dir / 'kernel_dense.csv')
    
    # Plot kernel shape
    val_dir = Path('data/validation')
    val_dir.mkdir(parents=True, exist_ok=True)
    plot_kernel_shape(t, K_learned, val_dir / 'kernel_shape.png', kernel_config)
    
    if args.dense:
        print("\n[Phase 1 Complete] Dense kernel evaluation only.")
        return
    
    # Continue with double-exponential fit
    print("\n[Phase 2] Double-Exponential Fit")
    print("-" * 40)
    print("Fitting: K(t) = A*exp(-t/tau_fast) - B*exp(-t/tau_slow)")
    
    fit_result = fit_double_exponential(t, K_learned)
    
    if fit_result['converged']:
        print("\nFit successful!")
        print(f"  A (fast amplitude): {fit_result['A']:.3f}")
        print(f"  B (slow amplitude): {fit_result['B']:.3f}")
        print(f"  tau_fast: {fit_result['tau_fast']:.3f} s")
        print(f"  tau_slow: {fit_result['tau_slow']:.3f} s")
        print(f"  R²: {fit_result['r_squared']:.3f}")
        print(f"  RMSE: {fit_result['rmse']:.4f}")
        
        # Physiological interpretation
        print("\n" + "=" * 50)
        print("PHYSIOLOGICAL INTERPRETATION")
        print("=" * 50)
        print(f"""
tau_fast ({fit_result['tau_fast']:.2f}s):
  - Reflects fast sensorimotor processes
  - Initial optogenetic activation and early circuit response
  - Matches early positive bump in PSTH

tau_slow ({fit_result['tau_slow']:.2f}s):
  - Reflects slower adaptation/habituation
  - Synaptic depression or network-level state change
  - Matches sustained suppression during LED-ON
""")
        
        # Compute analytic kernel
        K_analytic = double_exp_kernel(t, fit_result['A'], fit_result['B'],
                                       fit_result['tau_fast'], fit_result['tau_slow'])
        
        # Plot
        plot_kernel_comparison(t, K_learned, K_analytic, fit_result,
                               val_dir / 'analytic_kernel_fit.png')
        
        # Save results
        output_path = Path('data/model/analytic_kernel_results.json')
        with open(output_path, 'w') as f:
            json.dump(fit_result, f, indent=2)
        print(f"\nSaved results to {output_path}")
        
    else:
        print(f"\nFit failed: {fit_result.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()




