#!/usr/bin/env python3
"""
Compare Alternative Kernel Forms

Fits multiple parametric kernel forms to the reference condition and 
compares via AIC/BIC to justify (or not) the gamma-difference choice.

Kernel forms tested:
1. Gamma-difference: A*Gamma(t;α₁,β₁) - B*Gamma(t;α₂,β₂)  [6 params]
2. Double-exponential: A*exp(-t/τ₁) - B*exp(-t/τ₂)        [4 params]
3. Alpha-difference: A*Alpha(t;τ₁) - B*Alpha(t;τ₂)        [4 params]
4. Triple-exponential: A*exp(-t/τ₁) - B*exp(-t/τ₂) + C*exp(-t/τ₃)  [6 params]

Usage:
    python scripts/compare_kernel_forms.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist
from scipy.special import factorial
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
from typing import Dict, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Kernel Functions
# ============================================================================

def gamma_diff_kernel(t: np.ndarray, A: float, a1: float, b1: float,
                      B: float, a2: float, b2: float) -> np.ndarray:
    """Gamma-difference kernel (6 parameters)."""
    pdf1 = gamma_dist.pdf(t, a1, scale=b1)
    pdf2 = gamma_dist.pdf(t, a2, scale=b2)
    pdf1 = np.nan_to_num(pdf1, nan=0.0)
    pdf2 = np.nan_to_num(pdf2, nan=0.0)
    return A * pdf1 - B * pdf2


def double_exp_kernel(t: np.ndarray, A: float, tau1: float, 
                      B: float, tau2: float) -> np.ndarray:
    """Double-exponential kernel (4 parameters)."""
    return A * np.exp(-t / tau1) - B * np.exp(-t / tau2)


def alpha_function(t: np.ndarray, tau: float) -> np.ndarray:
    """Alpha function: (t/tau) * exp(-t/tau), peaks at t=tau."""
    return (t / tau) * np.exp(-t / tau)


def alpha_diff_kernel(t: np.ndarray, A: float, tau1: float,
                      B: float, tau2: float) -> np.ndarray:
    """Alpha-difference kernel (4 parameters)."""
    return A * alpha_function(t, tau1) - B * alpha_function(t, tau2)


def triple_exp_kernel(t: np.ndarray, A: float, tau1: float,
                      B: float, tau2: float, C: float, tau3: float) -> np.ndarray:
    """Triple-exponential kernel (6 parameters)."""
    return A * np.exp(-t / tau1) - B * np.exp(-t / tau2) + C * np.exp(-t / tau3)


def single_exp_kernel(t: np.ndarray, A: float, tau: float) -> np.ndarray:
    """Single exponential decay (2 parameters).
    
    Note: Cannot capture biphasic dynamics; included for comparison only.
    """
    return A * np.exp(-t / tau)


def raised_cosine_basis(t: np.ndarray, n_bases: int = 12, 
                        t_peak_min: float = 0.1, t_peak_max: float = 8.0) -> np.ndarray:
    """Generate raised-cosine basis functions (Pillow et al., 2008).
    
    Args:
        t: Time array
        n_bases: Number of basis functions
        t_peak_min: Minimum peak time
        t_peak_max: Maximum peak time
        
    Returns:
        Basis matrix of shape (len(t), n_bases)
    """
    log_peaks = np.linspace(np.log(t_peak_min + 0.01), np.log(t_peak_max), n_bases)
    peaks = np.exp(log_peaks)
    delta = (log_peaks[-1] - log_peaks[0]) / (n_bases - 1) if n_bases > 1 else 1.0
    
    basis = np.zeros((len(t), n_bases))
    for i, peak in enumerate(peaks):
        log_t = np.log(t + 0.01)
        log_peak = np.log(peak + 0.01)
        arg = (log_t - log_peak) / delta
        valid = np.abs(arg) <= 1
        basis[valid, i] = 0.5 * (1 + np.cos(np.pi * arg[valid]))
    
    return basis


# ============================================================================
# Model Fitting and Comparison
# ============================================================================

def fit_kernel(kernel_func: Callable, t: np.ndarray, K: np.ndarray,
               p0: list, bounds: tuple, n_params: int, name: str) -> Dict:
    """Fit a kernel function and compute metrics."""
    try:
        popt, pcov = curve_fit(kernel_func, t, K, p0=p0, bounds=bounds, maxfev=10000)
        K_fit = kernel_func(t, *popt)
        
        # Compute metrics
        n = len(t)
        residuals = K - K_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((K - np.mean(K)) ** 2)
        
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Log-likelihood (assuming Gaussian errors)
        sigma2 = ss_res / n
        if sigma2 > 0:
            log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
        else:
            log_lik = 0
        
        # AIC and BIC
        k = n_params
        aic = 2 * k - 2 * log_lik
        bic = np.log(n) * k - 2 * log_lik
        
        return {
            'name': name,
            'n_params': n_params,
            'params': popt.tolist(),
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'ss_res': float(ss_res),
            'log_lik': float(log_lik),
            'aic': float(aic),
            'bic': float(bic),
            'K_fit': K_fit.tolist(),
            'converged': True
        }
    except Exception as e:
        return {
            'name': name,
            'n_params': n_params,
            'error': str(e),
            'converged': False
        }


def fit_raised_cosine(t: np.ndarray, K: np.ndarray, n_bases: int = 12) -> Dict:
    """Fit raised-cosine basis via least squares."""
    basis = raised_cosine_basis(t, n_bases)
    
    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(basis, K, rcond=None)
    K_fit = basis @ coeffs
    
    n = len(t)
    ss_res = np.sum((K - K_fit) ** 2)
    ss_tot = np.sum((K - np.mean(K)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((K - K_fit) ** 2))
    
    # Log-likelihood (Gaussian errors)
    sigma2 = ss_res / n
    log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2 if sigma2 > 0 else 0
    
    k = n_bases
    aic = 2 * k - 2 * log_lik
    bic = np.log(n) * k - 2 * log_lik
    
    return {
        'name': f'Raised-cosine ({n_bases} basis)',
        'n_params': n_bases,
        'params': coeffs.tolist(),
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'ss_res': float(ss_res),
        'log_lik': float(log_lik),
        'aic': float(aic),
        'bic': float(bic),
        'K_fit': K_fit.tolist(),
        'converged': True,
        'interpretation': 'Overparameterized; no biological interpretability'
    }


def compare_kernels(t: np.ndarray, K: np.ndarray) -> Dict:
    """Compare all kernel forms.
    
    Includes:
    - Single exponential (2 params) - too simple, cannot capture biphasic
    - Double exponential (4 params) - no shape control
    - Alpha-difference (4 params) - intermediate complexity
    - Gamma-difference (6 params) - biologically interpretable
    - Triple exponential (6 params) - overparameterized
    - Raised-cosine 12-basis (12 params) - flexible but uninterpretable
    """
    
    results = {}
    
    # 1. Single-exponential (2 params) - CANNOT capture biphasic dynamics
    print("Fitting single-exponential...")
    results['single_exp'] = fit_kernel(
        single_exp_kernel, t, K,
        p0=[1.0, 1.0],
        bounds=([0.001, 0.01], [100.0, 20.0]),
        n_params=2,
        name='Single-exponential'
    )
    # Note: R² < 0 is expected for biphasic kernels
    if results['single_exp'].get('converged'):
        results['single_exp']['interpretation'] = 'Too simple; cannot capture biphasic dynamics'
    
    # 2. Double-exponential (4 params)
    print("Fitting double-exponential...")
    results['double_exp'] = fit_kernel(
        double_exp_kernel, t, K,
        p0=[2.0, 0.3, 1.5, 3.0],
        bounds=([0.01, 0.1, 0.01, 0.5], [10.0, 2.0, 10.0, 10.0]),
        n_params=4,
        name='Double-exponential'
    )
    if results['double_exp'].get('converged'):
        results['double_exp']['interpretation'] = 'No shape control'
    
    # 3. Alpha-difference (4 params)
    print("Fitting alpha-difference...")
    results['alpha_diff'] = fit_kernel(
        alpha_diff_kernel, t, K,
        p0=[5.0, 0.3, 2.0, 3.0],
        bounds=([0.01, 0.1, 0.01, 0.5], [20.0, 2.0, 20.0, 10.0]),
        n_params=4,
        name='Alpha-difference'
    )
    if results['alpha_diff'].get('converged'):
        results['alpha_diff']['interpretation'] = 'Intermediate complexity'
    
    # 4. Gamma-difference (6 params) - PREFERRED MODEL
    print("Fitting gamma-difference...")
    results['gamma_diff'] = fit_kernel(
        gamma_diff_kernel, t, K,
        p0=[0.5, 2.2, 0.13, 12.0, 4.4, 0.87],
        bounds=([0.01, 1.0, 0.05, 0.1, 2.0, 0.3], [5.0, 6.0, 0.5, 50.0, 8.0, 2.0]),
        n_params=6,
        name='Gamma-difference'
    )
    if results['gamma_diff'].get('converged'):
        results['gamma_diff']['interpretation'] = 'Biologically interpretable; timescales map to neural processes'
    
    # 5. Triple-exponential (6 params)
    print("Fitting triple-exponential...")
    results['triple_exp'] = fit_kernel(
        triple_exp_kernel, t, K,
        p0=[2.0, 0.3, 3.0, 2.0, 0.5, 5.0],
        bounds=([0.01, 0.1, 0.01, 0.5, -5.0, 1.0], [10.0, 2.0, 20.0, 5.0, 5.0, 15.0]),
        n_params=6,
        name='Triple-exponential'
    )
    if results['triple_exp'].get('converged'):
        results['triple_exp']['interpretation'] = 'Overparameterized'
    
    # 6. Raised-cosine (12 basis) - flexible but uninterpretable
    print("Fitting raised-cosine (12 basis)...")
    results['raised_cosine_12'] = fit_raised_cosine(t, K, n_bases=12)
    
    return results


def plot_comparison(t: np.ndarray, K: np.ndarray, results: Dict, output_path: Path):
    """Plot all kernel fits overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'single_exp': '#d62728',      # Red - too simple
        'double_exp': 'tab:orange',
        'alpha_diff': 'tab:purple',
        'gamma_diff': 'tab:green',    # Green - preferred model
        'triple_exp': 'tab:brown',
        'raised_cosine_12': 'tab:blue'
    }
    
    # Left: All fits
    ax = axes[0]
    ax.plot(t, K, 'k-', linewidth=2, alpha=0.5, label='Raised-cosine reference')
    
    for key, result in results.items():
        if result.get('converged', False):
            K_fit = np.array(result['K_fit'])
            label = f"{result['name']} (R²={result['r_squared']:.3f})"
            ax.plot(t, K_fit, color=colors[key], linewidth=1.5, linestyle='--', label=label)
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel value (log-hazard)')
    ax.set_title('Kernel Form Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # Right: AIC/BIC comparison
    ax = axes[1]
    
    names = []
    aics = []
    bics = []
    
    for key, result in results.items():
        if result.get('converged', False):
            names.append(result['name'])
            aics.append(result['aic'])
            bics.append(result['bic'])
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, aics, width, label='AIC', color='steelblue')
    ax.bar(x + width/2, bics, width, label='BIC', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Information Criterion (lower is better)')
    ax.set_title('Model Comparison: AIC and BIC')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mark the winner
    if aics:
        best_aic_idx = np.argmin(aics)
        best_bic_idx = np.argmin(bics)
        ax.annotate('Best AIC', xy=(best_aic_idx - width/2, aics[best_aic_idx]),
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    print("=" * 70)
    print("KERNEL FORM COMPARISON")
    print("=" * 70)
    
    # Load the dense kernel (reconstructed from raised-cosine fit)
    kernel_path = Path('data/model/kernel_dense.csv')
    
    if not kernel_path.exists():
        print(f"Kernel not found: {kernel_path}")
        print("Run fit_hybrid_model.py first to generate the reference kernel.")
        return
    
    df = pd.read_csv(kernel_path)
    t = df['time'].values
    K = df['kernel_value'].values
    
    print(f"Loaded reference kernel: {len(t)} points")
    print(f"  Range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"\nAIC/BIC computation: All models use Gaussian likelihood with n={len(t)} points.")
    print("  AIC = 2k - 2·log_lik, BIC = log(n)·k - 2·log_lik")
    print("  Values are directly comparable across all models.")
    
    # Compare kernels
    print("\nFitting alternative kernel forms...")
    results = compare_kernels(t, K)
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Params':>6} {'R²':>8} {'AIC':>12} {'BIC':>12} {'ΔAIC':>8} {'ΔBIC':>8}")
    print("-" * 80)
    
    # Find best AIC/BIC
    valid_results = {k: v for k, v in results.items() if v.get('converged', False)}
    if valid_results:
        best_aic = min(r['aic'] for r in valid_results.values())
        best_bic = min(r['bic'] for r in valid_results.values())
    else:
        best_aic = best_bic = 0
    
    for key, result in results.items():
        if result.get('converged', False):
            delta_aic = result['aic'] - best_aic
            delta_bic = result['bic'] - best_bic
            print(f"{result['name']:<20} {result['n_params']:>6} {result['r_squared']:>8.4f} "
                  f"{result['aic']:>12.1f} {result['bic']:>12.1f} "
                  f"{delta_aic:>8.1f} {delta_bic:>8.1f}")
        else:
            print(f"{result.get('name', key):<20} FAILED: {result.get('error', 'Unknown')[:40]}")
    
    print("-" * 80)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if valid_results:
        # Find winners
        aic_winner = min(valid_results.items(), key=lambda x: x[1]['aic'])
        bic_winner = min(valid_results.items(), key=lambda x: x[1]['bic'])
        
        print(f"  Best AIC: {aic_winner[1]['name']}")
        print(f"  Best BIC: {bic_winner[1]['name']}")
        
        # Check if gamma-difference is justified
        if 'gamma_diff' in valid_results:
            gd = valid_results['gamma_diff']
            gd_delta_aic = gd['aic'] - best_aic
            gd_delta_bic = gd['bic'] - best_bic
            
            if gd_delta_aic < 2 and gd_delta_bic < 2:
                print(f"\n  Gamma-difference is WITHIN 2 units of best model.")
                print(f"  This supports using gamma-difference for its interpretability.")
            elif gd_delta_aic < 10 and gd_delta_bic < 10:
                print(f"\n  Gamma-difference is within 10 units of best model.")
                print(f"  Moderate support; consider mentioning alternatives.")
            else:
                print(f"\n  WARNING: Gamma-difference is NOT well-supported.")
                print(f"  ΔAIC = {gd_delta_aic:.1f}, ΔBIC = {gd_delta_bic:.1f}")
                print(f"  Consider using {aic_winner[1]['name']} instead.")
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove non-serializable data
    save_results = {}
    for key, result in results.items():
        save_results[key] = {k: v for k, v in result.items() if k != 'K_fit'}
    
    output_path = output_dir / 'kernel_form_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Plot
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(t, K, results, fig_dir / 'kernel_form_comparison.png')
    
    return results


if __name__ == '__main__':
    main()


