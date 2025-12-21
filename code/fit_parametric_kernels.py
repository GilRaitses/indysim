#!/usr/bin/env python3
"""
Fit Parametric Kernel Models to Learned Raised-Cosine Kernel

This script fits 5 candidate functional forms to the dense kernel evaluation
and compares them using R², AIC, and BIC.

Candidate models:
1. Double-exponential: A*exp(-t/tau1) - B*exp(-t/tau2)
2. Triple-exponential: A*exp(-t/tau1) - B*exp(-t/tau2) + C*exp(-t/tau3)
3. Alpha + exponential: A*(t/tau1)*exp(-t/tau1) - B*exp(-t/tau2)
4. Gamma-difference: A*Gamma(t;a1,b1) - B*Gamma(t;a2,b2)
5. Piecewise exponential: Phase-specific exponentials

Usage:
    python scripts/fit_parametric_kernels.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.special import gamma as gamma_func
from scipy.stats import gamma as gamma_dist
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Parametric Kernel Models
# ============================================================================

def double_exp_kernel(t: np.ndarray, A: float, B: float, 
                      tau1: float, tau2: float) -> np.ndarray:
    """
    Double-exponential kernel.
    K(t) = A * exp(-t/tau1) - B * exp(-t/tau2)
    
    Parameters: A (fast amp), B (slow amp), tau1 (fast), tau2 (slow)
    """
    return A * np.exp(-t / tau1) - B * np.exp(-t / tau2)


def triple_exp_kernel(t: np.ndarray, A: float, B: float, C: float,
                      tau1: float, tau2: float, tau3: float) -> np.ndarray:
    """
    Triple-exponential kernel.
    K(t) = A * exp(-t/tau1) - B * exp(-t/tau2) + C * exp(-t/tau3)
    
    Parameters: A, B, C (amplitudes), tau1, tau2, tau3 (time constants)
    """
    return (A * np.exp(-t / tau1) - 
            B * np.exp(-t / tau2) + 
            C * np.exp(-t / tau3))


def alpha_exp_kernel(t: np.ndarray, A: float, tau1: float, 
                     B: float, tau2: float) -> np.ndarray:
    """
    Alpha function + exponential kernel.
    K(t) = A * (t/tau1) * exp(-t/tau1) - B * exp(-t/tau2)
    
    The alpha function has a delayed peak at t=tau1.
    """
    return A * (t / tau1) * np.exp(-t / tau1) - B * np.exp(-t / tau2)


def gamma_diff_kernel(t: np.ndarray, A: float, a1: float, b1: float,
                      B: float, a2: float, b2: float) -> np.ndarray:
    """
    Gamma-difference kernel.
    K(t) = A * Gamma_pdf(t; a1, scale=b1) - B * Gamma_pdf(t; a2, scale=b2)
    
    Gamma PDF provides flexible shape with delayed peak.
    """
    # Gamma pdf: (t^(a-1) * exp(-t/b)) / (b^a * Gamma(a))
    pdf1 = gamma_dist.pdf(t, a1, scale=b1)
    pdf2 = gamma_dist.pdf(t, a2, scale=b2)
    # Handle edge case at t=0
    pdf1 = np.nan_to_num(pdf1, nan=0.0)
    pdf2 = np.nan_to_num(pdf2, nan=0.0)
    return A * pdf1 - B * pdf2


def piecewise_exp_kernel(t: np.ndarray, A: float, tau1: float, t_switch: float,
                         B: float, tau2: float) -> np.ndarray:
    """
    Piecewise exponential kernel.
    K(t) = A * exp(-t/tau1)                          for t < t_switch
    K(t) = A * exp(-t_switch/tau1) - B*(1-exp(-(t-t_switch)/tau2))  for t >= t_switch
    
    This captures early rise followed by suppression.
    """
    K = np.zeros_like(t)
    early = t < t_switch
    late = ~early
    
    K[early] = A * np.exp(-t[early] / tau1)
    
    # Value at switch point
    K_switch = A * np.exp(-t_switch / tau1)
    # Late phase: decay to suppression
    K[late] = K_switch - B * (1 - np.exp(-(t[late] - t_switch) / tau2))
    
    return K


# ============================================================================
# Model Fitting
# ============================================================================

def compute_metrics(K_true: np.ndarray, K_fit: np.ndarray, n_params: int) -> Dict:
    """Compute R², RMSE, AIC, and BIC."""
    n = len(K_true)
    residuals = K_true - K_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((K_true - np.mean(K_true))**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Log-likelihood (assuming Gaussian errors)
    sigma2 = ss_res / n
    if sigma2 > 0:
        log_lik = -n/2 * np.log(2*np.pi*sigma2) - n/2
    else:
        log_lik = 0
    
    # AIC and BIC
    aic = 2 * n_params - 2 * log_lik
    bic = np.log(n) * n_params - 2 * log_lik
    
    return {
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'aic': float(aic),
        'bic': float(bic),
        'ss_res': float(ss_res),
        'n_params': n_params
    }


def fit_model(model_func: Callable, t: np.ndarray, K: np.ndarray,
              p0: List[float], bounds: Tuple, param_names: List[str],
              model_name: str) -> Dict:
    """
    Fit a parametric model to kernel data.
    
    Returns dict with fitted parameters, metrics, and status.
    """
    try:
        popt, pcov = curve_fit(model_func, t, K, p0=p0, bounds=bounds, 
                               maxfev=10000, method='trf')
        
        K_fit = model_func(t, *popt)
        metrics = compute_metrics(K, K_fit, len(popt))
        
        # Parameter standard errors
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.zeros(len(popt))
        
        result = {
            'model_name': model_name,
            'formula': get_formula(model_name),
            'converged': True,
            'parameters': {name: float(val) for name, val in zip(param_names, popt)},
            'std_errors': {name: float(val) for name, val in zip(param_names, perr)},
            'K_fit': K_fit.tolist(),
            **metrics
        }
        
        return result
        
    except Exception as e:
        return {
            'model_name': model_name,
            'formula': get_formula(model_name),
            'converged': False,
            'error': str(e),
            'r_squared': 0.0,
            'aic': np.inf,
            'bic': np.inf,
            'n_params': len(p0)
        }


def get_formula(model_name: str) -> str:
    """Return formula string for each model."""
    formulas = {
        'double_exp': 'K(t) = A*exp(-t/τ₁) - B*exp(-t/τ₂)',
        'triple_exp': 'K(t) = A*exp(-t/τ₁) - B*exp(-t/τ₂) + C*exp(-t/τ₃)',
        'alpha_exp': 'K(t) = A*(t/τ₁)*exp(-t/τ₁) - B*exp(-t/τ₂)',
        'gamma_diff': 'K(t) = A*Γ(t;α₁,β₁) - B*Γ(t;α₂,β₂)',
        'piecewise': 'K(t) = A*exp(-t/τ₁) for t<t_s, then decay'
    }
    return formulas.get(model_name, 'Unknown')


def fit_all_models(t: np.ndarray, K: np.ndarray) -> Dict[str, Dict]:
    """Fit all 5 parametric models and return results."""
    
    results = {}
    
    # 1. Double-exponential (4 params)
    print("  Fitting double-exponential...")
    results['double_exp'] = fit_model(
        model_func=double_exp_kernel,
        t=t, K=K,
        p0=[2.0, 1.5, 0.3, 3.0],  # A, B, tau1, tau2
        bounds=([0, 0, 0.05, 0.5], [10, 10, 2.0, 15.0]),
        param_names=['A', 'B', 'tau_fast', 'tau_slow'],
        model_name='double_exp'
    )
    
    # 2. Triple-exponential (6 params)
    print("  Fitting triple-exponential...")
    results['triple_exp'] = fit_model(
        model_func=triple_exp_kernel,
        t=t, K=K,
        p0=[2.0, 3.0, 0.5, 0.3, 2.0, 6.0],  # A, B, C, tau1, tau2, tau3
        bounds=([0, 0, -3, 0.05, 0.5, 2.5], [10, 10, 3, 1.0, 8.0, 15.0]),
        param_names=['A', 'B', 'C', 'tau_fast', 'tau_slow', 'tau_rec'],
        model_name='triple_exp'
    )
    
    # 3. Alpha + exponential (4 params)
    print("  Fitting alpha + exponential...")
    results['alpha_exp'] = fit_model(
        model_func=alpha_exp_kernel,
        t=t, K=K,
        p0=[5.0, 0.3, 1.5, 3.0],  # A, tau1, B, tau2
        bounds=([0, 0.05, 0, 0.5], [50, 2.0, 10, 15.0]),
        param_names=['A', 'tau_fast', 'B', 'tau_slow'],
        model_name='alpha_exp'
    )
    
    # 4. Gamma-difference (6 params)
    print("  Fitting gamma-difference...")
    results['gamma_diff'] = fit_model(
        model_func=gamma_diff_kernel,
        t=t, K=K,
        p0=[5.0, 2.0, 0.2, 3.0, 3.0, 1.0],  # A, a1, b1, B, a2, b2
        bounds=([0, 1.0, 0.05, 0, 1.0, 0.1], [50, 10, 2.0, 30, 10, 5.0]),
        param_names=['A', 'alpha1', 'beta1', 'B', 'alpha2', 'beta2'],
        model_name='gamma_diff'
    )
    
    # 5. Piecewise exponential (5 params)
    print("  Fitting piecewise exponential...")
    results['piecewise'] = fit_model(
        model_func=piecewise_exp_kernel,
        t=t, K=K,
        p0=[1.5, 0.5, 1.0, 2.0, 3.0],  # A, tau1, t_switch, B, tau2
        bounds=([0, 0.1, 0.3, 0, 0.5], [10, 2.0, 3.0, 10, 15.0]),
        param_names=['A', 'tau_rise', 't_switch', 'B', 'tau_decay'],
        model_name='piecewise'
    )
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(t: np.ndarray, K_true: np.ndarray, 
                    results: Dict[str, Dict], output_path: Path):
    """Plot comparison of all fitted models."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Colors for each model
    colors = {
        'double_exp': 'red',
        'triple_exp': 'blue',
        'alpha_exp': 'green',
        'gamma_diff': 'purple',
        'piecewise': 'orange'
    }
    
    # Plot 1: All models overlaid
    ax = axes[0, 0]
    ax.plot(t, K_true, 'k-', linewidth=2.5, label='Learned kernel', alpha=0.8)
    for name, res in results.items():
        if res['converged']:
            K_fit = np.array(res['K_fit'])
            r2 = res['r_squared']
            ax.plot(t, K_fit, '--', color=colors[name], linewidth=1.5, 
                    label=f'{name} (R²={r2:.3f})')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kernel value')
    ax.set_title('All Models Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2-6: Individual models with residuals
    model_order = ['double_exp', 'triple_exp', 'alpha_exp', 'gamma_diff', 'piecewise']
    
    for idx, name in enumerate(model_order):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]
        
        res = results[name]
        if res['converged']:
            K_fit = np.array(res['K_fit'])
            ax.plot(t, K_true, 'k-', linewidth=2, label='Learned', alpha=0.7)
            ax.plot(t, K_fit, '-', color=colors[name], linewidth=2, label='Fitted')
            ax.fill_between(t, K_true, K_fit, alpha=0.2, color=colors[name])
            
            r2 = res['r_squared']
            aic = res['aic']
            n_params = res['n_params']
            ax.set_title(f'{name}\nR²={r2:.3f}, AIC={aic:.1f}, params={n_params}')
        else:
            ax.text(0.5, 0.5, f'{name}\nFailed to converge', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name)
        
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kernel value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def plot_best_model(t: np.ndarray, K_true: np.ndarray, 
                    best_result: Dict, output_path: Path):
    """Detailed plot of the best model."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    K_fit = np.array(best_result['K_fit'])
    residuals = K_true - K_fit
    
    # Top-left: Full comparison
    ax = axes[0, 0]
    ax.plot(t, K_true, 'k-', linewidth=2.5, label='Learned kernel')
    ax.plot(t, K_fit, 'r--', linewidth=2, label=f'Best fit: {best_result["model_name"]}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Kernel value', fontsize=12)
    ax.set_title(f'Best Model: {best_result["model_name"]} (R²={best_result["r_squared"]:.4f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Early portion (0-2s)
    ax = axes[0, 1]
    mask = t <= 2.0
    ax.plot(t[mask], K_true[mask], 'k-', linewidth=2.5, label='Learned')
    ax.plot(t[mask], K_fit[mask], 'r--', linewidth=2, label='Fitted')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Kernel value', fontsize=12)
    ax.set_title('Early Response (0-2s)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Residuals
    ax = axes[1, 0]
    ax.plot(t, residuals, 'g-', linewidth=1.5)
    ax.fill_between(t, residuals, 0, alpha=0.3, color='green')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title(f'Residuals (RMSE={best_result["rmse"]:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Parameter table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create parameter table
    params = best_result['parameters']
    std_errs = best_result.get('std_errors', {})
    
    table_data = []
    for name, val in params.items():
        err = std_errs.get(name, 0)
        table_data.append([name, f'{val:.4f}', f'±{err:.4f}'])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Parameter', 'Value', 'Std Error'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Add metrics
    metrics_text = (f"Model: {best_result['model_name']}\n"
                   f"Formula: {best_result['formula']}\n"
                   f"R² = {best_result['r_squared']:.4f}\n"
                   f"RMSE = {best_result['rmse']:.4f}\n"
                   f"AIC = {best_result['aic']:.1f}\n"
                   f"BIC = {best_result['bic']:.1f}")
    ax.text(0.5, 0.1, metrics_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved best model plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("PARAMETRIC KERNEL FITTING (Phase 2)")
    print("=" * 70)
    
    # Load dense kernel
    kernel_path = Path('data/model/kernel_dense.csv')
    if not kernel_path.exists():
        print(f"Dense kernel not found: {kernel_path}")
        print("Run 'python scripts/fit_analytic_kernel.py --dense' first.")
        return
    
    df = pd.read_csv(kernel_path)
    t = df['time'].values
    K = df['kernel_value'].values
    
    print(f"\nLoaded kernel: {len(t)} points, t=[{t.min():.2f}, {t.max():.2f}]s")
    print(f"Kernel range: [{K.min():.3f}, {K.max():.3f}]")
    
    # Fit all models
    print("\nFitting 5 parametric models...")
    results = fit_all_models(t, K)
    
    # Summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<15} {'R²':<10} {'RMSE':<10} {'AIC':<12} {'BIC':<12} {'Params':<8} {'Status'}")
    print("-" * 75)
    
    for name, res in results.items():
        if res['converged']:
            print(f"{name:<15} {res['r_squared']:.4f}     {res['rmse']:.4f}     "
                  f"{res['aic']:<12.1f} {res['bic']:<12.1f} {res['n_params']:<8} OK")
        else:
            print(f"{name:<15} {'--':<10} {'--':<10} {'--':<12} {'--':<12} "
                  f"{res['n_params']:<8} FAILED")
    
    # Find best model (by R²)
    converged = {k: v for k, v in results.items() if v['converged']}
    if not converged:
        print("\nNo models converged!")
        return
    
    best_name = max(converged.keys(), key=lambda k: converged[k]['r_squared'])
    best_result = results[best_name]
    
    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'=' * 70}")
    print(f"Formula: {best_result['formula']}")
    print(f"R² = {best_result['r_squared']:.4f}")
    print(f"Parameters:")
    for name, val in best_result['parameters'].items():
        std_err = best_result.get('std_errors', {}).get(name, 0)
        print(f"  {name}: {val:.4f} ± {std_err:.4f}")
    
    # Check decision gate
    print(f"\n{'=' * 70}")
    print("DECISION GATE")
    print(f"{'=' * 70}")
    if best_result['r_squared'] >= 0.95:
        print(f"✓ R² = {best_result['r_squared']:.4f} >= 0.95")
        print("  SKIP Phases 3-4 (PySR/SINDy)")
        print("  PROCEED to Phase 5 (Interpretation)")
    else:
        print(f"✗ R² = {best_result['r_squared']:.4f} < 0.95")
        print("  CONTINUE to Phase 3 (PySR) and Phase 4 (SINDy)")
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for name, res in results.items():
        json_results[name] = {k: v for k, v in res.items() if k != 'K_fit'}
        if res['converged']:
            # Don't save full K_fit array in JSON
            json_results[name]['K_fit_sample'] = res['K_fit'][:10]  # First 10 values
    
    json_results['best_model'] = best_name
    json_results['decision'] = 'skip_symbolic' if best_result['r_squared'] >= 0.95 else 'continue_symbolic'
    
    output_path = output_dir / 'parametric_fits.json'
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Save best model separately
    best_output = {
        'model_name': best_result['model_name'],
        'formula': best_result['formula'],
        'parameters': best_result['parameters'],
        'std_errors': best_result.get('std_errors', {}),
        'r_squared': best_result['r_squared'],
        'rmse': best_result['rmse'],
        'aic': best_result['aic'],
        'bic': best_result['bic'],
        'n_params': best_result['n_params']
    }
    with open(output_dir / 'best_parametric_kernel.json', 'w') as f:
        json.dump(best_output, f, indent=2)
    
    # Plots
    val_dir = Path('data/validation')
    val_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(t, K, results, val_dir / 'parametric_comparison.png')
    plot_best_model(t, K, best_result, val_dir / 'best_parametric_fit.png')
    
    print("\nPhase 2 complete!")


if __name__ == '__main__':
    main()


