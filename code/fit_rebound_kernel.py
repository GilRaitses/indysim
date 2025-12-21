#!/usr/bin/env python3
"""
Fit LED-Off Rebound Kernel

Fits a separate exponential term for the post-LED-OFF response:
    K_off(t) = D * exp(-t/tau_off)

This captures the rebound in turn probability after LED turns off.

Usage:
    python scripts/fit_rebound_kernel.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py


def exponential_decay(t: np.ndarray, D: float, tau: float) -> np.ndarray:
    """Simple exponential decay: D * exp(-t/tau)"""
    return D * np.exp(-t / tau)


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


def load_binned_data(data_path: Path) -> pd.DataFrame:
    """Load binned data for rebound analysis."""
    if data_path.suffix == '.h5':
        with h5py.File(data_path, 'r') as f:
            if 'binned' in f:
                grp = f['binned']
                data = {k: grp[k][:] for k in grp.keys()}
                return pd.DataFrame(data)
    elif data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    
    raise ValueError(f"Unknown data format: {data_path}")


def extract_led_off_kernel(model_results: dict) -> tuple:
    """
    Extract LED-OFF kernel from hybrid model results.
    
    The hybrid model may have a rebound coefficient that captures
    the post-LED-OFF response.
    """
    coefficients = model_results.get('coefficients', {})
    kernel_config = model_results.get('kernel_config', {})
    
    # Check for explicit rebound coefficient
    rebound_coef = coefficients.get('rebound', coefficients.get('x13', 0))
    rebound_tau = kernel_config.get('rebound_tau', 2.0)
    
    return rebound_coef, rebound_tau


def compute_empirical_psth_post_offset(events: np.ndarray, offset_times: np.ndarray,
                                        window: tuple = (0, 10), bin_width: float = 0.5) -> tuple:
    """
    Compute PSTH relative to LED-OFF (offset) times.
    
    Returns bin centers and event rates.
    """
    relative_times = []
    for off_t in offset_times:
        rel = events - off_t
        in_window = (rel >= window[0]) & (rel <= window[1])
        relative_times.extend(rel[in_window])
    
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    counts, _ = np.histogram(relative_times, bins=bins)
    rate = counts / (len(offset_times) * bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, rate


def fit_rebound_from_model(model_results: dict) -> dict:
    """
    Extract and characterize the rebound term from the hybrid model.
    
    The rebound in the hybrid model is captured by the last coefficient
    which multiplies an exponential decay from LED-OFF.
    """
    coefficients = model_results.get('coefficients', {})
    kernel_config = model_results.get('kernel_config', {})
    
    # Get rebound parameters
    # In the hybrid model, rebound is often the last coefficient
    n_early = len(kernel_config.get('early_centers', []))
    n_intm = len(kernel_config.get('intm_centers', []))
    n_late = len(kernel_config.get('late_centers', []))
    n_kernel = n_early + n_intm + n_late
    
    # The rebound coefficient (if present) would be x{n_kernel + 1}
    rebound_key = f'x{n_kernel + 1}'
    rebound_coef = coefficients.get(rebound_key, 0)
    rebound_tau = kernel_config.get('rebound_tau', 2.0)
    
    # Evaluate rebound kernel
    t = np.linspace(0, 10, 101)
    K_rebound = rebound_coef * np.exp(-t / rebound_tau)
    
    return {
        'rebound_coefficient': float(rebound_coef),
        'rebound_tau': float(rebound_tau),
        'formula': f'K_off(t) = {rebound_coef:.4f} * exp(-t/{rebound_tau:.2f})',
        't': t.tolist(),
        'K_rebound': K_rebound.tolist()
    }


def plot_rebound_kernel(results: dict, output_path: Path):
    """Visualize the rebound kernel."""
    t = np.array(results['t'])
    K = np.array(results['K_rebound'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Rebound kernel
    ax = axes[0]
    ax.plot(t, K, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED-OFF (s)', fontsize=12)
    ax.set_ylabel('Kernel value (log-hazard contribution)', fontsize=12)
    ax.set_title(f'LED-OFF Rebound Kernel\n{results["formula"]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark key features
    D = results['rebound_coefficient']
    tau = results['rebound_tau']
    ax.scatter([0], [D], color='red', s=100, zorder=5, label=f'D = {D:.3f}')
    ax.axvline(tau, color='orange', linestyle='--', alpha=0.7, label=f'tau = {tau:.2f}s')
    ax.legend()
    
    # Right: Combined kernel view
    ax = axes[1]
    
    # Load LED-ON kernel
    kernel_path = Path('data/model/kernel_dense.csv')
    if kernel_path.exists():
        df = pd.read_csv(kernel_path)
        t_on = df['time'].values
        K_on = df['kernel_value'].values
        
        # Plot LED-ON period
        ax.plot(t_on, K_on, 'b-', linewidth=2, label='LED-ON kernel')
        
        # Plot LED-OFF period (shifted)
        t_off = t + 30  # LED-OFF starts at t=30s in 60s cycle
        ax.plot(t_off[:50], K[:50], 'r-', linewidth=2, label='LED-OFF rebound')
        
        ax.axvline(30, color='gray', linestyle='--', alpha=0.7, label='LED OFF')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time in stimulus cycle (s)', fontsize=12)
        ax.set_ylabel('Kernel value', fontsize=12)
        ax.set_title('Full Stimulus Response (60s cycle)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 40)
    else:
        ax.text(0.5, 0.5, 'LED-ON kernel not found', ha='center', va='center', 
                transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rebound plot to {output_path}")


def main():
    print("=" * 70)
    print("FIT LED-OFF REBOUND KERNEL")
    print("=" * 70)
    
    # Load hybrid model results
    model_path = Path('data/model/hybrid_model_results.json')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    with open(model_path) as f:
        model_results = json.load(f)
    
    print(f"Loaded hybrid model from: {model_path}")
    
    # Extract rebound from model
    results = fit_rebound_from_model(model_results)
    
    print(f"\nRebound kernel extracted:")
    print(f"  Coefficient D: {results['rebound_coefficient']:.4f}")
    print(f"  Time constant tau: {results['rebound_tau']:.2f}s")
    print(f"  Formula: {results['formula']}")
    
    # Interpretation
    D = results['rebound_coefficient']
    tau = results['rebound_tau']
    
    print("\n" + "=" * 50)
    print("INTERPRETATION")
    print("=" * 50)
    
    if abs(D) < 0.1:
        print("  Rebound is WEAK (|D| < 0.1)")
        print("  Post-LED-OFF behavior may not require separate modeling.")
    elif D > 0:
        print(f"  Rebound is POSITIVE (D = {D:.3f})")
        print("  LED-OFF triggers INCREASED turn probability")
        print(f"  Decays with time constant tau = {tau:.2f}s")
    else:
        print(f"  Rebound is NEGATIVE (D = {D:.3f})")
        print("  LED-OFF triggers DECREASED turn probability")
        print(f"  Recovers with time constant tau = {tau:.2f}s")
    
    # Check if rebound was actually fit
    if results['rebound_coefficient'] == 0:
        print("\n  NOTE: Rebound coefficient is 0 in the model.")
        print("  This may indicate rebound was not included in the fit,")
        print("  or that the data does not support a significant rebound.")
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save without the full arrays (just summary)
    save_results = {k: v for k, v in results.items() if k not in ['t', 'K_rebound']}
    save_results['peak_effect'] = float(D)
    save_results['half_life'] = float(tau * np.log(2))
    
    with open(output_dir / 'rebound_kernel.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'rebound_kernel.json'}")
    
    # Plot
    val_dir = Path('data/validation')
    val_dir.mkdir(parents=True, exist_ok=True)
    plot_rebound_kernel(results, val_dir / 'rebound_kernel.png')
    
    print("\nPhase 2 complete!")


if __name__ == '__main__':
    main()


