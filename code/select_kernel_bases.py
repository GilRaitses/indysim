#!/usr/bin/env python3
"""
AIC-Based Kernel Selection for LNP Model

Compares models with different:
- Number of raised-cosine bases (2, 3, 4, 5)
- Temporal windows (0-2s, 0-4s, 0-6s)

Selects optimal configuration by minimum AIC.

Usage:
    python scripts/select_kernel_bases.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import itertools

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed")


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


def create_kernel_design(
    led1Val_ton: np.ndarray,
    n_bases: int,
    window: Tuple[float, float],
    width: float = 0.6
) -> np.ndarray:
    """Create kernel design matrix columns."""
    centers = np.linspace(window[0], window[1], n_bases)
    t_relative = -led1Val_ton  # Negative = past
    return raised_cosine_basis(t_relative, centers, width)


def build_simple_design_matrix(
    data: pd.DataFrame,
    n_bases: int = 4,
    window: Tuple[float, float] = (-3.0, 0.0)
) -> np.ndarray:
    """
    Build simple design matrix with intercept, LED1, and kernel bases.
    """
    n = len(data)
    features = []
    
    # Intercept
    features.append(np.ones(n))
    
    # LED1 intensity
    if 'LED1_scaled' in data.columns:
        features.append(data['LED1_scaled'].values)
    elif 'led1Val' in data.columns:
        features.append(data['led1Val'].values / 250.0)
    
    # Temporal kernel
    if 'led1Val_ton' in data.columns:
        time_since = data['led1Val_ton'].fillna(999).values
        kernel = create_kernel_design(time_since, n_bases, window)
        for j in range(n_bases):
            features.append(kernel[:, j])
    
    return np.column_stack(features)


def fit_model_and_get_aic(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1
) -> Tuple[float, float, bool]:
    """
    Fit NB-GLM and return AIC.
    
    Returns
    -------
    aic : float
        AIC value
    deviance : float
        Model deviance
    converged : bool
        Whether model converged
    """
    if not HAS_STATSMODELS:
        return np.inf, np.inf, False
    
    try:
        model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
        fit = model.fit()
        return fit.aic, fit.deviance, fit.converged
    except Exception as e:
        return np.inf, np.inf, False


def run_kernel_selection(
    data: pd.DataFrame,
    y: np.ndarray,
    n_bases_options: List[int] = [2, 3, 4, 5],
    window_options: List[Tuple[float, float]] = [(-2.0, 0.0), (-4.0, 0.0), (-6.0, 0.0)]
) -> Dict:
    """
    Compare all combinations of kernel parameters by AIC.
    """
    results = {
        'tested_configs': [],
        'best_config': None,
        'best_aic': np.inf
    }
    
    print(f"Testing {len(n_bases_options) * len(window_options)} configurations...")
    
    for n_bases, window in itertools.product(n_bases_options, window_options):
        config = {'n_bases': n_bases, 'window': list(window)}
        
        X = build_simple_design_matrix(data, n_bases=n_bases, window=window)
        aic, deviance, converged = fit_model_and_get_aic(X, y)
        
        n_params = 1 + 1 + n_bases  # intercept + LED1 + kernel bases
        
        config_result = {
            'n_bases': n_bases,
            'window': list(window),
            'aic': float(aic),
            'deviance': float(deviance),
            'converged': bool(converged),
            'n_params': n_params
        }
        results['tested_configs'].append(config_result)
        
        status = "✓" if converged else "✗"
        print(f"  {status} n_bases={n_bases}, window={window}: AIC={aic:.1f}, deviance={deviance:.1f}")
        
        if aic < results['best_aic'] and converged:
            results['best_aic'] = aic
            results['best_config'] = config_result
    
    return results


def load_model_data(data_dir: Path, n_files: int = 4) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load data for model fitting."""
    event_files = sorted(data_dir.glob('*_0to250PWM_*_events.csv'))[:n_files]
    
    if not event_files:
        event_files = sorted(data_dir.glob('*_events.csv'))[:n_files]
    
    if not event_files:
        raise FileNotFoundError(f"No event files in {data_dir}")
    
    dfs = []
    for f in event_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Response variable
    if 'is_reorientation_start' in data.columns:
        y = data['is_reorientation_start'].astype(int).values
    elif 'is_reorientation' in data.columns:
        y = data['is_reorientation'].astype(int).values
    else:
        raise ValueError("No reorientation column found")
    
    return data, y


def main():
    print("=" * 60)
    print("AIC-BASED KERNEL SELECTION")
    print("=" * 60)
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    try:
        data, y = load_model_data(data_dir)
        print(f"  Loaded {len(data):,} observations, {y.sum():,} events")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run kernel selection
    print("\nRunning kernel selection...")
    results = run_kernel_selection(
        data, y,
        n_bases_options=[2, 3, 4, 5],
        window_options=[(-2.0, 0.0), (-4.0, 0.0), (-6.0, 0.0)]
    )
    
    # Report best configuration
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    
    if results['best_config']:
        best = results['best_config']
        print(f"  Number of bases: {best['n_bases']}")
        print(f"  Window: {best['window']} seconds")
        print(f"  AIC: {best['aic']:.2f}")
        print(f"  Deviance: {best['deviance']:.2f}")
        print(f"  Number of parameters: {best['n_params']}")
    else:
        print("  No valid configuration found!")
    
    # AIC table
    print("\n" + "=" * 60)
    print("AIC COMPARISON TABLE")
    print("=" * 60)
    
    configs = sorted(results['tested_configs'], key=lambda x: x['aic'])
    print(f"{'n_bases':>8} {'window':>12} {'AIC':>12} {'deviance':>12} {'delta_AIC':>10}")
    print("-" * 56)
    
    best_aic = configs[0]['aic'] if configs else 0
    for c in configs:
        delta = c['aic'] - best_aic
        window_str = f"[{c['window'][0]:.0f}, {c['window'][1]:.0f}]"
        print(f"{c['n_bases']:>8} {window_str:>12} {c['aic']:>12.1f} {c['deviance']:>12.1f} {delta:>10.1f}")
    
    # Save results
    output_path = Path('data/model/kernel_selection.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()




