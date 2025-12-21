#!/usr/bin/env python3
"""
Lightweight Cross-Validation for Kernel Parameters

Memory-efficient version that:
1. Processes one configuration at a time
2. Clears memory between fits
3. Uses subset sampling for large datasets

Usage:
    python scripts/run_cv_lightweight.py
"""

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

# Reduce memory by importing only what we need
import warnings
warnings.filterwarnings('ignore')


def load_data_subset(parquet_path: str, sample_frac: float = 0.3) -> pd.DataFrame:
    """Load a random subset of data to reduce memory."""
    df = pd.read_parquet(parquet_path)
    
    # Sample by experiment to maintain structure
    experiments = df['experiment_id'].unique()
    n_exp = len(experiments)
    
    # Use subset of experiments for CV
    n_sample = max(3, int(n_exp * sample_frac))
    sampled_exps = np.random.choice(experiments, n_sample, replace=False)
    
    df_subset = df[df['experiment_id'].isin(sampled_exps)].copy()
    print(f"Using {len(df_subset):,} rows from {n_sample} experiments")
    
    return df_subset


def build_kernel_bases(t: np.ndarray, n_bases: int, window: tuple) -> np.ndarray:
    """Build raised-cosine kernel bases."""
    centers = np.linspace(window[0], window[1], n_bases)
    width = (window[1] - window[0]) / (n_bases - 1) * 0.8
    
    basis = np.zeros((len(t), n_bases))
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def fit_single_config(df: pd.DataFrame, n_bases: int, window: tuple) -> dict:
    """Fit a single kernel configuration and return metrics."""
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.families import NegativeBinomial
    
    # Rebuild kernel columns for this configuration
    if 'led1Val_ton' in df.columns:
        t = df['led1Val_ton'].values
    else:
        t = np.zeros(len(df))
    
    basis = build_kernel_bases(t, n_bases, window)
    
    # Build design matrix
    X = pd.DataFrame({
        'intercept': 1.0,
        'LED1_scaled': df['LED1_scaled'] if 'LED1_scaled' in df.columns else df['led1Val'] / 250.0,
        'LED2_scaled': df['LED2_scaled'] if 'LED2_scaled' in df.columns else df['led2Val'] / 15.0,
    })
    
    for j in range(n_bases):
        X[f'kernel_{j+1}'] = basis[:, j]
    
    y = df['Y'].values
    exposure = df['bin_width'].values if 'bin_width' in df.columns else np.full(len(df), 0.5)
    offset = np.log(exposure)
    
    # Fit model
    try:
        model = GLM(y, X, family=NegativeBinomial(alpha=0.1), offset=offset)
        fit = model.fit(maxiter=50)
        
        result = {
            'n_bases': n_bases,
            'window': window,
            'aic': float(fit.aic),
            'deviance': float(fit.deviance),
            'converged': bool(fit.converged),
            'n_obs': len(y)
        }
    except Exception as e:
        result = {
            'n_bases': n_bases,
            'window': window,
            'aic': float('inf'),
            'deviance': float('inf'),
            'converged': False,
            'error': str(e)
        }
    
    # Clear memory
    del X, y, basis, model
    gc.collect()
    
    return result


def main():
    print("=" * 60)
    print("Lightweight Cross-Validation")
    print("=" * 60)
    
    # Load subset of data
    data_path = Path("data/processed/binned_0.5s.parquet")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    np.random.seed(42)
    df = load_data_subset(str(data_path), sample_frac=0.5)
    
    # Grid of configurations
    n_bases_options = [3, 4, 5]
    window_options = [(0.0, 2.0), (0.0, 3.0), (0.0, 4.0)]
    
    results = []
    total = len(n_bases_options) * len(window_options)
    
    print(f"\nTesting {total} configurations...")
    print("-" * 60)
    
    for i, (n_bases, window) in enumerate(product(n_bases_options, window_options)):
        print(f"[{i+1}/{total}] n_bases={n_bases}, window={window}...", end=" ", flush=True)
        
        result = fit_single_config(df, n_bases, window)
        results.append(result)
        
        if result['converged']:
            print(f"AIC={result['aic']:.1f}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')}")
        
        gc.collect()
    
    # Find best
    valid = [r for r in results if r['converged']]
    if valid:
        best = min(valid, key=lambda x: x['aic'])
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION:")
        print(f"  n_bases: {best['n_bases']}")
        print(f"  window: {best['window']}")
        print(f"  AIC: {best['aic']:.2f}")
        print("=" * 60)
    
    # Save results
    output_path = Path("data/model/cv_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'best': best if valid else None
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()




