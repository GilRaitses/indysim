#!/usr/bin/env python3
"""
Run NB-GLM LNP Model Pipeline

Loads binned data, fits LNP model with cluster-robust SEs,
runs cross-validation and diagnostics, generates outputs.

Usage:
    python scripts/run_hazard_pipeline.py --input data/processed/binned_0.5s.parquet
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from hazard_model module
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hazard_model import (
    fit_nb_glm,
    run_all_diagnostics,
    cross_validate_kernel_params,
    extract_kernel_shape,
    kernel_confidence_bands,
    find_peak_latency,
    generate_coefficient_table,
    raised_cosine_basis,
    estimate_dispersion
)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_binned_data(parquet_path: Path) -> pd.DataFrame:
    """Load binned data from parquet file."""
    print(f"Loading binned data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} bins, {df.columns.tolist()}")
    return df


def prepare_design_matrix(df: pd.DataFrame, add_ar_term: bool = True) -> tuple:
    """
    Prepare design matrix for NB-GLM.
    
    Returns X (design matrix), y (response), feature_names, and cluster_groups.
    
    Parameters
    ----------
    add_ar_term : bool
        If True, add AR(1) lag term (Y_lag1) to handle temporal autocorrelation
    """
    # Add intercept
    df['intercept'] = 1.0
    
    # Add AR(1) term if requested (per research recommendation for ACF=0.999)
    if add_ar_term:
        df = df.sort_values(['experiment_id', 'track_id', 'bin_start'])
        df['Y_lag1'] = df.groupby(['experiment_id', 'track_id'])['Y'].shift(1, fill_value=0)
        print("  Added AR(1) lag term (Y_lag1)")
    
    # Feature columns
    base_features = ['intercept', 'LED1_scaled', 'LED2_scaled', 'LED1xLED2',
                     'phase_sin', 'phase_cos', 'speed_z', 'curvature_z']
    
    # Add AR term to features
    if add_ar_term:
        base_features.append('Y_lag1')
    
    # Kernel columns
    kernel_cols = [c for c in df.columns if c.startswith('kernel_')]
    
    feature_names = base_features + kernel_cols
    
    # Check for missing columns
    available = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"  Warning: Missing columns {missing}")
    
    X = df[available].copy()
    y = df['Y'].values
    
    # Exposure (bin width)
    if 'bin_width' in df.columns:
        exposure = df['bin_width'].values
    else:
        exposure = np.full(len(df), 0.5)  # Default 0.5s bins
    
    # Cluster groups (experiment_id)
    if 'experiment_id' in df.columns:
        cluster_groups = df['experiment_id'].values
    else:
        cluster_groups = None
    
    return X, y, available, exposure, cluster_groups


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_and_diagnose(
    X: pd.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray,
    cluster_groups: np.ndarray,
    feature_names: list
) -> dict:
    """Fit NB-GLM and run diagnostics."""
    print("\n=== Fitting NB-GLM ===")
    
    # Estimate dispersion from data
    mean_y = np.mean(y)
    var_y = np.var(y)
    if mean_y > 0:
        alpha = max(0.1, (var_y - mean_y) / max(mean_y**2, 0.001))
    else:
        alpha = 1.0
    print(f"  Estimated dispersion alpha: {alpha:.3f}")
    
    # Fit model
    results = fit_nb_glm(
        X, y, 
        exposure=exposure, 
        alpha=alpha,
        cluster_groups=cluster_groups
    )
    
    if results.get('converged', False):
        print("  Model converged!")
        print(f"  Deviance: {results.get('deviance', 0):.2f}")
        print(f"  Dispersion ratio: {results.get('dispersion_ratio', 0):.3f} (target ~1.0)")
        print(f"  AIC: {results.get('aic', 0):.2f}")
        
        if results.get('robust_se', False):
            print(f"  Using cluster-robust SEs ({results.get('n_clusters', 0)} clusters)")
    else:
        print(f"  Model failed: {results.get('error', 'Unknown')}")
        return results
    
    # Run diagnostics
    print("\n=== Running Diagnostics ===")
    diagnostics = run_all_diagnostics(results, y)
    
    for name, diag in diagnostics.items():
        if isinstance(diag, dict) and 'status' in diag:
            print(f"  {name}: {diag['status']}")
    
    results['diagnostics'] = diagnostics
    
    return results


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cv(df: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> dict:
    """Run cross-validation for kernel parameters."""
    print("\n=== Cross-Validation ===")
    
    cv_results = cross_validate_kernel_params(
        df, y,
        n_bases_options=[3, 4, 5],
        window_options=[(0.0, 2.0), (0.0, 3.0), (0.0, 4.0)],
        exposure=exposure
    )
    
    best = cv_results.get('best_params', {})
    print(f"  Best params: {best}")
    print(f"  Best deviance: {cv_results.get('best_deviance', 0):.2f}")
    
    return cv_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_kernel_response(
    results: dict,
    feature_names: list,
    output_path: Path
):
    """Plot temporal kernel response with confidence bands."""
    # Extract kernel coefficients
    kernel_names = [n for n in feature_names if n.startswith('kernel_')]
    if not kernel_names:
        print("  No kernel features found, skipping kernel plot")
        return
    
    phi_hat = np.array([results['coefficients'].get(n, 0) for n in kernel_names])
    
    # Get kernel parameters
    n_bases = len(kernel_names)
    window = (0.0, 3.0)  # Default window
    width = 0.6
    centers = np.linspace(window[0], window[1], n_bases)
    
    # Extract kernel shape
    t_grid, K, RR = extract_kernel_shape(phi_hat, centers, width)
    
    # Get confidence bands if we have covariance
    try:
        # Construct phi covariance from SEs (diagonal approximation)
        phi_se = np.array([results['std_errors'].get(n, 0.1) for n in kernel_names])
        phi_cov = np.diag(phi_se**2)
        
        _, RR, RR_lower, RR_upper = kernel_confidence_bands(
            phi_hat, phi_cov, centers, width, t_grid
        )
    except:
        RR_lower = RR
        RR_upper = RR
    
    # Find peak
    peak_latency, peak_rr = find_peak_latency(t_grid, K)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_grid, RR, 'b-', lw=2, label='Rate Ratio')
    ax.fill_between(t_grid, RR_lower, RR_upper, alpha=0.3, color='blue', 
                    label='95% CI')
    ax.axhline(y=1, color='gray', linestyle='--', lw=1)
    ax.axvline(x=peak_latency, color='red', linestyle=':', lw=1.5,
               label=f'Peak at {peak_latency:.2f}s (RR={peak_rr:.2f})')
    
    ax.set_xlabel('Time since stimulus onset (s)')
    ax.set_ylabel('Rate Ratio (exp(kernel))')
    ax.set_title('Temporal Kernel: Stimulus-Response Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def save_coefficient_table(results: dict, feature_names: list, output_path: Path):
    """Generate and save coefficient table."""
    table = generate_coefficient_table(results, feature_names)
    table.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")
    
    # Print significant coefficients
    print("\n  Significant coefficients (p < 0.05):")
    sig = table[table['p'] < 0.05]
    for _, row in sig.iterrows():
        print(f"    {row['feature']}: RR={row['RR']:.3f} ({row['pct_change']:+.1f}%), p={row['p']:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LNP model pipeline')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to binned parquet file')
    parser.add_argument('--output', type=str, default='data/model/',
                        help='Output directory')
    parser.add_argument('--run-cv', action='store_true',
                        help='Run cross-validation for kernel params')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_binned_data(input_path)
    
    # Prepare design matrix
    X, y, feature_names, exposure, cluster_groups = prepare_design_matrix(df)
    print(f"\n  Design matrix: {X.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Events: {y.sum()} ({100*y.mean():.2f}%)")
    
    # Cross-validation (optional)
    if args.run_cv:
        cv_results = run_cv(df, y, exposure)
        with open(output_dir / 'cv_results.json', 'w') as f:
            # Convert numpy types
            cv_clean = {k: v for k, v in cv_results.items() if k != 'tested_params'}
            cv_clean['tested_params'] = [str(p) for p in cv_results.get('tested_params', [])]
            json.dump(cv_clean, f, indent=2, default=str)
        print(f"  Saved {output_dir / 'cv_results.json'}")
    
    # Fit model
    results = fit_and_diagnose(X, y, exposure, cluster_groups, feature_names)
    
    if not results.get('converged', False):
        print("\nModel did not converge. Exiting.")
        return
    
    # Save results
    print("\n=== Saving Results ===")
    
    # Model results JSON
    results_clean = {k: v for k, v in results.items() 
                     if k not in ['fitted_values', 'resid_pearson']}
    with open(output_dir / 'model_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"  Saved {output_dir / 'model_results.json'}")
    
    # Coefficient table
    save_coefficient_table(results, feature_names, output_dir / 'coefficient_table.csv')
    
    # Kernel plot
    plot_kernel_response(results, feature_names, output_dir / 'kernel_response.png')
    
    print("\n=== Pipeline Complete ===")


if __name__ == '__main__':
    main()




