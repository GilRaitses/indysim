#!/usr/bin/env python3
"""
Refit LNP Model with Optimal Kernel Parameters

Uses AIC-selected optimal configuration:
- 5 raised-cosine bases
- [-6, 0] second temporal window

Usage:
    python scripts/refit_optimal_model.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

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
    n_bases: int = 5,
    window: Tuple[float, float] = (-6.0, 0.0),
    width: float = 1.0
) -> np.ndarray:
    """Create kernel design matrix with optimal parameters."""
    centers = np.linspace(window[0], window[1], n_bases)
    t_relative = -led1Val_ton
    return raised_cosine_basis(t_relative, centers, width)


def build_optimal_design_matrix(
    data: pd.DataFrame,
    n_bases: int = 5,
    window: Tuple[float, float] = (-6.0, 0.0)
) -> Tuple[np.ndarray, List[str]]:
    """
    Build design matrix with optimal kernel configuration.
    """
    n = len(data)
    features = []
    feature_names = []
    
    # Intercept
    features.append(np.ones(n))
    feature_names.append('intercept')
    
    # LED1 intensity
    if 'LED1_scaled' in data.columns:
        features.append(data['LED1_scaled'].values)
        feature_names.append('LED1_scaled')
    elif 'led1Val' in data.columns:
        features.append(data['led1Val'].values / 250.0)
        feature_names.append('LED1_scaled')
    
    # LED2 intensity (optional)
    if 'LED2_scaled' in data.columns:
        features.append(data['LED2_scaled'].values)
        feature_names.append('LED2_scaled')
    
    # LED interaction (optional)
    if 'LED1xLED2' in data.columns:
        features.append(data['LED1xLED2'].values)
        feature_names.append('LED1xLED2')
    
    # Phase covariates
    if 'phase_sin' in data.columns:
        features.append(data['phase_sin'].values)
        feature_names.append('phase_sin')
    if 'phase_cos' in data.columns:
        features.append(data['phase_cos'].values)
        feature_names.append('phase_cos')
    
    # Speed and curvature
    if 'speed_z' in data.columns:
        features.append(data['speed_z'].values)
        feature_names.append('speed_z')
    if 'curvature_z' in data.columns:
        features.append(data['curvature_z'].values)
        feature_names.append('curvature_z')
    
    # Temporal kernel with optimal parameters
    if 'led1Val_ton' in data.columns:
        time_since = data['led1Val_ton'].fillna(999).values
        kernel = create_kernel_design(time_since, n_bases=n_bases, window=window)
        for j in range(n_bases):
            features.append(kernel[:, j])
            feature_names.append(f'kernel_{j+1}')
    
    X = np.column_stack(features)
    return X, feature_names


def fit_optimal_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cluster_groups: np.ndarray = None,
    alpha: float = 0.1
) -> Dict:
    """Fit NB-GLM with optimal kernel and return results."""
    
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels not installed', 'converged': False}
    
    try:
        model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
        
        if cluster_groups is not None:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
            robust_se = True
            n_clusters = len(np.unique(cluster_groups))
        else:
            fit = model.fit()
            robust_se = False
            n_clusters = 0
        
        # Build results dict
        results = {
            'coefficients': dict(zip(feature_names, fit.params)),
            'std_errors': dict(zip(feature_names, fit.bse)),
            'pvalues': dict(zip(feature_names, fit.pvalues)),
            'conf_int_lower': dict(zip(feature_names, fit.conf_int()[0])),
            'conf_int_upper': dict(zip(feature_names, fit.conf_int()[1])),
            'robust_se': robust_se,
            'n_clusters': n_clusters,
            'deviance': float(fit.deviance),
            'pearson_chi2': float(fit.pearson_chi2),
            'df_resid': int(fit.df_resid),
            'aic': float(fit.aic),
            'bic': float(fit.bic),
            'llf': float(fit.llf),
            'dispersion': alpha,
            'n_obs': len(y),
            'n_events': int(y.sum()),
            'converged': bool(fit.converged),
            'dispersion_ratio': float(fit.pearson_chi2 / fit.df_resid),
            'kernel_config': {
                'n_bases': 5,
                'window': [-6.0, 0.0],
                'width': 1.0
            }
        }
        
        # Add pseudo-R2
        # Fit null model for comparison
        null_model = GLM(y, np.ones((len(y), 1)), family=NegativeBinomial(alpha=alpha))
        null_fit = null_model.fit()
        null_llf = float(null_fit.llf)
        
        results['null_llf'] = null_llf
        results['pseudo_r2'] = 1 - (fit.llf / null_llf) if null_llf < 0 else 0.0
        
        return results
        
    except Exception as e:
        return {'error': str(e), 'converged': False}


def load_model_data(data_dir: Path, n_files: int = 4) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load data for model fitting."""
    # Use 0to250PWM condition
    event_files = sorted(data_dir.glob('*_0to250PWM_*_events.csv'))[:n_files]
    
    if not event_files:
        event_files = sorted(data_dir.glob('*_events.csv'))[:n_files]
    
    if not event_files:
        raise FileNotFoundError(f"No event files in {data_dir}")
    
    dfs = []
    for i, f in enumerate(event_files):
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Response variable
    if 'is_reorientation_start' in data.columns:
        y = data['is_reorientation_start'].astype(int).values
    elif 'is_reorientation' in data.columns:
        y = data['is_reorientation'].astype(int).values
    else:
        raise ValueError("No reorientation column found")
    
    # Cluster groups
    cluster_groups = data['experiment_id'].values
    
    return data, y, cluster_groups


def main():
    print("=" * 60)
    print("REFIT LNP MODEL WITH OPTIMAL KERNEL")
    print("=" * 60)
    print("\nOptimal configuration (from AIC selection):")
    print("  - 5 raised-cosine bases")
    print("  - [-6, 0] second temporal window")
    print("  - Width: 1.0s")
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    try:
        data, y, cluster_groups = load_model_data(data_dir)
        print(f"  Loaded {len(data):,} observations")
        print(f"  Events: {y.sum():,} ({100*y.mean():.3f}%)")
        print(f"  Experiments: {len(np.unique(cluster_groups))}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Build design matrix with optimal kernel
    print("\nBuilding design matrix...")
    X, feature_names = build_optimal_design_matrix(data, n_bases=5, window=(-6.0, 0.0))
    print(f"  Features: {feature_names}")
    
    # Fit model
    print("\nFitting NB-GLM with cluster-robust SEs...")
    results = fit_optimal_model(X, y, feature_names, cluster_groups)
    
    if results.get('converged', False):
        print("\n✓ Model converged!")
        print(f"  Deviance: {results['deviance']:.2f}")
        print(f"  AIC: {results['aic']:.2f}")
        print(f"  Pseudo-R²: {results['pseudo_r2']:.4f}")
        print(f"  Dispersion ratio: {results['dispersion_ratio']:.3f}")
        
        print("\nSignificant coefficients (p < 0.05):")
        for name in feature_names:
            p = results['pvalues'].get(name, 1.0)
            if p < 0.05:
                coef = results['coefficients'][name]
                se = results['std_errors'][name]
                print(f"  {name}: {coef:.4f} (SE={se:.4f}, p={p:.4f})")
        
        # Save results
        output_path = Path('data/model/model_results_optimal.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to {output_path}")
        
        # Also update the main model_results.json
        main_path = Path('data/model/model_results.json')
        with open(main_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Updated {main_path}")
        
    else:
        print(f"\n✗ Model failed: {results.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()




