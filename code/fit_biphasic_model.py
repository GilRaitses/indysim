#!/usr/bin/env python3
"""
Fit LNP Model with Biphasic Temporal Kernel

Implements separate early and late kernel phases:
- Early (0-1.5s): Fine-resolution bases, non-negative constraint
- Late (1.5-6s): Broader bases, unconstrained

Also includes LED-off rebound term and regularization.

Usage:
    python scripts/fit_biphasic_model.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.optimize import minimize

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


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


def create_biphasic_kernel(
    led1Val_ton: np.ndarray,
    split_point: float = 1.5,
    n_early: int = 3,
    n_late: int = 4
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create biphasic kernel design matrix.
    
    Parameters
    ----------
    led1Val_ton : ndarray
        Time since last LED onset (seconds)
    split_point : float
        Time to split early/late phases (seconds)
    n_early : int
        Number of early kernel bases
    n_late : int
        Number of late kernel bases
    
    Returns
    -------
    early_basis : ndarray
        Early kernel design matrix (n_obs x n_early)
    late_basis : ndarray
        Late kernel design matrix (n_obs x n_late)
    feature_names : list
        Names of kernel features
    """
    t_relative = led1Val_ton.copy()
    
    # Early kernel: 0 to split_point (fine resolution)
    early_centers = np.linspace(0, split_point, n_early)
    early_width = split_point / (n_early - 1) * 0.8 if n_early > 1 else split_point * 0.5
    early_basis = raised_cosine_basis(t_relative, early_centers, early_width)
    
    # Late kernel: split_point to 6s (broader)
    late_centers = np.linspace(split_point, 6.0, n_late)
    late_width = (6.0 - split_point) / (n_late - 1) * 0.8 if n_late > 1 else 1.0
    late_basis = raised_cosine_basis(t_relative, late_centers, late_width)
    
    feature_names = [f'kernel_early_{i+1}' for i in range(n_early)]
    feature_names += [f'kernel_late_{i+1}' for i in range(n_late)]
    
    return early_basis, late_basis, feature_names


def compute_led_off_rebound(
    data: pd.DataFrame,
    tau: float = 1.5
) -> np.ndarray:
    """
    Compute LED-off rebound term.
    
    rebound(t) = exp(-(t - t_off) / tau) if t > t_off else 0
    """
    led_threshold = 50
    
    # Detect LED off times per track
    data = data.sort_values(['experiment_id', 'track_id', 'time'])
    
    rebound = np.zeros(len(data))
    
    for (exp, track), group in data.groupby(['experiment_id', 'track_id']):
        times = group['time'].values
        led = group['led1Val'].values if 'led1Val' in group.columns else np.zeros(len(group))
        
        led_on = led > led_threshold
        
        # Find offset times (transition from on to off)
        offsets = []
        for i in range(1, len(led_on)):
            if led_on[i-1] and not led_on[i]:
                offsets.append(times[i])
        
        # Compute time since most recent offset
        idx = group.index
        for i, t in enumerate(times):
            # Find most recent offset before t
            recent_offsets = [off for off in offsets if off < t]
            if recent_offsets:
                t_since_off = t - max(recent_offsets)
                rebound[idx[i]] = np.exp(-t_since_off / tau)
    
    return rebound


def build_biphasic_design_matrix(
    data: pd.DataFrame,
    split_point: float = 1.5,
    n_early: int = 3,
    n_late: int = 4,
    include_rebound: bool = True,
    rebound_tau: float = 1.5
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Build full design matrix with biphasic kernel.
    
    Returns
    -------
    X : ndarray
        Design matrix
    feature_names : list
        Feature names
    early_idx : list
        Indices of early kernel coefficients (for constraints)
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
    elif 'led1Val' in data.columns:
        features.append(data['led1Val'].values / 250.0)
    else:
        features.append(np.zeros(n))
    feature_names.append('LED1_scaled')
    
    # Biphasic kernel
    if 'led1Val_ton' in data.columns:
        time_since = data['led1Val_ton'].fillna(999).values
        early_basis, late_basis, kernel_names = create_biphasic_kernel(
            time_since, split_point, n_early, n_late
        )
        
        # Track early kernel indices for constraints
        early_start_idx = len(features)
        
        for j in range(early_basis.shape[1]):
            features.append(early_basis[:, j])
        for j in range(late_basis.shape[1]):
            features.append(late_basis[:, j])
        
        feature_names.extend(kernel_names)
        early_idx = list(range(early_start_idx, early_start_idx + n_early))
    else:
        early_idx = []
    
    # LED-off rebound
    if include_rebound and 'led1Val' in data.columns:
        rebound = compute_led_off_rebound(data, tau=rebound_tau)
        features.append(rebound)
        feature_names.append('led_off_rebound')
    
    X = np.column_stack(features)
    return X, feature_names, early_idx


def fit_with_constraints(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    early_idx: List[int],
    alpha: float = 0.1,
    lambda_ridge: float = 0.01,
    lambda_smooth: float = 0.01
) -> Dict:
    """
    Fit NB-GLM with non-negative constraints on early kernel
    and ridge + smoothness regularization.
    """
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels not installed', 'converged': False}
    
    # First fit unconstrained to get starting values
    try:
        model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
        fit_init = model.fit()
        beta_init = fit_init.params.values
    except Exception as e:
        beta_init = np.zeros(X.shape[1])
        beta_init[0] = -4.5  # Reasonable intercept
    
    # Define objective with penalties
    def objective(beta):
        # Log-likelihood (NB)
        mu = np.exp(X @ beta)
        mu = np.clip(mu, 1e-10, 1e10)
        
        # NB log-likelihood
        r = 1 / alpha  # NB dispersion parameter
        ll = np.sum(
            y * np.log(mu / (mu + r)) + 
            r * np.log(r / (mu + r))
        )
        
        # Ridge penalty on kernel coefficients
        kernel_idx = [i for i, name in enumerate(feature_names) if 'kernel' in name]
        ridge_penalty = lambda_ridge * np.sum(beta[kernel_idx]**2)
        
        # Smoothness penalty (adjacent kernel coefficients)
        smooth_penalty = 0
        for i in range(len(kernel_idx) - 1):
            smooth_penalty += (beta[kernel_idx[i+1]] - beta[kernel_idx[i]])**2
        smooth_penalty *= lambda_smooth
        
        return -(ll - ridge_penalty - smooth_penalty)
    
    # Bounds: early kernel >= 0, others unconstrained
    bounds = [(None, None)] * len(beta_init)
    for idx in early_idx:
        bounds[idx] = (0, None)  # Non-negative
    
    # Optimize
    result = minimize(
        objective,
        beta_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if result.success:
        beta = result.x
        
        # Compute approximate SEs using Hessian
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            n_params = len(beta)
            hess = np.zeros((n_params, n_params))
            for i in range(n_params):
                def grad_i(b):
                    b_plus = b.copy()
                    b_plus[i] += eps
                    b_minus = b.copy()
                    b_minus[i] -= eps
                    return (objective(b_plus) - objective(b_minus)) / (2 * eps)
                hess[i, :] = approx_fprime(beta, grad_i, eps)
            
            try:
                cov = np.linalg.inv(hess)
                se = np.sqrt(np.diag(cov))
            except:
                se = np.full(n_params, np.nan)
        except:
            se = np.full(len(beta), np.nan)
        
        # Compute AIC
        mu = np.exp(X @ beta)
        r = 1 / alpha
        ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
        k = len(beta)
        aic = 2 * k - 2 * ll
        
        return {
            'coefficients': dict(zip(feature_names, beta)),
            'std_errors': dict(zip(feature_names, se)),
            'aic': float(aic),
            'llf': float(ll),
            'converged': True,
            'n_obs': len(y),
            'n_events': int(y.sum()),
            'optimizer_success': result.success,
            'optimizer_message': result.message
        }
    else:
        return {
            'error': result.message,
            'converged': False
        }


def load_model_data(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load data for model fitting."""
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        csv_files = sorted(data_dir.glob('*_events.csv'))[:2]
    
    dfs = []
    for f in csv_files:
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
    
    return data, y


def main():
    print("=" * 60)
    print("FIT BIPHASIC LNP MODEL")
    print("=" * 60)
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    data, y = load_model_data(data_dir)
    print(f"  Loaded {len(data):,} observations")
    print(f"  Events: {y.sum():,}")
    
    # Test different split points
    split_points = [1.0, 1.5, 2.0]
    results = []
    
    for split in split_points:
        print(f"\n--- Testing split point: {split}s ---")
        
        X, feature_names, early_idx = build_biphasic_design_matrix(
            data,
            split_point=split,
            n_early=3,
            n_late=4,
            include_rebound=True,
            rebound_tau=1.5
        )
        
        print(f"  Features: {len(feature_names)}")
        print(f"  Early kernel indices: {early_idx}")
        
        result = fit_with_constraints(
            X, y, feature_names, early_idx,
            alpha=0.1,
            lambda_ridge=0.01,
            lambda_smooth=0.01
        )
        
        if result.get('converged', False):
            print(f"  AIC: {result['aic']:.2f}")
            print(f"  Significant coefficients:")
            for name, coef in result['coefficients'].items():
                se = result['std_errors'].get(name, np.nan)
                if not np.isnan(se) and abs(coef / se) > 1.96:
                    print(f"    {name}: {coef:.4f} (SE={se:.4f})")
        else:
            print(f"  Failed: {result.get('error', 'Unknown')}")
        
        result['split_point'] = split
        results.append(result)
    
    # Select best by AIC
    valid_results = [r for r in results if r.get('converged', False)]
    if valid_results:
        best = min(valid_results, key=lambda x: x['aic'])
        print(f"\n\n{'='*60}")
        print(f"BEST MODEL: split_point = {best['split_point']}s")
        print(f"{'='*60}")
        print(f"AIC: {best['aic']:.2f}")
        print(f"Log-likelihood: {best['llf']:.2f}")
        
        print("\nCoefficients:")
        for name, coef in best['coefficients'].items():
            se = best['std_errors'].get(name, np.nan)
            z = coef / se if not np.isnan(se) and se > 0 else 0
            sig = "*" if abs(z) > 1.96 else ""
            print(f"  {name}: {coef:.4f} (SE={se:.4f}) {sig}")
        
        # Save best model
        output_path = Path('data/model/biphasic_model_results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON
        save_result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                       for k, v in best.items()}
        save_result['coefficients'] = {k: float(v) for k, v in best['coefficients'].items()}
        save_result['std_errors'] = {k: float(v) if not np.isnan(v) else None 
                                     for k, v in best['std_errors'].items()}
        
        with open(output_path, 'w') as f:
            json.dump(save_result, f, indent=2)
        
        print(f"\nSaved to {output_path}")
    else:
        print("\nNo models converged!")


if __name__ == '__main__':
    main()




