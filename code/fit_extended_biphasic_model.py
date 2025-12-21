#!/usr/bin/env python3
"""
Fit LNP Model with Extended Triphasic Temporal Kernel

Triphasic temporal kernel design (BO-optimized 2025-12-11):
- Early kernel: 0-1.5s with centers at [0.2, 0.63, 1.07, 1.5] - width 0.30s
- Intermediate kernel: 1.5-3s with centers at [2.0, 2.5] - width 0.60s  
- Late kernel: 3-9s with centers at [3.0, 4.2, 5.4, 6.6, 7.8, 9.0] - width 2.49s
- Only first early basis constrained non-negative
- Intercept anchored to empirical baseline (~-7.3)
- Physics-informed refractory penalty for IEI regularization

Usage:
    python scripts/fit_extended_biphasic_model.py
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


def create_triphasic_kernel(
    led1Val_ton: np.ndarray,
    late_end: float = 10.0,
    use_bo_optimal: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """
    Create triphasic kernel design matrix.
    
    Structure (BO-optimized):
    - Early kernel (0-1.5s): 4 bases at [0.2, 0.63, 1.07, 1.5]s, width 0.30s
      Only first basis (0.2s) is constrained non-negative
    - Intermediate kernel (1.5-3s): 2 bases at [2.0, 2.5]s, width 0.6s
      All unconstrained - captures rising suppression
    - Late kernel (3-9s): 6 bases at [3.0, 4.2, 5.4, 6.6, 7.8, 9.0]s, width 2.49s
      All unconstrained
    
    Parameters
    ----------
    led1Val_ton : ndarray
        Time since last LED onset (seconds)
    late_end : float
        End of late kernel window (seconds)
    use_bo_optimal : bool
        If True, use BO-optimized config. If False, use legacy config.
    
    Returns
    -------
    early_basis : ndarray
        Early kernel design matrix (n_obs x n_early)
    intm_basis : ndarray
        Intermediate kernel design matrix (n_obs x 2)
    late_basis : ndarray
        Late kernel design matrix (n_obs x n_late)
    feature_names : list
        Names of kernel features
    kernel_config : dict
        Configuration used for kernel construction
    """
    t_relative = led1Val_ton.copy()
    
    if use_bo_optimal:
        # BO-optimized configuration (2025-12-11)
        # Score: 0.8186, W-ISE: 0.444, Correlation: 0.882
        early_centers = np.array([0.2, 0.6333, 1.0667, 1.5])
        early_width = 0.30
        intm_centers = np.array([2.0, 2.5])
        intm_width = 0.6
        late_centers = np.array([3.0, 4.2, 5.4, 6.6, 7.8, 9.0])
        late_width = 2.494
    else:
        # Legacy configuration
        early_centers = np.array([0.2, 0.7, 1.4])
        early_width = 0.4
        intm_centers = np.array([2.0, 2.5])
        intm_width = 0.6
        late_centers = np.array([3.0, 5.0, 7.0, 9.0])
        late_width = 1.8
    
    early_basis = raised_cosine_basis(t_relative, early_centers, early_width)
    intm_basis = raised_cosine_basis(t_relative, intm_centers, intm_width)
    late_basis = raised_cosine_basis(t_relative, late_centers, late_width)
    
    # Feature names
    feature_names = [f'kernel_early_{i+1}' for i in range(len(early_centers))]
    feature_names += [f'kernel_intm_{i+1}' for i in range(len(intm_centers))]
    feature_names += [f'kernel_late_{i+1}' for i in range(len(late_centers))]
    
    kernel_config = {
        'early_centers': early_centers.tolist(),
        'early_width': early_width,
        'intm_centers': intm_centers.tolist(),
        'intm_width': intm_width,
        'late_centers': late_centers.tolist(),
        'late_width': late_width,
        'late_end': late_end
    }
    
    return early_basis, intm_basis, late_basis, feature_names, kernel_config


def compute_led_off_rebound(
    data: pd.DataFrame,
    tau: float = 2.0
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


def build_triphasic_design_matrix(
    data: pd.DataFrame,
    late_end: float = 10.0,
    include_led_main_effect: bool = False,  # Disabled by default
    include_rebound: bool = True,
    rebound_tau: float = 2.0
) -> Tuple[np.ndarray, List[str], int, Dict]:
    """
    Build full design matrix with triphasic kernel (early + intermediate + late).
    
    Returns
    -------
    X : ndarray
        Design matrix
    feature_names : list
        Feature names
    first_early_idx : int
        Index of FIRST early kernel coefficient (for non-negative constraint)
    kernel_config : dict
        Configuration used for kernel construction
    """
    n = len(data)
    features = []
    feature_names = []
    
    # Intercept
    features.append(np.ones(n))
    feature_names.append('intercept')
    
    # LED1 intensity (optional - disabled by default, kernel carries dynamics)
    if include_led_main_effect:
        if 'LED1_scaled' in data.columns:
            features.append(data['LED1_scaled'].values)
        elif 'led1Val' in data.columns:
            features.append(data['led1Val'].values / 250.0)
        else:
            features.append(np.zeros(n))
        feature_names.append('LED1_scaled')
    
    # Triphasic kernel (early + intermediate + late)
    if 'led1Val_ton' in data.columns:
        time_since = data['led1Val_ton'].fillna(999).values
        early_basis, intm_basis, late_basis, kernel_names, kernel_config = create_triphasic_kernel(
            time_since, late_end
        )
        
        # Track FIRST early kernel index (for non-negative constraint)
        first_early_idx = len(features)  # Index of first early basis
        
        # Add all basis functions
        for j in range(early_basis.shape[1]):
            features.append(early_basis[:, j])
        for j in range(intm_basis.shape[1]):
            features.append(intm_basis[:, j])
        for j in range(late_basis.shape[1]):
            features.append(late_basis[:, j])
        
        feature_names.extend(kernel_names)
    else:
        first_early_idx = -1
        kernel_config = {}
    
    # LED-off rebound
    if include_rebound and 'led1Val' in data.columns:
        rebound = compute_led_off_rebound(data, tau=rebound_tau)
        features.append(rebound)
        feature_names.append('led_off_rebound')
        kernel_config['rebound_tau'] = rebound_tau
    
    X = np.column_stack(features)
    return X, feature_names, first_early_idx, kernel_config


def refractory_penalty(
    lambda_series: np.ndarray,
    event_indices: np.ndarray,
    dt: float = 0.05,
    tau_refrac: float = 0.8
) -> float:
    """
    Physics-informed penalty for refractory period.
    
    Penalizes if post-event hazard doesn't decay exponentially.
    After an event, hazard should be suppressed and recover with time constant tau_refrac.
    
    Design based on empirical IEI statistics:
    - tau_refrac ~ 0.8s matches empirical mean IEI of 0.84s
    - Penalize only if predicted hazard is HIGHER than expected recovery curve
    
    Parameters
    ----------
    lambda_series : ndarray
        Predicted hazard at each frame
    event_indices : ndarray
        Indices of frames with events
    dt : float
        Time step (seconds)
    tau_refrac : float
        Refractory time constant (seconds)
    
    Returns
    -------
    penalty : float
        Mean squared deviation from expected recovery curve
    """
    max_steps = int(1.5 / dt)  # Look 1.5s ahead
    penalties = []
    baseline = np.mean(lambda_series)
    
    for idx in event_indices:
        for k in range(1, min(max_steps, len(lambda_series) - idx)):
            t_rel = k * dt
            lam_pred = lambda_series[idx + k]
            # Expected hazard: starts at 0, recovers exponentially to baseline
            lam_target = baseline * (1 - np.exp(-t_rel / tau_refrac))
            # Only penalize if predicted is TOO HIGH (not enough suppression)
            if lam_pred > lam_target:
                penalties.append((lam_pred - lam_target)**2)
    
    return np.mean(penalties) if penalties else 0.0


def fit_with_constraints(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    first_early_idx: int,
    alpha: float = 0.1,
    lambda_ridge: float = 0.01,
    lambda_smooth: float = 0.01,
    lambda_refrac: float = 0.02,
    tau_refrac: float = 0.8
) -> Dict:
    """
    Fit NB-GLM with constraints:
    - Only FIRST early basis (0.2s) is constrained non-negative
    - Intercept anchored to empirical baseline (-7.3 to -7.5)
    - Ridge + smoothness regularization on kernel coefficients
    - Physics-informed refractory penalty for IEI regularization
    
    Parameters
    ----------
    lambda_refrac : float
        Weight for refractory penalty (default 0.02)
    tau_refrac : float
        Refractory time constant in seconds (default 0.8, matches empirical IEI)
    """
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels not installed', 'converged': False}
    
    # Initialize parameters
    # FIX (2025-12-11): For paper/power analysis, FIX intercept near empirical baseline
    # Empirical baseline: ln(5.9e-4) = -7.44 (matches 0.71 events/min/track)
    # MLE intercept: -6.66 (best shape match but 1.8x rate)
    # Compromise: -7.0 balances rate (~1.1x) with reasonable shape
    FIXED_INTERCEPT = -7.2
    
    # Initialize kernel coefficients (not intercept - it's fixed)
    beta_kernel_init = np.zeros(X.shape[1] - 1)
    
    # Try to get better starting values from unconstrained fit
    try:
        model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
        fit_init = model.fit()
        # Use kernel params from unconstrained fit as starting point
        beta_kernel_init = fit_init.params.values[1:]
    except Exception as e:
        pass  # Keep zeros for kernel params
    
    # Get event indices for refractory penalty
    event_indices = np.where(y > 0)[0]
    
    # Define objective with FIXED intercept and refractory penalty
    def objective(beta_kernel):
        # Reconstruct full beta with fixed intercept
        beta_full = np.concatenate([[FIXED_INTERCEPT], beta_kernel])
        
        # Log-likelihood (NB)
        mu = np.exp(X @ beta_full)
        mu = np.clip(mu, 1e-10, 1e10)
        
        # NB log-likelihood
        r = 1 / alpha  # NB dispersion parameter
        ll = np.sum(
            y * np.log(mu / (mu + r)) + 
            r * np.log(r / (mu + r))
        )
        
        # Ridge penalty on kernel coefficients
        kernel_idx = [i for i, name in enumerate(feature_names[1:]) if 'kernel' in name]
        ridge_penalty = lambda_ridge * np.sum(beta_kernel[kernel_idx]**2)
        
        # Smoothness penalty (adjacent kernel coefficients)
        smooth_penalty = 0
        for i in range(len(kernel_idx) - 1):
            smooth_penalty += (beta_kernel[kernel_idx[i+1]] - beta_kernel[kernel_idx[i]])**2
        smooth_penalty *= lambda_smooth
        
        # Physics-informed refractory penalty
        refrac_pen = refractory_penalty(mu, event_indices, dt=0.05, tau_refrac=tau_refrac)
        
        return -(ll - ridge_penalty - smooth_penalty) + lambda_refrac * refrac_pen
    
    # Bounds for kernel coefficients only (intercept is fixed)
    # - Only FIRST early basis (0.2s) >= 0
    # - All other coefficients unconstrained
    bounds = [(None, None)] * len(beta_kernel_init)
    # Adjust first_early_idx since we're now optimizing without intercept
    if first_early_idx >= 1:  # It was index in full X, now offset by 1
        bounds[first_early_idx - 1] = (0, None)  # Only first early basis non-negative
    
    # Optimize (kernel coefficients only - intercept is fixed)
    result = minimize(
        objective,
        beta_kernel_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if result.success:
        beta_kernel = result.x
        # Reconstruct full beta with fixed intercept
        beta = np.concatenate([[FIXED_INTERCEPT], beta_kernel])
        
        # Compute approximate SEs using Hessian (for kernel params only)
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            n_kernel_params = len(beta_kernel)
            hess = np.zeros((n_kernel_params, n_kernel_params))
            for i in range(n_kernel_params):
                def grad_i(b):
                    b_plus = b.copy()
                    b_plus[i] += eps
                    b_minus = b.copy()
                    b_minus[i] -= eps
                    return (objective(b_plus) - objective(b_minus)) / (2 * eps)
                hess[i, :] = approx_fprime(beta_kernel, grad_i, eps)
            
            try:
                cov = np.linalg.inv(hess)
                se_kernel = np.sqrt(np.diag(cov))
            except:
                se_kernel = np.full(n_kernel_params, np.nan)
        except:
            se_kernel = np.full(len(beta_kernel), np.nan)
        
        # SE for intercept is 0 since it's fixed
        se = np.concatenate([[0.0], se_kernel])
        
        # Compute AIC
        mu = np.exp(X @ beta)
        r = 1 / alpha
        ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
        k = len(beta)  # Count all params including fixed intercept for AIC
        aic = 2 * k - 2 * ll
        
        # Compute pseudo-R2
        # Null model: intercept only
        y_mean = y.mean()
        mu_null = np.full(len(y), y_mean)
        ll_null = np.sum(y * np.log(mu_null / (mu_null + r)) + r * np.log(r / (mu_null + r)))
        pseudo_r2 = 1 - (ll / ll_null) if ll_null < 0 else 0.0
        
        return {
            'coefficients': dict(zip(feature_names, beta)),
            'std_errors': dict(zip(feature_names, se)),
            'aic': float(aic),
            'llf': float(ll),
            'llf_null': float(ll_null),
            'pseudo_r2': float(pseudo_r2),
            'converged': True,
            'n_obs': len(y),
            'n_events': int(y.sum()),
            'optimizer_success': result.success,
            'optimizer_message': result.message,
            'fixed_intercept': FIXED_INTERCEPT,
            'note': 'Intercept fixed at empirical baseline for paper/power analysis'
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
    print("=" * 70)
    print("FIT TRIPHASIC LNP MODEL")
    print("Early kernel: 0-1.5s at [0.2, 0.7, 1.4] - only first non-negative")
    print("Intermediate kernel: 1.5-3s at [2.0, 2.5] - unconstrained")
    print("Late kernel: 3-10s at [3, 5, 7, 9] - unconstrained")
    print("Intercept anchored to empirical baseline (~-7.3)")
    print("LED main effect REMOVED")
    print("=" * 70)
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    data, y = load_model_data(data_dir)
    print(f"  Loaded {len(data):,} observations")
    print(f"  Events: {y.sum():,}")
    
    # Calculate empirical rate for reference
    # BUG FIX: track_id is not unique across experiments, use (experiment_id, track_id) pairs
    n_tracks = data.groupby(['experiment_id', 'track_id']).ngroups
    duration_min = 20.0  # 20 minute experiments
    emp_rate = y.sum() / n_tracks / duration_min
    print(f"  Unique tracks: {n_tracks}")
    print(f"  Empirical rate: {emp_rate:.2f} events/min/track")
    
    # Build triphasic design matrix
    print(f"\n{'='*50}")
    print("Building triphasic kernel design matrix...")
    print(f"{'='*50}")
    
    X, feature_names, first_early_idx, kernel_config = build_triphasic_design_matrix(
        data,
        late_end=10.0,
        include_led_main_effect=False,  # Removed per recommendation
        include_rebound=True,
        rebound_tau=2.0
    )
    
    print(f"  Features: {feature_names}")
    print(f"  First early basis index (non-negative): {first_early_idx}")
    print(f"  Kernel config: {kernel_config}")
    
    # Fit model
    print(f"\n{'='*50}")
    print("Fitting NB-GLM with triphasic kernel...")
    print(f"{'='*50}")
    
    result = fit_with_constraints(
        X, y, feature_names, first_early_idx,
        alpha=0.1,
        lambda_ridge=0.01,
        lambda_smooth=0.01
    )
    
    if result.get('converged', False):
        print(f"\n  AIC: {result['aic']:.2f}")
        print(f"  Pseudo-R2: {result['pseudo_r2']:.4f}")
        print(f"\n  Coefficients:")
        for name, coef in result['coefficients'].items():
            se = result['std_errors'].get(name, np.nan)
            z = coef / se if not np.isnan(se) and se > 0 else 0
            sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.645 else ""
            constraint = "(non-neg)" if name == 'kernel_early_1' else ""
            print(f"    {name}: {coef:+.4f} (SE={se:.4f}) {sig} {constraint}")
        
        result['kernel_config'] = kernel_config
        
        # Save model
        output_path = Path('data/model/extended_biphasic_model_results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON
        save_result = {}
        for k, v in result.items():
            if isinstance(v, (np.floating, np.integer)):
                save_result[k] = float(v)
            elif isinstance(v, dict):
                save_result[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else 
                         (None if isinstance(vv, float) and np.isnan(vv) else vv))
                    for kk, vv in v.items()
                }
            else:
                save_result[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(save_result, f, indent=2)
        
        print(f"\nSaved to {output_path}")
        
        # Summary
        print(f"\n{'='*70}")
        print("MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"Intercept: {result['coefficients']['intercept']:.4f}")
        print(f"  -> Baseline hazard: {np.exp(result['coefficients']['intercept']):.6f} per frame")
        print(f"  -> Baseline rate: {np.exp(result['coefficients']['intercept']) * 20 * 60:.2f} events/min")
        
        # Check kernel structure
        print("\nKernel structure:")
        print("  Early (0.2s): should be positive (early bump)")
        print("  Early (0.7s, 1.4s) + Intermediate (2.0s, 2.5s): can be negative (suppression onset)")
        print("  Late (3s, 5s, 7s, 9s): negative (sustained suppression)")
    else:
        print(f"  Failed: {result.get('error', 'Unknown')}")


if __name__ == '__main__':
    main()




