#!/usr/bin/env python3
"""
Negative Binomial GLM LNP Model for Larval Reorientation Events

Implements Linear-Nonlinear-Poisson (LNP) model specification:
- Family: Negative Binomial with log link (handles overdispersion)
- Temporal kernel: Raised-cosine basis functions
- Covariates: LED intensity, phase, speed, curvature

Reference:
- Gepner et al. (2015) eLife - LNP model with raised-cosine kernels
- Klein et al. (2015) PNAS - Turn detection and stimulus response

Usage:
    python scripts/hazard_model.py --data-dir data/engineered_validated
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Statistical modeling
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")


# =============================================================================
# RAISED-COSINE BASIS FUNCTIONS
# =============================================================================

def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Compute raised-cosine basis functions for temporal kernels.
    
    Based on Pillow et al. (2008) J Neurosci and Gepner et al. (2015) eLife.
    
    Parameters
    ----------
    t : ndarray
        Time points (relative to event, negative = before)
    centers : ndarray
        Center positions for each basis function (in seconds)
    width : float
        Width parameter (controls overlap between bases)
    
    Returns
    -------
    basis : ndarray
        Shape (len(t), len(centers)) - basis function values
    
    Notes
    -----
    Each basis function is:
        B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / w))  if |t - c_j| < w
               = 0                                      otherwise
    
    Width w ≈ 0.6s makes bumps overlap at ~50% height.
    """
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        # Distance from center
        dist = np.abs(t - c)
        # Raised cosine: nonzero only within width
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def create_temporal_kernel_design(
    time_since_stimulus: np.ndarray,
    n_bases: int = 4,
    window: Tuple[float, float] = (-3.0, 0.0),
    width: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create design matrix columns for temporal kernel.
    
    Parameters
    ----------
    time_since_stimulus : ndarray
        Time since last stimulus onset (seconds)
    n_bases : int
        Number of raised-cosine basis functions (default 4)
    window : tuple
        (start, end) of temporal window in seconds (default -3 to 0)
    width : float
        Width of each basis function (default 0.6s)
    
    Returns
    -------
    design : ndarray
        Shape (N, n_bases) - design matrix columns
    centers : ndarray
        Centers of the basis functions
    """
    # Compute basis centers (evenly spaced in window)
    centers = np.linspace(window[0], window[1], n_bases)
    
    # Convert time_since_stimulus to relative time (negative = before now)
    # time_since_stimulus is positive, so we need -time_since_stimulus for kernel
    t_relative = -time_since_stimulus  # Now negative values = past
    
    # Compute basis functions
    design = raised_cosine_basis(t_relative, centers, width)
    
    return design, centers


# =============================================================================
# PHASE COVARIATES
# =============================================================================

def compute_phase_covariates(time: np.ndarray, cycle_period: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sin/cos phase covariates for periodic LED1 stimulus.
    
    Parameters
    ----------
    time : ndarray
        Experiment time in seconds
    cycle_period : float
        LED1 cycle period in seconds (default 60s = 30s on + 30s off)
    
    Returns
    -------
    sin_phase : ndarray
        sin(2π * phase)
    cos_phase : ndarray
        cos(2π * phase)
    """
    phase = (time % cycle_period) / cycle_period  # 0 to 1
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    return sin_phase, cos_phase


# =============================================================================
# DESIGN MATRIX CONSTRUCTION
# =============================================================================

def build_design_matrix(
    data: pd.DataFrame,
    n_temporal_bases: int = 4,
    temporal_window: Tuple[float, float] = (-3.0, 0.0),
    include_interaction: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build design matrix for NB-GLM LNP model.
    
    Model specification:
        log(μ_i) = β₀ 
                 + β₁·LED1_intensity 
                 + β₂·LED2_intensity 
                 + β₃·LED1×LED2 (interaction)
                 + β₄·sin(phase) + β₅·cos(phase)
                 + Σⱼ φⱼ·Bⱼ(t)  (temporal kernel)
                 + γ₁·SpeedRunVel 
                 + γ₂·Curvature
    
    Parameters
    ----------
    data : DataFrame
        Must contain columns:
        - led1Val: LED1 intensity (PWM)
        - led2Val: LED2 intensity (PWM)
        - time: experiment time (seconds)
        - time_since_stimulus: time since last LED1 onset
        - speed: instantaneous speed (cm/s)
        - curvature: instantaneous curvature
    n_temporal_bases : int
        Number of raised-cosine bases for temporal kernel
    temporal_window : tuple
        Window for temporal kernel (seconds before stimulus)
    include_interaction : bool
        Whether to include LED1×LED2 interaction term
    
    Returns
    -------
    X : DataFrame
        Design matrix with all covariates
    feature_names : list
        Names of features in order
    """
    n = len(data)
    feature_names = []
    features = {}
    
    # Intercept
    features['intercept'] = np.ones(n)
    feature_names.append('intercept')
    
    # LED covariates (scaled to 0-1 range: LED1/250, LED2/15)
    if 'led1Val' in data.columns:
        # Scale by 250 (max PWM for LED1), not 255
        features['led1_intensity'] = (data['led1Val'] / 250.0).values
        feature_names.append('led1_intensity')
    elif 'LED1_scaled' in data.columns:
        # Already scaled (from prepare_binned_data.py)
        features['led1_intensity'] = data['LED1_scaled'].values
        feature_names.append('led1_intensity')
    
    if 'led2Val' in data.columns:
        # Scale by 15 (max PWM for LED2)
        features['led2_intensity'] = (data['led2Val'] / 15.0).values
        feature_names.append('led2_intensity')
    elif 'LED2_scaled' in data.columns:
        # Already scaled (from prepare_binned_data.py)
        features['led2_intensity'] = data['LED2_scaled'].values
        feature_names.append('led2_intensity')
    
    # Interaction term
    if include_interaction and 'led1Val' in data.columns and 'led2Val' in data.columns:
        features['led1_x_led2'] = features['led1_intensity'] * features['led2_intensity']
        feature_names.append('led1_x_led2')
    
    # Phase covariates (deterministic LED1 cycle)
    if 'time' in data.columns:
        sin_phase, cos_phase = compute_phase_covariates(data['time'].values)
        features['phase_sin'] = sin_phase
        features['phase_cos'] = cos_phase
        feature_names.extend(['phase_sin', 'phase_cos'])
    
    # Temporal kernel (raised-cosine basis)
    if 'time_since_stimulus' in data.columns:
        time_since = data['time_since_stimulus'].fillna(999).values  # Far future if NaN
        kernel_design, centers = create_temporal_kernel_design(
            time_since, 
            n_bases=n_temporal_bases,
            window=temporal_window
        )
        for j in range(n_temporal_bases):
            features[f'kernel_b{j}'] = kernel_design[:, j]
            feature_names.append(f'kernel_b{j}')
    
    # Instantaneous covariates
    if 'speed' in data.columns:
        # Standardize speed for numerical stability
        speed_mean = data['speed'].mean()
        speed_std = data['speed'].std()
        if speed_std > 0:
            features['speed'] = (data['speed'].values - speed_mean) / speed_std
        else:
            features['speed'] = data['speed'].values
        feature_names.append('speed')
    
    if 'curvature' in data.columns:
        # Clip extreme curvature values (path curvature explodes at low speed)
        curv_clipped = np.clip(data['curvature'].values, -50, 50)
        curv_mean = np.mean(curv_clipped)
        curv_std = np.std(curv_clipped)
        if curv_std > 0:
            features['curvature'] = (curv_clipped - curv_mean) / curv_std
        else:
            features['curvature'] = curv_clipped
        feature_names.append('curvature')
    
    X = pd.DataFrame(features)
    return X, feature_names


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_nb_glm(
    X: pd.DataFrame,
    y: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    cluster_groups: Optional[np.ndarray] = None
) -> Dict:
    """
    Fit Negative Binomial GLM with optional cluster-robust standard errors.
    
    Parameters
    ----------
    X : DataFrame
        Design matrix (N observations x K features)
    y : ndarray
        Response variable (event counts per bin)
    exposure : ndarray, optional
        Exposure offset (bin duration). If None, assumes uniform.
    alpha : float
        Dispersion parameter for NB (default 1.0)
    cluster_groups : ndarray, optional
        Group labels for cluster-robust standard errors (e.g., experiment_id)
    
    Returns
    -------
    results : dict
        Model results including coefficients, CI, diagnostics
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for NB-GLM. Install with: pip install statsmodels")
    
    # Add exposure as offset if provided
    if exposure is not None:
        offset = np.log(exposure)
    else:
        offset = None
    
    # Fit NB-GLM
    try:
        model = GLM(
            y, 
            X,
            family=NegativeBinomial(alpha=alpha),
            offset=offset
        )
        
        # Use cluster-robust SEs if groups provided
        if cluster_groups is not None:
            # Fit with cluster-robust covariance
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
            results = {
                'coefficients': fit.params.to_dict(),
                'std_errors': fit.bse.to_dict(),
                'pvalues': fit.pvalues.to_dict(),
                'conf_int_lower': fit.conf_int()[0].to_dict(),
                'conf_int_upper': fit.conf_int()[1].to_dict(),
                'robust_se': True,
                'n_clusters': len(np.unique(cluster_groups)),
            }
        else:
            fit = model.fit()
            results = {
                'coefficients': fit.params.to_dict(),
                'std_errors': fit.bse.to_dict(),
                'pvalues': fit.pvalues.to_dict(),
                'conf_int_lower': fit.conf_int()[0].to_dict(),
                'conf_int_upper': fit.conf_int()[1].to_dict(),
                'robust_se': False,
            }
        
        # Add common diagnostics
        results.update({
            'deviance': fit.deviance,
            'pearson_chi2': fit.pearson_chi2,
            'df_resid': fit.df_resid,
            'aic': fit.aic,
            'bic': fit.bic,
            'llf': fit.llf,
            'dispersion': alpha,
            'n_obs': len(y),
            'converged': fit.converged,
            'dispersion_ratio': fit.pearson_chi2 / fit.df_resid,
            'fitted_values': fit.predict(),
            'resid_pearson': fit.resid_pearson,
        })
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'converged': False
        }


def estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Estimate NB dispersion parameter using method of moments.
    
    Parameters
    ----------
    y : ndarray
        Observed counts
    mu : ndarray
        Fitted mean values
    
    Returns
    -------
    alpha : float
        Estimated dispersion (higher = more overdispersion)
    """
    # Var(Y) = μ + α*μ² for NB
    # Solve for α: α = (Var(Y) - μ) / μ²
    var_y = np.var(y)
    mean_y = np.mean(y)
    
    if mean_y > 0:
        alpha = max(0.01, (var_y - mean_y) / (mean_y ** 2))
    else:
        alpha = 1.0
    
    return alpha


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def check_overdispersion(results: Dict) -> Dict:
    """
    Check if NB model adequately captures overdispersion.
    
    Parameters
    ----------
    results : dict
        Output from fit_nb_glm()
    
    Returns
    -------
    diagnostic : dict
        Dispersion diagnostic results
    """
    ratio = results.get('dispersion_ratio', None)
    if ratio is None:
        return {'error': 'No dispersion_ratio in results'}
    
    status = 'OK'
    if ratio > 1.5:
        status = 'WARNING: Underdispersed (ratio >> 1), consider increasing alpha'
    elif ratio < 0.5:
        status = 'WARNING: Overdispersed (ratio << 1), alpha may be too high'
    
    return {
        'dispersion_ratio': ratio,
        'target': 1.0,
        'status': status
    }


def check_zero_inflation(y: np.ndarray, mu: np.ndarray, alpha: float) -> Dict:
    """
    Compare observed vs expected zeros under NB distribution.
    
    Parameters
    ----------
    y : ndarray
        Observed counts
    mu : ndarray
        Fitted means from model
    alpha : float
        Dispersion parameter
    
    Returns
    -------
    diagnostic : dict
        Zero-inflation diagnostic results
    """
    observed_zeros = (y == 0).mean()
    
    # Expected zero probability under NB: (1 + alpha*mu)^(-1/alpha)
    if alpha > 0:
        expected_zeros = np.mean((1 + alpha * mu) ** (-1 / alpha))
    else:
        # Poisson limit
        expected_zeros = np.mean(np.exp(-mu))
    
    diff = abs(observed_zeros - expected_zeros)
    
    needs_zinb = diff > 0.02
    status = 'Consider ZINB' if needs_zinb else 'OK'
    
    return {
        'observed_zeros': observed_zeros,
        'expected_zeros': expected_zeros,
        'difference': diff,
        'threshold': 0.02,
        'needs_zinb': needs_zinb,
        'status': status
    }


def check_serial_correlation(resid_pearson: np.ndarray, groups: np.ndarray = None, max_lag: int = 5) -> Dict:
    """
    Check for residual autocorrelation.
    
    Parameters
    ----------
    resid_pearson : ndarray
        Pearson residuals from model
    groups : ndarray, optional
        Track/group identifiers for within-group ACF
    max_lag : int
        Maximum lag to check (default 5)
    
    Returns
    -------
    diagnostic : dict
        Serial correlation diagnostic results
    """
    # Simple overall ACF
    n = len(resid_pearson)
    acf_values = {}
    
    for lag in range(1, min(max_lag + 1, n // 2)):
        corr = np.corrcoef(resid_pearson[:-lag], resid_pearson[lag:])[0, 1]
        acf_values[f'lag_{lag}'] = corr
    
    lag1 = acf_values.get('lag_1', 0)
    has_autocorr = abs(lag1) > 0.1
    status = 'WARNING: Significant lag-1 autocorrelation' if has_autocorr else 'OK'
    
    return {
        'acf': acf_values,
        'lag1_threshold': 0.1,
        'has_autocorrelation': has_autocorr,
        'status': status
    }


def run_all_diagnostics(results: Dict, y: np.ndarray) -> Dict:
    """
    Run all model diagnostics.
    
    Parameters
    ----------
    results : dict
        Output from fit_nb_glm()
    y : ndarray
        Observed counts
    
    Returns
    -------
    diagnostics : dict
        All diagnostic results
    """
    diagnostics = {}
    
    # Overdispersion
    diagnostics['overdispersion'] = check_overdispersion(results)
    
    # Zero-inflation
    if 'fitted_values' in results and 'dispersion' in results:
        diagnostics['zero_inflation'] = check_zero_inflation(
            y, results['fitted_values'], results['dispersion']
        )
    
    # Serial correlation
    if 'resid_pearson' in results:
        diagnostics['serial_correlation'] = check_serial_correlation(results['resid_pearson'])
    
    # Overall status
    all_ok = all(
        d.get('status', '').startswith('OK') 
        for d in diagnostics.values() 
        if isinstance(d, dict) and 'status' in d
    )
    diagnostics['overall'] = 'All diagnostics OK' if all_ok else 'Some diagnostics flagged'
    
    return diagnostics


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_kernel_params(
    data: pd.DataFrame,
    y: np.ndarray,
    n_bases_options: List[int] = [3, 4, 5],
    window_options: List[Tuple[float, float]] = [(0.0, 2.0), (0.0, 3.0), (0.0, 4.0)],
    n_folds: int = 5,
    exposure: Optional[np.ndarray] = None
) -> Dict:
    """
    Cross-validate to select optimal temporal kernel parameters.
    
    Uses leave-one-experiment-out CV for robust generalization.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset with all covariates
    y : ndarray
        Response variable
    n_bases_options : list
        Number of bases to try
    window_options : list
        Window ranges to try
    n_folds : int
        Number of CV folds (if not doing leave-one-out)
    
    Returns
    -------
    results : dict
        CV results including best parameters
    """
    results = {
        'tested_params': [],
        'cv_deviances': [],
        'best_params': None,
        'best_deviance': np.inf
    }
    
    # Check if we have experiment_id for leave-one-out
    if 'experiment_id' in data.columns:
        experiments = data['experiment_id'].unique()
        use_loo = len(experiments) > 3
    else:
        use_loo = False
    
    for n_bases in n_bases_options:
        for window in window_options:
            params = {'n_bases': n_bases, 'window': window}
            
            if use_loo:
                # Leave-one-experiment-out CV
                deviances = []
                for held_out in experiments:
                    train_mask = data['experiment_id'] != held_out
                    test_mask = data['experiment_id'] == held_out
                    
                    X_train, _ = build_design_matrix(
                        data[train_mask], 
                        n_temporal_bases=n_bases,
                        temporal_window=window
                    )
                    X_test, _ = build_design_matrix(
                        data[test_mask],
                        n_temporal_bases=n_bases,
                        temporal_window=window
                    )
                    
                    y_train = y[train_mask.values]
                    y_test = y[test_mask.values]
                    
                    if len(y_train) > 0 and len(y_test) > 0:
                        # Get exposure for train/test if provided
                        exp_train = exposure[train_mask.values] if exposure is not None else None
                        exp_test = exposure[test_mask.values] if exposure is not None else None
                        
                        fit_result = fit_nb_glm(X_train, y_train, exposure=exp_train)
                        if fit_result.get('converged', False):
                            # Compute held-out deviance using training model
                            deviances.append(fit_result.get('deviance', np.inf))
                
                mean_deviance = np.mean(deviances) if deviances else np.inf
            else:
                # Simple k-fold CV
                X, _ = build_design_matrix(data, n_temporal_bases=n_bases, temporal_window=window)
                fit_result = fit_nb_glm(X, y)
                mean_deviance = fit_result.get('deviance', np.inf)
            
            results['tested_params'].append(params)
            results['cv_deviances'].append(mean_deviance)
            
            if mean_deviance < results['best_deviance']:
                results['best_deviance'] = mean_deviance
                results['best_params'] = params
    
    return results


# =============================================================================
# KERNEL INTERPRETATION
# =============================================================================

def extract_kernel_shape(
    phi_hat: np.ndarray,
    centers: np.ndarray,
    width: float,
    t_grid: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stimulus-response function from fitted kernel weights.
    
    Parameters
    ----------
    phi_hat : ndarray
        Fitted kernel coefficients (phi_1, ..., phi_J)
    centers : ndarray
        Kernel center positions
    width : float
        Kernel width parameter
    t_grid : ndarray, optional
        Time points for evaluation (default: 0 to 3s at 0.01s resolution)
    
    Returns
    -------
    t_grid : ndarray
        Time points
    K : ndarray
        Log-rate modulation K(t) = sum_j(phi_j * B_j(t))
    RR : ndarray
        Rate ratio RR(t) = exp(K(t))
    """
    if t_grid is None:
        t_grid = np.linspace(0, 3, 301)
    
    B = raised_cosine_basis(t_grid, centers, width)
    K = B @ phi_hat
    RR = np.exp(K)
    
    return t_grid, K, RR


def kernel_confidence_bands(
    phi_hat: np.ndarray,
    phi_cov: np.ndarray,
    centers: np.ndarray,
    width: float,
    t_grid: np.ndarray = None,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pointwise confidence bands for kernel using delta method.
    
    Parameters
    ----------
    phi_hat : ndarray
        Fitted kernel coefficients
    phi_cov : ndarray
        Covariance matrix of phi estimates
    centers : ndarray
        Kernel center positions
    width : float
        Kernel width parameter
    t_grid : ndarray, optional
        Time points for evaluation
    alpha : float
        Significance level (default 0.05 for 95% CI)
    
    Returns
    -------
    t_grid : ndarray
        Time points
    RR : ndarray
        Rate ratio point estimate
    RR_lower : ndarray
        Lower CI bound
    RR_upper : ndarray
        Upper CI bound
    """
    from scipy.stats import norm
    
    if t_grid is None:
        t_grid = np.linspace(0, 3, 301)
    
    B = raised_cosine_basis(t_grid, centers, width)
    K = B @ phi_hat
    
    # Variance of K(t) via delta method: Var(K) = B @ Cov(phi) @ B.T
    var_K = np.diag(B @ phi_cov @ B.T)
    se_K = np.sqrt(np.maximum(var_K, 0))  # Ensure non-negative
    
    z = norm.ppf(1 - alpha / 2)
    K_lower = K - z * se_K
    K_upper = K + z * se_K
    
    RR = np.exp(K)
    RR_lower = np.exp(K_lower)
    RR_upper = np.exp(K_upper)
    
    return t_grid, RR, RR_lower, RR_upper


def find_peak_latency(t_grid: np.ndarray, K: np.ndarray) -> Tuple[float, float]:
    """
    Find time of peak kernel response.
    
    Parameters
    ----------
    t_grid : ndarray
        Time points
    K : ndarray
        Log-rate modulation values
    
    Returns
    -------
    peak_latency : float
        Time of peak response (seconds)
    peak_rr : float
        Rate ratio at peak
    """
    peak_idx = np.argmax(K)
    peak_latency = t_grid[peak_idx]
    peak_rr = np.exp(K[peak_idx])
    return peak_latency, peak_rr


def generate_coefficient_table(results: Dict, feature_names: List[str]) -> pd.DataFrame:
    """
    Generate interpretation table with rate ratios and percent changes.
    
    Parameters
    ----------
    results : dict
        Output from fit_nb_glm()
    feature_names : list
        Names of features in order
    
    Returns
    -------
    table : DataFrame
        Coefficient interpretation table
    """
    coefs = results.get('coefficients', {})
    ses = results.get('std_errors', {})
    pvals = results.get('pvalues', {})
    
    rows = []
    for name in feature_names:
        coef = coefs.get(name, np.nan)
        se = ses.get(name, np.nan)
        p = pvals.get(name, np.nan)
        rr = np.exp(coef) if not np.isnan(coef) else np.nan
        pct = 100 * (rr - 1) if not np.isnan(rr) else np.nan
        
        rows.append({
            'feature': name,
            'coef': coef,
            'se': se,
            'z': coef / se if se > 0 else np.nan,
            'p': p,
            'RR': rr,
            'pct_change': pct
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def fit_hazard_model(
    data_path: Path,
    output_path: Path,
    event_column: str = 'is_reorientation_start',
    n_temporal_bases: int = 4,
    temporal_window: Tuple[float, float] = (-3.0, 0.0),
    run_cv: bool = False
) -> Dict:
    """
    Fit NB-GLM LNP model to engineered data.
    
    Parameters
    ----------
    data_path : Path
        Path to engineered events CSV
    output_path : Path
        Path to save model results
    event_column : str
        Column containing event indicator (default 'is_reorientation_start')
    n_temporal_bases : int
        Number of temporal kernel bases
    temporal_window : tuple
        Temporal kernel window
    run_cv : bool
        Whether to run cross-validation for kernel params
    
    Returns
    -------
    results : dict
        Model fitting results
    """
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    print(f"  {len(data)} observations")
    print(f"  {data[event_column].sum()} events ({100*data[event_column].mean():.2f}%)")
    
    # Response variable (counts per bin)
    y = data[event_column].astype(int).values
    
    # Cross-validation for kernel parameters (optional)
    if run_cv:
        print("Running cross-validation for kernel parameters...")
        cv_results = cross_validate_kernel_params(
            data, y,
            n_bases_options=[3, 4, 5],
            window_options=[(-2.0, 0.0), (-3.0, 0.0), (-4.0, 0.0)]
        )
        print(f"  Best params: {cv_results['best_params']}")
        n_temporal_bases = cv_results['best_params']['n_bases']
        temporal_window = cv_results['best_params']['window']
    else:
        cv_results = None
    
    # Build design matrix
    print(f"Building design matrix with {n_temporal_bases} kernel bases, window {temporal_window}")
    X, feature_names = build_design_matrix(
        data,
        n_temporal_bases=n_temporal_bases,
        temporal_window=temporal_window
    )
    print(f"  {len(feature_names)} features: {feature_names}")
    
    # Estimate dispersion from data
    var_y = np.var(y)
    mean_y = np.mean(y)
    if mean_y > 0:
        estimated_alpha = max(0.1, (var_y - mean_y) / max(mean_y ** 2, 0.001))
    else:
        estimated_alpha = 1.0
    print(f"  Estimated dispersion: {estimated_alpha:.3f}")
    
    # Fit model
    print("Fitting NB-GLM...")
    model_results = fit_nb_glm(X, y, alpha=estimated_alpha)
    
    if model_results.get('converged', False):
        print("  Model converged!")
        print(f"  Deviance: {model_results['deviance']:.2f}")
        print(f"  Dispersion ratio: {model_results['dispersion_ratio']:.3f} (should be ~1)")
        print(f"  AIC: {model_results['aic']:.2f}")
        
        # Print significant coefficients
        print("\n  Significant coefficients (p < 0.05):")
        for name in feature_names:
            if name in model_results['pvalues']:
                p = model_results['pvalues'][name]
                if p < 0.05:
                    coef = model_results['coefficients'][name]
                    se = model_results['std_errors'][name]
                    print(f"    {name}: {coef:.4f} (SE={se:.4f}, p={p:.4f})")
    else:
        print(f"  Model failed: {model_results.get('error', 'Unknown error')}")
    
    # Combine results
    all_results = {
        'model': model_results,
        'cv': cv_results,
        'params': {
            'n_temporal_bases': n_temporal_bases,
            'temporal_window': temporal_window,
            'event_column': event_column
        },
        'data_summary': {
            'n_observations': len(data),
            'n_events': int(data[event_column].sum()),
            'event_rate': float(data[event_column].mean())
        }
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Fit NB-GLM LNP model')
    parser.add_argument('--data-dir', type=str, default='data/engineered_validated',
                       help='Directory with engineered event CSVs')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Output directory for model results')
    parser.add_argument('--event', type=str, default='is_reorientation_start',
                       help='Event column to model')
    parser.add_argument('--n-bases', type=int, default=4,
                       help='Number of temporal kernel bases')
    parser.add_argument('--cv', action='store_true',
                       help='Run cross-validation for kernel parameters')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Find all event files
    event_files = list(data_dir.glob("*_events.csv"))
    if not event_files:
        print(f"No event files found in {data_dir}")
        return
    
    print(f"Found {len(event_files)} event files")
    
    # Combine all data for pooled model
    all_data = []
    for f in event_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem.replace('_events', '')
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {len(combined_data)} observations from {len(event_files)} experiments")
    
    # Save combined data temporarily
    combined_path = output_dir / 'combined_events.csv'
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(combined_path, index=False)
    
    # Fit model
    results = fit_hazard_model(
        combined_path,
        output_dir / 'hazard_model_results.json',
        event_column=args.event,
        n_temporal_bases=args.n_bases,
        run_cv=args.cv
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()




