#!/usr/bin/env python3
"""
Mixed-Effects NB-GLM with Random Intercepts

Uses glmmTMB via rpy2 to fit a hierarchical model with:
- Fixed effects: temporal kernel bases
- Random intercepts: per track

This allows individual larvae to have different baseline rates,
which may help resolve the rate-vs-shape trade-off.

Usage:
    python scripts/fit_mixed_effects_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

# Check for rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("Warning: rpy2 not available. Install with: pip install rpy2")

# Parameters
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


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


def build_design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Build design matrix with temporal kernel bases."""
    times = data['time'].values
    n = len(times)
    
    # Compute time since LED onset for each frame
    time_since_onset = np.zeros(n)
    is_led_on = np.zeros(n, dtype=bool)
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            time_since_onset[i] = -1
            is_led_on[i] = False
        else:
            time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
            if time_in_cycle < LED_ON_DURATION:
                time_since_onset[i] = time_in_cycle
                is_led_on[i] = True
            else:
                time_since_onset[i] = -1
                is_led_on[i] = False
    
    # Kernel configuration
    early_centers = np.array([0.2, 0.7, 1.4])
    intm_centers = np.array([2.0, 2.5])
    late_centers = np.array([3.0, 5.0, 7.0, 9.0])
    
    # Compute basis functions
    early_basis = np.zeros((n, 3))
    intm_basis = np.zeros((n, 2))
    late_basis = np.zeros((n, 4))
    
    on_mask = is_led_on
    tso = time_since_onset[on_mask]
    
    early_basis[on_mask, :] = raised_cosine_basis(tso, early_centers, 0.4)
    intm_basis[on_mask, :] = raised_cosine_basis(tso, intm_centers, 0.6)
    late_basis[on_mask, :] = raised_cosine_basis(tso, late_centers, 1.8)
    
    # LED-off rebound
    rebound = np.zeros(n)
    off_mask = ~is_led_on & (times >= FIRST_LED_ONSET)
    for i in np.where(off_mask)[0]:
        t = times[i]
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        time_since_offset = time_in_cycle - LED_ON_DURATION
        if time_since_offset > 0:
            rebound[i] = np.exp(-time_since_offset / 2.0)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'time': times,
        'track_id': data['track_id'] if 'track_id' in data.columns else 0,
        'experiment_id': data['experiment_id'] if 'experiment_id' in data.columns else 'exp1',
        'is_event': data['is_reorientation_start'] if 'is_reorientation_start' in data.columns 
                   else data.get('is_reorientation', 0),
        'kernel_early_1': early_basis[:, 0],
        'kernel_early_2': early_basis[:, 1],
        'kernel_early_3': early_basis[:, 2],
        'kernel_intm_1': intm_basis[:, 0],
        'kernel_intm_2': intm_basis[:, 1],
        'kernel_late_1': late_basis[:, 0],
        'kernel_late_2': late_basis[:, 1],
        'kernel_late_3': late_basis[:, 2],
        'kernel_late_4': late_basis[:, 3],
        'led_off_rebound': rebound
    })
    
    # Create unique track identifier
    result['track_uid'] = result['experiment_id'].astype(str) + '_' + result['track_id'].astype(str)
    
    return result


def fit_glmm_via_r(data: pd.DataFrame) -> dict:
    """Fit mixed-effects NB-GLM using glmmTMB in R."""
    if not HAS_RPY2:
        return {'error': 'rpy2 not available'}
    
    # Import R packages
    try:
        glmmTMB = importr('glmmTMB')
        base = importr('base')
        stats = importr('stats')
    except Exception as e:
        return {'error': f'R package not found: {e}. Install with: install.packages("glmmTMB")'}
    
    # Convert to R dataframe using context manager
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
    ro.globalenv['data'] = r_data
    
    # Fit model with random intercept per track
    formula = """
    is_event ~ kernel_early_1 + kernel_early_2 + kernel_early_3 +
               kernel_intm_1 + kernel_intm_2 +
               kernel_late_1 + kernel_late_2 + kernel_late_3 + kernel_late_4 +
               led_off_rebound + (1|track_uid)
    """
    
    print("Fitting glmmTMB model (this may take a few minutes)...")
    
    try:
        ro.r(f'''
        library(glmmTMB)
        model <- glmmTMB({formula}, 
                        data = data, 
                        family = nbinom2,
                        control = glmmTMBControl(optCtrl = list(iter.max = 1000)))
        ''')
        
        # Extract results
        ro.r('''
        coefs <- fixef(model)$cond
        se <- sqrt(diag(vcov(model)$cond))
        aic <- AIC(model)
        ranef_var <- VarCorr(model)$cond$track_uid[1,1]
        ''')
        
        coefs = dict(zip(ro.r('names(coefs)'), ro.r('coefs')))
        se = dict(zip(ro.r('names(se)'), ro.r('se')))
        aic = ro.r('aic')[0]
        ranef_var = ro.r('ranef_var')[0]
        
        return {
            'coefficients': coefs,
            'std_errors': se,
            'aic': float(aic),
            'random_intercept_variance': float(ranef_var),
            'random_intercept_sd': float(np.sqrt(ranef_var)),
            'converged': True
        }
        
    except Exception as e:
        return {'error': str(e), 'converged': False}


def fit_approximation_with_offset(data: pd.DataFrame) -> dict:
    """
    Approximate mixed-effects by fitting separate intercepts per track,
    then pooling. This is a poor-man's random effects.
    """
    from scipy.optimize import minimize
    from scipy.special import gammaln
    
    track_uids = data['track_uid'].unique()
    n_tracks = len(track_uids)
    
    print(f"Fitting approximate mixed model with {n_tracks} tracks...")
    
    # First pass: estimate per-track intercepts
    track_intercepts = {}
    
    for uid in track_uids:
        track_data = data[data['track_uid'] == uid]
        n_events = track_data['is_event'].sum()
        n_frames = len(track_data)
        
        if n_frames > 0:
            rate = n_events / n_frames
            track_intercepts[uid] = np.log(max(rate, 1e-10))
        else:
            track_intercepts[uid] = -10
    
    # Compute statistics of intercept distribution
    intercepts = np.array(list(track_intercepts.values()))
    mean_intercept = np.mean(intercepts)
    std_intercept = np.std(intercepts)
    
    print(f"  Per-track intercept distribution:")
    print(f"    Mean: {mean_intercept:.3f}")
    print(f"    Std:  {std_intercept:.3f}")
    print(f"    Range: [{intercepts.min():.3f}, {intercepts.max():.3f}]")
    
    # Second pass: fit kernel with mean intercept fixed
    feature_cols = [c for c in data.columns if c.startswith('kernel') or c == 'led_off_rebound']
    X = data[feature_cols].values
    y = data['is_event'].values.astype(float)
    
    alpha = 0.1  # NB dispersion
    
    def objective(beta_kernel):
        eta = mean_intercept + X @ beta_kernel
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        
        r = 1 / alpha
        ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
        
        # Ridge penalty
        ridge = 0.01 * np.sum(beta_kernel**2)
        
        return -(ll - ridge)
    
    beta_init = np.zeros(len(feature_cols))
    bounds = [(None, None)] * len(feature_cols)
    bounds[0] = (0, None)  # First early basis non-negative
    
    result = minimize(objective, beta_init, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        beta_kernel = result.x
        
        return {
            'coefficients': {
                'intercept': float(mean_intercept),
                **dict(zip(feature_cols, beta_kernel))
            },
            'random_intercept_mean': float(mean_intercept),
            'random_intercept_std': float(std_intercept),
            'converged': True,
            'method': 'approximate_mixed_effects'
        }
    else:
        return {'error': result.message, 'converged': False}


def main():
    print("=" * 70)
    print("MIXED-EFFECTS NB-GLM")
    print("Random intercept per track to capture individual variability")
    print("=" * 70)
    
    # Load data
    data_dir = Path('data/engineered')
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        print("No data files found")
        return
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem.split('_')[0]
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(data):,} observations from {len(csv_files)} experiments")
    
    # Build design matrix
    print("\nBuilding design matrix...")
    design_data = build_design_matrix(data)
    
    n_tracks = design_data['track_uid'].nunique()
    n_events = design_data['is_event'].sum()
    print(f"  Unique tracks: {n_tracks}")
    print(f"  Events: {n_events}")
    
    # Try R-based glmmTMB first
    if HAS_RPY2:
        print("\n" + "=" * 50)
        print("Attempting glmmTMB fit...")
        print("=" * 50)
        results = fit_glmm_via_r(design_data)
        
        if results.get('converged'):
            print("\nglmmTMB Results:")
            print(f"  AIC: {results['aic']:.2f}")
            print(f"  Random intercept SD: {results['random_intercept_sd']:.3f}")
            print("\n  Fixed Effects:")
            for name, coef in results['coefficients'].items():
                se = results['std_errors'].get(name, np.nan)
                sig = '***' if abs(coef/se) > 3.3 else ('**' if abs(coef/se) > 2.6 else ('*' if abs(coef/se) > 2.0 else ''))
                print(f"    {name}: {coef:+.4f} (SE={se:.4f}) {sig}")
        else:
            print(f"  glmmTMB failed: {results.get('error')}")
    
    # Always try approximate method as backup
    print("\n" + "=" * 50)
    print("Fitting approximate mixed-effects model...")
    print("=" * 50)
    approx_results = fit_approximation_with_offset(design_data)
    
    if approx_results.get('converged'):
        print("\nApproximate Mixed-Effects Results:")
        print(f"  Mean intercept: {approx_results['random_intercept_mean']:.4f}")
        print(f"  Intercept SD (between tracks): {approx_results['random_intercept_std']:.4f}")
        
        print("\n  Kernel coefficients:")
        for name, coef in approx_results['coefficients'].items():
            if name != 'intercept':
                print(f"    {name}: {coef:+.4f}")
        
        # Save results
        output_path = Path('data/model/mixed_effects_results.json')
        with open(output_path, 'w') as f:
            json.dump(approx_results, f, indent=2)
        print(f"\nSaved to {output_path}")
        
        # Key insight
        print("\n" + "=" * 50)
        print("KEY INSIGHT")
        print("=" * 50)
        baseline_from_mean = np.exp(approx_results['random_intercept_mean']) * 20 * 60
        print(f"Mean per-track baseline: {baseline_from_mean:.2f} events/min")
        print(f"Empirical rate: 0.71 events/min")
        print(f"Ratio: {baseline_from_mean/0.71:.2f}x")
        print("\nIf the between-track intercept SD is high, this suggests")
        print("individual variability is a major source of rate mismatch.")


if __name__ == '__main__':
    main()




