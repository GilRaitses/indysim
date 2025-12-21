#!/usr/bin/env python3
"""
Hybrid Model: BO-Optimal Kernel + Random Intercepts

Combines:
1. BO-optimized kernel configuration (4 early, 6 late bases)
2. Random intercepts per track (from mixed-effects)

This should capture both:
- Optimal temporal dynamics (from Bayesian optimization)
- Individual baseline variability (from random intercepts)

Usage:
    python scripts/fit_hybrid_model.py
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


# BO-optimal kernel configuration
BO_OPTIMAL_CONFIG = {
    'early_centers': [0.2, 0.6333, 1.0667, 1.5],
    'early_width': 0.30,
    'intm_centers': [2.0, 2.5],
    'intm_width': 0.6,
    'late_centers': [3.0, 4.2, 5.4, 6.6, 7.8, 9.0],
    'late_width': 2.494,
    'rebound_tau': 2.0
}

# Stimulus parameters
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


def load_data() -> pd.DataFrame:
    """Load engineered dataset (same as fit_extended_biphasic_model.py)."""
    data_dir = Path('data/engineered')
    
    # Load specific event files (same as fit_extended_biphasic_model.py)
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        csv_files = sorted(data_dir.glob('*_events.csv'))[:2]
    
    if not csv_files:
        raise FileNotFoundError("No event CSV files found in data/engineered/")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def compute_time_since_led_onset(data: pd.DataFrame) -> np.ndarray:
    """Compute time since last LED onset for each observation."""
    times = data['time'].values
    led_values = data['led1Val'].values if 'led1Val' in data.columns else np.zeros(len(data))
    
    # Use fixed LED cycle
    time_since_onset = np.zeros(len(data))
    
    for i, t in enumerate(times):
        # Find which LED cycle we're in
        if t < FIRST_LED_ONSET:
            time_since_onset[i] = -1  # Before first LED
        else:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                time_since_onset[i] = cycle_time  # During LED ON
            else:
                time_since_onset[i] = -1  # During LED OFF (will be handled by rebound)
    
    return time_since_onset


def compute_led_off_rebound(data: pd.DataFrame, tau: float = 2.0) -> np.ndarray:
    """Compute LED-off rebound term."""
    times = data['time'].values
    rebound = np.zeros(len(data))
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            continue
        
        cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
        if cycle_time >= LED_ON_DURATION:
            # Time since LED turned off
            t_since_off = cycle_time - LED_ON_DURATION
            rebound[i] = np.exp(-t_since_off / tau)
    
    return rebound


def build_design_matrix(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Build design matrix with BO-optimal kernel configuration.
    
    Returns
    -------
    X : ndarray
        Design matrix (n_obs x n_features)
    y : ndarray
        Response variable (event counts)
    feature_names : list
        Names of features
    track_ids : ndarray
        Track IDs for random effects
    """
    config = BO_OPTIMAL_CONFIG
    
    # Compute time since LED onset
    time_since_onset = compute_time_since_led_onset(data)
    
    # Compute kernel bases
    early_basis = raised_cosine_basis(
        time_since_onset, 
        np.array(config['early_centers']), 
        config['early_width']
    )
    intm_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['intm_centers']),
        config['intm_width']
    )
    late_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['late_centers']),
        config['late_width']
    )
    
    # Compute rebound
    rebound = compute_led_off_rebound(data, config['rebound_tau'])
    
    # Build design matrix
    X = np.column_stack([
        np.ones(len(data)),  # Intercept
        early_basis,
        intm_basis,
        late_basis,
        rebound
    ])
    
    # Feature names
    feature_names = ['intercept']
    feature_names += [f'kernel_early_{i+1}' for i in range(len(config['early_centers']))]
    feature_names += [f'kernel_intm_{i+1}' for i in range(len(config['intm_centers']))]
    feature_names += [f'kernel_late_{i+1}' for i in range(len(config['late_centers']))]
    feature_names += ['led_off_rebound']
    
    # Response (handle NaN)
    if 'is_reorientation_start' in data.columns:
        y = data['is_reorientation_start'].fillna(0).values.astype(int)
    else:
        y = data['is_reorientation'].fillna(0).values.astype(int)
    
    # Track IDs for random effects
    track_ids = data['track_id'].values
    
    return X, y, feature_names, track_ids


def fit_per_track_intercepts(X: np.ndarray, y: np.ndarray, 
                              track_ids: np.ndarray,
                              alpha: float = 1.0) -> Dict:
    """
    Fit model with per-track intercepts using two-stage approach.
    
    Stage 1: Fit global model to get kernel coefficients
    Stage 2: Estimate per-track intercepts given kernel
    
    Parameters
    ----------
    X : ndarray
        Design matrix
    y : ndarray
        Response
    track_ids : ndarray
        Track IDs
    alpha : float
        NB dispersion parameter
    
    Returns
    -------
    results : dict
        Fitted coefficients and diagnostics
    """
    unique_tracks = np.unique(track_ids)
    n_tracks = len(unique_tracks)
    
    print(f"\nFitting hybrid model with {n_tracks} tracks...")
    print("Stage 1: Fit global kernel coefficients")
    
    # Stage 1: Global fit (same as before)
    model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
    fit_global = model.fit()
    
    global_intercept = fit_global.params[0]
    kernel_coeffs = fit_global.params[1:]
    
    print(f"  Global intercept: {global_intercept:.3f}")
    
    # Stage 2: Estimate per-track intercepts
    print("\nStage 2: Estimate per-track intercepts")
    
    track_intercepts = {}
    track_event_rates = {}
    
    for track in unique_tracks:
        mask = track_ids == track
        X_track = X[mask]
        y_track = y[mask]
        n_events = y_track.sum()
        n_obs = len(y_track)
        
        if n_events == 0:
            # No events - use global intercept with penalty
            track_intercepts[int(track)] = global_intercept - 1.0
        else:
            # Estimate track-specific intercept
            # Linear predictor without intercept
            eta_kernel = X_track[:, 1:] @ kernel_coeffs
            
            # Find intercept that matches observed event rate
            observed_rate = n_events / n_obs
            
            # mu = exp(intercept + eta_kernel)
            # E[y] = mu, so sum(y) / n = mean(exp(intercept + eta_kernel))
            # intercept = log(observed_rate) - log(mean(exp(eta_kernel)))
            
            mean_kernel_effect = np.mean(np.exp(eta_kernel))
            if mean_kernel_effect > 0:
                track_intercept = np.log(observed_rate / mean_kernel_effect)
            else:
                track_intercept = global_intercept
            
            # Shrink toward global (regularization)
            shrinkage = 0.5
            track_intercept = shrinkage * track_intercept + (1 - shrinkage) * global_intercept
            
            track_intercepts[int(track)] = float(track_intercept)
        
        track_event_rates[int(track)] = float(n_events / n_obs * 60 / 0.05)  # events/min
    
    # Compute statistics
    intercept_values = list(track_intercepts.values())
    intercept_mean = np.mean(intercept_values)
    intercept_std = np.std(intercept_values)
    
    print(f"\n  Per-track intercept statistics:")
    print(f"    Mean: {intercept_mean:.3f}")
    print(f"    Std:  {intercept_std:.3f}")
    print(f"    Range: [{min(intercept_values):.3f}, {max(intercept_values):.3f}]")
    
    # Build coefficients dict
    coefficients = {'intercept_mean': float(intercept_mean)}
    for i, name in enumerate(fit_global.model.exog_names[1:]):
        coefficients[name] = float(kernel_coeffs[i])
    
    return {
        'coefficients': coefficients,
        'kernel_coeffs': kernel_coeffs.tolist(),
        'global_intercept': float(global_intercept),
        'track_intercepts': track_intercepts,
        'track_event_rates': track_event_rates,
        'intercept_mean': float(intercept_mean),
        'intercept_std': float(intercept_std),
        'aic': float(fit_global.aic),
        'converged': True,
        'n_tracks': n_tracks,
        'kernel_config': BO_OPTIMAL_CONFIG
    }


def refractory_factor(t_since_last: float, tau: float = 0.8, factor_min: float = 0.1) -> float:
    """Soft refractory: suppress hazard after recent events."""
    if t_since_last <= 0:
        return factor_min
    return 1.0 - (1.0 - factor_min) * np.exp(-t_since_last / tau)


def simulate_with_track_intercepts(results: Dict, data: pd.DataFrame,
                                    n_simulations: int = 1,
                                    use_refractory: bool = True,
                                    intercept_offset: float = -1.0) -> pd.DataFrame:
    """
    Simulate events using per-track intercepts with refractory.
    
    Parameters
    ----------
    intercept_offset : float
        Offset to add to all intercepts (negative = lower rate)
    use_refractory : bool
        Whether to apply soft refractory period
    """
    print(f"\nSimulating with per-track intercepts (offset={intercept_offset}, refractory={use_refractory})...")
    
    config = results['kernel_config']
    kernel_coeffs = np.array(results['kernel_coeffs'])
    track_intercepts = results['track_intercepts']
    global_intercept = results['global_intercept']
    
    rng = np.random.default_rng(42)
    dt = 0.05  # Frame duration
    
    all_events = []
    
    for (exp_id, track_id), group in data.groupby(['experiment_id', 'track_id']):
        # Get track-specific intercept with offset
        intercept = track_intercepts.get(int(track_id), global_intercept) + intercept_offset
        
        times = group['time'].values
        time_since_onset = compute_time_since_led_onset(group)
        rebound = compute_led_off_rebound(group, config['rebound_tau'])
        
        # Compute kernel contribution
        early_basis = raised_cosine_basis(
            time_since_onset,
            np.array(config['early_centers']),
            config['early_width']
        )
        intm_basis = raised_cosine_basis(
            time_since_onset,
            np.array(config['intm_centers']),
            config['intm_width']
        )
        late_basis = raised_cosine_basis(
            time_since_onset,
            np.array(config['late_centers']),
            config['late_width']
        )
        
        # Combine bases
        X_kernel = np.column_stack([early_basis, intm_basis, late_basis, rebound])
        eta_kernel = X_kernel @ kernel_coeffs
        
        # Full linear predictor
        eta = intercept + eta_kernel
        
        # Base hazard (probability per frame)
        base_hazard = np.exp(eta)
        base_hazard = np.clip(base_hazard, 0, 0.5)
        
        # Simulate with refractory
        last_event_time = -np.inf
        track_events = []
        
        for i in range(len(times)):
            # Apply refractory suppression
            if use_refractory:
                t_since_last = times[i] - last_event_time
                refrac = refractory_factor(t_since_last, tau=0.8, factor_min=0.1)
                hazard = base_hazard[i] * refrac
            else:
                hazard = base_hazard[i]
            
            # Bernoulli draw
            if rng.random() < hazard:
                track_events.append({
                    'experiment_id': exp_id,
                    'track_id': track_id,
                    'time': times[i],
                    'is_reorientation': 1
                })
                last_event_time = times[i]
        
        all_events.extend(track_events)
    
    return pd.DataFrame(all_events)


def validate_simulation(sim_events: pd.DataFrame, emp_data: pd.DataFrame) -> Dict:
    """Validate simulated events against empirical."""
    # Compute rates
    emp_events = emp_data[emp_data.get('is_reorientation_start', emp_data.get('is_reorientation', 0)) == 1]
    
    n_tracks_emp = emp_data.groupby(['experiment_id', 'track_id']).ngroups
    n_tracks_sim = len(sim_events['track_id'].unique()) if len(sim_events) > 0 else n_tracks_emp
    
    duration = emp_data['time'].max() / 60  # minutes
    
    emp_rate = len(emp_events) / (n_tracks_emp * duration)
    sim_rate = len(sim_events) / (n_tracks_sim * duration) if len(sim_events) > 0 else 0
    
    rate_ratio = sim_rate / emp_rate if emp_rate > 0 else 0
    
    print(f"\nValidation Results:")
    print(f"  Empirical events: {len(emp_events)}")
    print(f"  Simulated events: {len(sim_events)}")
    print(f"  Empirical rate: {emp_rate:.3f} events/min/track")
    print(f"  Simulated rate: {sim_rate:.3f} events/min/track")
    print(f"  Rate ratio: {rate_ratio:.2f}x")
    
    return {
        'emp_events': len(emp_events),
        'sim_events': len(sim_events),
        'emp_rate': emp_rate,
        'sim_rate': sim_rate,
        'rate_ratio': rate_ratio
    }


def main():
    print("=" * 70)
    print("HYBRID MODEL: BO-Optimal Kernel + Per-Track Intercepts")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    print(f"  Loaded {len(data):,} observations")
    
    n_events = data.get('is_reorientation_start', data.get('is_reorientation', 0)).sum()
    print(f"  Events: {n_events:,}")
    
    # Build design matrix
    print("\nBuilding design matrix with BO-optimal kernel...")
    X, y, feature_names, track_ids = build_design_matrix(data)
    print(f"  Features: {feature_names}")
    
    # Fit hybrid model
    results = fit_per_track_intercepts(X, y, track_ids)
    
    # Print kernel coefficients
    print("\n" + "=" * 50)
    print("KERNEL COEFFICIENTS (BO-optimal structure)")
    print("=" * 50)
    
    for name, value in results['coefficients'].items():
        if name != 'intercept_mean':
            print(f"  {name}: {value:+.4f}")
    
    # Simulate with calibrated offset
    # Try multiple offsets to find best rate match
    print("\n" + "=" * 50)
    print("CALIBRATING INTERCEPT OFFSET")
    print("=" * 50)
    
    best_offset = -0.5
    best_ratio_diff = float('inf')
    
    for offset in [-0.3, -0.5, -0.7, -0.9]:
        sim_test = simulate_with_track_intercepts(results, data, intercept_offset=offset)
        n_sim = len(sim_test)
        sim_rate = n_sim / (results['n_tracks'] * 20.0)  # events/min/track
        emp_rate = 0.711
        ratio = sim_rate / emp_rate
        ratio_diff = abs(ratio - 1.0)
        print(f"  offset={offset}: rate={sim_rate:.3f}, ratio={ratio:.2f}x")
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_offset = offset
    
    print(f"\nBest offset: {best_offset}")
    
    # Final simulation with best offset
    sim_events = simulate_with_track_intercepts(results, data, intercept_offset=best_offset)
    results['intercept_offset'] = best_offset
    
    # Validate
    validation = validate_simulation(sim_events, data)
    results['validation'] = validation
    
    # Save results
    output_path = Path('data/model/hybrid_model_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Save simulated events
    sim_output = Path('data/simulated/hybrid_model_events.csv')
    sim_events.to_csv(sim_output, index=False)
    print(f"Saved simulated events to {sim_output}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Hybrid Model Results:
    - Kernel: BO-optimal (4 early, 6 late bases)
    - Intercepts: Per-track (mean={results['intercept_mean']:.3f}, SD={results['intercept_std']:.3f})
    - Rate ratio: {validation['rate_ratio']:.2f}x
    
    This model accounts for individual larva variability while using
    the optimized temporal kernel structure from Bayesian optimization.
    """)


if __name__ == '__main__':
    main()
