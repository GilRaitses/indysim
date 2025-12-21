#!/usr/bin/env python3
"""
Bayesian Optimization for Kernel Basis Configuration

Automatically searches for optimal kernel parameters:
- Number of early/intermediate/late bases
- Center positions and widths
- Objective: minimize W-ISE with rate constraint

Usage:
    python scripts/run_bayesian_optimization.py

Requirements:
    pip install scikit-optimize
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Try to import skopt, fall back to manual grid search
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("Warning: scikit-optimize not found. Using grid search instead.")
    print("Install with: pip install scikit-optimize")

# Parameters
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3
EXPERIMENT_DURATION = 1200.0
EMPIRICAL_RATE = 0.71  # events/min/track


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


def build_design_matrix(data: pd.DataFrame, kernel_config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build design matrix with given kernel configuration."""
    times = data['time'].values
    n = len(times)
    
    # Get response variable
    if 'is_reorientation_start' in data.columns:
        y = data['is_reorientation_start'].values.astype(float)
    else:
        y = data['is_reorientation'].values.astype(float)
    
    # Compute time since LED onset
    time_since_onset = np.zeros(n)
    is_led_on = np.zeros(n, dtype=bool)
    
    for i, t in enumerate(times):
        if t >= FIRST_LED_ONSET:
            time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
            if time_in_cycle < LED_ON_DURATION:
                time_since_onset[i] = time_in_cycle
                is_led_on[i] = True
    
    # Build kernel bases
    tso = time_since_onset[is_led_on]
    
    early_centers = np.array(kernel_config['early_centers'])
    intm_centers = np.array(kernel_config['intm_centers'])
    late_centers = np.array(kernel_config['late_centers'])
    
    n_early = len(early_centers)
    n_intm = len(intm_centers)
    n_late = len(late_centers)
    
    early_basis = np.zeros((n, n_early))
    intm_basis = np.zeros((n, n_intm))
    late_basis = np.zeros((n, n_late))
    
    early_basis[is_led_on] = raised_cosine_basis(tso, early_centers, kernel_config['early_width'])
    intm_basis[is_led_on] = raised_cosine_basis(tso, intm_centers, kernel_config['intm_width'])
    late_basis[is_led_on] = raised_cosine_basis(tso, late_centers, kernel_config['late_width'])
    
    # LED-off rebound
    rebound = np.zeros(n)
    off_mask = ~is_led_on & (times >= FIRST_LED_ONSET)
    for i in np.where(off_mask)[0]:
        t = times[i]
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        time_since_offset = time_in_cycle - LED_ON_DURATION
        if time_since_offset > 0:
            rebound[i] = np.exp(-time_since_offset / 2.0)
    
    # Build design matrix
    intercept = np.ones((n, 1))
    X = np.hstack([intercept, early_basis, intm_basis, late_basis, rebound.reshape(-1, 1)])
    
    feature_names = ['intercept']
    feature_names += [f'kernel_early_{i+1}' for i in range(n_early)]
    feature_names += [f'kernel_intm_{i+1}' for i in range(n_intm)]
    feature_names += [f'kernel_late_{i+1}' for i in range(n_late)]
    feature_names += ['led_off_rebound']
    
    return X, y, feature_names


def fit_nb_glm(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, 
               fixed_intercept: float = None) -> Dict:
    """Fit NB-GLM with optional fixed intercept."""
    n_params = X.shape[1]
    
    if fixed_intercept is not None:
        # Optimize only kernel coefficients
        def objective(beta_kernel):
            beta = np.concatenate([[fixed_intercept], beta_kernel])
            mu = np.exp(X @ beta)
            mu = np.clip(mu, 1e-10, 1e10)
            r = 1 / alpha
            ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
            ridge = 0.01 * np.sum(beta_kernel**2)
            return -(ll - ridge)
        
        beta_init = np.zeros(n_params - 1)
        bounds = [(None, None)] * (n_params - 1)
        bounds[0] = (0, None)  # First early basis non-negative
        
        result = minimize(objective, beta_init, method='L-BFGS-B', bounds=bounds)
        beta = np.concatenate([[fixed_intercept], result.x])
    else:
        def objective(beta):
            mu = np.exp(X @ beta)
            mu = np.clip(mu, 1e-10, 1e10)
            r = 1 / alpha
            ll = np.sum(y * np.log(mu / (mu + r)) + r * np.log(r / (mu + r)))
            ridge = 0.01 * np.sum(beta[1:]**2)
            return -(ll - ridge)
        
        beta_init = np.zeros(n_params)
        beta_init[0] = -7.0
        bounds = [(None, None)] * n_params
        bounds[0] = (-8.0, -6.0)
        bounds[1] = (0, None)
        
        result = minimize(objective, beta_init, method='L-BFGS-B', bounds=bounds)
        beta = result.x
    
    return {'coefficients': beta, 'converged': result.success}


def simulate_and_validate(coefficients: np.ndarray, kernel_config: Dict, 
                          emp_psth: np.ndarray, n_tracks: int = 99) -> Dict:
    """Simulate events and compute validation metrics."""
    # Build hazard function
    def hazard_func(t):
        if t < FIRST_LED_ONSET:
            return np.exp(coefficients[0])
        
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        
        if time_in_cycle < LED_ON_DURATION:
            tso = np.array([time_in_cycle])
            
            early_basis = raised_cosine_basis(tso, np.array(kernel_config['early_centers']), 
                                              kernel_config['early_width'])
            intm_basis = raised_cosine_basis(tso, np.array(kernel_config['intm_centers']), 
                                             kernel_config['intm_width'])
            late_basis = raised_cosine_basis(tso, np.array(kernel_config['late_centers']), 
                                             kernel_config['late_width'])
            
            idx = 1
            kernel_contrib = 0
            for j in range(len(kernel_config['early_centers'])):
                kernel_contrib += coefficients[idx] * early_basis[0, j]
                idx += 1
            for j in range(len(kernel_config['intm_centers'])):
                kernel_contrib += coefficients[idx] * intm_basis[0, j]
                idx += 1
            for j in range(len(kernel_config['late_centers'])):
                kernel_contrib += coefficients[idx] * late_basis[0, j]
                idx += 1
            
            return np.exp(coefficients[0] + kernel_contrib)
        else:
            time_since_offset = time_in_cycle - LED_ON_DURATION
            rebound = np.exp(-time_since_offset / 2.0)
            rebound_coef = coefficients[-1]
            return np.exp(coefficients[0] + rebound_coef * rebound)
    
    # Simulate events (discrete-time Bernoulli)
    rng = np.random.default_rng(42)
    dt = 0.05
    all_events = []
    
    for track in range(n_tracks):
        t = 0
        while t < EXPERIMENT_DURATION:
            p = hazard_func(t)
            p = np.clip(p, 0, 1)
            if rng.random() < p:
                all_events.append(t)
            t += dt
    
    sim_events = np.array(all_events)
    
    # Compute rate
    sim_rate = len(sim_events) / (EXPERIMENT_DURATION / 60) / n_tracks
    
    # Compute PSTH
    led_onsets = np.arange(FIRST_LED_ONSET, EXPERIMENT_DURATION, LED_CYCLE)
    bin_size = 0.2
    bins = np.arange(-3.0, 8.0 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    counts = np.zeros(len(bin_centers))
    for onset in led_onsets:
        for i, bc in enumerate(bin_centers):
            t_start = onset + bc - bin_size / 2
            t_end = onset + bc + bin_size / 2
            if t_start >= 0 and t_end <= EXPERIMENT_DURATION:
                counts[i] += np.sum((sim_events >= t_start) & (sim_events < t_end))
    
    sim_psth = counts / (len(led_onsets) * bin_size * n_tracks)
    
    # Normalize
    baseline_idx = bin_centers < 0
    emp_baseline = emp_psth[baseline_idx].mean() if emp_psth[baseline_idx].mean() > 0 else 1
    sim_baseline = sim_psth[baseline_idx].mean() if sim_psth[baseline_idx].mean() > 0 else 1
    
    emp_norm = emp_psth / emp_baseline
    sim_norm = sim_psth / sim_baseline
    
    # Compute metrics
    correlation = pearsonr(emp_norm, sim_norm)[0] if len(emp_norm) > 2 else 0
    wise = np.sqrt(np.mean((emp_norm - sim_norm)**2))
    rate_error = abs(sim_rate - EMPIRICAL_RATE) / EMPIRICAL_RATE
    
    return {
        'rate': sim_rate,
        'rate_error': rate_error,
        'correlation': correlation,
        'wise': wise,
        'sim_psth': sim_psth
    }


def compute_empirical_psth(data: pd.DataFrame) -> np.ndarray:
    """Compute empirical PSTH."""
    times = data['time'].values
    if 'is_reorientation_start' in data.columns:
        events = times[data['is_reorientation_start'] == 1]
    else:
        events = times[data['is_reorientation'] == 1]
    
    led_onsets = np.arange(FIRST_LED_ONSET, EXPERIMENT_DURATION, LED_CYCLE)
    bin_size = 0.2
    bins = np.arange(-3.0, 8.0 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    n_tracks = data.groupby(['experiment_id', 'track_id']).ngroups if 'experiment_id' in data.columns else data['track_id'].nunique()
    
    counts = np.zeros(len(bin_centers))
    for onset in led_onsets:
        for i, bc in enumerate(bin_centers):
            t_start = onset + bc - bin_size / 2
            t_end = onset + bc + bin_size / 2
            if t_start >= 0 and t_end <= EXPERIMENT_DURATION:
                counts[i] += np.sum((events >= t_start) & (events < t_end))
    
    return counts / (len(led_onsets) * bin_size * n_tracks)


def objective_function(params: List, data: pd.DataFrame, emp_psth: np.ndarray, 
                       fixed_intercept: float = -7.2) -> float:
    """
    Objective function for Bayesian optimization.
    Returns negative score (we minimize).
    """
    # Unpack parameters
    n_early = int(round(params[0]))
    n_late = int(round(params[1]))
    early_width = params[2]
    late_width = params[3]
    
    # Generate center positions
    early_centers = np.linspace(0.2, 1.5, n_early).tolist()
    intm_centers = [2.0, 2.5]  # Fixed intermediate
    late_centers = np.linspace(3.0, 9.0, n_late).tolist()
    
    kernel_config = {
        'early_centers': early_centers,
        'early_width': early_width,
        'intm_centers': intm_centers,
        'intm_width': 0.6,
        'late_centers': late_centers,
        'late_width': late_width
    }
    
    try:
        # Build design matrix and fit
        X, y, feature_names = build_design_matrix(data, kernel_config)
        fit_result = fit_nb_glm(X, y, fixed_intercept=fixed_intercept)
        
        if not fit_result['converged']:
            return 1e6
        
        # Validate
        val_result = simulate_and_validate(fit_result['coefficients'], kernel_config, emp_psth)
        
        # Objective: minimize W-ISE with rate penalty and physics-informed terms
        wise = val_result['wise']
        rate_error = val_result['rate_error']
        
        # Heavy penalty if rate is off by more than 100% (2x)
        # The 1.6x mismatch is fundamental; we optimize shape within that constraint
        if rate_error > 1.0:
            return 1e6
        
        # Compute kernel smoothness penalty (penalize jagged kernels)
        coeffs = fit_result['coefficients']
        kernel_coeffs = coeffs[1:-1]  # Exclude intercept and rebound
        smoothness = np.sum(np.diff(kernel_coeffs)**2)
        
        # Score: W-ISE + rate penalty + smoothness penalty
        # Per plan: score = WISE + 0.5 * rate_error + 0.05 * smoothness
        score = wise + 0.5 * rate_error + 0.05 * smoothness
        
        print(f"  n_early={n_early}, n_late={n_late}, early_w={early_width:.2f}, late_w={late_width:.2f}")
        print(f"    -> WISE={wise:.4f}, rate_err={rate_error:.2%}, corr={val_result['correlation']:.3f}, smooth={smoothness:.2f}")
        
        return score
        
    except Exception as e:
        print(f"  Error: {e}")
        return 1e6


def run_bayesian_optimization(data: pd.DataFrame, emp_psth: np.ndarray, n_calls: int = 30):
    """Run Bayesian optimization to find best kernel configuration."""
    
    if HAS_SKOPT:
        # Define search space
        space = [
            Integer(2, 5, name='n_early'),
            Integer(3, 6, name='n_late'),
            Real(0.3, 0.8, name='early_width'),
            Real(1.2, 2.5, name='late_width')
        ]
        
        def objective_wrapper(params):
            return objective_function(params, data, emp_psth)
        
        print(f"Running Bayesian optimization with {n_calls} evaluations...")
        result = gp_minimize(objective_wrapper, space, n_calls=n_calls, random_state=42, verbose=True)
        
        best_params = {
            'n_early': int(result.x[0]),
            'n_late': int(result.x[1]),
            'early_width': float(result.x[2]),
            'late_width': float(result.x[3]),
            'best_score': float(result.fun)
        }
        
    else:
        # Grid search fallback
        print("Running grid search (install scikit-optimize for Bayesian optimization)...")
        
        best_score = 1e6
        best_params = None
        
        for n_early in [2, 3, 4]:
            for n_late in [3, 4, 5]:
                for early_width in [0.4, 0.5, 0.6]:
                    for late_width in [1.5, 1.8, 2.0]:
                        params = [n_early, n_late, early_width, late_width]
                        score = objective_function(params, data, emp_psth)
                        
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'n_early': n_early,
                                'n_late': n_late,
                                'early_width': early_width,
                                'late_width': late_width,
                                'best_score': score
                            }
        
        result = None
    
    return best_params, result


def main():
    print("=" * 70)
    print("BAYESIAN OPTIMIZATION FOR KERNEL DESIGN")
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
    print(f"\nLoaded {len(data):,} observations")
    
    # Compute empirical PSTH
    emp_psth = compute_empirical_psth(data)
    print(f"Computed empirical PSTH ({len(emp_psth)} bins)")
    
    # Run optimization
    print("\n" + "=" * 50)
    best_params, result = run_bayesian_optimization(data, emp_psth, n_calls=25)
    
    print("\n" + "=" * 50)
    print("BEST CONFIGURATION FOUND")
    print("=" * 50)
    print(f"  n_early: {best_params['n_early']}")
    print(f"  n_late: {best_params['n_late']}")
    print(f"  early_width: {best_params['early_width']:.3f}")
    print(f"  late_width: {best_params['late_width']:.3f}")
    print(f"  score: {best_params['best_score']:.4f}")
    
    # Generate final configuration
    final_config = {
        'early_centers': np.linspace(0.2, 1.5, best_params['n_early']).tolist(),
        'early_width': best_params['early_width'],
        'intm_centers': [2.0, 2.5],
        'intm_width': 0.6,
        'late_centers': np.linspace(3.0, 9.0, best_params['n_late']).tolist(),
        'late_width': best_params['late_width']
    }
    
    print("\nOptimal kernel configuration:")
    print(json.dumps(final_config, indent=2))
    
    # Save results
    output_path = Path('data/model/bayesian_opt_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_params': best_params,
        'optimal_kernel_config': final_config
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Suggest next steps
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("1. Update fit_extended_biphasic_model.py with optimal config")
    print("2. Refit the model and validate")
    print("3. Consider adding PIML refractory penalty if IEI still off")


if __name__ == '__main__':
    main()




