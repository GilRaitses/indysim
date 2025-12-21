#!/usr/bin/env python3
"""
Factorial NB-GLM Model Fitting

Fits a pooled negative-binomial GLM with factorial effects for intensity
and temperature conditions, using additive kernel modulation.

Model:
    log λ(t) = β₀ + β_I·I + β_T·T + β_{IT}·(I×T) 
             + α·K_on(t) + α_I·I·K_on(t) + α_T·T·K_on(t) 
             + γ·K_off(t)

Where:
    I = 0 (0→250) or 1 (50→250)
    T = 0 (Control) or 1 (Temp)
    K_on(t) = gamma-difference kernel value at each frame
    K_off(t) = exponential rebound value at each frame

Usage:
    python scripts/fit_factorial_model.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.stats import gamma as gamma_dist

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available")


# Kernel parameters (from reference condition)
KERNEL_PARAMS = {
    'A': 0.456,
    'alpha1': 2.22,
    'beta1': 0.132,
    'B': 12.54,
    'alpha2': 4.38,
    'beta2': 0.869,
    'D': -0.114,
    'tau_off': 2.0
}

# LED timing
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


def gamma_pdf(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute gamma PDF."""
    t = np.maximum(t, 1e-10)
    return gamma_dist.pdf(t, alpha, scale=beta)


def compute_kernel_on(t: np.ndarray) -> np.ndarray:
    """
    Compute the gamma-difference LED-ON kernel.
    
    K_on(t) = A * Γ(t; α₁, β₁) - B * Γ(t; α₂, β₂)
    """
    p = KERNEL_PARAMS
    fast = p['A'] * gamma_pdf(t, p['alpha1'], p['beta1'])
    slow = p['B'] * gamma_pdf(t, p['alpha2'], p['beta2'])
    return fast - slow


def compute_kernel_off(t: np.ndarray) -> np.ndarray:
    """
    Compute the exponential LED-OFF rebound kernel.
    
    K_off(t) = D * exp(-t / τ_off)
    """
    p = KERNEL_PARAMS
    return p['D'] * np.exp(-t / p['tau_off'])


def compute_time_since_led_onset(times: np.ndarray) -> np.ndarray:
    """Compute time since last LED onset for each frame."""
    time_since_onset = np.full(len(times), -1.0)
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            time_since_onset[i] = -1
        else:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                time_since_onset[i] = cycle_time
            else:
                time_since_onset[i] = -1
    
    return time_since_onset


def compute_time_since_led_offset(times: np.ndarray) -> np.ndarray:
    """Compute time since last LED offset for each frame."""
    time_since_offset = np.full(len(times), -1.0)
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET + LED_ON_DURATION:
            time_since_offset[i] = -1
        else:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time >= LED_ON_DURATION:
                time_since_offset[i] = cycle_time - LED_ON_DURATION
            else:
                # In next ON period, time since last OFF
                time_since_offset[i] = LED_OFF_DURATION + cycle_time
    
    return time_since_offset


def get_condition_files(data_dir: str = "data/engineered") -> Dict[str, List[Path]]:
    """Get files for each condition."""
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob('*_events.csv'))
    
    conditions = {
        '0→250 | Control': [],
        '0→250 | Temp': [],
        '50→250 | Control': [],
        '50→250 | Temp': [],
    }
    
    # Anomalous files to exclude
    anomalous = ['202510291652', '202510291713']
    
    for f in all_files:
        if any(a in f.name for a in anomalous):
            continue
        
        # Parse condition
        if '0to250PWM' in f.name and '50to250PWM' not in f.name:
            intensity = '0→250'
        elif '50to250PWM' in f.name:
            intensity = '50→250'
        else:
            continue
        
        if '#C_Bl' in f.name:
            background = 'Control'
        elif '#T_Bl' in f.name:
            background = 'Temp'
        else:
            continue
        
        key = f'{intensity} | {background}'
        conditions[key].append(f)
    
    return conditions


def build_factorial_design_matrix(sample_rate: int = 100) -> pd.DataFrame:
    """
    Build the pooled factorial design matrix.
    
    Parameters
    ----------
    sample_rate : int
        Sample every Nth frame to reduce memory (default 100)
    
    Returns
    -------
    df : DataFrame
        Design matrix with columns:
        - events (0/1)
        - I, T, IT (factorial indicators)
        - K_on, I_K_on, T_K_on, K_off (kernel terms)
        - track, experiment, condition
    """
    conditions = get_condition_files()
    
    all_rows = []
    
    for condition, files in conditions.items():
        # Parse indicators
        I = 1 if '50→250' in condition else 0
        T = 1 if 'Temp' in condition else 0
        IT = I * T
        
        for f in files:
            print(f"Processing {f.name}...")
            df = pd.read_csv(f)
            
            # Sample frames
            df = df.iloc[::sample_rate].copy()
            
            # Get times and events
            times = df['time'].values
            events = df['is_reorientation_start'].fillna(0).values.astype(int)
            
            # Compute kernel values
            t_on = compute_time_since_led_onset(times)
            t_off = compute_time_since_led_offset(times)
            
            K_on = np.zeros(len(times))
            K_off = np.zeros(len(times))
            
            # LED-ON kernel (only during ON period)
            on_mask = t_on >= 0
            K_on[on_mask] = compute_kernel_on(t_on[on_mask])
            
            # LED-OFF kernel (only during OFF period)
            off_mask = (t_off >= 0) & (t_off < LED_OFF_DURATION)
            K_off[off_mask] = compute_kernel_off(t_off[off_mask])
            
            # Interaction terms
            I_K_on = I * K_on
            T_K_on = T * K_on
            
            # Build rows
            for track_id in df['track_id'].unique():
                track_mask = df['track_id'] == track_id
                n_track = track_mask.sum()
                
                track_data = {
                    'events': events[track_mask],
                    'I': np.full(n_track, I),
                    'T': np.full(n_track, T),
                    'IT': np.full(n_track, IT),
                    'K_on': K_on[track_mask],
                    'I_K_on': I_K_on[track_mask],
                    'T_K_on': T_K_on[track_mask],
                    'K_off': K_off[track_mask],
                    'track': [f"{f.stem}_{track_id}"] * n_track,
                    'experiment': [f.stem] * n_track,
                    'condition': [condition] * n_track,
                }
                
                all_rows.append(pd.DataFrame(track_data))
    
    result = pd.concat(all_rows, ignore_index=True)
    print(f"\nDesign matrix: {len(result):,} rows, {result['events'].sum():,} events")
    
    return result


def fit_factorial_glm(df: pd.DataFrame, alpha: float = 1.0) -> Dict:
    """
    Fit the factorial NB-GLM.
    
    Note: This uses fixed-effects GLM as statsmodels doesn't support
    NB with random effects. For full GLMM, use Bambi or R.
    
    Parameters
    ----------
    df : DataFrame
        Design matrix from build_factorial_design_matrix()
    alpha : float
        NB dispersion parameter
    
    Returns
    -------
    results : dict
        Fitted coefficients and diagnostics
    """
    # Build design matrix
    X = df[['I', 'T', 'IT', 'K_on', 'I_K_on', 'T_K_on', 'K_off']].values
    X = sm.add_constant(X)
    y = df['events'].values
    
    feature_names = ['intercept', 'I', 'T', 'IT', 'K_on', 'I_K_on', 'T_K_on', 'K_off']
    
    print("\nFitting NB-GLM...")
    model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
    fit = model.fit()
    
    print(fit.summary())
    
    # Extract coefficients
    coefficients = {}
    for i, name in enumerate(feature_names):
        coef = fit.params[i]
        se = fit.bse[i]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        pval = fit.pvalues[i]
        
        # Map to factorial notation
        if name == 'intercept':
            key = 'beta_0'
        elif name == 'I':
            key = 'beta_I'
        elif name == 'T':
            key = 'beta_T'
        elif name == 'IT':
            key = 'beta_IT'
        elif name == 'K_on':
            key = 'alpha'
        elif name == 'I_K_on':
            key = 'alpha_I'
        elif name == 'T_K_on':
            key = 'alpha_T'
        elif name == 'K_off':
            key = 'gamma'
        else:
            key = name
        
        coefficients[key] = {
            'mean': float(coef),
            'se': float(se),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'pvalue': float(pval),
            'significant': bool(pval < 0.05)
        }
    
    # Per-condition suppression amplitude
    conditions = df['condition'].unique()
    condition_amplitudes = {}
    
    for cond in conditions:
        I = 1 if '50→250' in cond else 0
        T = 1 if 'Temp' in cond else 0
        
        alpha_base = coefficients['alpha']['mean']
        alpha_I = coefficients['alpha_I']['mean']
        alpha_T = coefficients['alpha_T']['mean']
        
        amplitude = alpha_base + alpha_I * I + alpha_T * T
        condition_amplitudes[cond] = float(amplitude)
    
    return {
        'coefficients': coefficients,
        'condition_amplitudes': condition_amplitudes,
        'n_observations': len(df),
        'n_events': int(df['events'].sum()),
        'n_tracks': df['track'].nunique(),
        'n_experiments': df['experiment'].nunique(),
        'aic': float(fit.aic),
        'bic': float(fit.bic),
        'deviance': float(fit.deviance),
        'converged': fit.converged
    }


def compute_per_condition_validation(df: pd.DataFrame, results: Dict) -> Dict:
    """
    Compute validation metrics per condition.
    """
    coefficients = results['coefficients']
    
    validation = {}
    
    for condition in df['condition'].unique():
        cond_df = df[df['condition'] == condition]
        
        # Get indicators
        I = cond_df['I'].iloc[0]
        T = cond_df['T'].iloc[0]
        
        # Compute predicted hazard
        beta_0 = coefficients['beta_0']['mean']
        beta_I = coefficients['beta_I']['mean']
        beta_T = coefficients['beta_T']['mean']
        beta_IT = coefficients['beta_IT']['mean']
        alpha = coefficients['alpha']['mean']
        alpha_I = coefficients['alpha_I']['mean']
        alpha_T = coefficients['alpha_T']['mean']
        gamma = coefficients['gamma']['mean']
        
        eta = (beta_0 + beta_I * I + beta_T * T + beta_IT * I * T +
               (alpha + alpha_I * I + alpha_T * T) * cond_df['K_on'].values +
               gamma * cond_df['K_off'].values)
        
        predicted_rate = np.exp(eta).sum()
        empirical_events = cond_df['events'].sum()
        
        rate_ratio = predicted_rate / empirical_events if empirical_events > 0 else 0
        
        validation[condition] = {
            'empirical_events': int(empirical_events),
            'predicted_rate': float(predicted_rate),
            'rate_ratio': float(rate_ratio),
            'n_tracks': cond_df['track'].nunique(),
            'n_frames': len(cond_df)
        }
    
    return validation


def main():
    print("=" * 70)
    print("FACTORIAL NB-GLM MODEL FITTING")
    print("=" * 70)
    
    # Build design matrix
    print("\nBuilding design matrix (full data, no sampling)...")
    df = build_factorial_design_matrix(sample_rate=1)
    
    # Save design matrix
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / 'factorial_design_matrix.parquet', index=False)
    print(f"\nSaved design matrix to {output_dir / 'factorial_design_matrix.parquet'}")
    
    # Fit model
    results = fit_factorial_glm(df)
    
    # Per-condition validation
    print("\n" + "=" * 70)
    print("PER-CONDITION VALIDATION")
    print("=" * 70)
    
    validation = compute_per_condition_validation(df, results)
    results['validation'] = validation
    
    print(f"\n{'Condition':<25} {'Events':>8} {'Rate Ratio':>12} {'Pass':>6}")
    print("-" * 55)
    for cond, v in validation.items():
        passed = 0.8 <= v['rate_ratio'] <= 1.25
        status = 'PASS' if passed else 'FAIL'
        print(f"{cond:<25} {v['empirical_events']:>8} {v['rate_ratio']:>12.3f} {status:>6}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COEFFICIENT SUMMARY")
    print("=" * 70)
    print(f"\n{'Parameter':<12} {'Estimate':>10} {'95% CI':>20} {'p-value':>10} {'Sig':>5}")
    print("-" * 60)
    for key, val in results['coefficients'].items():
        sig = '*' if val['significant'] else ''
        ci = f"[{val['ci_low']:.3f}, {val['ci_high']:.3f}]"
        print(f"{key:<12} {val['mean']:>10.4f} {ci:>20} {val['pvalue']:>10.4f} {sig:>5}")
    
    print("\n" + "=" * 70)
    print("CONDITION AMPLITUDES")
    print("=" * 70)
    for cond, amp in results['condition_amplitudes'].items():
        print(f"  {cond}: {amp:.3f}")
    
    # Save results
    model_dir = Path('data/model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / 'factorial_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {model_dir / 'factorial_model_results.json'}")
    
    return results


if __name__ == '__main__':
    main()
