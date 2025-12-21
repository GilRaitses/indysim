#!/usr/bin/env python3
"""
Cross-Condition Fitting

Fits the hazard model to all 4 conditions in the 2×2 factorial design:
1. 0→250 | Control (reference - already fitted)
2. 0→250 | Temp
3. 50→250 | Control
4. 50→250 | Temp

Compares kernel parameters across conditions to test stability.

Usage:
    python scripts/fit_cross_condition.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.optimize import minimize

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available")


# BO-optimal kernel configuration (from original model)
BO_OPTIMAL_CONFIG = {
    'early_centers': [0.2, 0.6333, 1.0667, 1.5],
    'early_width': 0.30,
    'intm_centers': [2.0, 2.5],
    'intm_width': 0.6,
    'late_centers': [3.0, 4.2, 5.4, 6.6, 7.8, 9.0],
    'late_width': 2.494,
    'rebound_tau': 2.0
}

# LED timing (verified identical across all conditions)
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


@dataclass
class ConditionData:
    """Container for condition-specific data."""
    name: str
    intensity: str  # '0→250' or '50→250'
    background: str  # 'Control' or 'Temp'
    files: List[Path]
    data: Optional[pd.DataFrame] = None
    n_tracks: int = 0
    n_events: int = 0
    n_frames: int = 0


def get_condition_files(data_dir: str = "data/engineered") -> Dict[str, ConditionData]:
    """
    Get files for each condition in the 2×2 factorial design.
    
    Returns dict mapping condition name to ConditionData.
    """
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob('*_events.csv'))
    
    conditions = {}
    
    # Define condition patterns
    condition_patterns = {
        '0→250 | Control': ('0to250PWM', '#C_Bl_7PWM'),
        '0→250 | Temp': ('0to250PWM', '#T_Bl_Sq_5to15PWM'),
        '50→250 | Control': ('50to250PWM', '#C_Bl_7PWM'),
        '50→250 | Temp': ('50to250PWM', '#T_Bl_Sq_5to15PWM'),
    }
    
    for name, (intensity_pattern, bg_pattern) in condition_patterns.items():
        matching_files = [
            f for f in all_files
            if intensity_pattern in f.name and bg_pattern in f.name
        ]
        
        # Parse intensity and background
        intensity = '0→250' if '0to250' in intensity_pattern else '50→250'
        background = 'Control' if 'C_Bl' in bg_pattern else 'Temp'
        
        conditions[name] = ConditionData(
            name=name,
            intensity=intensity,
            background=background,
            files=matching_files
        )
    
    return conditions


def load_condition_data(
    condition: ConditionData,
    exclude_anomalous: bool = True
) -> pd.DataFrame:
    """
    Load data for a specific condition.
    
    Parameters
    ----------
    condition : ConditionData
        Condition to load
    exclude_anomalous : bool
        If True, exclude experiments with anomalously high event counts
        (only applies to 0→250 | Control)
    
    Returns
    -------
    data : DataFrame
        Combined data from all files in condition
    """
    # Anomalous files to exclude (10-20x higher event counts)
    anomalous_files = ['202510291652', '202510291713']
    
    dfs = []
    for f in condition.files:
        # Check for anomalous files
        if exclude_anomalous and any(a in f.name for a in anomalous_files):
            print(f"  Excluding anomalous file: {f.name}")
            continue
        
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Update condition stats
    condition.data = data
    condition.n_frames = len(data)
    condition.n_tracks = data['track_id'].nunique()
    condition.n_events = data['is_reorientation_start'].sum()
    
    return data


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


def compute_time_since_led_onset(data: pd.DataFrame) -> np.ndarray:
    """Compute time since last LED onset for each observation."""
    times = data['time'].values
    time_since_onset = np.zeros(len(data))
    
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


def compute_led_off_rebound(data: pd.DataFrame, tau: float = 2.0) -> np.ndarray:
    """Compute LED-off rebound term."""
    times = data['time'].values
    rebound = np.zeros(len(data))
    
    for i, t in enumerate(times):
        if t < FIRST_LED_ONSET:
            continue
        
        cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
        if cycle_time >= LED_ON_DURATION:
            t_since_off = cycle_time - LED_ON_DURATION
            rebound[i] = np.exp(-t_since_off / tau)
    
    return rebound


def build_design_matrix(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build design matrix with BO-optimal kernel configuration."""
    config = BO_OPTIMAL_CONFIG
    
    time_since_onset = compute_time_since_led_onset(data)
    
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
    
    rebound = compute_led_off_rebound(data, config['rebound_tau'])
    
    X = np.column_stack([
        np.ones(len(data)),
        early_basis,
        intm_basis,
        late_basis,
        rebound
    ])
    
    feature_names = ['intercept']
    feature_names += [f'kernel_early_{i+1}' for i in range(len(config['early_centers']))]
    feature_names += [f'kernel_intm_{i+1}' for i in range(len(config['intm_centers']))]
    feature_names += [f'kernel_late_{i+1}' for i in range(len(config['late_centers']))]
    feature_names += ['led_off_rebound']
    
    y = data['is_reorientation_start'].fillna(0).values.astype(int)
    track_ids = data['track_id'].values
    
    return X, y, feature_names, track_ids


def fit_condition(
    condition: ConditionData,
    alpha: float = 1.0
) -> Dict:
    """
    Fit hazard model to a single condition.
    
    Returns dict with:
    - coefficients
    - kernel_coeffs
    - global_intercept
    - track_intercepts
    - diagnostics
    """
    data = condition.data
    if data is None or len(data) == 0:
        return {'error': 'No data'}
    
    print(f"\n{'='*60}")
    print(f"Fitting: {condition.name}")
    print(f"  Files: {len(condition.files)}")
    print(f"  Tracks: {condition.n_tracks}")
    print(f"  Events: {condition.n_events}")
    print(f"{'='*60}")
    
    X, y, feature_names, track_ids = build_design_matrix(data)
    unique_tracks = np.unique(track_ids)
    n_tracks = len(unique_tracks)
    
    # Stage 1: Global fit
    print("Stage 1: Fit global kernel coefficients...")
    model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
    
    try:
        fit_global = model.fit()
    except Exception as e:
        print(f"  Fit failed: {e}")
        return {'error': str(e)}
    
    global_intercept = fit_global.params[0]
    kernel_coeffs = fit_global.params[1:]
    
    print(f"  Global intercept: {global_intercept:.3f}")
    
    # Stage 2: Per-track intercepts
    print("Stage 2: Estimate per-track intercepts...")
    track_intercepts = {}
    
    for track in unique_tracks:
        mask = track_ids == track
        X_track = X[mask]
        y_track = y[mask]
        n_events = y_track.sum()
        n_obs = len(y_track)
        
        if n_events == 0:
            track_intercepts[int(track)] = global_intercept - 1.0
        else:
            eta_kernel = X_track[:, 1:] @ kernel_coeffs
            observed_rate = n_events / n_obs
            
            mean_kernel_effect = np.mean(np.exp(eta_kernel))
            if mean_kernel_effect > 0:
                track_intercept = np.log(observed_rate / mean_kernel_effect)
            else:
                track_intercept = global_intercept
            
            # Shrink toward global
            shrinkage = 0.5
            track_intercept = shrinkage * track_intercept + (1 - shrinkage) * global_intercept
            track_intercepts[int(track)] = float(track_intercept)
    
    intercept_values = list(track_intercepts.values())
    intercept_mean = np.mean(intercept_values)
    intercept_std = np.std(intercept_values)
    
    print(f"  Track intercepts: mean={intercept_mean:.3f}, std={intercept_std:.3f}")
    
    # Build coefficients dict
    coefficients = {'intercept_mean': float(intercept_mean)}
    for i, name in enumerate(feature_names[1:]):
        coefficients[name] = float(kernel_coeffs[i])
    
    return {
        'condition': condition.name,
        'intensity': condition.intensity,
        'background': condition.background,
        'n_tracks': n_tracks,
        'n_events': int(condition.n_events),
        'n_frames': int(condition.n_frames),
        'coefficients': coefficients,
        'kernel_coeffs': kernel_coeffs.tolist(),
        'global_intercept': float(global_intercept),
        'track_intercepts': track_intercepts,
        'intercept_mean': float(intercept_mean),
        'intercept_std': float(intercept_std),
        'aic': float(fit_global.aic),
        'converged': True
    }


def compute_kernel_from_coeffs(coeffs: List[float], t: np.ndarray) -> np.ndarray:
    """Reconstruct kernel values from basis coefficients."""
    config = BO_OPTIMAL_CONFIG
    
    early_basis = raised_cosine_basis(t, np.array(config['early_centers']), config['early_width'])
    intm_basis = raised_cosine_basis(t, np.array(config['intm_centers']), config['intm_width'])
    late_basis = raised_cosine_basis(t, np.array(config['late_centers']), config['late_width'])
    
    basis = np.column_stack([early_basis, intm_basis, late_basis])
    kernel_coeffs = coeffs[:-1]  # Exclude rebound
    
    return basis @ kernel_coeffs


def compare_kernels(results: Dict[str, Dict]) -> Dict:
    """
    Compare kernel parameters across conditions.
    
    Returns comparison statistics and stability assessment.
    """
    t = np.linspace(0, 10, 200)
    
    # Reference: 0→250 | Control
    ref_name = '0→250 | Control'
    ref_result = results.get(ref_name)
    
    if ref_result is None or 'kernel_coeffs' not in ref_result:
        return {'error': 'Reference condition missing'}
    
    ref_kernel = compute_kernel_from_coeffs(ref_result['kernel_coeffs'], t)
    
    comparisons = {}
    
    for name, result in results.items():
        if 'kernel_coeffs' not in result:
            continue
        
        kernel = compute_kernel_from_coeffs(result['kernel_coeffs'], t)
        
        # Compute comparison metrics
        correlation = np.corrcoef(ref_kernel, kernel)[0, 1]
        rmse = np.sqrt(np.mean((ref_kernel - kernel) ** 2))
        max_abs_diff = np.max(np.abs(ref_kernel - kernel))
        
        # Suppression magnitude (minimum of kernel)
        suppression = -np.min(kernel)
        ref_suppression = -np.min(ref_kernel)
        suppression_ratio = suppression / ref_suppression if ref_suppression != 0 else 0
        
        # Peak timing (time of maximum suppression)
        peak_time = t[np.argmin(kernel)]
        ref_peak_time = t[np.argmin(ref_kernel)]
        peak_time_diff = peak_time - ref_peak_time
        
        comparisons[name] = {
            'correlation_with_ref': float(correlation),
            'rmse_vs_ref': float(rmse),
            'max_abs_diff': float(max_abs_diff),
            'suppression_magnitude': float(suppression),
            'suppression_ratio': float(suppression_ratio),
            'peak_time': float(peak_time),
            'peak_time_diff': float(peak_time_diff),
            'intercept_mean': result['intercept_mean'],
            'n_events': result['n_events'],
            'n_tracks': result['n_tracks']
        }
    
    # Assess stability
    correlations = [c['correlation_with_ref'] for c in comparisons.values()]
    min_corr = min(correlations)
    
    stability_assessment = {
        'min_correlation': float(min_corr),
        'mean_correlation': float(np.mean(correlations)),
        'kernel_stable': min_corr > 0.8,  # Threshold for stability
        'recommendation': 'proceed_factorial' if min_corr > 0.8 else 'condition_specific_kernels'
    }
    
    return {
        'per_condition': comparisons,
        'stability': stability_assessment
    }


def main():
    print("=" * 70)
    print("CROSS-CONDITION FITTING")
    print("=" * 70)
    
    # Get condition files
    conditions = get_condition_files()
    
    print("\nCondition files found:")
    for name, cond in conditions.items():
        print(f"  {name}: {len(cond.files)} files")
    
    # Load data for each condition
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    for name, cond in conditions.items():
        print(f"\nLoading {name}...")
        load_condition_data(cond, exclude_anomalous=True)
        print(f"  Tracks: {cond.n_tracks}, Events: {cond.n_events}, Frames: {cond.n_frames:,}")
    
    # Fit model to each condition
    print("\n" + "=" * 70)
    print("FITTING MODELS")
    print("=" * 70)
    
    results = {}
    for name, cond in conditions.items():
        if cond.n_events > 0:
            results[name] = fit_condition(cond)
        else:
            print(f"\nSkipping {name}: no data")
    
    # Compare kernels
    print("\n" + "=" * 70)
    print("KERNEL COMPARISON")
    print("=" * 70)
    
    comparison = compare_kernels(results)
    
    print("\nPer-condition metrics vs reference (0→250 | Control):")
    print("-" * 60)
    print(f"{'Condition':<25} {'Corr':>8} {'RMSE':>8} {'Suppr':>8} {'Peak':>8}")
    print("-" * 60)
    
    for name, metrics in comparison['per_condition'].items():
        print(f"{name:<25} {metrics['correlation_with_ref']:>8.3f} "
              f"{metrics['rmse_vs_ref']:>8.3f} {metrics['suppression_ratio']:>8.2f}x "
              f"{metrics['peak_time']:>7.1f}s")
    
    print("-" * 60)
    
    stability = comparison['stability']
    print(f"\nStability Assessment:")
    print(f"  Min correlation: {stability['min_correlation']:.3f}")
    print(f"  Mean correlation: {stability['mean_correlation']:.3f}")
    print(f"  Kernel stable: {stability['kernel_stable']}")
    print(f"  Recommendation: {stability['recommendation']}")
    
    # Save results
    output_path = Path('data/model/cross_condition_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'conditions': {name: {k: v for k, v in r.items() if k != 'track_intercepts'} 
                      for name, r in results.items()},
        'comparison': comparison,
        'config': BO_OPTIMAL_CONFIG
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Tracks':>8} {'Events':>8} {'Intercept':>10} {'AIC':>12}")
    print("-" * 70)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<25} {result['n_tracks']:>8} {result['n_events']:>8} "
                  f"{result['intercept_mean']:>10.3f} {result['aic']:>12.1f}")
    
    print("-" * 70)
    
    return results, comparison


if __name__ == '__main__':
    main()


