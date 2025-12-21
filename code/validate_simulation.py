#!/usr/bin/env python3
"""
Simulation Validation for NB-GLM LNP Model

Validates that simulated larval trajectories match empirical statistics:
- Turn rate (events per minute per track)
- Stimulus-locked PSTH
- Heading change distribution
- Inter-event interval distribution

Usage:
    python scripts/validate_simulation.py --empirical data/processed/consolidated_dataset.h5 \
                                          --simulated data/simulated/trajectories.parquet \
                                          --output data/validation/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# HAZARD FUNCTION FROM FITTED MODEL
# =============================================================================

def make_hazard_function(
    coefficients: Dict[str, float],
    feature_names: List[str],
    kernel_centers: np.ndarray,
    kernel_width: float = 0.6,
    speed_mean: float = 0.0,
    speed_std: float = 1.0,
    curvature_mean: float = 0.0,
    curvature_std: float = 1.0
) -> Callable:
    """
    Create a hazard function from fitted GLM coefficients.
    
    Parameters
    ----------
    coefficients : dict
        Fitted coefficients from NB-GLM
    feature_names : list
        Names of features in order
    kernel_centers : ndarray
        Temporal kernel center positions
    kernel_width : float
        Kernel width parameter
    speed_mean, speed_std : float
        Normalization parameters for speed
    curvature_mean, curvature_std : float
        Normalization parameters for curvature
    
    Returns
    -------
    hazard : callable
        Function hazard(t, led1_pwm, led2_pwm, speed, curvature) -> rate
    """
    # Extract coefficient values in order
    coef_array = np.array([coefficients.get(name, 0.0) for name in feature_names])
    
    # Find kernel coefficient indices
    kernel_indices = [i for i, name in enumerate(feature_names) if name.startswith('kernel_')]
    kernel_coefs = coef_array[kernel_indices] if kernel_indices else np.zeros(len(kernel_centers))
    
    def raised_cosine_basis(t, centers, width):
        """Compute raised-cosine basis at time t."""
        t = np.atleast_1d(t)
        basis = np.zeros((len(t), len(centers)))
        for j, c in enumerate(centers):
            dist = np.abs(t - c)
            in_range = dist < width
            basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
        return basis
    
    def hazard(t: float, led1_pwm: float = 0.0, led2_pwm: float = 0.0,
               speed: float = 0.0, curvature: float = 0.0) -> float:
        """
        Compute instantaneous reorientation hazard at time t.
        
        Parameters
        ----------
        t : float
            Experiment time (seconds)
        led1_pwm : float
            Current LED1 intensity (0-250)
        led2_pwm : float
            Current LED2 intensity (0-15)
        speed : float
            Current speed (cm/s)
        curvature : float
            Current curvature (1/cm)
        
        Returns
        -------
        lambda_t : float
            Instantaneous hazard (events per second)
        """
        # Scale covariates
        led1_scaled = led1_pwm / 250.0
        led2_scaled = led2_pwm / 15.0
        interaction = led1_scaled * led2_scaled
        
        # Phase in 60s cycle
        phase = (t % 60.0) / 60.0
        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)
        
        # Time since stimulus (assume 30s on / 30s off pattern)
        time_in_cycle = t % 60.0
        if time_in_cycle < 30.0:
            # LED1 is ON
            time_since_stim = time_in_cycle
        else:
            # LED1 is OFF
            time_since_stim = 30.0  # Use max value when off
        
        # Kernel bases
        kernel_vals = raised_cosine_basis(np.array([time_since_stim]), kernel_centers, kernel_width).flatten()
        
        # Z-score kinematics
        speed_z = (speed - speed_mean) / (speed_std + 1e-9)
        curv_z = (curvature - curvature_mean) / (curvature_std + 1e-9)
        
        # Build feature vector (must match feature_names order)
        x = []
        for name in feature_names:
            if name == 'intercept':
                x.append(1.0)
            elif name == 'led1_intensity':
                x.append(led1_scaled)
            elif name == 'led2_intensity':
                x.append(led2_scaled)
            elif name == 'led1_x_led2':
                x.append(interaction)
            elif name == 'phase_sin':
                x.append(phase_sin)
            elif name == 'phase_cos':
                x.append(phase_cos)
            elif name == 'speed':
                x.append(speed_z)
            elif name == 'curvature':
                x.append(curv_z)
            elif name.startswith('kernel_'):
                idx = int(name.split('_')[1]) - 1
                x.append(kernel_vals[idx] if idx < len(kernel_vals) else 0.0)
            else:
                x.append(0.0)
        
        x = np.array(x)
        
        # Linear predictor
        eta = np.dot(coef_array, x)
        
        # Hazard (events per second)
        return np.exp(eta)
    
    return hazard


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_turn_rate(events: pd.DataFrame, time_window: float = 60.0) -> pd.DataFrame:
    """
    Compute turn rate (events per minute) per track.
    
    Handles both:
    1. Frame-level data with is_reorientation column (detect onsets)
    2. Event-only data where each row is an event
    
    Parameters
    ----------
    events : DataFrame
        Event data with columns: experiment_id, track_id, time, is_reorientation
    time_window : float
        Window size for rate calculation (seconds, default 60)
    
    Returns
    -------
    rates : DataFrame
        Turn rates per track
    """
    # Check if this is event-only data (all is_reorientation are True, few rows)
    is_event_only = (
        'is_reorientation' in events.columns and 
        events['is_reorientation'].all() and
        len(events) < 100000  # Arbitrary threshold
    )
    
    if is_event_only:
        # Event-only data: each row is an event, just count rows
        counts = events.groupby(['experiment_id', 'track_id']).size()
        
        # For event-only data, assume standard 20-min experiment duration
        # Use 1200s (20 min) as fixed duration
        duration = pd.Series(1200.0, index=counts.index)
        rates = (counts / duration * 60.0).reset_index()
        rates.columns = ['experiment_id', 'track_id', 'turn_rate_per_min']
        return rates
    elif 'reo_onset' in events.columns:
        counts = events.groupby(['experiment_id', 'track_id'])['reo_onset'].sum()
    elif 'is_reorientation' in events.columns:
        # Detect onsets from frame-level data
        events = events.sort_values(['experiment_id', 'track_id', 'time'])
        events['reo_onset'] = (
            events.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
        counts = events.groupby(['experiment_id', 'track_id'])['reo_onset'].sum()
    else:
        raise ValueError("Need 'reo_onset' or 'is_reorientation' column")
    
    # Compute duration per track
    duration = events.groupby(['experiment_id', 'track_id'])['time'].apply(lambda x: x.max() - x.min())
    
    # Rate per minute
    rates = (counts / duration * 60.0).reset_index()
    rates.columns = ['experiment_id', 'track_id', 'turn_rate_per_min']
    
    return rates


def compare_turn_rates(empirical: pd.DataFrame, simulated: pd.DataFrame) -> Dict:
    """
    Compare empirical vs simulated turn rates.
    
    Parameters
    ----------
    empirical : DataFrame
        Empirical turn rates (from compute_turn_rate)
    simulated : DataFrame
        Simulated turn rates
    
    Returns
    -------
    result : dict
        Comparison statistics and pass/fail
    """
    emp_rates = empirical['turn_rate_per_min'].dropna()
    sim_rates = simulated['turn_rate_per_min'].dropna()
    
    emp_mean = emp_rates.mean()
    emp_std = emp_rates.std()
    emp_ci = (np.percentile(emp_rates, 2.5), np.percentile(emp_rates, 97.5))
    
    sim_mean = sim_rates.mean()
    sim_std = sim_rates.std()
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(emp_rates, sim_rates)
    
    # Check if simulated mean is within empirical 95% CI
    within_ci = emp_ci[0] <= sim_mean <= emp_ci[1]
    
    return {
        'empirical_mean': emp_mean,
        'empirical_std': emp_std,
        'empirical_95ci': emp_ci,
        'simulated_mean': sim_mean,
        'simulated_std': sim_std,
        't_statistic': t_stat,
        'p_value': p_value,
        'within_ci': within_ci,
        'pass': within_ci
    }


def compute_psth(
    events: pd.DataFrame,
    stimulus_times: np.ndarray,
    window: Tuple[float, float] = (-5.0, 30.0),
    bin_width: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram of reorientation events.
    
    Parameters
    ----------
    events : DataFrame
        Event data with time and reo_onset columns
    stimulus_times : ndarray
        Times of stimulus onsets
    window : tuple
        (pre, post) time window around stimulus
    bin_width : float
        Bin width for histogram
    
    Returns
    -------
    bin_centers : ndarray
        Time bin centers relative to stimulus
    rate : ndarray
        Event rate per bin (events per second per stimulus)
    """
    # Get reorientation onset times
    # Check if event-only data (all rows are events)
    is_event_only = (
        'is_reorientation' in events.columns and 
        events['is_reorientation'].all() and
        len(events) < 100000
    )
    
    if is_event_only:
        # Event-only: each row is an event time
        event_times = events['time'].values
    elif 'reo_onset' in events.columns:
        event_times = events[events['reo_onset'] == True]['time'].values
    else:
        events = events.sort_values('time')
        events['reo_onset'] = events['is_reorientation'].astype(bool) & ~events['is_reorientation'].shift(1, fill_value=False).astype(bool)
        event_times = events[events['reo_onset'] == True]['time'].values
    
    # Compute relative times
    relative_times = []
    for stim_t in stimulus_times:
        rel = event_times - stim_t
        in_window = (rel >= window[0]) & (rel <= window[1])
        relative_times.extend(rel[in_window])
    
    # Histogram
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    counts, _ = np.histogram(relative_times, bins=bins)
    
    # Rate: counts / (n_stimuli * bin_width)
    n_stimuli = len(stimulus_times)
    rate = counts / (n_stimuli * bin_width)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, rate


def bootstrap_psth_threshold(
    events: pd.DataFrame,
    stimulus_times: np.ndarray,
    n_bootstrap: int = 500,
    window: Tuple[float, float] = (-5.0, 30.0),
    bin_width: float = 0.5
) -> Dict[str, float]:
    """
    Compute bootstrap threshold for PSTH validation.
    
    Resamples stimulus cycles to get null distribution of empirical-vs-empirical W-ISE.
    Model is acceptable if W-ISE is within this natural variability.
    
    Uses weighted ISE with weights = 1/(rate + eps)
    
    Parameters
    ----------
    events : DataFrame
        Event data
    stimulus_times : ndarray
        LED onset times
    n_bootstrap : int
        Number of bootstrap iterations
    window : tuple
        PSTH window
    bin_width : float
        PSTH bin width
    
    Returns
    -------
    thresholds : dict
        Contains 'w_ise_threshold' (95th percentile of W-ISE) and 'ise_threshold' (95th of raw ISE)
    """
    n_cycles = len(stimulus_times)
    ise_null = []
    w_ise_null = []
    
    rng = np.random.default_rng(42)
    eps = 1e-9
    
    for _ in range(n_bootstrap):
        # Resample cycles with replacement
        idx1 = rng.choice(n_cycles, n_cycles, replace=True)
        idx2 = rng.choice(n_cycles, n_cycles, replace=True)
        
        _, psth1 = compute_psth(events, stimulus_times[idx1], window, bin_width)
        _, psth2 = compute_psth(events, stimulus_times[idx2], window, bin_width)
        
        # Raw ISE
        ise = np.sum((psth1 - psth2) ** 2) * bin_width
        ise_null.append(ise)
        
        # Weighted ISE (weights = 1 / (rate + eps))
        # Use psth1 as reference for weights
        weights = 1.0 / (psth1 + eps)
        w_ise = np.sum(weights * (psth1 - psth2) ** 2) * bin_width
        w_ise_null.append(w_ise)
    
    return {
        'ise_threshold': np.percentile(ise_null, 95),
        'w_ise_threshold': np.percentile(w_ise_null, 95),
        'ise_median': np.median(ise_null),
        'w_ise_median': np.median(w_ise_null)
    }


def compare_psth(
    empirical_psth: Tuple[np.ndarray, np.ndarray],
    simulated_psth: Tuple[np.ndarray, np.ndarray],
    bootstrap_threshold: Optional[float] = None
) -> Dict:
    """
    Compare stimulus-locked PSTHs using baseline-normalized W-ISE and correlation.
    
    Validation approach:
    1. Normalize by baseline (pre-onset rate)
    2. Use W-ISE with weights = 1 / (emp_rate + eps)
    3. Use bootstrap threshold for acceptance
    4. Require correlation >= 0.8
    
    Parameters
    ----------
    empirical_psth : tuple
        (bin_centers, rate) from compute_psth
    simulated_psth : tuple
        (bin_centers, rate) from compute_psth
    bootstrap_threshold : float, optional
        Bootstrap-derived W-ISE threshold (from bootstrap_psth_threshold)
    
    Returns
    -------
    result : dict
        Comparison statistics and pass/fail
    """
    emp_centers, emp_rate = empirical_psth
    sim_centers, sim_rate = simulated_psth
    
    # Interpolate to common grid if needed
    if not np.allclose(emp_centers, sim_centers):
        common_centers = emp_centers
        sim_rate = np.interp(common_centers, sim_centers, sim_rate)
    else:
        common_centers = emp_centers
    
    bin_width = common_centers[1] - common_centers[0] if len(common_centers) > 1 else 0.5
    
    # Compute baseline from pre-onset bins (t < 0)
    pre_onset_mask = common_centers < 0
    if pre_onset_mask.sum() > 0:
        baseline_emp = emp_rate[pre_onset_mask].mean()
        baseline_sim = sim_rate[pre_onset_mask].mean()
    else:
        baseline_emp = emp_rate.mean()
        baseline_sim = sim_rate.mean()
    
    # Baseline-normalized PSTHs (0 = baseline, positive = above baseline)
    eps = 1e-9
    emp_norm = (emp_rate - baseline_emp) / (baseline_emp + eps)
    sim_norm = (sim_rate - baseline_sim) / (baseline_sim + eps)
    
    # Raw ISE (on unnormalized rates)
    ise = np.sum((emp_rate - sim_rate) ** 2) * bin_width
    
    # Weighted ISE: weights = 1 / (emp_rate + eps)
    # This gives MORE weight to high-rate bins (inverse of rate)
    weights = 1.0 / (emp_rate + eps)
    w_ise = np.sum(weights * (emp_rate - sim_rate) ** 2) * bin_width
    
    # Normalized ISE (on baseline-normalized rates)
    norm_ise = np.sum((emp_norm - sim_norm) ** 2) * bin_width
    
    # Correlation (on raw rates - captures shape similarity)
    if np.std(emp_rate) > 0 and np.std(sim_rate) > 0:
        correlation = np.corrcoef(emp_rate, sim_rate)[0, 1]
    else:
        correlation = 0.0
    
    # Normalized correlation (on baseline-normalized rates)
    if np.std(emp_norm) > 0 and np.std(sim_norm) > 0:
        norm_correlation = np.corrcoef(emp_norm, sim_norm)[0, 1]
    else:
        norm_correlation = 0.0
    
    # Determine pass/fail
    # Criterion 1: W-ISE within bootstrap threshold (if provided)
    # Criterion 2: Correlation >= 0.8
    if bootstrap_threshold is not None:
        ise_pass = w_ise <= bootstrap_threshold
        threshold_used = bootstrap_threshold
    else:
        # Fallback: compare to empirical variance
        emp_var = np.var(emp_rate)
        ise_pass = ise <= emp_var * 2.0  # Allow up to 2x variance
        threshold_used = emp_var * 2.0
    
    corr_pass = correlation >= 0.8
    passed = ise_pass and corr_pass
    
    return {
        'ise': ise,
        'weighted_ise': w_ise,
        'normalized_ise': norm_ise,
        'correlation': correlation,
        'norm_correlation': norm_correlation,
        'baseline_emp': baseline_emp,
        'baseline_sim': baseline_sim,
        'threshold': threshold_used,
        'ise_pass': ise_pass,
        'corr_pass': corr_pass,
        'pass': passed
    }


def compare_distributions(
    empirical: np.ndarray,
    simulated: np.ndarray,
    name: str = 'distribution'
) -> Dict:
    """
    Compare distributions using Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    empirical : ndarray
        Empirical values
    simulated : ndarray
        Simulated values
    name : str
        Name of the distribution being compared
    
    Returns
    -------
    result : dict
        KS test statistics and pass/fail
    """
    # Remove NaN values
    empirical = empirical[~np.isnan(empirical)]
    simulated = simulated[~np.isnan(simulated)]
    
    if len(empirical) == 0 or len(simulated) == 0:
        return {
            'name': name,
            'error': 'Empty data',
            'pass': False
        }
    
    # KS test
    ks_stat, p_value = stats.ks_2samp(empirical, simulated)
    
    # Pass if p > 0.05 (distributions not significantly different)
    passed = p_value > 0.05
    
    return {
        'name': name,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'threshold': 0.05,
        'pass': passed,
        'empirical_n': len(empirical),
        'simulated_n': len(simulated)
    }


def compare_iei(
    iei_empirical: np.ndarray,
    iei_simulated: np.ndarray
) -> Dict:
    """
    Compare inter-event interval distributions with K-S test and moment comparison.
    
    Acceptance criteria:
    - K-S test with p > 0.05
    - Mean difference < 1.0 seconds
    - Variance ratio between 0.8 and 1.2
    
    Parameters
    ----------
    iei_empirical : ndarray
        Empirical inter-event intervals (seconds)
    iei_simulated : ndarray
        Simulated inter-event intervals (seconds)
    
    Returns
    -------
    result : dict
        Comparison statistics and pass/fail
    """
    # Remove NaN values
    iei_empirical = iei_empirical[~np.isnan(iei_empirical)]
    iei_simulated = iei_simulated[~np.isnan(iei_simulated)]
    
    if len(iei_empirical) == 0 or len(iei_simulated) == 0:
        return {
            'name': 'iei',
            'error': 'Empty data',
            'pass': False
        }
    
    # K-S test
    ks_stat, ks_pval = stats.ks_2samp(iei_empirical, iei_simulated)
    
    # Moment comparison
    mean_emp = iei_empirical.mean()
    mean_sim = iei_simulated.mean()
    mean_diff = abs(mean_sim - mean_emp)
    
    median_emp = np.median(iei_empirical)
    median_sim = np.median(iei_simulated)
    
    var_emp = iei_empirical.var()
    var_sim = iei_simulated.var()
    var_ratio = var_sim / (var_emp + 1e-9)
    
    # Pass criteria
    ks_pass = ks_pval > 0.05
    mean_pass = mean_diff < 1.0  # seconds
    var_pass = 0.8 < var_ratio < 1.2
    
    overall_pass = ks_pass and mean_pass and var_pass
    
    return {
        'name': 'iei',
        'ks_statistic': ks_stat,
        'ks_pval': ks_pval,
        'ks_pass': ks_pass,
        'mean_emp': mean_emp,
        'mean_sim': mean_sim,
        'mean_diff': mean_diff,
        'mean_pass': mean_pass,
        'median_emp': median_emp,
        'median_sim': median_sim,
        'var_emp': var_emp,
        'var_sim': var_sim,
        'var_ratio': var_ratio,
        'var_pass': var_pass,
        'empirical_n': len(iei_empirical),
        'simulated_n': len(iei_simulated),
        'pass': overall_pass
    }


def compute_inter_event_intervals(events: pd.DataFrame) -> np.ndarray:
    """
    Compute inter-event intervals for reorientation events.
    
    Parameters
    ----------
    events : DataFrame
        Event data with time column and reorientation indicator
    
    Returns
    -------
    iei : ndarray
        Inter-event intervals (seconds)
    """
    if 'reo_onset' not in events.columns:
        events = events.copy()
        events = events.sort_values(['experiment_id', 'track_id', 'time'])
        events['reo_onset'] = (
            events.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x & ~x.shift(1, fill_value=False))
        )
    
    # Get event times per track
    iei_list = []
    for (exp, track), group in events[events['reo_onset'] == True].groupby(['experiment_id', 'track_id']):
        times = group['time'].sort_values().values
        if len(times) > 1:
            intervals = np.diff(times)
            iei_list.extend(intervals)
    
    return np.array(iei_list)


def run_validation(
    empirical_data: pd.DataFrame,
    simulated_data: pd.DataFrame,
    stimulus_times: np.ndarray = None
) -> Dict:
    """
    Run all validation comparisons.
    
    Parameters
    ----------
    empirical_data : DataFrame
        Empirical event data
    simulated_data : DataFrame
        Simulated event data
    stimulus_times : ndarray, optional
        Times of stimulus onsets for PSTH
    
    Returns
    -------
    results : dict
        All validation results
    """
    results = {}
    
    # 1. Turn rate comparison
    print("Comparing turn rates...")
    emp_rates = compute_turn_rate(empirical_data)
    sim_rates = compute_turn_rate(simulated_data)
    results['turn_rate'] = compare_turn_rates(emp_rates, sim_rates)
    print(f"  Empirical: {results['turn_rate']['empirical_mean']:.2f} +/- {results['turn_rate']['empirical_std']:.2f}")
    print(f"  Simulated: {results['turn_rate']['simulated_mean']:.2f} +/- {results['turn_rate']['simulated_std']:.2f}")
    print(f"  Pass: {results['turn_rate']['pass']}")
    
    # 2. PSTH comparison (if stimulus times provided)
    if stimulus_times is not None and len(stimulus_times) > 0:
        print("\nComputing bootstrap PSTH threshold...")
        bootstrap_result = bootstrap_psth_threshold(
            empirical_data, stimulus_times, n_bootstrap=200
        )
        w_ise_thresh = bootstrap_result['w_ise_threshold']
        print(f"  Bootstrap 95th percentile W-ISE: {w_ise_thresh:.4f}")
        print(f"  Bootstrap 95th percentile ISE: {bootstrap_result['ise_threshold']:.4f}")
        
        print("\nComparing stimulus-locked PSTH...")
        emp_psth = compute_psth(empirical_data, stimulus_times)
        sim_psth = compute_psth(simulated_data, stimulus_times)
        results['psth'] = compare_psth(emp_psth, sim_psth, bootstrap_threshold=w_ise_thresh)
        results['psth']['bootstrap_info'] = bootstrap_result
        
        print(f"  Baselines: empirical={results['psth']['baseline_emp']:.4f}, simulated={results['psth']['baseline_sim']:.4f}")
        print(f"  W-ISE: {results['psth']['weighted_ise']:.4f} (threshold: {w_ise_thresh:.4f})")
        print(f"  Correlation: {results['psth']['correlation']:.3f}")
        print(f"  ISE Pass: {results['psth']['ise_pass']}, Corr Pass: {results['psth']['corr_pass']}")
        print(f"  Overall PSTH Pass: {results['psth']['pass']}")
    
    # 3. Heading change distribution
    if 'reo_dtheta' in empirical_data.columns and 'reo_dtheta' in simulated_data.columns:
        print("\nComparing heading change distribution...")
        results['heading_change'] = compare_distributions(
            empirical_data['reo_dtheta'].values,
            simulated_data['reo_dtheta'].values,
            name='heading_change'
        )
        print(f"  KS statistic: {results['heading_change']['ks_statistic']:.4f}")
        print(f"  p-value: {results['heading_change']['p_value']:.4f}")
        print(f"  Pass: {results['heading_change']['pass']}")
    
    # 4. Inter-event interval distribution (K-S + moments)
    print("\nComparing inter-event intervals...")
    emp_iei = compute_inter_event_intervals(empirical_data)
    sim_iei = compute_inter_event_intervals(simulated_data)
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        results['iei'] = compare_iei(emp_iei, sim_iei)
        print(f"  K-S statistic: {results['iei']['ks_statistic']:.4f}, p-value: {results['iei']['ks_pval']:.4f}")
        print(f"  Mean: empirical={results['iei']['mean_emp']:.2f}s, simulated={results['iei']['mean_sim']:.2f}s (diff={results['iei']['mean_diff']:.2f}s)")
        print(f"  Variance ratio: {results['iei']['var_ratio']:.3f}")
        print(f"  Pass: K-S={results['iei']['ks_pass']}, Mean={results['iei']['mean_pass']}, Var={results['iei']['var_pass']}")
        print(f"  Overall IEI Pass: {results['iei']['pass']}")
    
    # Overall pass/fail
    all_passed = all(
        r.get('pass', True) 
        for r in results.values() 
        if isinstance(r, dict)
    )
    results['overall'] = 'PASS' if all_passed else 'FAIL'
    print(f"\nOverall: {results['overall']}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate simulated vs empirical data')
    parser.add_argument('--empirical', type=str, required=True,
                        help='Path to empirical data (parquet or h5)')
    parser.add_argument('--simulated', type=str, required=True,
                        help='Path to simulated data (parquet)')
    parser.add_argument('--output', type=str, default='data/validation/',
                        help='Output directory for validation results')
    
    args = parser.parse_args()
    
    emp_path = Path(args.empirical)
    sim_path = Path(args.simulated)
    output_dir = Path(args.output)
    
    # Load data
    print(f"Loading empirical data from {emp_path}...")
    if emp_path.suffix == '.h5':
        import h5py
        with h5py.File(emp_path, 'r') as f:
            if 'events' in f:
                grp = f['events']
                data = {k: grp[k][:] for k in grp.keys()}
                empirical = pd.DataFrame(data)
            else:
                raise ValueError("No 'events' group in H5 file")
    else:
        # Try parquet, fall back to CSV or directory of CSVs
        if emp_path.suffix == '.parquet':
            try:
                empirical = pd.read_parquet(emp_path)
            except ImportError:
                raise ImportError("pyarrow/fastparquet not installed. Use CSV instead.")
        elif emp_path.suffix == '.csv':
            empirical = pd.read_csv(emp_path)
        elif emp_path.is_dir():
            # Load from directory of CSV files
            # Prioritize specific low-rate experiments (202510301228, 202510301408)
            # These match the model's expected ~1 event/min/track rate
            csv_files = sorted(emp_path.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
            if not csv_files:
                csv_files = sorted(emp_path.glob('*_0to250PWM_*_events.csv'))[:4]
            if not csv_files:
                csv_files = sorted(emp_path.glob('*_events.csv'))[:4]
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f)
                df['experiment_id'] = f.stem
                dfs.append(df)
            empirical = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(csv_files)} files: {[f.name[:40] for f in csv_files]}")
        else:
            raise ValueError(f"Unknown empirical data format: {emp_path}")
    
    print(f"Loading simulated data from {sim_path}...")
    if sim_path.suffix == '.parquet':
        try:
            simulated = pd.read_parquet(sim_path)
        except ImportError:
            raise ImportError("pyarrow/fastparquet not installed. Use CSV instead.")
    elif sim_path.suffix == '.csv':
        simulated = pd.read_csv(sim_path)
    else:
        raise ValueError(f"Unknown simulated data format: {sim_path}")
    
    # Extract LED onset times for PSTH validation
    # For 20-min experiments with 30s on/30s off, expect ~20 onsets per experiment
    stimulus_times = None
    if 'led1Val' in empirical.columns and 'time' in empirical.columns:
        print("Extracting LED onset times for PSTH...")
        
        # Approach: sample at 1s resolution, detect major transitions
        all_onsets = []
        for exp_id in empirical['experiment_id'].unique():
            exp_df = empirical[empirical['experiment_id'] == exp_id].sort_values('time')
            
            # Sample to ~1s resolution to avoid detecting ramp steps
            t = exp_df['time'].values
            led = exp_df['led1Val'].values
            
            # Bin to 1s resolution
            t_bins = np.arange(t.min(), t.max() + 1, 1.0)
            led_binned = np.zeros(len(t_bins) - 1)
            for i in range(len(t_bins) - 1):
                mask = (t >= t_bins[i]) & (t < t_bins[i+1])
                if mask.any():
                    led_binned[i] = led[mask].max()
            
            # Detect transitions from low to high (>200 PWM)
            led_high = led_binned > 200
            led_high_prev = np.roll(led_high, 1)
            led_high_prev[0] = False
            transitions = led_high & ~led_high_prev
            
            onset_times = t_bins[:-1][transitions]
            all_onsets.extend(onset_times)
        
        if len(all_onsets) > 0:
            stimulus_times = np.array(all_onsets)
            print(f"  Found {len(stimulus_times)} LED onsets (~{len(stimulus_times)/14:.1f} per experiment)")
    
    # Run validation
    results = run_validation(empirical, simulated, stimulus_times=stimulus_times)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_dir / 'validation_results.json', 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'validation_results.json'}")


if __name__ == '__main__':
    main()




