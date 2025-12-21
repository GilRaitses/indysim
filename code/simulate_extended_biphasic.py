#!/usr/bin/env python3
"""
Simulate Events Using Extended Biphasic LNP Model

Uses the model with:
- Late kernel: centers at [3, 5, 7, 9]s (shifted for correct timing)
- Early kernel: narrower bases at [0.2, 0.7, 1.4]s
- No LED main effect

Usage:
    python scripts/simulate_extended_biphasic.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Callable, List
from scipy.stats import pearsonr, ks_2samp

# Parameters matching empirical data
LED_ON_DURATION = 10.0  # seconds
LED_OFF_DURATION = 20.0  # seconds
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3  # seconds (from empirical data)
EXPERIMENT_DURATION = 1200.0  # 20 minutes


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


def make_hazard_from_triphasic_model(model_results: dict, kernel_config: dict) -> Callable:
    """
    Create hazard function from triphasic model (early + intermediate + late).
    
    Parameters
    ----------
    model_results : dict
        Model coefficients
    kernel_config : dict
        Kernel configuration (centers, widths)
    
    Returns
    -------
    hazard_func : callable
        Function that takes time (seconds) and returns hazard rate
    """
    coefficients = model_results['coefficients']
    
    intercept = coefficients.get('intercept', -6.5)
    
    # Get kernel coefficients by type
    early_coefs = []
    intm_coefs = []
    late_coefs = []
    for name, val in coefficients.items():
        if 'kernel_early' in name:
            early_coefs.append(val)
        elif 'kernel_intm' in name:
            intm_coefs.append(val)
        elif 'kernel_late' in name:
            late_coefs.append(val)
    
    early_coefs = np.array(early_coefs)
    intm_coefs = np.array(intm_coefs) if intm_coefs else np.array([])
    late_coefs = np.array(late_coefs)
    
    # Get kernel configuration
    early_centers = np.array(kernel_config.get('early_centers', [0.2, 0.7, 1.4]))
    intm_centers = np.array(kernel_config.get('intm_centers', [2.0, 2.5]))
    late_centers = np.array(kernel_config.get('late_centers', [3.0, 5.0, 7.0, 9.0]))
    early_width = kernel_config.get('early_width', 0.4)
    intm_width = kernel_config.get('intm_width', 0.6)
    late_width = kernel_config.get('late_width', 1.8)
    
    rebound_coef = coefficients.get('led_off_rebound', 0.0)
    rebound_tau = kernel_config.get('rebound_tau', 2.0)
    
    def hazard_func(t: float) -> float:
        """Compute hazard rate at time t."""
        # Determine LED state and time since onset
        if t < FIRST_LED_ONSET:
            # Before first LED onset - baseline only
            return np.exp(intercept)
        
        time_in_cycle = (t - FIRST_LED_ONSET) % LED_CYCLE
        
        if time_in_cycle < LED_ON_DURATION:
            # LED is ON
            time_since_onset = time_in_cycle
            
            # Compute kernel contribution from all three phases
            tso = np.array([time_since_onset])
            early_basis = raised_cosine_basis(tso, early_centers, early_width)
            intm_basis = raised_cosine_basis(tso, intm_centers, intm_width) if len(intm_coefs) > 0 else np.zeros((1, 0))
            late_basis = raised_cosine_basis(tso, late_centers, late_width)
            
            kernel_contrib = 0.0
            for j, c in enumerate(early_coefs):
                kernel_contrib += c * early_basis[0, j]
            for j, c in enumerate(intm_coefs):
                kernel_contrib += c * intm_basis[0, j]
            for j, c in enumerate(late_coefs):
                kernel_contrib += c * late_basis[0, j]
            
            eta = intercept + kernel_contrib
            return np.exp(eta)
        else:
            # LED is OFF - compute rebound if applicable
            time_since_offset = time_in_cycle - LED_ON_DURATION
            rebound = np.exp(-time_since_offset / rebound_tau)
            eta = intercept + rebound_coef * rebound
            return np.exp(eta)
    
    return hazard_func


def generate_events_discrete(
    hazard_func: Callable,
    t_start: float,
    t_end: float,
    dt: float = 0.05,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate events using discrete-time Bernoulli simulation.
    
    This method matches the GLM exactly:
    - GLM is fit on frame-wise binary events (0/1 per 50ms frame)
    - exp(eta) is the per-frame event probability
    - We draw Bernoulli(p = exp(eta)) for each frame
    
    FIX (2025-12-11): Removed double-scaling issue. Previously we:
    1. Treated exp(eta) as per-frame probability
    2. Multiplied by frame_rate to get per-second
    3. Integrated with dt
    This caused a 1.77x rate overestimate.
    
    Now we directly use exp(eta) as per-frame probability in Bernoulli draws.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    events = []
    t = t_start
    
    while t < t_end:
        # Get per-frame event probability from hazard function
        p = hazard_func(t)
        # Clip to valid probability range
        p = np.clip(p, 0, 1)
        
        # Draw Bernoulli event
        if rng.random() < p:
            events.append(t)
        
        t += dt
    
    return np.array(events)


def simulate_tracks(
    hazard_func: Callable,
    n_tracks: int = 99,
    duration: float = EXPERIMENT_DURATION,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate events for multiple tracks using discrete-time Bernoulli simulation.
    
    FIX (2025-12-11): Now uses discrete-time simulation that matches the GLM exactly.
    """
    rng = np.random.default_rng(seed)
    
    all_events = []
    for track_id in range(n_tracks):
        track_seed = rng.integers(0, 2**32)
        track_rng = np.random.default_rng(track_seed)
        
        events = generate_events_discrete(
            hazard_func,
            t_start=0,
            t_end=duration,
            dt=0.05,
            rng=track_rng
        )
        
        for ev_time in events:
            all_events.append({
                'track_id': track_id,
                'time': ev_time,
                'experiment_id': 'simulated'
            })
    
    return pd.DataFrame(all_events)


def compute_psth(events: np.ndarray, n_tracks: int = 99, bin_size: float = 0.2, window: tuple = (-3.0, 8.0)) -> tuple:
    """
    Compute PSTH relative to LED onsets.
    
    FIX (2025-12-11): Now divides by n_tracks to get per-track rate.
    This makes empirical and simulated PSTHs directly comparable.
    
    Parameters
    ----------
    events : ndarray
        Event times (seconds)
    n_tracks : int
        Number of tracks (for per-track normalization)
    bin_size : float
        PSTH bin size (seconds)
    window : tuple
        (start, end) relative to LED onset (seconds)
    
    Returns
    -------
    bin_centers : ndarray
        Time points (seconds relative to LED onset)
    rates : ndarray
        Event rate per track per second
    """
    # LED onset times
    led_onsets = np.arange(FIRST_LED_ONSET, EXPERIMENT_DURATION, LED_CYCLE)
    
    # Bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count events in each bin for each onset
    counts = np.zeros(len(bin_centers))
    n_onsets = 0
    
    for onset in led_onsets:
        for i, bc in enumerate(bin_centers):
            t_start = onset + bc - bin_size / 2
            t_end = onset + bc + bin_size / 2
            counts[i] += np.sum((events >= t_start) & (events < t_end))
        n_onsets += 1
    
    # Convert to rate per track (events per second per track)
    # FIX: Divide by n_tracks to get per-track rate
    rates = counts / (n_onsets * bin_size * n_tracks)
    
    return bin_centers, rates


def compute_wise(psth1: np.ndarray, psth2: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute Weighted Integrated Squared Error between two PSTHs.
    
    W-ISE = sum(w_i * (psth1_i - psth2_i)^2) / sum(w_i)
    
    Parameters
    ----------
    psth1, psth2 : ndarray
        PSTH values to compare
    weights : ndarray, optional
        Weights for each bin (default: 1/variance for empirical weighting)
    
    Returns
    -------
    wise : float
        Weighted integrated squared error
    """
    if weights is None:
        # Default: equal weights
        weights = np.ones(len(psth1))
    
    # Handle zero weights
    weights = np.maximum(weights, 1e-10)
    
    # Compute W-ISE
    sq_diff = (psth1 - psth2) ** 2
    wise = np.sum(weights * sq_diff) / np.sum(weights)
    
    return wise


def bootstrap_psth_threshold(
    events: np.ndarray,
    n_tracks: int,
    n_bootstrap: int = 100,
    bin_size: float = 0.2,
    window: tuple = (-3.0, 8.0),
    rng_seed: int = 42
) -> dict:
    """
    Compute bootstrap-derived threshold for PSTH comparison.
    
    Splits empirical data into halves and computes W-ISE and correlation
    between them to establish a "self-consistency" baseline.
    
    Parameters
    ----------
    events : ndarray
        All empirical event times
    n_tracks : int
        Number of tracks
    n_bootstrap : int
        Number of bootstrap iterations
    bin_size : float
        PSTH bin size
    window : tuple
        PSTH window relative to LED onset
    rng_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    thresholds : dict
        'wise_95': 95th percentile of W-ISE (upper threshold)
        'corr_05': 5th percentile of correlation (lower threshold)
        'wise_values': all bootstrap W-ISE values
        'corr_values': all bootstrap correlation values
    """
    rng = np.random.default_rng(rng_seed)
    
    # LED onset times
    led_onsets = np.arange(FIRST_LED_ONSET, EXPERIMENT_DURATION, LED_CYCLE)
    n_onsets = len(led_onsets)
    
    wise_values = []
    corr_values = []
    
    for _ in range(n_bootstrap):
        # Randomly split onsets into two halves
        perm = rng.permutation(n_onsets)
        half1_idx = perm[:n_onsets // 2]
        half2_idx = perm[n_onsets // 2:]
        
        half1_onsets = led_onsets[half1_idx]
        half2_onsets = led_onsets[half2_idx]
        
        # Compute PSTH for each half
        bins = np.arange(window[0], window[1] + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        counts1 = np.zeros(len(bin_centers))
        counts2 = np.zeros(len(bin_centers))
        
        for onset in half1_onsets:
            for i, bc in enumerate(bin_centers):
                t_start = onset + bc - bin_size / 2
                t_end = onset + bc + bin_size / 2
                counts1[i] += np.sum((events >= t_start) & (events < t_end))
        
        for onset in half2_onsets:
            for i, bc in enumerate(bin_centers):
                t_start = onset + bc - bin_size / 2
                t_end = onset + bc + bin_size / 2
                counts2[i] += np.sum((events >= t_start) & (events < t_end))
        
        # Normalize to rates
        rates1 = counts1 / (len(half1_onsets) * bin_size * n_tracks)
        rates2 = counts2 / (len(half2_onsets) * bin_size * n_tracks)
        
        # Normalize by baseline
        baseline1 = np.mean(rates1[bin_centers < 0])
        baseline2 = np.mean(rates2[bin_centers < 0])
        
        if baseline1 > 0 and baseline2 > 0:
            norm_rates1 = rates1 / baseline1
            norm_rates2 = rates2 / baseline2
            
            # Compute W-ISE and correlation
            wise = compute_wise(norm_rates1, norm_rates2)
            corr, _ = pearsonr(norm_rates1, norm_rates2)
            
            wise_values.append(wise)
            corr_values.append(corr)
    
    wise_values = np.array(wise_values)
    corr_values = np.array(corr_values)
    
    return {
        'wise_95': np.percentile(wise_values, 95),
        'wise_mean': np.mean(wise_values),
        'wise_std': np.std(wise_values),
        'corr_05': np.percentile(corr_values, 5),
        'corr_mean': np.mean(corr_values),
        'corr_std': np.std(corr_values),
        'wise_values': wise_values,
        'corr_values': corr_values
    }


def load_empirical_events(data_dir: Path) -> np.ndarray:
    """Load empirical event times."""
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        csv_files = sorted(data_dir.glob('*_events.csv'))[:2]
    
    all_events = []
    for f in csv_files:
        df = pd.read_csv(f)
        if 'is_reorientation_start' in df.columns:
            event_times = df.loc[df['is_reorientation_start'] == 1, 'time'].values
        elif 'is_reorientation' in df.columns:
            event_times = df.loc[df['is_reorientation'] == 1, 'time'].values
        else:
            continue
        all_events.extend(event_times)
    
    return np.array(all_events)


def main():
    print("=" * 70)
    print("SIMULATE WITH TRIPHASIC MODEL")
    print("=" * 70)
    
    # Load model
    model_path = Path('data/model/extended_biphasic_model_results.json')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    with open(model_path) as f:
        model_results = f.read()
        model_results = json.loads(model_results)
    
    print("\nModel coefficients:")
    for name, val in model_results['coefficients'].items():
        print(f"  {name}: {val:+.4f}")
    
    kernel_config = model_results.get('kernel_config', {
        'early_centers': [0.2, 0.7, 1.4],
        'intm_centers': [2.0, 2.5],
        'late_centers': [3.0, 5.0, 7.0, 9.0],
        'early_width': 0.4,
        'intm_width': 0.6,
        'late_width': 1.8,
        'rebound_tau': 2.0
    })
    
    print("\nKernel configuration:")
    for k, v in kernel_config.items():
        print(f"  {k}: {v}")
    
    # Create hazard function
    hazard_func = make_hazard_from_triphasic_model(model_results, kernel_config)
    
    # Debug hazard at key times
    print("\nHazard function debug:")
    for tso in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        t = FIRST_LED_ONSET + tso
        h = hazard_func(t)
        print(f"  t={tso:.1f}s: hazard = {h:.6f}")
    
    # Post-LED hazard
    print("\n  Post-LED (t=15s into cycle):")
    t = FIRST_LED_ONSET + 15.0
    print(f"  hazard = {hazard_func(t):.6f}")
    
    # Simulate
    print("\nSimulating 99 tracks...")
    sim_df = simulate_tracks(hazard_func, n_tracks=99, seed=42)
    
    n_events = len(sim_df)
    n_tracks = sim_df['track_id'].nunique()
    rate = n_events / (n_tracks * EXPERIMENT_DURATION / 60)
    
    print(f"  Generated {n_events} events")
    print(f"  Rate: {rate:.2f} events/min/track")
    
    # Load empirical events for comparison
    data_dir = Path('data/engineered')
    emp_events = load_empirical_events(data_dir)
    print(f"\nEmpirical events: {len(emp_events)}")
    
    # Compute PSTHs with Mirna's window
    # FIX (2025-12-11): Now using per-track normalization
    print("\n=== PSTH WITH MIRNA WINDOW (-3s to +8s) ===")
    print("  (Per-track normalized)")
    
    sim_events = sim_df['time'].values
    window = (-3.0, 8.0)
    n_tracks_val = 99  # Both empirical and simulated have 99 tracks
    
    emp_bins, emp_psth = compute_psth(emp_events, n_tracks=n_tracks_val, window=window)
    sim_bins, sim_psth = compute_psth(sim_events, n_tracks=n_tracks_val, window=window)
    
    # Phase-specific rates
    pre_idx = (emp_bins >= -3.0) & (emp_bins < 0.0)
    early_idx = (emp_bins >= 0.0) & (emp_bins < 3.0)
    late_idx = (emp_bins >= 3.0) & (emp_bins <= 8.0)
    
    print("\nEmpirical PSTH:")
    print(f"  Pre-onset (-3 to 0s): {emp_psth[pre_idx].mean():.3f}")
    print(f"  Early (0 to 3s): {emp_psth[early_idx].mean():.3f}")
    print(f"  Late (3 to 8s): {emp_psth[late_idx].mean():.3f}")
    
    print("\nSimulated PSTH:")
    print(f"  Pre-onset (-3 to 0s): {sim_psth[pre_idx].mean():.3f}")
    print(f"  Early (0 to 3s): {sim_psth[early_idx].mean():.3f}")
    print(f"  Late (3 to 8s): {sim_psth[late_idx].mean():.3f}")
    
    # Overall correlation
    corr, _ = pearsonr(emp_psth, sim_psth)
    print(f"\nOverall PSTH correlation: {corr:.3f}")
    
    # Normalized PSTH
    emp_baseline = emp_psth[pre_idx].mean()
    sim_baseline = sim_psth[pre_idx].mean()
    
    emp_norm = emp_psth / emp_baseline if emp_baseline > 0 else emp_psth
    sim_norm = sim_psth / sim_baseline if sim_baseline > 0 else sim_psth
    
    print("\n=== NORMALIZED PSTH (relative to baseline) ===")
    print("\nEmpirical (normalized):")
    print(f"  Pre-onset: {emp_norm[pre_idx].mean():.3f}")
    print(f"  Early (0-3s): {emp_norm[early_idx].mean():.3f}")
    print(f"  Late (3-8s): {emp_norm[late_idx].mean():.3f}")
    
    print("\nSimulated (normalized):")
    print(f"  Pre-onset: {sim_norm[pre_idx].mean():.3f}")
    print(f"  Early (0-3s): {sim_norm[early_idx].mean():.3f}")
    print(f"  Late (3-8s): {sim_norm[late_idx].mean():.3f}")
    
    norm_corr, _ = pearsonr(emp_norm, sim_norm)
    print(f"\nNormalized PSTH correlation: {norm_corr:.3f}")
    
    # Compute W-ISE
    print("\n=== WEIGHTED INTEGRATED SQUARED ERROR (W-ISE) ===")
    wise = compute_wise(emp_norm, sim_norm)
    print(f"  W-ISE (normalized): {wise:.4f}")
    
    # Bootstrap threshold
    print("\n=== BOOTSTRAP THRESHOLD (empirical self-consistency) ===")
    print("  Computing bootstrap (100 iterations)...")
    bootstrap_results = bootstrap_psth_threshold(
        emp_events, n_tracks=n_tracks_val, n_bootstrap=100, 
        bin_size=0.2, window=window
    )
    
    print(f"\n  Bootstrap W-ISE (95th percentile): {bootstrap_results['wise_95']:.4f}")
    print(f"  Bootstrap W-ISE (mean ± std): {bootstrap_results['wise_mean']:.4f} ± {bootstrap_results['wise_std']:.4f}")
    print(f"  Bootstrap correlation (5th percentile): {bootstrap_results['corr_05']:.3f}")
    print(f"  Bootstrap correlation (mean ± std): {bootstrap_results['corr_mean']:.3f} ± {bootstrap_results['corr_std']:.3f}")
    
    # Compare to thresholds
    wise_pass = wise <= bootstrap_results['wise_95']
    corr_pass = norm_corr >= bootstrap_results['corr_05']
    
    print(f"\n  W-ISE vs threshold: {wise:.4f} vs {bootstrap_results['wise_95']:.4f} -> {'PASS' if wise_pass else 'FAIL'}")
    print(f"  Correlation vs threshold: {norm_corr:.3f} vs {bootstrap_results['corr_05']:.3f} -> {'PASS' if corr_pass else 'FAIL'}")
    
    # Save simulated events
    output_path = Path('data/simulated/extended_biphasic_events.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(output_path, index=False)
    print(f"\nSaved simulated events to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Calculate empirical rate for comparison
    emp_rate = len(emp_events) / (n_tracks_val * EXPERIMENT_DURATION / 60)
    
    print(f"\n| Metric | Empirical | Simulated | Threshold | Status |")
    print(f"|--------|-----------|-----------|-----------|--------|")
    print(f"| Event rate (events/min/track) | {emp_rate:.2f} | {rate:.2f} | - | {'PASS' if abs(rate - emp_rate) / emp_rate < 0.3 else 'WARN'} |")
    print(f"| PSTH correlation | - | {norm_corr:.3f} | >= {bootstrap_results['corr_05']:.3f} | {'PASS' if corr_pass else 'FAIL'} |")
    print(f"| W-ISE | - | {wise:.4f} | <= {bootstrap_results['wise_95']:.4f} | {'PASS' if wise_pass else 'FAIL'} |")
    print(f"| Early suppression | {emp_norm[early_idx].mean():.2f} | {sim_norm[early_idx].mean():.2f} | - | {abs(emp_norm[early_idx].mean() - sim_norm[early_idx].mean()):.0%} diff |")
    print(f"| Late suppression | {emp_norm[late_idx].mean():.2f} | {sim_norm[late_idx].mean():.2f} | - | {abs(emp_norm[late_idx].mean() - sim_norm[late_idx].mean()):.0%} diff |")
    
    # Check if timing is now correct
    print("\nTiming check (normalized, suppression should build over time):")
    print(f"  Empirical:  Pre=1.00 -> Early={emp_norm[early_idx].mean():.2f} -> Late={emp_norm[late_idx].mean():.2f}")
    print(f"  Simulated:  Pre=1.00 -> Early={sim_norm[early_idx].mean():.2f} -> Late={sim_norm[late_idx].mean():.2f}")
    
    emp_builds = emp_norm[late_idx].mean() < emp_norm[early_idx].mean()
    sim_builds = sim_norm[late_idx].mean() < sim_norm[early_idx].mean()
    
    if emp_builds and sim_builds:
        print("  ✓ Both show building suppression (correct pattern)")
    elif emp_builds and not sim_builds:
        print("  ✗ Empirical shows building suppression, simulated does not")
    else:
        print("  ? Pattern comparison inconclusive")
    
    # Save validation results
    validation_results = {
        'event_rate_sim': rate,
        'event_rate_emp': emp_rate,
        'psth_correlation': float(norm_corr),
        'wise': float(wise),
        'bootstrap_wise_95': float(bootstrap_results['wise_95']),
        'bootstrap_corr_05': float(bootstrap_results['corr_05']),
        'early_suppression_emp': float(emp_norm[early_idx].mean()),
        'early_suppression_sim': float(sim_norm[early_idx].mean()),
        'late_suppression_emp': float(emp_norm[late_idx].mean()),
        'late_suppression_sim': float(sim_norm[late_idx].mean()),
        'wise_pass': bool(wise_pass),
        'corr_pass': bool(corr_pass)
    }
    
    validation_path = Path('data/model/validation_results.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nSaved validation results to {validation_path}")


if __name__ == '__main__':
    main()




