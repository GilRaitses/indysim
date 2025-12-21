#!/usr/bin/env python3
"""
Inter-Event Interval (IEI) Analysis

Analyzes the distribution of times between consecutive reorientation events
to check for:
1. Bimodal distribution (short during LED-OFF, long spanning LED-ON)
2. Refractory period signature (sharp drop in very short IEIs)
3. Model fit quality

Usage:
    python scripts/analyze_iei.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_empirical_events(data_dir: Path) -> pd.DataFrame:
    """Load empirical event data."""
    event_files = sorted(data_dir.glob('*_events.csv'))
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in {data_dir}")
    
    all_events = []
    for i, f in enumerate(event_files):
        df = pd.read_csv(f)
        track_id = i + 1
        
        if 'is_reorientation_start' in df.columns:
            events = df[df['is_reorientation_start'] == True]['time'].values
        elif 'event_type' in df.columns:
            events = df[df['event_type'] == 'reorientation']['time'].values
        else:
            events = df['time'].values if 'time' in df.columns else np.array([])
        
        for t in events:
            all_events.append({'track_id': track_id, 'time': t})
    
    return pd.DataFrame(all_events)


def compute_iei(events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inter-event intervals within each track.
    
    Returns
    -------
    ieis : ndarray
        All inter-event intervals
    track_ids : ndarray
        Track ID for each IEI
    """
    ieis = []
    track_ids = []
    
    for track_id, group in events_df.groupby('track_id'):
        times = group['time'].sort_values().values
        if len(times) > 1:
            intervals = np.diff(times)
            ieis.extend(intervals)
            track_ids.extend([track_id] * len(intervals))
    
    return np.array(ieis), np.array(track_ids)


def classify_iei_by_led(ieis: np.ndarray, 
                         event_times: np.ndarray,
                         led_period: float = 60.0,
                         led_duty: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify IEIs by whether they span an LED-ON period.
    
    Returns
    -------
    within_off : ndarray
        IEIs that occur entirely within LED-OFF periods
    spans_on : ndarray
        IEIs that span at least part of an LED-ON period
    """
    led_on_duration = led_period * led_duty
    
    within_off = []
    spans_on = []
    
    # For each IEI, check if it spans an LED-ON period
    for i, iei in enumerate(ieis):
        t_start = event_times[i]
        t_end = t_start + iei
        
        # Check phase within LED cycle
        phase_start = t_start % led_period
        phase_end = t_end % led_period
        
        # LED is ON from 0 to led_on_duration within each cycle
        # IEI spans ON if:
        # 1. Starts during ON, or
        # 2. Ends during ON, or
        # 3. Spans a full cycle
        
        starts_in_on = phase_start < led_on_duration
        ends_in_on = phase_end < led_on_duration
        spans_cycle = iei >= led_period
        
        if spans_cycle or starts_in_on or ends_in_on:
            spans_on.append(iei)
        else:
            within_off.append(iei)
    
    return np.array(within_off), np.array(spans_on)


def fit_exponential(ieis: np.ndarray) -> Tuple[float, float]:
    """Fit exponential distribution to IEIs. Returns rate and R²."""
    if len(ieis) < 2:
        return np.nan, np.nan
    
    # MLE for exponential: rate = 1/mean
    rate = 1.0 / np.mean(ieis)
    
    # Compute R² for histogram fit
    bins = np.linspace(0, np.percentile(ieis, 99), 50)
    hist, _ = np.histogram(ieis, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    expected = rate * np.exp(-rate * bin_centers)
    
    ss_res = np.sum((hist - expected)**2)
    ss_tot = np.sum((hist - hist.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return rate, r2


def fit_gamma(ieis: np.ndarray) -> Tuple[float, float, float]:
    """Fit gamma distribution to IEIs. Returns shape, scale, R²."""
    if len(ieis) < 2:
        return np.nan, np.nan, np.nan
    
    # Filter out zero and negative values
    ieis_positive = ieis[ieis > 0]
    if len(ieis_positive) < 2:
        return np.nan, np.nan, np.nan
    
    # MLE for gamma
    try:
        shape, loc, scale = stats.gamma.fit(ieis_positive, floc=0)
    except Exception:
        return np.nan, np.nan, np.nan
    
    # Compute R² for histogram fit
    bins = np.linspace(0, np.percentile(ieis_positive, 99), 50)
    hist, _ = np.histogram(ieis_positive, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    expected = stats.gamma.pdf(bin_centers, shape, scale=scale)
    
    ss_res = np.sum((hist - expected)**2)
    ss_tot = np.sum((hist - hist.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return shape, scale, r2


def detect_refractory(ieis: np.ndarray, 
                       threshold: float = 0.3) -> Tuple[bool, float, float]:
    """
    Detect refractory period signature.
    
    Looks for a sharp drop in IEI density near zero relative to exponential expectation.
    
    Parameters
    ----------
    ieis : ndarray
        Inter-event intervals
    threshold : float
        Time threshold to check for refractory (seconds)
    
    Returns
    -------
    has_refractory : bool
        Whether refractory signature detected
    deficit_ratio : float
        Ratio of observed to expected IEIs below threshold
    estimated_tau_ref : float
        Estimated refractory time constant
    """
    if len(ieis) < 10:
        return False, np.nan, np.nan
    
    # Fit exponential to bulk of distribution (exclude very short IEIs)
    bulk_ieis = ieis[ieis > threshold]
    if len(bulk_ieis) < 10:
        return False, np.nan, np.nan
    
    rate = 1.0 / np.mean(bulk_ieis)
    
    # Expected number of IEIs below threshold under exponential
    p_below = 1 - np.exp(-rate * threshold)
    expected_below = len(ieis) * p_below
    
    # Observed number below threshold
    observed_below = np.sum(ieis < threshold)
    
    # Deficit ratio (0 = complete refractory, 1 = no refractory)
    deficit_ratio = observed_below / max(expected_below, 1)
    
    # Detect refractory if deficit > 50%
    has_refractory = deficit_ratio < 0.5
    
    # Estimate tau_ref from the shortest IEIs
    if has_refractory and observed_below > 0:
        short_ieis = ieis[ieis < threshold]
        estimated_tau_ref = np.median(short_ieis)
    else:
        estimated_tau_ref = 0.0
    
    return has_refractory, deficit_ratio, estimated_tau_ref


def fit_refractory_model(ieis: np.ndarray) -> dict:
    """
    Fit a model with refractory period.
    
    Models IEI as: tau_ref + Exponential(rate)
    
    Returns parameters and goodness of fit.
    """
    if len(ieis) < 10:
        return {'converged': False}
    
    # Estimate tau_ref as minimum IEI (or 5th percentile for robustness)
    tau_ref = np.percentile(ieis, 5)
    
    # Fit exponential to shifted IEIs
    shifted = ieis - tau_ref
    shifted = shifted[shifted > 0]  # Remove any negative values
    
    if len(shifted) < 10:
        return {'converged': False, 'tau_ref': tau_ref}
    
    rate = 1.0 / np.mean(shifted)
    
    # Compute log-likelihood
    ll = np.sum(np.log(rate) - rate * shifted)
    
    # Compare to pure exponential
    rate_exp = 1.0 / np.mean(ieis)
    ll_exp = np.sum(np.log(rate_exp) - rate_exp * ieis)
    
    # Likelihood ratio test (1 extra parameter)
    lr_stat = 2 * (ll - ll_exp)
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    
    return {
        'converged': True,
        'tau_ref': float(tau_ref),
        'rate': float(rate),
        'mean_after_ref': float(1.0 / rate),
        'll_refractory': float(ll),
        'll_exponential': float(ll_exp),
        'lr_statistic': float(lr_stat),
        'p_value': float(p_value),
        'refractory_significant': bool(p_value < 0.05)
    }


def plot_iei_analysis(ieis: np.ndarray, 
                       within_off: np.ndarray,
                       spans_on: np.ndarray,
                       refractory_results: dict,
                       output_path: Path):
    """Create comprehensive IEI analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall IEI histogram
    ax = axes[0, 0]
    bins = np.linspace(0, min(120, np.percentile(ieis, 99)), 60)
    ax.hist(ieis, bins=bins, density=True, alpha=0.7, color='steelblue', 
            edgecolor='white', label='All IEIs')
    
    # Overlay exponential fit
    rate = 1.0 / np.mean(ieis)
    x = np.linspace(0, bins[-1], 100)
    ax.plot(x, rate * np.exp(-rate * x), 'r-', linewidth=2, 
            label=f'Exponential (rate={rate:.3f}/s)')
    
    ax.set_xlabel('Inter-event interval (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'IEI Distribution (n={len(ieis)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats
    ax.text(0.95, 0.95, 
            f'Mean: {np.mean(ieis):.1f}s\nMedian: {np.median(ieis):.1f}s\nCV: {np.std(ieis)/np.mean(ieis):.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Log-scale histogram (to see tail)
    ax = axes[0, 1]
    log_ieis = np.log10(ieis[ieis > 0])
    bins = np.linspace(log_ieis.min(), log_ieis.max(), 50)
    ax.hist(log_ieis, bins=bins, density=True, alpha=0.7, color='steelblue', 
            edgecolor='white')
    ax.set_xlabel('log₁₀(IEI) (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IEI Distribution (log scale)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark LED period
    ax.axvline(np.log10(30), color='orange', linestyle='--', alpha=0.7, 
               label='LED half-period (30s)')
    ax.axvline(np.log10(60), color='red', linestyle='--', alpha=0.7, 
               label='LED period (60s)')
    ax.legend()
    
    # 3. Short IEI analysis (refractory check)
    ax = axes[1, 0]
    short_ieis = ieis[ieis < 5.0]
    if len(short_ieis) > 0:
        bins = np.linspace(0, 5, 50)
        ax.hist(short_ieis, bins=bins, density=True, alpha=0.7, color='steelblue', 
                edgecolor='white', label='Observed')
        
        # Expected from exponential
        rate = 1.0 / np.mean(ieis)
        x = np.linspace(0, 5, 100)
        ax.plot(x, rate * np.exp(-rate * x), 'r-', linewidth=2, 
                label='Exponential expectation')
        
        # Mark potential refractory period
        if refractory_results.get('converged', False):
            tau_ref = refractory_results.get('tau_ref', 0)
            ax.axvline(tau_ref, color='green', linestyle=':', linewidth=2,
                       label=f'Est. τ_ref = {tau_ref:.2f}s')
    
    ax.set_xlabel('Inter-event interval (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Short IEI Analysis (Refractory Check)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
IEI ANALYSIS SUMMARY
====================

Basic Statistics:
  N IEIs: {len(ieis)}
  Mean: {np.mean(ieis):.2f}s
  Median: {np.median(ieis):.2f}s
  Std: {np.std(ieis):.2f}s
  CV: {np.std(ieis)/np.mean(ieis):.2f}
  Min: {np.min(ieis):.3f}s
  Max: {np.max(ieis):.1f}s

LED-Based Classification:
  Within LED-OFF: {len(within_off)} ({100*len(within_off)/max(len(ieis),1):.1f}%)
  Spanning LED-ON: {len(spans_on)} ({100*len(spans_on)/max(len(ieis),1):.1f}%)

Refractory Analysis:
  Converged: {refractory_results.get('converged', False)}
  Estimated τ_ref: {refractory_results.get('tau_ref', 'N/A'):.3f}s
  Refractory significant: {refractory_results.get('refractory_significant', 'N/A')}
  p-value: {refractory_results.get('p_value', 'N/A'):.4f}

Distribution Fit:
  Exponential rate: {1/np.mean(ieis):.4f}/s
  Expected mean: {np.mean(ieis):.2f}s
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved IEI plot to {output_path}")


def main():
    print("=" * 70)
    print("INTER-EVENT INTERVAL ANALYSIS")
    print("=" * 70)
    
    # Load empirical events
    data_dir = Path('data/engineered')
    try:
        events_df = load_empirical_events(data_dir)
        print(f"Loaded {len(events_df)} events")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Compute IEIs
    ieis, track_ids = compute_iei(events_df)
    print(f"Computed {len(ieis)} inter-event intervals")
    
    # Basic statistics
    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    print(f"  Mean IEI: {np.mean(ieis):.2f}s")
    print(f"  Median IEI: {np.median(ieis):.2f}s")
    print(f"  Std IEI: {np.std(ieis):.2f}s")
    print(f"  CV: {np.std(ieis)/np.mean(ieis):.2f}")
    print(f"  Min: {np.min(ieis):.3f}s")
    print(f"  Max: {np.max(ieis):.1f}s")
    
    # Get event times for LED classification
    event_times = []
    for track_id, group in events_df.groupby('track_id'):
        times = group['time'].sort_values().values
        if len(times) > 1:
            event_times.extend(times[:-1])  # All but last (since IEI needs a next event)
    event_times = np.array(event_times)
    
    # Classify by LED state
    within_off, spans_on = classify_iei_by_led(ieis, event_times)
    
    print("\n" + "=" * 50)
    print("LED-BASED CLASSIFICATION")
    print("=" * 50)
    print(f"  Within LED-OFF: {len(within_off)} ({100*len(within_off)/len(ieis):.1f}%)")
    print(f"  Spanning LED-ON: {len(spans_on)} ({100*len(spans_on)/len(ieis):.1f}%)")
    
    if len(within_off) > 0:
        print(f"  Mean IEI within OFF: {np.mean(within_off):.2f}s")
    if len(spans_on) > 0:
        print(f"  Mean IEI spanning ON: {np.mean(spans_on):.2f}s")
    
    # Check for bimodality
    if len(within_off) > 10 and len(spans_on) > 10:
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(within_off, spans_on)
        print(f"\n  KS test (within-OFF vs spans-ON): stat={ks_stat:.3f}, p={ks_p:.2e}")
        if ks_p < 0.05:
            print("  -> Distributions are SIGNIFICANTLY DIFFERENT (supports bimodality)")
        else:
            print("  -> Distributions are similar")
    
    # Refractory analysis
    print("\n" + "=" * 50)
    print("REFRACTORY PERIOD ANALYSIS")
    print("=" * 50)
    
    has_ref, deficit, tau_est = detect_refractory(ieis, threshold=0.3)
    print(f"  Threshold: 0.3s")
    print(f"  Deficit ratio: {deficit:.3f} (0=complete refractory, 1=none)")
    print(f"  Refractory detected: {has_ref}")
    print(f"  Estimated τ_ref: {tau_est:.3f}s")
    
    # Fit refractory model
    ref_results = fit_refractory_model(ieis)
    if ref_results.get('converged', False):
        print(f"\n  Refractory model fit:")
        print(f"    τ_ref: {ref_results['tau_ref']:.3f}s")
        print(f"    Mean after refractory: {ref_results['mean_after_ref']:.2f}s")
        print(f"    LR statistic: {ref_results['lr_statistic']:.2f}")
        print(f"    p-value: {ref_results['p_value']:.4f}")
        print(f"    Significant: {ref_results['refractory_significant']}")
    
    # Distribution fitting
    print("\n" + "=" * 50)
    print("DISTRIBUTION FITTING")
    print("=" * 50)
    
    exp_rate, exp_r2 = fit_exponential(ieis)
    print(f"  Exponential: rate={exp_rate:.4f}/s, R²={exp_r2:.3f}")
    
    gamma_shape, gamma_scale, gamma_r2 = fit_gamma(ieis)
    print(f"  Gamma: shape={gamma_shape:.2f}, scale={gamma_scale:.2f}s, R²={gamma_r2:.3f}")
    
    # Save results
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'n_ieis': int(len(ieis)),
        'mean': float(np.mean(ieis)),
        'median': float(np.median(ieis)),
        'std': float(np.std(ieis)),
        'cv': float(np.std(ieis) / np.mean(ieis)),
        'min': float(np.min(ieis)),
        'max': float(np.max(ieis)),
        'n_within_off': int(len(within_off)),
        'n_spans_on': int(len(spans_on)),
        'mean_within_off': float(np.mean(within_off)) if len(within_off) > 0 else None,
        'mean_spans_on': float(np.mean(spans_on)) if len(spans_on) > 0 else None,
        'refractory_detected': bool(has_ref),
        'refractory_deficit_ratio': float(deficit),
        'refractory_tau_estimate': float(tau_est),
        'refractory_model': ref_results,
        'exponential_rate': float(exp_rate),
        'exponential_r2': float(exp_r2),
        'gamma_shape': float(gamma_shape),
        'gamma_scale': float(gamma_scale),
        'gamma_r2': float(gamma_r2)
    }
    
    with open(output_dir / 'iei_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'iei_analysis.json'}")
    
    # Plot
    plot_iei_analysis(ieis, within_off, spans_on, ref_results,
                      output_dir / 'iei_analysis.png')
    
    # Recommendation
    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    
    if ref_results.get('refractory_significant', False):
        print("  REFRACTORY PERIOD DETECTED")
        print(f"  Recommend adding post-event kernel with τ_ref ≈ {ref_results['tau_ref']:.2f}s")
        print("  K_post(t) = -2.0 * exp(-t / τ_ref)")
    else:
        print("  No significant refractory period detected.")
        print("  Current model (Poisson with time-varying hazard) is adequate.")
    
    print("\nPhase 3 complete!")


if __name__ == '__main__':
    main()


