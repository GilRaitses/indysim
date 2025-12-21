#!/usr/bin/env python3
"""
Track-Level Validation for Hazard Model

Compares simulated vs empirical data at the track level:
1. Event counts per track
2. Inter-event interval (IEI) distribution
3. PSTH around LED onset/offset
4. Clustering (Fano factor)

Usage:
    python scripts/validate_track_level.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from analytic_hazard import AnalyticHazardModel, KernelParams


def load_empirical_events(data_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load empirical event data from engineered files."""
    # Look for event files
    event_files = sorted(data_dir.glob('*_events.csv'))
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in {data_dir}")
    
    all_events = []
    track_info = {}
    
    for i, f in enumerate(event_files):
        df = pd.read_csv(f)
        track_id = i + 1
        
        # Get reorientation events
        if 'is_reorientation_start' in df.columns:
            events = df[df['is_reorientation_start'] == True]['time'].values
        elif 'event_type' in df.columns:
            events = df[df['event_type'] == 'reorientation']['time'].values
        else:
            # Assume all rows are events
            events = df['time'].values if 'time' in df.columns else np.array([])
        
        for t in events:
            all_events.append({'track_id': track_id, 'time': t})
        
        track_info[track_id] = {
            'n_events': len(events),
            'duration': df['time'].max() - df['time'].min() if 'time' in df.columns else 0,
            'file': f.name
        }
    
    events_df = pd.DataFrame(all_events)
    return events_df, track_info


def load_binned_data(data_dir: Path) -> pd.DataFrame:
    """Load binned data for LED timing information."""
    binned_path = data_dir / 'binned_data.parquet'
    if binned_path.exists():
        return pd.read_parquet(binned_path)
    
    # Try CSV
    binned_path = data_dir / 'binned_data.csv'
    if binned_path.exists():
        return pd.read_csv(binned_path)
    
    return None


def simulate_tracks(model: AnalyticHazardModel, 
                    n_tracks: int,
                    duration: float,
                    led_period: float = 60.0,
                    led_duty: float = 0.5,
                    track_intercept_std: float = 0.47,
                    seed: int = 42) -> pd.DataFrame:
    """
    Simulate multiple tracks with the hazard model.
    
    Parameters
    ----------
    model : AnalyticHazardModel
        The hazard model to use
    n_tracks : int
        Number of tracks to simulate
    duration : float
        Duration per track (seconds)
    led_period : float
        LED cycle period (seconds)
    led_duty : float
        LED duty cycle (fraction ON)
    track_intercept_std : float
        Standard deviation of track random effects
    seed : int
        Random seed
    
    Returns
    -------
    DataFrame with columns: track_id, time
    """
    rng = np.random.default_rng(seed)
    
    # LED timing
    n_cycles = int(np.ceil(duration / led_period))
    led_on_duration = led_period * led_duty
    led_onsets = np.array([i * led_period for i in range(n_cycles)])
    led_offsets = led_onsets + led_on_duration
    
    all_events = []
    
    for track_id in range(1, n_tracks + 1):
        # Sample track intercept
        track_intercept = rng.normal(0, track_intercept_std)
        
        # Simulate events
        events = model.simulate_events_thinning(
            duration=duration,
            led_onset_times=led_onsets,
            led_offset_times=led_offsets,
            track_intercept=track_intercept,
            dt=0.05,
            seed=seed + track_id
        )
        
        for t in events:
            all_events.append({'track_id': track_id, 'time': t})
    
    return pd.DataFrame(all_events) if all_events else pd.DataFrame(columns=['track_id', 'time'])


def compute_event_counts(events_df: pd.DataFrame, n_tracks: int) -> np.ndarray:
    """Compute event counts per track."""
    if events_df.empty:
        return np.zeros(n_tracks)
    
    counts = events_df.groupby('track_id').size()
    result = np.zeros(n_tracks)
    for track_id, count in counts.items():
        if track_id <= n_tracks:
            result[track_id - 1] = count
    return result


def compute_iei(events_df: pd.DataFrame) -> np.ndarray:
    """Compute inter-event intervals within each track."""
    if events_df.empty:
        return np.array([])
    
    ieis = []
    for track_id, group in events_df.groupby('track_id'):
        times = group['time'].sort_values().values
        if len(times) > 1:
            ieis.extend(np.diff(times))
    
    return np.array(ieis)


def compute_psth(events_df: pd.DataFrame, 
                 trigger_times: np.ndarray,
                 window: Tuple[float, float] = (-5, 15),
                 bin_width: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram.
    
    Parameters
    ----------
    events_df : DataFrame
        Events with 'time' column
    trigger_times : ndarray
        Times to align events to (e.g., LED onsets)
    window : tuple
        (pre, post) window around triggers
    bin_width : float
        Histogram bin width
    
    Returns
    -------
    bin_centers, rates
    """
    if events_df.empty:
        bins = np.arange(window[0], window[1] + bin_width, bin_width)
        return (bins[:-1] + bins[1:]) / 2, np.zeros(len(bins) - 1)
    
    relative_times = []
    for t_trigger in trigger_times:
        for _, row in events_df.iterrows():
            t_rel = row['time'] - t_trigger
            if window[0] <= t_rel <= window[1]:
                relative_times.append(t_rel)
    
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    counts, _ = np.histogram(relative_times, bins=bins)
    
    # Normalize by number of triggers and bin width
    rates = counts / (len(trigger_times) * bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, rates


def compute_fano_factor(events_df: pd.DataFrame,
                        epoch_starts: np.ndarray,
                        epoch_ends: np.ndarray) -> float:
    """
    Compute Fano factor (variance/mean) of event counts per epoch.
    
    Used to assess clustering beyond Poisson expectation.
    """
    if events_df.empty:
        return np.nan
    
    counts = []
    for start, end in zip(epoch_starts, epoch_ends):
        n = ((events_df['time'] >= start) & (events_df['time'] < end)).sum()
        counts.append(n)
    
    counts = np.array(counts)
    if counts.mean() == 0:
        return np.nan
    
    return counts.var() / counts.mean()


def plot_validation(emp_counts: np.ndarray, sim_counts: np.ndarray,
                    emp_iei: np.ndarray, sim_iei: np.ndarray,
                    emp_psth: Tuple, sim_psth: Tuple,
                    emp_fano: float, sim_fano: float,
                    output_path: Path):
    """Create comprehensive validation plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Event counts histogram
    ax = axes[0, 0]
    max_count = max(emp_counts.max() if len(emp_counts) > 0 else 0,
                    sim_counts.max() if len(sim_counts) > 0 else 0, 1)
    bins = np.arange(0, max_count + 2) - 0.5
    
    ax.hist(emp_counts, bins=bins, alpha=0.6, label=f'Empirical (n={len(emp_counts)})', 
            color='blue', density=True)
    ax.hist(sim_counts, bins=bins, alpha=0.6, label=f'Simulated (n={len(sim_counts)})', 
            color='orange', density=True)
    ax.set_xlabel('Events per track', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Event Count Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats
    emp_mean = emp_counts.mean() if len(emp_counts) > 0 else 0
    sim_mean = sim_counts.mean() if len(sim_counts) > 0 else 0
    ax.text(0.95, 0.95, f'Emp mean: {emp_mean:.1f}\nSim mean: {sim_mean:.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. IEI distribution
    ax = axes[0, 1]
    if len(emp_iei) > 0:
        ax.hist(emp_iei, bins=50, alpha=0.6, label='Empirical', color='blue', density=True)
    if len(sim_iei) > 0:
        ax.hist(sim_iei, bins=50, alpha=0.6, label='Simulated', color='orange', density=True)
    ax.set_xlabel('Inter-event interval (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IEI Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(120, max(emp_iei.max() if len(emp_iei) > 0 else 60,
                                 sim_iei.max() if len(sim_iei) > 0 else 60)))
    
    # Stats
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        ks_stat, ks_p = stats.ks_2samp(emp_iei, sim_iei)
        ax.text(0.95, 0.95, f'KS stat: {ks_stat:.3f}\np-value: {ks_p:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. PSTH
    ax = axes[1, 0]
    if emp_psth[1] is not None:
        ax.plot(emp_psth[0], emp_psth[1], 'b-', linewidth=2, label='Empirical')
    if sim_psth[1] is not None:
        ax.plot(sim_psth[0], sim_psth[1], 'r--', linewidth=2, label='Simulated')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.7, label='LED onset')
    ax.set_xlabel('Time relative to LED onset (s)', fontsize=12)
    ax.set_ylabel('Event rate (Hz)', fontsize=12)
    ax.set_title('PSTH around LED Onset', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    if emp_psth[1] is not None and sim_psth[1] is not None and len(emp_psth[1]) > 0:
        corr = np.corrcoef(emp_psth[1], sim_psth[1])[0, 1]
        ax.text(0.95, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Summary metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute metrics
    rate_ratio = sim_counts.sum() / max(emp_counts.sum(), 1)
    
    metrics_text = f"""
VALIDATION SUMMARY
==================

Event Counts:
  Empirical total: {emp_counts.sum():.0f}
  Simulated total: {sim_counts.sum():.0f}
  Rate ratio: {rate_ratio:.3f}
  Target: 0.8-1.25

IEI Statistics:
  Empirical: mean={np.mean(emp_iei):.1f}s, median={np.median(emp_iei):.1f}s, CV={np.std(emp_iei)/np.mean(emp_iei):.2f}
  Simulated: mean={np.mean(sim_iei) if len(sim_iei) > 0 else 0:.1f}s, median={np.median(sim_iei) if len(sim_iei) > 0 else 0:.1f}s

Fano Factor (clustering):
  Empirical: {emp_fano:.2f}
  Simulated: {sim_fano:.2f}
  (Poisson expectation: 1.0)

PASS/FAIL:
  Rate ratio: {'PASS' if 0.8 <= rate_ratio <= 1.25 else 'FAIL'}
    """ if len(emp_iei) > 0 else "No empirical IEI data available"
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved validation plot to {output_path}")


def main():
    print("=" * 70)
    print("TRACK-LEVEL VALIDATION")
    print("=" * 70)
    
    # Load model results for track info
    model_path = Path('data/model/hybrid_model_results.json')
    with open(model_path) as f:
        model_results = json.load(f)
    
    n_tracks = model_results['n_tracks']
    track_intercepts = model_results.get('track_intercepts', {})
    track_event_rates = model_results.get('track_event_rates', {})
    
    print(f"Model has {n_tracks} tracks")
    
    # Get empirical event counts from model results
    emp_counts = np.array([track_event_rates.get(str(i+1), 0) for i in range(n_tracks)])
    
    # Since event rates are per minute, and we have track durations of ~20 min,
    # estimate total events per track
    # Actually, these ARE already rates per minute
    # We need to know track durations to get total counts
    # For now, assume ~20 min per track based on typical experiment length
    track_duration_min = 20.0
    emp_counts_estimated = emp_counts * track_duration_min
    
    print(f"Empirical events (estimated from rates): {emp_counts_estimated.sum():.0f}")
    print(f"Mean events/track: {emp_counts_estimated.mean():.1f}")
    
    # Create hazard model
    model = AnalyticHazardModel()
    
    # Simulate tracks
    print(f"\nSimulating {n_tracks} tracks...")
    duration = track_duration_min * 60  # seconds
    
    # Use actual track intercepts
    rng = np.random.default_rng(42)
    
    all_sim_events = []
    led_onsets = np.arange(0, duration, 60)  # LED on every 60s
    led_offsets = led_onsets + 30  # LED off after 30s
    
    for track_id in range(1, n_tracks + 1):
        # Get track intercept (relative to global)
        track_int = track_intercepts.get(str(track_id), model.params.intercept)
        track_effect = track_int - model.params.intercept
        
        # Use discrete (frame-by-frame) simulation to match GLM exactly
        events = model.simulate_events_discrete(
            duration=duration,
            led_onset_times=led_onsets,
            led_offset_times=led_offsets,
            track_intercept=track_effect,
            seed=42 + track_id
        )
        
        for t in events:
            all_sim_events.append({'track_id': track_id, 'time': t})
    
    sim_events_df = pd.DataFrame(all_sim_events) if all_sim_events else pd.DataFrame(columns=['track_id', 'time'])
    
    # Compute simulated counts
    sim_counts = compute_event_counts(sim_events_df, n_tracks)
    
    print(f"Simulated events: {sim_counts.sum():.0f}")
    print(f"Mean events/track: {sim_counts.mean():.1f}")
    
    # Compute IEIs
    # For empirical IEI, we need actual event times, not just rates
    # Try to load from engineered data
    try:
        data_dir = Path('data/engineered')
        emp_events_df, _ = load_empirical_events(data_dir)
        emp_iei = compute_iei(emp_events_df)
        emp_counts_actual = compute_event_counts(emp_events_df, n_tracks)
        print(f"\nLoaded {len(emp_events_df)} empirical events")
    except FileNotFoundError:
        print("\nNo empirical event files found, using estimated counts")
        emp_events_df = pd.DataFrame(columns=['track_id', 'time'])
        emp_iei = np.array([])
        emp_counts_actual = emp_counts_estimated
    
    sim_iei = compute_iei(sim_events_df)
    
    print(f"Empirical IEIs: {len(emp_iei)}")
    print(f"Simulated IEIs: {len(sim_iei)}")
    
    # Compute PSTH
    if not emp_events_df.empty:
        emp_psth = compute_psth(emp_events_df, led_onsets)
    else:
        emp_psth = (np.array([]), None)
    
    sim_psth = compute_psth(sim_events_df, led_onsets)
    
    # Compute Fano factor for LED-OFF epochs
    led_off_starts = led_offsets
    led_off_ends = np.minimum(led_offsets + 30, duration)
    
    emp_fano = compute_fano_factor(emp_events_df, led_off_starts, led_off_ends) if not emp_events_df.empty else np.nan
    sim_fano = compute_fano_factor(sim_events_df, led_off_starts, led_off_ends)
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    rate_ratio = sim_counts.sum() / max(emp_counts_actual.sum(), 1)
    print(f"\nRate ratio (sim/emp): {rate_ratio:.3f}")
    print(f"  Target: 0.8-1.25")
    print(f"  Status: {'PASS' if 0.8 <= rate_ratio <= 1.25 else 'FAIL'}")
    
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        ks_stat, ks_p = stats.ks_2samp(emp_iei, sim_iei)
        print(f"\nIEI KS test: stat={ks_stat:.3f}, p={ks_p:.3f}")
    
    print(f"\nFano factor:")
    print(f"  Empirical: {emp_fano:.2f}")
    print(f"  Simulated: {sim_fano:.2f}")
    
    # Save results
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'n_tracks': n_tracks,
        'emp_total_events': float(emp_counts_actual.sum()),
        'sim_total_events': float(sim_counts.sum()),
        'rate_ratio': float(rate_ratio),
        'rate_pass': bool(0.8 <= rate_ratio <= 1.25),
        'emp_mean_events_per_track': float(emp_counts_actual.mean()),
        'sim_mean_events_per_track': float(sim_counts.mean()),
        'emp_fano': float(emp_fano) if not np.isnan(emp_fano) else None,
        'sim_fano': float(sim_fano) if not np.isnan(sim_fano) else None
    }
    
    if len(emp_iei) > 0:
        results['emp_iei_mean'] = float(np.mean(emp_iei))
        results['emp_iei_median'] = float(np.median(emp_iei))
        results['emp_iei_cv'] = float(np.std(emp_iei) / np.mean(emp_iei))
    
    if len(sim_iei) > 0:
        results['sim_iei_mean'] = float(np.mean(sim_iei))
        results['sim_iei_median'] = float(np.median(sim_iei))
        results['sim_iei_cv'] = float(np.std(sim_iei) / np.mean(sim_iei))
    
    with open(output_dir / 'track_level_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'track_level_validation.json'}")
    
    # Plot
    plot_validation(emp_counts_actual, sim_counts,
                    emp_iei, sim_iei,
                    emp_psth, sim_psth,
                    emp_fano, sim_fano,
                    output_dir / 'track_level_validation.png')
    
    print("\nPhase 2 complete!")


if __name__ == '__main__':
    main()


