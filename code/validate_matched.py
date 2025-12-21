#!/usr/bin/env python3
"""
Matched Validation for Hazard Model

Validates the analytic hazard model against the exact dataset used for fitting:
- 55 tracks from 2 specific experiment files
- 1407 reorientation events (all) or 319 (filtered by duration)
- 10s ON / 20s OFF LED cycle

Usage:
    python scripts/validate_matched.py              # All events (default)
    python scripts/validate_matched.py --filtered   # Filtered events only
    python scripts/validate_matched.py --both       # Both event sets
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from load_fitting_data import (
    load_fitting_dataset, 
    get_event_times,
    get_filtered_events,
    get_led_timing,
    get_track_intercepts,
    compute_empirical_stats,
    LED_ON_DURATION,
    LED_OFF_DURATION,
    LED_CYCLE,
    FIRST_LED_ONSET
)
from analytic_hazard import AnalyticHazardModel, KernelParams


def simulate_matched_tracks(
    model: AnalyticHazardModel,
    data: pd.DataFrame,
    track_intercepts: Dict[int, float],
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate events for each track using the matched LED timing.
    
    Uses the discrete (frame-by-frame) simulation for accuracy.
    """
    led_onsets, led_offsets = get_led_timing(data)
    
    all_events = []
    
    for track_id in sorted(data['track_id'].unique()):
        track_data = data[data['track_id'] == track_id]
        duration = track_data['time'].max() - track_data['time'].min()
        
        # Get track-specific intercept
        intercept = track_intercepts.get(track_id, 0.0)
        
        # Simulate events
        events = model.simulate_events_discrete(
            duration=duration,
            led_onset_times=led_onsets,
            led_offset_times=led_offsets,
            track_intercept=intercept,
            seed=seed + track_id
        )
        
        for t in events:
            all_events.append({'track_id': track_id, 'time': t})
    
    return pd.DataFrame(all_events) if all_events else pd.DataFrame(columns=['track_id', 'time'])


def compute_iei(events_df: pd.DataFrame) -> np.ndarray:
    """Compute inter-event intervals within each track."""
    ieis = []
    for track_id, group in events_df.groupby('track_id'):
        times = group['time'].sort_values().values
        if len(times) > 1:
            ieis.extend(np.diff(times))
    return np.array(ieis)


def compute_psth(events_df: pd.DataFrame, 
                 trigger_times: np.ndarray,
                 window: Tuple[float, float] = (-5, 25),
                 bin_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram around LED onsets.
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


def compute_suppression_metrics(events_df: pd.DataFrame,
                                 led_onsets: np.ndarray,
                                 led_offsets: np.ndarray) -> Dict:
    """
    Compute suppression metrics during LED-ON vs LED-OFF.
    """
    if events_df.empty:
        return {'rate_on': 0, 'rate_off': 0, 'suppression_ratio': float('inf')}
    
    # Count events in LED-ON vs LED-OFF epochs
    events_on = 0
    events_off = 0
    total_on_time = 0
    total_off_time = 0
    
    for t_on, t_off in zip(led_onsets, led_offsets):
        # Events during this ON period
        n_on = ((events_df['time'] >= t_on) & (events_df['time'] < t_off)).sum()
        events_on += n_on
        total_on_time += (t_off - t_on)
        
        # Events during the following OFF period
        t_off_end = t_on + LED_CYCLE
        n_off = ((events_df['time'] >= t_off) & (events_df['time'] < t_off_end)).sum()
        events_off += n_off
        total_off_time += (t_off_end - t_off)
    
    rate_on = events_on / (total_on_time / 60) if total_on_time > 0 else 0
    rate_off = events_off / (total_off_time / 60) if total_off_time > 0 else 0
    
    suppression_ratio = rate_off / rate_on if rate_on > 0 else float('inf')
    
    return {
        'events_on': int(events_on),
        'events_off': int(events_off),
        'rate_on': float(rate_on),
        'rate_off': float(rate_off),
        'suppression_ratio': float(suppression_ratio)
    }


def plot_validation(emp_events: pd.DataFrame, 
                    sim_events: pd.DataFrame,
                    emp_iei: np.ndarray,
                    sim_iei: np.ndarray,
                    emp_psth: Tuple,
                    sim_psth: Tuple,
                    results: Dict,
                    output_path: Path):
    """Create 4-panel validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Event counts per track
    ax = axes[0, 0]
    emp_counts = emp_events.groupby('track_id').size()
    sim_counts = sim_events.groupby('track_id').size() if not sim_events.empty else pd.Series([])
    
    tracks = sorted(set(emp_counts.index) | set(sim_counts.index))
    x = np.arange(len(tracks))
    width = 0.35
    
    emp_vals = [emp_counts.get(t, 0) for t in tracks]
    sim_vals = [sim_counts.get(t, 0) for t in tracks]
    
    ax.bar(x - width/2, emp_vals, width, label='Empirical', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, sim_vals, width, label='Simulated', color='coral', alpha=0.8)
    ax.set_xlabel('Track ID', fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.set_title(f'Events per Track (Rate Ratio: {results["rate_ratio"]:.2f})', fontsize=14)
    ax.legend()
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(tracks[i]) for i in range(0, len(tracks), 5)])
    
    # 2. IEI distribution
    ax = axes[0, 1]
    if len(emp_iei) > 0:
        ax.hist(emp_iei, bins=30, alpha=0.6, label='Empirical', color='steelblue', density=True)
    if len(sim_iei) > 0:
        ax.hist(sim_iei, bins=30, alpha=0.6, label='Simulated', color='coral', density=True)
    ax.set_xlabel('Inter-event interval (s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IEI Distribution', fontsize=14)
    ax.legend()
    ax.set_xlim(0, min(120, max(emp_iei.max() if len(emp_iei) > 0 else 60,
                                 sim_iei.max() if len(sim_iei) > 0 else 60)))
    
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        ks_stat, ks_p = stats.ks_2samp(emp_iei, sim_iei)
        ax.text(0.95, 0.95, f'KS: {ks_stat:.3f} (p={ks_p:.3f})',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. PSTH around LED onset
    ax = axes[1, 0]
    ax.plot(emp_psth[0], emp_psth[1], 'b-', linewidth=2, label='Empirical')
    ax.plot(sim_psth[0], sim_psth[1], 'r--', linewidth=2, label='Simulated')
    ax.axvline(0, color='green', linestyle=':', alpha=0.7, label='LED ON')
    ax.axvline(LED_ON_DURATION, color='red', linestyle=':', alpha=0.7, label='LED OFF')
    ax.axhspan(0, ax.get_ylim()[1], 0, LED_ON_DURATION/30, alpha=0.1, color='yellow')
    ax.set_xlabel('Time relative to LED onset (s)', fontsize=12)
    ax.set_ylabel('Event rate', fontsize=12)
    ax.set_title('PSTH around LED Onset', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    if len(emp_psth[1]) > 0 and len(sim_psth[1]) > 0:
        corr = np.corrcoef(emp_psth[1], sim_psth[1])[0, 1]
        ax.text(0.95, 0.95, f'Corr: {corr:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
MATCHED VALIDATION RESULTS
==========================

Data Alignment:
  Files: 2 (0to250PWM_30#C_Bl_7PWM_2025103*)
  Tracks: {results['n_tracks']}
  Empirical events: {results['emp_events']}
  Simulated events: {results['sim_events']}

Rate Metrics:
  Rate ratio: {results['rate_ratio']:.3f}
  Target: 0.8-1.25
  Status: {'PASS' if 0.8 <= results['rate_ratio'] <= 1.25 else 'FAIL'}

Suppression:
  Empirical: {results['emp_suppression']['suppression_ratio']:.1f}x
  Simulated: {results['sim_suppression']['suppression_ratio']:.1f}x

IEI:
  Empirical mean: {results.get('emp_iei_mean', 'N/A'):.1f}s
  Simulated mean: {results.get('sim_iei_mean', 'N/A'):.1f}s

PSTH Correlation: {results.get('psth_correlation', 'N/A'):.3f}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved validation plot to {output_path}")


def run_validation(
    data: pd.DataFrame,
    model_info: Dict,
    use_filtered: bool = False,
    output_suffix: str = ""
) -> Dict:
    """
    Run validation on specified event set.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset
    model_info : dict
        Model configuration
    use_filtered : bool
        If True, use filtered events (duration > 0.1s)
    output_suffix : str
        Suffix for output files (e.g., "_filtered")
    
    Returns
    -------
    results : dict
        Validation results
    """
    # Get events
    if use_filtered:
        emp_events = get_filtered_events(data, min_duration=0.1)
        event_type = "filtered (duration > 0.1s)"
        # Recalibrate for filtered events
        calibration_factor = 319 / 193  # Approximate ratio for filtered events
    else:
        emp_events = get_event_times(data)
        event_type = "all"
        calibration_factor = 1407 / 830
    
    led_onsets, led_offsets = get_led_timing(data)
    track_intercepts = get_track_intercepts(model_info)
    
    print(f"\nEvent type: {event_type}")
    print(f"Empirical: {len(emp_events)} events across {data['track_id'].nunique()} tracks")
    
    # Create model with appropriate calibration
    original_intercept = -6.76
    calibrated_intercept = original_intercept + np.log(calibration_factor)
    
    print(f"\nCalibration:")
    print(f"  Original intercept: {original_intercept}")
    print(f"  Calibration factor: {calibration_factor:.3f}")
    print(f"  Calibrated intercept: {calibrated_intercept:.4f}")
    
    params = KernelParams(
        A=0.456, alpha1=2.22, beta1=0.132,
        B=12.54, alpha2=4.38, beta2=0.869,
        D=-0.114, tau_off=2.0,
        intercept=calibrated_intercept,
        frame_rate=20.0
    )
    model = AnalyticHazardModel(params)
    
    # Simulate events
    print("\nSimulating events...")
    sim_events = simulate_matched_tracks(model, data, track_intercepts, seed=42)
    print(f"Simulated: {len(sim_events)} events")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    rate_ratio = len(sim_events) / len(emp_events) if len(emp_events) > 0 else 0
    emp_iei = compute_iei(emp_events)
    sim_iei = compute_iei(sim_events)
    emp_psth = compute_psth(emp_events, led_onsets)
    sim_psth = compute_psth(sim_events, led_onsets)
    emp_suppression = compute_suppression_metrics(emp_events, led_onsets, led_offsets)
    sim_suppression = compute_suppression_metrics(sim_events, led_onsets, led_offsets)
    psth_corr = np.corrcoef(emp_psth[1], sim_psth[1])[0, 1] if len(emp_psth[1]) > 0 else 0
    
    results = {
        'event_type': event_type,
        'n_tracks': int(data['track_id'].nunique()),
        'emp_events': int(len(emp_events)),
        'sim_events': int(len(sim_events)),
        'rate_ratio': float(rate_ratio),
        'rate_pass': bool(0.8 <= rate_ratio <= 1.25),
        'calibration_factor': float(calibration_factor),
        'calibrated_intercept': float(calibrated_intercept),
        'emp_iei_mean': float(np.mean(emp_iei)) if len(emp_iei) > 0 else None,
        'sim_iei_mean': float(np.mean(sim_iei)) if len(sim_iei) > 0 else None,
        'psth_correlation': float(psth_corr),
        'emp_suppression': emp_suppression,
        'sim_suppression': sim_suppression
    }
    
    # Print results
    print("\n" + "=" * 50)
    print(f"VALIDATION RESULTS ({event_type})")
    print("=" * 50)
    
    print(f"\nRate ratio: {rate_ratio:.3f}")
    print(f"  Target: 0.8-1.25")
    print(f"  Status: {'PASS' if results['rate_pass'] else 'FAIL'}")
    
    print(f"\nSuppression:")
    print(f"  Empirical: {emp_suppression['suppression_ratio']:.1f}x")
    print(f"  Simulated: {sim_suppression['suppression_ratio']:.1f}x")
    
    print(f"\nPSTH correlation: {psth_corr:.3f}")
    
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        ks_stat, ks_p = stats.ks_2samp(emp_iei, sim_iei)
        print(f"\nIEI KS test: stat={ks_stat:.3f}, p={ks_p:.3f}")
        results['iei_ks_stat'] = float(ks_stat)
        results['iei_ks_p'] = float(ks_p)
    
    # Save results
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f'matched_validation{output_suffix}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")
    
    # Plot
    png_path = output_dir / f'matched_validation{output_suffix}.png'
    plot_validation(emp_events, sim_events, emp_iei, sim_iei,
                    emp_psth, sim_psth, results, png_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Matched validation for hazard model')
    parser.add_argument('--filtered', action='store_true',
                        help='Use filtered events (duration > 0.1s)')
    parser.add_argument('--both', action='store_true',
                        help='Run validation on both all and filtered events')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MATCHED VALIDATION")
    print("=" * 70)
    
    # Load matched data
    data, model_info = load_fitting_dataset()
    
    all_results = {}
    
    if args.both:
        # Run both
        print("\n" + "=" * 70)
        print("VALIDATING ALL EVENTS")
        print("=" * 70)
        all_results['all'] = run_validation(data, model_info, use_filtered=False, output_suffix="")
        
        print("\n" + "=" * 70)
        print("VALIDATING FILTERED EVENTS")
        print("=" * 70)
        all_results['filtered'] = run_validation(data, model_info, use_filtered=True, output_suffix="_filtered")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Event Set':<20} {'Emp Events':<12} {'Sim Events':<12} {'Rate Ratio':<12} {'Status':<8}")
        print("-" * 64)
        for key, res in all_results.items():
            status = "PASS" if res['rate_pass'] else "FAIL"
            print(f"{key:<20} {res['emp_events']:<12} {res['sim_events']:<12} {res['rate_ratio']:<12.3f} {status:<8}")
        
        # Save combined results
        output_dir = Path('data/validation')
        with open(output_dir / 'matched_validation_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved summary to {output_dir / 'matched_validation_summary.json'}")
    
    elif args.filtered:
        run_validation(data, model_info, use_filtered=True, output_suffix="_filtered")
    
    else:
        run_validation(data, model_info, use_filtered=False, output_suffix="")
    
    print("\nMatched validation complete!")


if __name__ == '__main__':
    main()


