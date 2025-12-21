#!/usr/bin/env python3
"""
Output Analysis for Larval Behavior Simulation (ECS630 Methodology)

Determines simulation type, replication requirements, and warm-up period:
- Simulation classification (terminating vs non-terminating)
- Replication count via exact t-distribution method
- Warm-up period determination
- CI half-width targets

Usage:
    python scripts/output_analysis.py --input data/processed/consolidated_dataset.h5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_events_from_h5(h5_path: Path) -> pd.DataFrame:
    """Load event data from consolidated H5 file."""
    print(f"Loading events from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'events' not in f:
            raise ValueError("No 'events' group in H5 file")
        
        grp = f['events']
        data = {}
        for key in grp.keys():
            arr = grp[key][:]
            if arr.dtype.kind == 'S':
                arr = arr.astype(str)
            data[key] = arr
        
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df):,} rows")
    return df


# =============================================================================
# KPI COMPUTATION
# =============================================================================

def compute_experiment_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KPIs per experiment for replication analysis.
    
    KPIs:
    - mean_turn_rate: turns per minute per track (averaged over tracks)
    - mean_latency: time to first reorientation after experiment start
    - stop_fraction: fraction of time in pause state
    - mean_run_duration: average run duration
    """
    results = []
    
    for exp_id in df['experiment_id'].unique():
        exp_df = df[df['experiment_id'] == exp_id]
        
        # Time range
        t_min = exp_df['time'].min()
        t_max = exp_df['time'].max()
        duration_min = (t_max - t_min) / 60.0
        
        # Turn rate per track
        if 'is_reorientation_start' in exp_df.columns:
            event_col = 'is_reorientation_start'
        elif 'is_reorientation' in exp_df.columns:
            # Detect onsets
            exp_df = exp_df.sort_values(['track_id', 'time'])
            exp_df['reo_onset'] = (
                exp_df.groupby('track_id')['is_reorientation']
                .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
            )
            event_col = 'reo_onset'
        else:
            event_col = None
        
        track_rates = []
        latencies = []
        
        for track_id in exp_df['track_id'].unique():
            track_df = exp_df[exp_df['track_id'] == track_id]
            track_duration = (track_df['time'].max() - track_df['time'].min()) / 60.0
            
            if event_col and track_duration > 0:
                n_events = track_df[event_col].sum()
                rate = n_events / track_duration
                track_rates.append(rate)
                
                # Latency to first event
                event_times = track_df[track_df[event_col].astype(bool)]['time']
                if len(event_times) > 0:
                    latency = event_times.min() - track_df['time'].min()
                    latencies.append(latency)
        
        # Stop fraction
        if 'is_pause' in exp_df.columns:
            stop_frac = exp_df['is_pause'].mean()
        else:
            stop_frac = 0.0
        
        # Run duration (from turn_duration or pause_duration)
        if 'turn_duration' in exp_df.columns:
            run_durations = exp_df[exp_df['turn_duration'] > 0]['turn_duration']
            mean_run_dur = run_durations.mean() if len(run_durations) > 0 else np.nan
        else:
            mean_run_dur = np.nan
        
        results.append({
            'experiment_id': str(exp_id),
            'duration_min': duration_min,
            'n_tracks': exp_df['track_id'].nunique(),
            'mean_turn_rate': np.mean(track_rates) if track_rates else np.nan,
            'std_turn_rate': np.std(track_rates) if len(track_rates) > 1 else np.nan,
            'mean_latency': np.mean(latencies) if latencies else np.nan,
            'stop_fraction': stop_frac,
            'mean_run_duration': mean_run_dur
        })
    
    return pd.DataFrame(results)


# =============================================================================
# REPLICATION COUNT (EXACT T-METHOD)
# =============================================================================

def compute_replication_count(
    sample_values: np.ndarray,
    target_half_width: float,
    confidence: float = 0.95,
    max_iterations: int = 100
) -> Dict:
    """
    Compute required replication count using exact t-distribution method.
    
    Iterates: n = ceil(t²_{n-1,1-α/2} × S² / h²) until convergence.
    
    Parameters
    ----------
    sample_values : ndarray
        Initial sample of KPI values (one per experiment)
    target_half_width : float
        Desired CI half-width
    confidence : float
        Confidence level (default 0.95)
    max_iterations : int
        Maximum iterations for convergence
    
    Returns
    -------
    result : dict
        Replication analysis results
    """
    alpha = 1 - confidence
    n0 = len(sample_values)
    
    if n0 < 2:
        return {'error': 'Need at least 2 samples'}
    
    sample_mean = np.nanmean(sample_values)
    sample_std = np.nanstd(sample_values, ddof=1)
    
    # Initial half-width
    t_crit_initial = stats.t.ppf(1 - alpha/2, df=n0-1)
    initial_half_width = t_crit_initial * sample_std / np.sqrt(n0)
    
    # Iterate to find required n
    n = n0
    iterations = []
    
    for i in range(max_iterations):
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        n_new = int(np.ceil(t_crit**2 * sample_std**2 / target_half_width**2))
        n_new = max(n_new, 2)  # At least 2 replications
        
        iterations.append({
            'iteration': i + 1,
            'n': n,
            't_critical': t_crit,
            'n_new': n_new
        })
        
        if n_new == n:
            break
        n = n_new
    
    return {
        'n0': n0,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'initial_half_width': initial_half_width,
        'target_half_width': target_half_width,
        'required_n': n,
        'additional_needed': max(0, n - n0),
        'converged': n_new == n,
        'iterations': iterations,
        'confidence': confidence
    }


def run_replication_analysis(kpi_df: pd.DataFrame) -> Dict:
    """Run replication analysis for all KPIs."""
    results = {}
    
    # Define target half-widths for each KPI
    targets = {
        'mean_turn_rate': 0.5,      # ±0.5 turns/min
        'mean_latency': 4.0,        # ±4 seconds
        'stop_fraction': 0.001,     # ±0.1%
        'mean_run_duration': 0.1    # ±0.1 seconds
    }
    
    for kpi, target_h in targets.items():
        if kpi in kpi_df.columns:
            values = kpi_df[kpi].dropna().values
            if len(values) >= 2:
                results[kpi] = compute_replication_count(values, target_h)
                results[kpi]['target_half_width'] = target_h
            else:
                results[kpi] = {'error': f'Insufficient data for {kpi}'}
    
    return results


# =============================================================================
# WARM-UP PERIOD
# =============================================================================

def compute_warmup_period(df: pd.DataFrame, bin_width: float = 30.0) -> Dict:
    """
    Determine warm-up period by analyzing turn rate over time.
    
    Uses visual inspection approach: find time when turn rate stabilizes.
    """
    # Bin time into intervals
    df = df.sort_values('time')
    t_min = df['time'].min()
    t_max = df['time'].max()
    
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    df['time_bin'] = pd.cut(df['time'], bins=bins, labels=False)
    
    # Detect reorientation onsets
    if 'is_reorientation_start' in df.columns:
        event_col = 'is_reorientation_start'
    elif 'is_reorientation' in df.columns:
        df = df.sort_values(['experiment_id', 'track_id', 'time'])
        df['reo_onset'] = (
            df.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
        event_col = 'reo_onset'
    else:
        return {'error': 'No reorientation column'}
    
    # Count events per time bin
    bin_counts = df.groupby('time_bin')[event_col].sum()
    bin_times = bins[:-1] + bin_width / 2
    
    # Normalize by number of tracks active in each bin
    track_counts = df.groupby('time_bin')['track_id'].nunique()
    
    rates = []
    times = []
    for b in range(len(bins) - 1):
        if b in bin_counts.index and b in track_counts.index:
            n_events = bin_counts[b]
            n_tracks = track_counts[b]
            duration_min = bin_width / 60.0
            if n_tracks > 0:
                rate = n_events / (n_tracks * duration_min)
                rates.append(rate)
                times.append(bin_times[b])
    
    times = np.array(times)
    rates = np.array(rates)
    
    if len(rates) < 5:
        return {'error': 'Insufficient time bins'}
    
    # Find stabilization point (when rate is within 10% of final mean)
    final_mean = np.mean(rates[-5:])  # Average of last 5 bins
    threshold = 0.1 * final_mean
    
    warmup_time = 0
    for i in range(len(rates)):
        if abs(rates[i] - final_mean) < threshold:
            # Check if subsequent values also stable
            if i + 3 < len(rates):
                subsequent = rates[i:i+3]
                if all(abs(s - final_mean) < threshold for s in subsequent):
                    warmup_time = times[i]
                    break
    
    # Default to 5 minutes if no clear stabilization
    if warmup_time == 0:
        warmup_time = 300  # 5 minutes
    
    return {
        'warmup_seconds': warmup_time,
        'warmup_minutes': warmup_time / 60.0,
        'final_mean_rate': final_mean,
        'time_series': {
            'times': times.tolist(),
            'rates': rates.tolist()
        },
        'recommendation': f"Use {warmup_time:.0f} seconds ({warmup_time/60:.1f} minutes) warm-up"
    }


# =============================================================================
# SIMULATION CLASSIFICATION
# =============================================================================

def classify_simulation(df: pd.DataFrame, kpi_df: pd.DataFrame) -> Dict:
    """
    Classify simulation as terminating or non-terminating.
    """
    # Check for natural start/stop
    experiment_durations = []
    for exp_id in df['experiment_id'].unique():
        exp_df = df[df['experiment_id'] == exp_id]
        duration = exp_df['time'].max() - exp_df['time'].min()
        experiment_durations.append(duration)
    
    mean_duration = np.mean(experiment_durations)
    std_duration = np.std(experiment_durations)
    
    # Check if durations are consistent (CV < 0.1)
    cv = std_duration / mean_duration if mean_duration > 0 else 0
    consistent_duration = cv < 0.1
    
    # Natural endpoints: experiments have fixed duration
    is_terminating = consistent_duration and mean_duration < 1800  # < 30 min
    
    return {
        'type': 'terminating' if is_terminating else 'non-terminating',
        'mean_duration_seconds': mean_duration,
        'mean_duration_minutes': mean_duration / 60.0,
        'duration_cv': cv,
        'consistent_duration': consistent_duration,
        'n_experiments': len(experiment_durations),
        'experimental_unit': 'track (independent within experiment)',
        'recommendation': (
            'Terminating simulation with fixed run length' if is_terminating
            else 'Non-terminating; use warm-up period'
        )
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_warmup_determination(warmup_results: Dict, output_path: Path):
    """Plot turn rate over time for warm-up determination."""
    if 'time_series' not in warmup_results:
        print("  No time series data for warmup plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    times = np.array(warmup_results['time_series']['times']) / 60.0  # Convert to minutes
    rates = warmup_results['time_series']['rates']
    
    ax.plot(times, rates, 'b-', lw=1.5, label='Turn rate')
    
    # Mark warm-up period
    warmup_min = warmup_results.get('warmup_minutes', 5)
    ax.axvline(x=warmup_min, color='r', linestyle='--', lw=2, 
               label=f'Warm-up end ({warmup_min:.1f} min)')
    
    # Mark final mean
    final_mean = warmup_results.get('final_mean_rate', 0)
    ax.axhline(y=final_mean, color='g', linestyle=':', lw=1.5,
               label=f'Steady-state mean ({final_mean:.2f})')
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Turn rate (events/min/track)')
    ax.set_title('Warm-Up Period Determination')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    classification: Dict,
    replication_results: Dict,
    warmup_results: Dict,
    kpi_df: pd.DataFrame,
    output_path: Path
):
    """Generate markdown report for output analysis."""
    lines = [
        "# Output Analysis Report",
        "",
        "## Simulation Classification",
        "",
        f"**Type:** {classification.get('type', 'Unknown')}",
        f"**Mean experiment duration:** {classification.get('mean_duration_minutes', 0):.1f} minutes",
        f"**Number of experiments:** {classification.get('n_experiments', 0)}",
        f"**Experimental unit:** {classification.get('experimental_unit', 'track')}",
        f"**Recommendation:** {classification.get('recommendation', 'N/A')}",
        "",
        "## Replication Requirements (Exact t-Method)",
        "",
        "| KPI | Sample S | Target h | Required n | Additional |",
        "|-----|----------|----------|------------|------------|",
    ]
    
    for kpi, result in replication_results.items():
        if 'error' not in result:
            s = result.get('sample_std', 0)
            h = result.get('target_half_width', 0)
            n = result.get('required_n', 0)
            add = result.get('additional_needed', 0)
            lines.append(f"| {kpi} | {s:.3f} | ±{h} | **{n}** | {add} |")
    
    lines.extend([
        "",
        "## Warm-Up Period",
        "",
        f"**Recommended warm-up:** {warmup_results.get('warmup_seconds', 300):.0f} seconds "
        f"({warmup_results.get('warmup_minutes', 5):.1f} minutes)",
        f"**Steady-state turn rate:** {warmup_results.get('final_mean_rate', 0):.2f} events/min/track",
        "",
        "## Experiment Summary",
        "",
    ])
    
    # Add KPI summary table
    lines.append("| Experiment | Duration (min) | Tracks | Turn Rate | Latency (s) |")
    lines.append("|------------|---------------|--------|-----------|-------------|")
    
    for _, row in kpi_df.head(10).iterrows():
        exp = row.get('experiment_id', 'N/A')[:20]
        dur = row.get('duration_min', 0)
        n_tracks = row.get('n_tracks', 0)
        rate = row.get('mean_turn_rate', np.nan)
        lat = row.get('mean_latency', np.nan)
        lines.append(f"| {exp} | {dur:.1f} | {n_tracks} | {rate:.2f} | {lat:.1f} |")
    
    if len(kpi_df) > 10:
        lines.append(f"| ... | ... | ... | ... | ... |")
        lines.append(f"| *({len(kpi_df)} total experiments)* | | | | |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Output analysis for simulation')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to consolidated H5 file')
    parser.add_argument('--output', type=str, default='data/analysis/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_events_from_h5(input_path)
    
    print("\n=== Computing KPIs ===")
    kpi_df = compute_experiment_kpis(df)
    print(f"  Computed KPIs for {len(kpi_df)} experiments")
    
    # Summary stats
    for col in ['mean_turn_rate', 'mean_latency', 'stop_fraction']:
        if col in kpi_df.columns:
            mean = kpi_df[col].mean()
            std = kpi_df[col].std()
            print(f"  {col}: {mean:.3f} ± {std:.3f}")
    
    print("\n=== Simulation Classification ===")
    classification = classify_simulation(df, kpi_df)
    print(f"  Type: {classification.get('type', 'Unknown')}")
    print(f"  Duration: {classification.get('mean_duration_minutes', 0):.1f} min")
    print(f"  Recommendation: {classification.get('recommendation', 'N/A')}")
    
    print("\n=== Replication Analysis ===")
    replication_results = run_replication_analysis(kpi_df)
    for kpi, result in replication_results.items():
        if 'error' not in result:
            n = result.get('required_n', 0)
            h = result.get('target_half_width', 0)
            print(f"  {kpi}: n = {n} for ±{h}")
    
    print("\n=== Warm-Up Period ===")
    warmup_results = compute_warmup_period(df)
    if 'error' not in warmup_results:
        print(f"  Recommended: {warmup_results.get('warmup_seconds', 300):.0f}s")
        print(f"  Steady-state rate: {warmup_results.get('final_mean_rate', 0):.2f}")
    else:
        print(f"  Warning: {warmup_results.get('error')}")
        warmup_results = {'warmup_seconds': 300, 'warmup_minutes': 5, 'final_mean_rate': 0}
    
    print("\n=== Generating Outputs ===")
    
    # Plots
    plot_warmup_determination(warmup_results, output_dir / 'warmup_determination.png')
    
    # Report
    generate_report(
        classification, replication_results, warmup_results, kpi_df,
        output_dir / 'replication_analysis.md'
    )
    
    # Save KPI table
    kpi_df.to_csv(output_dir / 'experiment_kpis.csv', index=False)
    print(f"  Saved {output_dir / 'experiment_kpis.csv'}")
    
    print("\n=== Output Analysis Complete ===")


if __name__ == '__main__':
    main()




