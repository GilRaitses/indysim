#!/usr/bin/env python3
"""
Generate PSTH comparison figure across all factorial conditions.

Shows turn rate (events/min) as a function of time since LED onset,
phase-locked across all stimulus cycles. Each condition in separate panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Stimulus timing
LED_CYCLE = 30.0  # seconds (10s ON, 20s OFF)
LED_ON_DURATION = 10.0  # seconds
FIRST_LED_ONSET = 30.0  # seconds

# Condition patterns
CONDITION_PATTERNS = {
    '0→250 | Constant': ('_0to250PWM', '#C_Bl_7PWM'),
    '0→250 | Cycling': ('_0to250PWM', '#T_Bl_Sq_5to15PWM'),
    '50→250 | Constant': ('50to250PWM', '#C_Bl_7PWM'),
    '50→250 | Cycling': ('50to250PWM', '#T_Bl_Sq_5to15PWM'),
}

ANOMALOUS_FILES = ['202510291652', '202510291713']


def load_experiments_by_condition(data_dir: Path = Path('data/engineered')) -> Dict[str, List[pd.DataFrame]]:
    """Load data for all conditions, keeping experiments separate for CI calculation."""
    conditions = {}
    
    for name, (intensity_pattern, bg_pattern) in CONDITION_PATTERNS.items():
        files = [
            f for f in sorted(data_dir.glob('*_events.csv'))
            if intensity_pattern in f.name and bg_pattern in f.name
            and not any(a in f.name for a in ANOMALOUS_FILES)
        ]
        
        if not files:
            print(f"  {name}: no files found")
            continue
        
        experiments = []
        for f in files:
            df = pd.read_csv(f)
            df['experiment_id'] = f.stem
            experiments.append(df)
        
        conditions[name] = experiments
        total_events = sum(df['is_reorientation_start'].sum() for df in experiments)
        print(f"  {name}: {len(files)} experiments, {total_events:.0f} events")
    
    return conditions


def compute_psth_single_experiment(df: pd.DataFrame, bin_width: float = 0.5) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute PSTH for a single experiment by averaging PER-TRACK rates.
    
    For each track:
      1. Count events in each phase bin
      2. Divide by (n_cycles * bin_width) * 60 to get events/min
    Then average across tracks.
    
    Returns:
        t_centers: bin centers (-10 to 20s, 0 = LED onset)
        mean_rate: mean rate across tracks (events/min)
        n_tracks: number of tracks
    """
    bins = np.arange(-10, 20 + bin_width, bin_width)
    t_centers = (bins[:-1] + bins[1:]) / 2
    
    if 'time_since_stimulus' not in df.columns:
        print("  WARNING: time_since_stimulus column not found!")
        return t_centers, np.zeros(len(bins) - 1), 0
    
    track_rates = []
    
    for track_id, track_df in df.groupby('track_id'):
        # Get events for this track
        events = track_df[track_df['is_reorientation_start'] == True]
        tss = events['time_since_stimulus'].values
        
        # Remap: 20-30s becomes -10 to 0 (pre-stimulus period)
        tss_remapped = np.where(tss >= 20, tss - 30, tss)
        tss_remapped = tss_remapped[(tss_remapped >= -10) & (tss_remapped <= 20)]
        
        # Bin events
        counts, _ = np.histogram(tss_remapped, bins=bins)
        
        # This track's duration and number of cycles
        track_duration = track_df['time'].max() - track_df['time'].min()
        n_cycles = max(1, track_duration / LED_CYCLE)
        
        # Rate for THIS TRACK: events/min at each phase
        # rate = counts / (time_spent_at_this_phase) * 60
        # time_spent_at_this_phase = n_cycles * bin_width
        rate = counts / (n_cycles * bin_width) * 60
        track_rates.append(rate)
    
    if not track_rates:
        return t_centers, np.zeros(len(bins) - 1), 0
    
    # Average across tracks
    track_rates = np.array(track_rates)
    mean_rate = np.mean(track_rates, axis=0)
    
    return t_centers, mean_rate, len(track_rates)


def compute_psth_with_ci(experiments: List[pd.DataFrame], bin_width: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute PSTH with confidence intervals by pooling all tracks across experiments.
    
    Each track contributes one rate estimate. CI is computed across all tracks.
    
    Returns:
        t_centers: bin centers (-10 to 20s)
        mean_rate: mean rate across all tracks (events/min)
        ci_lower: lower 95% CI
        ci_upper: upper 95% CI
        n_tracks: total number of tracks
    """
    bins = np.arange(-10, 20 + bin_width, bin_width)
    t_centers = (bins[:-1] + bins[1:]) / 2
    
    all_track_rates = []
    
    for df in experiments:
        if 'time_since_stimulus' not in df.columns:
            continue
            
        for track_id, track_df in df.groupby('track_id'):
            events = track_df[track_df['is_reorientation_start'] == True]
            tss = events['time_since_stimulus'].values
            
            # Remap: 20-30s becomes -10 to 0
            tss_remapped = np.where(tss >= 20, tss - 30, tss)
            tss_remapped = tss_remapped[(tss_remapped >= -10) & (tss_remapped <= 20)]
            
            # Bin events
            counts, _ = np.histogram(tss_remapped, bins=bins)
            
            # Track duration and cycles
            track_duration = track_df['time'].max() - track_df['time'].min()
            n_cycles = max(1, track_duration / LED_CYCLE)
            
            # Rate for this track (events/min)
            rate = counts / (n_cycles * bin_width) * 60
            all_track_rates.append(rate)
    
    if not all_track_rates:
        return t_centers, np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), 0
    
    all_track_rates = np.array(all_track_rates)
    n_tracks = len(all_track_rates)
    
    # Mean and SEM across ALL tracks
    mean_rate = np.mean(all_track_rates, axis=0)
    
    if n_tracks > 1:
        sem = stats.sem(all_track_rates, axis=0)
        ci_lower = mean_rate - 1.96 * sem
        ci_upper = mean_rate + 1.96 * sem
    else:
        ci_lower = mean_rate
        ci_upper = mean_rate
    
    return t_centers, mean_rate, ci_lower, ci_upper, n_tracks


def plot_psth_panels(conditions: Dict[str, List[pd.DataFrame]], output_path: Path):
    """Generate PSTH figure with each condition in separate panel."""
    
    n_conditions = len(conditions)
    fig, axes = plt.subplots(1, n_conditions, figsize=(4 * n_conditions, 5), sharey=True)
    
    if n_conditions == 1:
        axes = [axes]
    
    colors = {
        '0→250 | Constant': '#2166ac',
        '0→250 | Cycling': '#67a9cf',
        '50→250 | Constant': '#b2182b',
        '50→250 | Cycling': '#ef8a62',
    }
    
    # Window: -10 to +20s (10s before, 10s LED ON, 10s after)
    WINDOW_START = -10
    WINDOW_END = 20
    
    for ax, (name, experiments) in zip(axes, conditions.items()):
        t, mean_rate, ci_lower, ci_upper = compute_psth_with_ci(experiments, bin_width=1.0)
        
        color = colors.get(name, 'gray')
        
        # Plot mean with CI band
        ax.fill_between(t, ci_lower, ci_upper, alpha=0.3, color=color)
        ax.plot(t, mean_rate, color=color, linewidth=2)
        
        # Mark LED ON period
        ax.axvspan(0, LED_ON_DURATION, alpha=0.15, color='#ffeb3b')
        ax.axvline(0, color='green', linestyle='-', alpha=0.7, linewidth=1.5)
        ax.axvline(LED_ON_DURATION, color='red', linestyle='-', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time rel. to LED onset (s)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(WINDOW_START, WINDOW_END)
        
        # Add experiment count
        ax.text(0.02, 0.98, f'n={len(experiments)} expts', transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
    
    axes[0].set_ylabel('Turn rate (events/min)', fontsize=11)
    
    fig.suptitle('Peri-Stimulus Time Histogram by Condition (mean ± 95% CI)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved figure to {output_path}")


def debug_timing(conditions: Dict[str, List[pd.DataFrame]]):
    """Debug: check timing of events using time_since_stimulus column."""
    print("\n" + "=" * 60)
    print("DEBUG: Event timing (using time_since_stimulus)")
    print("=" * 60)
    
    for name, experiments in conditions.items():
        print(f"\n{name}:")
        
        # Pool all experiments
        all_tss = []
        for df in experiments:
            events = df[df['is_reorientation_start'] == True]
            if 'time_since_stimulus' in events.columns:
                all_tss.extend(events['time_since_stimulus'].values)
        
        all_tss = np.array(all_tss)
        
        # Check for clustering in each 10s window
        for window_name, start, end in [('LED ON', 0, 10), ('LED OFF early', 10, 20), ('LED OFF late', 20, 30)]:
            count = np.sum((all_tss >= start) & (all_tss < end))
            rate = count / 10  # per second (since 10s window)
            print(f"  {window_name} ({start}-{end}s): {count} events ({rate:.1f}/s)")


def main():
    print("=" * 60)
    print("PSTH COMPARISON ACROSS CONDITIONS")
    print("=" * 60)
    
    print("\nLoading data...")
    conditions = load_experiments_by_condition()
    
    if not conditions:
        print("No data found!")
        return
    
    # Debug timing
    debug_timing(conditions)
    
    # Generate figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plot_psth_panels(conditions, output_dir / 'psth_comparison.png')
    
    # Also save to docs/paper
    paper_dir = Path('docs/paper')
    if paper_dir.exists():
        plot_psth_panels(conditions, paper_dir / 'figure_psth_comparison.png')


if __name__ == '__main__':
    main()
