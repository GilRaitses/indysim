#!/usr/bin/env python3
"""
Ramp Detection for LED Stimulus Patterns

Classifies LED1 signal into ramp phase (0→250 PWM transition) and 
plateau phase (steady high intensity) using derivative-based detection.

Uses derivative-based detection with Gaussian smoothing.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional


def label_ramp_plateau(
    df: pd.DataFrame,
    led_col: str = 'led1Val',
    time_col: str = 'time',
    fps: float = 20.0,
    sigma_frames: float = 2.0,
    dled_thresh: float = 1.0,
    plateau_threshold: float = 200.0
) -> pd.DataFrame:
    """
    Label time points as ramp phase, plateau phase, or LED-off.
    
    Uses derivative-based detection with Gaussian smoothing to identify
    when LED1 is actively ramping vs steady at high intensity.
    
    Parameters
    ----------
    df : DataFrame
        Data with LED values and time
    led_col : str
        Name of LED1 column (default 'led1Val')
    time_col : str
        Name of time column (default 'time')
    fps : float
        Frame rate in Hz (default 20.0)
    sigma_frames : float
        Gaussian smoothing width in frames (default 2.0)
    dled_thresh : float
        Derivative threshold in PWM/frame after smoothing (default 1.0)
        For 0→250 PWM over 5s at 20Hz: avg derivative = 2.5 PWM/frame
    plateau_threshold : float
        LED value above which is considered plateau if not ramping (default 200)
    
    Returns
    -------
    df : DataFrame
        Original DataFrame with added columns:
        - 'is_ramp': 1 if in ramp phase, 0 otherwise
        - 'is_plateau': 1 if in plateau phase, 0 otherwise
        - 'led_smoothed': Smoothed LED values
        - 'led_derivative': Derivative of smoothed LED
    
    Notes
    -----
    Ramp phase: |dLED/dt| > threshold AND LED < plateau_threshold
    Plateau phase: LED >= plateau_threshold AND not ramping
    LED-off: LED < threshold AND not ramping
    """
    df = df.copy()
    
    # Get LED values
    led = df[led_col].values.astype(float)
    
    # Gaussian smooth to reduce noise
    led_smooth = gaussian_filter1d(led, sigma=sigma_frames)
    
    # Compute derivative (PWM/frame)
    dled = np.gradient(led_smooth)
    
    # Classify phases
    ramp = (np.abs(dled) > dled_thresh) & (led_smooth < plateau_threshold)
    plateau = (led_smooth >= plateau_threshold) & (~ramp)
    led_on = led_smooth > 10  # LED is on (above noise floor)
    
    # Store results
    df['is_ramp'] = ramp.astype(np.uint8)
    df['is_plateau'] = plateau.astype(np.uint8)
    df['is_led_on'] = led_on.astype(np.uint8)
    df['led_smoothed'] = led_smooth
    df['led_derivative'] = dled
    
    return df


def compute_time_since_led_onset(
    df: pd.DataFrame,
    time_col: str = 'time',
    led_on_col: str = 'is_led_on',
    group_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute time since most recent LED onset for each time point.
    
    Parameters
    ----------
    df : DataFrame
        Data with time and LED-on indicator
    time_col : str
        Name of time column
    led_on_col : str
        Name of LED-on indicator column
    group_cols : list, optional
        Columns to group by (e.g., ['experiment_id', 'track_id'])
    
    Returns
    -------
    df : DataFrame
        Original DataFrame with added 'time_since_onset' column
    """
    df = df.copy()
    
    if group_cols is None:
        group_cols = []
    
    def _compute_onset_times(group_df):
        """Find onset times and compute time since onset."""
        time = group_df[time_col].values
        led_on = group_df[led_on_col].values.astype(bool)
        
        # Detect onsets (transition from off to on)
        led_on_prev = np.roll(led_on, 1)
        led_on_prev[0] = False
        onsets = led_on & ~led_on_prev
        
        onset_times = time[onsets]
        
        # For each time point, find time since most recent onset
        time_since = np.full(len(time), np.inf)
        
        if len(onset_times) > 0:
            for i, t in enumerate(time):
                # Find most recent onset before or at this time
                past_onsets = onset_times[onset_times <= t]
                if len(past_onsets) > 0:
                    time_since[i] = t - past_onsets[-1]
        
        return time_since
    
    if group_cols:
        # Apply per group
        df['time_since_onset'] = np.nan
        for name, group in df.groupby(group_cols):
            idx = group.index
            df.loc[idx, 'time_since_onset'] = _compute_onset_times(group)
    else:
        df['time_since_onset'] = _compute_onset_times(df)
    
    return df


def visualize_ramp_detection(
    df: pd.DataFrame,
    time_col: str = 'time',
    led_col: str = 'led1Val',
    output_path: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None
):
    """
    Visualize ramp detection results.
    
    Parameters
    ----------
    df : DataFrame
        Data with ramp detection columns
    time_col : str
        Name of time column
    led_col : str
        Name of LED column
    output_path : str, optional
        Path to save figure
    time_range : tuple, optional
        (start, end) time range to plot
    """
    import matplotlib.pyplot as plt
    
    if time_range is not None:
        mask = (df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])
        plot_df = df[mask]
    else:
        plot_df = df
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    t = plot_df[time_col].values
    
    # Panel 1: Raw and smoothed LED
    axes[0].plot(t, plot_df[led_col].values, 'b-', alpha=0.5, label='Raw LED1')
    axes[0].plot(t, plot_df['led_smoothed'].values, 'r-', label='Smoothed')
    axes[0].axhline(200, color='gray', linestyle='--', label='Plateau threshold')
    axes[0].set_ylabel('LED1 (PWM)')
    axes[0].legend()
    axes[0].set_title('LED1 Signal')
    
    # Panel 2: Derivative
    axes[1].plot(t, plot_df['led_derivative'].values, 'g-')
    axes[1].axhline(1.0, color='red', linestyle='--', label='Ramp threshold')
    axes[1].axhline(-1.0, color='red', linestyle='--')
    axes[1].set_ylabel('dLED/dt (PWM/frame)')
    axes[1].legend()
    axes[1].set_title('LED1 Derivative')
    
    # Panel 3: Phase classification
    axes[2].fill_between(t, plot_df['is_ramp'].values * 0.5, alpha=0.5, 
                         label='Ramp', color='orange')
    axes[2].fill_between(t, plot_df['is_plateau'].values, alpha=0.5,
                         label='Plateau', color='green')
    axes[2].set_ylabel('Phase')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].set_title('Detected Phases')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    import argparse
    import h5py
    
    parser = argparse.ArgumentParser(description='Label ramp vs plateau phases in LED data')
    parser.add_argument('--input', type=str, default='data/processed/consolidated_dataset.h5',
                        help='Input H5 file')
    parser.add_argument('--output', type=str, default='data/processed/ramp_labeled.parquet',
                        help='Output parquet file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--dled-thresh', type=float, default=1.0,
                        help='Derivative threshold (PWM/frame)')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    with h5py.File(args.input, 'r') as f:
        if 'events' in f:
            grp = f['events']
            data = {k: grp[k][:] for k in grp.keys()}
            df = pd.DataFrame(data)
        else:
            raise ValueError("No 'events' group in H5 file")
    
    print(f"Loaded {len(df)} rows")
    print(f"LED1 range: {df['led1Val'].min():.1f} - {df['led1Val'].max():.1f}")
    
    print(f"\nLabeling ramp/plateau phases (dled_thresh={args.dled_thresh})...")
    df = label_ramp_plateau(df, dled_thresh=args.dled_thresh)
    
    print(f"  Ramp frames: {df['is_ramp'].sum()} ({100*df['is_ramp'].mean():.1f}%)")
    print(f"  Plateau frames: {df['is_plateau'].sum()} ({100*df['is_plateau'].mean():.1f}%)")
    
    print("\nComputing time since LED onset...")
    if 'experiment_id' in df.columns and 'track_id' in df.columns:
        df = compute_time_since_led_onset(df, group_cols=['experiment_id', 'track_id'])
    else:
        df = compute_time_since_led_onset(df)
    
    print(f"\nSaving to {args.output}...")
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows")
    
    if args.visualize:
        print("\nGenerating visualization...")
        # Take first experiment, first 200s
        if 'experiment_id' in df.columns:
            exp0 = df['experiment_id'].unique()[0]
            viz_df = df[df['experiment_id'] == exp0]
        else:
            viz_df = df
        
        visualize_ramp_detection(
            viz_df,
            output_path='data/validation/ramp_detection.png',
            time_range=(0, 200)
        )




