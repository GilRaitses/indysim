#!/usr/bin/env python3
"""
Reverse Correlation Kernel Estimation for LNP Model

Computes stimulus-triggered average (STA) of LED signal preceding 
reorientation events to reveal the empirical temporal kernel shape.

This is a diagnostic tool to:
1. Confirm biphasic response structure (early positive, late negative)
2. Determine optimal kernel window and split point
3. Verify the 0.2s peak is real behavior, not detection artifact

References:
- Hernandez-Nunez et al. 2015 (larval navigation reverse correlation)
- Gepner et al. 2015 (larval phototaxis LNP)

Usage:
    python scripts/compute_reverse_correlation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import json


def compute_reverse_correlation_kernel(
    event_times: np.ndarray,
    led_signal: np.ndarray,
    time_axis: np.ndarray,
    window: float = 6.0,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stimulus-triggered average (reverse correlation kernel).
    
    K(tau) = (1/N) * sum_i LED(t_event_i - tau)
    
    Parameters
    ----------
    event_times : ndarray
        Times of reorientation events (seconds)
    led_signal : ndarray
        LED intensity time series
    time_axis : ndarray
        Time points corresponding to led_signal
    window : float
        Window size before event (seconds)
    dt : float
        Time resolution for kernel (seconds)
    
    Returns
    -------
    tau : ndarray
        Time lags (negative = before event)
    kernel : ndarray
        Stimulus-triggered average at each lag
    """
    n_bins = int(window / dt)
    kernel = np.zeros(n_bins)
    valid_events = 0
    
    for t_event in event_times:
        # Find LED signal in [t_event - window, t_event]
        t_start = t_event - window
        
        if t_start < time_axis[0]:
            continue
        
        idx_start = np.searchsorted(time_axis, t_start)
        idx_end = np.searchsorted(time_axis, t_event)
        
        if idx_end - idx_start >= n_bins:
            # Resample to consistent resolution
            segment = led_signal[idx_start:idx_start + n_bins]
            if len(segment) == n_bins:
                kernel += segment
                valid_events += 1
    
    if valid_events > 0:
        kernel = kernel / valid_events
    
    # Time lags (negative = before event, 0 = at event)
    tau = np.linspace(-window, 0, n_bins)
    
    return tau, kernel


def compute_shuffled_baseline(
    event_times: np.ndarray,
    led_signal: np.ndarray,
    time_axis: np.ndarray,
    window: float = 6.0,
    dt: float = 0.1,
    n_shuffles: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shuffled baseline for significance testing.
    
    Shuffles event times randomly and computes kernel for each shuffle.
    
    Returns
    -------
    mean_shuffled : ndarray
        Mean shuffled kernel
    std_shuffled : ndarray
        Std of shuffled kernels (for CI)
    """
    n_bins = int(window / dt)
    shuffled_kernels = []
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_shuffles):
        # Random times within valid range
        t_min = time_axis[0] + window
        t_max = time_axis[-1]
        shuffled_times = rng.uniform(t_min, t_max, len(event_times))
        
        _, kernel = compute_reverse_correlation_kernel(
            shuffled_times, led_signal, time_axis, window, dt
        )
        shuffled_kernels.append(kernel)
    
    shuffled_kernels = np.array(shuffled_kernels)
    mean_shuffled = np.mean(shuffled_kernels, axis=0)
    std_shuffled = np.std(shuffled_kernels, axis=0)
    
    return mean_shuffled, std_shuffled


def find_zero_crossing(tau: np.ndarray, kernel: np.ndarray) -> float:
    """Find where kernel crosses from positive to negative (approximate split point)."""
    # Smooth kernel for robust detection
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(kernel, sigma=3)
    
    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(smoothed)))[0]
    
    if len(sign_changes) > 0:
        # First crossing from positive to negative
        for idx in sign_changes:
            if smoothed[idx] > 0 and smoothed[idx + 1] < 0:
                return tau[idx]
    
    return -2.0  # Default split if no crossing found


def load_empirical_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load event times and LED signal from empirical data."""
    # Load matching condition files
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        csv_files = sorted(data_dir.glob('*_events.csv'))[:2]
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Event times
    event_mask = data['is_reorientation_start'] == True
    event_times = data[event_mask]['time'].values
    
    # LED signal (sampled at data resolution)
    led_by_time = data.groupby('time').agg({
        'led1Val': 'mean'
    }).reset_index().sort_values('time')
    
    time_axis = led_by_time['time'].values
    led_signal = led_by_time['led1Val'].values
    
    return event_times, led_signal, time_axis


def plot_reverse_correlation(
    tau: np.ndarray,
    kernel: np.ndarray,
    mean_shuffled: np.ndarray,
    std_shuffled: np.ndarray,
    split_point: float,
    output_path: Path
):
    """Create diagnostic plot of reverse correlation kernel."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Raw kernel with shuffled baseline
    ax = axes[0]
    ax.fill_between(tau, mean_shuffled - 2*std_shuffled, mean_shuffled + 2*std_shuffled,
                    alpha=0.3, color='gray', label='95% CI (shuffled)')
    ax.plot(tau, kernel, 'b-', linewidth=2, label='Empirical kernel')
    ax.plot(tau, mean_shuffled, 'k--', alpha=0.5, label='Shuffled mean')
    ax.axvline(split_point, color='red', linestyle=':', label=f'Split point: {split_point:.1f}s')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time before event (s)')
    ax.set_ylabel('Mean LED intensity')
    ax.set_title('Reverse Correlation Kernel (Stimulus-Triggered Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Normalized kernel (deviation from shuffled)
    ax = axes[1]
    z_score = (kernel - mean_shuffled) / (std_shuffled + 1e-6)
    ax.fill_between(tau, -2, 2, alpha=0.2, color='gray', label='95% CI')
    ax.plot(tau, z_score, 'b-', linewidth=2, label='Z-scored kernel')
    ax.axvline(split_point, color='red', linestyle=':', label=f'Split point')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time before event (s)')
    ax.set_ylabel('Z-score')
    ax.set_title('Normalized Kernel (Significance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    print("=" * 60)
    print("REVERSE CORRELATION KERNEL ESTIMATION")
    print("=" * 60)
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    event_times, led_signal, time_axis = load_empirical_data(data_dir)
    print(f"  Events: {len(event_times)}")
    print(f"  Time range: {time_axis.min():.1f} to {time_axis.max():.1f} s")
    
    # Compute reverse correlation kernel
    print("\nComputing reverse correlation kernel...")
    tau, kernel = compute_reverse_correlation_kernel(
        event_times, led_signal, time_axis, window=6.0, dt=0.1
    )
    
    # Compute shuffled baseline
    print("Computing shuffled baseline (100 shuffles)...")
    mean_shuffled, std_shuffled = compute_shuffled_baseline(
        event_times, led_signal, time_axis, window=6.0, dt=0.1, n_shuffles=100
    )
    
    # Find zero crossing (split point)
    split_point = find_zero_crossing(tau, kernel - mean_shuffled)
    print(f"\nEstimated split point: {split_point:.2f} s")
    
    # Analyze kernel shape
    print("\nKernel analysis:")
    early_mask = tau > split_point
    late_mask = tau <= split_point
    
    early_mean = (kernel[early_mask] - mean_shuffled[early_mask]).mean()
    late_mean = (kernel[late_mask] - mean_shuffled[late_mask]).mean()
    
    print(f"  Early phase (>{split_point:.1f}s): mean deviation = {early_mean:.2f}")
    print(f"  Late phase (<={split_point:.1f}s): mean deviation = {late_mean:.2f}")
    
    if early_mean > 0 and late_mean < 0:
        print("  Shape: BIPHASIC (positive early, negative late) - as expected")
    elif early_mean > 0:
        print("  Shape: POSITIVE (early response only)")
    elif late_mean < 0:
        print("  Shape: NEGATIVE (suppression only)")
    else:
        print("  Shape: UNCLEAR")
    
    # Check for artifact at 0.2s
    peak_idx = np.argmax(kernel - mean_shuffled)
    peak_time = tau[peak_idx]
    peak_z = (kernel[peak_idx] - mean_shuffled[peak_idx]) / (std_shuffled[peak_idx] + 1e-6)
    
    print(f"\nPeak analysis:")
    print(f"  Peak at t = {peak_time:.2f} s")
    print(f"  Z-score at peak: {peak_z:.2f}")
    if abs(peak_z) > 2:
        print(f"  Peak is SIGNIFICANT (|z| > 2)")
    else:
        print(f"  Peak is NOT significant (|z| <= 2) - may be artifact")
    
    # Save results
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_reverse_correlation(
        tau, kernel, mean_shuffled, std_shuffled, split_point,
        output_dir / 'reverse_correlation_kernel.png'
    )
    
    results = {
        'split_point': float(split_point),
        'peak_time': float(peak_time),
        'peak_z_score': float(peak_z),
        'early_mean_deviation': float(early_mean),
        'late_mean_deviation': float(late_mean),
        'n_events': int(len(event_times)),
        'is_biphasic': bool(early_mean > 0 and late_mean < 0),
        'peak_significant': bool(abs(peak_z) > 2)
    }
    
    with open(output_dir / 'reverse_correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()




