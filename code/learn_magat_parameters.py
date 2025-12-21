#!/usr/bin/env python3
"""
MAGAT Parameter Learning System

Automatically learns optimal segmentation thresholds from empirical data
by analyzing distributions and finding thresholds that detect reasonable
numbers of runs while maintaining biological plausibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from magat_segmentation import MaggotSegmentOptions

def analyze_speed_distribution(speed: np.ndarray) -> Dict[str, float]:
    """
    Analyze speed distribution to learn optimal thresholds.
    
    Returns percentiles and suggested thresholds.
    """
    speed_positive = speed[speed > 0]
    
    if len(speed_positive) == 0:
        return {
            'median': 0.0,
            'percentiles': {},
            'suggested_stop': 0.0,
            'suggested_start': 0.0
        }
    
    percentiles = {
        'p5': np.percentile(speed_positive, 5),
        'p10': np.percentile(speed_positive, 10),
        'p15': np.percentile(speed_positive, 15),
        'p25': np.percentile(speed_positive, 25),
        'p50': np.percentile(speed_positive, 50),
        'p75': np.percentile(speed_positive, 75),
        'p90': np.percentile(speed_positive, 90),
        'p95': np.percentile(speed_positive, 95)
    }
    
    median = percentiles['p50']
    
    # Suggested thresholds based on Klein methodology
    # Stop speed: 5th percentile (very slow = stop)
    # Start speed: 15th percentile (need reasonable speed to start)
    suggested_stop = max(percentiles['p5'], median * 0.1)
    suggested_start = max(percentiles['p15'], median * 0.2)
    
    return {
        'median': median,
        'percentiles': percentiles,
        'suggested_stop': suggested_stop,
        'suggested_start': suggested_start
    }

def analyze_curvature_distribution(curvature: np.ndarray) -> Dict[str, float]:
    """
    Analyze curvature distribution to learn optimal threshold.
    
    High curvature ends runs, so we want a threshold that marks only
    extreme curvature values.
    """
    curv_abs = np.abs(curvature)
    curv_positive = curv_abs[curv_abs > 0]
    
    if len(curv_positive) == 0:
        return {
            'median': 0.0,
            'percentiles': {},
            'suggested_cut': 0.4
        }
    
    percentiles = {
        'p50': np.percentile(curv_positive, 50),
        'p75': np.percentile(curv_positive, 75),
        'p90': np.percentile(curv_positive, 90),
        'p95': np.percentile(curv_positive, 95)
    }
    
    median = percentiles['p50']
    
    # Use 75th percentile - marks high curvature but not too strict
    # This means 25% of frames have high curvature (should end runs)
    suggested_cut = percentiles['p75']
    
    # But don't go below default if values are very small
    if suggested_cut < 0.1:
        suggested_cut = 0.4  # Default MAGAT value
    
    return {
        'median': median,
        'percentiles': percentiles,
        'suggested_cut': suggested_cut
    }

def analyze_body_theta_distribution(body_theta: np.ndarray) -> Dict[str, float]:
    """
    Analyze body bend angle distribution to learn optimal threshold.
    
    Head swinging (high body_theta) ends runs. We want a threshold
    that marks only extreme bends.
    """
    theta_abs = np.abs(body_theta)
    theta_positive = theta_abs[theta_abs > 0]
    
    if len(theta_positive) == 0:
        return {
            'median': 0.0,
            'percentiles': {},
            'suggested_cut': np.pi / 2
        }
    
    percentiles = {
        'p50': np.percentile(theta_positive, 50),
        'p75': np.percentile(theta_positive, 75),
        'p90': np.percentile(theta_positive, 90),
        'p95': np.percentile(theta_positive, 95)
    }
    
    median = percentiles['p50']
    
    # Use 90th percentile - only extreme bends end runs
    # This matches the current adaptive approach
    suggested_cut = min(percentiles['p90'], np.pi / 2)  # Cap at 90°
    
    return {
        'median': median,
        'percentiles': percentiles,
        'suggested_cut': suggested_cut
    }

def learn_optimal_parameters(trajectory_df: pd.DataFrame,
                            target_runs_per_minute: float = 1.0,
                            min_run_duration: float = 2.5) -> MaggotSegmentOptions:
    """
    Learn optimal MAGAT segmentation parameters from trajectory data.
    
    Uses data-driven approach to find thresholds that detect reasonable
    numbers of runs while maintaining biological plausibility.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory data with columns: time, speed, curvature, heading
        Optionally: spineTheta (body bend angle)
    target_runs_per_minute : float
        Target number of runs per minute (default: 1.0)
        Used to guide threshold selection
    min_run_duration : float
        Minimum run duration in seconds (default: 2.5)
    
    Returns
    -------
    segment_options : MaggotSegmentOptions
        Calibrated segmentation options
    """
    speed = trajectory_df['speed'].values
    curvature = trajectory_df['curvature'].values
    
    # Get body_theta if available
    if 'spineTheta' in trajectory_df.columns:
        body_theta = trajectory_df['spineTheta'].values
    elif 'spineTheta_magat' in trajectory_df.columns:
        body_theta = trajectory_df['spineTheta_magat'].values
    else:
        # Approximate from curvature
        body_theta = np.abs(curvature) * 10
    
    # Analyze distributions
    speed_analysis = analyze_speed_distribution(speed)
    curv_analysis = analyze_curvature_distribution(curvature)
    theta_analysis = analyze_body_theta_distribution(body_theta)
    
    # Create calibrated options
    options = MaggotSegmentOptions()
    
    # Speed thresholds (already adaptive in magat_segmentation.py, but set here too)
    options.stop_speed_cut = speed_analysis['suggested_stop']
    options.start_speed_cut = speed_analysis['suggested_start']
    
    # Curvature threshold
    options.curv_cut = curv_analysis['suggested_cut']
    
    # Body theta threshold (use adaptive approach from code)
    options.theta_cut = theta_analysis['suggested_cut']
    
    # Run duration (keep default)
    options.minRunTime = min_run_duration
    
    # Store analysis results for debugging
    options._learned_params = {
        'speed_analysis': speed_analysis,
        'curvature_analysis': curv_analysis,
        'theta_analysis': theta_analysis
    }
    
    return options

def tune_parameters_to_target_runs(trajectory_df: pd.DataFrame,
                                   target_runs: int,
                                   min_run_duration: float = 2.5,
                                   frame_rate: float = 10.0) -> MaggotSegmentOptions:
    """
    Iteratively tune parameters to achieve target number of runs.
    
    This uses a binary search approach to find thresholds that produce
    approximately the target number of runs.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory data
    target_runs : int
        Target number of runs to detect
    min_run_duration : float
        Minimum run duration in seconds
    frame_rate : float
        Frame rate in Hz
    
    Returns
    -------
    segment_options : MaggotSegmentOptions
        Tuned segmentation options
    """
    from magat_segmentation import magat_segment_track
    
    speed = trajectory_df['speed'].values
    curvature = trajectory_df['curvature'].values
    
    if 'spineTheta' in trajectory_df.columns:
        body_theta = trajectory_df['spineTheta'].values
    elif 'spineTheta_magat' in trajectory_df.columns:
        body_theta = trajectory_df['spineTheta_magat'].values
    else:
        body_theta = np.abs(curvature) * 10
    
    # Start with learned parameters
    options = learn_optimal_parameters(trajectory_df, min_run_duration=min_run_duration)
    
    # Prepare MAGAT DataFrame
    magat_df = pd.DataFrame({
        'time': trajectory_df['time'],
        'speed': speed,
        'curvature': curvature,
        'curv': curvature,
        'spineTheta': body_theta,
        'sspineTheta': body_theta,
        'heading': trajectory_df['heading'],
        'x': trajectory_df['x'],
        'y': trajectory_df['y']
    })
    magat_df['vel_dp'] = np.ones(len(magat_df)) * 0.707
    
    # Binary search on curvature threshold (most impactful)
    curv_min = np.percentile(np.abs(curvature[curvature != 0]), 50)  # Start at median
    curv_max = np.percentile(np.abs(curvature[curvature != 0]), 95)  # End at 95th percentile
    
    best_options = options
    best_runs = 0
    best_diff = float('inf')
    
    # Try different curvature thresholds
    for _ in range(10):  # Max 10 iterations
        curv_cut = (curv_min + curv_max) / 2
        options.curv_cut = curv_cut
        
        # Test segmentation
        segmentation = magat_segment_track(magat_df, segment_options=options, frame_rate=frame_rate)
        n_runs = segmentation['n_runs']
        
        diff = abs(n_runs - target_runs)
        if diff < best_diff:
            best_diff = diff
            best_runs = n_runs
            best_options = MaggotSegmentOptions()
            best_options.__dict__.update(options.__dict__)
        
        if n_runs < target_runs:
            # Too few runs - lower threshold (more lenient)
            curv_max = curv_cut
        elif n_runs > target_runs:
            # Too many runs - raise threshold (more strict)
            curv_min = curv_cut
        else:
            break
    
    return best_options

def print_learned_parameters(options: MaggotSegmentOptions):
    """
    Print learned parameters in a readable format.
    """
    print("\n" + "="*60)
    print("LEARNED MAGAT PARAMETERS")
    print("="*60)
    
    print(f"\nSpeed Thresholds:")
    print(f"  stop_speed_cut:  {options.stop_speed_cut:.6f}")
    print(f"  start_speed_cut: {options.start_speed_cut:.6f}")
    
    print(f"\nCurvature Threshold:")
    print(f"  curv_cut: {options.curv_cut:.6f}")
    
    print(f"\nBody Theta Threshold:")
    print(f"  theta_cut: {np.rad2deg(options.theta_cut):.2f}° ({options.theta_cut:.6f} rad)")
    
    print(f"\nRun Quality Filters:")
    print(f"  minRunTime: {options.minRunTime:.2f} seconds")
    print(f"  minRunLength: {options.minRunLength:.2f} cm")
    print(f"  minRunSpeed: {options.minRunSpeed:.6f}")
    
    if hasattr(options, '_learned_params'):
        print(f"\nDistribution Analysis:")
        speed = options._learned_params['speed_analysis']
        print(f"  Speed median: {speed['median']:.6f}")
        print(f"  Speed p5: {speed['percentiles'].get('p5', 0):.6f}")
        print(f"  Speed p15: {speed['percentiles'].get('p15', 0):.6f}")
        
        curv = options._learned_params['curvature_analysis']
        print(f"  Curvature median: {curv['median']:.6f}")
        print(f"  Curvature p75: {curv['percentiles'].get('p75', 0):.6f}")
        
        theta = options._learned_params['theta_analysis']
        print(f"  Body theta median: {np.rad2deg(theta['median']):.2f}°")
        print(f"  Body theta p90: {np.rad2deg(theta['percentiles'].get('p90', 0)):.2f}°")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Learn MAGAT parameters from data')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--track', help='Track key (default: first track)')
    parser.add_argument('--target-runs', type=int, help='Target number of runs for tuning')
    args = parser.parse_args()
    
    from engineer_dataset_from_h5 import load_h5_file, extract_trajectory_features
    
    h5_data = load_h5_file(args.h5_file)
    if args.track:
        track_key = args.track
    else:
        track_keys = list(h5_data['tracks'].keys())
        track_key = track_keys[0]
    
    track_data = h5_data['tracks'][track_key]
    df = extract_trajectory_features(track_data, frame_rate=10.0)
    
    if args.target_runs:
        options = tune_parameters_to_target_runs(df, args.target_runs)
    else:
        options = learn_optimal_parameters(df)
    
    print_learned_parameters(options)

