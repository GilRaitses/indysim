#!/usr/bin/env python3
"""
Load Fitting Data

Loads the exact dataset used to fit the hybrid hazard model:
- 2 files: *_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv
- 55 tracks: track_id 1-55
- 1407 events: is_reorientation_start == True

Usage:
    from scripts.load_fitting_data import load_fitting_dataset
    data, model_info = load_fitting_dataset()
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional


# Constants matching fit_hybrid_model.py
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3


def load_fitting_dataset(
    data_dir: str = "data/engineered",
    model_json: str = "data/model/hybrid_model_results.json",
    validate_counts: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load the exact dataset used for model fitting.
    
    This matches the data loading logic in fit_hybrid_model.py:
    1. Load files matching *_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv
    2. Use track_id 1-55
    3. Use is_reorientation_start as event flag
    
    Parameters
    ----------
    data_dir : str
        Path to engineered data directory
    model_json : str
        Path to hybrid model results JSON
    validate_counts : bool
        If True, assert that counts match expected values
    
    Returns
    -------
    data : DataFrame
        The frame-level data with columns:
        - experiment_id, track_id, time
        - is_reorientation_start (event flag)
        - led1Val, speed, curvature, etc.
    model_info : dict
        Model coefficients and track intercepts
    """
    data_path = Path(data_dir)
    
    # Load files matching the model's pattern
    csv_files = sorted(data_path.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))
    
    if not csv_files:
        # Fallback to first 2 files (same as fit_hybrid_model.py)
        csv_files = sorted(data_path.glob('*_events.csv'))[:2]
    
    if not csv_files:
        raise FileNotFoundError(f"No event CSV files found in {data_dir}")
    
    # Load and concatenate
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Load model info
    with open(model_json) as f:
        model_info = json.load(f)
    
    # Validate
    n_tracks = data['track_id'].nunique()
    n_events = data['is_reorientation_start'].sum()
    expected_tracks = model_info.get('n_tracks', 55)
    expected_events = model_info.get('validation', {}).get('emp_events', 1407)
    
    print(f"Loaded fitting dataset:")
    print(f"  Files: {[f.name for f in csv_files]}")
    print(f"  Rows: {len(data):,}")
    print(f"  Tracks: {n_tracks} (expected: {expected_tracks})")
    print(f"  Events: {n_events} (expected: {expected_events})")
    
    if validate_counts:
        assert n_tracks == expected_tracks, f"Track count mismatch: {n_tracks} vs {expected_tracks}"
        assert n_events == expected_events, f"Event count mismatch: {n_events} vs {expected_events}"
    
    return data, model_info


def get_event_times(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract event times from the dataset (all events, no filtering).
    
    Returns DataFrame with columns: track_id, time
    """
    events = data[data['is_reorientation_start'] == True][['track_id', 'time']].copy()
    events = events.sort_values(['track_id', 'time']).reset_index(drop=True)
    return events


def get_filtered_events(
    data: pd.DataFrame,
    min_duration: float = 0.1,
    min_angle: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract filtered event times from the dataset.
    
    Filters events to include only "true" reorientations with sufficient
    duration, excluding brief micro-movements and noise.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset from load_fitting_dataset()
    min_duration : float
        Minimum turn_duration to include (seconds). Default 0.1s.
        Set to 0 to include all events.
    min_angle : float, optional
        Minimum absolute turn angle to include (radians).
        Only applied if 'turn_angle' or 'reo_dtheta' column exists.
    
    Returns
    -------
    events : DataFrame
        Filtered events with columns: track_id, time, turn_duration
        (and turn_angle if available)
    
    Notes
    -----
    The original 1,407 events include 77% with turn_duration = 0.
    For trajectory simulation, filtering to duration > 0.1s yields ~319 events.
    This represents behaviorally meaningful reorientations rather than
    micro-movements or detection noise.
    """
    # Start with all reorientation events
    mask = data['is_reorientation_start'] == True
    
    # Apply duration filter
    if min_duration > 0 and 'turn_duration' in data.columns:
        mask = mask & (data['turn_duration'] > min_duration)
    
    # Apply angle filter if column exists and threshold specified
    if min_angle is not None:
        angle_col = None
        if 'turn_angle' in data.columns:
            angle_col = 'turn_angle'
        elif 'reo_dtheta' in data.columns:
            angle_col = 'reo_dtheta'
        
        if angle_col is not None:
            mask = mask & (data[angle_col].abs() > min_angle)
    
    # Select columns
    cols = ['track_id', 'time']
    if 'turn_duration' in data.columns:
        cols.append('turn_duration')
    if 'turn_angle' in data.columns:
        cols.append('turn_angle')
    elif 'reo_dtheta' in data.columns:
        cols.append('reo_dtheta')
    
    events = data[mask][cols].copy()
    events = events.sort_values(['track_id', 'time']).reset_index(drop=True)
    
    return events


def get_event_statistics(data: pd.DataFrame, filtered: bool = False) -> Dict:
    """
    Get statistics comparing all events vs filtered events.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset from load_fitting_dataset()
    filtered : bool
        If True, also compute stats for filtered events
    
    Returns
    -------
    stats : dict
        Event statistics including counts, durations, thresholds
    """
    all_events = get_event_times(data)
    
    stats = {
        'all_events': len(all_events),
        'tracks_with_events': all_events['track_id'].nunique(),
    }
    
    if 'turn_duration' in data.columns:
        events_with_dur = data[data['is_reorientation_start'] == True]
        stats['events_with_duration_gt_0'] = int((events_with_dur['turn_duration'] > 0).sum())
        stats['events_with_duration_gt_0.1'] = int((events_with_dur['turn_duration'] > 0.1).sum())
        stats['events_with_duration_gt_0.2'] = int((events_with_dur['turn_duration'] > 0.2).sum())
        stats['pct_zero_duration'] = float((events_with_dur['turn_duration'] == 0).mean() * 100)
        stats['duration_mean'] = float(events_with_dur['turn_duration'].mean())
        stats['duration_median'] = float(events_with_dur['turn_duration'].median())
    
    if filtered:
        filtered_events = get_filtered_events(data, min_duration=0.1)
        stats['filtered_events'] = len(filtered_events)
        stats['filtered_tracks'] = filtered_events['track_id'].nunique()
    
    return stats


def get_led_timing(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LED onset and offset times from the data.
    
    Uses the fixed LED cycle parameters from the model.
    
    Returns
    -------
    led_onsets : ndarray
        Times when LED turns on
    led_offsets : ndarray
        Times when LED turns off
    """
    max_time = data['time'].max()
    n_cycles = int(np.ceil((max_time - FIRST_LED_ONSET) / LED_CYCLE)) + 1
    
    led_onsets = np.array([FIRST_LED_ONSET + i * LED_CYCLE for i in range(n_cycles)])
    led_offsets = led_onsets + LED_ON_DURATION
    
    # Filter to valid range
    led_onsets = led_onsets[led_onsets < max_time]
    led_offsets = led_offsets[led_offsets < max_time]
    
    return led_onsets, led_offsets


def get_track_intercepts(model_info: Dict) -> Dict[int, float]:
    """
    Get track intercepts from model results.
    
    Returns dict mapping track_id (int) to intercept value.
    """
    raw_intercepts = model_info.get('track_intercepts', {})
    global_intercept = model_info.get('global_intercept', model_info.get('intercept_mean', -6.76))
    
    intercepts = {}
    for key, value in raw_intercepts.items():
        track_id = int(key)
        # Intercept relative to global
        intercepts[track_id] = value - global_intercept
    
    return intercepts


def compute_empirical_stats(data: pd.DataFrame) -> Dict:
    """
    Compute empirical statistics for validation comparison.
    """
    events = data[data['is_reorientation_start'] == True]
    
    # Events per track
    events_per_track = events.groupby('track_id').size()
    
    # Duration per track
    duration_per_track = data.groupby('track_id')['time'].agg(['min', 'max'])
    duration_per_track['duration'] = duration_per_track['max'] - duration_per_track['min']
    
    # Event rate per track
    rate_per_track = events_per_track / (duration_per_track['duration'] / 60)
    
    # IEI
    ieis = []
    for track_id, group in events.groupby('track_id'):
        times = group['time'].sort_values().values
        if len(times) > 1:
            ieis.extend(np.diff(times))
    ieis = np.array(ieis)
    
    return {
        'n_tracks': data['track_id'].nunique(),
        'n_events': len(events),
        'mean_events_per_track': events_per_track.mean(),
        'mean_rate_per_track': rate_per_track.mean(),
        'std_rate_per_track': rate_per_track.std(),
        'iei_mean': float(np.mean(ieis)) if len(ieis) > 0 else None,
        'iei_median': float(np.median(ieis)) if len(ieis) > 0 else None,
        'iei_cv': float(np.std(ieis) / np.mean(ieis)) if len(ieis) > 0 else None,
    }


def main():
    """Test the data loader."""
    print("=" * 70)
    print("LOAD FITTING DATA TEST")
    print("=" * 70)
    
    data, model_info = load_fitting_dataset()
    
    print("\n" + "=" * 50)
    print("EMPIRICAL STATISTICS")
    print("=" * 50)
    
    stats = compute_empirical_stats(data)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("EVENT FILTERING")
    print("=" * 50)
    
    event_stats = get_event_statistics(data, filtered=True)
    print(f"  All events: {event_stats['all_events']}")
    print(f"  Zero-duration events: {event_stats.get('pct_zero_duration', 0):.1f}%")
    print(f"  Events with duration > 0: {event_stats.get('events_with_duration_gt_0', 'N/A')}")
    print(f"  Events with duration > 0.1s: {event_stats.get('events_with_duration_gt_0.1', 'N/A')}")
    print(f"  Filtered events (min_duration=0.1s): {event_stats.get('filtered_events', 'N/A')}")
    
    # Show filtered events example
    filtered = get_filtered_events(data, min_duration=0.1)
    print(f"\n  Filtered event sample:")
    print(filtered.head(5).to_string(index=False))
    
    print("\n" + "=" * 50)
    print("LED TIMING")
    print("=" * 50)
    
    led_onsets, led_offsets = get_led_timing(data)
    print(f"  First LED onset: {led_onsets[0]:.1f}s")
    print(f"  LED cycle: {LED_CYCLE:.0f}s ({LED_ON_DURATION:.0f}s ON / {LED_OFF_DURATION:.0f}s OFF)")
    print(f"  Number of cycles: {len(led_onsets)}")
    
    print("\n" + "=" * 50)
    print("TRACK INTERCEPTS")
    print("=" * 50)
    
    intercepts = get_track_intercepts(model_info)
    print(f"  Tracks with intercepts: {len(intercepts)}")
    print(f"  Range: [{min(intercepts.values()):.3f}, {max(intercepts.values()):.3f}]")
    
    print("\nData loader ready!")


if __name__ == '__main__':
    main()


