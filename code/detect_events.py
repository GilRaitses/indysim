#!/usr/bin/env python3
"""
Detect pauses and quantify turn durations from trajectory data.

Pauses: Periods where speed < threshold for minimum duration
Turn durations: Time from turn start (heading change onset) to turn end (heading stabilizes)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

def detect_pauses(speed: np.ndarray, time: np.ndarray, 
                  speed_threshold: float = 0.001,
                  min_duration: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect pause events (stops) from speed trajectory.
    
    Parameters
    ----------
    speed : ndarray
        Speed values (1D array)
    time : ndarray
        Time values (1D array)
    speed_threshold : float
        Speed threshold below which larva is considered paused (default: 0.001)
    min_duration : float
        Minimum duration for pause to be considered valid (default: 0.2 seconds)
    
    Returns
    -------
    is_pause : ndarray
        Boolean array indicating pause frames
    pause_durations : ndarray
        Duration of each pause event (for frames in pause)
    """
    is_pause = speed < speed_threshold
    
    # Find pause start/end events
    pause_starts = []
    pause_ends = []
    in_pause = False
    
    for i in range(len(is_pause)):
        if is_pause[i] and not in_pause:
            # Entering pause
            pause_starts.append(i)
            in_pause = True
        elif not is_pause[i] and in_pause:
            # Exiting pause
            pause_ends.append(i)
            in_pause = False
    
    # Handle pause that extends to end of trajectory
    if in_pause:
        pause_ends.append(len(is_pause))
    
    # Calculate pause durations and filter by minimum duration
    pause_durations = np.zeros(len(time))
    valid_pauses = np.zeros(len(time), dtype=bool)
    
    for start_idx, end_idx in zip(pause_starts, pause_ends):
        pause_duration = time[end_idx - 1] - time[start_idx] if end_idx > start_idx else 0
        
        if pause_duration >= min_duration:
            # Valid pause - mark all frames in this pause
            pause_durations[start_idx:end_idx] = pause_duration
            valid_pauses[start_idx:end_idx] = True
    
    return valid_pauses, pause_durations

def quantify_turn_durations(heading_change: np.ndarray, time: np.ndarray,
                            turn_threshold: float = np.pi/6,
                            stabilization_threshold: float = np.pi/18) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantify turn durations from heading change trajectory.
    
    A turn starts when heading_change exceeds threshold.
    A turn ends when heading_change drops below stabilization threshold.
    
    Parameters
    ----------
    heading_change : ndarray
        Heading change magnitude (1D array)
    time : ndarray
        Time values (1D array)
    turn_threshold : float
        Heading change threshold to start turn (default: π/6 = 30°)
    stabilization_threshold : float
        Heading change threshold below which turn is considered ended (default: π/18 = 10°)
    
    Returns
    -------
    turn_durations : ndarray
        Duration of each turn event (for frames in turn, 0 otherwise)
    turn_event_ids : ndarray
        Unique ID for each turn event (0 = not in turn)
    """
    is_turning = heading_change > turn_threshold
    
    # Find turn start/end events
    turn_starts = []
    turn_ends = []
    in_turn = False
    
    for i in range(len(is_turning)):
        if is_turning[i] and not in_turn:
            # Entering turn
            turn_starts.append(i)
            in_turn = True
        elif heading_change[i] < stabilization_threshold and in_turn:
            # Exiting turn (heading stabilized)
            turn_ends.append(i)
            in_turn = False
    
    # Handle turn that extends to end of trajectory
    if in_turn:
        turn_ends.append(len(is_turning))
    
    # Calculate turn durations
    turn_durations = np.zeros(len(time))
    turn_event_ids = np.zeros(len(time), dtype=int)
    
    for turn_id, (start_idx, end_idx) in enumerate(zip(turn_starts, turn_ends), 1):
        if end_idx > start_idx:
            turn_duration = time[end_idx - 1] - time[start_idx]
            # Mark all frames in this turn
            turn_durations[start_idx:end_idx] = turn_duration
            turn_event_ids[start_idx:end_idx] = turn_id
    
    return turn_durations, turn_event_ids

def detect_reverse_crawl(trajectory_df: pd.DataFrame,
                         angle_threshold: float = np.pi/2,
                         speed_threshold: float = 0.0005,
                         min_duration_frames: int = 5) -> np.ndarray:
    """
    Detect reverse crawl using Klein methodology: movement direction opposite to head-tail orientation.
    
    Methodology: Compare movement vector (velocity) with head-tail orientation vector.
    Reverse crawl occurs when movement is opposite to body orientation (> 90° angle).
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        DataFrame with columns: x, y, head_x, head_y, tail_x, tail_y, speed
    angle_threshold : float
        Angle threshold in radians (default: π/2 = 90°)
        Movement-orientation angle > threshold indicates reverse crawl
    speed_threshold : float
        Minimum speed threshold to filter stationary periods (default: 0.0005)
    min_duration_frames : int
        Minimum consecutive frames for valid reverse crawl event (default: 5)
    
    Returns
    -------
    is_reverse_crawl : ndarray
        Boolean array indicating reverse crawl frames
    """
    n_frames = len(trajectory_df)
    is_reverse_crawl = np.zeros(n_frames, dtype=bool)
    
    if n_frames < 2:
        return is_reverse_crawl
    
    # Calculate movement vectors (velocity) from centroid positions
    # Movement vector = [x(t+1) - x(t), y(t+1) - y(t)]
    dx = np.diff(trajectory_df['x'].values)
    dy = np.diff(trajectory_df['y'].values)
    movement_speeds = np.sqrt(dx**2 + dy**2)
    
    # Calculate head-tail orientation vectors
    head_tail_x = trajectory_df['head_x'].values - trajectory_df['tail_x'].values
    head_tail_y = trajectory_df['head_y'].values - trajectory_df['tail_y'].values
    head_tail_lengths = np.sqrt(head_tail_x**2 + head_tail_y**2)
    
    # Calculate angles between movement and orientation for each frame
    # Movement vector uses frame i to i+1, so we check angle at frame i
    for i in range(len(dx)):  # dx has n_frames-1 elements
        frame_idx = i  # Frame index (movement from frame i to i+1)
        
        if movement_speeds[i] < speed_threshold:
            continue  # Skip stationary frames
        
        if head_tail_lengths[frame_idx] < 1e-6:
            continue  # Skip frames with invalid head-tail vectors
        
        # Normalize vectors
        movement_norm = np.array([dx[i], dy[i]]) / movement_speeds[i]
        orientation_norm = np.array([head_tail_x[frame_idx], head_tail_y[frame_idx]]) / head_tail_lengths[frame_idx]
        
        # Calculate angle between movement and orientation vectors
        # Use absolute dot product to get angle in [0, π]
        dot_product = np.abs(np.dot(movement_norm, orientation_norm))
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid arccos input
        angle = np.arccos(dot_product)
        
        # Reverse crawl: angle > 90° (movement opposite to orientation)
        if angle > angle_threshold:
            is_reverse_crawl[frame_idx] = True
    
    # Filter by minimum duration (group consecutive frames)
    if min_duration_frames > 1:
        # Find consecutive reverse crawl segments
        reverse_segments = []
        in_segment = False
        segment_start = None
        
        for i in range(len(is_reverse_crawl)):
            if is_reverse_crawl[i] and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_reverse_crawl[i] and in_segment:
                segment_end = i - 1
                if segment_end - segment_start + 1 >= min_duration_frames:
                    reverse_segments.append((segment_start, segment_end))
                in_segment = False
        
        # Handle segment extending to end
        if in_segment and segment_start is not None:
            segment_end = len(is_reverse_crawl) - 1
            if segment_end - segment_start + 1 >= min_duration_frames:
                reverse_segments.append((segment_start, segment_end))
        
        # Reconstruct is_reverse_crawl with only valid segments
        is_reverse_crawl_filtered = np.zeros(n_frames, dtype=bool)
        for start, end in reverse_segments:
            is_reverse_crawl_filtered[start:end+1] = True
        
        is_reverse_crawl = is_reverse_crawl_filtered
    
    return is_reverse_crawl

def detect_reversals(heading_change: np.ndarray,
                    reversal_threshold: float = np.pi/2) -> np.ndarray:
    """
    DEPRECATED: Simple reversal detection based on heading change.
    
    Use detect_reverse_crawl() for proper reverse crawl detection using
    movement-orientation angle analysis (Klein methodology).
    
    Parameters
    ----------
    heading_change : ndarray
        Heading change magnitude (1D array)
    reversal_threshold : float
        Heading change threshold for reversal (default: π/2 = 90°)
    
    Returns
    -------
    is_reversal : ndarray
        Boolean array indicating reversal frames
    """
    # Simple method: large heading changes (> 90°)
    is_reversal = heading_change > reversal_threshold
    return is_reversal

def add_event_detection(trajectory_df: pd.DataFrame,
                       speed_threshold: float = 0.001,
                       pause_min_duration: float = 0.2,
                       turn_threshold: float = np.pi/6,
                       reversal_threshold: float = np.pi/2) -> pd.DataFrame:
    """
    Add pause detection and turn duration quantification to trajectory DataFrame.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory DataFrame with 'time', 'speed', 'heading_change' columns
    speed_threshold : float
        Speed threshold for pause detection
    pause_min_duration : float
        Minimum pause duration (seconds)
    turn_threshold : float
        Heading change threshold for turn detection
    reversal_threshold : float
        Heading change threshold for reversal detection
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - is_pause: boolean pause indicator
        - pause_duration: duration of pause event
        - turn_duration: duration of turn event
        - turn_event_id: unique ID for each turn
        - is_reversal: boolean reversal indicator
    """
    df = trajectory_df.copy()
    
    # Detect pauses
    if 'speed' in df.columns and 'time' in df.columns:
        is_pause, pause_durations = detect_pauses(
            df['speed'].values,
            df['time'].values,
            speed_threshold=speed_threshold,
            min_duration=pause_min_duration
        )
        df['is_pause'] = is_pause
        df['pause_duration'] = pause_durations
    else:
        df['is_pause'] = False
        df['pause_duration'] = 0.0
    
    # Quantify turn durations
    if 'heading_change' in df.columns and 'time' in df.columns:
        turn_durations, turn_event_ids = quantify_turn_durations(
            df['heading_change'].values,
            df['time'].values,
            turn_threshold=turn_threshold
        )
        df['turn_duration'] = turn_durations
        df['turn_event_id'] = turn_event_ids
        
        # Detect reverse crawl using proper Klein methodology
        # Requires: x, y, head_x, head_y, tail_x, tail_y, speed columns
        required_cols = ['x', 'y', 'head_x', 'head_y', 'tail_x', 'tail_y', 'speed']
        if all(col in df.columns for col in required_cols):
            is_reverse_crawl = detect_reverse_crawl(
                df,
                angle_threshold=np.pi/2,  # 90° threshold
                speed_threshold=0.0005,  # Minimum speed to filter noise
                min_duration_frames=5  # Minimum 5 consecutive frames
            )
            df['is_reversal'] = is_reverse_crawl
            df['is_reverse_crawl'] = is_reverse_crawl  # Alias for clarity
        else:
            # Fallback to simple heading change method if required columns missing
            is_reversal = detect_reversals(
                df['heading_change'].values,
                reversal_threshold=reversal_threshold
            )
            df['is_reversal'] = is_reversal
    else:
        df['turn_duration'] = 0.0
        df['turn_event_id'] = 0
        df['is_reversal'] = False
    
    return df

if __name__ == '__main__':
    # Test with sample data
    import numpy as np
    
    n_frames = 1000
    time = np.arange(n_frames) * 0.1  # 10 fps
    
    # Simulate speed with pauses
    speed = np.random.exponential(0.01, n_frames)
    speed[100:150] = 0.0005  # Pause
    speed[300:320] = 0.0003  # Short pause (will be filtered)
    speed[500:600] = 0.0008  # Long pause
    
    # Simulate heading changes with turns
    heading_change = np.abs(np.random.normal(0, 0.1, n_frames))
    heading_change[50:80] = np.pi/4  # Turn
    heading_change[200:250] = np.pi/3  # Longer turn
    heading_change[400:410] = np.pi  # Reversal
    
    df = pd.DataFrame({
        'time': time,
        'speed': speed,
        'heading_change': heading_change
    })
    
    df = add_event_detection(df)
    
    print("Event Detection Test:")
    print(f"  Pauses detected: {df['is_pause'].sum()}")
    print(f"  Unique pause events: {len(df[df['pause_duration'] > 0]['pause_duration'].unique())}")
    print(f"  Pause durations: {df[df['pause_duration'] > 0]['pause_duration'].unique()}")
    
    print(f"\n  Turns detected: {(df['turn_event_id'] > 0).sum()}")
    print(f"  Unique turn events: {df['turn_event_id'].max()}")
    print(f"  Turn durations: {df[df['turn_duration'] > 0]['turn_duration'].unique()}")
    
    print(f"\n  Reversals detected: {df['is_reversal'].sum()}")

