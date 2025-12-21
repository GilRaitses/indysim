#!/usr/bin/env python3
"""
Klein Run Table Generator

Generates Klein-style 18-column run tables from MAGAT segmentation output.
NO FALLBACKS - strict requirements, raises errors if data is missing.

Based on Mason Klein's run table methodology.
Reference: docs/mason_klein_run_table_methodology.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def wrap_angle_diff(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """
    Calculate angular difference with proper wrapping (Klein methodology).
    
    Δθ = θ₂ - θ₁
    if Δθ < -π: Δθ += 2π
    if Δθ > +π: Δθ -= 2π
    
    Parameters
    ----------
    theta1 : ndarray
        First angle(s) in radians
    theta2 : ndarray
        Second angle(s) in radians
    
    Returns
    -------
    delta_theta : ndarray
        Wrapped angular difference in [-π, +π]
    """
    delta_theta = theta2 - theta1
    delta_theta = np.where(delta_theta < -np.pi, delta_theta + 2*np.pi, delta_theta)
    delta_theta = np.where(delta_theta > np.pi, delta_theta - 2*np.pi, delta_theta)
    return delta_theta

def calculate_path_length(x: np.ndarray, y: np.ndarray, start_idx: int, end_idx: int) -> float:
    """
    Calculate path length (cumulative distance) along trajectory.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    start_idx : int
        Start frame index
    end_idx : int
        End frame index (inclusive)
    
    Returns
    -------
    path_length : float
        Cumulative distance traveled
    """
    if end_idx <= start_idx:
        return 0.0
    
    x_segment = x[start_idx:end_idx+1]
    y_segment = y[start_idx:end_idx+1]
    
    if len(x_segment) < 2:
        return 0.0
    
    dx = np.diff(x_segment)
    dy = np.diff(y_segment)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)

def associate_head_swings_with_turns(head_swings: List[Tuple[int, int]], 
                                     reorientations: List[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Associate head swings with their corresponding turns (reorientations).
    
    Parameters
    ----------
    head_swings : List[Tuple[int, int]]
        List of (start_idx, end_idx) for each head swing
    reorientations : List[Tuple[int, int]]
        List of (start_idx, end_idx) for each reorientation
    
    Returns
    -------
    turn_head_swings : Dict[int, List[Tuple[int, int]]]
        Dictionary mapping reorientation index to list of head swings within it
    """
    turn_head_swings = {i: [] for i in range(len(reorientations))}
    
    for hs_start, hs_end in head_swings:
        # Find which reorientation contains this head swing
        # Head swing is within reorientation if it overlaps or is contained
        for reo_idx, (reo_start, reo_end) in enumerate(reorientations):
            # Check if head swing overlaps with reorientation
            # Overlap: hs_start <= reo_end and hs_end >= reo_start
            if hs_start <= reo_end and hs_end >= reo_start:
                turn_head_swings[reo_idx].append((hs_start, hs_end))
                break
    
    return turn_head_swings

def calculate_first_head_swing(head_swings: List[Tuple[int, int]], 
                              heading: np.ndarray, 
                              reo_start: int, 
                              reo_end: int) -> Tuple[float, Optional[int]]:
    """
    Calculate first head swing magnitude and direction.
    
    Parameters
    ----------
    head_swings : List[Tuple[int, int]]
        List of (start_idx, end_idx) for head swings in this turn
    heading : ndarray
        Heading angles in radians
    reo_start : int
        Reorientation start frame
    reo_end : int
        Reorientation end frame
    
    Returns
    -------
    reoHS1 : float
        First head swing magnitude in radians (positive=LEFT, negative=RIGHT)
    first_hs_idx : Optional[int]
        Index of first head swing in list, or None if no head swings
    """
    if len(head_swings) == 0:
        return 0.0, None
    
    # Find first head swing (earliest start)
    first_hs = min(head_swings, key=lambda hs: hs[0])
    hs_start, hs_end = first_hs
    
    # Get heading at start and end of first head swing
    if hs_start < len(heading) and hs_end < len(heading):
        heading_start = heading[hs_start]
        heading_end = heading[hs_end]
        
        # Calculate change: positive = LEFT, negative = RIGHT
        delta_heading = wrap_angle_diff(np.array([heading_start]), np.array([heading_end]))[0]
        return delta_heading, 0
    else:
        return 0.0, None

def generate_klein_run_table(trajectory_df: pd.DataFrame,
                            segmentation: Dict,
                            track_id: int,
                            experiment_id: int = 1,
                            set_id: int = 1) -> pd.DataFrame:
    """
    Generate Klein-style 18-column run table from MAGAT segmentation.
    
    NO FALLBACKS - strict requirements:
    - trajectory_df must contain: time, x, y, heading, speed
    - segmentation must contain: runs, head_swings, reorientations
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Frame-level trajectory data with columns:
        - time: time in seconds
        - x, y: centroid positions
        - heading: heading angle in radians
        - speed: speed
    segmentation : Dict
        MAGAT segmentation output with keys:
        - runs: List[Tuple[int, int]] (start_idx, end_idx)
        - head_swings: List[Tuple[int, int]] (start_idx, end_idx)
        - reorientations: List[Tuple[int, int]] (start_idx, end_idx)
    track_id : int
        Track identifier
    experiment_id : int
        Experiment identifier (default: 1)
    set_id : int
        Set identifier (default: 1)
    
    Returns
    -------
    run_table : pd.DataFrame
        18-column Klein run table
    """
    # Validate required columns
    required_cols = ['time', 'x', 'y', 'heading']
    missing_cols = [col for col in required_cols if col not in trajectory_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trajectory_df: {missing_cols}")
    
    # Validate segmentation data
    if 'runs' not in segmentation:
        raise ValueError("segmentation must contain 'runs'")
    if 'head_swings' not in segmentation:
        raise ValueError("segmentation must contain 'head_swings'")
    if 'reorientations' not in segmentation:
        raise ValueError("segmentation must contain 'reorientations'")
    
    runs = segmentation['runs']
    head_swings = segmentation['head_swings']
    reorientations = segmentation['reorientations']
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in segmentation for track {track_id}. "
                        f"MAGAT segmentation detected {len(head_swings)} head swings and "
                        f"{len(reorientations)} reorientations, but 0 runs. "
                        f"This usually means run detection thresholds are too strict. "
                        f"Check: speed thresholds, curvature thresholds, minRunTime.")
    
    # Extract arrays
    time = trajectory_df['time'].values
    x = trajectory_df['x'].values
    y = trajectory_df['y'].values
    heading = trajectory_df['heading'].values
    
    n_frames = len(trajectory_df)
    
    # Associate head swings with reorientations
    turn_head_swings = associate_head_swings_with_turns(head_swings, reorientations)
    
    # Create run table rows
    run_table_rows = []
    
    for run_idx, (run_start, run_end) in enumerate(runs):
        # Validate indices
        if run_start < 0 or run_end >= n_frames or run_end < run_start:
            raise ValueError(f"Invalid run indices for track {track_id}, run {run_idx}: "
                           f"start={run_start}, end={run_end}, n_frames={n_frames}")
        
        # Column 1-4: Identification
        time0 = time[run_start]
        
        # Column 5: reoYN (whether run ends in a turn)
        # Check if there's a reorientation starting right after this run
        reoYN = 0
        reo_idx = None
        if run_idx < len(runs) - 1:
            # Check if next run starts right after this one (no gap = reorientation)
            next_run_start = runs[run_idx + 1][0]
            if next_run_start > run_end + 1:
                # There's a gap = reorientation exists
                # Find matching reorientation
                for reo_i, (reo_start, reo_end_reo) in enumerate(reorientations):
                    if reo_start == run_end + 1:
                        reoYN = 1
                        reo_idx = reo_i
                        break
        
        # Column 6: runQ (average direction during run)
        run_headings = heading[run_start:run_end+1]
        # Handle angle wrapping for average
        cos_mean = np.mean(np.cos(run_headings))
        sin_mean = np.mean(np.sin(run_headings))
        runQ = np.arctan2(sin_mean, cos_mean)
        
        # Column 7: runL (path length)
        runL = calculate_path_length(x, y, run_start, run_end)
        
        # Column 8: runT (duration)
        runT = time[run_end] - time[run_start]
        if runT < 0:
            raise ValueError(f"Negative run duration for track {track_id}, run {run_idx}")
        
        # Column 9: runX (end X position, redundant with column 17)
        runX = x[run_end]
        
        # Columns 10-13: Turn analysis (only if reoYN == 1)
        reo_numHS = 0
        reoQ1 = np.nan
        reoQ2 = np.nan
        reoHS1 = np.nan
        
        if reoYN == 1 and reo_idx is not None:
            # Column 10: reo#HS (number of head swings)
            hs_list = turn_head_swings.get(reo_idx, [])
            reo_numHS = len(hs_list)
            
            # Column 11: reoQ1 (direction at end of run)
            reoQ1 = heading[run_end]
            
            # Column 12: reoQ2 (direction at end of turn = start of next run)
            if run_idx + 1 < len(runs):
                next_run_start_idx = runs[run_idx + 1][0]
                if next_run_start_idx < n_frames:
                    reoQ2 = heading[next_run_start_idx]
                else:
                    raise ValueError(f"Next run start index {next_run_start_idx} >= n_frames {n_frames}")
            else:
                raise ValueError(f"Run {run_idx} has reoYN=1 but no next run")
            
            # Column 13: reoHS1 (first head swing magnitude)
            if len(hs_list) > 0:
                reo_start_reo, reo_end_reo = reorientations[reo_idx]
                reoHS1, _ = calculate_first_head_swing(hs_list, heading, reo_start_reo, reo_end_reo)
        
        # Column 14: runQ0 (direction at start of run)
        runQ0 = heading[run_start]
        
        # Columns 15-18: Spatial coordinates
        runX0 = x[run_start]
        runY0 = y[run_start]
        runX1 = x[run_end]
        runY1 = y[run_end]
        
        # Create row
        row = {
            'set': set_id,
            'expt': experiment_id,
            'track': track_id,
            'time0': time0,
            'reoYN': reoYN,
            'runQ': runQ,
            'runL': runL,
            'runT': runT,
            'runX': runX,
            'reo#HS': reo_numHS,
            'reoQ1': reoQ1,
            'reoQ2': reoQ2,
            'reoHS1': reoHS1,
            'runQ0': runQ0,
            'runX0': runX0,
            'runY0': runY0,
            'runX1': runX1,
            'runY1': runY1
        }
        
        run_table_rows.append(row)
    
    # Create DataFrame
    run_table = pd.DataFrame(run_table_rows)
    
    # Calculate derived metrics
    run_table = calculate_klein_derived_metrics(run_table)
    
    return run_table

def calculate_klein_derived_metrics(run_table: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Klein-derived metrics from run table.
    
    Adds columns:
    - turn_magnitude: Δθ = reoQ2 - reoQ1 (with angle wrapping)
    - turn_direction: 'LEFT' or 'RIGHT' based on turn_magnitude sign
    - run_efficiency: displacement / runL
    - run_displacement: Euclidean distance from start to end
    - run_drift: reoQ1 - runQ0 (with angle wrapping)
    - run_drift_direction: 'LEFT' or 'RIGHT'
    - run_speed: runL / runT
    
    Parameters
    ----------
    run_table : pd.DataFrame
        18-column Klein run table
    
    Returns
    -------
    run_table : pd.DataFrame
        Run table with added derived metrics
    """
    # Turn magnitude (only for rows with reoYN == 1)
    turn_mask = run_table['reoYN'] == 1
    if turn_mask.any():
        theta1 = run_table.loc[turn_mask, 'reoQ1'].values
        theta2 = run_table.loc[turn_mask, 'reoQ2'].values
        
        # Check for NaN values
        valid_turns = ~(np.isnan(theta1) | np.isnan(theta2))
        if not np.all(valid_turns):
            invalid_idx = np.where(turn_mask)[0][~valid_turns]
            raise ValueError(f"NaN values in reoQ1 or reoQ2 for turns at indices: {invalid_idx}")
        
        turn_magnitude = wrap_angle_diff(theta1, theta2)
        
        run_table.loc[turn_mask, 'turn_magnitude'] = turn_magnitude
        run_table.loc[~turn_mask, 'turn_magnitude'] = np.nan
        
        # Turn direction
        run_table.loc[turn_mask, 'turn_direction'] = np.where(
            turn_magnitude > 0, 'LEFT', 'RIGHT'
        )
        run_table.loc[~turn_mask, 'turn_direction'] = np.nan
    else:
        run_table['turn_magnitude'] = np.nan
        run_table['turn_direction'] = np.nan
    
    # Run efficiency and displacement
    displacement = np.sqrt(
        (run_table['runX1'] - run_table['runX0'])**2 + 
        (run_table['runY1'] - run_table['runY0'])**2
    )
    run_table['run_displacement'] = displacement
    
    # Avoid division by zero
    run_table['run_efficiency'] = np.where(
        run_table['runL'] > 0,
        displacement / run_table['runL'],
        np.nan
    )
    
    # Run drift (reoQ1 - runQ0)
    theta0 = run_table['runQ0'].values
    theta1 = run_table['reoQ1'].values
    
    # Only calculate for runs that have reoQ1 (have turns)
    valid_for_drift = ~np.isnan(theta1)
    if valid_for_drift.any():
        drift = wrap_angle_diff(theta0[valid_for_drift], theta1[valid_for_drift])
        run_table.loc[valid_for_drift, 'run_drift'] = drift
        run_table.loc[~valid_for_drift, 'run_drift'] = np.nan
        
        run_table.loc[valid_for_drift, 'run_drift_direction'] = np.where(
            drift > 0, 'LEFT', 'RIGHT'
        )
        run_table.loc[~valid_for_drift, 'run_drift_direction'] = np.nan
    else:
        run_table['run_drift'] = np.nan
        run_table['run_drift_direction'] = np.nan
    
    # Run speed
    run_table['run_speed'] = np.where(
        run_table['runT'] > 0,
        run_table['runL'] / run_table['runT'],
        np.nan
    )
    
    return run_table

def calculate_turn_rate(run_table: pd.DataFrame) -> float:
    """
    Calculate turn rate using Klein's formula.
    
    turn_rate = (Σ reoYN) / (Σ runT) * 60  # turns per minute
    
    Parameters
    ----------
    run_table : pd.DataFrame
        Klein run table
    
    Returns
    -------
    turn_rate : float
        Turns per minute
    """
    total_turns = run_table['reoYN'].sum()
    total_run_time = run_table['runT'].sum()
    
    if total_run_time <= 0:
        raise ValueError("Total run time must be > 0 for turn rate calculation")
    
    turn_rate_per_sec = total_turns / total_run_time
    turn_rate_per_min = turn_rate_per_sec * 60.0
    
    return turn_rate_per_min
