#!/usr/bin/env python3
"""
Engineer dataset from H5 files for term project modeling.

Extracts trajectory data, stimulus timing, and behavioral events from H5 files
and creates feature matrices suitable for LNP model fitting.

GOLD STANDARD: tier2_complete.h5 format
- LED data: global_quantities/led1Val/yData and global_quantities/led2Val/yData
- Track structure: tracks/track_N/points/{head,mid,tail} and derived_quantities/{speed,theta,curv}
- Creates addTonToff-equivalent fields: led1Val_ton, led1Val_toff, led2Val_ton, led2Val_toff

Usage:
    python scripts/engineer_dataset_from_h5.py \
        --h5-dir /Users/gilraitses/mechanosensation/h5tests \
        --output-dir data/engineered \
        --experiment-id GMR61_202509051201
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("WARNING: h5py not installed. Install with: pip install h5py")

# Reverse crawl detection via retrovibez (Mason Klein methodology)
# Uses derived_quantities/shead, smid, sloc, eti (NOT points/head, points/mid)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _retrovibez_path = _Path(__file__).parent.parent / 'external' / 'retrovibez'
    if _retrovibez_path.exists() and str(_retrovibez_path) not in _sys.path:
        _sys.path.insert(0, str(_retrovibez_path))
    from core.h5_reader import compute_speed_run_vel, detect_reversals
    HAS_RETROVIBEZ = True
except ImportError:
    HAS_RETROVIBEZ = False

def load_h5_file(h5_path: Path) -> Dict:
    """Load H5 file and return dictionary of datasets."""
    if not HAS_H5PY:
        raise ImportError("h5py required. Install with: pip install h5py")
    
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # CRITICAL POLICY: ETI MUST ALWAYS BE LOADED FROM ROOT LEVEL
        # See docs/ETI_TIME_CALCULATION_POLICY.md
        if 'eti' in f:
            data['eti'] = f['eti'][:]  # Load ETI array
        else:
            raise ValueError(f"CRITICAL ERROR: ETI not found at root level in {h5_path.name}. "
                           "ETI is REQUIRED for time calculation. "
                           "See docs/ETI_TIME_CALCULATION_POLICY.md")
        
        # Load metadata
        if 'metadata' in f:
            data['metadata'] = dict(f['metadata'].attrs)
        
        # Load tracks
        if 'tracks' in f:
            data['tracks'] = {}
            for track_key in f['tracks'].keys():
                track_group = f[f'tracks/{track_key}']
                track_data = {}
                
                # Load position data (head, mid, tail, and spine points)
                # Gold standard (tier2_complete): positions in points/head, points/mid, points/tail
                # Also includes spine_points and spine_indices for full spine tracking
                if 'points' in track_group and isinstance(track_group['points'], h5py.Group):
                    # Tier2 structure: positions in points/head, points/mid, points/tail
                    points_group = track_group['points']
                    for pos_key in ['head', 'mid', 'tail']:
                        if pos_key in points_group:
                            track_data[pos_key] = points_group[pos_key][:]
                    
                    # Load spine points (multiple points per frame)
                    if 'spine_points' in points_group:
                        spine_data = points_group['spine_points'][:]
                        # Ensure it's a 2D array (N_points, 2)
                        if spine_data.size > 0:
                            if spine_data.ndim == 1:
                                # Reshape if needed: might be (2*N,) -> (N, 2)
                                if len(spine_data) % 2 == 0:
                                    spine_data = spine_data.reshape(-1, 2)
                            track_data['spine_points'] = spine_data
                        else:
                            print(f"      Warning: Empty spine_points for {track_key}")
                    if 'spine_indices' in points_group:
                        idx_data = points_group['spine_indices'][:]
                        if idx_data.size > 0:
                            track_data['spine_indices'] = idx_data.flatten()
                        else:
                            print(f"      Warning: Empty spine_indices for {track_key}")
                else:
                    # Tier1 fallback: positions directly in track_group
                    for pos_key in ['head', 'mid', 'tail']:
                        if pos_key in track_group:
                            track_data[pos_key] = track_group[pos_key][:]
                    # Spine points may not be available in Tier1
                    if 'spine_points' in track_group:
                        spine_data = track_group['spine_points'][:]
                        # Ensure it's a 2D array (N_points, 2)
                        if spine_data.size > 0:
                            if spine_data.ndim == 1:
                                # Reshape if needed: might be (2*N,) -> (N, 2)
                                if len(spine_data) % 2 == 0:
                                    spine_data = spine_data.reshape(-1, 2)
                            track_data['spine_points'] = spine_data
                        else:
                            print(f"      Warning: Empty spine_points for {track_key}")
                    if 'spine_indices' in track_group:
                        idx_data = track_group['spine_indices'][:]
                        if idx_data.size > 0:
                            track_data['spine_indices'] = idx_data.flatten()
                        else:
                            print(f"      Warning: Empty spine_indices for {track_key}")
                
                # Load derived features if available
                # Gold standard (tier2_complete): uses 'derived_quantities' with speed, theta, curv
                derived_group = None
                if 'derived_quantities' in track_group:
                    # Tier2 structure (gold standard)
                    derived_group = track_group['derived_quantities']
                elif 'derived' in track_group:
                    # Tier1 fallback structure
                    derived_group = track_group['derived']
                
                if derived_group is not None:
                    track_data['derived'] = {}
                    for derived_key in derived_group.keys():
                        if isinstance(derived_group[derived_key], h5py.Dataset):
                            track_data['derived'][derived_key] = derived_group[derived_key][:]
                
                # Load track attributes
                track_data['attrs'] = dict(track_group.attrs)
                
                # Load track metadata (contains startFrame, endFrame for ETI mapping)
                if 'metadata' in track_group:
                    metadata_group = track_group['metadata']
                    if isinstance(metadata_group, h5py.Group) and metadata_group.attrs:
                        track_data['metadata_attrs'] = dict(metadata_group.attrs)
                    elif isinstance(metadata_group, h5py.Dataset):
                        # Some H5 files might have metadata as dataset
                        track_data['metadata_attrs'] = {}
                
                data['tracks'][track_key] = track_data
        
        # Load LED data - check multiple possible locations
        # Gold standard (tier2_complete): global_quantities/led1Val/yData and global_quantities/led2Val/yData
        led1_found = False
        led2_found = False
        
        # Method 1: Check global_quantities (gold standard tier2_complete format)
        if 'global_quantities' in f:
            gq = f['global_quantities']
            
            # Helper to load LED field (handles both Group and Dataset)
            def load_led_field(gq, field_name):
                if field_name not in gq:
                    return None
                item = gq[field_name]
                if isinstance(item, h5py.Group) and 'yData' in item:
                    return item['yData'][:]
                elif isinstance(item, h5py.Dataset):
                    return item[:]
                return None
            
            # Load LED1 value, derivative, and diff
            led1_val = load_led_field(gq, 'led1Val')
            if led1_val is not None:
                data['led1Val'] = led1_val
                led1_found = True
            data['led1ValDeriv'] = load_led_field(gq, 'led1ValDeriv')
            data['led1ValDiff'] = load_led_field(gq, 'led1ValDiff')
            
            # Load LED2 value, derivative, and diff
            led2_val = load_led_field(gq, 'led2Val')
            if led2_val is not None:
                data['led2Val'] = led2_val
                led2_found = True
            data['led2ValDeriv'] = load_led_field(gq, 'led2ValDeriv')
            data['led2ValDiff'] = load_led_field(gq, 'led2ValDiff')
        
        # Method 2: Check top-level led_data (fallback for led1Val)
        if not led1_found and 'led_data' in f:
            data['led_data'] = f['led_data'][:]
            data['led1Val'] = f['led_data'][:]
            led1_found = True
        
        # Method 3: Check separate led2_data dataset
        if not led2_found and 'led2_data' in f:
            data['led2Val'] = f['led2_data'][:]
            led2_found = True
        
        # If led1Val found but led2Val not found, create placeholder
        if led1_found and not led2_found:
            n_frames = len(data['led1Val'])
            data['led2Val'] = np.zeros(n_frames)  # Placeholder
            print(f"  WARNING: led2Val not found, creating zero placeholder")
        
        # Also store led_data for backwards compatibility
        if 'led1Val' in data and 'led_data' not in data:
            data['led_data'] = data['led1Val']
        
        # Load stimulus group (onset frames/times)
        if 'stimulus' in f:
            data['stimulus'] = {}
            for subkey in f['stimulus'].keys():
                if isinstance(f['stimulus'][subkey], h5py.Dataset):
                    data['stimulus'][subkey] = f['stimulus'][subkey][:]
        
        # Load metadata attributes
        if 'metadata' in f:
            data['metadata'] = {'attrs': dict(f['metadata'].attrs)}
        
        # Load derivation_rules from root (added by MagatFairy 2025-12-10)
        # These are required for MAGAT segmentation head-swing buffer calculation
        if 'derivation_rules' in f:
            dr_group = f['derivation_rules']
            data['derivation_rules'] = {
                'smoothTime': float(dr_group.attrs.get('smoothTime', 0.2)),
                'derivTime': float(dr_group.attrs.get('derivTime', 0.1)),
                'interpTime': float(dr_group.attrs.get('interpTime', 0.05))
            }
            print(f"  Loaded derivation_rules: smoothTime={data['derivation_rules']['smoothTime']:.3f}s, "
                  f"derivTime={data['derivation_rules']['derivTime']:.3f}s, "
                  f"interpTime={data['derivation_rules']['interpTime']:.4f}s")
        else:
            # Use sensible defaults (matches MagatFairy fallback)
            data['derivation_rules'] = {
                'smoothTime': 0.2,
                'derivTime': 0.1,
                'interpTime': 0.05
            }
            print(f"  WARNING: derivation_rules not found in H5, using defaults")
    
    return data

def extract_trajectory_features(track_data: Dict, frame_rate: float = 10.0, eti: np.ndarray = None, track_frame_indices: np.ndarray = None, derivation_rules: Dict = None) -> pd.DataFrame:
    """
    Extract trajectory features from track data.
    
    Based on actual H5 structure: head, mid, tail positions and derived features.
    
    Parameters
    ----------
    track_data : dict
        Track data dictionary with head/mid/tail positions and derived features
    frame_rate : float
        Frame rate in Hz (default 10 fps from H5 metadata)
    eti : ndarray, optional
        Experiment Time Index array from H5 root. If provided, use ETI for time calculation.
    track_frame_indices : ndarray, optional
        Frame indices mapping track frames to ETI indices. If None, assumes continuous frames.
    derivation_rules : dict, optional
        MAGAT derivation rules with smoothTime, derivTime, interpTime.
        Required for accurate head-swing buffer calculation in segmentation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, x, y, speed, heading, etc.
    """
    # Use mid position as primary location (centroid)
    if 'mid' not in track_data:
        return pd.DataFrame()
    
    mid_pos = track_data['mid']  # N×2 array (x, y)
    n_frames = len(mid_pos)
    
    if n_frames == 0:
        return pd.DataFrame()
    
    x = mid_pos[:, 0]
    y = mid_pos[:, 1]
    
    # CRITICAL POLICY: ETI MUST ALWAYS BE USED FOR TIME CALCULATION
    # Using frame_rate-based time calculation is FORBIDDEN (causes 37+ minute tracks)
    # See docs/ETI_TIME_CALCULATION_POLICY.md
    
    # PREFERRED: Use track-level ETI from derived_quantities (uniform interpolated time)
    # This matches track data exactly since MATLAB interpolates to interpTime grid
    # See docs/logs/2025-12-10/ETI-OVERFLOW-RESEARCH-PROMPT.md for analysis
    track_eti_used = False
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'eti' in track_data['derived']:
            track_eti = track_data['derived']['eti']
            # Flatten if 2D (tier2 structure: shape (1, N))
            if track_eti.ndim > 1:
                track_eti = track_eti.flatten()
            
            if len(track_eti) == n_frames:
                # Perfect match - use track-level ETI directly
                time = track_eti.copy()
                track_eti_used = True
            elif len(track_eti) >= n_frames:
                # Track ETI longer than position data - use first n_frames
                time = track_eti[:n_frames].copy()
                track_eti_used = True
    
    # FALLBACK: Map to global ETI if track-level ETI not available
    if not track_eti_used:
        if eti is None:
            raise ValueError("CRITICAL ERROR: ETI is REQUIRED for time calculation. "
                            "Neither track-level nor global ETI available. "
                            "See docs/ETI_TIME_CALCULATION_POLICY.md")
        
        # Map track frames to global ETI indices using track metadata
        if track_frame_indices is not None and len(track_frame_indices) == n_frames:
            # Track has explicit frame indices mapping to ETI (provided externally)
            time = eti[track_frame_indices]
        elif 'metadata_attrs' in track_data and 'startFrame' in track_data['metadata_attrs']:
            # Use startFrame from metadata to map track frames to ETI indices
            start_frame = int(track_data['metadata_attrs']['startFrame'])
            track_eti_indices = np.arange(start_frame, start_frame + n_frames, dtype=int)
            
            max_eti_index = track_eti_indices[-1]
            eti_length = len(eti)
            
            if max_eti_index >= eti_length:
                # Track has more frames than global ETI - handle overflow gracefully
                n_overflow = max_eti_index - eti_length + 1
                n_valid = n_frames - n_overflow
                
                print(f"    WARNING: Track has {n_frames} frames but global ETI has {eti_length} elements. "
                      f"Track frames exceed ETI by {n_overflow} frames. "
                      f"Using global ETI for {n_valid} frames, last ETI value for overflow frames.")
                
                time = np.zeros(n_frames)
                valid_mask = track_eti_indices < eti_length
                valid_indices = track_eti_indices[valid_mask]
                
                if len(valid_indices) > 0:
                    time[:len(valid_indices)] = eti[valid_indices]
                    if len(valid_indices) < n_frames:
                        time[len(valid_indices):] = eti[-1]
                else:
                    raise ValueError(f"CRITICAL ERROR: No valid ETI indices for track. "
                                   f"startFrame={start_frame}, n_frames={n_frames}, ETI length={eti_length}")
            else:
                time = eti[track_eti_indices].copy()
        elif len(eti) == n_frames:
            time = eti.copy()
        else:
            raise ValueError(f"CRITICAL ERROR: Cannot map track frames to ETI indices. "
                           f"ETI length={len(eti)}, track frames={n_frames}. "
                           f"Track metadata must contain 'startFrame' attribute. "
                           f"See docs/ETI_TIME_CALCULATION_POLICY.md")
    
    # CRITICAL VALIDATION: Check if calculated duration exceeds expected experiment duration (20 minutes = 1200 seconds)
    # Experiments are exactly 20 minutes (1200 seconds), so allow max_time <= 1200
    max_time = time.max() if len(time) > 0 else 0
    if max_time > 1200.1:  # Allow up to 1200 seconds (20 min) with small tolerance for floating point
        raise ValueError(f"CRITICAL ERROR: Time exceeds 20 minutes: {max_time:.1f}s ({max_time/60:.1f} min). "
                        f"Experiments are exactly 20 minutes (1200s) long. This indicates ETI data corruption or incorrect usage.")
    
    # Use pre-computed derived features if available (tier2_complete is gold standard)
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        derived = track_data['derived']
        if 'speed' in derived:
            speed = derived['speed']
            # Tier2 structure: speed is shape (1, N), flatten to (N,)
            if speed.ndim > 1:
                speed = speed.flatten()
            # Ensure correct length
            if len(speed) > n_frames:
                speed = speed[:n_frames]
            elif len(speed) < n_frames:
                # Pad with last value if shorter
                speed = np.pad(speed, (0, n_frames - len(speed)), mode='edge')
        else:
            # Compute speed from positions
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            dt = np.diff(time, prepend=time[0])
            speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        
        if 'direction' in derived:
            heading = derived['direction']
            # Tier2 structure: may be 2D, flatten
            if heading.ndim > 1:
                heading = heading.flatten()
            if len(heading) > n_frames:
                heading = heading[:n_frames]
            elif len(heading) < n_frames:
                heading = np.pad(heading, (0, n_frames - len(heading)), mode='edge')
        elif 'theta' in derived:
            # Tier2 uses 'theta' instead of 'direction'
            heading = derived['theta']
            if heading.ndim > 1:
                heading = heading.flatten()
            if len(heading) > n_frames:
                heading = heading[:n_frames]
            elif len(heading) < n_frames:
                heading = np.pad(heading, (0, n_frames - len(heading)), mode='edge')
        else:
            # Compute heading from positions
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            heading = np.arctan2(dy, dx)
            heading = np.concatenate([[heading[0]], heading[1:]])
        
        if 'curvature' in derived:
            curvature = derived['curvature']
            if curvature.ndim > 1:
                curvature = curvature.flatten()
            if len(curvature) > n_frames:
                curvature = curvature[:n_frames]
            elif len(curvature) < n_frames:
                curvature = np.pad(curvature, (0, n_frames - len(curvature)), mode='constant', constant_values=0)
        elif 'curv' in derived:
            # Tier2 uses 'curv' instead of 'curvature'
            curvature = derived['curv']
            if curvature.ndim > 1:
                curvature = curvature.flatten()
            if len(curvature) > n_frames:
                curvature = curvature[:n_frames]
            elif len(curvature) < n_frames:
                curvature = np.pad(curvature, (0, n_frames - len(curvature)), mode='constant', constant_values=0)
        else:
            curvature = np.zeros(n_frames)
    else:
        # Compute speed using MAGAT algorithm (if not already computed)
        if not speed_magat_computed:
            try:
                import sys
                from pathlib import Path
                script_dir = Path(__file__).parent
                sys.path.insert(0, str(script_dir))
                from magat_speed_analysis import calculate_speed_magat
                
                # Compute MAGAT speed from positions
                positions = np.array([x, y])  # (2, n_frames)
                interp_time = np.mean(np.diff(time)) if len(time) > 1 else 0.1
                smooth_time = 0.1  # MAGAT default
                deriv_time = 0.1   # MAGAT default
                
                speed, velocity, smoothed_locs = calculate_speed_magat(
                    positions, interp_time, smooth_time, deriv_time
                )
                speed_magat_computed = True
                
            except Exception as e:
                # Fallback to simple diff-based speed
                print(f"  Warning: MAGAT speed calculation failed ({e}), using simple diff")
                dx = np.diff(x, prepend=x[0])
                dy = np.diff(y, prepend=y[0])
                dt = np.diff(time, prepend=time[0])
                speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        
        # Compute heading and curvature
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        heading = np.arctan2(dy, dx)
        heading = np.concatenate([[heading[0]], heading[1:]])
        curvature = np.zeros(n_frames)
    
    # Ensure speed is 1D before computing acceleration
    if speed.ndim > 1:
        speed = speed.flatten()
    if len(speed) != n_frames:
        speed = speed[:n_frames] if len(speed) > n_frames else np.pad(speed, (0, n_frames - len(speed)), mode='edge')
    
    # Compute acceleration
    dt = np.diff(time, prepend=time[0])
    accel = np.diff(speed, prepend=speed[0]) / np.maximum(dt, 1e-6)
    
    # Extract deltatheta (heading change per frame) from H5 if available
    # This is the basis for reorientation detection (more accurate than diff(heading))
    # 
    # NOTE: In MAGAT (https://github.com/GilRaitses/magniphyq), reorientations are accessed as 
    # track.reorientation (abbreviated "reo") and are MaggotReorientation objects with startInd/endInd.
    # In tier2_complete H5, reorientations are not pre-computed, so we detect them from deltatheta
    # using MAGAT-compatible thresholds. MAGAT's reorientation detection algorithm uses deltatheta
    # along with speed thresholds and head swing detection (numHS >= 1).
    #
    # CRITICAL: We now also calculate MATLAB-compatible reorientation angles (reo_dtheta) using
    # the same formula as MATLAB spatial calculations: diff(unwrap([prevDir;nextDir]))
    # This is calculated in magat_segmentation.py and stored in the DataFrame as 'reo_dtheta'
    #
    # MAGAT Reference: @MaggotReorientation class in magniphyq repository
    # MATLAB Reference: spatialMaggotCalculations.m line 175
    deltatheta = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'deltatheta' in track_data['derived']:
            deltatheta_raw = track_data['derived']['deltatheta']
            # Flatten if 2D (tier2 structure: (1, N))
            if deltatheta_raw.ndim > 1:
                deltatheta_raw = deltatheta_raw.flatten()
            # Ensure correct length
            if len(deltatheta_raw) >= n_frames:
                deltatheta = deltatheta_raw[:n_frames]  # Keep sign for directional info
            else:
                # Pad with zeros if shorter
                deltatheta = np.pad(deltatheta_raw, (0, n_frames - len(deltatheta_raw)), mode='constant')
        # TODO: If tier2_complete adds a 'reo' or 'reorientation' field (from MAGAT), use it directly
    
    # Compute heading_change: use deltatheta if available, otherwise compute from heading
    if deltatheta is not None:
        # Use pre-computed deltatheta from H5 (more accurate, already accounts for angle wrapping)
        # Keep absolute value for magnitude
        heading_change = np.abs(deltatheta)
    else:
        # Fallback: compute from heading differences
        heading_change = np.abs(np.diff(heading, prepend=heading[0]))
        # Wrap angles to [-pi, pi]
        heading_change = np.mod(heading_change + np.pi, 2*np.pi) - np.pi
        heading_change = np.abs(heading_change)
    
    # Initialize MAGAT spine analysis flags (computed in spine analysis section below)
    spine_theta_magat_computed = False
    spine_theta_magat = None
    spine_theta_smoothed = None
    
    # Initialize MAGAT segmentation (will be set if segmentation succeeds)
    magat_segmentation = None
    
    # Extract spineTheta (body bend angle) from derived quantities if available (for fallback)
    spine_theta = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'sspineTheta' in track_data['derived']:
            spine_theta_raw = track_data['derived']['sspineTheta']
            if spine_theta_raw.ndim > 1:
                spine_theta_raw = spine_theta_raw.flatten()
            if len(spine_theta_raw) >= n_frames:
                spine_theta = spine_theta_raw[:n_frames]
        elif 'spineTheta' in track_data['derived']:
            spine_theta_raw = track_data['derived']['spineTheta']
            if spine_theta_raw.ndim > 1:
                spine_theta_raw = spine_theta_raw.flatten()
            if len(spine_theta_raw) >= n_frames:
                spine_theta = spine_theta_raw[:n_frames]
    
    # Get vel_dp (velocity dot product) if available
    vel_dp = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'vel_dp' in track_data['derived']:
            vel_dp_raw = track_data['derived']['vel_dp']
            if vel_dp_raw.ndim > 1:
                vel_dp_raw = vel_dp_raw.flatten()
            if len(vel_dp_raw) >= n_frames:
                vel_dp = vel_dp_raw[:n_frames]
    
    # Simple turn detection for backwards compatibility (will be replaced by segmentation-based turns)
    is_turn_simple = heading_change > np.pi/6  # 30 degrees threshold (simpler detection)
    
    # Extract spine points per frame (if available)
    # Multiple spine points allow more accurate curvature computation
    spine_points_per_frame = None
    if 'spine_points' in track_data and 'spine_indices' in track_data:
        spine_points_all = track_data['spine_points']  # (N_total, 2) array
        spine_indices = track_data['spine_indices'].astype(int)  # Indices marking frame boundaries
        
        # Debug: print spine data info
        print(f"    Loading spine points: {len(spine_points_all)} total points, {len(spine_indices)} frame indices")
        
        # Extract spine points for each frame
        if len(spine_indices) > 1:
            n_spine_points_per_frame = int(spine_indices[1] - spine_indices[0])
        else:
            # Fallback: estimate from total points
            n_spine_points_per_frame = len(spine_points_all) // max(n_frames, 1) if n_frames > 0 else 11
            if n_spine_points_per_frame == 0:
                n_spine_points_per_frame = 11  # Default
        
        print(f"    Spine points per frame: {n_spine_points_per_frame}")
        spine_points_per_frame = np.zeros((n_frames, n_spine_points_per_frame, 2))
        
        for i in range(n_frames):
            if i < len(spine_indices) - 1:
                idx_start = spine_indices[i]
                idx_end = spine_indices[i + 1]
                if idx_end <= len(spine_points_all) and idx_start < len(spine_points_all):
                    frame_spine = spine_points_all[idx_start:idx_end]
                    if len(frame_spine) == n_spine_points_per_frame:
                        spine_points_per_frame[i] = frame_spine
                    elif len(frame_spine) > 0:
                        # Handle variable-length spines (pad or truncate)
                        min_len = min(len(frame_spine), n_spine_points_per_frame)
                        spine_points_per_frame[i, :min_len] = frame_spine[:min_len]
        
        frames_with_spines = np.sum(np.any(spine_points_per_frame != 0, axis=(1,2)))
        print(f"    Extracted spine points for {frames_with_spines}/{n_frames} frames")
        
        if frames_with_spines == 0:
            track_id = track_data.get('attrs', {}).get('id', 'unknown')
            raise ValueError(f"No valid spine points extracted from track (id={track_id}). "
                           f"Check that spine_points and spine_indices are correctly formatted in H5 file.")
    else:
        missing = []
        if 'spine_points' not in track_data:
            missing.append('spine_points')
        if 'spine_indices' not in track_data:
            missing.append('spine_indices')
        track_id = track_data.get('attrs', {}).get('id', 'unknown')
        raise ValueError(f"REQUIRED: Spine data not available in H5 file for track (id={track_id}). "
                        f"Missing: {missing}. "
                        f"The H5 file must contain spine_points and spine_indices in tracks/track_*/points/")
    
    # Compute spineTheta and curvature using MAGAT algorithms
    # MAGAT Reference: @MaggotTrack/calculateDerivedQuantity.m
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from magat_spine_analysis import calculate_spine_theta_magat, calculate_spine_curv_magat, calculate_spine_curve_energy_magat, lowpass1d
        
        if spine_points_per_frame is not None:
            # Use MAGAT's algorithm to compute spineTheta (body bend angle)
            spine_theta_magat = calculate_spine_theta_magat(spine_points_per_frame)  # (n_frames,)
            
            # Use MAGAT's algorithm to compute spineCurv
            spine_curv_magat = calculate_spine_curv_magat(spine_points_per_frame)  # (n_frames,)
            
            # Compute smoothed spineTheta (sspineTheta) - MAGAT lowpass filters spineTheta
            # MAGAT: sspineTheta = lowpass1D(spineTheta, smoothTime/interpTime)
            if len(spine_theta_magat) > 1:
                # Estimate smoothTime/interpTime from frame rate
                dt_avg = np.mean(dt[dt > 0]) if np.any(dt > 0) else 1.0 / frame_rate
                smooth_time = 0.1  # Typical MAGAT smoothTime (seconds)
                sigma_samples = smooth_time / dt_avg
                spine_theta_smoothed = lowpass1d(spine_theta_magat, sigma_samples)
            else:
                spine_theta_smoothed = spine_theta_magat
            
            # Use MAGAT-computed values
            # Override curvature with MAGAT's calculation
            curvature = spine_curv_magat
            
            # Store MAGAT spineTheta (body bend angle) for use in segmentation
            spine_theta_magat_computed = True
            print(f"    MAGAT spine analysis: computed spineTheta and spineCurv from {spine_points_per_frame.shape[1]} spine points")
        else:
            spine_theta_magat = None
            spine_theta_smoothed = None
            spine_theta_magat_computed = False
    except Exception as e:
        # Fallback to original method if MAGAT algorithms fail
        import traceback
        print(f"    Warning: MAGAT spine analysis failed ({e}), using simplified calculation")
        traceback.print_exc()
        spine_theta_magat = None
        spine_theta_smoothed = None
        spine_theta_magat_computed = False
        
        # Original simple curvature computation
        if spine_points_per_frame is not None:
            spine_curvatures = []
            for i in range(n_frames):
                spine = spine_points_per_frame[i]
                if len(spine) >= 3:
                    vecs = np.diff(spine, axis=0)
                    vec_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vec_norms[vec_norms < 1e-6] = 1e-6
                    vecs_normalized = vecs / vec_norms
                    angles = []
                    for j in range(len(vecs_normalized) - 1):
                        dot = np.clip(np.dot(vecs_normalized[j], vecs_normalized[j+1]), -1, 1)
                        angle = np.arccos(dot)
                        angles.append(angle)
                    if len(angles) > 0:
                        total_angle = np.sum(angles)
                        total_length = np.sum(vec_norms.flatten())
                        frame_curvature = total_angle / (total_length + 1e-6)
                    else:
                        frame_curvature = 0.0
                else:
                    frame_curvature = 0.0
                spine_curvatures.append(frame_curvature)
            spine_curvature_from_points = np.array(spine_curvatures)
            if len(spine_curvature_from_points) == n_frames:
                curvature = spine_curvature_from_points
    
    # Compute spine curve energy using MAGAT method
    # MAGAT uses spineTheta^2 for curve energy, or curvature^2
    if spine_theta_magat_computed and spine_theta_magat is not None:
        # Use MAGAT's spineTheta-based energy
        spine_curve_energy = calculate_spine_curve_energy_magat(spine_theta_magat, spine_curv=None)
    else:
        # Fallback: use curvature^2
        curvature_normalized = np.clip(curvature, -1e6, 1e6)  # Clip extreme outliers
        spine_curve_energy = curvature_normalized ** 2  # Energy ∝ curvature²
    
    # Also compute absolute curvature for additional features
    curvature_abs = np.abs(curvature)
    
    # FULL MAGAT SEGMENTATION ALGORITHM (after spine analysis)
    # MAGAT Reference: https://github.com/GilRaitses/magniphyq
    # 
    # MAGAT's Algorithm (from @MaggotTrack/segmentTrack.m):
    #   1. Find runs (periods of forward movement: high speed, head aligned, low curvature)
    #   2. Find head swings (head swinging wide periods between runs)
    #   3. Group head swings into reorientations - a reorientation is the period BETWEEN runs
    #      (whether or not it contains head swings). Reorientations are gaps between runs.
    #
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from magat_segmentation import magat_segment_track, MaggotSegmentOptions
        
        # Prepare DataFrame for MAGAT segmentation
        # Need: time, speed, curvature, body_theta (spineTheta), vel_dp
        # Use MAGAT-computed spineTheta if available (from spine analysis above)
        # Otherwise use H5 values or approximate
        if spine_theta_magat_computed and spine_theta_magat is not None:
            body_theta_for_seg = spine_theta_magat
            body_theta_smooth = spine_theta_smoothed if spine_theta_smoothed is not None else spine_theta_magat
        elif spine_theta is not None:
            body_theta_for_seg = spine_theta
            body_theta_smooth = spine_theta  # Approximate
        else:
            # Fallback approximation
            body_theta_for_seg = np.abs(curvature) * 10
            body_theta_smooth = body_theta_for_seg
        
        magat_df = pd.DataFrame({
            'time': time,
            'speed': speed,
            'curvature': curvature,
            'curv': curvature,  # MAGAT uses 'curv'
            'spineTheta': body_theta_for_seg,  # MAGAT body bend angle
            'sspineTheta': body_theta_smooth,  # MAGAT smoothed body bend angle
            'heading': heading,
            'x': x,
            'y': y
        })
        
        # Attach derivation_rules for MAGAT segmentation head-swing buffer calculation
        if derivation_rules is not None:
            magat_df.attrs['derivation_rules'] = derivation_rules
        
        # Add vel_dp if available
        if vel_dp is not None:
            magat_df['vel_dp'] = vel_dp
        else:
            # Approximate: assume good alignment if speed is high
            magat_df['vel_dp'] = np.ones(n_frames) * 0.707  # cos(45°)
        
        # Learn optimal segmentation parameters from data (parameter learning)
        try:
            from learn_magat_parameters import learn_optimal_parameters
            print("    Learning optimal MAGAT parameters from data...")
            segment_options = learn_optimal_parameters(
                trajectory_df=pd.DataFrame({
                    'time': time,
                    'speed': speed,
                    'curvature': curvature,
                    'heading': heading,
                    'spineTheta': body_theta_for_seg
                }),
                target_runs_per_minute=1.0,
                min_run_duration=2.5
            )
            print(f"    Learned parameters: curv_cut={segment_options.curv_cut:.4f}, "
                  f"theta_cut={np.rad2deg(segment_options.theta_cut):.1f}°, "
                  f"stop_speed={segment_options.stop_speed_cut:.6f}, "
                  f"start_speed={segment_options.start_speed_cut:.6f}")
        except Exception as e:
            # Fallback to defaults if learning fails
            print(f"    Warning: Parameter learning failed ({e}), using defaults")
            segment_options = MaggotSegmentOptions()
            segment_options.minRunTime = 2.5
            segment_options.minHeadSwingDuration = 0.05
            segment_options.minHeadSwingAmplitude = np.deg2rad(10)
        
        # Run quality filters
        segment_options.minRunLength = 0.0  # Minimum path length in cm (0 = disabled)
        segment_options.minRunSpeed = 0.0  # Minimum average speed during run (0 = disabled)
        segment_options.requireRunContinuous = True  # Require runs to be continuous (no gaps)
        
        # Head swing quality filters (if not set by learning)
        if not hasattr(segment_options, 'minHeadSwingDuration'):
            segment_options.minHeadSwingDuration = 0.05  # 50ms minimum
        if not hasattr(segment_options, 'minHeadSwingAmplitude'):
            segment_options.minHeadSwingAmplitude = np.deg2rad(10)  # 10° minimum
        segment_options.requireAccepted = False  # Set to True for MAGAT strict mode
        segment_options.requireValid = False  # Set to True if htValid data is available
        
        # Run MAGAT segmentation
        frame_rate_actual = 1.0 / np.mean(dt[dt > 0]) if np.any(dt > 0) else frame_rate
        segmentation = magat_segment_track(magat_df, segment_options=segment_options, frame_rate=frame_rate_actual)
        
        # Extract MAGAT results
        is_reorientation = segmentation['is_reorientation']  # Start events only
        is_run = segmentation['is_run']
        n_reorientations = segmentation['n_reorientations']
        
        # Extract MATLAB-compatible reorientation angle calculations
        reo_prevdir = segmentation.get('reo_prevdir', np.array([]))
        reo_nextdir = segmentation.get('reo_nextdir', np.array([]))
        reo_dtheta = segmentation.get('reo_dtheta', np.array([]))
        
        print(f"    MAGAT segmentation: {segmentation['n_runs']} runs, {segmentation['n_head_swings']} head swings, {n_reorientations} reorientations")
        if len(reo_dtheta) > 0:
            print(f"    MATLAB-compatible reo_dtheta: {len(reo_dtheta)} reorientations, range [{np.rad2deg(np.min(reo_dtheta)):.1f}°, {np.rad2deg(np.max(reo_dtheta)):.1f}°]")
        
        # Store segmentation for Klein run table generation (turns will be added after pause detection)
        magat_segmentation = segmentation
        
    except Exception as e:
        # Fallback to simplified detection if MAGAT segmentation fails
        import traceback
        print(f"    Warning: MAGAT segmentation failed ({e}), using simplified detection")
        traceback.print_exc()
        frame_rate_actual = 1.0 / np.mean(dt[dt > 0]) if np.any(dt > 0) else 10.0
        angular_velocity = heading_change / np.maximum(dt, 1e-6)  # rad/s
        
        # Simplified thresholds
        turn_threshold_rad_per_sec = 2.3
        speed_threshold = 0.0003
        
        is_reorientation_frame = (angular_velocity > turn_threshold_rad_per_sec) & (speed > speed_threshold)
        is_reorientation = np.zeros(n_frames, dtype=bool)
        if n_frames > 1:
            is_reorientation[1:] = is_reorientation_frame[1:] & (~is_reorientation_frame[:-1])
            if is_reorientation_frame[0]:
                is_reorientation[0] = True
        is_run = np.zeros(n_frames, dtype=bool)  # No run info in fallback
        
        # Fallback: turns = reorientations that contain pauses (if pause detection available)
        # Note: df doesn't exist yet in fallback, so we'll set is_turn after df is created
        is_turn = np.zeros(n_frames, dtype=bool)
    
    # Also include head and tail positions
    head_x = track_data.get('head', mid_pos)[:, 0] if 'head' in track_data else x
    head_y = track_data.get('head', mid_pos)[:, 1] if 'head' in track_data else y
    tail_x = track_data.get('tail', mid_pos)[:, 0] if 'tail' in track_data else x
    tail_y = track_data.get('tail', mid_pos)[:, 1] if 'tail' in track_data else y
    
    # Store spine points data (if available)
    spine_data = {}
    if spine_points_per_frame is not None:
        # Store spine points as columns (spine_x_0, spine_y_0, spine_x_1, spine_y_1, ...)
        n_spine_pts = spine_points_per_frame.shape[1]
        for i in range(n_spine_pts):
            spine_data[f'spine_x_{i}'] = spine_points_per_frame[:, i, 0]
            spine_data[f'spine_y_{i}'] = spine_points_per_frame[:, i, 1]
    
    # Create frame array (0-indexed track frame numbers)
    frames = np.arange(n_frames)
    
    # Create arrays for reorientation angle data (MATLAB-compatible)
    # These arrays have one value per reorientation, not per frame
    # We'll store them as attributes or create per-frame arrays
    reo_dtheta_per_frame = np.full(n_frames, np.nan)  # NaN for non-reorientation frames
    reo_prevdir_per_frame = np.full(n_frames, np.nan)
    reo_nextdir_per_frame = np.full(n_frames, np.nan)
    
    # Map reorientation angles to frames (at reorientation start)
    if 'magat_segmentation' in locals() and magat_segmentation is not None:
        reorientations = magat_segmentation.get('reorientations', [])
        reo_dtheta_array = magat_segmentation.get('reo_dtheta', np.array([]))
        reo_prevdir_array = magat_segmentation.get('reo_prevdir', np.array([]))
        reo_nextdir_array = magat_segmentation.get('reo_nextdir', np.array([]))
        
        for i, (reo_start, reo_end) in enumerate(reorientations):
            if i < len(reo_dtheta_array):
                # Store at reorientation start frame (MATLAB convention)
                if reo_start < n_frames:
                    reo_dtheta_per_frame[reo_start] = reo_dtheta_array[i]
                    reo_prevdir_per_frame[reo_start] = reo_prevdir_array[i]
                    reo_nextdir_per_frame[reo_start] = reo_nextdir_array[i]
    
    df = pd.DataFrame({
        'frame': frames,
        'time': time,
        'x': x,  # mid/centroid x
        'y': y,  # mid/centroid y
        'head_x': head_x,
        'head_y': head_y,
        'tail_x': tail_x,
        'tail_y': tail_y,
        'speed': speed,
        'heading': heading,
        'curvature': curvature,
        'curvature_abs': curvature_abs,
        'spine_curve_energy': spine_curve_energy,
        'acceleration': accel,
        'heading_change': np.concatenate([[0], heading_change[1:]]),
        'is_turn_simple': is_turn_simple,  # Simple heading change detection (backwards compatibility)
        'is_reorientation': is_reorientation,  # MAGAT reorientation detection (start events)
        'is_run': is_run if 'is_run' in locals() else np.zeros(n_frames, dtype=bool),  # MAGAT run detection
        'reo_dtheta': reo_dtheta_per_frame,  # MATLAB-compatible: diff(unwrap([prevDir;nextDir]))
        'reo_prevdir': reo_prevdir_per_frame,  # Direction before reorientation
        'reo_nextdir': reo_nextdir_per_frame,  # Direction after reorientation
        **spine_data  # Add spine point columns
    })
    
    # ==========================================================================
    # TRACK-LEVEL LED EXTRACTION (uses track's own LED data - no merge needed)
    # This is cleaner than merging with global stimulus data because:
    # 1. Track LED is on same uniform time grid as track data (no alignment issues)
    # 2. No NaN values from merge_asof tolerance failures
    # 3. LED values were interpolated to same grid as position data by MATLAB
    # ==========================================================================
    derived = track_data.get('derived', {})
    
    # Extract LED1 values
    led1_val = derived.get('led1Val')
    if led1_val is not None:
        if led1_val.ndim > 1:
            led1_val = led1_val.flatten()
        if len(led1_val) >= n_frames:
            led1_val = led1_val[:n_frames]
        else:
            led1_val = np.pad(led1_val, (0, n_frames - len(led1_val)), mode='edge')
        
        # Compute LED1 ON/OFF state
        led1_max = np.max(led1_val)
        if led1_max > 0:
            led1_threshold = led1_max * 0.1
            led1_ton = led1_val > led1_threshold
        else:
            led1_ton = np.zeros(n_frames, dtype=bool)
        led1_toff = ~led1_ton
        
        # Detect LED1 onsets/offsets using diff
        led1_diff = derived.get('led1ValDiff')
        if led1_diff is not None:
            if led1_diff.ndim > 1:
                led1_diff = led1_diff.flatten()
            if len(led1_diff) >= n_frames:
                led1_diff = led1_diff[:n_frames]
            else:
                led1_diff = np.pad(led1_diff, (0, n_frames - len(led1_diff)), mode='constant')
            
            # Onset: large positive diff, Offset: large negative diff
            # Use 30% of max diff magnitude as threshold (robust to noise)
            # LED transitions show large spikes (>1000), noise is small (<100)
            diff_max = max(np.abs(led1_diff).max(), 1.0)
            diff_threshold = diff_max * 0.3  # 30% of max diff
            led1_onset = led1_diff > diff_threshold
            led1_offset = led1_diff < -diff_threshold
        else:
            # Fallback: detect from state transitions
            led1_padded = np.concatenate([[False], led1_ton])
            led1_onset = (~led1_padded[:-1]) & led1_padded[1:]
            led1_offset = led1_padded[:-1] & (~led1_padded[1:])
        
        df['led1Val'] = led1_val
        df['led1Val_ton'] = led1_ton
        df['led1Val_toff'] = led1_toff
        df['led1Val_onset'] = led1_onset
        df['led1Val_offset'] = led1_offset
        
        # Compute time_since_stimulus (from LED1 onsets)
        time_since_stimulus = np.zeros(n_frames)
        last_onset_time = np.nan
        for i in range(n_frames):
            if led1_onset[i]:
                last_onset_time = time[i]
            if not np.isnan(last_onset_time):
                time_since_stimulus[i] = time[i] - last_onset_time
        
        df['stimulus_on'] = led1_ton
        df['stimulus_onset'] = led1_onset
        df['stimulus_offset'] = led1_offset
        df['time_since_stimulus'] = time_since_stimulus
    else:
        # No LED1 data - create placeholders
        df['led1Val'] = 0.0
        df['led1Val_ton'] = False
        df['led1Val_toff'] = True
        df['stimulus_on'] = False
        df['stimulus_onset'] = False
        df['stimulus_offset'] = False
        df['time_since_stimulus'] = 0.0
    
    # Extract LED2 values (if available)
    led2_val = derived.get('led2Val')
    if led2_val is not None:
        if led2_val.ndim > 1:
            led2_val = led2_val.flatten()
        if len(led2_val) >= n_frames:
            led2_val = led2_val[:n_frames]
        else:
            led2_val = np.pad(led2_val, (0, n_frames - len(led2_val)), mode='edge')
        
        led2_max = np.max(led2_val)
        if led2_max > 0:
            led2_threshold = led2_max * 0.1
            led2_ton = led2_val > led2_threshold
        else:
            led2_ton = np.zeros(n_frames, dtype=bool)
        led2_toff = ~led2_ton
        
        df['led2Val'] = led2_val
        df['led2Val_ton'] = led2_ton
        df['led2Val_toff'] = led2_toff
    else:
        df['led2Val'] = 0.0
        df['led2Val_ton'] = False
        df['led2Val_toff'] = True
    
    # After df is created, handle fallback turn detection if needed
    # CRITICAL: Use MAGAT definition (turns = reorientations with head swings)
    if 'is_turn' not in df.columns:
        # Fallback: if MAGAT segmentation failed, try to detect turns from head swings
        # This requires head swing information from segmentation
        if 'magat_segmentation' in locals() and magat_segmentation is not None:
            # Use the turn detection from above (already set in df)
            pass  # Already handled above
        else:
            # No segmentation available - cannot determine turns without head swing info
            # Set is_turn to False (turns require MAGAT segmentation)
            df['is_turn'] = np.zeros(n_frames, dtype=bool)
            print(f"    Warning: No MAGAT segmentation available, cannot detect turns (requires head swing info)")
    
    # Add MAGAT-computed spineTheta fields if available
    if spine_theta_magat_computed and spine_theta_magat is not None:
        df['spineTheta_magat'] = spine_theta_magat  # MAGAT body bend angle
        if spine_theta_smoothed is not None:
            df['sspineTheta_magat'] = spine_theta_smoothed  # MAGAT smoothed body bend angle
    
    # Add pause detection and turn duration quantification
    try:
        import sys
        from pathlib import Path
        # Add scripts directory to path if not already there
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from detect_events import add_event_detection
        df = add_event_detection(df)
    except (ImportError, Exception) as e:
        # Fallback: simple pause detection if module not available
        print(f"  Warning: Could not import detect_events, using simple detection: {e}")
        speed_threshold = 0.001
        pause_min_duration = 0.2  # seconds
        is_pause = (df['speed'] < speed_threshold).values
        pause_durations = np.zeros(len(df))
        turn_durations = np.zeros(len(df))
        turn_event_ids = np.zeros(len(df), dtype=int)
        is_reversal = (df['heading_change'] > np.pi/2).values
        
        df['is_pause'] = is_pause
        df['pause_duration'] = pause_durations
        df['turn_duration'] = turn_durations
        df['turn_event_id'] = turn_event_ids
        df['is_reversal'] = is_reversal
    
    # CRITICAL FIX: Extract turns from reorientations using MAGAT definition
    # MAGAT Definition: Turns = reorientations with numHS >= 1 (at least 1 head swing)
    # NOT: Turns = reorientations with pauses (this was WRONG)
    # Reference: @MaggotReorientation/MaggotReorientation.m, load_multi_data_gr21a.m line 169
    # See docs/logs/2025-11-14/MAGAT_TURN_DEFINITION_ANALYSIS.md
    if 'magat_segmentation' in locals() and magat_segmentation is not None:
        # Extract turns from reorientations (turns = reorientations with head swings)
        # This matches MAGAT's definition: turnStartTime = reorientation with numHS >= 1
        turns = []
        is_turn = np.zeros(n_frames, dtype=bool)
        
        reorientations = magat_segmentation['reorientations']
        head_swings = magat_segmentation.get('head_swings', [])
        
        # Associate head swings with reorientations (count numHS per reorientation)
        for reo_idx, (reo_start, reo_end) in enumerate(reorientations):
            # Count head swings within this reorientation
            numHS = 0
            for hs_start, hs_end in head_swings:
                # Head swing is within reorientation if it overlaps
                # Overlap: hs_start <= reo_end and hs_end >= reo_start
                if hs_start <= reo_end and hs_end >= reo_start:
                    numHS += 1
            
            # MAGAT definition: Turn = reorientation with numHS >= 1
            if numHS >= 1:
                # This reorientation has at least 1 head swing → it's a turn
                turns.append((reo_start, reo_end))
                is_turn[reo_start:reo_end+1] = True
        
        # Add turns to segmentation
        magat_segmentation['turns'] = turns
        magat_segmentation['is_turn'] = is_turn
        magat_segmentation['n_turns'] = len(turns)
        
        print(f"    Turns (reorientations with numHS >= 1): {len(turns)} out of {len(reorientations)} reorientations")
        
        # Set is_turn in DataFrame
        df['is_turn'] = is_turn  # MAGAT turn detection: reorientations with head swings
    elif 'is_turn' not in df.columns:
        # No segmentation or no pause detection - set is_turn to False
        df['is_turn'] = np.zeros(n_frames, dtype=bool)
    
    # Generate Klein run table if MAGAT segmentation succeeded (NO FALLBACKS)
    klein_run_table = None
    if 'magat_segmentation' in locals() and magat_segmentation is not None:
        # Check if we have runs - Klein run table requires at least 1 run
        if magat_segmentation.get('n_runs', 0) == 0:
            print(f"    Skipping Klein run table generation: No runs detected (need at least 1 run)")
        else:
            try:
                import sys
                from pathlib import Path
                scripts_dir = Path(__file__).parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from klein_run_table import generate_klein_run_table
                
                # Get track and experiment IDs
                track_id = track_data.get('attrs', {}).get('id', 1)
                experiment_id = track_data.get('attrs', {}).get('experiment_id', 1)
                
                # Generate Klein run table (NO FALLBACKS - will raise error if data missing)
                klein_run_table = generate_klein_run_table(
                    trajectory_df=df,
                    segmentation=magat_segmentation,
                    track_id=track_id,
                    experiment_id=experiment_id,
                    set_id=1
                )
                
                print(f"    Generated Klein run table: {len(klein_run_table)} runs/turns")
                
            except Exception as e:
                # NO FALLBACKS - re-raise error for data quality issues
                # But provide helpful error message
                track_id = track_data.get('attrs', {}).get('id', 1)
                raise ValueError(f"Failed to generate Klein run table for track {track_id}: {e}") from e
    
    # Store run table as attribute (for later access)
    if klein_run_table is not None:
        df.attrs['klein_run_table'] = klein_run_table
    
    # ==========================================================================
    # REVERSE CRAWL DETECTION (Mason Klein methodology via retrovibez)
    # SpeedRunVel = speed * dot(velocity_direction, heading_direction)
    # Reverse crawl = SpeedRunVel < 0 for >= 3 seconds
    # CRITICAL: Uses derived_quantities/shead, smid, sloc (smoothed real-world coords)
    #           NOT points/head, points/mid (raw pixel coords) - these differ by ~0.28 cm!
    # Reference: Klein et al. 2015 PNAS, retrovibez/matlab/mason_analysis.m
    # ==========================================================================
    if HAS_RETROVIBEZ:
        derived = track_data.get('derived', {})
        
        # Extract required arrays from derived_quantities (MUST use s-prefixed versions)
        has_required = all(k in derived for k in ['shead', 'smid', 'sloc', 'eti'])
        
        if has_required:
            try:
                # Get arrays and ensure correct shape (2, N)
                def ensure_2xN(arr):
                    arr = np.asarray(arr).squeeze()
                    if arr.ndim == 1:
                        return arr.reshape(1, -1)
                    if arr.shape[0] != 2 and arr.shape[1] == 2:
                        return arr.T
                    return arr
                
                shead = ensure_2xN(derived['shead'])
                smid = ensure_2xN(derived['smid'])
                sloc = ensure_2xN(derived['sloc'])
                track_eti = np.asarray(derived['eti']).ravel()
                
                # Ensure lengths match (use minimum length)
                min_len = min(shead.shape[1], smid.shape[1], sloc.shape[1], len(track_eti))
                shead = shead[:, :min_len]
                smid = smid[:, :min_len]
                sloc = sloc[:, :min_len]
                track_eti = track_eti[:min_len]
                
                # Compute SpeedRunVel
                speed_run_vel = compute_speed_run_vel(track_eti, sloc, shead, smid)
                
                # Detect reversals (>= 3 seconds duration)
                times_srv = track_eti[:-1]  # Match SpeedRunVel length
                reversals = detect_reversals(times_srv, speed_run_vel, min_duration=3.0)
                
                # Create boolean masks matching DataFrame length
                speed_run_vel_padded = np.full(n_frames, np.nan)
                is_reverse_crawl = np.zeros(n_frames, dtype=bool)
                is_reverse_crawl_start = np.zeros(n_frames, dtype=bool)
                
                # Fill SpeedRunVel (length n-1, pad last frame with NaN)
                srv_len = min(len(speed_run_vel), n_frames)
                speed_run_vel_padded[:srv_len] = speed_run_vel[:srv_len]
                
                # Mark reversal frames
                for rev in reversals:
                    start = max(0, rev.start_idx)
                    end = min(n_frames, rev.end_idx + 1)
                    is_reverse_crawl[start:end] = True
                    if start < n_frames:
                        is_reverse_crawl_start[start] = True
                
                df['speed_run_vel'] = speed_run_vel_padded
                df['is_reverse_crawl'] = is_reverse_crawl
                df['is_reverse_crawl_start'] = is_reverse_crawl_start
                
                if len(reversals) > 0:
                    total_rev_duration = sum(r.duration for r in reversals)
                    print(f"    Reverse crawls (retrovibez): {len(reversals)} events, {total_rev_duration:.1f}s total")
                    
            except Exception as e:
                print(f"    Warning: Reverse crawl detection failed: {e}")
                df['speed_run_vel'] = np.full(n_frames, np.nan)
                df['is_reverse_crawl'] = np.zeros(n_frames, dtype=bool)
                df['is_reverse_crawl_start'] = np.zeros(n_frames, dtype=bool)
        else:
            # Missing required derived quantities
            missing = [k for k in ['shead', 'smid', 'sloc', 'eti'] if k not in derived]
            print(f"    Warning: Cannot detect reverse crawls - missing derived_quantities: {missing}")
            df['speed_run_vel'] = np.full(n_frames, np.nan)
            df['is_reverse_crawl'] = np.zeros(n_frames, dtype=bool)
            df['is_reverse_crawl_start'] = np.zeros(n_frames, dtype=bool)
    else:
        # retrovibez not available
        df['speed_run_vel'] = np.full(n_frames, np.nan)
        df['is_reverse_crawl'] = np.zeros(n_frames, dtype=bool)
        df['is_reverse_crawl_start'] = np.zeros(n_frames, dtype=bool)
    
    return df

def compute_ton_toff(led_values: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute _ton and _toff boolean arrays from LED values (equivalent to MAGAT addTonToff).
    
    Parameters
    ----------
    led_values : ndarray
        LED intensity values (1D array)
    threshold : float, optional
        Threshold for detecting ON state. If None, uses 10% of max value.
    
    Returns
    -------
    ton : ndarray
        Boolean array where True indicates LED is ON
    toff : ndarray
        Boolean array where True indicates LED is OFF
    """
    if threshold is None:
        threshold = np.max(led_values) * 0.1  # 10% of max
    
    # Determine ON/OFF state
    is_on = led_values > threshold
    
    # ton: True when LED is ON
    ton = is_on
    
    # toff: True when LED is OFF
    toff = ~is_on
    
    return ton, toff

def extract_stimulus_timing(h5_data: Dict, frame_rate: float = 10.0) -> pd.DataFrame:
    """
    Extract stimulus timing from H5 data.
    
    Uses stimulus onset frames to create 10-second pulses (fixed duration).
    
    CRITICAL: Uses ETI for time calculation. frame_rate parameter is only used for pulse duration calculations.
    
    Parameters
    ----------
    h5_data : dict
        H5 data dictionary with 'led_data' and 'stimulus' groups. MUST contain 'eti' at root level.
    frame_rate : float
        Frame rate in Hz (default 10 fps) - ONLY used for pulse duration calculations, NOT for time array
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, stimulus_on, intensity, time_since_stimulus, etc.
    """
    # CRITICAL POLICY: ETI MUST ALWAYS BE USED FOR TIME CALCULATION
    # See docs/ETI_TIME_CALCULATION_POLICY.md
    if 'eti' not in h5_data or h5_data['eti'] is None:
        raise ValueError("CRITICAL ERROR: ETI not found in h5_data. "
                       "ETI must be loaded from H5 root level. "
                       "See docs/ETI_TIME_CALCULATION_POLICY.md")
    
    eti = h5_data['eti']
    n_frames = len(eti)
    
    # Use ETI directly for time array (NOT frame_rate-based calculation)
    times = eti.copy()
    
    # Get LED1 data (red pulsing)
    if 'led1Val' in h5_data:
        led1_values = h5_data['led1Val']
        if len(led1_values) != n_frames:
            raise ValueError(f"CRITICAL ERROR: LED1 length ({len(led1_values)}) doesn't match ETI length ({n_frames})")
    elif 'led_data' in h5_data:
        led1_values = h5_data['led_data']
        if len(led1_values) != n_frames:
            raise ValueError(f"CRITICAL ERROR: LED data length ({len(led1_values)}) doesn't match ETI length ({n_frames})")
    else:
        return pd.DataFrame()
    
    # Compute led1Val_ton and led1Val_toff (equivalent to addTonToff)
    led1Val_ton, led1Val_toff = compute_ton_toff(led1_values)
    
    # Get LED2 data (blue constant) if available
    led2_values = None
    led2Val_ton = None
    led2Val_toff = None
    if 'led2Val' in h5_data:
        led2_values = h5_data['led2Val']
        if len(led2_values) == n_frames:
            led2Val_ton, led2Val_toff = compute_ton_toff(led2_values)
    
    # FIXED: Pulse duration is always 10 seconds
    pulse_duration = 10.0
    pulse_duration_frames = int(pulse_duration * frame_rate)  # 100 frames at 10 fps
    
    # ==========================================================================
    # DETECT LED TRANSITIONS USING DIFF FIELDS FOR PRECISE ONSET/OFFSET
    # ==========================================================================
    
    def detect_led_transitions_from_diff(led_values, led_diff, led_name, frame_rate):
        """
        Detect ON/OFF transitions using the Diff field for precise timing.
        
        The Diff field contains frame-to-frame differences:
        - Large positive spike = onset (LED turning ON)
        - Large negative spike = offset (LED turning OFF)
        
        Returns:
            led_on_state: bool array - True when LED is ON
            onset_indices: array - frame indices where LED turns ON
            offset_indices: array - frame indices where LED turns OFF
            is_pulsing: bool - True if LED is pulsing (square wave), False if constant
            pulse_duration_sec: float - mean pulse duration in seconds (if pulsing)
        """
        n = len(led_values)
        led_max = np.max(led_values)
        
        # Check if LED has any signal
        if led_max <= 0:
            return np.zeros(n, dtype=bool), np.array([]), np.array([]), False, 0.0
        
        # Use Diff field if available for precise transition detection
        if led_diff is not None and len(led_diff) == n:
            # Threshold for transition detection (10% of LED range)
            diff_threshold = led_max * 0.1
            
            # Onset: large positive diff (LED turning ON)
            onset_indices = np.where(led_diff > diff_threshold)[0]
            
            # Offset: large negative diff (LED turning OFF)
            offset_indices = np.where(led_diff < -diff_threshold)[0]
        else:
            # Fallback: detect from value transitions
            threshold = led_max * 0.1
            led_on_state_temp = led_values > threshold
            led_padded = np.concatenate([[False], led_on_state_temp])
            onset_indices = np.where((~led_padded[:-1]) & led_padded[1:])[0]
            offset_indices = np.where(led_padded[:-1] & (~led_padded[1:]))[0]
        
        # Determine ON state from value threshold
        threshold = led_max * 0.1
        led_on_state = led_values > threshold
        
        # Determine if pulsing (square wave) or constant
        n_transitions = len(onset_indices) + len(offset_indices)
        is_pulsing = n_transitions >= 4  # At least 2 complete cycles
        
        # Compute pulse duration if pulsing
        pulse_duration_sec = 0.0
        if is_pulsing and len(onset_indices) > 0 and len(offset_indices) > 0:
            pulse_durations = []
            for onset_idx in onset_indices:
                offsets_after = offset_indices[offset_indices > onset_idx]
                if len(offsets_after) > 0:
                    offset_idx = offsets_after[0]
                    pulse_durations.append(offset_idx - onset_idx)
            if pulse_durations:
                pulse_duration_sec = np.mean(pulse_durations) / frame_rate
        
        return led_on_state, onset_indices, offset_indices, is_pulsing, pulse_duration_sec
    
    # Get LED Diff fields from h5_data
    led1_diff = h5_data.get('led1ValDiff')
    led2_diff = h5_data.get('led2ValDiff')
    
    # Detect LED1 (Red) transitions
    led1_on_state, led1_onset_indices, led1_offset_indices, led1_is_pulsing, led1_pulse_sec = \
        detect_led_transitions_from_diff(led1_values, led1_diff, "LED1", frame_rate)
    
    # Report LED1 status
    if led1_is_pulsing:
        print(f"  LED1 (Red): PULSING - {len(led1_onset_indices)} onsets, {len(led1_offset_indices)} offsets, {led1_pulse_sec:.1f}s pulse")
    else:
        led1_duty = np.mean(led1_on_state) * 100
        print(f"  LED1 (Red): CONSTANT - {led1_duty:.0f}% duty cycle")
    
    # Detect LED2 (Blue) transitions if available
    led2_on_state = np.zeros(n_frames, dtype=bool)
    led2_onset_indices = np.array([], dtype=int)
    led2_offset_indices = np.array([], dtype=int)
    led2_is_pulsing = False
    led2_pulse_sec = 0.0
    
    if led2_values is not None:
        led2_on_state, led2_onset_indices, led2_offset_indices, led2_is_pulsing, led2_pulse_sec = \
            detect_led_transitions_from_diff(led2_values, led2_diff, "LED2", frame_rate)
        
        if led2_is_pulsing:
            print(f"  LED2 (Blue): PULSING - {len(led2_onset_indices)} onsets, {len(led2_offset_indices)} offsets, {led2_pulse_sec:.1f}s pulse")
        else:
            led2_duty = np.mean(led2_on_state) * 100
            if led2_duty > 0:
                print(f"  LED2 (Blue): CONSTANT ON - {led2_duty:.0f}% duty cycle")
            else:
                print(f"  LED2 (Blue): OFF")
    
    # Create stimulus arrays
    stimulus_on = led1_on_state.copy()
    stimulus_onset = np.zeros(n_frames, dtype=bool)
    stimulus_onset[led1_onset_indices] = True
    stimulus_offset = np.zeros(n_frames, dtype=bool)
    stimulus_offset[led1_offset_indices] = True
    
    # LED2 onset/offset arrays (for pulsing LED2)
    led2_stimulus_onset = np.zeros(n_frames, dtype=bool)
    led2_stimulus_offset = np.zeros(n_frames, dtype=bool)
    if led2_is_pulsing:
        led2_stimulus_onset[led2_onset_indices] = True
        led2_stimulus_offset[led2_offset_indices] = True
    
    # Compute time since last stimulus onset
    time_since_stimulus = np.full(n_frames, np.nan)
    last_onset_time = np.nan
    
    for i, t in enumerate(times):
        if stimulus_onset[i]:
            last_onset_time = t
        
        if not np.isnan(last_onset_time):
            time_since_stimulus[i] = t - last_onset_time
    
    # Build DataFrame with all LED timing fields
    df_dict = {
        'time': times,
        'frame': np.arange(len(times)),
        # LED1 value and state
        'led1Val': led1_values,
        'led1Val_ton': led1_on_state,
        'led1Val_toff': ~led1_on_state,
        'led1Val_onset': stimulus_onset,
        'led1Val_offset': stimulus_offset,
        # LED1 derivatives (for fine-grained dynamics)
        'led1ValDeriv': h5_data.get('led1ValDeriv', np.zeros(n_frames)),
        'led1ValDiff': h5_data.get('led1ValDiff', np.zeros(n_frames)),
        # LED1 pulse info
        'led1_is_pulsing': led1_is_pulsing,
        # Combined stimulus
        'stimulus_on': stimulus_on,
        'stimulus_onset': stimulus_onset,
        'stimulus_offset': stimulus_offset,
        'time_since_stimulus': time_since_stimulus
    }
    
    # Add LED2 fields if available
    if led2_values is not None:
        df_dict['led2Val'] = led2_values
        df_dict['led2Val_ton'] = led2_on_state
        df_dict['led2Val_toff'] = ~led2_on_state
        df_dict['led2Val_onset'] = led2_stimulus_onset
        df_dict['led2Val_offset'] = led2_stimulus_offset
        # LED2 derivatives
        if h5_data.get('led2ValDeriv') is not None:
            df_dict['led2ValDeriv'] = h5_data['led2ValDeriv']
        if h5_data.get('led2ValDiff') is not None:
            df_dict['led2ValDiff'] = h5_data['led2ValDiff']
        # LED2 pulse info
        df_dict['led2_is_pulsing'] = led2_is_pulsing
    
    df = pd.DataFrame(df_dict)
    
    return df

def align_trajectory_with_stimulus(trajectory_df: pd.DataFrame, 
                                   stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align trajectory data with stimulus timing.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory features
    stimulus_df : pd.DataFrame
        Stimulus timing
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with aligned data
    """
    # Merge on time (with tolerance for frame alignment)
    # NOTE: Tolerance increased to 0.3s to handle camera gaps (up to 250ms gaps observed)
    # Camera sometimes drops frames, creating gaps in global ETI that don't exist in
    # track ETI (which is uniformly interpolated). See ETI-OVERFLOW-RESEARCH-PROMPT.md
    # FUTURE: Use track-level LED data directly instead of merge (cleaner architecture)
    merged = pd.merge_asof(
        trajectory_df.sort_values('time'),
        stimulus_df.sort_values('time'),
        on='time',
        direction='nearest',
        tolerance=0.3  # 300ms tolerance to handle camera gaps
    )
    
    # Compute time since last stimulus onset
    stimulus_onsets = merged[merged['stimulus_onset'] == True]['time'].values
    if len(stimulus_onsets) > 0:
        time_since_stimulus = np.zeros(len(merged))
        for i, t in enumerate(merged['time']):
            prev_onsets = stimulus_onsets[stimulus_onsets <= t]
            if len(prev_onsets) > 0:
                time_since_stimulus[i] = t - prev_onsets[-1]
        merged['time_since_stimulus'] = time_since_stimulus
    else:
        merged['time_since_stimulus'] = np.inf
    
    return merged

def create_event_records(trajectory_df: pd.DataFrame, 
                        track_id: int,
                        experiment_id: str) -> pd.DataFrame:
    """
    Create event records for LNP modeling.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Aligned trajectory+stimulus data
    track_id : int
        Track identifier
    experiment_id : str
        Experiment identifier
    
    Returns
    -------
    pd.DataFrame
        Event records with time bins and event indicators
    """
    # Create time bins (e.g., 50ms bins)
    bin_width = 0.05  # 50ms
    time_bins = np.arange(
        trajectory_df['time'].min(),
        trajectory_df['time'].max() + bin_width,
        bin_width
    )
    
    # Assign each time point to a bin
    trajectory_df['time_bin'] = np.digitize(trajectory_df['time'], time_bins) - 1
    
    # CRITICAL FIX: Detect event START times before binning
    # Count only start events (False->True transitions), not all frames during events
    trajectory_df = trajectory_df.sort_values('time').reset_index(drop=True)
    
    # Detect reorientation START events (False->True transitions)
    if 'is_reorientation' in trajectory_df.columns:
        is_reo = trajectory_df['is_reorientation'].values
        if len(is_reo) > 1:
            # Pad with False at start to detect first True as start
            is_reo_padded = np.concatenate([[False], is_reo])
            reo_start_mask = (~is_reo_padded[:-1]) & is_reo_padded[1:]
            trajectory_df['is_reorientation_start'] = False
            trajectory_df.loc[reo_start_mask, 'is_reorientation_start'] = True
        else:
            trajectory_df['is_reorientation_start'] = False
    
    # Detect turn START events (False->True transitions) for backwards compatibility
    if 'is_turn' in trajectory_df.columns:
        is_turn = trajectory_df['is_turn'].values
        if len(is_turn) > 1:
            is_turn_padded = np.concatenate([[False], is_turn])
            turn_start_mask = (~is_turn_padded[:-1]) & is_turn_padded[1:]
            trajectory_df['is_turn_start'] = False
            trajectory_df.loc[turn_start_mask, 'is_turn_start'] = True
        else:
            trajectory_df['is_turn_start'] = False
    
    # Detect pause START events
    if 'is_pause' in trajectory_df.columns:
        is_pause = trajectory_df['is_pause'].values
        if len(is_pause) > 1:
            is_pause_padded = np.concatenate([[False], is_pause])
            pause_start_mask = (~is_pause_padded[:-1]) & is_pause_padded[1:]
            trajectory_df['is_pause_start'] = False
            trajectory_df.loc[pause_start_mask, 'is_pause_start'] = True
        else:
            trajectory_df['is_pause_start'] = False
    
    # Aggregate to bins (only aggregate columns that exist)
    # NOTE: Spine point coordinates (spine_x_*, spine_y_*) are NOT aggregated here
    # They remain at full resolution in the trajectory DataFrame. Only derived features
    # like spine_curve_energy are aggregated for event records.
    # CRITICAL: Use *_start columns for event counting (only count start events, not duration)
    agg_dict = {
        'time': 'first',  # Use first time in bin (more accurate for event timing)
        'speed': 'mean',
        'heading': 'mean',
        'x': 'mean',
        'y': 'mean',
        'stimulus_on': 'any',
        'time_since_stimulus': 'mean',
        'is_turn': 'any',  # Keep for backwards compatibility (but use is_turn_start for counting)
        'is_turn_start': 'any',  # Count only turn START events
        'is_reorientation': 'any',  # Keep for backwards compatibility (but use is_reorientation_start for counting)
        'is_reorientation_start': 'any',  # Count only reorientation START events (USE THIS FOR TURN RATES)
        'is_pause': 'any',  # Keep for backwards compatibility
        'is_pause_start': 'any',  # Count only pause START events
        'is_reversal': 'any',  # Event occurred if any frame in bin was reversal (heading-based, deprecated)
        'is_reverse_crawl': 'any',  # Mason Klein: SpeedRunVel < 0 for >= 3s
        'is_reverse_crawl_start': 'any',  # Count only reverse crawl START events
        'curvature': 'mean',
        'spine_curve_energy': 'mean',  # Average bending energy per bin
        'turn_duration': 'mean',  # Average turn duration in bin
        'pause_duration': 'mean'  # Average pause duration in bin
    }
    
    # Add LED columns if they exist
    if 'led1Val' in trajectory_df.columns:
        agg_dict['led1Val'] = 'mean'
    if 'led1Val_ton' in trajectory_df.columns:
        agg_dict['led1Val_ton'] = 'any'
    if 'led1Val_toff' in trajectory_df.columns:
        agg_dict['led1Val_toff'] = 'any'
    if 'led2Val' in trajectory_df.columns:
        agg_dict['led2Val'] = 'mean'
    if 'led2Val_ton' in trajectory_df.columns:
        agg_dict['led2Val_ton'] = 'any'
    if 'led2Val_toff' in trajectory_df.columns:
        agg_dict['led2Val_toff'] = 'any'
    
    # Add speed_run_vel if available (from retrovibez reverse crawl detection)
    if 'speed_run_vel' in trajectory_df.columns:
        agg_dict['speed_run_vel'] = 'mean'
    
    binned = trajectory_df.groupby('time_bin').agg(agg_dict).reset_index()
    
    # Add metadata
    binned['track_id'] = track_id
    binned['experiment_id'] = experiment_id
    
    return binned

def process_h5_file(h5_path: Path, output_dir: Path, experiment_id: str):
    """Process a single H5 file and extract data for modeling."""
    print(f"\nProcessing: {h5_path.name}")
    
    # Load H5 file
    try:
        h5_data = load_h5_file(h5_path)
    except Exception as e:
        print(f"  ERROR loading H5 file: {e}")
        return
    
    # Get frame rate from metadata
    frame_rate = 10.0  # default
    if 'metadata' in h5_data and 'attrs' in h5_data['metadata']:
        metadata_attrs = h5_data['metadata']['attrs']
        if 'fps' in metadata_attrs:
            frame_rate = float(metadata_attrs['fps'])
    
    # Extract stimulus timing
    stimulus_df = extract_stimulus_timing(h5_data, frame_rate=frame_rate)
    if len(stimulus_df) == 0:
        print("  WARNING: No stimulus data found")
    else:
        print(f"  Stimulus data: {len(stimulus_df)} frames, {stimulus_df['stimulus_onset'].sum()} onsets")
    
    # Process each track
    all_event_records = []
    all_trajectories = []
    all_klein_run_tables = []  # Collect Klein run tables for all tracks
    
    if 'tracks' in h5_data:
        # Sort track keys by numeric value (track_1, track_2, ..., track_9, ..., track_64)
        # This ensures consistent ordering regardless of how they were stored in H5
        def extract_track_number(track_key):
            """Extract numeric part from track key (e.g., 'track_9' -> 9)."""
            try:
                return int(track_key.split('_')[-1])
            except (ValueError, IndexError):
                return 999999  # Put non-standard keys at the end
        
        track_keys = sorted(h5_data['tracks'].keys(), key=extract_track_number)
        
        for track_key in track_keys:
            track_data = h5_data['tracks'][track_key]
            # Extract track ID from track_key (e.g., "track_1" -> 1)
            try:
                track_id = int(track_key.split('_')[-1])
            except:
                track_id = len(all_trajectories) + 1
            
            # Extract trajectory features - CRITICAL: Must pass ETI and derivation_rules
            if 'eti' not in h5_data or h5_data['eti'] is None:
                raise ValueError(f"CRITICAL ERROR: ETI not available in h5_data. "
                                "ETI must be loaded from H5 root level.")
            traj_df = extract_trajectory_features(
                track_data, 
                frame_rate=frame_rate, 
                eti=h5_data['eti'],
                derivation_rules=h5_data.get('derivation_rules')
            )
            if len(traj_df) == 0:
                continue
            
            # LED data is now extracted directly from track-level derived quantities
            # in extract_trajectory_features (no merge needed - same time grid)
            aligned_df = traj_df  # LED columns already included
            
            # Create event records (aggregated for LNP modeling)
            event_records = create_event_records(aligned_df, track_id, experiment_id)
            all_event_records.append(event_records)
            
            # Keep full-resolution trajectories (including all spine points)
            # These are NOT aggregated - preserve full frame-level resolution
            all_trajectories.append(aligned_df)
            
            # Collect Klein run table if available
            if hasattr(traj_df, 'attrs') and 'klein_run_table' in traj_df.attrs:
                klein_rt = traj_df.attrs['klein_run_table'].copy()
                # Ensure track_id and experiment_id are set
                klein_rt['track_id'] = track_id
                klein_rt['experiment_id'] = experiment_id
                all_klein_run_tables.append(klein_rt)
            
            # CRITICAL FIX: Count only START events, not all bins during events
            n_turns = event_records['is_turn_start'].sum() if 'is_turn_start' in event_records.columns else (event_records['is_turn'].sum() if 'is_turn' in event_records.columns else 0)
            n_reorientations = event_records['is_reorientation_start'].sum() if 'is_reorientation_start' in event_records.columns else (event_records['is_reorientation'].sum() if 'is_reorientation' in event_records.columns else 0)
            n_runs = len(traj_df.attrs.get('klein_run_table', [])) if hasattr(traj_df, 'attrs') and 'klein_run_table' in traj_df.attrs else 0
            print(f"  Track {track_id}: {len(traj_df)} frames, {n_turns} turns (simple), {n_reorientations} reorientations (proper), {n_runs} runs (Klein)")
    
    # Combine all tracks
    if all_event_records:
        combined_events = pd.concat(all_event_records, ignore_index=True)
        combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
        
        # =======================================================================
        # GLOBAL time_since_stimulus RECALCULATION (vectorized for speed)
        # Per-track calculation gives wrong values for tracks starting mid-pulse.
        # Recalculate using global pulse onset times from experiment timeline.
        # =======================================================================
        if 'stimulus_onset' in combined_trajectories.columns:
            onset_times_raw = combined_trajectories[combined_trajectories['stimulus_onset'] == True]['time'].values
            if len(onset_times_raw) > 0:
                # Round to nearest 0.1s to cluster near-identical onset times
                unique_onsets = np.unique(np.round(onset_times_raw, 1))
                
                # Vectorized: use searchsorted for O(n log n) instead of O(n²)
                time_values = combined_trajectories['time'].values
                onset_indices = np.searchsorted(unique_onsets, time_values, side='right') - 1
                onset_indices = np.clip(onset_indices, 0, len(unique_onsets) - 1)
                tss_global = time_values - unique_onsets[onset_indices]
                tss_global[onset_indices < 0] = time_values[onset_indices < 0]  # Before first onset
                
                combined_trajectories['time_since_stimulus'] = tss_global
                
                # Same for events
                event_times = combined_events['time'].values
                event_onset_idx = np.searchsorted(unique_onsets, event_times, side='right') - 1
                event_onset_idx = np.clip(event_onset_idx, 0, len(unique_onsets) - 1)
                combined_events['time_since_stimulus'] = event_times - unique_onsets[event_onset_idx]
                
                print(f"  Recalculated time_since_stimulus globally using {len(unique_onsets)} unique onsets")
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        events_file = output_dir / f"{experiment_id}_events.parquet"
        combined_events.to_parquet(events_file, index=False)
        print(f"  Saved {len(combined_events)} event records to {events_file}")
        
        trajectories_file = output_dir / f"{experiment_id}_trajectories.parquet"
        combined_trajectories.to_parquet(trajectories_file, index=False)
        print(f"  Saved trajectory data to {trajectories_file}")
        
        # Save Klein run tables if available
        if all_klein_run_tables:
            combined_klein_runs = pd.concat(all_klein_run_tables, ignore_index=True)
            klein_runs_file = output_dir / f"{experiment_id}_klein_run_table.parquet"
            combined_klein_runs.to_parquet(klein_runs_file, index=False)
            print(f"  Saved {len(combined_klein_runs)} Klein run table rows to {klein_runs_file}")
            
            # Add to summary
            total_runs = int(len(combined_klein_runs))
            total_turns = int(combined_klein_runs['reoYN'].sum())
            total_head_swings = int(combined_klein_runs['reo#HS'].sum())
        else:
            total_runs = 0
            total_turns = 0
            total_head_swings = 0
        
        # Save summary
        summary = {
            'experiment_id': experiment_id,
            'n_tracks': len(all_event_records),
            'n_event_records': len(combined_events),
            'n_trajectory_points': len(combined_trajectories),
            'n_turns': int(combined_events['is_turn'].sum()),
            'n_reorientations': int(combined_events['is_reorientation'].sum()) if 'is_reorientation' in combined_events.columns else 0,
            'mean_turn_rate': float(combined_events['is_turn'].mean() / 0.05 * 60),  # turns/min (simple detection)
            'mean_reorientation_rate': float(combined_events['is_reorientation'].mean() / 0.05 * 60) if 'is_reorientation' in combined_events.columns else 0.0,  # reorientations/min (proper detection)
            'n_klein_runs': total_runs,
            'n_klein_turns': total_turns,
            'n_klein_head_swings': total_head_swings
        }
        
        summary_file = output_dir / f"{experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary to {summary_file}")
    else:
        print("  WARNING: No tracks processed")

def main():
    parser = argparse.ArgumentParser(description='Engineer dataset from H5 files')
    parser.add_argument('--h5-dir', type=str, 
                       default='/Users/gilraitses/mechanosensation/h5tests',
                       help='Directory containing H5 files')
    parser.add_argument('--output-dir', type=str,
                       default='data/engineered',
                       help='Output directory for engineered data')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Experiment ID (if None, uses filename)')
    parser.add_argument('--file', type=str, default=None,
                       help='Process single file (if None, processes all)')
    
    args = parser.parse_args()
    
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    
    if not HAS_H5PY:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        sys.exit(1)
    
    # Find H5 files
    if args.file:
        # Handle both absolute and relative paths
        file_path = Path(args.file)
        if file_path.is_absolute():
            h5_files = [file_path]
        else:
            h5_files = [h5_dir / args.file]
    else:
        h5_files = sorted(h5_dir.rglob("*.h5"))  # Recursive search for H5 files
    
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} H5 file(s)")
    
    # Process each file
    for h5_file in h5_files:
        experiment_id = args.experiment_id or h5_file.stem.replace(' ', '_')
        process_h5_file(h5_file, output_dir, experiment_id)
    
    print(f"\nProcessing complete. Outputs in {output_dir}")
    
    # Consolidate all experiments into single HDF5 files
    if len(h5_files) > 1:
        print("\n" + "="*60)
        print("CONSOLIDATING TO HDF5")
        print("="*60)
        
        consolidated_h5 = output_dir / "consolidated_dataset.h5"
        
        # Find all parquet files
        traj_files = sorted(output_dir.glob("*_trajectories.parquet"))
        event_files = sorted(output_dir.glob("*_events.parquet"))
        klein_files = sorted(output_dir.glob("*_klein_run_table.parquet"))
        
        with h5py.File(consolidated_h5, 'w') as h5f:
            # Trajectories
            if traj_files:
                print(f"  Consolidating {len(traj_files)} trajectory files...")
                all_traj = pd.concat([pd.read_parquet(f) for f in traj_files], ignore_index=True)
                traj_grp = h5f.create_group('trajectories')
                for col in all_traj.columns:
                    data = all_traj[col].values
                    if data.dtype == object:
                        data = data.astype(str)
                        traj_grp.create_dataset(col, data=data.astype('S'))
                    else:
                        traj_grp.create_dataset(col, data=data, compression='gzip')
                print(f"    {len(all_traj)} total trajectory rows")
            
            # Events
            if event_files:
                print(f"  Consolidating {len(event_files)} event files...")
                all_events = pd.concat([pd.read_parquet(f) for f in event_files], ignore_index=True)
                event_grp = h5f.create_group('events')
                for col in all_events.columns:
                    data = all_events[col].values
                    if data.dtype == object:
                        data = data.astype(str)
                        event_grp.create_dataset(col, data=data.astype('S'))
                    else:
                        event_grp.create_dataset(col, data=data, compression='gzip')
                print(f"    {len(all_events)} total event rows")
            
            # Klein run tables
            if klein_files:
                print(f"  Consolidating {len(klein_files)} klein run table files...")
                all_klein = pd.concat([pd.read_parquet(f) for f in klein_files], ignore_index=True)
                klein_grp = h5f.create_group('klein_run_tables')
                for col in all_klein.columns:
                    data = all_klein[col].values
                    if data.dtype == object:
                        data = data.astype(str)
                        klein_grp.create_dataset(col, data=data.astype('S'))
                    else:
                        klein_grp.create_dataset(col, data=data, compression='gzip')
                print(f"    {len(all_klein)} total klein run table rows")
        
        consolidated_size = consolidated_h5.stat().st_size / (1024*1024)
        print(f"\n  Consolidated HDF5: {consolidated_h5} ({consolidated_size:.1f} MB)")

if __name__ == '__main__':
    main()

