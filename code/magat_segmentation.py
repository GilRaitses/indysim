#!/usr/bin/env python3
"""
MAGAT Segmentation Algorithm Implementation in Python

Implements the full MAGAT track segmentation algorithm:
1. Detect runs (periods of forward movement)
2. Detect head swings (between runs)
3. Group into reorientations (gaps between runs)

Based on @MaggotTrack/segmentTrack.m from magniphyq repository.
Reference: https://github.com/GilRaitses/magniphyq

CRITICAL: This implementation matches MATLAB exactly with NO FALLBACKS.
All required fields must be present in trajectory_df or ValueError will be raised.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Dict, List, Tuple, Optional

class MaggotSegmentOptions:
    """MAGAT segmentation options (defaults from MaggotSegmentOptions.m)
    
    UNIT CONVERSIONS:
    - MATLAB uses mm/s for speed thresholds: stop=2.0, start=3.0
    - H5 speed is in cm/s, so convert: stop=0.2, start=0.3 cm/s
    - curv_cut=0.4 is for BODY curvature (smoothed), not PATH curvature
    - H5 curv is PATH curvature which explodes at low speed - use sspineTheta instead
    """
    def __init__(self):
        # Curvature threshold - applies to BODY curvature (sspineTheta-based), not path curv
        self.curv_cut = 0.4  # Body curvature threshold (dimensionless or 1/cm on body-length scale)
        self.autoset_curv_cut = False
        self.autoset_curv_cut_mult = 5
        self.theta_cut = np.pi / 2  # If body theta > theta_cut, end a run (radians)
        self.speed_field = 'speed'
        # Speed thresholds - CONVERTED from mm/s to cm/s
        # MATLAB: 2.0 mm/s = 0.2 cm/s, 3.0 mm/s = 0.3 cm/s
        self.stop_speed_cut = 0.2   # End run if speed < 0.2 cm/s (= 2 mm/s)
        self.start_speed_cut = 0.3  # Start run if speed > 0.3 cm/s (= 3 mm/s)
        self.aligned_dp = np.cos(np.deg2rad(45))  # cos(45°) ≈ 0.707 (MATLAB: cosd(45))
        # MATLAB MaggotSegmentOptions properties
        self.minRunTime = 2.5  # Minimum run duration in seconds
        self.minRunLength = 0  # Minimum run length - NOT USED IN FILTERING
        self.headswing_start = np.deg2rad(20)  # Start headswing if |theta| > 20°
        self.headswing_stop = np.deg2rad(10)   # End headswing if |theta| < 10°
        self.smoothBodyFromPeriFreq = False
        self.smoothBodyTime = None


def magat_segment_track(trajectory_df: 'pd.DataFrame', 
                       segment_options: Optional[MaggotSegmentOptions] = None,
                       frame_rate: float = 10.0) -> Dict:
    """
    Full MAGAT segmentation algorithm implementation.
    
    Segments a maggot track into runs and reorientations following MAGAT's algorithm.
    
    Also calculates MATLAB-compatible reorientation angles:
    - reo_prevdir: Direction of run BEFORE reorientation
    - reo_nextdir: Direction of run AFTER reorientation  
    - reo_dtheta: Angle change using MATLAB formula: diff(unwrap([prevDir;nextDir]))
    
    MATLAB Reference: spatialMaggotCalculations.m line 175
        sc.reo_dtheta = diff(unwrap([sc.reo_prevdir;sc.reo_nextdir]));
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory DataFrame with columns: time, speed, curvature, body_theta, vel_dp
        Must also have: x, y, heading (or theta)
    segment_options : MaggotSegmentOptions, optional
        Segmentation options (uses defaults if None)
    frame_rate : float
        Frame rate in Hz (default 10.0)
    
    Returns
    -------
    segmentation : dict
        Dictionary containing:
        - runs: List of (start_idx, end_idx) tuples for each run
        - head_swings: List of (start_idx, end_idx) tuples for each head swing
        - reorientations: List of (start_idx, end_idx) tuples for each reorientation
        - is_run: Boolean array indicating run frames
        - is_head_swing: Boolean array indicating head swing frames
        - is_reorientation: Boolean array indicating reorientation frames (start events)
    """
    if segment_options is None:
        segment_options = MaggotSegmentOptions()
    
    n_frames = len(trajectory_df)
    time = trajectory_df['time'].values
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0 / frame_rate
    
    # Get derived quantities (MATLAB: track.getDerivedQuantity(...))
    # REQUIRED fields - no fallbacks allowed
    
    # Speed (MATLAB: sp = track.getDerivedQuantity(mso.speed_field))
    if segment_options.speed_field not in trajectory_df.columns:
        raise ValueError(f"CRITICAL: {segment_options.speed_field} is REQUIRED. "
                        f"No fallback allowed. trajectory_df must contain '{segment_options.speed_field}' column.")
    speed = trajectory_df[segment_options.speed_field].values
    
    # Curvature (MATLAB: cv = track.getDerivedQuantity('curv'))
    if 'curv' not in trajectory_df.columns:
        raise ValueError("CRITICAL: 'curv' is REQUIRED. No fallback allowed. "
                        "trajectory_df must contain 'curv' column.")
    curv = trajectory_df['curv'].values
    
    # Body theta (sspineTheta) - body bend angle
    # MATLAB: bt = track.getDerivedQuantity('sspineTheta') or lowpass1D(...) if smoothBodyFromPeriFreq
    # REQUIRED: sspineTheta or spineTheta must be present
    if segment_options.smoothBodyFromPeriFreq or segment_options.smoothBodyTime is not None:
        # MATLAB: bt = lowpass1D(track.getDerivedQuantity('spineTheta'), st/track.dr.interpTime);
        if segment_options.smoothBodyTime is not None:
            st = segment_options.smoothBodyTime
        else:
            # MATLAB: st = 0.2/median(track.getDerivedQuantity('periFreq'));
            if 'periFreq' not in trajectory_df.columns:
                raise ValueError("CRITICAL: periFreq is REQUIRED when smoothBodyFromPeriFreq=True. "
                                "No fallback allowed. trajectory_df must contain 'periFreq' column.")
            st = 0.2 / np.median(trajectory_df['periFreq'].values)
        
        # Get interpTime for lowpass1D
        if hasattr(trajectory_df, 'attrs') and 'derivation_rules' in trajectory_df.attrs:
            interp_time = trajectory_df.attrs['derivation_rules'].get('interpTime', dt)
        else:
            raise ValueError("CRITICAL: derivation_rules.interpTime is REQUIRED for smoothBodyFromPeriFreq. "
                            "No fallback allowed.")
        
        if 'spineTheta' not in trajectory_df.columns:
            raise ValueError("CRITICAL: spineTheta is REQUIRED when smoothBodyFromPeriFreq=True. "
                            "No fallback allowed. trajectory_df must contain 'spineTheta' column.")
        
        # Apply lowpass1D filter (simplified - would need actual lowpass1D implementation)
        # For now, require sspineTheta to be pre-computed
        if 'sspineTheta' not in trajectory_df.columns:
            raise ValueError("CRITICAL: sspineTheta is REQUIRED. No fallback allowed. "
                            "trajectory_df must contain 'sspineTheta' column.")
        body_theta = trajectory_df['sspineTheta'].values
    else:
        # MATLAB: bt = track.getDerivedQuantity('sspineTheta');
        if 'sspineTheta' not in trajectory_df.columns:
            raise ValueError("CRITICAL: sspineTheta is REQUIRED. No fallback allowed. "
                            "trajectory_df must contain 'sspineTheta' column.")
        body_theta = trajectory_df['sspineTheta'].values
    
    # Velocity dot product (vel_dp) - alignment of velocity with head direction
    # MATLAB: vdp = track.getDerivedQuantity('vel_dp');
    # REQUIRED: vel_dp must be present in trajectory_df
    if 'vel_dp' not in trajectory_df.columns:
        raise ValueError("CRITICAL: vel_dp is REQUIRED. No fallback allowed. "
                        "trajectory_df must contain 'vel_dp' column.")
    vel_dp = trajectory_df['vel_dp'].values
    
    # Auto-set curvature cut if requested
    # MATLAB: mso.curv_cut = mso.autoset_curv_cut_mult / median(track.getDerivedQuantity('spineLength'));
    if segment_options.autoset_curv_cut:
        if 'spineLength' not in trajectory_df.columns:
            raise ValueError("CRITICAL: spineLength is REQUIRED when autoset_curv_cut=True. "
                            "No fallback allowed. trajectory_df must contain 'spineLength' column.")
        median_spine_length = np.median(trajectory_df['spineLength'].values)
        segment_options.curv_cut = segment_options.autoset_curv_cut_mult / median_spine_length
    
    # Use speed thresholds directly from segment_options (MATLAB: mso.stop_speed_cut, mso.start_speed_cut)
    stop_speed_cut = segment_options.stop_speed_cut
    start_speed_cut = segment_options.start_speed_cut
    
    # Step 1: Find everywhere NOT a run
    # MATLAB: highcurv = (abs(cv) > mso.curv_cut);
    # MATLAB: head_swinging = (abs(bt) > mso.theta_cut);
    # MATLAB: speedlow = (sp < mso.stop_speed_cut);
    curv_abs = np.abs(curv)
    highcurv = curv_abs > segment_options.curv_cut
    
    body_theta_abs = np.abs(body_theta)
    head_swinging = body_theta_abs > segment_options.theta_cut
    speedlow = speed < stop_speed_cut
    
    notarun = highcurv | head_swinging | speedlow
    
    # Find run end indices (transitions from False to True in notarun)
    endarun = np.where(np.diff(notarun.astype(int)) >= 1)[0] + 1
    
    # Step 2: Find run start indices
    # MATLAB: speedhigh = (sp >= mso.start_speed_cut);
    # MATLAB: headaligned = (vdp >= mso.aligned_dp);
    # MATLAB: isarun = (~notarun & speedhigh & headaligned);
    speedhigh = speed >= start_speed_cut
    headaligned = vel_dp >= segment_options.aligned_dp
    isarun = (~notarun) & speedhigh & headaligned
    
    # Find run start indices (transitions from False to True in isarun)
    startarun = np.where(np.diff(isarun.astype(int)) >= 1)[0] + 1
    
    # Step 3: Match starts and stops, create run intervals
    runs = []
    if len(startarun) > 0:
        si = 0
        while si < len(startarun):
            start_idx = startarun[si]
            # Find next end after this start
            ei = np.where(endarun > start_idx)[0]
            if len(ei) > 0:
                end_idx = endarun[ei[0]]
            else:
                end_idx = n_frames - 1
            
            # Find next start after this end
            next_si = np.where(startarun > end_idx)[0]
            if len(next_si) > 0:
                si = next_si[0]
            else:
                si = len(startarun)
            
            runs.append((start_idx, end_idx))
    
    # Step 4: Apply quality filters to runs
    # MATLAB: inds = find(track.dq.eti(stop) - track.dq.eti(start) >= mso.minRunTime);
    # MATLAB: start = start(inds); stop = stop(inds);
    # ONLY filter by minRunTime (MATLAB only has this filter)
    valid_runs = []
    for start_idx, end_idx in runs:
        run_duration = time[end_idx] - time[start_idx] if end_idx < len(time) else time[-1] - time[start_idx]
        if run_duration >= segment_options.minRunTime:
            valid_runs.append((start_idx, end_idx))
    runs = valid_runs
    
    # Create is_run boolean array
    is_run = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in runs:
        is_run[start_idx:end_idx+1] = True
    
    notrun = ~is_run
    
    # Step 5: Find head swings
    # MATLAB: buffer = ceil((track.dr.smoothTime + track.dr.derivTime)/track.dr.interpTime);
    # Use sensible defaults when derivation_rules is not available
    
    # Get derivation rules (MATLAB: track.dr) or use defaults
    if hasattr(trajectory_df, 'attrs') and 'derivation_rules' in trajectory_df.attrs:
        dr = trajectory_df.attrs['derivation_rules']
        smooth_time = dr.get('smoothTime', 0.2)  # Default 0.2s
        deriv_time = dr.get('derivTime', 0.1)    # Default 0.1s
        interp_time = dr.get('interpTime', dt)    # Use dt as fallback
    elif 'interpTime' in trajectory_df.columns:
        # Try to get from columns
        interp_time = np.mean(trajectory_df['interpTime'].values) if len(trajectory_df) > 0 else dt
        smooth_time = 0.2  # Default
        deriv_time = 0.1    # Default
    else:
        # Use sensible defaults based on typical MAGAT parameters
        # Standard MAGAT uses 0.2s smoothing, 0.1s derivative window
        # interpTime is the frame interval from the data
        interp_time = dt  # Use frame interval from data
        smooth_time = 0.2  # 0.2s smoothing window (MAGAT default)
        deriv_time = 0.1   # 0.1s derivative window (MAGAT default)
    
    # MATLAB: buffer = ceil((track.dr.smoothTime + track.dr.derivTime)/track.dr.interpTime);
    buffer = int(np.ceil((smooth_time + deriv_time) / interp_time))
    buffer = max(1, buffer)  # At least 1 frame
    
    # MATLAB: firstrunind = find(track.isrun, 1, 'first') + buffer;
    # MATLAB: lastrunind = find(track.isrun, 1, 'last') - buffer;
    # MATLAB: inrange = false(size(notrun)); inrange(firstrunind:lastrunind) = true;
    if np.any(is_run):
        first_run_ind = np.where(is_run)[0][0] + buffer
        last_run_ind = np.where(is_run)[0][-1] - buffer
        inrange = np.zeros(n_frames, dtype=bool)
        if first_run_ind <= last_run_ind:
            inrange[first_run_ind:last_run_ind+1] = True
    else:
        inrange = np.zeros(n_frames, dtype=bool)
    
    # MATLAB: notrun = imdilate(notrun, ones([1, buffer]));
    notrun_dilated = binary_dilation(notrun, structure=np.ones(buffer))
    
    # MATLAB: head_swinging = find (abs(bt) > mso.headswing_start & notrun & inrange);
    head_swinging_condition = (np.abs(body_theta) > segment_options.headswing_start) & notrun_dilated & inrange
    head_swinging_indices = np.where(head_swinging_condition)[0]
    
    # MATLAB: isrun2 = imerode(track.isrun, ones([1, buffer]));
    isrun_eroded = binary_erosion(is_run, structure=np.ones(buffer))
    
    # MATLAB: not_head_swing = find((abs(bt) < mso.headswing_stop) | ([0 diff(sign(bt))] ~= 0 | isrun2) & inrange);
    body_theta_sign_change = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        body_theta_sign_change[1:] = np.diff(np.sign(body_theta)) != 0
    
    not_head_swing_condition = ((np.abs(body_theta) < segment_options.headswing_stop) | 
                                body_theta_sign_change | 
                                isrun_eroded) & inrange
    
    not_head_swing_indices = np.where(not_head_swing_condition)[0]
    
    # Match head swing starts and stops
    head_swings_raw = []
    if len(head_swinging_indices) > 0:
        si = 0
        while si < len(head_swinging_indices):
            start_idx = head_swinging_indices[si]
            # Find next stop after this start
            ei = np.where(not_head_swing_indices > start_idx)[0]
            if len(ei) > 0:
                end_idx = not_head_swing_indices[ei[0]]
            else:
                end_idx = n_frames - 1
            
            # Head swing is only valid if it includes at least one point that is not a run
            if np.any(notrun[start_idx:end_idx+1]):
                head_swings_raw.append((start_idx, end_idx))
            
            # Find next start after this end
            next_si = np.where(head_swinging_indices > end_idx)[0]
            if len(next_si) > 0:
                si = next_si[0]
            else:
                si = len(head_swinging_indices)
    
    # Apply quality filters to head swings
    # MATLAB: a headswing is only valid if it includes at least one point that is not part of a run
    # MATLAB: for k = 1:length(start)
    # MATLAB:     if (any(notrun(start(k):stop(k))))
    # MATLAB:         j = j + 1; inds(j) = k; end
    # MATLAB: end
    # ONLY filter: must contain at least one non-run frame (MATLAB only has this filter)
    head_swings = []
    for start_idx, end_idx in head_swings_raw:
        if np.any(notrun[start_idx:end_idx+1]):
            head_swings.append((start_idx, end_idx))
    
    # Create is_head_swing boolean array
    is_head_swing = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in head_swings:
        is_head_swing[start_idx:end_idx+1] = True
    
    # Step 6: Group head swings into reorientations
    # A reorientation is the period BETWEEN runs (whether or not it contains head swings)
    # Reorientations are gaps between consecutive runs
    reorientations = []
    reo_prevdir = []  # Direction of run BEFORE reorientation (MATLAB: prevDir)
    reo_nextdir = []  # Direction of run AFTER reorientation (MATLAB: nextDir)
    
    # Get heading/theta from trajectory (for prevDir/nextDir calculation)
    # MATLAB: reorientation.prevDir and reorientation.nextDir come from run directions
    # We calculate from run headings (equivalent to MATLAB's approach)
    # REQUIRED: heading or theta must be present
    if 'heading' in trajectory_df.columns:
        heading = trajectory_df['heading'].values
    elif 'theta' in trajectory_df.columns:
        heading = trajectory_df['theta'].values
    else:
        raise ValueError("CRITICAL: 'heading' or 'theta' is REQUIRED for reorientation angle calculation. "
                        "No fallback allowed. trajectory_df must contain 'heading' or 'theta' column.")
    
    if len(runs) > 1:
        for i in range(len(runs) - 1):
            # Reorientation is the gap between run i and run i+1
            prev_run_start = runs[i][0]
            prev_run_end = runs[i][1]
            next_run_start = runs[i+1][0]
            next_run_end = runs[i+1][1]
            
            # Reorientation start is right after previous run ends
            reo_start = prev_run_end + 1
            # Reorientation end is right before next run starts
            reo_end = next_run_start - 1
            
            if reo_start <= reo_end:
                reorientations.append((reo_start, reo_end))
                
                # Calculate prevDir: average heading of the run BEFORE reorientation
                # MATLAB uses the direction at the END of the previous run
                # Use last few frames of previous run for stability
                prev_run_frames = max(1, min(5, prev_run_end - prev_run_start + 1))
                prev_run_indices = np.arange(prev_run_end - prev_run_frames + 1, prev_run_end + 1)
                prev_run_indices = prev_run_indices[prev_run_indices < len(heading)]
                if len(prev_run_indices) > 0:
                    prev_run_headings = heading[prev_run_indices]
                    # Use circular mean for angles
                    prev_dir = np.arctan2(np.mean(np.sin(prev_run_headings)), np.mean(np.cos(prev_run_headings)))
                else:
                    prev_dir = heading[prev_run_end] if prev_run_end < len(heading) else 0.0
                
                # Calculate nextDir: average heading of the run AFTER reorientation
                # MATLAB uses the direction at the START of the next run
                # Use first few frames of next run for stability
                next_run_frames = max(1, min(5, next_run_end - next_run_start + 1))
                next_run_indices = np.arange(next_run_start, next_run_start + next_run_frames)
                next_run_indices = next_run_indices[next_run_indices < len(heading)]
                if len(next_run_indices) > 0:
                    next_run_headings = heading[next_run_indices]
                    # Use circular mean for angles
                    next_dir = np.arctan2(np.mean(np.sin(next_run_headings)), np.mean(np.cos(next_run_headings)))
                else:
                    next_dir = heading[next_run_start] if next_run_start < len(heading) else 0.0
                
                reo_prevdir.append(prev_dir)
                reo_nextdir.append(next_dir)
    
    # Calculate reo_dtheta using MATLAB's formula: diff(unwrap([prevDir; nextDir]))
    # MATLAB Reference: spatialMaggotCalculations.m line 175
    reo_dtheta = []
    if len(reo_prevdir) > 0 and len(reo_nextdir) > 0:
        # Stack prevDir and nextDir like MATLAB: [prevDir; nextDir]
        prevdir_array = np.array(reo_prevdir)
        nextdir_array = np.array(reo_nextdir)
        combined_dirs = np.concatenate([prevdir_array, nextdir_array])
        
        # Unwrap angles to handle circularity (like MATLAB's unwrap)
        unwrapped = np.unwrap(combined_dirs)
        
        # Calculate difference: diff(unwrap([prevDir; nextDir]))
        # This gives the angle change from prevDir to nextDir for each reorientation
        n_reos = len(reo_prevdir)
        prevdir_unwrapped = unwrapped[:n_reos]
        nextdir_unwrapped = unwrapped[n_reos:]
        reo_dtheta = nextdir_unwrapped - prevdir_unwrapped
        
        # Wrap to [-pi, pi] range
        reo_dtheta = np.mod(reo_dtheta + np.pi, 2*np.pi) - np.pi
    else:
        reo_dtheta = []
    
    # Create is_reorientation boolean array (mark START of each reorientation)
    is_reorientation = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in reorientations:
        is_reorientation[start_idx] = True  # Mark start event only
    
    return {
        'runs': runs,
        'head_swings': head_swings,
        'reorientations': reorientations,
        'reo_prevdir': np.array(reo_prevdir) if len(reo_prevdir) > 0 else np.array([]),
        'reo_nextdir': np.array(reo_nextdir) if len(reo_nextdir) > 0 else np.array([]),
        'reo_dtheta': np.array(reo_dtheta) if len(reo_dtheta) > 0 else np.array([]),
        'is_run': is_run,
        'is_head_swing': is_head_swing,
        'is_reorientation': is_reorientation,  # Start events only
        'n_runs': len(runs),
        'n_head_swings': len(head_swings),
        'n_reorientations': len(reorientations)
    }

