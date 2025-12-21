#!/usr/bin/env python3
"""
MAGAT Spine Analysis Algorithms Implementation in Python

Implements MAGAT's algorithms for computing:
1. spineTheta (body bend angle) - optimal split point between head and tail
2. spineCurv (spine curvature)
3. Spine curve energy calculations

Based on MAGAT's calculateSpineTheta, calculateSpineCurv from:
@MaggotTrack/calculateDerivedQuantity.m
Reference: https://github.com/GilRaitses/magniphyq
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import uniform_filter1d


def fit_line(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a line to a set of points (MAGAT's fitLine algorithm).
    
    Parameters
    ----------
    pts : ndarray
        Points array of shape (2, N) or (2, N, M) for (x, y) coordinates
    
    Returns
    -------
    midpt : ndarray
        Midpoint of the line fit
    dirvec : ndarray
        Direction vector (normalized) of the line
    square_error : float or ndarray
        Sum of squared errors from line fit
    """
    # Handle different input shapes
    if pts.ndim == 2:
        # Single set of points: (2, N)
        pts = pts[:, :, np.newaxis]  # Add third dimension: (2, N, 1)
    
    # pts is now (2, N, M) where M is number of point sets
    n_pts = pts.shape[1]
    n_sets = pts.shape[2]
    
    # Compute midpoint (mean of points)
    midpt = np.mean(pts, axis=1)  # (2, M)
    
    # Compute direction vector using principal component analysis
    # Direction is eigenvector corresponding to largest eigenvalue
    dirvec = np.zeros((2, n_sets))
    square_error = np.zeros(n_sets)
    
    for i in range(n_sets):
        pts_set = pts[:, :, i]  # (2, N)
        pts_centered = pts_set - midpt[:, i:i+1]  # Center around mean
        
        # Compute covariance matrix
        cov = np.cov(pts_centered)
        
        # Find principal direction (eigenvector of largest eigenvalue)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        max_idx = np.argmax(eigenvals)
        dirvec[:, i] = eigenvecs[:, max_idx]
        
        # Normalize direction vector
        dirvec_norm = np.linalg.norm(dirvec[:, i])
        if dirvec_norm > 1e-6:
            dirvec[:, i] = dirvec[:, i] / dirvec_norm
        
        # Compute sum of squared errors (distance from points to line)
        # Project points onto direction vector, compute residuals
        projections = np.dot(pts_centered.T, dirvec[:, i])  # (N,)
        projected_pts = np.outer(projections, dirvec[:, i]).T  # (2, N)
        residuals = pts_centered - projected_pts
        square_error[i] = np.sum(residuals**2)
    
    # Remove extra dimension if single set
    if pts.shape[2] == 1:
        midpt = midpt[:, 0]
        dirvec = dirvec[:, 0]
        square_error = square_error[0]
    
    return midpt, dirvec, square_error


def calculate_spine_theta_magat(spine_points: np.ndarray) -> np.ndarray:
    """
    Calculate spineTheta (body bend angle) using MAGAT's algorithm.
    
    MAGAT's algorithm:
    1. Divides spine into head and tail segments at different split points
    2. Fits lines to each segment
    3. Finds split point that minimizes sum squared error
    4. Returns angle difference between head and tail directions at optimal split
    
    Parameters
    ----------
    spine_points : ndarray
        Spine points array of shape (n_frames, n_spine_pts, 2) or (n_spine_pts, 2) for single frame
    
    Returns
    -------
    spine_theta : ndarray
        Body bend angle for each frame (n_frames,) in radians
    """
    if spine_points.ndim == 2:
        # Single frame: (n_spine_pts, 2) -> add frame dimension
        spine_points = spine_points[np.newaxis, :, :]
    
    n_frames, n_spine_pts, _ = spine_points.shape
    
    # MAGAT uses range = max(2, round(nspinepts/5))
    range_size = max(2, round(n_spine_pts / 5))
    mid_idx = int(np.ceil(n_spine_pts / 2))
    midind = np.arange(mid_idx - range_size, mid_idx + range_size + 1)
    midind = midind[(midind >= 1) & (midind < n_spine_pts)]  # Valid indices
    
    spine_theta = np.zeros(n_frames)
    
    for frame_idx in range(n_frames):
        spine_frame = spine_points[frame_idx]  # (n_spine_pts, 2)
        
        # Transpose to (2, n_spine_pts) for fit_line
        spine_frame_T = spine_frame.T  # (2, n_spine_pts)
        
        sqe = np.zeros(len(midind))
        dt = np.zeros(len(midind))
        
        for j, split_idx in enumerate(midind):
            # Split spine into tail and head segments
            tail_segment = spine_frame_T[:, :split_idx]  # (2, split_idx)
            head_segment = spine_frame_T[:, split_idx-1:]  # (2, n_spine_pts - split_idx + 1)
            
            # Fit lines to each segment
            _, dvt, sqet = fit_line(tail_segment)
            _, dvh, sqeh = fit_line(head_segment)
            
            # Compute angles
            tht = np.arctan2(dvt[1], dvt[0])  # Tail angle
            thh = np.arctan2(dvh[1], dvh[0])  # Head angle
            
            # Unwrap angles and compute difference
            angles = np.array([tht, thh])
            angles_unwrapped = np.unwrap(angles)
            dt[j] = np.diff(angles_unwrapped)[0]
            
            # Sum squared error (minimize this)
            sqe[j] = sqet + sqeh
        
        # Find split point that minimizes squared error
        I = np.argmin(sqe)
        spine_theta[frame_idx] = dt[I]
    
    return spine_theta


def calculate_spine_curv_magat(spine_points: np.ndarray, 
                              interp_factor: int = 100) -> np.ndarray:
    """
    Calculate spine curvature using MAGAT's algorithm.
    
    MAGAT's algorithm:
    1. Interpolates spine to 100 points for smoothness
    2. Computes velocity (first derivative) and acceleration (second derivative)
    3. Uses formula: curvature = (v_x * a_y - v_y * a_x) / |v|^3
    
    Parameters
    ----------
    spine_points : ndarray
        Spine points array of shape (n_frames, n_spine_pts, 2) or (n_spine_pts, 2) for single frame
    interp_factor : int
        Number of points to interpolate to (default 100, matching MAGAT)
    
    Returns
    -------
    spine_curv : ndarray
        Curvature for each frame (n_frames,)
    """
    if spine_points.ndim == 2:
        # Single frame: (n_spine_pts, 2) -> add frame dimension
        spine_points = spine_points[np.newaxis, :, :]
    
    n_frames, n_spine_pts, _ = spine_points.shape
    
    spine_curv = np.zeros(n_frames)
    
    for frame_idx in range(n_frames):
        spine_frame = spine_points[frame_idx]  # (n_spine_pts, 2)
        
        # Interpolate spine to 100 points (MAGAT algorithm)
        # Use arc-length parameterization
        if n_spine_pts < 2:
            spine_curv[frame_idx] = 0.0
            continue
        
        # Compute arc lengths
        diffs = np.diff(spine_frame, axis=0)
        arc_lengths = np.cumsum(np.linalg.norm(diffs, axis=1))
        total_length = arc_lengths[-1] if len(arc_lengths) > 0 else 0
        
        if total_length < 1e-6:
            spine_curv[frame_idx] = 0.0
            continue
        
        # Parameterize from 0 to 1
        s_old = np.concatenate([[0], arc_lengths]) / total_length
        
        # New evenly spaced parameterization
        s_new = np.linspace(0, 1, interp_factor)
        
        # Interpolate x and y coordinates
        spine_interp = np.zeros((interp_factor, 2))
        spine_interp[:, 0] = np.interp(s_new, s_old, spine_frame[:, 0])
        spine_interp[:, 1] = np.interp(s_new, s_old, spine_frame[:, 1])
        
        # Compute velocity (first derivative)
        v = np.diff(spine_interp, axis=0)  # (interp_factor-1, 2)
        # Average adjacent differences for smoother velocity
        v_smooth = 0.5 * (v[1:] + v[:-1]) if len(v) > 1 else v
        
        # Compute acceleration (second derivative)
        if len(v_smooth) > 1:
            a = np.diff(v_smooth, axis=0)  # (interp_factor-2, 2)
            # Average adjacent differences
            if len(a) > 1:
                a_smooth = 0.5 * (a[1:] + a[:-1])
                v_for_a = v_smooth[1:-1]  # Match dimensions
            else:
                a_smooth = a
                v_for_a = v_smooth[:len(a)]
        else:
            spine_curv[frame_idx] = 0.0
            continue
        
        # Compute curvature: cv = (v_x * a_y - v_y * a_x) / |v|^3
        v_mag_sq = np.sum(v_for_a**2, axis=1)
        v_mag_cubed = v_mag_sq ** 1.5
        v_mag_cubed[v_mag_cubed < 1e-10] = 1e-10  # Avoid division by zero
        
        numerator = v_for_a[:, 0] * a_smooth[:, 1] - v_for_a[:, 1] * a_smooth[:, 0]
        curv_vals = numerator / v_mag_cubed
        
        # Mean curvature (MAGAT's approach)
        spine_curv[frame_idx] = np.mean(curv_vals)
    
    return spine_curv


def calculate_spine_curve_energy_magat(spine_theta: np.ndarray, 
                                       spine_curv: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate spine curve energy from spineTheta or curvature.
    
    MAGAT uses spineTheta for body bend angle. Curve energy can be computed as:
    - Energy = curvature^2 (for curvature-based)
    - Energy = (spineTheta)^2 (for bend angle-based)
    
    Parameters
    ----------
    spine_theta : ndarray
        Body bend angle (spineTheta) for each frame (n_frames,)
    spine_curv : ndarray, optional
        Curvature for each frame (n_frames,). If provided, uses curvature^2.
        Otherwise uses spineTheta^2.
    
    Returns
    -------
    spine_energy : ndarray
        Curve energy for each frame (n_frames,)
    """
    if spine_curv is not None:
        # Use curvature-based energy
        spine_energy = spine_curv ** 2
    else:
        # Use bend angle-based energy (MAGAT's approach)
        spine_energy = spine_theta ** 2
    
    return spine_energy


def lowpass1d(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Low-pass filter 1D data (MAGAT's lowpass1D function).
    
    Parameters
    ----------
    data : ndarray
        1D array to filter
    sigma : float
        Standard deviation for Gaussian filter (in samples)
    
    Returns
    -------
    filtered : ndarray
        Low-pass filtered data
    """
    if sigma < 1.0:
        return data
    
    # Use uniform filter as approximation (MAGAT may use different method)
    # Convert sigma to window size (rough approximation)
    window_size = int(2 * sigma + 1)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    
    filtered = uniform_filter1d(data.astype(float), size=window_size, mode='nearest')
    return filtered



