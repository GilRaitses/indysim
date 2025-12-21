#!/usr/bin/env python3
"""
MAGAT Speed Analysis Algorithms Implementation in Python

Implements MAGAT's algorithms for computing:
1. Speed (from velocity)
2. Normalized speed (nspeed)
3. Adjusted speed (adjspeed)
4. Speed difference (speed_diff_local)

Based on MAGAT's calculateDerivedQuantity speed calculation from:
@Track/calculateDerivedQuantity.m
@MaggotTrack/calculateDerivedQuantity.m
Reference: https://github.com/GilRaitses/magniphyq
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import uniform_filter1d
from scipy import ndimage


def gausskernel(sigma: float) -> np.ndarray:
    """
    Generate normalized Gaussian kernel (MAGAT's gaussKernel).
    
    MAGAT: Returns a normalized Gaussian 6 sigma in total width, with standard
    deviation sigma.
    
    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian (in samples)
    
    Returns
    -------
    kernel : ndarray
        Normalized Gaussian kernel
    """
    if sigma <= 0:
        return np.array([1.0])
    
    # MAGAT: x = floor(-3*sigma):ceil(3*sigma)
    x = np.arange(int(np.floor(-3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    
    # Gaussian: G(x) = exp(-x^2 / (2*sigma^2))
    g = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize: g = g / sum(g)
    g = g / np.sum(g)
    
    return g


def dgausskernel(sigma: float) -> np.ndarray:
    """
    Generate Gaussian derivative kernel (MAGAT's dgausskernel).
    
    MAGAT Algorithm:
    1. Get Gaussian kernel: gK = gaussKernel(sigma)
    2. Compute differences: dgd = diff(gK)
    3. Average adjacent differences: dg = [dgd[0], (dgd[0]+dgd[1])/2, ..., dgd[-1]]
    
    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian (in samples)
    
    Returns
    -------
    kernel : ndarray
        Gaussian derivative kernel
    """
    # Get Gaussian kernel
    gK = gausskernel(sigma)
    
    # Compute differences
    dgd = np.diff(gK)
    
    # Average adjacent differences (MAGAT's algorithm)
    dg = np.zeros(len(gK))
    dg[1:-1] = (dgd[:-1] + dgd[1:]) / 2
    dg[0] = dgd[0]
    dg[-1] = dgd[-1]
    
    return dg


def deriv(x: np.ndarray, sigma: float, padtype: str = 'linear') -> np.ndarray:
    """
    Compute derivative using Gaussian derivative kernel (MAGAT's deriv function).
    
    Parameters
    ----------
    x : ndarray
        Input signal (can be 1D or 2D, with shape (n_features, n_samples) for 2D)
    sigma : float
        Standard deviation of Gaussian derivative kernel (in samples)
    padtype : str
        Padding type: 'linear' (default) or 'circular'
    
    Returns
    -------
    dx : ndarray
        Derivative of x (same shape as x)
    """
    # Handle 1D vs 2D input
    if x.ndim == 1:
        xx = x.reshape(1, -1)  # Make it (1, n_samples)
        transpose_back = True
    else:
        xx = x
        transpose_back = False
    
    # Generate derivative kernel
    dg = dgausskernel(sigma)
    
    # Handle edge case: sigma too large for data
    padfront = int(np.ceil((len(dg) - 1) / 2))
    padback = len(dg) - padfront - 1
    
    if padfront >= xx.shape[1] or padback >= xx.shape[1]:
        # Sigma too large - reduce kernel size
        ks = int(np.floor((xx.shape[1] - 1) / 2))
        center_idx = len(dg) // 2
        inds = np.arange(center_idx - ks, center_idx + ks + 1)
        dg = dg[inds]
        padfront = int(np.ceil((len(dg) - 1) / 2))
        padback = len(dg) - padfront - 1
    
    # Padding
    if padtype == 'circular':
        frontpad = xx[:, -padfront:]
        backpad = xx[:, :padback]
    else:
        # Linear padding: mirror at boundaries
        frontpad = 2 * xx[:, 0:1] - xx[:, padfront:0:-1] if padfront > 0 else np.array([]).reshape(xx.shape[0], 0)
        backpad = 2 * xx[:, -1:] - xx[:, -2:-padback-2:-1] if padback > 0 else np.array([]).reshape(xx.shape[0], 0)
    
    # Concatenate padded data
    if padfront > 0 and padback > 0:
        xx_padded = np.concatenate([frontpad, xx, backpad], axis=1)
    elif padfront > 0:
        xx_padded = np.concatenate([frontpad, xx], axis=1)
    elif padback > 0:
        xx_padded = np.concatenate([xx, backpad], axis=1)
    else:
        xx_padded = xx
    
    # Convolve with derivative kernel
    dx = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        dx[i, :] = np.convolve(xx_padded[i, :], dg, mode='valid')
    
    # Transpose back if needed
    if transpose_back:
        dx = dx[0, :]
    
    return dx


def calculate_speed_magat(positions: np.ndarray, 
                         interp_time: float = 0.1,
                         smooth_time: float = 0.1,
                         deriv_time: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate speed using MAGAT's algorithm.
    
    MAGAT Algorithm:
    1. Interpolate locations (iloc) - already done if positions are pre-interpolated
    2. Smooth locations (sloc) using lowpass1D with sigma = smoothTime/interpTime
    3. Compute velocity: vel = deriv(sloc, sigma) / interpTime where sigma = derivTime/interpTime
    4. Compute speed: speed = sqrt(sum(vel^2))
    
    Parameters
    ----------
    positions : ndarray
        Position array, shape (n_frames, 2) or (2, n_frames) for (x, y) coordinates
    interp_time : float
        Interpolation time step (seconds). Default 0.1s (10 Hz)
    smooth_time : float
        Smoothing time constant (seconds). Default 0.1s
    deriv_time : float
        Derivative time constant (seconds). Default 0.1s
    
    Returns
    -------
    speed : ndarray
        Speed array (n_frames,)
    velocity : ndarray
        Velocity array (2, n_frames) for (vx, vy)
    smoothed_locations : ndarray
        Smoothed location array (2, n_frames) for (x, y)
    """
    # Normalize to (2, n_frames) format
    if positions.ndim == 2:
        if positions.shape[0] == 2:
            sloc = positions.copy()
        elif positions.shape[1] == 2:
            sloc = positions.T
        else:
            raise ValueError(f"positions must be (n_frames, 2) or (2, n_frames), got {positions.shape}")
    else:
        raise ValueError(f"positions must be 2D, got {positions.ndim}D")
    
    # Smooth locations using lowpass1D (Gaussian filter)
    from magat_spine_analysis import lowpass1d
    sigma_smooth = smooth_time / interp_time
    sloc_smooth = np.zeros_like(sloc)
    sloc_smooth[0, :] = lowpass1d(sloc[0, :], sigma_smooth)
    sloc_smooth[1, :] = lowpass1d(sloc[1, :], sigma_smooth)
    
    # Compute velocity using derivative
    sigma_deriv = deriv_time / interp_time
    vel = deriv(sloc_smooth, sigma_deriv) / interp_time  # Velocity in units/second
    
    # Compute speed: sqrt(sum(vel^2))
    speed = np.sqrt(np.sum(vel**2, axis=0))
    
    return speed, vel, sloc_smooth


def calculate_normalized_speed(speed: np.ndarray) -> np.ndarray:
    """
    Calculate normalized speed (nspeed) - normalized by median speed.
    
    MAGAT: nspeed = speed / median(speed)
    
    Parameters
    ----------
    speed : ndarray
        Speed array
    
    Returns
    -------
    nspeed : ndarray
        Normalized speed
    """
    median_speed = np.median(speed[speed > 0])  # Use non-zero speeds
    if median_speed <= 0:
        median_speed = np.mean(speed) if np.any(speed > 0) else 1.0
    
    nspeed = speed / median_speed
    return nspeed


def calculate_adjusted_speed(speed: np.ndarray, 
                            is_run: Optional[np.ndarray] = None,
                            interp_time: float = 0.1,
                            smooth_time: float = 0.1) -> np.ndarray:
    """
    Calculate adjusted speed (adjspeed) - normalized by median run speed, then smoothed.
    
    MAGAT Algorithm:
    - If runs are available: adjspeed = lowpass1D(speed / median(speed[is_run]), 3*smoothTime/interpTime)
    - Otherwise: adjspeed = lowpass1D(nspeed, 3*smoothTime/interpTime)
    
    Parameters
    ----------
    speed : ndarray
        Speed array
    is_run : ndarray, optional
        Boolean array indicating run periods
    interp_time : float
        Interpolation time step (seconds)
    smooth_time : float
        Smoothing time constant (seconds)
    
    Returns
    -------
    adjspeed : ndarray
        Adjusted speed
    """
    from magat_spine_analysis import lowpass1d
    
    lrtime = smooth_time * 3
    sigma_lr = lrtime / interp_time
    
    if is_run is not None and np.any(is_run):
        # Normalize by median run speed
        median_run_speed = np.median(speed[is_run])
        if median_run_speed <= 0:
            median_run_speed = np.mean(speed[is_run]) if np.any(speed[is_run] > 0) else 1.0
        
        normalized = speed / median_run_speed
    else:
        # Use normalized speed instead
        normalized = calculate_normalized_speed(speed)
    
    # Apply long-term smoothing
    adjspeed = lowpass1d(normalized, sigma_lr)
    
    return adjspeed


def calculate_speed_diff_local(speed: np.ndarray, interp_time: float = 0.1) -> np.ndarray:
    """
    Calculate local speed difference (speed_diff_local).
    
    MAGAT Algorithm:
    - Uses median filter with window size = 240 seconds / interpTime
    - speed_diff_local = speed - medfilt(speed)
    
    Parameters
    ----------
    speed : ndarray
        Speed array
    interp_time : float
        Interpolation time step (seconds)
    
    Returns
    -------
    speed_diff_local : ndarray
        Local speed difference
    """
    npts = int(240 / interp_time)
    
    if len(speed) < npts:
        # Use median of all speeds if track is too short
        spfilt = np.full_like(speed, np.median(speed))
    else:
        # Median filter with symmetric padding
        spfilt = ndimage.median_filter(speed, size=npts, mode='reflect')
    
    speed_diff_local = speed - spfilt
    
    return speed_diff_local


def calculate_velocity_from_trajectory(trajectory_df: 'pd.DataFrame',
                                      x_col: str = 'x',
                                      y_col: str = 'y',
                                      time_col: str = 'time',
                                      interp_time: float = 0.1,
                                      smooth_time: float = 0.1,
                                      deriv_time: float = 0.1) -> 'pd.DataFrame':
    """
    Calculate MAGAT-compatible speed and velocity from trajectory DataFrame.
    
    Parameters
    ----------
    trajectory_df : DataFrame
        Trajectory DataFrame with x, y, time columns
    x_col : str
        Column name for x coordinates
    y_col : str
        Column name for y coordinates
    time_col : str
        Column name for time
    interp_time : float
        Interpolation time step (seconds)
    smooth_time : float
        Smoothing time constant (seconds)
    deriv_time : float
        Derivative time constant (seconds)
    
    Returns
    -------
    trajectory_df : DataFrame
        Original DataFrame with added columns:
        - speed_magat: MAGAT-computed speed
        - velocity_x, velocity_y: Velocity components
        - nspeed: Normalized speed
        - adjspeed: Adjusted speed (requires is_run column)
        - speed_diff_local: Local speed difference
    """
    # Extract positions
    positions = np.array([trajectory_df[x_col].values, trajectory_df[y_col].values])
    
    # Calculate speed and velocity
    speed, velocity, smoothed_locs = calculate_speed_magat(
        positions, interp_time, smooth_time, deriv_time
    )
    
    # Add to DataFrame
    trajectory_df = trajectory_df.copy()
    trajectory_df['speed_magat'] = speed
    trajectory_df['velocity_x'] = velocity[0, :]
    trajectory_df['velocity_y'] = velocity[1, :]
    
    # Calculate normalized speed
    trajectory_df['nspeed'] = calculate_normalized_speed(speed)
    
    # Calculate adjusted speed (if is_run available)
    if 'is_run' in trajectory_df.columns:
        trajectory_df['adjspeed'] = calculate_adjusted_speed(
            speed, trajectory_df['is_run'].values, interp_time, smooth_time
        )
    else:
        trajectory_df['adjspeed'] = calculate_adjusted_speed(
            speed, None, interp_time, smooth_time
        )
    
    # Calculate local speed difference
    trajectory_df['speed_diff_local'] = calculate_speed_diff_local(speed, interp_time)
    
    return trajectory_df

