#!/usr/bin/env python3
"""
Analytic Hazard Model for INDYsim

Provides efficient kernel evaluation using pre-computed lookup tables
for the gamma-difference LED-ON kernel and exponential LED-OFF rebound.

Model:
    λ(t) = exp(β₀ + u_track + K_on(time_since_LED_onset) + K_off(time_since_LED_offset))

Where:
    K_on(t) = A × Γ(t; α₁, β₁) - B × Γ(t; α₂, β₂)
    K_off(t) = D × exp(-t/τ_off)

Usage:
    from scripts.analytic_hazard import AnalyticHazardModel
    
    model = AnalyticHazardModel()
    hazard = model.compute_hazard(time_since_onset, time_since_offset, track_intercept)
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class KernelParams:
    """Parameters for the gamma-difference kernel."""
    # LED-ON kernel (gamma-difference)
    A: float = 0.456       # Fast excitatory amplitude
    alpha1: float = 2.22   # Fast shape
    beta1: float = 0.132   # Fast scale (seconds)
    B: float = 12.54       # Slow suppressive amplitude
    alpha2: float = 4.38   # Slow shape
    beta2: float = 0.869   # Slow scale (seconds)
    
    # LED-OFF rebound
    D: float = -0.114      # Rebound amplitude
    tau_off: float = 2.0   # Rebound timescale (seconds)
    
    # Global intercept (log events per FRAME at 20 Hz)
    # Calibrated to match empirical event rate of ~12.7 turns/min
    intercept: float = -4.5485
    
    # Frame rate used in GLM fitting
    frame_rate: float = 20.0  # Hz
    
    # Kernel support
    kernel_duration: float = 10.0  # seconds


class AnalyticHazardModel:
    """
    Efficient hazard model using pre-computed kernel lookup table.
    
    The gamma-difference kernel is pre-computed on a fine grid and
    accessed via linear interpolation for fast evaluation.
    """
    
    def __init__(self, params: Optional[KernelParams] = None, 
                 resolution: float = 0.01):
        """
        Initialize the hazard model with kernel lookup table.
        
        Parameters
        ----------
        params : KernelParams, optional
            Kernel parameters. Uses fitted defaults if not provided.
        resolution : float
            Time resolution for lookup table (seconds). Default 0.01s.
        """
        self.params = params or KernelParams()
        self.resolution = resolution
        
        # Pre-compute lookup table
        self._build_lookup_table()
    
    def _build_lookup_table(self):
        """Pre-compute K_on values on a fine grid."""
        n_points = int(self.params.kernel_duration / self.resolution) + 1
        self.t_grid = np.linspace(0, self.params.kernel_duration, n_points)
        
        # Compute gamma-difference kernel
        self.K_table = self._gamma_diff_kernel(self.t_grid)
        
        # Store derived quantities
        self.peak_fast = (self.params.alpha1 - 1) * self.params.beta1
        self.peak_slow = (self.params.alpha2 - 1) * self.params.beta2
        self.mean_fast = self.params.alpha1 * self.params.beta1
        self.mean_slow = self.params.alpha2 * self.params.beta2
    
    def _gamma_diff_kernel(self, t: np.ndarray) -> np.ndarray:
        """
        Compute gamma-difference kernel.
        
        K_on(t) = A × Γ(t; α₁, β₁) - B × Γ(t; α₂, β₂)
        """
        p = self.params
        pdf1 = gamma_dist.pdf(t, p.alpha1, scale=p.beta1)
        pdf2 = gamma_dist.pdf(t, p.alpha2, scale=p.beta2)
        return p.A * np.nan_to_num(pdf1) - p.B * np.nan_to_num(pdf2)
    
    def get_K_on(self, time_since_onset: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get LED-ON kernel value via lookup table interpolation.
        
        Parameters
        ----------
        time_since_onset : float or ndarray
            Time since LED turned on (seconds). 
            Negative values or values > kernel_duration return 0.
        
        Returns
        -------
        float or ndarray
            Kernel value (log-hazard contribution)
        """
        t = np.asarray(time_since_onset)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        result = np.zeros_like(t, dtype=float)
        valid = (t >= 0) & (t <= self.params.kernel_duration)
        
        if np.any(valid):
            result[valid] = np.interp(t[valid], self.t_grid, self.K_table)
        
        return float(result[0]) if scalar_input else result
    
    def get_K_off(self, time_since_offset: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get LED-OFF rebound kernel value.
        
        K_off(t) = D × exp(-t/τ_off)
        
        Parameters
        ----------
        time_since_offset : float or ndarray
            Time since LED turned off (seconds).
            Negative values return 0.
        
        Returns
        -------
        float or ndarray
            Kernel value (log-hazard contribution)
        """
        t = np.asarray(time_since_offset)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        result = np.zeros_like(t, dtype=float)
        valid = t >= 0
        
        if np.any(valid):
            result[valid] = self.params.D * np.exp(-t[valid] / self.params.tau_off)
        
        return float(result[0]) if scalar_input else result
    
    def compute_log_hazard(self, 
                           time_since_onset: Union[float, np.ndarray],
                           time_since_offset: Union[float, np.ndarray],
                           track_intercept: float = 0.0,
                           led_on: Union[bool, np.ndarray] = True) -> Union[float, np.ndarray]:
        """
        Compute log-hazard at given time(s).
        
        log λ(t) = β₀ + u_track + K_on(t_onset) + K_off(t_offset)
        
        Parameters
        ----------
        time_since_onset : float or ndarray
            Time since LED onset (used when LED is on)
        time_since_offset : float or ndarray
            Time since LED offset (used when LED is off)
        track_intercept : float
            Track-specific random effect (u_track)
        led_on : bool or ndarray
            Whether LED is currently on
        
        Returns
        -------
        float or ndarray
            Log-hazard value(s)
        """
        log_hazard = self.params.intercept + track_intercept
        
        # Add K_on contribution when LED is on
        led_on = np.asarray(led_on)
        if np.any(led_on):
            K_on = self.get_K_on(time_since_onset)
            if led_on.ndim == 0:
                if led_on:
                    log_hazard = log_hazard + K_on
            else:
                log_hazard = log_hazard + np.where(led_on, K_on, 0)
        
        # Add K_off contribution when LED is off
        if np.any(~led_on):
            K_off = self.get_K_off(time_since_offset)
            if led_on.ndim == 0:
                if not led_on:
                    log_hazard = log_hazard + K_off
            else:
                log_hazard = log_hazard + np.where(~led_on, K_off, 0)
        
        return log_hazard
    
    def compute_hazard(self,
                       time_since_onset: Union[float, np.ndarray],
                       time_since_offset: Union[float, np.ndarray],
                       track_intercept: float = 0.0,
                       led_on: Union[bool, np.ndarray] = True,
                       per_second: bool = True) -> Union[float, np.ndarray]:
        """
        Compute instantaneous hazard rate.
        
        The GLM intercept is in log(events per FRAME). To get events per second,
        we multiply by the frame rate.
        
        Parameters
        ----------
        per_second : bool
            If True, return hazard in events/second (multiply by frame_rate).
            If False, return hazard in events/frame (raw GLM output).
        
        Returns
        -------
        float or ndarray
            Hazard rate (events per second if per_second=True, else per frame)
        """
        log_h = self.compute_log_hazard(time_since_onset, time_since_offset,
                                         track_intercept, led_on)
        hazard_per_frame = np.exp(log_h)
        
        if per_second:
            # Convert from per-frame to per-second
            return hazard_per_frame * self.params.frame_rate
        else:
            return hazard_per_frame
    
    def simulate_events_thinning(self,
                                  duration: float,
                                  led_onset_times: np.ndarray,
                                  led_offset_times: np.ndarray,
                                  track_intercept: float = 0.0,
                                  dt: float = 0.05,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate events using thinning algorithm with proper unit handling.
        
        The GLM intercept is in log(events per frame). We convert to events/second
        for continuous-time simulation, then use thinning.
        
        Parameters
        ----------
        duration : float
            Total simulation duration (seconds)
        led_onset_times : ndarray
            Times when LED turns on
        led_offset_times : ndarray
            Times when LED turns off
        track_intercept : float
            Track-specific random effect
        dt : float
            Time step for simulation (seconds). Should match frame interval (0.05s at 20Hz).
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        ndarray
            Event times
        """
        rng = np.random.default_rng(seed)
        
        # Compute hazard on time grid
        t = np.arange(0, duration, dt)
        n_times = len(t)
        
        # Determine LED state and times since onset/offset for each timepoint
        led_on = np.zeros(n_times, dtype=bool)
        time_since_onset = np.zeros(n_times)
        time_since_offset = np.zeros(n_times)
        
        for i, ti in enumerate(t):
            # Find most recent LED onset before ti
            onset_mask = led_onset_times <= ti
            if np.any(onset_mask):
                last_onset = led_onset_times[onset_mask].max()
                time_since_onset[i] = ti - last_onset
            
            # Find most recent LED offset before ti
            offset_mask = led_offset_times <= ti
            if np.any(offset_mask):
                last_offset = led_offset_times[offset_mask].max()
                time_since_offset[i] = ti - last_offset
            
            # LED is on if last onset is more recent than last offset
            if np.any(onset_mask):
                if np.any(offset_mask):
                    led_on[i] = last_onset > last_offset
                else:
                    led_on[i] = True
        
        # Compute hazard in events/second (properly scaled from per-frame)
        hazard_per_sec = self.compute_hazard(time_since_onset, time_since_offset,
                                              track_intercept, led_on, per_second=True)
        
        # Thinning algorithm for inhomogeneous Poisson process
        # Upper bound for acceptance probability
        lambda_max = hazard_per_sec.max() * 1.1  # 10% buffer
        
        if lambda_max <= 0:
            return np.array([])
        
        # Generate candidate events from homogeneous Poisson with rate lambda_max
        expected_candidates = lambda_max * duration
        n_candidates = rng.poisson(expected_candidates)
        
        if n_candidates == 0:
            return np.array([])
        
        candidate_times = rng.uniform(0, duration, n_candidates)
        candidate_times.sort()
        
        # Accept/reject based on hazard ratio
        events = []
        for tc in candidate_times:
            # Find hazard at candidate time via interpolation
            idx = int(tc / dt)
            if idx >= n_times:
                idx = n_times - 1
            h_tc = hazard_per_sec[idx]
            
            # Accept with probability h(tc) / lambda_max
            if rng.random() < h_tc / lambda_max:
                events.append(tc)
        
        return np.array(events)
    
    def simulate_events_discrete(self,
                                  duration: float,
                                  led_onset_times: np.ndarray,
                                  led_offset_times: np.ndarray,
                                  track_intercept: float = 0.0,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate events using discrete-time (frame-by-frame) Bernoulli sampling.
        
        This exactly matches the GLM fitting procedure: at each frame,
        sample event ~ Bernoulli(p_frame) where p_frame = exp(η).
        
        Parameters
        ----------
        duration : float
            Total simulation duration (seconds)
        led_onset_times : ndarray
            Times when LED turns on
        led_offset_times : ndarray
            Times when LED turns off
        track_intercept : float
            Track-specific random effect
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        ndarray
            Event times
        """
        rng = np.random.default_rng(seed)
        
        # Frame-by-frame simulation at the GLM's frame rate
        dt = 1.0 / self.params.frame_rate  # e.g., 0.05s at 20 Hz
        t = np.arange(0, duration, dt)
        n_frames = len(t)
        
        # Determine LED state and times for each frame
        led_on = np.zeros(n_frames, dtype=bool)
        time_since_onset = np.zeros(n_frames)
        time_since_offset = np.zeros(n_frames)
        
        for i, ti in enumerate(t):
            onset_mask = led_onset_times <= ti
            if np.any(onset_mask):
                last_onset = led_onset_times[onset_mask].max()
                time_since_onset[i] = ti - last_onset
            
            offset_mask = led_offset_times <= ti
            if np.any(offset_mask):
                last_offset = led_offset_times[offset_mask].max()
                time_since_offset[i] = ti - last_offset
            
            if np.any(onset_mask):
                if np.any(offset_mask):
                    led_on[i] = last_onset > last_offset
                else:
                    led_on[i] = True
        
        # Compute hazard in events/frame (raw GLM output)
        p_frame = self.compute_hazard(time_since_onset, time_since_offset,
                                       track_intercept, led_on, per_second=False)
        
        # Clip probabilities to valid range [0, 1]
        p_frame = np.clip(p_frame, 0, 1)
        
        # Bernoulli sampling at each frame
        event_occurred = rng.random(n_frames) < p_frame
        event_times = t[event_occurred]
        
        return event_times
    
    def get_kernel_summary(self) -> dict:
        """Get summary of kernel parameters and derived quantities."""
        return {
            'params': {
                'A': self.params.A,
                'alpha1': self.params.alpha1,
                'beta1': self.params.beta1,
                'B': self.params.B,
                'alpha2': self.params.alpha2,
                'beta2': self.params.beta2,
                'D': self.params.D,
                'tau_off': self.params.tau_off,
                'intercept': self.params.intercept
            },
            'derived': {
                'peak_fast': self.peak_fast,
                'peak_slow': self.peak_slow,
                'mean_fast': self.mean_fast,
                'mean_slow': self.mean_slow,
                'amplitude_ratio': self.params.A / self.params.B
            },
            'lookup_table': {
                'resolution': self.resolution,
                'n_points': len(self.t_grid),
                'duration': self.params.kernel_duration
            }
        }


def create_default_model() -> AnalyticHazardModel:
    """Create hazard model with default (fitted) parameters."""
    return AnalyticHazardModel()


def main():
    """Test the analytic hazard model."""
    print("=" * 70)
    print("ANALYTIC HAZARD MODEL TEST")
    print("=" * 70)
    
    # Create model
    model = AnalyticHazardModel()
    
    # Print summary
    summary = model.get_kernel_summary()
    print("\nKernel Parameters:")
    for k, v in summary['params'].items():
        print(f"  {k}: {v}")
    
    print("\nDerived Quantities:")
    for k, v in summary['derived'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nLookup Table:")
    for k, v in summary['lookup_table'].items():
        print(f"  {k}: {v}")
    
    # Test kernel values
    print("\nKernel Samples:")
    test_times = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    for t in test_times:
        K_on = model.get_K_on(t)
        print(f"  K_on({t:.1f}s) = {K_on:.4f}")
    
    print("\nRebound Samples:")
    for t in [0, 0.5, 1.0, 2.0, 5.0]:
        K_off = model.get_K_off(t)
        print(f"  K_off({t:.1f}s) = {K_off:.4f}")
    
    # Test hazard scaling
    print("\nHazard Scaling Test (at baseline, no kernel contribution):")
    h_frame = model.compute_hazard(0, 100, 0, False, per_second=False)
    h_sec = model.compute_hazard(0, 100, 0, False, per_second=True)
    print(f"  Hazard per frame: {h_frame:.6f}")
    print(f"  Hazard per second: {h_sec:.6f}")
    print(f"  Frame rate: {model.params.frame_rate} Hz")
    print(f"  Ratio (should be frame_rate): {h_sec/h_frame:.1f}")
    
    # Test simulation with DISCRETE method (matches GLM exactly)
    print("\n" + "=" * 50)
    print("SIMULATION TEST: DISCRETE (FRAME-BY-FRAME)")
    print("=" * 50)
    
    led_onsets = np.array([0, 60, 120, 180, 240])
    led_offsets = np.array([30, 90, 150, 210, 270])
    duration = 300
    
    events_discrete = model.simulate_events_discrete(
        duration=duration,
        led_onset_times=led_onsets,
        led_offset_times=led_offsets,
        track_intercept=0.0,
        seed=42
    )
    
    print(f"  Duration: {duration}s (5 LED cycles)")
    print(f"  Events generated: {len(events_discrete)}")
    print(f"  Event rate: {len(events_discrete) / duration * 60:.2f} events/min")
    
    # Count events by LED state
    n_on = sum(1 for e in events_discrete if any(
        (led_onsets[i] <= e < led_offsets[i]) for i in range(len(led_onsets)) if led_offsets[i] <= duration
    ))
    n_off = len(events_discrete) - n_on
    
    print(f"  Events during LED-ON: {n_on}")
    print(f"  Events during LED-OFF: {n_off}")
    if n_on > 0:
        print(f"  Suppression ratio: {n_off / n_on:.1f}x more during OFF")
    
    # Test simulation with THINNING method
    print("\n" + "=" * 50)
    print("SIMULATION TEST: THINNING (CONTINUOUS-TIME)")
    print("=" * 50)
    
    events_thinning = model.simulate_events_thinning(
        duration=duration,
        led_onset_times=led_onsets,
        led_offset_times=led_offsets,
        track_intercept=0.0,
        dt=0.05,  # Match frame rate
        seed=42
    )
    
    print(f"  Duration: {duration}s (5 LED cycles)")
    print(f"  Events generated: {len(events_thinning)}")
    print(f"  Event rate: {len(events_thinning) / duration * 60:.2f} events/min")
    
    # Expected rate comparison
    print("\n" + "=" * 50)
    print("EXPECTED RATE ANALYSIS")
    print("=" * 50)
    
    # Baseline hazard (LED off, no kernel)
    baseline_per_frame = np.exp(model.params.intercept)
    baseline_per_sec = baseline_per_frame * model.params.frame_rate
    baseline_per_min = baseline_per_sec * 60
    
    print(f"  Baseline (LED-OFF, no kernel):")
    print(f"    Per frame: {baseline_per_frame:.6f}")
    print(f"    Per second: {baseline_per_sec:.4f}")
    print(f"    Per minute: {baseline_per_min:.2f}")
    
    # During LED-ON with suppression (kernel ~= -2 at peak suppression)
    suppressed_per_frame = np.exp(model.params.intercept - 2.0)
    suppressed_per_min = suppressed_per_frame * model.params.frame_rate * 60
    
    print(f"  During LED-ON (with K ~ -2 suppression):")
    print(f"    Per minute: {suppressed_per_min:.4f}")
    print(f"    Suppression factor: {baseline_per_min / suppressed_per_min:.1f}x")
    
    print("\nAnalytic hazard model ready.")


if __name__ == '__main__':
    main()


