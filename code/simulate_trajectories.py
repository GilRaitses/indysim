#!/usr/bin/env python3
"""
Trajectory Simulator

Simulates full larval trajectories using:
1. Hazard model for event timing
2. Empirical turn angle and duration distributions
3. Simple RUN/TURN state machine

Usage:
    python scripts/simulate_trajectories.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy import stats
import matplotlib.pyplot as plt

from analytic_hazard import AnalyticHazardModel, KernelParams
from load_fitting_data import get_led_timing, LED_ON_DURATION, LED_OFF_DURATION, FIRST_LED_ONSET


@dataclass
class TrajectoryParams:
    """Parameters for trajectory simulation."""
    # Run kinematics
    run_speed: float = 1.0  # mm/s
    heading_noise: float = 0.03  # rad/sqrt(s) - diffusion coefficient
    
    # Turn kinematics
    turn_speed_factor: float = 0.4  # Speed during turn (fraction of run speed)
    
    # Turn angle distribution (Normal)
    turn_angle_mu: float = 0.12  # rad (~7 deg rightward bias)
    turn_angle_sigma: float = 1.50  # rad (~86 deg)
    
    # Turn duration distribution (Lognormal)
    turn_duration_s: float = 0.589  # shape parameter
    turn_duration_scale: float = 1.287  # scale parameter
    
    # Simulation parameters
    dt: float = 0.05  # Time step (s) - matches 20 Hz frame rate


class TrajectorySimulator:
    """
    Simulate larval trajectories with RUN/TURN state machine.
    
    States:
    - RUN: Forward motion with small heading noise
    - TURN: Heading change with reduced speed
    """
    
    def __init__(
        self,
        hazard_model: AnalyticHazardModel,
        params: TrajectoryParams = None
    ):
        self.model = hazard_model
        self.params = params or TrajectoryParams()
        
        # Load turn distributions if available
        dist_path = Path('data/model/turn_distributions.json')
        if dist_path.exists():
            with open(dist_path) as f:
                dists = json.load(f)
            
            # Update params from fitted distributions
            angle = dists['turn_angle']['distribution']['normal']
            self.params.turn_angle_mu = angle['mu']
            self.params.turn_angle_sigma = angle['sigma']
            
            dur = dists['turn_duration']['distribution']['lognormal']
            self.params.turn_duration_s = dur['s']
            self.params.turn_duration_scale = dur['scale']
            
            print(f"Loaded turn distributions from {dist_path}")
    
    def sample_turn_angle(self, rng: np.random.Generator) -> float:
        """Sample a turn angle from the fitted distribution."""
        return rng.normal(self.params.turn_angle_mu, self.params.turn_angle_sigma)
    
    def sample_turn_duration(self, rng: np.random.Generator) -> float:
        """Sample a turn duration from the lognormal distribution."""
        # Lognormal: X = exp(mu + sigma * Z) where Z ~ N(0,1)
        # scipy parameterization: s=sigma, scale=exp(mu)
        return stats.lognorm.rvs(
            self.params.turn_duration_s,
            scale=self.params.turn_duration_scale,
            random_state=rng
        )
    
    def simulate(
        self,
        duration: float,
        led_onsets: np.ndarray,
        led_offsets: np.ndarray,
        track_intercept: float = 0.0,
        seed: int = None,
        filter_true_turns: bool = True
    ) -> pd.DataFrame:
        """
        Simulate a single larval trajectory.
        
        Parameters
        ----------
        duration : float
            Simulation duration in seconds
        led_onsets : ndarray
            LED onset times
        led_offsets : ndarray
            LED offset times
        track_intercept : float
            Track-specific intercept adjustment
        seed : int
            Random seed
        filter_true_turns : bool
            If True, only count turns with duration > 0.1s as "true turns"
        
        Returns
        -------
        trajectory : DataFrame
            Trajectory with columns: time, x, y, theta, state, led_on, is_turn
        """
        rng = np.random.default_rng(seed)
        dt = self.params.dt
        
        # Initialize state
        x, y, theta = 0.0, 0.0, rng.uniform(-np.pi, np.pi)
        state = 'RUN'
        turn_end_time = 0.0
        turn_angle = 0.0
        turn_duration = 0.0
        turn_angle_rate = 0.0
        
        # Generate event times using hazard model
        event_times = self.model.simulate_events_discrete(
            duration=duration,
            led_onset_times=led_onsets,
            led_offset_times=led_offsets,
            track_intercept=track_intercept,
            seed=seed
        )
        event_idx = 0
        
        # Storage
        records = []
        
        # Time loop
        t = 0.0
        while t < duration:
            # Determine LED state
            led_on = False
            for i in range(len(led_onsets)):
                if i < len(led_offsets):
                    if led_onsets[i] <= t < led_offsets[i]:
                        led_on = True
                        break
            
            # State machine
            if state == 'RUN':
                # Check for turn event
                if event_idx < len(event_times) and t >= event_times[event_idx]:
                    state = 'TURN'
                    turn_angle = self.sample_turn_angle(rng)
                    turn_duration = self.sample_turn_duration(rng)
                    turn_end_time = t + turn_duration
                    turn_angle_rate = turn_angle / turn_duration
                    event_idx += 1
                else:
                    # Run dynamics
                    speed = self.params.run_speed
                    x += speed * np.cos(theta) * dt
                    y += speed * np.sin(theta) * dt
                    
                    # Heading noise (Brownian)
                    theta += rng.normal(0, self.params.heading_noise * np.sqrt(dt))
                    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi  # Wrap to [-π, π]
            
            elif state == 'TURN':
                # Apply heading change
                theta += turn_angle_rate * dt
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
                
                # Reduced speed during turn
                speed = self.params.run_speed * self.params.turn_speed_factor
                x += speed * np.cos(theta) * dt
                y += speed * np.sin(theta) * dt
                
                # Check if turn complete
                if t >= turn_end_time:
                    state = 'RUN'
            
            # Record
            is_true_turn = (state == 'TURN' and turn_duration > 0.1) if filter_true_turns else (state == 'TURN')
            records.append({
                'time': t,
                'x': x,
                'y': y,
                'theta': theta,
                'state': state,
                'led_on': led_on,
                'is_turn': is_true_turn
            })
            
            t += dt
        
        return pd.DataFrame(records)
    
    def simulate_batch(
        self,
        n_tracks: int,
        duration: float,
        led_onsets: np.ndarray,
        led_offsets: np.ndarray,
        track_intercept_std: float = 0.47,
        seed: int = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Simulate multiple trajectories.
        
        Parameters
        ----------
        n_tracks : int
            Number of tracks to simulate
        duration : float
            Duration per track (seconds)
        led_onsets, led_offsets : ndarray
            LED timing
        track_intercept_std : float
            Standard deviation of track intercepts
        seed : int
            Random seed
        
        Returns
        -------
        trajectories : dict
            Mapping track_id -> trajectory DataFrame
        """
        rng = np.random.default_rng(seed)
        trajectories = {}
        
        for track_id in range(1, n_tracks + 1):
            intercept = rng.normal(0, track_intercept_std)
            traj = self.simulate(
                duration=duration,
                led_onsets=led_onsets,
                led_offsets=led_offsets,
                track_intercept=intercept,
                seed=seed + track_id if seed else None
            )
            traj['track_id'] = track_id
            trajectories[track_id] = traj
            
            # Progress
            if track_id % 10 == 0:
                print(f"  Simulated {track_id}/{n_tracks} tracks")
        
        return trajectories


def plot_trajectories(
    trajectories: Dict[int, pd.DataFrame],
    led_onsets: np.ndarray,
    led_offsets: np.ndarray,
    output_path: Path,
    n_show: int = 3
):
    """Plot example trajectories."""
    fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 5))
    if n_show == 1:
        axes = [axes]
    
    track_ids = list(trajectories.keys())[:n_show]
    
    for ax, track_id in zip(axes, track_ids):
        traj = trajectories[track_id]
        
        # Color by state
        run_mask = traj['state'] == 'RUN'
        turn_mask = traj['state'] == 'TURN'
        
        ax.plot(traj.loc[run_mask, 'x'], traj.loc[run_mask, 'y'],
                'b.', markersize=1, alpha=0.5, label='Run')
        ax.plot(traj.loc[turn_mask, 'x'], traj.loc[turn_mask, 'y'],
                'r.', markersize=2, alpha=0.8, label='Turn')
        
        # Mark start
        ax.plot(traj.iloc[0]['x'], traj.iloc[0]['y'], 'go', markersize=10, label='Start')
        
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(f'Track {track_id}', fontsize=14)
        ax.legend(loc='upper right')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to {output_path}")


def plot_trajectory_with_led(
    traj: pd.DataFrame,
    led_onsets: np.ndarray,
    led_offsets: np.ndarray,
    output_path: Path
):
    """Plot single trajectory with LED timing overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Trajectory plot
    ax = axes[0]
    
    # Color by LED state
    led_off_mask = ~traj['led_on']
    led_on_mask = traj['led_on']
    
    ax.plot(traj.loc[led_off_mask, 'x'], traj.loc[led_off_mask, 'y'],
            'b.', markersize=1, alpha=0.3, label='LED OFF')
    ax.plot(traj.loc[led_on_mask, 'x'], traj.loc[led_on_mask, 'y'],
            'orange', marker='.', markersize=1, alpha=0.3, linestyle='none', label='LED ON')
    
    # Mark turns
    turn_mask = traj['is_turn']
    ax.plot(traj.loc[turn_mask, 'x'], traj.loc[turn_mask, 'y'],
            'r.', markersize=3, alpha=0.8, label='Turn')
    
    ax.plot(traj.iloc[0]['x'], traj.iloc[0]['y'], 'go', markersize=10, label='Start')
    ax.plot(traj.iloc[-1]['x'], traj.iloc[-1]['y'], 'k*', markersize=10, label='End')
    
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_title('Simulated Larval Trajectory', fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # LED and event timing
    ax = axes[1]
    
    # LED trace
    t = traj['time'].values
    led = traj['led_on'].astype(float).values
    ax.fill_between(t, 0, led, color='yellow', alpha=0.5, label='LED ON')
    
    # Turn events
    turn_times = traj[traj['is_turn'] & (traj['state'].shift(1) == 'RUN')]['time'].values
    ax.scatter(turn_times, np.ones_like(turn_times) * 0.5, c='red', s=20, 
               marker='|', label=f'Turns (n={len(turn_times)})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('LED / Events', fontsize=12)
    ax.set_xlim(0, t.max())
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory+LED plot to {output_path}")


def main():
    print("=" * 70)
    print("TRAJECTORY SIMULATION")
    print("=" * 70)
    
    # Create hazard model
    model = AnalyticHazardModel()
    
    # Create simulator
    simulator = TrajectorySimulator(model)
    
    # LED timing (matching experiment)
    duration = 1200.0  # 20 minutes
    n_cycles = int(np.ceil((duration - FIRST_LED_ONSET) / (LED_ON_DURATION + LED_OFF_DURATION))) + 1
    led_onsets = np.array([FIRST_LED_ONSET + i * (LED_ON_DURATION + LED_OFF_DURATION) 
                           for i in range(n_cycles)])
    led_offsets = led_onsets + LED_ON_DURATION
    led_onsets = led_onsets[led_onsets < duration]
    led_offsets = led_offsets[led_offsets < duration]
    
    print(f"\nSimulation parameters:")
    print(f"  Duration: {duration/60:.1f} min")
    print(f"  LED cycles: {len(led_onsets)}")
    print(f"  Run speed: {simulator.params.run_speed} mm/s")
    print(f"  Turn angle σ: {np.degrees(simulator.params.turn_angle_sigma):.1f}°")
    print(f"  Turn duration scale: {simulator.params.turn_duration_scale:.2f}s")
    
    # Simulate batch
    print(f"\nSimulating 5 trajectories...")
    trajectories = simulator.simulate_batch(
        n_tracks=5,
        duration=duration,
        led_onsets=led_onsets,
        led_offsets=led_offsets,
        seed=42
    )
    
    # Statistics
    print("\n" + "=" * 50)
    print("SIMULATION STATISTICS")
    print("=" * 50)
    
    total_turns = 0
    for track_id, traj in trajectories.items():
        n_turns = traj['is_turn'].sum() // int(1 / simulator.params.dt)  # Approximate
        # Better: count state transitions
        state_changes = (traj['state'] != traj['state'].shift()).sum()
        n_turns = state_changes // 2
        total_turns += n_turns
        print(f"  Track {track_id}: ~{n_turns} turns")
    
    print(f"\n  Total turns: ~{total_turns}")
    print(f"  Mean rate: {total_turns / 5 / (duration/60):.2f} turns/min/track")
    
    # Create figures directory
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot trajectories
    plot_trajectories(trajectories, led_onsets, led_offsets,
                      figures_dir / 'example_trajectories.png', n_show=3)
    
    # Plot single trajectory with LED overlay
    plot_trajectory_with_led(trajectories[1], led_onsets, led_offsets,
                             figures_dir / 'trajectory_with_led.png')
    
    # Save sample trajectories
    output_dir = Path('data/simulated')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for track_id, traj in trajectories.items():
        traj.to_parquet(output_dir / f'trajectory_{track_id}.parquet', index=False)
    print(f"\nSaved trajectories to {output_dir}/")
    
    print("\nTrajectory simulation complete!")


if __name__ == '__main__':
    main()


