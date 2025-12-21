#!/usr/bin/env python3
"""
Generate Simulated Tracks for Phenotyping Study

Uses the validated INDYsim trajectory simulator to generate complete 20-minute tracks
that can be used for:
1. Testing phenotype clustering methods
2. Power analysis for individual-level variability studies
3. Validating phenotyping pipelines before applying to real data

The simulator is validated (rate ratio = 0.97, PSTH correlation r = 0.84) and can
generate tracks with controlled variability in kernel parameters.
"""

import sys
import os
from pathlib import Path

# Add InDySim code directory to path
INDYSIM_CODE = Path('/Users/gilraitses/InDySim/code')
if INDYSIM_CODE.exists() and str(INDYSIM_CODE) not in sys.path:
    sys.path.insert(0, str(INDYSIM_CODE))

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats

try:
    from analytic_hazard import AnalyticHazardModel, KernelParams
except ImportError:
    print("Error: Could not import AnalyticHazardModel")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# LED timing constants (from load_fitting_data)
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
FIRST_LED_ONSET = 21.3


class TrajectorySimulator:
    """
    Simulate larval trajectories with RUN/TURN state machine.
    Simplified version for phenotyping track generation.
    """
    
    def __init__(self, hazard_model, params=None):
        self.model = hazard_model
        self.params = params or self._default_params()
    
    def _default_params(self):
        """Default trajectory parameters."""
        class Params:
            dt = 0.05  # 20 Hz
            run_speed = 1.0  # mm/s
            heading_noise = 0.03  # rad/sqrt(s)
            turn_speed_factor = 0.4
            turn_angle_mu = 0.12  # rad (~7 deg)
            turn_angle_sigma = 1.50  # rad (~86 deg)
            turn_duration_s = 0.589
            turn_duration_scale = 1.287
        return Params()
    
    def sample_turn_angle(self, rng):
        return rng.normal(self.params.turn_angle_mu, self.params.turn_angle_sigma)
    
    def sample_turn_duration(self, rng):
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
        """Simulate a single larval trajectory."""
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
                    theta += rng.normal(0, self.params.heading_noise * np.sqrt(dt))
                    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
            
            elif state == 'TURN':
                # Apply heading change
                theta += turn_angle_rate * dt
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
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


def generate_phenotype_tracks(
    n_tracks: int = 300,
    duration: float = 1200.0,  # 20 minutes
    condition: str = '0-250_Constant',
    phenotype_variation: bool = True,
    track_intercept_std: float = 0.47,
    tau1_variation: bool = True,
    tau2_variation: bool = False,
    amplitude_variation: bool = True,
    seed: int = 42,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Generate simulated tracks with controlled phenotype variation.
    """
    print("=" * 70)
    print("GENERATING SIMULATED TRACKS FOR PHENOTYPING")
    print("=" * 70)
    
    rng = np.random.default_rng(seed)
    
    # Base kernel parameters (from reference condition)
    base_params = KernelParams(
        A=0.456,
        alpha1=2.22,
        beta1=0.132,  # τ₁ = 0.29 s
        B=12.54,
        alpha2=4.38,
        beta2=0.869,  # τ₂ = 3.81 s
        D=-0.114,
        tau_off=2.0
    )
    
    # Condition-specific amplitude modulation (from factorial analysis)
    condition_amplitudes = {
        '0-250_Constant': 1.005,      # Reference
        '0-250_Cycling': 1.157,       # +15%
        '50-250_Constant': 0.340,     # -66%
        '50-250_Cycling': 0.492       # Combined
    }
    base_amplitude = condition_amplitudes.get(condition, 1.0)
    
    # LED timing
    n_cycles = int(np.ceil((duration - FIRST_LED_ONSET) / (LED_ON_DURATION + LED_OFF_DURATION))) + 1
    led_onsets = np.array([FIRST_LED_ONSET + i * (LED_ON_DURATION + LED_OFF_DURATION) 
                           for i in range(n_cycles)])
    led_offsets = led_onsets + LED_ON_DURATION
    led_onsets = led_onsets[led_onsets < duration]
    led_offsets = led_offsets[led_offsets < duration]
    
    print(f"\nSimulation parameters:")
    print(f"  Condition: {condition}")
    print(f"  Base amplitude: {base_amplitude:.3f}")
    print(f"  Duration: {duration/60:.1f} min")
    print(f"  LED cycles: {len(led_onsets)}")
    print(f"  Track intercept SD: {track_intercept_std}")
    print(f"  Phenotype variation: {phenotype_variation}")
    
    # Generate tracks with phenotype variation
    trajectories = {}
    track_summaries = []
    
    for track_id in range(1, n_tracks + 1):
        # Sample track-specific parameters
        track_intercept = rng.normal(0, track_intercept_std)
        
        # Phenotype variation in kernel parameters
        if phenotype_variation:
            if tau1_variation:
                # τ₁ varies 4-fold: 0.26-1.18 s (from timescale variability analysis)
                tau1 = rng.uniform(0.26, 1.18)
                beta1 = tau1 / base_params.alpha1
            else:
                beta1 = base_params.beta1
            
            if tau2_variation:
                # τ₂ relatively stable: 3.7-4.5 s
                tau2 = rng.uniform(3.7, 4.5)
                beta2 = tau2 / base_params.alpha2
            else:
                beta2 = base_params.beta2
            
            if amplitude_variation:
                # Amplitude variation: ±30% around condition mean
                amplitude_factor = rng.normal(1.0, 0.3)
                amplitude_factor = np.clip(amplitude_factor, 0.5, 1.5)
                A = base_params.A * amplitude_factor * base_amplitude
                B = base_params.B * amplitude_factor * base_amplitude
            else:
                A = base_params.A * base_amplitude
                B = base_params.B * base_amplitude
        else:
            # No variation (all tracks identical except intercept)
            beta1 = base_params.beta1
            beta2 = base_params.beta2
            A = base_params.A * base_amplitude
            B = base_params.B * base_amplitude
        
        # Create track-specific kernel parameters
        track_params = KernelParams(
            A=A,
            alpha1=base_params.alpha1,
            beta1=beta1,
            B=B,
            alpha2=base_params.alpha2,
            beta2=beta2,
            D=base_params.D,
            tau_off=base_params.tau_off
        )
        
        # Create hazard model with track-specific parameters
        model = AnalyticHazardModel(params=track_params)
        
        # Create simulator
        simulator = TrajectorySimulator(model)
        
        # Simulate track
        traj = simulator.simulate(
            duration=duration,
            led_onsets=led_onsets,
            led_offsets=led_offsets,
            track_intercept=track_intercept,
            seed=seed + track_id,
            filter_true_turns=True
        )
        
        traj['track_id'] = track_id
        traj['condition'] = condition
        trajectories[track_id] = traj
        
        # Compute track statistics
        n_events = traj['is_turn'].sum()
        turn_rate = n_events / (duration / 60)  # turns/min
        
        # Store summary
        track_summaries.append({
            'track_id': track_id,
            'condition': condition,
            'tau1': track_params.alpha1 * track_params.beta1,
            'tau2': track_params.alpha2 * track_params.beta2,
            'amplitude_A': A,
            'amplitude_B': B,
            'track_intercept': track_intercept,
            'n_events': n_events,
            'turn_rate': turn_rate,
            'duration': duration
        })
        
        if track_id % 50 == 0:
            print(f"  Generated {track_id}/{n_tracks} tracks")
    
    # Save trajectories
    if output_dir is None:
        output_dir = Path('data/simulated_phenotyping')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving trajectories to {output_dir}/...")
    for track_id, traj in trajectories.items():
        traj.to_parquet(output_dir / f'track_{track_id:04d}.parquet', index=False)
    
    # Save summary
    summary_df = pd.DataFrame(track_summaries)
    summary_df.to_csv(output_dir / 'track_summary.csv', index=False)
    summary_df.to_parquet(output_dir / 'track_summary.parquet', index=False)
    
    print(f"\nSummary statistics:")
    print(f"  Total tracks: {len(track_summaries)}")
    print(f"  Mean events per track: {summary_df['n_events'].mean():.1f}")
    print(f"  Mean turn rate: {summary_df['turn_rate'].mean():.2f} turns/min")
    print(f"  τ₁ range: {summary_df['tau1'].min():.2f} - {summary_df['tau1'].max():.2f} s")
    print(f"  τ₂ range: {summary_df['tau2'].min():.2f} - {summary_df['tau2'].max():.2f} s")
    
    return summary_df


def generate_multi_condition_dataset(
    n_tracks_per_condition: int = 75,
    duration: float = 1200.0,
    phenotype_variation: bool = True,
    seed: int = 42,
    output_dir: Path = None
) -> pd.DataFrame:
    """Generate simulated tracks for all 4 factorial conditions."""
    conditions = [
        '0-250_Constant',
        '0-250_Cycling',
        '50-250_Constant',
        '50-250_Cycling'
    ]
    
    if output_dir is None:
        output_dir = Path('data/simulated_phenotyping')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    
    for i, condition in enumerate(conditions):
        print(f"\n{'='*70}")
        print(f"CONDITION {i+1}/4: {condition}")
        print(f"{'='*70}")
        
        condition_dir = output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        
        condition_summary = generate_phenotype_tracks(
            n_tracks=n_tracks_per_condition,
            duration=duration,
            condition=condition,
            phenotype_variation=phenotype_variation,
            seed=seed + i * 1000,
            output_dir=condition_dir
        )
        
        all_summaries.append(condition_summary)
    
    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)
    
    combined.to_csv(output_dir / 'all_tracks_summary.csv', index=False)
    combined.to_parquet(output_dir / 'all_tracks_summary.parquet', index=False)
    
    print(f"\n{'='*70}")
    print("COMPLETE DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total tracks: {len(combined)}")
    print(f"Tracks per condition: {n_tracks_per_condition}")
    print(f"\nBy condition:")
    for cond in combined['condition'].unique():
        subset = combined[combined['condition'] == cond]
        print(f"  {cond}: {len(subset)} tracks, {subset['n_events'].mean():.1f} events/track")
    
    return combined


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate simulated tracks for phenotyping')
    parser.add_argument('--n-tracks', type=int, default=300,
                       help='Total number of tracks to generate (default: 300)')
    parser.add_argument('--per-condition', action='store_true',
                       help='Generate equal number per condition (n_tracks/4 each)')
    parser.add_argument('--condition', type=str, default='0-250_Constant',
                       choices=['0-250_Constant', '0-250_Cycling', '50-250_Constant', '50-250_Cycling'],
                       help='Single condition to generate (if not --per-condition)')
    parser.add_argument('--no-variation', action='store_true',
                       help='Disable phenotype variation (all tracks identical)')
    parser.add_argument('--output-dir', type=str, default='data/simulated_phenotyping',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.per_condition:
        n_per_cond = args.n_tracks // 4
        print(f"Generating {n_per_cond} tracks per condition (total: {n_per_cond * 4})")
        summary = generate_multi_condition_dataset(
            n_tracks_per_condition=n_per_cond,
            phenotype_variation=not args.no_variation,
            seed=args.seed,
            output_dir=output_dir
        )
    else:
        print(f"Generating {args.n_tracks} tracks for condition: {args.condition}")
        summary = generate_phenotype_tracks(
            n_tracks=args.n_tracks,
            condition=args.condition,
            phenotype_variation=not args.no_variation,
            seed=args.seed,
            output_dir=output_dir
        )
    
    print(f"\n✓ Complete! Tracks saved to {output_dir}/")
    print(f"  Use track_summary.csv for phenotype clustering analysis")


if __name__ == '__main__':
    main()
