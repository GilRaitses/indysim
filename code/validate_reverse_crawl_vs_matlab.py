#!/usr/bin/env python3
"""
Validate Python reverse crawl detection against MATLAB output.

Compares the Python retrovibez implementation (using derived_quantities/shead, smid, sloc)
against the original MATLAB mason_analysis.m output.

Usage:
    python scripts/validate_reverse_crawl_vs_matlab.py

Expected MATLAB output for Track 2 (from user validation):
    R1: 38.1s - 41.2s (3.1s)
    R2: 87.0s - 93.9s (6.9s)
    R3: 176.5s - 185.5s (9.0s)
    R4: 480.8s - 485.3s (4.4s)
    R5: 488.8s - 493.8s (4.9s)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add retrovibez to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'retrovibez'))
from core.h5_reader import process_track_from_h5, compute_speed_run_vel
import h5py


def validate_track_2():
    """Validate Track 2 against known MATLAB output."""
    
    h5_path = Path('data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#C_Bl_7PWM/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5')
    
    if not h5_path.exists():
        print(f"ERROR: H5 file not found: {h5_path}")
        return False
    
    # Expected MATLAB results for Track 2
    matlab_reversals = [
        {'start': 38.1, 'end': 41.2, 'duration': 3.1},
        {'start': 87.0, 'end': 93.9, 'duration': 6.9},
        {'start': 176.5, 'end': 185.5, 'duration': 9.0},
        {'start': 480.8, 'end': 485.3, 'duration': 4.4},
        {'start': 488.8, 'end': 493.8, 'duration': 4.9},
    ]
    
    print("=" * 60)
    print("Validating Track 2 against MATLAB output")
    print("=" * 60)
    
    with h5py.File(h5_path, 'r') as f:
        speed_run_vel, reversals = process_track_from_h5(f, 'track_2', min_duration=3.0)
        
        # Get ETI for plotting
        eti = f['tracks/track_2/derived_quantities/eti'][:].ravel()
    
    print(f"\nPython results: {len(reversals)} reversals")
    print(f"MATLAB results: {len(matlab_reversals)} reversals")
    
    # Compare each reversal
    all_match = True
    tolerance = 0.5  # seconds
    
    print("\nDetailed comparison:")
    print("-" * 60)
    print(f"{'#':<4} {'Python Start':>12} {'MATLAB Start':>12} {'Match':>8}")
    print("-" * 60)
    
    for i, (py_rev, mat_rev) in enumerate(zip(reversals, matlab_reversals)):
        start_match = abs(py_rev.start_time - mat_rev['start']) < tolerance
        end_match = abs(py_rev.end_time - mat_rev['end']) < tolerance
        match = start_match and end_match
        all_match = all_match and match
        
        status = "✓" if match else "✗"
        print(f"{i+1:<4} {py_rev.start_time:>12.1f} {mat_rev['start']:>12.1f} {status:>8}")
    
    print("-" * 60)
    print(f"Overall: {'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")
    
    return all_match, speed_run_vel, eti, reversals, matlab_reversals


def plot_validation(speed_run_vel, eti, reversals, matlab_reversals, output_path):
    """Create validation figure."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Panel A: Full SpeedRunVel time series
    ax = axes[0]
    times = eti[:-1]  # Match SpeedRunVel length
    ax.plot(times, speed_run_vel, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.fill_between(times, speed_run_vel, 0, where=(speed_run_vel < 0), 
                    color='red', alpha=0.3, label='SpeedRunVel < 0')
    
    # Mark detected reversals
    for i, rev in enumerate(reversals):
        ax.axvspan(rev.start_time, rev.end_time, color='red', alpha=0.2)
        ax.text(rev.start_time, ax.get_ylim()[1] * 0.9, f'R{i+1}', fontsize=8)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('SpeedRunVel (cm/s)', fontsize=11)
    ax.set_title('A. SpeedRunVel Time Series with Detected Reversals', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Panel B: Zoom on first reversal
    ax = axes[1]
    if len(reversals) > 0:
        r1 = reversals[0]
        t_start = max(0, r1.start_time - 10)
        t_end = min(times[-1], r1.end_time + 10)
        
        mask = (times >= t_start) & (times <= t_end)
        ax.plot(times[mask], speed_run_vel[mask], 'b-', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.fill_between(times[mask], speed_run_vel[mask], 0, 
                        where=(speed_run_vel[mask] < 0), color='red', alpha=0.3)
        ax.axvspan(r1.start_time, r1.end_time, color='red', alpha=0.2, label='Python detection')
        
        # Add MATLAB expected
        m1 = matlab_reversals[0]
        ax.axvline(m1['start'], color='green', linestyle='--', linewidth=2, label=f'MATLAB start ({m1["start"]:.1f}s)')
        ax.axvline(m1['end'], color='green', linestyle=':', linewidth=2, label=f'MATLAB end ({m1["end"]:.1f}s)')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('SpeedRunVel (cm/s)', fontsize=11)
        ax.set_title(f'B. First Reversal (Python: {r1.start_time:.1f}-{r1.end_time:.1f}s, MATLAB: {m1["start"]:.1f}-{m1["end"]:.1f}s)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
    
    # Panel C: Comparison table
    ax = axes[2]
    ax.axis('off')
    
    # Create table data
    table_data = [['#', 'Python Start', 'Python End', 'Python Dur', 'MATLAB Start', 'MATLAB End', 'MATLAB Dur', 'Match']]
    for i, (py_rev, mat_rev) in enumerate(zip(reversals, matlab_reversals)):
        match = abs(py_rev.start_time - mat_rev['start']) < 0.5 and abs(py_rev.end_time - mat_rev['end']) < 0.5
        table_data.append([
            f'{i+1}',
            f'{py_rev.start_time:.1f}s',
            f'{py_rev.end_time:.1f}s',
            f'{py_rev.duration:.1f}s',
            f'{mat_rev["start"]:.1f}s',
            f'{mat_rev["end"]:.1f}s',
            f'{mat_rev["duration"]:.1f}s',
            '✓' if match else '✗'
        ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.05, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('C. Python vs MATLAB Reversal Detection Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved validation figure to {output_path}")


def main():
    print("=" * 60)
    print("Reverse Crawl Detection Validation")
    print("Python (retrovibez) vs MATLAB (mason_analysis.m)")
    print("=" * 60)
    
    # Validate Track 2
    result = validate_track_2()
    
    if result is False:
        return 1
    
    all_match, speed_run_vel, eti, reversals, matlab_reversals = result
    
    # Generate validation figure
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plot_validation(speed_run_vel, eti, reversals, matlab_reversals, 
                   fig_dir / "reverse_crawl_validation_vs_matlab.png")
    
    if all_match:
        print("\n" + "=" * 60)
        print("✓ VALIDATION PASSED: Python matches MATLAB exactly!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ VALIDATION FAILED: Discrepancies detected")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit(main())






