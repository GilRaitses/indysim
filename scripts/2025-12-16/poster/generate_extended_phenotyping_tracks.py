#!/usr/bin/env python3
"""
Generate Extended Phenotyping Dataset

Generates more complete tracks (200-300 per condition) for better phenotype discovery.
Higher sample size = more robust clustering and better phenotype identification.

This script wraps the existing generate_simulated_tracks_for_phenotyping.py
to generate more tracks per condition.
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Add InDySim code directory to path
INDYSIM_CODE = Path('/Users/gilraitses/InDySim/code')
if INDYSIM_CODE.exists() and str(INDYSIM_CODE) not in sys.path:
    sys.path.insert(0, str(INDYSIM_CODE))

# Import the existing generation function
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the generation module
import importlib.util
gen_script_path = SCRIPT_DIR / 'generate_simulated_tracks_for_phenotyping.py'
spec = importlib.util.spec_from_file_location("gen_tracks", gen_script_path)
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)


def main():
    parser = argparse.ArgumentParser(
        description='Generate extended phenotyping dataset with more tracks per condition'
    )
    parser.add_argument('--n-tracks-per-condition', type=int, default=250,
                       help='Number of tracks per condition (default: 250)')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/gilraitses/InDySim/data/simulated_phenotyping_extended',
                       help='Output directory')
    parser.add_argument('--duration', type=float, default=1200.0,
                       help='Track duration in seconds (default: 1200 = 20 min)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    n_per_condition = args.n_tracks_per_condition
    total_tracks = n_per_condition * 4  # 4 conditions
    
    print("=" * 80)
    print("EXTENDED PHENOTYPING DATASET GENERATION")
    print("=" * 80)
    print(f"\nGenerating {total_tracks} complete tracks:")
    print(f"  - {n_per_condition} tracks per condition")
    print(f"  - 4 conditions (0-250 Constant, 0-250 Cycling, 50-250 Constant, 50-250 Cycling)")
    print(f"  - Duration: {args.duration/60:.1f} minutes per track")
    print(f"\nOutput: {output_dir}")
    print(f"\nEstimated time: ~{total_tracks * 0.5 / 60:.1f} minutes")
    print("\nStarting generation...\n")
    
    start_time = datetime.now()
    
    # Use the existing function
    summary = gen_module.generate_multi_condition_dataset(
        n_tracks_per_condition=n_per_condition,
        duration=args.duration,
        phenotype_variation=True,
        seed=args.seed,
        output_dir=output_dir
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal tracks: {len(summary)}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"\nSummary saved to: {output_dir}/all_tracks_summary.csv")
    print(f"\nNext steps:")
    print(f"  1. Run phenotyping analysis on extended dataset")
    print(f"  2. Compare results to original 300-track dataset")
    print(f"  3. Assess if more tracks improve phenotype discovery")

if __name__ == '__main__':
    main()

