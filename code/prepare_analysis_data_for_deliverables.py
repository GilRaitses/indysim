#!/usr/bin/env python3
"""
Prepare Analysis Data for Deliverables

Copies analysis results and figures needed for manuscript deliverables.
This script prepares the data structure for generating figures without
including the original experimental data.

Usage:
    python scripts/prepare_analysis_data_for_deliverables.py --output deliverables/mechanosensory_manuscript_2025
"""

import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict

# Essential model results files needed for figure generation
ESSENTIAL_MODEL_FILES = [
    'factorial_model_results.json',
    'kernel_bootstrap_ci.json',
    'per_condition_timescales.json',
    'turn_distributions.json',
    'reverse_crawl_modulation.json',
    'model_comparison_full.json',
    'kernel_form_comparison.json',
    'event_duration_summary.json',
    'validation_results.json',
    'cv_results.json',
]

# Essential figure files
ESSENTIAL_FIGURES = [
    'figure1_kernel.png',
    'figure2_validation.png',
    'figure3_trajectories.png',
    'figure5_factorial.png',
    'psth_comparison.png',
    'timescale_variability.png',
    'reverse_crawl_led_modulation.png',
    'event_durations.png',
    'turn_distributions.png',
]

# Diagnostic figures
DIAGNOSTIC_FIGURES = [
    'factorial_diagnostics/factorial_diagnostics.png',
    'factorial_diagnostics/time_rescaling.png',
]


def copy_model_results(source_dir: Path, dest_dir: Path) -> List[str]:
    """Copy essential model results files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    
    for filename in ESSENTIAL_MODEL_FILES:
        source = source_dir / filename
        if source.exists():
            dest = dest_dir / filename
            shutil.copy2(source, dest)
            copied.append(filename)
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: {filename} not found")
    
    return copied


def copy_figures(source_dir: Path, dest_dir: Path) -> List[str]:
    """Copy essential figure files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    
    # Copy main figures
    for filename in ESSENTIAL_FIGURES:
        source = source_dir / filename
        if source.exists():
            dest = dest_dir / filename
            shutil.copy2(source, dest)
            copied.append(filename)
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: {filename} not found")
    
    # Copy diagnostic figures
    for rel_path in DIAGNOSTIC_FIGURES:
        source = source_dir / rel_path
        if source.exists():
            dest = dest_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(rel_path)
            print(f"  Copied: {rel_path}")
    
    return copied


def create_manifest(output_dir: Path, model_files: List[str], figure_files: List[str]):
    """Create a manifest file listing what's included."""
    manifest = {
        'analysis_results': {
            'count': len(model_files),
            'files': sorted(model_files)
        },
        'figures': {
            'count': len(figure_files),
            'files': sorted(figure_files)
        },
        'note': 'This deliverable contains analysis outputs only. Original experimental data are available upon request (contact: mmihovil@syr.edu).'
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  Created: manifest.json")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare analysis data for deliverables'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('deliverables/mechanosensory_manuscript_2025'),
        help='Output directory for deliverables'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('data/model'),
        help='Source directory for model results'
    )
    parser.add_argument(
        '--figures-dir',
        type=Path,
        default=Path('figures'),
        help='Source directory for figures'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preparing Analysis Data for Deliverables")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Model source: {args.model_dir}")
    print(f"Figures source: {args.figures_dir}")
    print()
    
    # Create output structure
    analysis_results_dir = args.output / 'analysis_results'
    figures_dir = args.output / 'figures'
    
    # Copy model results
    print("Copying model results...")
    model_files = copy_model_results(args.model_dir, analysis_results_dir)
    print(f"  Copied {len(model_files)} model result files\n")
    
    # Copy figures
    print("Copying figures...")
    figure_files = copy_figures(args.figures_dir, figures_dir)
    print(f"  Copied {len(figure_files)} figure files\n")
    
    # Create manifest
    print("Creating manifest...")
    create_manifest(args.output, model_files, figure_files)
    
    print()
    print("=" * 60)
    print("SUCCESS: Analysis data prepared for deliverables")
    print("=" * 60)
    print(f"Location: {args.output}")
    print(f"  - Analysis results: {len(model_files)} files")
    print(f"  - Figures: {len(figure_files)} files")
    print()
    print("Note: Original experimental data are NOT included.")
    print("      Data available upon request: mmihovil@syr.edu")
    print("=" * 60)


if __name__ == '__main__':
    main()

