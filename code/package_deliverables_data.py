#!/usr/bin/env python3
"""
Package Analysis Data for Deliverables

Copies analysis results and figures into a data/ folder structure
and distributes it to all three deliverable locations:
- deliverables/Raitses_et_al_2025/data/
- deliverables/mechanosensory_manuscript_2025/data/
- deliverables/sensorimotor-habituation-model/data/

Usage:
    python scripts/package_deliverables_data.py
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

# Deliverable locations
DELIVERABLE_LOCATIONS = [
    Path('deliverables/Raitses_et_al_2025'),
    Path('deliverables/mechanosensory_manuscript_2025'),
    Path('deliverables/sensorimotor-habituation-model'),
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


def copy_analysis_script(source_dir: Path, dest_dir: Path):
    """Copy the end-to-end analysis script."""
    script_source = source_dir / 'run_analysis_pipeline.py'
    script_dest = dest_dir / 'run_analysis_pipeline.py'
    
    if script_source.exists():
        shutil.copy2(script_source, script_dest)
        # Make executable
        script_dest.chmod(0o755)
        print(f"  Copied: run_analysis_pipeline.py")
        return True
    else:
        print(f"  Warning: run_analysis_pipeline.py not found")
        return False


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


def package_to_deliverable(base_dir: Path, deliverable_path: Path):
    """Package data folder to a specific deliverable location."""
    data_dir = deliverable_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Packaging to: {deliverable_path.name}")
    print(f"{'='*60}")
    
    # Copy model results
    model_dir = data_dir / 'analysis_results'
    print("\nCopying model results...")
    model_files = copy_model_results(base_dir / 'data' / 'model', model_dir)
    
    # Copy figures
    figures_dir = data_dir / 'figures'
    print("\nCopying figures...")
    figure_files = copy_figures(base_dir / 'figures', figures_dir)
    
    # Copy analysis script
    code_dir = data_dir / 'code'
    code_dir.mkdir(parents=True, exist_ok=True)
    print("\nCopying analysis script...")
    copy_analysis_script(base_dir / 'scripts', code_dir)
    
    # Create manifest
    print("\nCreating manifest...")
    create_manifest(data_dir, model_files, figure_files)
    
    return len(model_files), len(figure_files)


def main():
    base_dir = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Packaging Analysis Data for Deliverables")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Deliverable locations: {len(DELIVERABLE_LOCATIONS)}")
    print()
    
    # Verify source directories exist
    model_dir = base_dir / 'data' / 'model'
    figures_dir = base_dir / 'figures'
    scripts_dir = base_dir / 'scripts'
    
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1
    
    if not figures_dir.exists():
        print(f"ERROR: Figures directory not found: {figures_dir}")
        return 1
    
    if not (scripts_dir / 'run_analysis_pipeline.py').exists():
        print(f"ERROR: Analysis script not found: {scripts_dir / 'run_analysis_pipeline.py'}")
        return 1
    
    # Package to each deliverable location
    results = []
    for deliverable_path in DELIVERABLE_LOCATIONS:
        if not deliverable_path.exists():
            print(f"Warning: Deliverable location not found: {deliverable_path}")
            continue
        
        n_models, n_figures = package_to_deliverable(base_dir, deliverable_path)
        results.append((deliverable_path.name, n_models, n_figures))
    
    # Summary
    print()
    print("=" * 60)
    print("SUCCESS: Analysis data packaged to all deliverables")
    print("=" * 60)
    for name, n_models, n_figures in results:
        print(f"  {name}: {n_models} model files, {n_figures} figures")
    print()
    print("Note: Original experimental data are NOT included.")
    print("      Data available upon request: mmihovil@syr.edu")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

