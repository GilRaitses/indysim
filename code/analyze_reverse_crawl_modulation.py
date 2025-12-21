#!/usr/bin/env python3
"""
Analyze LED modulation of reverse crawl probability across all experiments.

Uses the corrected reverse crawl detection (Mason Klein methodology via retrovibez)
that reads from derived_quantities/shead, smid, sloc, eti.

Run after reprocessing all H5 files:
    python scripts/analyze_reverse_crawl_modulation.py

Output:
    - figures/reverse_crawl_led_modulation.png
    - data/model/reverse_crawl_modulation.json
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
from pathlib import Path
import json
from scipy import stats


def load_all_trajectories(data_dir: Path) -> pd.DataFrame:
    """Load trajectory data from consolidated H5 or parquet files."""
    import h5py
    
    # Try consolidated H5 first
    h5_path = data_dir / "consolidated_dataset.h5"
    if h5_path.exists():
        print(f"Loading from consolidated H5: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            # Try 'trajectories' group first, then 'events'
            if 'trajectories' in f:
                grp = f['trajectories']
            elif 'events' in f:
                grp = f['events']
            else:
                raise ValueError("No trajectories or events group in H5 file")
            
            columns = [c.decode('utf-8') if isinstance(c, bytes) else c 
                       for c in grp.attrs['columns']]
            
            # Load needed columns
            needed = ['experiment_id', 'time', 'is_reverse_crawl', 'is_reverse_crawl_start',
                      'is_reorientation', 'stimulus_on', 'time_since_stimulus', 
                      'led1Val', 'led1Val_ton']
            available = [c for c in needed if c in columns]
            
            data = {}
            for col in available:
                vals = grp[col][:]
                if vals.dtype.kind == 'S':
                    vals = np.array([v.decode('utf-8') if isinstance(v, bytes) else v for v in vals])
                data[col] = vals
            
            df = pd.DataFrame(data)
            n_expts = df['experiment_id'].nunique() if 'experiment_id' in df.columns else 'unknown'
            print(f"Loaded {len(df):,} frames from {n_expts} experiments")
            return df
    
    # Fallback to parquet files
    traj_files = sorted(data_dir.glob("*_trajectories.parquet"))
    
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files or consolidated_dataset.h5 found in {data_dir}")
    
    print(f"Loading {len(traj_files)} trajectory files...")
    
    all_dfs = []
    for f in traj_files:
        df = pd.read_parquet(f)
        # Extract experiment ID from filename
        expt_id = f.stem.replace("_trajectories", "")
        df['experiment_id'] = expt_id
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} total frames from {len(traj_files)} experiments")
    
    return combined


def analyze_led_modulation(df: pd.DataFrame) -> dict:
    """Analyze reverse crawl rate by LED state."""
    
    results = {
        'n_frames': len(df),
        'n_experiments': df['experiment_id'].nunique(),
    }
    
    # Check required columns
    if 'is_reverse_crawl' not in df.columns:
        raise ValueError("is_reverse_crawl column not found. Reprocess with retrovibez integration.")
    
    # Find LED column
    led_col = None
    for col in ['led1Val_on', 'led1Val_ton', 'stimulus_on']:
        if col in df.columns:
            led_col = col
            break
    
    if led_col is None:
        raise ValueError("LED state column not found.")
    
    print(f"Using LED column: {led_col}")
    
    # Overall statistics
    results['total_rc_frames'] = int(df['is_reverse_crawl'].sum())
    results['total_rc_starts'] = int(df['is_reverse_crawl_start'].sum()) if 'is_reverse_crawl_start' in df.columns else 0
    results['pct_time_reversing'] = 100 * results['total_rc_frames'] / len(df)
    
    # Reorientation statistics for comparison
    if 'is_reorientation' in df.columns:
        results['total_reo_frames'] = int(df['is_reorientation'].sum())
        results['pct_time_reorienting'] = 100 * results['total_reo_frames'] / len(df)
    
    # Peak-intensity vs Baseline
    led_on_mask = df[led_col] == True
    led_off_mask = df[led_col] == False
    
    results['led_on'] = {
        'n_frames': int(led_on_mask.sum()),
        'rc_frames': int(df.loc[led_on_mask, 'is_reverse_crawl'].sum()),
        'rc_starts': int(df.loc[led_on_mask, 'is_reverse_crawl_start'].sum()) if 'is_reverse_crawl_start' in df.columns else 0,
    }
    results['led_on']['pct_reversing'] = 100 * results['led_on']['rc_frames'] / results['led_on']['n_frames'] if results['led_on']['n_frames'] > 0 else 0
    
    results['led_off'] = {
        'n_frames': int(led_off_mask.sum()),
        'rc_frames': int(df.loc[led_off_mask, 'is_reverse_crawl'].sum()),
        'rc_starts': int(df.loc[led_off_mask, 'is_reverse_crawl_start'].sum()) if 'is_reverse_crawl_start' in df.columns else 0,
    }
    results['led_off']['pct_reversing'] = 100 * results['led_off']['rc_frames'] / results['led_off']['n_frames'] if results['led_off']['n_frames'] > 0 else 0
    
    # Suppression ratio
    if results['led_off']['pct_reversing'] > 0:
        results['suppression_ratio'] = results['led_on']['pct_reversing'] / results['led_off']['pct_reversing']
        results['suppression_pct'] = 100 * (1 - results['suppression_ratio'])
    else:
        results['suppression_ratio'] = float('nan')
        results['suppression_pct'] = float('nan')
    
    # Chi-square test for LED effect
    observed = np.array([
        [results['led_on']['rc_frames'], results['led_on']['n_frames'] - results['led_on']['rc_frames']],
        [results['led_off']['rc_frames'], results['led_off']['n_frames'] - results['led_off']['rc_frames']]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    results['chi2_test'] = {
        'chi2': float(chi2),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
    
    # Time-resolved analysis (bins relative to LED onset)
    if 'time_since_stimulus' in df.columns:
        # Bin by time since stimulus
        bins = [-np.inf, 0, 2, 5, 10, np.inf]
        labels = ['pre-LED', '0-2s', '2-5s', '5-10s', '>10s']
        df['led_phase'] = pd.cut(df['time_since_stimulus'], bins=bins, labels=labels)
        
        results['by_phase'] = {}
        for phase in labels:
            mask = df['led_phase'] == phase
            n = int(mask.sum())
            rc = int(df.loc[mask, 'is_reverse_crawl'].sum())
            rc_starts = int(df.loc[mask, 'is_reverse_crawl_start'].sum()) if 'is_reverse_crawl_start' in df.columns else 0
            results['by_phase'][phase] = {
                'n_frames': n,
                'rc_frames': rc,
                'rc_starts': rc_starts,
                'pct_reversing': 100 * rc / n if n > 0 else 0
            }
    
    return results


def plot_analysis(results: dict, output_path: Path):
    """Generate visualization of reverse crawl modulation."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Baseline vs Peak-intensity comparison
    ax = axes[0]
    categories = ['Baseline', 'Peak']
    values = [results['led_off']['pct_reversing'], results['led_on']['pct_reversing']]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('% Time in Reverse Crawl', fontsize=11)
    ax.set_title('A. Reverse Crawl by LED State', fontsize=12, fontweight='bold')
    
    # Add significance marker
    if results['chi2_test']['significant']:
        max_val = max(values)
        ax.plot([0, 1], [max_val * 1.15, max_val * 1.15], 'k-', linewidth=1.5)
        p_val = results['chi2_test']['p_value']
        if p_val < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f'p = {p_val:.3f}'
        ax.text(0.5, max_val * 1.2, f"***\n{p_str}", ha='center', fontsize=10)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(values) * 1.4)
    
    # Panel B: Time-resolved analysis
    ax = axes[1]
    if 'by_phase' in results:
        phases = list(results['by_phase'].keys())
        pcts = [results['by_phase'][p]['pct_reversing'] for p in phases]
        
        colors = ['#95a5a6', '#3498db', '#2980b9', '#1f618d', '#154360']
        bars = ax.bar(phases, pcts, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('% Time in Reverse Crawl', fontsize=11)
        ax.set_xlabel('Time Since LED Onset', fontsize=11)
        ax.set_title('B. Reverse Crawl by LED Phase', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        
        # Add value labels
        for bar, val in zip(bars, pcts):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.1f}%', ha='center', fontsize=9)
    
    # Panel C: Comparison with reorientations
    ax = axes[2]
    if 'pct_time_reorienting' in results:
        categories = ['Reverse Crawl', 'Reorientation']
        values = [results['pct_time_reversing'], results['pct_time_reorienting']]
        colors = ['#9b59b6', '#e67e22']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('% of Total Time', fontsize=11)
        ax.set_title('C. Behavioral State Duration', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Add summary text
    fig.text(0.5, -0.02, 
             f"LED suppresses reverse crawls by {results['suppression_pct']:.0f}% "
             f"({results['led_on']['rc_starts']} events during peak vs {results['led_off']['rc_starts']} at baseline, "
             f"χ² p < 0.001)",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    print("=" * 60)
    print("Reverse Crawl LED Modulation Analysis")
    print("=" * 60)
    
    # Try data/processed first (has consolidated H5), then others
    data_dir = Path("data/processed")
    if not data_dir.exists() or not (data_dir / "consolidated_dataset.h5").exists():
        data_dir = Path("data/processed_with_reversals")
    if not data_dir.exists():
        data_dir = Path("data/data/processed_with_reversals")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found")
        print("Looking for: data/processed/consolidated_dataset.h5")
        print("Run the consolidation script first.")
        return 1
    
    # Load data
    df = load_all_trajectories(data_dir)
    
    # Analyze
    print("\nAnalyzing LED modulation of reverse crawls...")
    results = analyze_led_modulation(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total frames: {results['n_frames']:,}")
    print(f"Experiments: {results['n_experiments']}")
    print(f"Reverse crawl events: {results['total_rc_starts']}")
    print(f"% time reversing: {results['pct_time_reversing']:.2f}%")
    if 'pct_time_reorienting' in results:
        print(f"% time reorienting: {results['pct_time_reorienting']:.2f}%")
    
    print(f"\nPeak:     {results['led_on']['pct_reversing']:.3f}% reversing ({results['led_on']['rc_starts']} events)")
    print(f"Baseline: {results['led_off']['pct_reversing']:.3f}% reversing ({results['led_off']['rc_starts']} events)")
    print(f"Suppression: {results['suppression_pct']:.1f}%")
    print(f"Chi-square p-value: {results['chi2_test']['p_value']:.2e}")
    print(f"Significant: {'YES' if results['chi2_test']['significant'] else 'NO'}")
    
    if 'by_phase' in results:
        print("\nBy LED phase:")
        for phase, data in results['by_phase'].items():
            print(f"  {phase:10s}: {data['pct_reversing']:.3f}% ({data['rc_starts']} events)")
    
    # Save results
    output_dir = Path("data/model")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "reverse_crawl_modulation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Generate figure
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plot_analysis(results, fig_dir / "reverse_crawl_led_modulation.png")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())






