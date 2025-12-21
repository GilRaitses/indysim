#!/usr/bin/env python3
"""
Fit Gamma-Difference Kernel Per Condition

Fits the 6-parameter gamma-difference kernel separately to each of the 
4 factorial conditions to test whether timescales (τ₁, τ₂) are conserved.

For each condition:
1. Fit raised-cosine GLM to get kernel shape
2. Fit gamma-difference to the reconstructed kernel
3. Bootstrap (track-level) for CIs on τ₁, τ₂

Output:
- data/model/per_condition_timescales.json
- figures/timescale_comparison.png

Usage:
    python scripts/fit_gamma_per_condition.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available")


# ============================================================================
# Configuration
# ============================================================================

BO_OPTIMAL_CONFIG = {
    'early_centers': [0.2, 0.6333, 1.0667, 1.5],
    'early_width': 0.30,
    'intm_centers': [2.0, 2.5],
    'intm_width': 0.6,
    'late_centers': [3.0, 4.2, 5.4, 6.6, 7.8, 9.0],
    'late_width': 2.494,
    'rebound_tau': 2.0
}

LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
FIRST_LED_ONSET = 21.3

N_BOOTSTRAP = 200  # Track-level bootstrap iterations
RANDOM_SEED = 42


# ============================================================================
# Data Loading (from fit_cross_condition.py)
# ============================================================================

@dataclass
class ConditionData:
    """Container for condition-specific data."""
    name: str
    intensity: str
    background: str
    files: List[Path]
    data: Optional[pd.DataFrame] = None
    n_tracks: int = 0
    n_events: int = 0
    n_frames: int = 0
    n_experiments: int = 0


def get_condition_files(data_dir: str = "data/engineered") -> Dict[str, ConditionData]:
    """Get files for each condition in the 2×2 factorial design."""
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob('*_events.csv'))
    
    conditions = {}
    # C_Bl = Constant blue background, T_Bl_Sq = Temporal/cycling blue square wave
    # Use _0to250 (with underscore) to avoid matching 50to250 which contains "0to250" as substring
    condition_patterns = {
        '0→250 | Constant': ('_0to250PWM', '#C_Bl_7PWM'),
        '0→250 | Cycling': ('_0to250PWM', '#T_Bl_Sq_5to15PWM'),
        '50→250 | Constant': ('50to250PWM', '#C_Bl_7PWM'),
        '50→250 | Cycling': ('50to250PWM', '#T_Re_Sq'),  # Uses #T_Re_Sq not #T_Bl_Sq
    }
    
    for name, (intensity_pattern, bg_pattern) in condition_patterns.items():
        matching_files = [
            f for f in all_files
            if intensity_pattern in f.name and bg_pattern in f.name
        ]
        intensity = '0→250' if '0to250' in intensity_pattern else '50→250'
        background = 'Constant' if 'C_Bl' in bg_pattern else 'Cycling'
        
        conditions[name] = ConditionData(
            name=name,
            intensity=intensity,
            background=background,
            files=matching_files
        )
    
    return conditions


def load_condition_data(condition: ConditionData, exclude_anomalous: bool = False) -> pd.DataFrame:
    """Load data for a specific condition."""
    # No experiments excluded by default
    anomalous_files = []  # Was: ['202510291652', '202510291713']
    
    dfs = []
    files_used = 0
    for f in condition.files:
        if exclude_anomalous and any(a in f.name for a in anomalous_files):
            continue
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        # Create unique track ID to avoid collisions across experiments
        df['unique_track_id'] = df['experiment_id'] + '_' + df['track_id'].astype(str)
        dfs.append(df)
        files_used += 1
    
    if not dfs:
        return pd.DataFrame()
    
    data = pd.concat(dfs, ignore_index=True)
    condition.data = data
    condition.n_frames = len(data)
    # Use unique_track_id for correct count
    condition.n_tracks = data['unique_track_id'].nunique()
    condition.n_events = data['is_reorientation_start'].sum()
    condition.n_experiments = files_used
    
    return data


# ============================================================================
# Kernel Fitting Functions
# ============================================================================

def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """Compute raised-cosine basis functions."""
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def compute_time_since_led_onset(data: pd.DataFrame) -> np.ndarray:
    """Compute time since last LED onset."""
    times = data['time'].values
    time_since_onset = np.full(len(data), -1.0)
    
    for i, t in enumerate(times):
        if t >= FIRST_LED_ONSET:
            cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                time_since_onset[i] = cycle_time
    
    return time_since_onset


def build_design_matrix(data: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build design matrix for GLM."""
    time_since_onset = compute_time_since_led_onset(data)
    
    early_basis = raised_cosine_basis(
        time_since_onset, 
        np.array(config['early_centers']), 
        config['early_width']
    )
    intm_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['intm_centers']),
        config['intm_width']
    )
    late_basis = raised_cosine_basis(
        time_since_onset,
        np.array(config['late_centers']),
        config['late_width']
    )
    
    X = np.column_stack([
        np.ones(len(data)),
        early_basis,
        intm_basis,
        late_basis
    ])
    
    y = data['is_reorientation_start'].fillna(0).values.astype(int)
    track_ids = data['track_id'].values
    
    return X, y, track_ids


def fit_glm_kernel(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit GLM and return kernel coefficients."""
    model = GLM(y, X, family=NegativeBinomial(alpha=alpha))
    try:
        result = model.fit(disp=False)
        return result.params[1:]  # Exclude intercept
    except Exception as e:
        print(f"  GLM fit failed: {e}")
        return None


def reconstruct_kernel(coeffs: np.ndarray, config: dict, t: np.ndarray) -> np.ndarray:
    """Reconstruct kernel from basis coefficients."""
    early_basis = raised_cosine_basis(t, np.array(config['early_centers']), config['early_width'])
    intm_basis = raised_cosine_basis(t, np.array(config['intm_centers']), config['intm_width'])
    late_basis = raised_cosine_basis(t, np.array(config['late_centers']), config['late_width'])
    
    basis = np.column_stack([early_basis, intm_basis, late_basis])
    return basis @ coeffs


def gamma_diff_kernel(t: np.ndarray, A: float, a1: float, b1: float,
                      B: float, a2: float, b2: float) -> np.ndarray:
    """Gamma-difference kernel."""
    pdf1 = gamma_dist.pdf(t, a1, scale=b1)
    pdf2 = gamma_dist.pdf(t, a2, scale=b2)
    pdf1 = np.nan_to_num(pdf1, nan=0.0)
    pdf2 = np.nan_to_num(pdf2, nan=0.0)
    return A * pdf1 - B * pdf2


def fit_gamma_difference(t: np.ndarray, K: np.ndarray) -> Dict:
    """Fit gamma-difference to kernel curve."""
    # Initial guesses and bounds
    p0 = [0.5, 2.2, 0.13, 12.0, 4.4, 0.87]  # A, a1, b1, B, a2, b2
    bounds = (
        [0.01, 1.0, 0.05, 0.1, 2.0, 0.3],   # Lower
        [5.0, 5.0, 0.5, 50.0, 8.0, 2.0]      # Upper
    )
    
    try:
        popt, pcov = curve_fit(gamma_diff_kernel, t, K, p0=p0, bounds=bounds, maxfev=10000)
        A, a1, b1, B, a2, b2 = popt
        
        # Compute timescales
        tau1 = a1 * b1
        tau2 = a2 * b2
        
        # Compute fit quality
        K_fit = gamma_diff_kernel(t, *popt)
        ss_res = np.sum((K - K_fit) ** 2)
        ss_tot = np.sum((K - np.mean(K)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'A': float(A),
            'alpha1': float(a1),
            'beta1': float(b1),
            'B': float(B),
            'alpha2': float(a2),
            'beta2': float(b2),
            'tau1': float(tau1),
            'tau2': float(tau2),
            'r_squared': float(r_squared),
            'converged': True
        }
    except Exception as e:
        return {'error': str(e), 'converged': False}


# ============================================================================
# Bootstrap for CIs
# ============================================================================

def bootstrap_gamma_fit(data: pd.DataFrame, config: dict, n_bootstrap: int = 200, 
                        seed: int = 42) -> Dict:
    """
    Track-level bootstrap for gamma-difference parameter CIs.
    
    Resamples tracks (not frames) to respect temporal autocorrelation.
    Uses unique_track_id to handle track ID collisions across experiments.
    """
    np.random.seed(seed)
    
    t_grid = np.linspace(0.01, 10, 500)
    # Use unique_track_id to properly distinguish tracks across experiments
    track_col = 'unique_track_id' if 'unique_track_id' in data.columns else 'track_id'
    track_ids = data[track_col].unique()
    n_tracks = len(track_ids)
    
    bootstrap_params = {
        'tau1': [], 'tau2': [],
        'alpha1': [], 'alpha2': [],
        'beta1': [], 'beta2': [],
        'A': [], 'B': [], 'r_squared': []
    }
    
    print(f"  Running {n_bootstrap} track-level bootstrap iterations ({n_tracks} tracks)...")
    
    import sys
    n_failed = 0
    
    for i in range(n_bootstrap):
        # Progress bar update every 5 iterations
        if i % 5 == 0 or i == n_bootstrap - 1:
            pct = (i + 1) / n_bootstrap * 100
            bar_width = 30
            filled = int(bar_width * (i + 1) / n_bootstrap)
            bar = '█' * filled + '░' * (bar_width - filled)
            sys.stdout.write(f"\r    [{bar}] {i+1:3d}/{n_bootstrap} ({pct:5.1f}%) | failed: {n_failed}")
            sys.stdout.flush()
        
        # Resample tracks with replacement
        sampled_tracks = np.random.choice(track_ids, size=n_tracks, replace=True)
        
        # Build bootstrap sample
        boot_dfs = []
        for track in sampled_tracks:
            boot_dfs.append(data[data[track_col] == track].copy())
        
        if not boot_dfs:
            n_failed += 1
            continue
            
        boot_data = pd.concat(boot_dfs, ignore_index=True)
        
        # Fit GLM
        X, y, _ = build_design_matrix(boot_data, config)
        coeffs = fit_glm_kernel(X, y)
        
        if coeffs is None:
            n_failed += 1
            continue
        
        # Reconstruct and fit gamma
        K = reconstruct_kernel(coeffs, config, t_grid)
        gamma_result = fit_gamma_difference(t_grid, K)
        
        if gamma_result.get('converged', False):
            for key in bootstrap_params:
                if key in gamma_result:
                    bootstrap_params[key].append(gamma_result[key])
        else:
            n_failed += 1
    
    # Newline after progress bar
    print()
    n_success = n_bootstrap - n_failed
    print(f"    Bootstrap complete: {n_success}/{n_bootstrap} successful")
    
    # Compute CIs and save raw samples for permutation tests
    results = {}
    for key, values in bootstrap_params.items():
        if len(values) >= 10:
            values = np.array(values)
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5)),
                'n_valid': len(values),
                'samples': values.tolist()  # Save raw samples for permutation tests
            }
    
    return results


# ============================================================================
# Main Analysis
# ============================================================================

def fit_condition_gamma(condition: ConditionData, config: dict, 
                        do_bootstrap: bool = True) -> Dict:
    """Fit gamma-difference kernel to a single condition."""
    print(f"\n{'='*60}")
    print(f"Fitting: {condition.name}")
    print(f"  Experiments: {condition.n_experiments}, Tracks: {condition.n_tracks}, Events: {condition.n_events}")
    print(f"{'='*60}")
    
    data = condition.data
    if data is None or len(data) == 0:
        return {'error': 'No data'}
    
    # Fit GLM to get kernel
    X, y, track_ids = build_design_matrix(data, config)
    coeffs = fit_glm_kernel(X, y)
    
    if coeffs is None:
        return {'error': 'GLM fit failed'}
    
    # Reconstruct kernel on dense grid
    t_grid = np.linspace(0.01, 10, 500)
    K = reconstruct_kernel(coeffs, config, t_grid)
    
    # Fit gamma-difference
    gamma_result = fit_gamma_difference(t_grid, K)
    
    if not gamma_result.get('converged', False):
        return gamma_result
    
    print(f"  Gamma-difference fit: R² = {gamma_result['r_squared']:.4f}")
    print(f"  τ₁ (fast) = {gamma_result['tau1']:.3f} s")
    print(f"  τ₂ (slow) = {gamma_result['tau2']:.3f} s")
    
    # Bootstrap for CIs
    if do_bootstrap:
        bootstrap_results = bootstrap_gamma_fit(data, config, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED)
        gamma_result['bootstrap'] = bootstrap_results
        
        if 'tau1' in bootstrap_results:
            tau1_ci = bootstrap_results['tau1']
            print(f"  τ₁ 95% CI: [{tau1_ci['ci_lower']:.3f}, {tau1_ci['ci_upper']:.3f}]")
        if 'tau2' in bootstrap_results:
            tau2_ci = bootstrap_results['tau2']
            print(f"  τ₂ 95% CI: [{tau2_ci['ci_lower']:.3f}, {tau2_ci['ci_upper']:.3f}]")
    
    # Store kernel for plotting
    gamma_result['kernel_t'] = t_grid.tolist()
    gamma_result['kernel_K'] = K.tolist()
    gamma_result['kernel_K_fit'] = gamma_diff_kernel(t_grid, 
        gamma_result['A'], gamma_result['alpha1'], gamma_result['beta1'],
        gamma_result['B'], gamma_result['alpha2'], gamma_result['beta2']
    ).tolist()
    
    return gamma_result


def plot_timescale_comparison(results: Dict[str, Dict], output_path: Path):
    """Create forest plot comparing τ₁, τ₂ across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = list(results.keys())
    y_pos = np.arange(len(conditions))
    
    # Plot τ₁
    ax = axes[0]
    tau1_means = []
    tau1_errors = []
    
    for cond in conditions:
        r = results[cond]
        if 'bootstrap' in r and 'tau1' in r['bootstrap']:
            boot = r['bootstrap']['tau1']
            tau1_means.append(boot['mean'])
            tau1_errors.append([[boot['mean'] - boot['ci_lower']], 
                               [boot['ci_upper'] - boot['mean']]])
        else:
            tau1_means.append(r.get('tau1', np.nan))
            tau1_errors.append([[0], [0]])
    
    tau1_errors = np.array(tau1_errors).reshape(len(conditions), 2).T
    
    ax.errorbar(tau1_means, y_pos, xerr=tau1_errors, fmt='o', capsize=5, 
                markersize=8, color='tab:blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions)
    ax.set_xlabel('τ₁ (fast timescale, s)')
    ax.set_title('Fast Component Timescale')
    ax.axvline(np.mean(tau1_means), color='gray', linestyle='--', alpha=0.5, 
               label=f'Mean: {np.mean(tau1_means):.3f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot τ₂
    ax = axes[1]
    tau2_means = []
    tau2_errors = []
    
    for cond in conditions:
        r = results[cond]
        if 'bootstrap' in r and 'tau2' in r['bootstrap']:
            boot = r['bootstrap']['tau2']
            tau2_means.append(boot['mean'])
            tau2_errors.append([[boot['mean'] - boot['ci_lower']], 
                               [boot['ci_upper'] - boot['mean']]])
        else:
            tau2_means.append(r.get('tau2', np.nan))
            tau2_errors.append([[0], [0]])
    
    tau2_errors = np.array(tau2_errors).reshape(len(conditions), 2).T
    
    ax.errorbar(tau2_means, y_pos, xerr=tau2_errors, fmt='o', capsize=5,
                markersize=8, color='tab:red')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions)
    ax.set_xlabel('τ₂ (slow timescale, s)')
    ax.set_title('Slow Component Timescale')
    ax.axvline(np.mean(tau2_means), color='gray', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(tau2_means):.3f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved timescale comparison plot to {output_path}")


def plot_kernel_comparison(results: Dict[str, Dict], output_path: Path):
    """Plot all condition kernels overlaid."""
    from scipy.stats import gamma as gamma_dist
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    t = np.linspace(0.01, 10, 200)
    
    for i, (cond, r) in enumerate(results.items()):
        # Compute kernel from saved parameters
        A = r.get('A', 0)
        alpha1 = r.get('alpha1', 2)
        beta1 = r.get('beta1', 0.1)
        B = r.get('B', 0)
        alpha2 = r.get('alpha2', 4)
        beta2 = r.get('beta2', 1)
        
        # Gamma-difference kernel
        K_fit = A * gamma_dist.pdf(t, alpha1, scale=beta1) - B * gamma_dist.pdf(t, alpha2, scale=beta2)
        
        ax.plot(t, K_fit, color=colors[i], linewidth=2, label=f"{cond} (t1={r['tau1']:.2f}, t2={r['tau2']:.2f})")
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel value (log-hazard contribution)')
    ax.set_title('Gamma-Difference Kernels Across Conditions')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved kernel comparison plot to {output_path}")


def main():
    print("=" * 70)
    print("FIT GAMMA-DIFFERENCE KERNEL PER CONDITION")
    print("=" * 70)
    
    # Load conditions
    conditions = get_condition_files()
    
    print("\nConditions found:")
    for name, cond in conditions.items():
        print(f"  {name}: {len(cond.files)} files")
    
    # Load data
    print("\nLoading data (all experiments included)...")
    for name, cond in conditions.items():
        load_condition_data(cond, exclude_anomalous=False)
        print(f"  {name}: {cond.n_experiments} experiments → {cond.n_tracks} tracks, {cond.n_events} events")
    
    # Check for existing partial results (resume capability)
    output_path = Path('data/model/per_condition_timescales.json')
    results = {}
    
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            
            # Check if existing results have bootstrap samples
            has_samples = False
            for cond_name, cond_data in existing.items():
                if 'bootstrap' in cond_data and 'tau1' in cond_data['bootstrap']:
                    if 'samples' in cond_data['bootstrap']['tau1']:
                        has_samples = True
                        break
            
            if has_samples:
                print(f"\nFound existing results WITH samples for {len(existing)} conditions - will resume")
                results = existing
            else:
                # Archive old file without samples and start fresh
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = output_path.parent / f'per_condition_timescales_{timestamp}_no_samples.json'
                import shutil
                shutil.copy(output_path, archive_path)
                print(f"\n⚠ Existing results lack bootstrap samples")
                print(f"  Archived old file to: {archive_path}")
                print(f"  Starting fresh to generate samples...")
                results = {}
        except Exception as e:
            print(f"  Error loading existing: {e}")
            pass
    
    # Fit each condition (skip if already done AND has samples)
    for name, cond in conditions.items():
        if name in results and results[name].get('converged', False):
            # Double-check this condition has samples
            has_samples = False
            if 'bootstrap' in results[name] and 'tau1' in results[name]['bootstrap']:
                has_samples = 'samples' in results[name]['bootstrap']['tau1']
            
            if has_samples:
                print(f"\n[SKIP] {name}: already completed with samples")
                continue
            else:
                print(f"\n[RERUN] {name}: exists but missing samples, refitting...")
        
        if cond.n_events > 0:
            results[name] = fit_condition_gamma(cond, BO_OPTIMAL_CONFIG, do_bootstrap=True)
            
            # Save checkpoint after each condition
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  [CHECKPOINT] Saved to {output_path}")
        else:
            print(f"\nSkipping {name}: no events")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: TIMESCALE COMPARISON")
    print("=" * 70)
    print(f"\n{'Condition':<22} {'τ₁ (s)':<20} {'τ₂ (s)':<20} {'R²':<8}")
    print("-" * 70)
    
    for cond, r in results.items():
        if 'error' in r:
            print(f"{cond:<22} ERROR: {r.get('error', 'Unknown')}")
            continue
        
        tau1_str = f"{r['tau1']:.3f}"
        tau2_str = f"{r['tau2']:.3f}"
        
        if 'bootstrap' in r:
            if 'tau1' in r['bootstrap']:
                b = r['bootstrap']['tau1']
                tau1_str = f"{b['mean']:.3f} [{b['ci_lower']:.3f}, {b['ci_upper']:.3f}]"
            if 'tau2' in r['bootstrap']:
                b = r['bootstrap']['tau2']
                tau2_str = f"{b['mean']:.3f} [{b['ci_lower']:.3f}, {b['ci_upper']:.3f}]"
        
        print(f"{cond:<22} {tau1_str:<20} {tau2_str:<20} {r['r_squared']:.4f}")
    
    print("-" * 70)
    
    # Check CI overlap
    print("\nCI OVERLAP ASSESSMENT:")
    tau1_cis = []
    tau2_cis = []
    
    for cond, r in results.items():
        if 'bootstrap' in r:
            if 'tau1' in r['bootstrap']:
                b = r['bootstrap']['tau1']
                tau1_cis.append((b['ci_lower'], b['ci_upper']))
            if 'tau2' in r['bootstrap']:
                b = r['bootstrap']['tau2']
                tau2_cis.append((b['ci_lower'], b['ci_upper']))
    
    if len(tau1_cis) >= 2:
        # Check if all CIs overlap
        tau1_max_lower = max(ci[0] for ci in tau1_cis)
        tau1_min_upper = min(ci[1] for ci in tau1_cis)
        tau1_overlap = tau1_max_lower < tau1_min_upper
        print(f"  τ₁: {'ALL CIs OVERLAP' if tau1_overlap else 'CIs DO NOT ALL OVERLAP'}")
        print(f"      Common range: [{tau1_max_lower:.3f}, {tau1_min_upper:.3f}]" if tau1_overlap else "")
    
    if len(tau2_cis) >= 2:
        tau2_max_lower = max(ci[0] for ci in tau2_cis)
        tau2_min_upper = min(ci[1] for ci in tau2_cis)
        tau2_overlap = tau2_max_lower < tau2_min_upper
        print(f"  τ₂: {'ALL CIs OVERLAP' if tau2_overlap else 'CIs DO NOT ALL OVERLAP'}")
        print(f"      Common range: [{tau2_max_lower:.3f}, {tau2_min_upper:.3f}]" if tau2_overlap else "")
    
    # Save results
    output_dir = Path('data/model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove non-serializable data
    save_results = {}
    for cond, r in results.items():
        save_results[cond] = {k: v for k, v in r.items() 
                             if k not in ['kernel_t', 'kernel_K', 'kernel_K_fit']}
    
    output_path = output_dir / 'per_condition_timescales.json'
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Generate plots
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_timescale_comparison(results, fig_dir / 'timescale_comparison.png')
    plot_kernel_comparison(results, fig_dir / 'kernel_per_condition.png')
    
    return results


if __name__ == '__main__':
    main()


