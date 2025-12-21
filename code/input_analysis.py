#!/usr/bin/env python3
"""
Input Analysis for Larval Behavior Simulation (ECS630 Methodology)

Characterizes input distributions before model fitting:
- Speed: Lognormal distribution (expected p > 0.05)
- Inter-event interval: Gamma distribution (exponential rejected)
- Turn angle: von Mises distribution
- Temporal structure: ACF, stationarity checks

Usage:
    python scripts/input_analysis.py --input data/processed/consolidated_dataset.h5
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.stats import lognorm, gamma, expon, kstest, vonmises
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectories_from_h5(h5_path: Path) -> pd.DataFrame:
    """Load trajectory data from consolidated H5 file."""
    print(f"Loading trajectories from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'trajectories' not in f:
            raise ValueError("No 'trajectories' group in H5 file")
        
        grp = f['trajectories']
        data = {}
        for key in grp.keys():
            arr = grp[key][:]
            if arr.dtype.kind == 'S':  # byte string
                arr = arr.astype(str)
            data[key] = arr
        
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df):,} rows")
    return df


def load_events_from_h5(h5_path: Path) -> pd.DataFrame:
    """Load event data from consolidated H5 file."""
    print(f"Loading events from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'events' not in f:
            raise ValueError("No 'events' group in H5 file")
        
        grp = f['events']
        data = {}
        for key in grp.keys():
            arr = grp[key][:]
            if arr.dtype.kind == 'S':
                arr = arr.astype(str)
            data[key] = arr
        
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df):,} rows")
    return df


# =============================================================================
# DISTRIBUTION FITTING
# =============================================================================

def fit_speed_distribution(speed_data: np.ndarray, max_samples: int = 100000) -> Dict:
    """
    Fit lognormal, gamma, and weibull to speed data.
    
    Returns dict with fit parameters and KS test results.
    """
    # Subsample for efficiency
    if len(speed_data) > max_samples:
        speed_data = np.random.choice(speed_data, max_samples, replace=False)
    
    # Filter valid speeds (positive)
    speed_data = speed_data[speed_data > 0]
    
    results = {}
    
    # Lognormal fit
    try:
        ln_params = lognorm.fit(speed_data, floc=0)
        ln_ks, ln_p = kstest(speed_data, 'lognorm', ln_params)
        results['lognormal'] = {
            'params': {'s': ln_params[0], 'loc': ln_params[1], 'scale': ln_params[2]},
            'ks_stat': ln_ks,
            'p_value': ln_p,
            'accepted': ln_p > 0.05
        }
    except Exception as e:
        results['lognormal'] = {'error': str(e)}
    
    # Gamma fit
    try:
        gam_params = gamma.fit(speed_data, floc=0)
        gam_ks, gam_p = kstest(speed_data, 'gamma', gam_params)
        results['gamma'] = {
            'params': {'a': gam_params[0], 'loc': gam_params[1], 'scale': gam_params[2]},
            'ks_stat': gam_ks,
            'p_value': gam_p,
            'accepted': gam_p > 0.05
        }
    except Exception as e:
        results['gamma'] = {'error': str(e)}
    
    # Weibull fit
    try:
        from scipy.stats import weibull_min
        wb_params = weibull_min.fit(speed_data, floc=0)
        wb_ks, wb_p = kstest(speed_data, 'weibull_min', wb_params)
        results['weibull'] = {
            'params': {'c': wb_params[0], 'loc': wb_params[1], 'scale': wb_params[2]},
            'ks_stat': wb_ks,
            'p_value': wb_p,
            'accepted': wb_p > 0.05
        }
    except Exception as e:
        results['weibull'] = {'error': str(e)}
    
    # Select best fit (highest p-value)
    valid_fits = {k: v for k, v in results.items() if 'p_value' in v}
    if valid_fits:
        best = max(valid_fits.keys(), key=lambda k: valid_fits[k]['p_value'])
        results['best_fit'] = best
    
    return results


def fit_iei_distribution(iei_data: np.ndarray, max_samples: int = 50000) -> Dict:
    """
    Fit exponential and gamma to inter-event intervals.
    
    Explicitly tests whether exponential is rejected (validates non-Poisson process).
    """
    if len(iei_data) > max_samples:
        iei_data = np.random.choice(iei_data, max_samples, replace=False)
    
    iei_data = iei_data[iei_data > 0]
    
    results = {}
    
    # Exponential fit
    try:
        exp_params = expon.fit(iei_data, floc=0)
        exp_ks, exp_p = kstest(iei_data, 'expon', exp_params)
        results['exponential'] = {
            'params': {'loc': exp_params[0], 'scale': exp_params[1]},
            'ks_stat': exp_ks,
            'p_value': exp_p,
            'accepted': exp_p > 0.05,
            'rejected': exp_p < 0.05  # Explicitly flag rejection
        }
    except Exception as e:
        results['exponential'] = {'error': str(e)}
    
    # Gamma fit
    try:
        gam_params = gamma.fit(iei_data, floc=0)
        gam_ks, gam_p = kstest(iei_data, 'gamma', gam_params)
        results['gamma'] = {
            'params': {'a': gam_params[0], 'loc': gam_params[1], 'scale': gam_params[2]},
            'ks_stat': gam_ks,
            'p_value': gam_p,
            'accepted': gam_p > 0.05
        }
    except Exception as e:
        results['gamma'] = {'error': str(e)}
    
    # Weibull fit
    try:
        from scipy.stats import weibull_min
        wb_params = weibull_min.fit(iei_data, floc=0)
        wb_ks, wb_p = kstest(iei_data, 'weibull_min', wb_params)
        results['weibull'] = {
            'params': {'c': wb_params[0], 'loc': wb_params[1], 'scale': wb_params[2]},
            'ks_stat': wb_ks,
            'p_value': wb_p,
            'accepted': wb_p > 0.05
        }
    except Exception as e:
        results['weibull'] = {'error': str(e)}
    
    # Implication for LNP model
    if 'exponential' in results and results['exponential'].get('rejected', False):
        results['implication'] = 'Non-Poisson process; NB-GLM with temporal kernels is appropriate'
    else:
        results['implication'] = 'Poisson process may be adequate'
    
    # Select best fit
    valid_fits = {k: v for k, v in results.items() 
                  if isinstance(v, dict) and 'p_value' in v}
    if valid_fits:
        best = max(valid_fits.keys(), key=lambda k: valid_fits[k]['p_value'])
        results['best_fit'] = best
    
    return results


def fit_turn_angle_distribution(angle_data: np.ndarray) -> Dict:
    """
    Fit von Mises distribution to turn angles.
    
    Tests for symmetry (H0: mean = 0).
    """
    # Ensure angles are in [-pi, pi]
    angle_data = np.arctan2(np.sin(angle_data), np.cos(angle_data))
    
    results = {}
    
    # von Mises fit
    try:
        # MLE for von Mises: kappa from R-bar
        n = len(angle_data)
        C = np.sum(np.cos(angle_data))
        S = np.sum(np.sin(angle_data))
        R_bar = np.sqrt(C**2 + S**2) / n
        mean_angle = np.arctan2(S, C)
        
        # Approximate kappa from R_bar
        if R_bar < 0.53:
            kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
        elif R_bar < 0.85:
            kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
        else:
            kappa = 1 / (R_bar**3 - 4 * R_bar**2 + 3 * R_bar)
        
        # KS test against von Mises
        vm_ks, vm_p = kstest(angle_data, lambda x: vonmises.cdf(x, kappa, loc=mean_angle))
        
        results['von_mises'] = {
            'params': {'kappa': kappa, 'mu': mean_angle},
            'ks_stat': vm_ks,
            'p_value': vm_p,
            'accepted': vm_p > 0.05
        }
        
        # Test for symmetry (mean = 0)
        from scipy.stats import circmean
        cm = circmean(angle_data, high=np.pi, low=-np.pi)
        results['symmetry'] = {
            'circular_mean': cm,
            'symmetric': np.abs(cm) < 0.1  # Within ~6 degrees of 0
        }
        
    except Exception as e:
        results['von_mises'] = {'error': str(e)}
    
    # Uniform test (should reject)
    try:
        # Rayleigh test for uniformity
        R = np.sqrt(C**2 + S**2) / n
        rayleigh_stat = 2 * n * R**2
        rayleigh_p = np.exp(-rayleigh_stat)  # Approximate p-value
        results['uniform'] = {
            'rayleigh_stat': rayleigh_stat,
            'p_value': rayleigh_p,
            'rejected': rayleigh_p < 0.05
        }
    except Exception as e:
        results['uniform'] = {'error': str(e)}
    
    return results


# =============================================================================
# TEMPORAL STRUCTURE
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 10) -> Dict:
    """Compute autocorrelation function."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return {'error': 'Zero variance'}
    
    acf = {}
    for lag in range(1, min(max_lag + 1, n // 2)):
        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf[f'lag_{lag}'] = cov / var
    
    lag1 = acf.get('lag_1', 0)
    
    return {
        'acf': acf,
        'lag1': lag1,
        'flag_autocorrelation': np.abs(lag1) > 0.1,
        'recommendation': 'Use cluster-robust SEs' if np.abs(lag1) > 0.1 else 'IID assumption OK'
    }


def test_stationarity(df: pd.DataFrame, time_col: str = 'time') -> Dict:
    """
    Compare turn rate in first vs second half of each experiment.
    
    Uses paired t-test across experiments.
    """
    results = {'experiments': []}
    
    # Detect reorientation onsets
    df = df.sort_values(['experiment_id', 'track_id', time_col])
    
    if 'is_reorientation' in df.columns:
        df['reo_onset'] = (
            df.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
    else:
        df['reo_onset'] = False
    
    first_half_rates = []
    second_half_rates = []
    
    for exp_id in df['experiment_id'].unique():
        exp_df = df[df['experiment_id'] == exp_id]
        
        # Get time range
        t_min = exp_df[time_col].min()
        t_max = exp_df[time_col].max()
        t_mid = (t_min + t_max) / 2
        
        # Count events in each half
        first_half = exp_df[exp_df[time_col] < t_mid]
        second_half = exp_df[exp_df[time_col] >= t_mid]
        
        n_first = first_half['reo_onset'].sum()
        n_second = second_half['reo_onset'].sum()
        
        # Duration in minutes
        dur_first = (t_mid - t_min) / 60
        dur_second = (t_max - t_mid) / 60
        
        rate_first = n_first / dur_first if dur_first > 0 else 0
        rate_second = n_second / dur_second if dur_second > 0 else 0
        
        first_half_rates.append(rate_first)
        second_half_rates.append(rate_second)
        
        results['experiments'].append({
            'experiment_id': str(exp_id),
            'rate_first_half': rate_first,
            'rate_second_half': rate_second,
            'difference': rate_second - rate_first
        })
    
    # Paired t-test
    if len(first_half_rates) > 1:
        t_stat, p_value = stats.ttest_rel(first_half_rates, second_half_rates)
        results['paired_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'stationary': p_value > 0.05
        }
    
    results['recommendation'] = (
        'No warm-up needed' if results.get('paired_ttest', {}).get('stationary', True)
        else 'Consider 5-min warm-up period'
    )
    
    return results


def compute_inter_event_intervals(df: pd.DataFrame) -> np.ndarray:
    """Compute inter-event intervals from event data."""
    # Check for required columns
    required_cols = ['experiment_id', 'track_id', 'time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  Warning: Missing columns {missing}, cannot compute IEI")
        return np.array([])
    
    df = df.sort_values(['experiment_id', 'track_id', 'time'])
    
    # Use is_reorientation_start if available, else detect from is_reorientation
    if 'is_reorientation_start' in df.columns:
        event_col = 'is_reorientation_start'
    elif 'is_reorientation' in df.columns:
        df['reo_onset'] = (
            df.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
        event_col = 'reo_onset'
    else:
        print("  Warning: No reorientation columns found")
        return np.array([])
    
    iei_list = []
    event_df = df[df[event_col].astype(bool)]
    
    for (exp, track), group in event_df.groupby(['experiment_id', 'track_id']):
        times = group['time'].sort_values().values
        if len(times) > 1:
            intervals = np.diff(times)
            iei_list.extend(intervals[intervals > 0])
    
    return np.array(iei_list)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_speed_distribution(speed_data: np.ndarray, fit_results: Dict, output_path: Path):
    """Plot speed histogram with fitted lognormal overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    speed_data = speed_data[speed_data > 0]
    ax.hist(speed_data, bins=100, density=True, alpha=0.7, label='Empirical')
    
    # Fitted lognormal
    if 'lognormal' in fit_results and 'params' in fit_results['lognormal']:
        params = fit_results['lognormal']['params']
        x = np.linspace(speed_data.min(), np.percentile(speed_data, 99), 200)
        pdf = lognorm.pdf(x, params['s'], params['loc'], params['scale'])
        p_val = fit_results['lognormal'].get('p_value', 0)
        ax.plot(x, pdf, 'r-', lw=2, label=f"Lognormal (p={p_val:.3f})")
    
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution with Lognormal Fit')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_iei_distribution(iei_data: np.ndarray, fit_results: Dict, output_path: Path):
    """Plot IEI histogram with gamma and rejected exponential overlays."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iei_data = iei_data[iei_data > 0]
    ax.hist(iei_data, bins=100, density=True, alpha=0.7, label='Empirical')
    
    x = np.linspace(iei_data.min(), np.percentile(iei_data, 99), 200)
    
    # Gamma fit
    if 'gamma' in fit_results and 'params' in fit_results['gamma']:
        params = fit_results['gamma']['params']
        pdf = gamma.pdf(x, params['a'], params['loc'], params['scale'])
        p_val = fit_results['gamma'].get('p_value', 0)
        ax.plot(x, pdf, 'g-', lw=2, label=f"Gamma (p={p_val:.3f})")
    
    # Exponential (rejected)
    if 'exponential' in fit_results and 'params' in fit_results['exponential']:
        params = fit_results['exponential']['params']
        pdf = expon.pdf(x, params['loc'], params['scale'])
        p_val = fit_results['exponential'].get('p_value', 0)
        ax.plot(x, pdf, 'r--', lw=2, label=f"Exponential REJECTED (p={p_val:.3f})")
    
    ax.set_xlabel('Inter-Event Interval (s)')
    ax.set_ylabel('Density')
    ax.set_title('IEI Distribution: Gamma Fit (Exponential Rejected)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_acf(acf_results: Dict, output_path: Path):
    """Plot ACF with significance bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    acf = acf_results.get('acf', {})
    lags = [int(k.split('_')[1]) for k in acf.keys()]
    values = list(acf.values())
    
    ax.bar(lags, values, color='steelblue', alpha=0.7)
    ax.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    ax.axhline(y=-0.1, color='r', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f"Speed ACF (Lag-1 = {acf_results.get('lag1', 0):.3f})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_turn_angle_distribution(angle_data: np.ndarray, fit_results: Dict, output_path: Path):
    """Plot polar histogram of turn angles with von Mises fit."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # Histogram
    bins = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(angle_data, bins=bins)
    counts = counts / counts.sum()  # Normalize
    
    theta = (bins[:-1] + bins[1:]) / 2
    ax.bar(theta, counts, width=np.diff(bins)[0], alpha=0.7, label='Empirical')
    
    # von Mises fit
    if 'von_mises' in fit_results and 'params' in fit_results['von_mises']:
        params = fit_results['von_mises']['params']
        x = np.linspace(-np.pi, np.pi, 200)
        pdf = vonmises.pdf(x, params['kappa'], loc=params['mu'])
        pdf = pdf / pdf.sum() * len(bins)  # Scale to match histogram
        ax.plot(x, pdf * 0.1, 'r-', lw=2, label=f"von Mises (κ={params['kappa']:.2f})")
    
    ax.set_title('Turn Angle Distribution')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    speed_results: Dict,
    iei_results: Dict,
    angle_results: Dict,
    acf_results: Dict,
    stationarity_results: Dict,
    output_path: Path
):
    """Generate markdown report."""
    lines = [
        "# Input Analysis Report",
        "",
        "## Summary",
        "",
        "| Variable | Best Fit | KS p-value | Status |",
        "|----------|----------|------------|--------|",
    ]
    
    # Speed
    if 'lognormal' in speed_results:
        p = speed_results['lognormal'].get('p_value', 0)
        status = 'ACCEPTED' if p > 0.05 else 'REJECTED'
        lines.append(f"| Speed | Lognormal | {p:.4f} | {status} |")
    
    # IEI
    if 'gamma' in iei_results:
        p = iei_results['gamma'].get('p_value', 0)
        status = 'ACCEPTED' if p > 0.05 else 'REJECTED'
        lines.append(f"| Inter-event interval | Gamma | {p:.4f} | {status} |")
    
    if 'exponential' in iei_results:
        p = iei_results['exponential'].get('p_value', 0)
        status = 'REJECTED' if p < 0.05 else 'ACCEPTED'
        lines.append(f"| IEI (Exponential) | Exponential | {p:.4f} | {status} |")
    
    # Angle
    if 'von_mises' in angle_results:
        p = angle_results['von_mises'].get('p_value', 0)
        status = 'ACCEPTED' if p > 0.05 else 'REJECTED'
        lines.append(f"| Turn angle | von Mises | {p:.4f} | {status} |")
    
    lines.extend([
        "",
        "## Temporal Structure",
        "",
        f"**ACF Lag-1:** {acf_results.get('lag1', 0):.3f}",
        f"**Recommendation:** {acf_results.get('recommendation', 'N/A')}",
        "",
        "## Stationarity",
        "",
    ])
    
    if 'paired_ttest' in stationarity_results:
        tt = stationarity_results['paired_ttest']
        lines.extend([
            f"**Paired t-test p-value:** {tt.get('p_value', 0):.4f}",
            f"**Stationary:** {'Yes' if tt.get('stationary', False) else 'No'}",
            f"**Recommendation:** {stationarity_results.get('recommendation', 'N/A')}",
        ])
    
    lines.extend([
        "",
        "## Implications for LNP Model",
        "",
        f"- {iei_results.get('implication', 'N/A')}",
        f"- Use cluster-robust SEs: {'Yes' if acf_results.get('flag_autocorrelation', False) else 'No'}",
        "",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved {output_path}")


def generate_summary_table(
    speed_results: Dict,
    iei_results: Dict,
    angle_results: Dict,
    output_path: Path
):
    """Generate CSV summary table."""
    rows = []
    
    # Speed
    if 'lognormal' in speed_results and 'params' in speed_results['lognormal']:
        rows.append({
            'variable': 'speed',
            'distribution': 'lognormal',
            'param1': speed_results['lognormal']['params']['s'],
            'param2': speed_results['lognormal']['params']['scale'],
            'ks_stat': speed_results['lognormal'].get('ks_stat', np.nan),
            'p_value': speed_results['lognormal'].get('p_value', np.nan),
            'accepted': speed_results['lognormal'].get('accepted', False)
        })
    
    # IEI - Gamma
    if 'gamma' in iei_results and 'params' in iei_results['gamma']:
        rows.append({
            'variable': 'iei',
            'distribution': 'gamma',
            'param1': iei_results['gamma']['params']['a'],
            'param2': iei_results['gamma']['params']['scale'],
            'ks_stat': iei_results['gamma'].get('ks_stat', np.nan),
            'p_value': iei_results['gamma'].get('p_value', np.nan),
            'accepted': iei_results['gamma'].get('accepted', False)
        })
    
    # IEI - Exponential (rejected)
    if 'exponential' in iei_results and 'params' in iei_results['exponential']:
        rows.append({
            'variable': 'iei',
            'distribution': 'exponential',
            'param1': iei_results['exponential']['params']['scale'],
            'param2': np.nan,
            'ks_stat': iei_results['exponential'].get('ks_stat', np.nan),
            'p_value': iei_results['exponential'].get('p_value', np.nan),
            'accepted': iei_results['exponential'].get('accepted', False)
        })
    
    # Turn angle
    if 'von_mises' in angle_results and 'params' in angle_results['von_mises']:
        rows.append({
            'variable': 'turn_angle',
            'distribution': 'von_mises',
            'param1': angle_results['von_mises']['params']['kappa'],
            'param2': angle_results['von_mises']['params']['mu'],
            'ks_stat': angle_results['von_mises'].get('ks_stat', np.nan),
            'p_value': angle_results['von_mises'].get('p_value', np.nan),
            'accepted': angle_results['von_mises'].get('accepted', False)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Input analysis for larval behavior')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to consolidated H5 file')
    parser.add_argument('--output', type=str, default='data/analysis/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_trajectories_from_h5(input_path)
    
    print("\n=== Distribution Fitting ===")
    
    # Speed distribution
    print("\nFitting speed distribution...")
    speed_col = 'speed' if 'speed' in df.columns else 'SpeedRunVel'
    if speed_col in df.columns:
        speed_data = df[speed_col].dropna().values
        speed_results = fit_speed_distribution(speed_data)
        print(f"  Best fit: {speed_results.get('best_fit', 'N/A')}")
        if 'lognormal' in speed_results:
            print(f"  Lognormal p-value: {speed_results['lognormal'].get('p_value', 0):.4f}")
    else:
        speed_results = {'error': 'No speed column found'}
        speed_data = np.array([])
    
    # Inter-event interval - use events data which has experiment_id and track_id
    print("\nComputing inter-event intervals...")
    try:
        events_df = load_events_from_h5(input_path)
        iei_data = compute_inter_event_intervals(events_df)
    except Exception as e:
        print(f"  Warning: Could not load events: {e}")
        iei_data = np.array([])
    
    if len(iei_data) > 0:
        print(f"  Found {len(iei_data):,} intervals")
        iei_results = fit_iei_distribution(iei_data)
        print(f"  Best fit: {iei_results.get('best_fit', 'N/A')}")
        if 'exponential' in iei_results:
            exp_rejected = iei_results['exponential'].get('rejected', False)
            print(f"  Exponential rejected: {exp_rejected}")
            print(f"  Implication: {iei_results.get('implication', 'N/A')}")
    else:
        iei_results = {'error': 'No IEI data available'}
    
    # Turn angle
    print("\nFitting turn angle distribution...")
    if 'reo_dtheta' in df.columns:
        angle_data = df['reo_dtheta'].dropna().values
        angle_results = fit_turn_angle_distribution(angle_data)
        if 'von_mises' in angle_results and 'params' in angle_results['von_mises']:
            kappa = angle_results['von_mises']['params']['kappa']
            print(f"  von Mises kappa: {kappa:.3f}")
    else:
        # Try to compute from heading changes during reorientations
        angle_data = np.array([])
        angle_results = {'error': 'No turn angle data available'}
    
    print("\n=== Temporal Structure ===")
    
    # ACF
    print("\nComputing speed ACF...")
    if len(speed_data) > 0:
        # Subsample for ACF
        acf_sample = speed_data[:min(100000, len(speed_data))]
        acf_results = compute_acf(acf_sample)
        print(f"  Lag-1 ACF: {acf_results.get('lag1', 0):.3f}")
        print(f"  Recommendation: {acf_results.get('recommendation', 'N/A')}")
    else:
        acf_results = {'error': 'No speed data for ACF'}
    
    # Stationarity - use events_df if available
    print("\nTesting stationarity...")
    if 'events_df' in dir() and len(events_df) > 0:
        stationarity_results = test_stationarity(events_df)
    else:
        stationarity_results = {'error': 'No events data for stationarity test'}
    
    if 'paired_ttest' in stationarity_results:
        p = stationarity_results['paired_ttest'].get('p_value', 0)
        print(f"  Paired t-test p-value: {p:.4f}")
        print(f"  Recommendation: {stationarity_results.get('recommendation', 'N/A')}")
    
    print("\n=== Generating Outputs ===")
    
    # Plots
    if len(speed_data) > 0:
        plot_speed_distribution(speed_data, speed_results, output_dir / 'speed_distribution.png')
    
    if len(iei_data) > 0:
        plot_iei_distribution(iei_data, iei_results, output_dir / 'iei_distribution.png')
    
    if 'acf' in acf_results:
        plot_acf(acf_results, output_dir / 'speed_acf.png')
    
    if len(angle_data) > 0:
        plot_turn_angle_distribution(angle_data, angle_results, output_dir / 'turn_angle_distribution.png')
    
    # Reports
    generate_report(
        speed_results, iei_results, angle_results,
        acf_results, stationarity_results,
        output_dir / 'input_analysis_report.md'
    )
    
    generate_summary_table(
        speed_results, iei_results, angle_results,
        output_dir / 'input_summary_table.csv'
    )
    
    print("\n=== Input Analysis Complete ===")


if __name__ == '__main__':
    main()
