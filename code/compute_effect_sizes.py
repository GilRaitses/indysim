#!/usr/bin/env python3
"""
Compute Effect Sizes for Rigor Upgrade

Computes Hedges' g for τ₁ differences between conditions using bootstrap SDs.
Also computes effect sizes for event durations.

Output:
- data/model/effect_sizes.json

Usage:
    python scripts/compute_effect_sizes.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from itertools import combinations


def hedges_g(mean1: float, std1: float, n1: int,
             mean2: float, std2: float, n2: int) -> Tuple[float, str]:
    """
    Compute Hedges' g (bias-corrected Cohen's d).
    
    Returns (effect_size, interpretation).
    """
    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return 0.0, "undefined"
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Hedges' correction for small samples
    # J = 1 - 3 / (4*(n1+n2) - 9)
    df = n1 + n2 - 2
    if df > 0:
        j = 1 - 3 / (4 * df - 1)
    else:
        j = 1
    
    g = d * j
    
    # Interpretation (Cohen's conventions)
    abs_g = abs(g)
    if abs_g < 0.2:
        interp = "negligible"
    elif abs_g < 0.5:
        interp = "small"
    elif abs_g < 0.8:
        interp = "medium"
    else:
        interp = "large"
    
    return g, interp


def compute_tau1_effect_sizes(timescales: Dict) -> Dict:
    """Compute pairwise Hedges' g for τ₁ across conditions."""
    conditions = list(timescales.keys())
    
    # Extract bootstrap stats for τ₁
    tau1_stats = {}
    for cond, data in timescales.items():
        if 'bootstrap' in data and 'tau1' in data['bootstrap']:
            boot = data['bootstrap']['tau1']
            tau1_stats[cond] = {
                'mean': boot['mean'],
                'std': boot['std'],
                'n': boot['n_valid']
            }
    
    # Compute pairwise effect sizes
    pairwise = {}
    for cond1, cond2 in combinations(tau1_stats.keys(), 2):
        s1 = tau1_stats[cond1]
        s2 = tau1_stats[cond2]
        
        g, interp = hedges_g(
            s1['mean'], s1['std'], s1['n'],
            s2['mean'], s2['std'], s2['n']
        )
        
        key = f"{cond1} vs {cond2}"
        pairwise[key] = {
            'hedges_g': round(g, 3),
            'interpretation': interp,
            'mean_diff': round(s1['mean'] - s2['mean'], 3),
            'condition1_mean': round(s1['mean'], 3),
            'condition2_mean': round(s2['mean'], 3)
        }
    
    # Summary
    g_values = [v['hedges_g'] for v in pairwise.values()]
    
    return {
        'tau1_pairwise': pairwise,
        'tau1_summary': {
            'max_effect': round(max(g_values, key=abs), 3),
            'mean_effect': round(np.mean(np.abs(g_values)), 3),
            'n_large_effects': sum(1 for g in g_values if abs(g) >= 0.8),
            'n_medium_effects': sum(1 for g in g_values if 0.5 <= abs(g) < 0.8),
            'n_small_effects': sum(1 for g in g_values if 0.2 <= abs(g) < 0.5)
        }
    }


def compute_tau2_effect_sizes(timescales: Dict) -> Dict:
    """Compute pairwise Hedges' g for τ₂ across conditions."""
    conditions = list(timescales.keys())
    
    # Extract bootstrap stats for τ₂
    tau2_stats = {}
    for cond, data in timescales.items():
        if 'bootstrap' in data and 'tau2' in data['bootstrap']:
            boot = data['bootstrap']['tau2']
            tau2_stats[cond] = {
                'mean': boot['mean'],
                'std': boot['std'],
                'n': boot['n_valid']
            }
    
    # Compute pairwise effect sizes
    pairwise = {}
    for cond1, cond2 in combinations(tau2_stats.keys(), 2):
        s1 = tau2_stats[cond1]
        s2 = tau2_stats[cond2]
        
        g, interp = hedges_g(
            s1['mean'], s1['std'], s1['n'],
            s2['mean'], s2['std'], s2['n']
        )
        
        key = f"{cond1} vs {cond2}"
        pairwise[key] = {
            'hedges_g': round(g, 3),
            'interpretation': interp,
            'mean_diff': round(s1['mean'] - s2['mean'], 3)
        }
    
    return {'tau2_pairwise': pairwise}


def compute_duration_effect_sizes(durations: Dict) -> Dict:
    """Compute effect sizes for event durations."""
    results = {}
    
    for event_type in ['run_duration', 'turn_duration', 'pause_duration', 'reverse_crawl_duration']:
        if event_type not in durations:
            continue
        
        event_data = durations[event_type]
        conditions = list(event_data.keys())
        
        pairwise = {}
        for cond1, cond2 in combinations(conditions, 2):
            d1 = event_data[cond1]
            d2 = event_data[cond2]
            
            # Use log-transformed stats for skewed durations
            # Approximate log mean and std from raw stats
            # For lognormal: log_mean ≈ log(mean) - 0.5*log(1 + (std/mean)²)
            def log_stats(mean, std):
                if mean <= 0:
                    return 0, 1
                cv_sq = (std / mean) ** 2
                log_mean = np.log(mean) - 0.5 * np.log(1 + cv_sq)
                log_std = np.sqrt(np.log(1 + cv_sq))
                return log_mean, log_std
            
            lm1, ls1 = log_stats(d1['mean'], d1['std'])
            lm2, ls2 = log_stats(d2['mean'], d2['std'])
            
            g, interp = hedges_g(lm1, ls1, d1['n'], lm2, ls2, d2['n'])
            
            key = f"{cond1} vs {cond2}"
            pairwise[key] = {
                'hedges_g_log': round(g, 3),
                'interpretation': interp,
                'ratio': round(d1['mean'] / d2['mean'], 2) if d2['mean'] > 0 else None
            }
        
        results[event_type] = pairwise
    
    return results


def main():
    print("=" * 70)
    print("COMPUTING EFFECT SIZES FOR RIGOR UPGRADE")
    print("=" * 70)
    
    # Load timescale data
    timescale_path = Path('data/model/per_condition_timescales.json')
    with open(timescale_path) as f:
        timescales = json.load(f)
    
    print(f"\nLoaded timescales for {len(timescales)} conditions")
    
    # Compute τ₁ effect sizes
    tau1_results = compute_tau1_effect_sizes(timescales)
    
    print("\n" + "=" * 50)
    print("τ₁ PAIRWISE EFFECT SIZES (Hedges' g)")
    print("=" * 50)
    for pair, stats in tau1_results['tau1_pairwise'].items():
        print(f"\n{pair}:")
        print(f"  Hedges' g = {stats['hedges_g']:.3f} ({stats['interpretation']})")
        print(f"  Δτ₁ = {stats['mean_diff']:.3f} s")
    
    print(f"\nSummary:")
    summary = tau1_results['tau1_summary']
    print(f"  Max effect: g = {summary['max_effect']}")
    print(f"  Large effects (|g| ≥ 0.8): {summary['n_large_effects']}")
    print(f"  Medium effects (0.5 ≤ |g| < 0.8): {summary['n_medium_effects']}")
    print(f"  Small effects (0.2 ≤ |g| < 0.5): {summary['n_small_effects']}")
    
    # Compute τ₂ effect sizes
    tau2_results = compute_tau2_effect_sizes(timescales)
    
    print("\n" + "=" * 50)
    print("τ₂ PAIRWISE EFFECT SIZES (Hedges' g)")
    print("=" * 50)
    for pair, stats in tau2_results['tau2_pairwise'].items():
        print(f"  {pair}: g = {stats['hedges_g']:.3f} ({stats['interpretation']})")
    
    # Load duration data
    duration_path = Path('data/model/event_duration_summary.json')
    with open(duration_path) as f:
        durations = json.load(f)
    
    duration_results = compute_duration_effect_sizes(durations)
    
    print("\n" + "=" * 50)
    print("EVENT DURATION EFFECT SIZES (log-scale Hedges' g)")
    print("=" * 50)
    for event_type, pairwise in duration_results.items():
        print(f"\n{event_type}:")
        for pair, stats in pairwise.items():
            print(f"  {pair}: g = {stats['hedges_g_log']:.3f} ({stats['interpretation']}), ratio = {stats['ratio']}")
    
    # Combine and save results
    all_results = {
        'tau1': tau1_results,
        'tau2': tau2_results,
        'durations': duration_results,
        'methodology': {
            'effect_size_metric': 'Hedges g (bias-corrected Cohen d)',
            'duration_transform': 'log-scale approximation from raw stats',
            'thresholds': {
                'small': '|g| >= 0.2',
                'medium': '|g| >= 0.5',
                'large': '|g| >= 0.8'
            },
            'note': 'Bootstrap SDs used as estimate of parameter uncertainty'
        }
    }
    
    output_path = Path('data/model/effect_sizes.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nSaved effect sizes to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()
