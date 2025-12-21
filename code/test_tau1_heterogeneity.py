#!/usr/bin/env python3
"""
Test τ₁ Heterogeneity Across Conditions

Runs a permutation test on bootstrap τ₁ samples to formally test whether
timescales differ significantly across the 4 experimental conditions.

If raw bootstrap samples are not available, falls back to computing
effect sizes from summary statistics.

Output:
- data/model/tau1_heterogeneity_test.json

Usage:
    python scripts/test_tau1_heterogeneity.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_timescales(path: Path) -> Tuple[Dict, bool]:
    """
    Load per-condition timescales and check for raw samples.
    
    Returns (data, has_samples).
    """
    with open(path) as f:
        data = json.load(f)
    
    # Check if any condition has raw samples
    has_samples = False
    for cond, vals in data.items():
        if 'bootstrap' in vals and 'tau1' in vals['bootstrap']:
            if 'samples' in vals['bootstrap']['tau1']:
                has_samples = True
                break
    
    return data, has_samples


def permutation_test_from_samples(samples_by_condition: Dict[str, List[float]], 
                                  n_permutations: int = 10000,
                                  seed: int = 42) -> Dict:
    """
    Run permutation test for τ₁ heterogeneity using raw bootstrap samples.
    
    H₀: τ₁ is the same across all conditions
    H₁: τ₁ differs across conditions
    
    Test statistic: variance of condition means
    """
    np.random.seed(seed)
    
    conditions = list(samples_by_condition.keys())
    n_conditions = len(conditions)
    
    # Flatten all samples with condition labels
    all_samples = []
    labels = []
    for cond in conditions:
        samples = samples_by_condition[cond]
        all_samples.extend(samples)
        labels.extend([cond] * len(samples))
    
    all_samples = np.array(all_samples)
    labels = np.array(labels)
    n_total = len(all_samples)
    
    # Compute observed test statistic
    observed_means = [np.mean(samples_by_condition[c]) for c in conditions]
    T_obs = np.var(observed_means)
    
    print(f"\nPermutation test for τ₁ heterogeneity:")
    print(f"  Conditions: {n_conditions}")
    print(f"  Total samples: {n_total}")
    print(f"  Observed variance of means: {T_obs:.6f}")
    
    # Permutation test
    T_perm = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle labels
        shuffled_labels = np.random.permutation(labels)
        
        # Compute means for each condition under permutation
        perm_means = []
        for cond in conditions:
            mask = shuffled_labels == cond
            perm_means.append(np.mean(all_samples[mask]))
        
        T_perm[i] = np.var(perm_means)
    
    # p-value: proportion of permuted T >= observed T
    p_value = np.mean(T_perm >= T_obs)
    
    print(f"  p-value: {p_value:.4f} (n_permutations = {n_permutations})")
    
    return {
        'method': 'permutation_test',
        'test_statistic': 'variance_of_condition_means',
        'T_observed': round(T_obs, 6),
        'n_permutations': n_permutations,
        'p_value': round(p_value, 4),
        'significant_at_alpha_05': p_value < 0.05,
        'condition_means': {c: round(m, 4) for c, m in zip(conditions, observed_means)},
        'interpretation': 'Significant heterogeneity' if p_value < 0.05 else 'No significant heterogeneity'
    }


def anova_on_samples(samples_by_condition: Dict[str, List[float]]) -> Dict:
    """
    Run one-way ANOVA on bootstrap samples as alternative test.
    """
    from scipy import stats
    
    conditions = list(samples_by_condition.keys())
    groups = [samples_by_condition[c] for c in conditions]
    
    F, p = stats.f_oneway(*groups)
    
    print(f"\nOne-way ANOVA on bootstrap samples:")
    print(f"  F = {F:.4f}")
    print(f"  p = {p:.4f}")
    
    return {
        'method': 'one_way_anova',
        'F_statistic': round(F, 4),
        'p_value': round(p, 4),
        'significant_at_alpha_05': p < 0.05
    }


def kruskal_on_samples(samples_by_condition: Dict[str, List[float]]) -> Dict:
    """
    Run Kruskal-Wallis H-test on bootstrap samples (non-parametric alternative).
    """
    from scipy import stats
    
    conditions = list(samples_by_condition.keys())
    groups = [samples_by_condition[c] for c in conditions]
    
    H, p = stats.kruskal(*groups)
    
    print(f"\nKruskal-Wallis H-test on bootstrap samples:")
    print(f"  H = {H:.4f}")
    print(f"  p = {p:.4f}")
    
    return {
        'method': 'kruskal_wallis',
        'H_statistic': round(H, 4),
        'p_value': round(p, 4),
        'significant_at_alpha_05': p < 0.05
    }


def fallback_summary_stats(data: Dict) -> Dict:
    """
    When raw samples are not available, compute effect sizes from summary stats.
    """
    print("\n⚠ WARNING: Raw bootstrap samples not available")
    print("  Using summary statistics only (Hedges' g)")
    print("  Re-run fit_gamma_per_condition.py to generate samples")
    
    conditions = list(data.keys())
    tau1_stats = {}
    
    for cond in conditions:
        if 'bootstrap' in data[cond] and 'tau1' in data[cond]['bootstrap']:
            boot = data[cond]['bootstrap']['tau1']
            tau1_stats[cond] = {
                'mean': boot['mean'],
                'std': boot['std'],
                'n': boot['n_valid']
            }
    
    # Compute pairwise effect sizes
    from itertools import combinations
    
    pairwise = {}
    for c1, c2 in combinations(tau1_stats.keys(), 2):
        s1, s2 = tau1_stats[c1], tau1_stats[c2]
        pooled_std = np.sqrt((s1['std']**2 + s2['std']**2) / 2)
        g = (s1['mean'] - s2['mean']) / pooled_std if pooled_std > 0 else 0
        
        pairwise[f"{c1} vs {c2}"] = {
            'hedges_g': round(g, 3),
            'mean_diff': round(s1['mean'] - s2['mean'], 4)
        }
    
    # Compute I² (heterogeneity index) from summary stats
    # I² = (Q - df) / Q × 100%, where Q = Σ wi(θi - θ̄)²
    means = [tau1_stats[c]['mean'] for c in tau1_stats]
    vars_ = [tau1_stats[c]['std']**2 for c in tau1_stats]
    weights = [1/v if v > 0 else 0 for v in vars_]
    
    weighted_mean = np.average(means, weights=weights) if sum(weights) > 0 else np.mean(means)
    Q = sum(w * (m - weighted_mean)**2 for w, m in zip(weights, means))
    df = len(means) - 1
    I_squared = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    
    print(f"\nSummary-based heterogeneity:")
    print(f"  τ₁ range: {min(means):.3f} - {max(means):.3f} s ({max(means)/min(means):.1f}x)")
    print(f"  I² = {I_squared:.1f}%")
    
    return {
        'method': 'summary_stats_only',
        'warning': 'Raw bootstrap samples not available. Re-run fit_gamma_per_condition.py to enable permutation test.',
        'tau1_point_estimates': {c: round(tau1_stats[c]['mean'], 4) for c in tau1_stats},
        'tau1_range': {
            'min': round(min(means), 4),
            'max': round(max(means), 4),
            'fold_range': round(max(means) / min(means), 2) if min(means) > 0 else None
        },
        'I_squared': round(I_squared, 1),
        'I_squared_interpretation': 'low' if I_squared < 25 else ('moderate' if I_squared < 50 else 'substantial'),
        'pairwise_effect_sizes': pairwise
    }


def main():
    print("=" * 70)
    print("TEST τ₁ HETEROGENEITY ACROSS CONDITIONS")
    print("=" * 70)
    
    timescale_path = Path('data/model/per_condition_timescales.json')
    
    if not timescale_path.exists():
        print(f"ERROR: {timescale_path} not found")
        return
    
    data, has_samples = load_timescales(timescale_path)
    
    print(f"\nLoaded data for {len(data)} conditions")
    print(f"Raw bootstrap samples available: {has_samples}")
    
    results = {}
    
    if has_samples:
        # Extract samples
        samples_by_condition = {}
        for cond, vals in data.items():
            if 'bootstrap' in vals and 'tau1' in vals['bootstrap']:
                if 'samples' in vals['bootstrap']['tau1']:
                    samples_by_condition[cond] = vals['bootstrap']['tau1']['samples']
        
        # Run all tests
        results['permutation'] = permutation_test_from_samples(samples_by_condition)
        results['anova'] = anova_on_samples(samples_by_condition)
        results['kruskal'] = kruskal_on_samples(samples_by_condition)
        
    else:
        results['fallback'] = fallback_summary_stats(data)
    
    # Save results
    output_path = Path('data/model/tau1_heterogeneity_test.json')
    
    # Convert booleans for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return obj
    
    results = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if has_samples:
        perm = results['permutation']
        print(f"Permutation test p-value: {perm['p_value']}")
        print(f"Interpretation: {perm['interpretation']}")
    else:
        fb = results['fallback']
        print(f"⚠ Only summary stats available")
        print(f"τ₁ range: {fb['tau1_range']['fold_range']}x")
        print(f"I² = {fb['I_squared']}% ({fb['I_squared_interpretation']} heterogeneity)")
        print("\nTo run permutation test:")
        print("  1. Run: python scripts/fit_gamma_per_condition.py")
        print("  2. Re-run this script")
    
    return results


if __name__ == '__main__':
    main()
