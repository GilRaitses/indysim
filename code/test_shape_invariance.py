#!/usr/bin/env python3
"""
Test Shape Invariance Across Conditions

Formal statistical test for whether kernel timescales (τ₁, τ₂) are 
conserved across conditions.

Approach:
1. Fit per-condition gamma kernels (from fit_gamma_per_condition.py output)
2. Compute pooled/shared τ₁, τ₂ estimates
3. Compare models via:
   - AIC/BIC (shared vs condition-specific)
   - Likelihood ratio test
   - CI overlap analysis

Usage:
    python scripts/test_shape_invariance.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_per_condition_results() -> Dict:
    """Load results from per-condition fitting."""
    results_path = Path('data/model/per_condition_timescales.json')
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Per-condition results not found: {results_path}\n"
            "Run fit_gamma_per_condition.py first."
        )
    
    with open(results_path) as f:
        return json.load(f)


def extract_timescales(results: Dict) -> Dict[str, Dict]:
    """Extract τ₁, τ₂ with CIs for each condition."""
    timescales = {}
    
    for cond, r in results.items():
        if 'error' in r or not r.get('converged', True):
            continue
        
        ts = {
            'tau1': r.get('tau1'),
            'tau2': r.get('tau2'),
        }
        
        # Get bootstrap CIs if available
        if 'bootstrap' in r:
            boot = r['bootstrap']
            if 'tau1' in boot:
                ts['tau1_ci'] = (boot['tau1']['ci_lower'], boot['tau1']['ci_upper'])
                ts['tau1_mean'] = boot['tau1']['mean']
                ts['tau1_std'] = boot['tau1']['std']
            if 'tau2' in boot:
                ts['tau2_ci'] = (boot['tau2']['ci_lower'], boot['tau2']['ci_upper'])
                ts['tau2_mean'] = boot['tau2']['mean']
                ts['tau2_std'] = boot['tau2']['std']
        
        timescales[cond] = ts
    
    return timescales


def test_ci_overlap(timescales: Dict[str, Dict]) -> Dict:
    """Test whether all CIs overlap."""
    results = {'tau1': {}, 'tau2': {}}
    
    for tau_name in ['tau1', 'tau2']:
        ci_key = f'{tau_name}_ci'
        
        # Collect all CIs
        cis = []
        conditions = []
        for cond, ts in timescales.items():
            if ci_key in ts:
                cis.append(ts[ci_key])
                conditions.append(cond)
        
        if len(cis) < 2:
            results[tau_name] = {'error': 'Not enough conditions with CIs'}
            continue
        
        # Find overlap region
        max_lower = max(ci[0] for ci in cis)
        min_upper = min(ci[1] for ci in cis)
        
        all_overlap = max_lower < min_upper
        
        # Pairwise overlap matrix
        n = len(cis)
        overlap_matrix = np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                # CIs overlap if max(lower) < min(upper)
                overlap = max(cis[i][0], cis[j][0]) < min(cis[i][1], cis[j][1])
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
        
        results[tau_name] = {
            'all_overlap': bool(all_overlap),
            'common_range': (float(max_lower), float(min_upper)) if all_overlap else None,
            'conditions': conditions,
            'cis': [(float(ci[0]), float(ci[1])) for ci in cis],
            'pairwise_overlap': overlap_matrix.tolist(),
            'n_non_overlapping_pairs': int(np.sum(~overlap_matrix) / 2)
        }
    
    return results


def compute_pooled_estimates(timescales: Dict[str, Dict]) -> Dict:
    """Compute pooled (inverse-variance weighted) τ estimates."""
    results = {}
    
    for tau_name in ['tau1', 'tau2']:
        means = []
        stds = []
        conditions = []
        
        for cond, ts in timescales.items():
            mean_key = f'{tau_name}_mean'
            std_key = f'{tau_name}_std'
            
            if mean_key in ts and std_key in ts and ts[std_key] > 0:
                means.append(ts[mean_key])
                stds.append(ts[std_key])
                conditions.append(cond)
        
        if len(means) < 2:
            results[tau_name] = {'error': 'Not enough conditions'}
            continue
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Inverse-variance weighting
        weights = 1 / (stds ** 2)
        pooled_mean = np.sum(weights * means) / np.sum(weights)
        pooled_se = 1 / np.sqrt(np.sum(weights))
        
        # Heterogeneity test (Cochran's Q)
        Q = np.sum(weights * (means - pooled_mean) ** 2)
        df = len(means) - 1
        p_heterogeneity = 1 - stats.chi2.cdf(Q, df)
        
        # I² statistic (proportion of variance due to heterogeneity)
        I2 = max(0, (Q - df) / Q) if Q > 0 else 0
        
        results[tau_name] = {
            'pooled_mean': float(pooled_mean),
            'pooled_se': float(pooled_se),
            'pooled_ci': (float(pooled_mean - 1.96 * pooled_se), 
                         float(pooled_mean + 1.96 * pooled_se)),
            'per_condition_means': means.tolist(),
            'per_condition_stds': stds.tolist(),
            'conditions': conditions,
            'Q_statistic': float(Q),
            'Q_df': df,
            'p_heterogeneity': float(p_heterogeneity),
            'I2': float(I2),
            'heterogeneity_significant': p_heterogeneity < 0.05
        }
    
    return results


def compute_aic_comparison(timescales: Dict[str, Dict]) -> Dict:
    """
    Compare AIC for shared vs condition-specific models.
    
    Approximate approach:
    - Shared model: 2 parameters (pooled τ₁, τ₂) for all conditions
    - Condition-specific: 2k parameters (τ₁, τ₂ per condition)
    
    Uses sum of squared deviations as proxy for likelihood.
    """
    results = {}
    
    for tau_name in ['tau1', 'tau2']:
        means = []
        stds = []
        
        for cond, ts in timescales.items():
            mean_key = f'{tau_name}_mean'
            std_key = f'{tau_name}_std'
            
            if mean_key in ts and std_key in ts:
                means.append(ts[mean_key])
                stds.append(ts[std_key])
        
        if len(means) < 2:
            continue
        
        means = np.array(means)
        stds = np.array(stds)
        k = len(means)  # Number of conditions
        
        # Pooled estimate
        weights = 1 / (stds ** 2)
        pooled = np.sum(weights * means) / np.sum(weights)
        
        # SS for shared model
        ss_shared = np.sum(weights * (means - pooled) ** 2)
        
        # SS for condition-specific model (perfect fit, SS = 0)
        ss_specific = 0
        
        # AIC approximation (using weighted SS as -2*log_lik proxy)
        # Shared: 1 parameter for this tau
        # Specific: k parameters for this tau
        aic_shared = ss_shared + 2 * 1
        aic_specific = ss_specific + 2 * k
        
        # BIC
        n_obs = k  # Each condition is one "observation" of the timescale
        bic_shared = ss_shared + np.log(n_obs) * 1
        bic_specific = ss_specific + np.log(n_obs) * k
        
        results[tau_name] = {
            'aic_shared': float(aic_shared),
            'aic_specific': float(aic_specific),
            'delta_aic': float(aic_shared - aic_specific),
            'bic_shared': float(bic_shared),
            'bic_specific': float(bic_specific),
            'delta_bic': float(bic_shared - bic_specific),
            'prefer_shared': aic_shared < aic_specific and bic_shared < bic_specific
        }
    
    return results


def main():
    print("=" * 70)
    print("SHAPE INVARIANCE TEST")
    print("=" * 70)
    
    # Load per-condition results
    try:
        results = load_per_condition_results()
    except FileNotFoundError as e:
        print(str(e))
        return
    
    print(f"Loaded results for {len(results)} conditions")
    
    # Extract timescales
    timescales = extract_timescales(results)
    print(f"  Valid conditions: {list(timescales.keys())}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("PER-CONDITION TIMESCALES")
    print("-" * 70)
    print(f"{'Condition':<22} {'τ₁ (s)':<25} {'τ₂ (s)':<25}")
    print("-" * 70)
    
    for cond, ts in timescales.items():
        tau1_str = f"{ts.get('tau1', 'N/A'):.3f}"
        tau2_str = f"{ts.get('tau2', 'N/A'):.3f}"
        
        if 'tau1_ci' in ts:
            tau1_str = f"{ts['tau1_mean']:.3f} [{ts['tau1_ci'][0]:.3f}, {ts['tau1_ci'][1]:.3f}]"
        if 'tau2_ci' in ts:
            tau2_str = f"{ts['tau2_mean']:.3f} [{ts['tau2_ci'][0]:.3f}, {ts['tau2_ci'][1]:.3f}]"
        
        print(f"{cond:<22} {tau1_str:<25} {tau2_str:<25}")
    
    # CI overlap test
    print("\n" + "=" * 70)
    print("TEST 1: CONFIDENCE INTERVAL OVERLAP")
    print("=" * 70)
    
    ci_results = test_ci_overlap(timescales)
    
    for tau_name, res in ci_results.items():
        if 'error' in res:
            print(f"\n{tau_name}: {res['error']}")
            continue
        
        print(f"\n{tau_name.upper()}:")
        print(f"  All CIs overlap: {res['all_overlap']}")
        if res['all_overlap']:
            print(f"  Common range: [{res['common_range'][0]:.3f}, {res['common_range'][1]:.3f}]")
        print(f"  Non-overlapping pairs: {res['n_non_overlapping_pairs']}")
    
    # Pooled estimates and heterogeneity test
    print("\n" + "=" * 70)
    print("TEST 2: HETEROGENEITY TEST (Cochran's Q)")
    print("=" * 70)
    
    pooled_results = compute_pooled_estimates(timescales)
    
    for tau_name, res in pooled_results.items():
        if 'error' in res:
            print(f"\n{tau_name}: {res['error']}")
            continue
        
        print(f"\n{tau_name.upper()}:")
        print(f"  Pooled estimate: {res['pooled_mean']:.3f} ± {res['pooled_se']:.3f}")
        print(f"  95% CI: [{res['pooled_ci'][0]:.3f}, {res['pooled_ci'][1]:.3f}]")
        print(f"  Cochran's Q: {res['Q_statistic']:.2f} (df={res['Q_df']}, p={res['p_heterogeneity']:.3f})")
        print(f"  I² (heterogeneity): {res['I2']*100:.1f}%")
        
        if res['heterogeneity_significant']:
            print(f"  ⚠️  SIGNIFICANT HETEROGENEITY (p < 0.05)")
        else:
            print(f"  ✓ No significant heterogeneity")
    
    # AIC/BIC comparison
    print("\n" + "=" * 70)
    print("TEST 3: MODEL COMPARISON (AIC/BIC)")
    print("=" * 70)
    
    aic_results = compute_aic_comparison(timescales)
    
    for tau_name, res in aic_results.items():
        print(f"\n{tau_name.upper()}:")
        print(f"  Shared model AIC:    {res['aic_shared']:.2f}")
        print(f"  Specific model AIC:  {res['aic_specific']:.2f}")
        print(f"  ΔAIC (shared - specific): {res['delta_aic']:.2f}")
        print(f"  ΔBIC (shared - specific): {res['delta_bic']:.2f}")
        
        if res['prefer_shared']:
            print(f"  ✓ SHARED MODEL PREFERRED (simpler is better)")
        else:
            print(f"  ⚠️  Condition-specific model may be justified")
    
    # Overall conclusion
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    
    tau1_conserved = True
    tau2_conserved = True
    
    if 'tau1' in ci_results and not ci_results['tau1'].get('all_overlap', True):
        tau1_conserved = False
    if 'tau1' in pooled_results and pooled_results['tau1'].get('heterogeneity_significant', False):
        tau1_conserved = False
    
    if 'tau2' in ci_results and not ci_results['tau2'].get('all_overlap', True):
        tau2_conserved = False
    if 'tau2' in pooled_results and pooled_results['tau2'].get('heterogeneity_significant', False):
        tau2_conserved = False
    
    print(f"\nτ₁ (fast timescale): {'CONSERVED ✓' if tau1_conserved else 'VARIES ACROSS CONDITIONS ⚠️'}")
    print(f"τ₂ (slow timescale): {'CONSERVED ✓' if tau2_conserved else 'VARIES ACROSS CONDITIONS ⚠️'}")
    
    if tau1_conserved and tau2_conserved:
        print("\n✓ DYNAMICS APPEAR CONSERVED ACROSS CONDITIONS")
        print("  The shared-dynamics model is statistically supported.")
    else:
        print("\n⚠️  DYNAMICS MAY NOT BE FULLY CONSERVED")
        print("  Consider condition-specific timescales or soften claims.")
    
    # Save results
    output = {
        'timescales': timescales,
        'ci_overlap': ci_results,
        'pooled_estimates': pooled_results,
        'aic_comparison': aic_results,
        'conclusion': {
            'tau1_conserved': tau1_conserved,
            'tau2_conserved': tau2_conserved,
            'overall_conserved': tau1_conserved and tau2_conserved
        }
    }
    
    output_path = Path('data/model/shape_invariance_test.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved results to {output_path}")
    
    return output


if __name__ == '__main__':
    main()


