#!/usr/bin/env python3
"""
LOEO Null Test

Tests whether the 7/12 LOEO pass rate significantly exceeds chance (50%).

Output:
- Binomial test p-value
- 95% CI for pass rate
- Updated validation results

Usage:
    python scripts/compute_loeo_null_test.py
"""

import json
from pathlib import Path
from scipy import stats
import numpy as np


def binomial_test(successes: int, trials: int, null_p: float = 0.5) -> dict:
    """
    Run binomial test and compute confidence interval.
    
    Args:
        successes: Number of passing experiments
        trials: Total number of experiments
        null_p: Null hypothesis probability (default 0.5 = chance)
    
    Returns:
        Dict with p-value, CI, and interpretation
    """
    # One-sided test: is pass rate greater than chance?
    # Using exact binomial test (scipy >= 1.7)
    result = stats.binomtest(successes, trials, null_p, alternative='greater')
    p_value = result.pvalue
    
    # 95% Clopper-Pearson exact confidence interval
    ci_lower = stats.beta.ppf(0.025, successes, trials - successes + 1)
    ci_upper = stats.beta.ppf(0.975, successes + 1, trials - successes)
    
    observed_rate = successes / trials
    
    # Interpretation
    if p_value < 0.05:
        interpretation = "Pass rate significantly exceeds chance"
    elif p_value < 0.10:
        interpretation = "Marginal trend above chance"
    else:
        interpretation = "Pass rate not significantly different from chance"
    
    return {
        'successes': successes,
        'trials': trials,
        'observed_rate': round(observed_rate, 3),
        'null_p': null_p,
        'p_value': round(p_value, 4),
        'ci_95_lower': round(ci_lower, 3),
        'ci_95_upper': round(ci_upper, 3),
        'interpretation': interpretation,
        'significant_at_alpha_05': p_value < 0.05
    }


def main():
    print("=" * 70)
    print("LOEO NULL TEST: Is 7/12 pass rate better than chance?")
    print("=" * 70)
    
    # Known LOEO results: 7 of 12 experiments pass (rate ratio in [0.8, 1.25])
    successes = 7
    trials = 12
    
    result = binomial_test(successes, trials, null_p=0.5)
    
    print(f"\nLOEO Validation Results:")
    print(f"  Passing experiments: {result['successes']}/{result['trials']}")
    print(f"  Observed pass rate: {result['observed_rate']:.1%}")
    print(f"  95% CI: [{result['ci_95_lower']:.1%}, {result['ci_95_upper']:.1%}]")
    
    print(f"\nNull Hypothesis Test:")
    print(f"  H₀: pass rate = {result['null_p']:.0%} (chance)")
    print(f"  H₁: pass rate > {result['null_p']:.0%}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if result['significant_at_alpha_05'] else 'No'}")
    
    print(f"\nInterpretation: {result['interpretation']}")
    
    # Additional context
    print("\n" + "-" * 50)
    print("Context for Manuscript:")
    print("-" * 50)
    
    if not result['significant_at_alpha_05']:
        print("""
The LOEO validation shows 7 of 12 experiments (58%) with rate ratios
within ±25% of 1.0. However, this pass rate does not significantly
exceed chance (binomial p = {:.2f}). 

Suggested manuscript language:
"Leave-one-experiment-out validation showed that model predictions
matched observed event rates within ±25% for 7 of 12 experiments
(58%; 95% CI: [{:.0%}, {:.0%}]). While this indicates reasonable
consistency, the pass rate does not significantly exceed chance
(binomial p = {:.2f}), suggesting the validation is modest rather
than strong."
""".format(result['p_value'], result['ci_95_lower'], result['ci_95_upper'], result['p_value']))
    
    # Also test with different thresholds for sensitivity
    print("\n" + "=" * 50)
    print("SENSITIVITY: Different tolerance windows")
    print("=" * 50)
    
    # These would need actual data - using hypothetical values
    tolerance_tests = [
        ("±25% [0.80, 1.25]", 7, 12),
        ("±20% [0.83, 1.20]", 5, 12),  # hypothetical - stricter
        ("±15% [0.87, 1.15]", 4, 12),  # hypothetical - stricter
    ]
    
    print(f"\n{'Tolerance':<20} {'Pass':<8} {'Rate':<10} {'p-value':<10}")
    print("-" * 50)
    for label, s, n in tolerance_tests:
        r = binomial_test(s, n)
        print(f"{label:<20} {s}/{n:<6} {r['observed_rate']:.1%}     {r['p_value']:.3f}")
    
    print("\nNote: ±20% and ±15% values are hypothetical - would need actual data.")
    
    # Save results - convert bool to python bool for JSON serialization
    result_serializable = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) 
                           for k, v in result.items()}
    
    output_path = Path('data/model/loeo_null_test.json')
    with open(output_path, 'w') as f:
        json.dump({
            'primary_test': result_serializable,
            'methodology': {
                'test': 'Exact binomial test (one-sided)',
                'null_hypothesis': 'Pass rate equals 0.5 (chance)',
                'alternative': 'Pass rate greater than 0.5',
                'ci_method': 'Clopper-Pearson exact',
                'tolerance_window': '[0.80, 1.25] rate ratio'
            }
        }, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return result


if __name__ == '__main__':
    main()
