#!/usr/bin/env python3
"""
Add deviance-based metrics to model results.

Computes:
- Null model log-likelihood (intercept-only)
- McFadden's pseudo-R2 = 1 - (llf / null_llf)

Usage:
    python scripts/add_model_metrics.py
"""

import json
import numpy as np
from pathlib import Path


def compute_null_llf_from_data(n_obs: int, n_events: int, dispersion: float = 0.1) -> float:
    """
    Compute null model log-likelihood (intercept-only NB model).
    
    For NB with intercept only:
        mu = n_events / n_obs (constant rate)
        llf = sum of NB log-probabilities
    
    Approximation using Poisson limit for low-rate events.
    """
    # Mean rate under null
    mean_rate = n_events / n_obs if n_obs > 0 else 0.01
    
    # For very sparse events (mean_rate << 1), Poisson approximation:
    # llf_null ≈ n_events * log(mean_rate) - n_obs * mean_rate - sum(log(y!))
    # For y in {0, 1}, log(y!) ≈ 0
    
    # More accurate: use NB log-likelihood formula
    # For y ~ NB(mu, alpha), log P(y) = ...
    # Simplified for y ∈ {0, 1}:
    
    alpha = dispersion
    mu = mean_rate
    
    # log P(Y=0) under NB
    log_p0 = -np.log(1 + alpha * mu) / alpha if alpha > 0 else -mu
    
    # log P(Y=1) under NB (approximation)
    log_p1 = np.log(mu) + log_p0 - np.log(1 + alpha * mu) if mu > 0 else -10
    
    # Weighted by counts
    n_zeros = n_obs - n_events
    null_llf = n_zeros * log_p0 + n_events * log_p1
    
    return null_llf


def add_metrics_to_model_results(model_path: Path) -> dict:
    """
    Add pseudo-R2 and null_llf to model results.
    """
    with open(model_path, 'r') as f:
        results = json.load(f)
    
    # Get existing values
    llf = results.get('llf', None)
    n_obs = results.get('n_obs', None)
    dispersion = results.get('dispersion', 0.1)
    
    if llf is None or n_obs is None:
        print("Missing llf or n_obs in model results")
        return results
    
    # Compute n_events from event rate (approximate)
    # From the diagnostics, ~0.9% are events
    event_rate = 1 - results.get('diagnostics', {}).get('zero_inflation', {}).get('observed_zeros', 0.99)
    n_events = int(n_obs * event_rate)
    
    print(f"Model stats:")
    print(f"  n_obs: {n_obs:,}")
    print(f"  n_events (approx): {n_events:,}")
    print(f"  Full model llf: {llf:.2f}")
    
    # Compute null llf
    null_llf = compute_null_llf_from_data(n_obs, n_events, dispersion)
    print(f"  Null model llf: {null_llf:.2f}")
    
    # McFadden's pseudo-R2
    if null_llf < 0:
        pseudo_r2 = 1 - (llf / null_llf)
    else:
        pseudo_r2 = 0.0
    print(f"  McFadden's pseudo-R2: {pseudo_r2:.4f}")
    
    # Typical interpretation: 0.2-0.4 is good for behavioral models
    if pseudo_r2 < 0.1:
        interpretation = "Low - model explains little variance"
    elif pseudo_r2 < 0.2:
        interpretation = "Fair - some predictive power"
    elif pseudo_r2 < 0.4:
        interpretation = "Good - typical for behavioral GLM"
    else:
        interpretation = "Excellent - strong predictive power"
    print(f"  Interpretation: {interpretation}")
    
    # Add to results
    results['null_llf'] = null_llf
    results['pseudo_r2'] = pseudo_r2
    results['pseudo_r2_interpretation'] = interpretation
    results['n_events_approx'] = n_events
    
    return results


def main():
    model_path = Path('data/model/model_results.json')
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Adding metrics to {model_path}")
    results = add_metrics_to_model_results(model_path)
    
    # Save updated results
    with open(model_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nUpdated {model_path}")


if __name__ == '__main__':
    main()




