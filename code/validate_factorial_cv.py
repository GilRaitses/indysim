#!/usr/bin/env python3
"""
Factorial Model Cross-Validation

Leave-one-experiment-out cross-validation for the factorial NB-GLM.

Usage:
    python scripts/validate_factorial_cv.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
except ImportError:
    print("Error: statsmodels required")
    exit(1)


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 1.0) -> Dict:
    """
    Fit on train, predict on test.
    
    Returns dict with predictions and metrics.
    """
    # Build design matrix
    feature_cols = ['I', 'T', 'IT', 'K_on', 'I_K_on', 'T_K_on', 'K_off']
    
    X_train = train_df[feature_cols].values
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    y_train = train_df['events'].values
    
    X_test = test_df[feature_cols].values
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    y_test = test_df['events'].values
    
    # Fit model
    try:
        model = GLM(y_train, X_train, family=NegativeBinomial(alpha=alpha))
        fit = model.fit()
        
        # Predict on test
        eta_test = X_test @ fit.params
        predicted_hazard = np.exp(eta_test)
        
        # Metrics
        empirical_events = y_test.sum()
        predicted_events = predicted_hazard.sum()
        rate_ratio = predicted_events / empirical_events if empirical_events > 0 else np.nan
        
        return {
            'empirical_events': int(empirical_events),
            'predicted_events': float(predicted_events),
            'rate_ratio': float(rate_ratio),
            'n_test_frames': len(test_df),
            'coefficients': fit.params.tolist(),
            'converged': bool(fit.converged)
        }
    except Exception as e:
        return {
            'empirical_events': int(y_test.sum()),
            'predicted_events': 0.0,
            'rate_ratio': np.nan,
            'n_test_frames': len(test_df),
            'coefficients': [],
            'error': str(e)
        }


def leave_one_experiment_out_cv(df: pd.DataFrame) -> Dict:
    """
    Leave-one-experiment-out cross-validation.
    
    Returns dict with per-experiment results and summary.
    """
    experiments = df['experiment'].unique()
    n_experiments = len(experiments)
    
    print(f"\nRunning {n_experiments}-fold leave-one-experiment-out CV...")
    print("=" * 60)
    
    results = {}
    rate_ratios = []
    
    for i, exp in enumerate(experiments):
        # Split
        test_df = df[df['experiment'] == exp]
        train_df = df[df['experiment'] != exp]
        
        # Get condition for this experiment
        condition = test_df['condition'].iloc[0]
        
        # Fit and predict
        exp_results = fit_and_predict(train_df, test_df)
        exp_results['condition'] = condition
        
        results[exp] = exp_results
        rate_ratios.append(exp_results['rate_ratio'])
        
        # Progress
        rr = exp_results['rate_ratio']
        status = 'PASS' if 0.8 <= rr <= 1.25 else 'FAIL'
        print(f"[{i+1:2d}/{n_experiments}] {exp[:40]:<40} RR={rr:.3f} {status}")
    
    # Summary statistics
    rate_ratios = np.array(rate_ratios)
    valid_rr = rate_ratios[~np.isnan(rate_ratios)]
    
    pass_count = np.sum((valid_rr >= 0.8) & (valid_rr <= 1.25))
    pass_rate = pass_count / len(valid_rr) * 100
    
    summary = {
        'n_experiments': n_experiments,
        'mean_rate_ratio': float(np.mean(valid_rr)),
        'std_rate_ratio': float(np.std(valid_rr)),
        'median_rate_ratio': float(np.median(valid_rr)),
        'min_rate_ratio': float(np.min(valid_rr)),
        'max_rate_ratio': float(np.max(valid_rr)),
        'pass_count': int(pass_count),
        'pass_rate': float(pass_rate)
    }
    
    return {
        'per_experiment': results,
        'summary': summary
    }


def per_condition_cv_summary(cv_results: Dict, df: pd.DataFrame) -> Dict:
    """
    Summarize CV results per condition.
    """
    conditions = df['condition'].unique()
    
    condition_summary = {}
    
    for condition in conditions:
        # Get experiments for this condition
        cond_exps = [exp for exp, res in cv_results['per_experiment'].items() 
                     if res['condition'] == condition]
        
        rrs = [cv_results['per_experiment'][exp]['rate_ratio'] for exp in cond_exps]
        
        condition_summary[condition] = {
            'n_experiments': len(cond_exps),
            'mean_rate_ratio': float(np.mean(rrs)),
            'std_rate_ratio': float(np.std(rrs)) if len(rrs) > 1 else 0.0,
            'experiments': cond_exps
        }
    
    return condition_summary


def main():
    print("=" * 70)
    print("FACTORIAL MODEL CROSS-VALIDATION")
    print("=" * 70)
    
    # Load design matrix with sampling to avoid memory issues
    dm_path = Path('data/processed/factorial_design_matrix.parquet')
    if not dm_path.exists():
        print(f"Error: Design matrix not found at {dm_path}")
        print("Run fit_factorial_model.py first.")
        return
    
    df = pd.read_parquet(dm_path)
    
    # Sample to reduce memory (keep all events, sample non-events)
    print(f"\nFull design matrix: {len(df):,} rows, {df['events'].sum():,} events")
    
    # Stratified sampling: keep all events, sample 10% of non-events
    events_df = df[df['events'] == 1]
    non_events_df = df[df['events'] == 0].sample(frac=0.1, random_state=42)
    df = pd.concat([events_df, non_events_df]).sort_index()
    
    print(f"Sampled for CV: {len(df):,} rows, {df['events'].sum():,} events")
    print(f"Experiments: {df['experiment'].nunique()}")
    print(f"Conditions: {df['condition'].nunique()}")
    
    # Run CV
    cv_results = leave_one_experiment_out_cv(df)
    
    # Per-condition summary
    condition_summary = per_condition_cv_summary(cv_results, df)
    cv_results['per_condition'] = condition_summary
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    
    summary = cv_results['summary']
    print(f"\nOverall:")
    print(f"  Mean rate ratio: {summary['mean_rate_ratio']:.3f} ± {summary['std_rate_ratio']:.3f}")
    print(f"  Range: [{summary['min_rate_ratio']:.3f}, {summary['max_rate_ratio']:.3f}]")
    print(f"  Pass rate: {summary['pass_count']}/{summary['n_experiments']} ({summary['pass_rate']:.1f}%)")
    
    print(f"\nPer condition:")
    print(f"{'Condition':<25} {'N':>3} {'Mean RR':>10} {'Std':>8}")
    print("-" * 50)
    for cond, stats in condition_summary.items():
        print(f"{cond:<25} {stats['n_experiments']:>3} {stats['mean_rate_ratio']:>10.3f} {stats['std_rate_ratio']:>8.3f}")
    
    # Save results
    output_path = Path('data/validation/factorial_cv_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\nSaved CV results to {output_path}")
    
    return cv_results


if __name__ == '__main__':
    main()
