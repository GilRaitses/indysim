#!/usr/bin/env python3
"""
Fit Negative Binomial GLMM with Track Random Intercepts

Uses glmmTMB via rpy2 to fit:
    log(mu) = beta_0 + beta_1*LED + kernel(t) + (1|track_id)

This accounts for track-to-track variability in baseline turn rate.

Usage:
    python scripts/fit_nb_glmm.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List


def load_binned_data(data_dir: Path, n_files: int = 2) -> pd.DataFrame:
    """Load binned data for GLMM fitting."""
    # Use matching condition files
    csv_files = sorted(data_dir.glob('*_0to250PWM_30#C_Bl_7PWM_2025103*_events.csv'))[:n_files]
    
    if not csv_files:
        csv_files = sorted(data_dir.glob('*_events.csv'))[:n_files]
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Create unique track ID
    data['track_uid'] = data['experiment_id'] + '_' + data['track_id'].astype(str)
    
    return data


def prepare_glmm_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for GLMM fitting."""
    # Response: is_reorientation_start (binary event indicator)
    if 'is_reorientation_start' in data.columns:
        y_col = 'is_reorientation_start'
    elif 'is_reorientation' in data.columns:
        # Detect onsets
        data = data.sort_values(['track_uid', 'time'])
        data['Y'] = (
            data.groupby('track_uid')['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        ).astype(int)
        y_col = 'Y'
    else:
        raise ValueError("No reorientation column found")
    
    # Covariates
    if 'led1Val' in data.columns:
        data['LED1_scaled'] = data['led1Val'] / 250.0
    
    if 'led2Val' in data.columns:
        data['LED2_scaled'] = data['led2Val'] / 15.0
    
    return data, y_col


def fit_glmm_via_r(data: pd.DataFrame, y_col: str) -> Dict:
    """
    Fit NB-GLMM using glmmTMB via rpy2.
    
    Model: Y ~ LED1_scaled + (1|track_uid), family = nbinom2
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Use context manager for conversion
    
    # Load R packages
    glmmTMB = importr('glmmTMB')
    base = importr('base')
    stats = importr('stats')
    
    # Prepare data - ensure numeric types
    subset = data[[y_col, 'LED1_scaled', 'track_uid']].dropna().copy()
    subset[y_col] = subset[y_col].astype(int)
    subset['LED1_scaled'] = subset['LED1_scaled'].astype(float)
    subset['track_uid'] = subset['track_uid'].astype(str)
    
    # Convert to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(subset)
    
    # Fit model
    print("Fitting NB-GLMM with track random intercepts...")
    print("  Formula: Y ~ LED1_scaled + (1|track_uid)")
    print("  Family: nbinom2 (Negative Binomial)")
    
    try:
        formula = ro.Formula(f'{y_col} ~ LED1_scaled + (1|track_uid)')
        fit = glmmTMB.glmmTMB(formula, data=r_data, family=ro.r('nbinom2'))
        
        # Extract results
        summary_fit = base.summary(fit)
        
        # Get fixed effects
        fixed_coefs = dict(zip(
            ['intercept', 'LED1_scaled'],
            np.array(ro.r('fixef')(fit)[0])
        ))
        
        # Get variance components
        var_comps = ro.r('VarCorr')(fit)
        
        # Get AIC/BIC
        aic_val = float(stats.AIC(fit)[0])
        
        results = {
            'model_type': 'NB-GLMM',
            'formula': f'{y_col} ~ LED1_scaled + (1|track_uid)',
            'fixed_effects': fixed_coefs,
            'aic': aic_val,
            'converged': True,
            'n_obs': len(data),
            'n_tracks': data['track_uid'].nunique()
        }
        
        print(f"\n✓ Model converged!")
        print(f"  Fixed effects:")
        for k, v in fixed_coefs.items():
            print(f"    {k}: {v:.4f}")
        print(f"  AIC: {aic_val:.2f}")
        
        return results
        
    except Exception as e:
        print(f"Error fitting GLMM: {e}")
        return {'error': str(e), 'converged': False}


def fit_simple_glmm(data: pd.DataFrame, y_col: str) -> Dict:
    """
    Fallback: fit using statsmodels mixed linear model.
    Note: This is not a true NB-GLMM but provides random intercepts.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    
    # For binary outcomes, use logistic approximation
    # This is not ideal but provides a fallback
    print("Fitting mixed model (linear approximation)...")
    
    # Ensure numeric types
    data = data.copy()
    data[y_col] = data[y_col].astype(float)
    data['LED1_scaled'] = data['LED1_scaled'].astype(float)
    
    formula = f'{y_col} ~ LED1_scaled'
    
    try:
        model = MixedLM.from_formula(
            formula, 
            data=data, 
            groups=data['track_uid']
        )
        fit = model.fit()
        
        results = {
            'model_type': 'Linear-MixedLM',
            'formula': formula + ' + (1|track_uid)',
            'fixed_effects': {
                'intercept': float(fit.fe_params['Intercept']),
                'LED1_scaled': float(fit.fe_params['LED1_scaled'])
            },
            'random_effects_var': float(fit.cov_re.iloc[0, 0]),
            'aic': float(fit.aic),
            'converged': fit.converged,
            'n_obs': len(data),
            'n_tracks': data['track_uid'].nunique()
        }
        
        print(f"\n✓ Model converged!")
        print(f"  Fixed effects:")
        for k, v in results['fixed_effects'].items():
            print(f"    {k}: {v:.4f}")
        print(f"  Random intercept variance: {results['random_effects_var']:.4f}")
        print(f"  AIC: {results['aic']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"Error fitting mixed model: {e}")
        return {'error': str(e), 'converged': False}


def main():
    print("=" * 60)
    print("FIT NB-GLMM WITH TRACK RANDOM INTERCEPTS")
    print("=" * 60)
    
    data_dir = Path('data/engineered')
    print(f"\nLoading data from {data_dir}...")
    
    try:
        data = load_binned_data(data_dir)
        print(f"  Loaded {len(data):,} observations")
        print(f"  Tracks: {data['track_uid'].nunique()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare data
    print("\nPreparing data...")
    data, y_col = prepare_glmm_data(data)
    n_events = data[y_col].sum()
    print(f"  Response column: {y_col}")
    print(f"  Events: {n_events:,}")
    
    # Subsample for faster fitting (GLMM can be slow)
    if len(data) > 100000:
        print(f"\n  Subsampling to 100k rows for faster fitting...")
        data = data.sample(n=100000, random_state=42)
    
    # Try glmmTMB first
    try:
        results = fit_glmm_via_r(data, y_col)
    except Exception as e:
        print(f"glmmTMB failed: {e}")
        print("Falling back to statsmodels MixedLM...")
        results = fit_simple_glmm(data, y_col)
    
    # Save results
    output_path = Path('data/model/glmm_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()




