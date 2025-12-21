#!/usr/bin/env python3
"""
Fit Negative Binomial GLMM with random track intercepts using Bambi.

This script refits the factorial model with (1|track) random intercepts
to properly account for the hierarchical structure of the data.

Usage:
    source .venv-glmm/bin/activate
    python scripts/fit_glmm.py
"""

import json
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
DESIGN_MATRIX_PATH = DATA_DIR / "processed" / "factorial_design_matrix.parquet"

# Sampling parameters
SAMPLE_FRAC = 0.02  # Use 2% of non-events to keep runtime reasonable
DRAWS = 1000
TUNE = 500
CHAINS = 1  # Single chain to avoid multiprocessing issues
TARGET_ACCEPT = 0.9
CORES = 1  # Single core


def load_data(sample_frac: float = SAMPLE_FRAC) -> pd.DataFrame:
    """Load design matrix with stratified sampling."""
    print(f"Loading data from {DESIGN_MATRIX_PATH}...")
    df = pd.read_parquet(DESIGN_MATRIX_PATH)
    
    print(f"Full dataset: {len(df):,} rows, {df['events'].sum():,} events")
    
    # Stratified sampling: keep all events, sample non-events
    events_df = df[df["events"] == 1]
    non_events_df = df[df["events"] == 0].sample(frac=sample_frac, random_state=42)
    
    df_sampled = pd.concat([events_df, non_events_df]).sort_index()
    
    print(f"Sampled dataset: {len(df_sampled):,} rows, {df_sampled['events'].sum():,} events")
    print(f"Sampling kept {100*len(df_sampled)/len(df):.1f}% of data")
    
    return df_sampled


def prepare_data_for_bambi(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe with proper column names for Bambi formula."""
    # Rename columns to be Bambi-friendly (no special characters)
    df = df.copy()
    
    # Create track ID as string for random effect
    if "track" not in df.columns:
        # Create track ID from experiment + track_id if available
        if "experiment" in df.columns and "track_id" in df.columns:
            df["track"] = df["experiment"].astype(str) + "_" + df["track_id"].astype(str)
        else:
            # Use index-based track assignment
            df["track"] = "track_" + (df.index // 1000).astype(str)
    
    # Ensure track is categorical
    df["track"] = df["track"].astype("category")
    
    # Rename indicator columns if needed
    rename_map = {
        "I": "intensity",
        "T": "cycling", 
        "IT": "int_cyc",
        "K_on": "k_on",
        "I_K_on": "int_k_on",
        "T_K_on": "cyc_k_on",
        "K_off": "k_off"
    }
    
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    return df


def fit_glmm(df: pd.DataFrame) -> tuple:
    """Fit NB-GLMM with random track intercepts."""
    print("\nFitting NB-GLMM with random track intercepts...")
    print(f"  Draws: {DRAWS}, Tune: {TUNE}, Chains: {CHAINS}")
    
    # Prepare data
    df = prepare_data_for_bambi(df)
    
    # Check required columns
    required = ["events", "intensity", "cycling", "k_on", "k_off", "track"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    n_tracks = df["track"].nunique()
    print(f"  Tracks: {n_tracks}")
    
    # Define model
    # Fixed effects: intensity, cycling, interaction, kernels
    # Random effects: (1|track) - random intercept per track
    formula = "events ~ intensity + cycling + intensity:cycling + k_on + intensity:k_on + cycling:k_on + k_off + (1|track)"
    
    print(f"  Formula: {formula}")
    
    model = bmb.Model(
        formula,
        df,
        family="negativebinomial"
    )
    
    print("\nModel summary:")
    print(model)
    
    # Fit model
    print("\nSampling (this may take several minutes)...")
    idata = model.fit(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CORES,
        target_accept=TARGET_ACCEPT,
        random_seed=42,
        progressbar=True
    )
    
    return model, idata


def extract_results(model, idata) -> dict:
    """Extract coefficient estimates and credible intervals."""
    summary = az.summary(idata, hdi_prob=0.95)
    
    results = {
        "coefficients": {},
        "random_effects": {},
        "diagnostics": {}
    }
    
    # Extract fixed effects
    fixed_params = [
        "Intercept", "intensity", "cycling", "intensity:cycling",
        "k_on", "intensity:k_on", "cycling:k_on", "k_off"
    ]
    
    for param in fixed_params:
        if param in summary.index:
            row = summary.loc[param]
            results["coefficients"][param] = {
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "hdi_2.5%": float(row["hdi_2.5%"]),
                "hdi_97.5%": float(row["hdi_97.5%"]),
                "ess_bulk": float(row["ess_bulk"]),
                "ess_tail": float(row["ess_tail"]),
                "r_hat": float(row["r_hat"])
            }
    
    # Extract random effect variance
    if "1|track_sigma" in summary.index:
        row = summary.loc["1|track_sigma"]
        results["random_effects"]["track_sigma"] = {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_2.5%": float(row["hdi_2.5%"]),
            "hdi_97.5%": float(row["hdi_97.5%"])
        }
    
    # Dispersion parameter
    if "events_alpha" in summary.index:
        row = summary.loc["events_alpha"]
        results["diagnostics"]["nb_alpha"] = {
            "mean": float(row["mean"]),
            "sd": float(row["sd"])
        }
    
    # Convergence diagnostics
    results["diagnostics"]["all_r_hat_ok"] = all(
        summary["r_hat"] < 1.05
    ) if "r_hat" in summary.columns else False
    
    results["diagnostics"]["min_ess_bulk"] = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else 0
    
    return results


def compare_with_fixed_effects(glmm_results: dict) -> dict:
    """Compare GLMM results with fixed-effects GLM."""
    # Load fixed-effects results
    fe_path = MODEL_DIR / "factorial_model_results.json"
    if not fe_path.exists():
        print("Warning: Fixed-effects results not found")
        return {}
    
    with open(fe_path) as f:
        fe_results = json.load(f)
    
    comparison = {}
    
    # Map parameter names
    param_map = {
        "Intercept": "beta_0",
        "intensity": "beta_I",
        "cycling": "beta_T",
        "intensity:cycling": "beta_IT",
        "k_on": "alpha",
        "intensity:k_on": "alpha_I",
        "cycling:k_on": "alpha_T",
        "k_off": "gamma"
    }
    
    for glmm_name, fe_name in param_map.items():
        if glmm_name in glmm_results["coefficients"] and fe_name in fe_results["coefficients"]:
            glmm_val = glmm_results["coefficients"][glmm_name]["mean"]
            fe_val = fe_results["coefficients"][fe_name]["mean"]
            
            comparison[fe_name] = {
                "fixed_effects": fe_val,
                "glmm": glmm_val,
                "difference": glmm_val - fe_val,
                "pct_change": 100 * (glmm_val - fe_val) / abs(fe_val) if fe_val != 0 else 0
            }
    
    return comparison


def main():
    """Main entry point."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df = load_data(sample_frac=SAMPLE_FRAC)
    
    # Fit GLMM
    model, idata = fit_glmm(df)
    
    # Extract results
    print("\nExtracting results...")
    results = extract_results(model, idata)
    
    # Add metadata
    results["metadata"] = {
        "n_observations": len(df),
        "n_events": int(df["events"].sum()),
        "n_tracks": int(df["track"].nunique()) if "track" in df.columns else 0,
        "sample_frac": SAMPLE_FRAC,
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS
    }
    
    # Compare with fixed effects
    print("\nComparing with fixed-effects model...")
    comparison = compare_with_fixed_effects(results)
    results["comparison"] = comparison
    
    # Save results
    results_path = MODEL_DIR / "glmm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save trace
    trace_path = MODEL_DIR / "glmm_trace.nc"
    idata.to_netcdf(trace_path)
    print(f"Trace saved to {trace_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("GLMM RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nFixed Effects (Posterior Means):")
    for param, vals in results["coefficients"].items():
        print(f"  {param:20s}: {vals['mean']:8.4f} [{vals['hdi_2.5%']:.4f}, {vals['hdi_97.5%']:.4f}]")
    
    if "track_sigma" in results.get("random_effects", {}):
        ts = results["random_effects"]["track_sigma"]
        print(f"\nRandom Effect SD (track): {ts['mean']:.4f} [{ts['hdi_2.5%']:.4f}, {ts['hdi_97.5%']:.4f}]")
    
    print("\nComparison with Fixed-Effects Model:")
    for param, vals in comparison.items():
        print(f"  {param:10s}: FE={vals['fixed_effects']:8.4f}  GLMM={vals['glmm']:8.4f}  Δ={vals['pct_change']:+.1f}%")
    
    print("\nConvergence Diagnostics:")
    print(f"  All R-hat < 1.05: {results['diagnostics']['all_r_hat_ok']}")
    print(f"  Min ESS (bulk): {results['diagnostics']['min_ess_bulk']:.0f}")
    
    return results


if __name__ == "__main__":
    main()


