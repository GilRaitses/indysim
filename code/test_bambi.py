#!/usr/bin/env python3
"""
Test Bambi NB-GLMM setup.

Run with: source .venv-larvaworld/bin/activate && python scripts/test_bambi.py
"""

import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az

print("Testing Bambi NB-GLMM...")

# Create minimal test data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'events': np.random.negative_binomial(n=2, p=0.3, size=n),
    'x1': np.random.randn(n),
    'x2': np.random.choice([0, 1], size=n),
    'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=n)
})

print(f"Test data: {len(df)} rows")
print(f"Events range: {df.events.min()} - {df.events.max()}")

# Fit simple NB-GLMM
print("\nFitting NB-GLMM with random intercepts...")
model = bmb.Model(
    "events ~ x1 + x2 + (1|group)",
    df,
    family="negativebinomial"
)

# Quick fit for testing (fewer draws)
idata = model.fit(draws=500, tune=500, random_seed=42, progressbar=False)

print("\nModel summary:")
summary = az.summary(idata, var_names=["Intercept", "x1", "x2"])
print(summary)

print("\nBambi NB-GLMM test PASSED!")
