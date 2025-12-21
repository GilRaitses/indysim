#!/usr/bin/env python3
"""
Composite Phenotype Validation & Power Curves (JAX GPU)

Validates that latent Precision and Burstiness dimensions can be reliably
recovered from sparse event data using GPU-accelerated simulation.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import time

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

OUTPUT_DIR = Path('results/composite_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
A, B, ALPHA, TAU1, TAU2 = 1.5, 12.0, 2.0, 0.63, 2.48
LED_ON, LED_OFF = 10.0, 20.0
CYCLE_DURATION = LED_ON + LED_OFF
DT = 0.05  # Coarser for speed
BETA_P, BETA_B = 0.25, 0.35
BURST_WINDOW = 0.6
BASELINES = [-3.5, -2.5, -1.5]
TARGET_EVENTS = [10, 20, 40, 80]
N_LARVAE = 1000
N_REPLICATES = 50

# =============================================================================
# JAX KERNEL FUNCTIONS
# =============================================================================

@jit
def gamma_pdf_jax(x, alpha, beta):
    log_pdf = ((alpha - 1) * jnp.log(jnp.maximum(x, 1e-10)) 
               - x / beta - alpha * jnp.log(beta) 
               - jax.scipy.special.gammaln(alpha))
    return jnp.where(x > 0, jnp.exp(log_pdf), 0.0)

@jit
def kernel_value_jax(t):
    beta1, beta2 = TAU1 / ALPHA, TAU2 / ALPHA
    return A * gamma_pdf_jax(t, ALPHA, beta1) - B * gamma_pdf_jax(t, ALPHA, beta2)

# =============================================================================
# JAX SIMULATION (vectorized over larvae)
# =============================================================================

@jit
def simulate_single_larva_jax(key, precision, burstiness, baseline, n_steps):
    """Simulate events for one larva using JAX scan."""
    times = jnp.arange(n_steps) * DT
    cycle_time = times % CYCLE_DURATION
    led_on = cycle_time < LED_ON
    time_since_onset = jnp.where(led_on, cycle_time, 0.0)
    
    b_on = baseline + BETA_P * precision
    b_off = baseline - BETA_P * precision / 2
    
    K_values = vmap(kernel_value_jax)(time_since_onset)
    log_hazard_base = jnp.where(led_on, b_on + K_values, b_off)
    
    # Simulate with scan
    def step_fn(carry, inputs):
        burst_until, key = carry
        t, log_h_base = inputs
        key, subkey = random.split(key)
        
        log_h = jnp.where(t < burst_until, log_h_base + BETA_B * burstiness, log_h_base)
        hazard = jnp.exp(log_h)
        p_event = 1 - jnp.exp(-hazard * DT)
        
        event = random.uniform(subkey) < p_event
        new_burst_until = jnp.where(event, t + BURST_WINDOW, burst_until)
        
        return (new_burst_until, key), (event, t, led_on[0])
    
    init = (-jnp.inf, key)
    _, (events, event_times, _) = lax.scan(step_fn, init, (times, log_hazard_base))
    
    return events.astype(jnp.float32), event_times, led_on

@jit
def compute_measures_jax(events, event_times, led_on, n_cycles):
    """Compute 7 behavioral measures from events."""
    cycle_time = event_times % CYCLE_DURATION
    in_on = cycle_time < LED_ON
    
    # 1. ON/OFF ratio
    on_events = jnp.sum(events * in_on)
    off_events = jnp.sum(events * (~in_on))
    on_off_ratio = (on_events + 0.5) / (off_events + 0.5)
    
    # 2. Median latency (approximate as mean of first events)
    first_event_idx = jnp.argmax(events * in_on)
    median_latency = jnp.where(on_events > 0, cycle_time[first_event_idx], LED_ON)
    
    # 3. IEI-CV
    event_indices = jnp.where(events, event_times, jnp.nan)
    valid = ~jnp.isnan(event_indices)
    sorted_times = jnp.sort(jnp.where(valid, event_indices, jnp.inf))
    ieis = jnp.diff(sorted_times[:100])  # Cap at 100
    valid_ieis = ieis[ieis < 1000]
    iei_mean = jnp.nanmean(valid_ieis)
    iei_std = jnp.nanstd(valid_ieis)
    iei_cv = iei_std / (iei_mean + 1e-6)
    
    # 4-7: Simplified versions
    total_events = jnp.sum(events)
    fano = jnp.var(events.reshape(-1, 100).sum(axis=1)) / (jnp.mean(events.reshape(-1, 100).sum(axis=1)) + 1e-6) if total_events > 0 else 1.0
    reliability = on_events / (n_cycles + 1e-6)
    habituation = 0.0  # Simplified
    phase_coherence = 0.5  # Simplified
    
    return jnp.array([on_off_ratio, median_latency, iei_cv, fano, reliability, habituation, phase_coherence])

# =============================================================================
# BATCH SIMULATION
# =============================================================================

def simulate_batch_jax(n_larvae, baseline, target_events, seed=42):
    """Run simulation for batch of larvae."""
    key = random.PRNGKey(seed)
    
    # Generate latent factors
    key, k1, k2 = random.split(key, 3)
    precision = random.normal(k1, (n_larvae,))
    burstiness = random.normal(k2, (n_larvae,))
    
    # Estimate steps needed
    base_rate = jnp.exp(baseline)
    est_events_per_cycle = base_rate * LED_ON * 0.3
    n_cycles = max(5, int(target_events / max(float(est_events_per_cycle), 0.01)))
    n_cycles = min(n_cycles, 60)
    n_steps = int(n_cycles * CYCLE_DURATION / DT)
    
    # Vectorized simulation
    keys = random.split(key, n_larvae)
    
    all_measures = []
    event_counts = []
    
    for i in range(n_larvae):
        events, event_times, led_on = simulate_single_larva_jax(
            keys[i], precision[i], burstiness[i], baseline, n_steps
        )
        measures = compute_measures_jax(events, event_times, led_on, n_cycles)
        all_measures.append(np.array(measures))
        event_counts.append(float(jnp.sum(events)))
    
    X = np.array(all_measures)
    return X, np.array(precision), np.array(burstiness), np.array(event_counts)

def recover_factors(X):
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        fa = FactorAnalysis(n_components=2, random_state=42)
        scores = fa.fit_transform(X_scaled)
        loadings = fa.components_
    except:
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_
    return scores, loadings

def compute_power(true_factor, recovered_factor, n_replicates=50, effect_size=0.5):
    n = len(true_factor)
    n_per_group = n // 2
    significant_count = 0
    for rep in range(n_replicates):
        np.random.seed(rep * 1000)
        indices = np.random.permutation(n)
        group_a = recovered_factor[indices[:n_per_group]]
        group_b = recovered_factor[indices[n_per_group:2*n_per_group]]
        group_b_shifted = group_b + effect_size
        _, p_val = stats.ttest_ind(group_a, group_b_shifted)
        if p_val < 0.05:
            significant_count += 1
    return significant_count / n_replicates

def main():
    print("="*70)
    print("COMPOSITE PHENOTYPE VALIDATION (JAX GPU)")
    print("="*70)
    print(f"\nN larvae: {N_LARVAE}, Baselines: {BASELINES}, Target events: {TARGET_EVENTS}")
    
    results = []
    start_time = time.time()
    
    for baseline in BASELINES:
        for target_events in TARGET_EVENTS:
            print(f"\nBaseline={baseline}, Target={target_events}")
            sim_start = time.time()
            
            X, true_P, true_B, event_counts = simulate_batch_jax(N_LARVAE, baseline, target_events, seed=42)
            sim_time = time.time() - sim_start
            mean_events = np.mean(event_counts)
            
            scores, loadings = recover_factors(X)
            recovered_P, recovered_B = scores[:, 0], scores[:, 1]
            
            corr_P0 = np.corrcoef(true_P, recovered_P)[0, 1]
            corr_P1 = np.corrcoef(true_P, recovered_B)[0, 1]
            corr_B0 = np.corrcoef(true_B, recovered_P)[0, 1]
            corr_B1 = np.corrcoef(true_B, recovered_B)[0, 1]
            
            if abs(corr_P0) > abs(corr_P1):
                corr_precision, corr_burstiness = abs(corr_P0), abs(corr_B1)
                best_P, best_B = recovered_P, recovered_B
            else:
                corr_precision, corr_burstiness = abs(corr_P1), abs(corr_B0)
                best_P, best_B = recovered_B, recovered_P
            
            power_P = compute_power(true_P, best_P, n_replicates=N_REPLICATES) * 100
            power_B = compute_power(true_B, best_B, n_replicates=N_REPLICATES) * 100
            
            print(f"  Time:{sim_time:.1f}s Events:{mean_events:.1f} CorrP:{corr_precision:.3f} CorrB:{corr_burstiness:.3f} PowP:{power_P:.0f}% PowB:{power_B:.0f}%")
            
            results.append({
                'baseline': baseline, 'target_events': target_events,
                'actual_events': mean_events,
                'corr_precision': corr_precision, 'corr_burstiness': corr_burstiness,
                'power_precision': power_P, 'power_burstiness': power_B
            })
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}\nTotal time: {total_time:.1f}s\n{'='*70}")
    
    # Summary tables
    print("\nSUMMARY: Correlations")
    print(f"{'Events':<10}", end="")
    for b in BASELINES: print(f"b={b:<10}", end="")
    print("\nPRECISION:")
    for t in TARGET_EVENTS:
        print(f"{t:<10}", end="")
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target_events']==t), None)
            print(f"{r['corr_precision']:<10.3f}" if r else "", end="")
        print()
    print("BURSTINESS:")
    for t in TARGET_EVENTS:
        print(f"{t:<10}", end="")
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target_events']==t), None)
            print(f"{r['corr_burstiness']:<10.3f}" if r else "", end="")
        print()
    
    print("\nSUMMARY: Power (%)")
    print(f"{'Events':<10}", end="")
    for b in BASELINES: print(f"b={b:<10}", end="")
    print("\nPRECISION:")
    for t in TARGET_EVENTS:
        print(f"{t:<10}", end="")
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target_events']==t), None)
            print(f"{r['power_precision']:<10.0f}" if r else "", end="")
        print()
    print("BURSTINESS:")
    for t in TARGET_EVENTS:
        print(f"{t:<10}", end="")
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target_events']==t), None)
            print(f"{r['power_burstiness']:<10.0f}" if r else "", end="")
        print()
    
    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {-3.5: 'coral', -2.5: 'gold', -1.5: 'forestgreen'}
    for i, (metric, title) in enumerate([
        ('corr_precision', 'Precision Recovery'), ('corr_burstiness', 'Burstiness Recovery'),
        ('power_precision', 'Power for Precision'), ('power_burstiness', 'Power for Burstiness')
    ]):
        ax = axes[i//2, i%2]
        for b in BASELINES:
            subset = [r for r in results if r['baseline'] == b]
            ax.plot([r['target_events'] for r in subset], [r[metric] for r in subset], 
                   'o-', label=f'b={b}', color=colors[b], lw=2, ms=8)
        ax.axhline(0.8 if 'corr' in metric else 80, color='red', ls='--', alpha=0.5)
        ax.set_ylim(0, 1 if 'corr' in metric else 100)
        ax.set_ylabel('Correlation' if 'corr' in metric else 'Power (%)')
        ax.set_xlabel('Events/Larva')
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'composite_validation_results.png', dpi=150)
    print(f"\nFigure saved: {OUTPUT_DIR / 'composite_validation_results.png'}")

if __name__ == '__main__':
    main()
