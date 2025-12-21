#!/usr/bin/env python3
"""
DOE Analysis for Larval Behavior Experiments

Analyzes factorial effects of LED intensity and timing on behavioral responses.
Computes main effects, interactions, and ranks conditions for subgroup discrimination.

Usage:
    python scripts/doe_analysis.py --input data/processed/consolidated_dataset.h5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_events_from_h5(h5_path: Path) -> pd.DataFrame:
    """Load event data from consolidated H5 file."""
    print(f"Loading events from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'events' not in f:
            raise ValueError("No 'events' group in H5 file")
        
        grp = f['events']
        data = {}
        for key in grp.keys():
            arr = grp[key][:]
            if arr.dtype.kind == 'S':
                arr = arr.astype(str)
            data[key] = arr
        
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df):,} rows")
    return df


# =============================================================================
# CONDITION EXTRACTION
# =============================================================================

def extract_experimental_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract experimental conditions from experiment IDs and LED values.
    """
    conditions = []
    
    for exp_id in df['experiment_id'].unique():
        exp_df = df[df['experiment_id'] == exp_id]
        
        # LED1 levels
        led1_max = exp_df['led1Val'].max()
        led1_min = exp_df['led1Val'].min()
        
        # LED2 levels
        led2_max = exp_df['led2Val'].max() if 'led2Val' in exp_df.columns else 0
        
        # Classify LED1 level
        if led1_max > 200:
            led1_level = 'high'
        elif led1_max > 100:
            led1_level = 'medium'
        else:
            led1_level = 'low'
        
        # Classify LED2 level
        led2_level = 'high' if led2_max > 7 else 'low'
        
        conditions.append({
            'experiment_id': str(exp_id),
            'led1_max': led1_max,
            'led1_min': led1_min,
            'led2_max': led2_max,
            'led1_level': led1_level,
            'led2_level': led2_level,
            'n_tracks': exp_df['track_id'].nunique(),
            'duration': exp_df['time'].max() - exp_df['time'].min()
        })
    
    return pd.DataFrame(conditions)


# =============================================================================
# KPI COMPUTATION
# =============================================================================

def compute_turn_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute turn rate per track per experiment."""
    results = []
    
    # Detect reorientation starts
    if 'is_reorientation_start' in df.columns:
        event_col = 'is_reorientation_start'
    elif 'is_reorientation' in df.columns:
        df = df.sort_values(['experiment_id', 'track_id', 'time'])
        df['reo_start'] = (
            df.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
        event_col = 'reo_start'
    else:
        return pd.DataFrame()
    
    for (exp_id, track_id), group in df.groupby(['experiment_id', 'track_id']):
        duration_min = (group['time'].max() - group['time'].min()) / 60.0
        n_events = group[event_col].sum()
        
        if duration_min > 0:
            rate = n_events / duration_min
            
            # Compute stimulus-specific rates
            if 'led1Val' in group.columns:
                stim_on = group[group['led1Val'] > 0]
                stim_off = group[group['led1Val'] == 0]
                
                stim_on_dur = len(stim_on) / len(group) * duration_min if len(group) > 0 else 0
                stim_off_dur = duration_min - stim_on_dur
                
                rate_on = stim_on[event_col].sum() / stim_on_dur if stim_on_dur > 0 else 0
                rate_off = stim_off[event_col].sum() / stim_off_dur if stim_off_dur > 0 else 0
            else:
                rate_on = rate_off = rate
            
            results.append({
                'experiment_id': str(exp_id),
                'track_id': track_id,
                'duration_min': duration_min,
                'n_events': n_events,
                'turn_rate': rate,
                'rate_stim_on': rate_on,
                'rate_stim_off': rate_off,
                'response_magnitude': rate_on - rate_off
            })
    
    return pd.DataFrame(results)


# =============================================================================
# FACTORIAL ANALYSIS
# =============================================================================

def compute_factorial_effects(
    conditions: pd.DataFrame,
    turn_rates: pd.DataFrame
) -> dict:
    """
    Compute main effects and interactions for 2^k factorial design.
    """
    # Merge conditions with turn rates
    merged = turn_rates.merge(conditions[['experiment_id', 'led1_level', 'led2_level']], 
                               on='experiment_id')
    
    # Code factors as -1/+1
    merged['A'] = merged['led1_level'].map({'low': -1, 'medium': 0, 'high': 1})
    merged['B'] = merged['led2_level'].map({'low': -1, 'high': 1})
    
    # Remove medium LED1 levels for clean 2^2 design
    factorial_df = merged[merged['A'].isin([-1, 1])].copy()
    
    if len(factorial_df) == 0:
        return {'error': 'No factorial data available'}
    
    results = {}
    
    # Main effect of LED1 (A)
    high_a = factorial_df[factorial_df['A'] == 1]['turn_rate']
    low_a = factorial_df[factorial_df['A'] == -1]['turn_rate']
    
    if len(high_a) > 0 and len(low_a) > 0:
        effect_a = high_a.mean() - low_a.mean()
        t_stat, p_val = stats.ttest_ind(high_a, low_a)
        ci = stats.t.interval(0.95, len(high_a) + len(low_a) - 2,
                              loc=effect_a,
                              scale=np.sqrt(high_a.var()/len(high_a) + low_a.var()/len(low_a)))
        results['LED1_effect'] = {
            'effect': effect_a,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    # Main effect of LED2 (B)
    high_b = factorial_df[factorial_df['B'] == 1]['turn_rate']
    low_b = factorial_df[factorial_df['B'] == -1]['turn_rate']
    
    if len(high_b) > 0 and len(low_b) > 0:
        effect_b = high_b.mean() - low_b.mean()
        t_stat, p_val = stats.ttest_ind(high_b, low_b)
        ci = stats.t.interval(0.95, len(high_b) + len(low_b) - 2,
                              loc=effect_b,
                              scale=np.sqrt(high_b.var()/len(high_b) + low_b.var()/len(low_b)))
        results['LED2_effect'] = {
            'effect': effect_b,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    # Interaction effect (A x B)
    # Effect of A at high B - Effect of A at low B
    high_b_df = factorial_df[factorial_df['B'] == 1]
    low_b_df = factorial_df[factorial_df['B'] == -1]
    
    if len(high_b_df) > 0 and len(low_b_df) > 0:
        effect_a_at_high_b = (high_b_df[high_b_df['A'] == 1]['turn_rate'].mean() - 
                              high_b_df[high_b_df['A'] == -1]['turn_rate'].mean())
        effect_a_at_low_b = (low_b_df[low_b_df['A'] == 1]['turn_rate'].mean() - 
                             low_b_df[low_b_df['A'] == -1]['turn_rate'].mean())
        
        interaction = effect_a_at_high_b - effect_a_at_low_b
        
        results['LED1xLED2_interaction'] = {
            'effect': interaction,
            'effect_a_at_high_b': effect_a_at_high_b,
            'effect_a_at_low_b': effect_a_at_low_b
        }
    
    return results


# =============================================================================
# CONDITION RANKING
# =============================================================================

def rank_conditions_by_response(
    conditions: pd.DataFrame,
    turn_rates: pd.DataFrame
) -> pd.DataFrame:
    """
    Rank experimental conditions by response magnitude.
    """
    # Aggregate by experiment
    exp_stats = turn_rates.groupby('experiment_id').agg({
        'turn_rate': 'mean',
        'response_magnitude': 'mean',
        'rate_stim_on': 'mean',
        'rate_stim_off': 'mean',
        'n_events': 'sum'
    }).reset_index()
    
    # Merge with conditions
    ranked = exp_stats.merge(conditions, on='experiment_id')
    
    # Compute Cohen's d for response magnitude
    overall_mean = ranked['response_magnitude'].mean()
    overall_std = ranked['response_magnitude'].std()
    
    if overall_std > 0:
        ranked['cohens_d'] = (ranked['response_magnitude'] - overall_mean) / overall_std
    else:
        ranked['cohens_d'] = 0
    
    # Rank by absolute response magnitude
    ranked['rank'] = ranked['response_magnitude'].abs().rank(ascending=False)
    
    return ranked.sort_values('rank')


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_interaction(
    conditions: pd.DataFrame,
    turn_rates: pd.DataFrame,
    output_path: Path
):
    """Plot interaction plot for LED1 x LED2."""
    # Merge
    merged = turn_rates.merge(conditions[['experiment_id', 'led1_level', 'led2_level']], 
                               on='experiment_id')
    
    # Aggregate
    summary = merged.groupby(['led1_level', 'led2_level'])['turn_rate'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    led1_levels = ['low', 'high']
    led2_levels = summary['led2_level'].unique()
    
    colors = {'low': 'blue', 'high': 'red'}
    markers = {'low': 'o', 'high': 's'}
    
    for led2 in led2_levels:
        subset = summary[summary['led2_level'] == led2]
        x_vals = [led1_levels.index(l) if l in led1_levels else 0 for l in subset['led1_level']]
        
        ax.errorbar(x_vals, subset['mean'], yerr=1.96*subset['se'],
                    marker=markers.get(led2, 'o'), color=colors.get(led2, 'gray'),
                    label=f'LED2={led2}', capsize=5, linewidth=2, markersize=10)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low', 'High'])
    ax.set_xlabel('LED1 Intensity')
    ax.set_ylabel('Turn Rate (events/min)')
    ax.set_title('LED1 x LED2 Interaction Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    conditions: pd.DataFrame,
    factorial_effects: dict,
    ranked_conditions: pd.DataFrame,
    output_path: Path
):
    """Generate DOE analysis report."""
    lines = [
        "# DOE Analysis Report",
        "",
        "## Experimental Conditions",
        "",
        f"**Number of experiments:** {len(conditions)}",
        "",
        "| Experiment | LED1 Max | LED2 Max | Level | Tracks |",
        "|------------|----------|----------|-------|--------|",
    ]
    
    for _, row in conditions.iterrows():
        lines.append(f"| {row['experiment_id'][:25]} | {row['led1_max']} | {row['led2_max']} | "
                     f"{row['led1_level']}/{row['led2_level']} | {row['n_tracks']} |")
    
    lines.extend([
        "",
        "## Factorial Effects",
        "",
        "| Effect | Estimate | 95% CI | p-value | Significant |",
        "|--------|----------|--------|---------|-------------|",
    ])
    
    for name, effect in factorial_effects.items():
        if isinstance(effect, dict) and 'effect' in effect:
            est = effect['effect']
            ci_l = effect.get('ci_lower', np.nan)
            ci_u = effect.get('ci_upper', np.nan)
            p = effect.get('p_value', np.nan)
            sig = 'Yes' if effect.get('significant', False) else 'No'
            lines.append(f"| {name} | {est:.3f} | [{ci_l:.3f}, {ci_u:.3f}] | {p:.4f} | {sig} |")
    
    lines.extend([
        "",
        "## Condition Ranking by Response Magnitude",
        "",
        "| Rank | Experiment | Response | Cohen's d |",
        "|------|------------|----------|-----------|",
    ])
    
    for _, row in ranked_conditions.head(10).iterrows():
        lines.append(f"| {int(row['rank'])} | {row['experiment_id'][:25]} | "
                     f"{row['response_magnitude']:.3f} | {row['cohens_d']:.2f} |")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    
    # Add interpretation
    if 'LED1_effect' in factorial_effects and factorial_effects['LED1_effect'].get('significant', False):
        eff = factorial_effects['LED1_effect']['effect']
        lines.append(f"- **LED1 main effect is significant** (effect = {eff:.3f} turns/min)")
    else:
        lines.append("- LED1 main effect is not significant")
    
    if 'LED2_effect' in factorial_effects and factorial_effects['LED2_effect'].get('significant', False):
        eff = factorial_effects['LED2_effect']['effect']
        lines.append(f"- **LED2 main effect is significant** (effect = {eff:.3f} turns/min)")
    else:
        lines.append("- LED2 main effect is not significant")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DOE analysis for behavioral experiments')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to consolidated H5 file')
    parser.add_argument('--output', type=str, default='data/analysis/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_events_from_h5(input_path)
    
    print("\n=== Extracting Conditions ===")
    conditions = extract_experimental_conditions(df)
    print(f"  {len(conditions)} experiments")
    print(f"  LED1 levels: {conditions['led1_level'].value_counts().to_dict()}")
    print(f"  LED2 levels: {conditions['led2_level'].value_counts().to_dict()}")
    
    print("\n=== Computing Turn Rates ===")
    turn_rates = compute_turn_rates(df)
    print(f"  {len(turn_rates)} track-level observations")
    print(f"  Mean turn rate: {turn_rates['turn_rate'].mean():.3f} ± {turn_rates['turn_rate'].std():.3f}")
    
    print("\n=== Factorial Analysis ===")
    factorial_effects = compute_factorial_effects(conditions, turn_rates)
    
    for name, effect in factorial_effects.items():
        if isinstance(effect, dict) and 'effect' in effect:
            sig = '*' if effect.get('significant', False) else ''
            print(f"  {name}: {effect['effect']:.3f} {sig}")
    
    print("\n=== Condition Ranking ===")
    ranked = rank_conditions_by_response(conditions, turn_rates)
    print(f"  Best condition: {ranked.iloc[0]['experiment_id'][:30]}")
    print(f"  Response magnitude: {ranked.iloc[0]['response_magnitude']:.3f}")
    
    print("\n=== Generating Outputs ===")
    
    # Save effects table
    effects_df = pd.DataFrame([
        {'effect': name, 
         'estimate': e.get('effect', np.nan) if isinstance(e, dict) else np.nan,
         'p_value': e.get('p_value', np.nan) if isinstance(e, dict) else np.nan,
         'significant': e.get('significant', False) if isinstance(e, dict) else False}
        for name, e in factorial_effects.items()
    ])
    effects_df.to_csv(output_dir / 'doe_effects_table.csv', index=False)
    print(f"  Saved {output_dir / 'doe_effects_table.csv'}")
    
    # Interaction plot
    plot_interaction(conditions, turn_rates, output_dir / 'interaction_plot.png')
    
    # Report
    generate_report(conditions, factorial_effects, ranked, output_dir / 'doe_analysis.md')
    
    print("\n=== DOE Analysis Complete ===")


if __name__ == '__main__':
    main()




