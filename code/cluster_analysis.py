#!/usr/bin/env python3
"""
Cluster Analysis for Larval Behavioral Phenotyping

Identifies behavioral subgroups (responders vs non-responders) using k-means
clustering on stimulus-response features.

Usage:
    python scripts/cluster_analysis.py --input data/processed/consolidated_dataset.h5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
# FEATURE EXTRACTION
# =============================================================================

def extract_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract behavioral features per track for clustering.
    
    Features:
    - baseline_rate: turn rate in first 30s (no/minimal stimulus)
    - stim_rate: turn rate during stimulus ON
    - response_increase: (stim_rate - baseline_rate) / baseline_rate
    - latency: time to first turn after stimulus onset
    - mean_speed: average speed
    """
    features = []
    
    # Detect events
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
        group = group.sort_values('time')
        t_start = group['time'].min()
        t_end = group['time'].max()
        duration = t_end - t_start
        
        if duration < 60:  # Skip very short tracks
            continue
        
        # Baseline: first 30 seconds
        baseline = group[group['time'] < t_start + 30]
        baseline_dur = min(30, (baseline['time'].max() - baseline['time'].min())) / 60.0
        baseline_events = baseline[event_col].sum()
        baseline_rate = baseline_events / baseline_dur if baseline_dur > 0 else 0
        
        # Stimulus ON periods
        if 'led1Val' in group.columns:
            stim_on = group[group['led1Val'] > 0]
            stim_off = group[group['led1Val'] == 0]
        else:
            stim_on = group
            stim_off = pd.DataFrame()
        
        stim_dur = len(stim_on) / len(group) * duration / 60.0 if len(group) > 0 else 0
        stim_events = stim_on[event_col].sum() if len(stim_on) > 0 else 0
        stim_rate = stim_events / stim_dur if stim_dur > 0 else 0
        
        # Response increase (percentage)
        if baseline_rate > 0:
            response_increase = (stim_rate - baseline_rate) / baseline_rate * 100
        else:
            response_increase = 0 if stim_rate == 0 else 100
        
        # Latency to first turn after first stimulus onset
        if 'led1Val' in group.columns:
            # Find first stimulus onset
            led_diff = group['led1Val'].diff()
            stim_onsets = group[led_diff > 0]
            
            if len(stim_onsets) > 0:
                first_stim_time = stim_onsets['time'].min()
                # Find first event after stimulus
                events_after = group[(group['time'] > first_stim_time) & group[event_col]]
                if len(events_after) > 0:
                    latency = events_after['time'].min() - first_stim_time
                else:
                    latency = np.nan
            else:
                latency = np.nan
        else:
            latency = np.nan
        
        # Mean speed
        mean_speed = group['speed'].mean() if 'speed' in group.columns else np.nan
        
        # Mean turn angle magnitude (if available)
        if 'turn_duration' in group.columns:
            mean_turn_dur = group[group['turn_duration'] > 0]['turn_duration'].mean()
        else:
            mean_turn_dur = np.nan
        
        features.append({
            'experiment_id': str(exp_id),
            'track_id': track_id,
            'baseline_rate': baseline_rate,
            'stim_rate': stim_rate,
            'response_increase': response_increase,
            'latency': latency,
            'mean_speed': mean_speed,
            'mean_turn_duration': mean_turn_dur,
            'duration': duration
        })
    
    return pd.DataFrame(features)


# =============================================================================
# CLUSTERING
# =============================================================================

def run_clustering(features_df: pd.DataFrame, k_values: list = [2, 3, 4]) -> dict:
    """
    Run k-means clustering for multiple k values and select optimal k.
    """
    # Feature columns for clustering
    feature_cols = ['baseline_rate', 'stim_rate', 'response_increase', 'latency', 'mean_speed']
    
    # Filter to valid rows
    valid_df = features_df.dropna(subset=feature_cols)
    
    if len(valid_df) < 10:
        return {'error': 'Insufficient valid data for clustering'}
    
    X = valid_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'feature_cols': feature_cols,
        'n_samples': len(valid_df),
        'k_results': {}
    }
    
    best_silhouette = -1
    best_k = 2
    
    for k in k_values:
        if len(valid_df) < k:
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        
        # Cluster statistics
        cluster_stats = []
        for c in range(k):
            mask = labels == c
            cluster_df = valid_df.iloc[mask]
            
            cluster_stats.append({
                'cluster': c,
                'n': mask.sum(),
                'pct': 100 * mask.sum() / len(valid_df),
                'mean_baseline_rate': cluster_df['baseline_rate'].mean(),
                'mean_stim_rate': cluster_df['stim_rate'].mean(),
                'mean_response_increase': cluster_df['response_increase'].mean(),
                'mean_latency': cluster_df['latency'].mean(),
                'mean_speed': cluster_df['mean_speed'].mean()
            })
        
        results['k_results'][k] = {
            'silhouette': silhouette,
            'inertia': kmeans.inertia_,
            'cluster_stats': cluster_stats,
            'labels': labels.tolist()
        }
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    results['optimal_k'] = best_k
    results['best_silhouette'] = best_silhouette
    
    # Assign cluster labels to dataframe for best k
    if best_k in results['k_results']:
        valid_df = valid_df.copy()
        valid_df['cluster'] = results['k_results'][best_k]['labels']
        results['labeled_features'] = valid_df
    
    return results


def classify_clusters(cluster_stats: list) -> list:
    """Assign interpretive labels to clusters based on response patterns."""
    # Sort by response increase
    sorted_stats = sorted(cluster_stats, key=lambda x: x['mean_response_increase'], reverse=True)
    
    labels = []
    for i, stat in enumerate(sorted_stats):
        resp = stat['mean_response_increase']
        lat = stat['mean_latency']
        
        if resp > 20:
            label = 'Responder'
        elif resp > 5:
            if lat > 30:
                label = 'Delayed Responder'
            else:
                label = 'Weak Responder'
        else:
            label = 'Non-Responder'
        
        labels.append({**stat, 'label': label})
    
    return labels


# =============================================================================
# CONDITION SALIENCE
# =============================================================================

def compute_condition_salience(
    features_df: pd.DataFrame,
    cluster_results: dict
) -> pd.DataFrame:
    """
    Compute silhouette score per experimental condition.
    """
    if 'labeled_features' not in cluster_results:
        return pd.DataFrame()
    
    labeled = cluster_results['labeled_features']
    feature_cols = cluster_results['feature_cols']
    
    salience = []
    
    for exp_id in labeled['experiment_id'].unique():
        exp_df = labeled[labeled['experiment_id'] == exp_id]
        
        if len(exp_df) < 5:
            continue
        
        X = exp_df[feature_cols].values
        labels = exp_df['cluster'].values
        
        # Need at least 2 clusters present
        if len(np.unique(labels)) < 2:
            sil = 0
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            sil = silhouette_score(X_scaled, labels)
        
        # F-ratio (between/within variance)
        overall_mean = X.mean(axis=0)
        between_var = 0
        within_var = 0
        
        for c in np.unique(labels):
            mask = labels == c
            cluster_mean = X[mask].mean(axis=0)
            n_c = mask.sum()
            between_var += n_c * np.sum((cluster_mean - overall_mean)**2)
            within_var += np.sum((X[mask] - cluster_mean)**2)
        
        f_ratio = between_var / (within_var + 1e-9)
        
        salience.append({
            'experiment_id': exp_id,
            'silhouette': sil,
            'f_ratio': f_ratio,
            'n_tracks': len(exp_df)
        })
    
    salience_df = pd.DataFrame(salience)
    salience_df['rank'] = salience_df['silhouette'].rank(ascending=False)
    
    return salience_df.sort_values('rank')


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_cluster_profiles(cluster_stats: list, output_path: Path):
    """Plot cluster profiles as bar chart."""
    n_clusters = len(cluster_stats)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Response increase
    ax = axes[0]
    clusters = [s['cluster'] for s in cluster_stats]
    values = [s['mean_response_increase'] for s in cluster_stats]
    colors = ['green' if v > 10 else 'orange' if v > 0 else 'red' for v in values]
    ax.bar(clusters, values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Response Increase (%)')
    ax.set_title('Stimulus Response Magnitude')
    ax.axhline(y=10, color='gray', linestyle='--', label='Responder threshold')
    
    # Latency
    ax = axes[1]
    values = [s['mean_latency'] for s in cluster_stats]
    ax.bar(clusters, values, color='steelblue')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Response Latency')
    
    # Cluster size
    ax = axes[2]
    values = [s['pct'] for s in cluster_stats]
    labels = [s.get('label', f'C{s["cluster"]}') for s in cluster_stats]
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=plt.cm.Set2.colors[:n_clusters])
    ax.set_title('Cluster Composition')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    cluster_results: dict,
    salience_df: pd.DataFrame,
    output_path: Path
):
    """Generate cluster analysis report."""
    lines = [
        "# Cluster Analysis Report",
        "",
        "## Clustering Results",
        "",
        f"**Optimal k:** {cluster_results.get('optimal_k', 'N/A')}",
        f"**Best silhouette score:** {cluster_results.get('best_silhouette', 0):.3f}",
        f"**Number of tracks analyzed:** {cluster_results.get('n_samples', 0)}",
        "",
        "## Cluster Profiles",
        "",
        "| Cluster | Label | n (%) | Response (%) | Latency (s) | Speed |",
        "|---------|-------|-------|--------------|-------------|-------|",
    ]
    
    optimal_k = cluster_results.get('optimal_k', 2)
    if optimal_k in cluster_results.get('k_results', {}):
        stats = cluster_results['k_results'][optimal_k]['cluster_stats']
        labeled_stats = classify_clusters(stats)
        
        for s in labeled_stats:
            lines.append(f"| {s['cluster']} | {s['label']} | {s['n']} ({s['pct']:.1f}%) | "
                        f"{s['mean_response_increase']:.1f} | {s['mean_latency']:.1f} | "
                        f"{s['mean_speed']:.4f} |")
    
    lines.extend([
        "",
        "## Subgroup Discovery",
        "",
    ])
    
    # Calculate responder/non-responder proportions
    if optimal_k in cluster_results.get('k_results', {}):
        stats = cluster_results['k_results'][optimal_k]['cluster_stats']
        non_resp = sum(s['pct'] for s in stats if s['mean_response_increase'] < 10)
        lines.append(f"**Non-responders (<10% increase):** {non_resp:.1f}% of larvae")
        lines.append(f"**Responders (>=10% increase):** {100-non_resp:.1f}% of larvae")
    
    lines.extend([
        "",
        "## Condition Salience Ranking",
        "",
        "| Rank | Experiment | Silhouette | F-ratio |",
        "|------|------------|------------|---------|",
    ])
    
    for _, row in salience_df.head(5).iterrows():
        lines.append(f"| {int(row['rank'])} | {row['experiment_id'][:25]} | "
                    f"{row['silhouette']:.3f} | {row['f_ratio']:.2f} |")
    
    if len(salience_df) > 0:
        best = salience_df.iloc[0]
        lines.extend([
            "",
            f"**Best condition for phenotype separation:** {best['experiment_id']}",
            f"**Silhouette score:** {best['silhouette']:.3f}",
        ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cluster analysis for behavioral phenotyping')
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
    
    print("\n=== Extracting Features ===")
    features = extract_track_features(df)
    print(f"  {len(features)} tracks with valid features")
    
    print("\n=== Clustering ===")
    cluster_results = run_clustering(features)
    
    if 'error' in cluster_results:
        print(f"  Error: {cluster_results['error']}")
        return
    
    print(f"  Optimal k: {cluster_results['optimal_k']}")
    print(f"  Best silhouette: {cluster_results['best_silhouette']:.3f}")
    
    # Print cluster summary
    optimal_k = cluster_results['optimal_k']
    if optimal_k in cluster_results['k_results']:
        stats = cluster_results['k_results'][optimal_k]['cluster_stats']
        labeled = classify_clusters(stats)
        for s in labeled:
            print(f"    Cluster {s['cluster']} ({s['label']}): {s['n']} tracks ({s['pct']:.1f}%)")
    
    print("\n=== Condition Salience ===")
    salience_df = compute_condition_salience(features, cluster_results)
    if len(salience_df) > 0:
        best = salience_df.iloc[0]
        print(f"  Best condition: {best['experiment_id'][:30]}")
        print(f"  Silhouette: {best['silhouette']:.3f}")
    
    print("\n=== Generating Outputs ===")
    
    # Cluster profiles
    if optimal_k in cluster_results['k_results']:
        stats = cluster_results['k_results'][optimal_k]['cluster_stats']
        labeled = classify_clusters(stats)
        plot_cluster_profiles(labeled, output_dir / 'cluster_profiles.png')
        
        # Save profiles
        profiles_df = pd.DataFrame(labeled)
        profiles_df.to_csv(output_dir / 'cluster_profiles.csv', index=False)
        print(f"  Saved {output_dir / 'cluster_profiles.csv'}")
    
    # Salience ranking
    if len(salience_df) > 0:
        salience_df.to_csv(output_dir / 'cluster_salience_ranking.csv', index=False)
        print(f"  Saved {output_dir / 'cluster_salience_ranking.csv'}")
    
    # Report
    generate_report(cluster_results, salience_df, output_dir / 'cluster_report.md')
    
    print("\n=== Cluster Analysis Complete ===")


if __name__ == '__main__':
    main()




