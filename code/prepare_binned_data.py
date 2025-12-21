#!/usr/bin/env python3
"""
Prepare Binned Data for NB-GLM LNP Model

Transforms frame-level trajectory/event data into binned format suitable for
Negative Binomial GLM LNP modeling.

Key transformations:
- Bin frames into 0.5s non-overlapping windows
- Detect reorientation ONSETS (transitions, not frame-level booleans)
- Scale covariates:
  - LED1: divide by 250
  - LED2: divide by 15
  - Speed, curvature: z-score
- Compute temporal kernel bases (raised-cosine)
- Compute phase covariates (sin/cos of LED1 cycle position)

Usage:
    python scripts/prepare_binned_data.py --input data/processed/consolidated_dataset.h5 \
                                          --output data/processed/binned_0.5s.parquet
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


# =============================================================================
# RAISED-COSINE BASIS FUNCTIONS
# =============================================================================

def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Compute raised-cosine basis functions for temporal kernels.
    
    Parameters
    ----------
    t : ndarray
        Time points (seconds)
    centers : ndarray
        Center positions for each basis function
    width : float
        Width parameter (controls overlap)
    
    Returns
    -------
    basis : ndarray
        Shape (len(t), len(centers))
    """
    t = np.atleast_1d(t).flatten()
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


# =============================================================================
# DATA LOADING
# =============================================================================

def load_consolidated_h5(h5_path: Path) -> pd.DataFrame:
    """
    Load consolidated HDF5 dataset into DataFrame.
    
    Parameters
    ----------
    h5_path : Path
        Path to consolidated_dataset.h5
    
    Returns
    -------
    df : DataFrame
        Frame-level data with all columns
    """
    print(f"Loading {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # Prefer events group as it has experiment_id and track_id columns
        if 'events' in f:
            grp = f['events']
            print("  Using 'events' group (has experiment_id and track_id)")
        elif 'trajectories' in f:
            grp = f['trajectories']
            print("  Using 'trajectories' group")
        else:
            raise ValueError(f"No 'trajectories' or 'events' group in {h5_path}")
        
        # Load each column
        data = {}
        for col in grp.keys():
            arr = grp[col][:]
            # Handle string columns
            if arr.dtype.kind == 'S':
                arr = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in arr])
            data[col] = arr
        
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# BINNING AND FEATURE ENGINEERING
# =============================================================================

def detect_reorientation_onsets(df: pd.DataFrame) -> pd.Series:
    """
    Detect reorientation onsets (transitions from False to True).
    
    This counts actual EVENT starts, not just frame-level booleans.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: experiment_id, track_id, time, is_reorientation
    
    Returns
    -------
    reo_onset : Series
        Boolean series where True = reorientation started this frame
    """
    # Sort by experiment, track, time to ensure correct ordering
    df = df.sort_values(['experiment_id', 'track_id', 'time'])
    
    # Detect transitions: was False, now True
    # Use shift within each track
    prev_reo = df.groupby(['experiment_id', 'track_id'])['is_reorientation'].shift(1, fill_value=False)
    reo_onset = df['is_reorientation'] & ~prev_reo
    
    return reo_onset.astype(int)


def bin_data_for_hazard(
    df: pd.DataFrame,
    bin_width: float = 0.5,
    n_bases: int = 4,
    kernel_window: tuple = (0.0, 3.0),
    kernel_width: float = 0.6
) -> pd.DataFrame:
    """
    Prepare binned dataset for NB-GLM LNP model.
    
    Parameters
    ----------
    df : DataFrame
        Frame-level data with columns: time, track_id, experiment_id,
        is_reorientation, led1Val, led2Val, speed, curvature, time_since_stimulus
    bin_width : float
        Bin size in seconds (default 0.5)
    n_bases : int
        Number of temporal kernel bases (default 4)
    kernel_window : tuple
        (min, max) time range for kernel centers
    kernel_width : float
        Width parameter for raised-cosine bases
    
    Returns
    -------
    binned : DataFrame
        Bin-level data ready for GLM fitting
    """
    print(f"Binning data to {bin_width}s windows...")
    df = df.copy()
    
    # Ensure required columns exist
    required = ['time', 'track_id', 'experiment_id', 'is_reorientation']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # 1. Detect reorientation onsets (transitions)
    print("  Detecting reorientation onsets...")
    df['reo_onset'] = detect_reorientation_onsets(df)
    n_onsets = df['reo_onset'].sum()
    print(f"    Found {n_onsets:,} reorientation onsets")
    
    # 2. Compute phase covariates (60s LED1 cycle)
    cycle_period = 60.0
    phase = (df['time'] % cycle_period) / cycle_period
    df['phase_sin'] = np.sin(2 * np.pi * phase)
    df['phase_cos'] = np.cos(2 * np.pi * phase)
    
    # 3. Compute temporal kernel bases
    print(f"  Computing {n_bases} temporal kernel bases (window {kernel_window})...")
    if 'time_since_stimulus' in df.columns:
        t = df['time_since_stimulus'].fillna(999).values
    else:
        # Fallback: use time mod 60 (within LED1 cycle)
        t = (df['time'] % 60.0).values
    
    centers = np.linspace(kernel_window[0], kernel_window[1], n_bases)
    basis = raised_cosine_basis(t, centers, kernel_width)
    for j in range(n_bases):
        df[f'kernel_{j+1}'] = basis[:, j]
    
    # 4. Assign bins
    df['bin_start'] = (df['time'] // bin_width) * bin_width
    
    # 5. Aggregate to bin level
    print("  Aggregating to bin level...")
    group_cols = ['experiment_id', 'track_id', 'bin_start']
    
    # Build aggregation dictionary
    agg_dict = {
        'reo_onset': 'sum',  # count of onsets in bin
        'phase_sin': 'mean',
        'phase_cos': 'mean',
        'time_since_stimulus': 'mean',
    }
    
    # Optional columns
    if 'led1Val' in df.columns:
        agg_dict['led1Val'] = 'mean'
    if 'led2Val' in df.columns:
        agg_dict['led2Val'] = 'mean'
    if 'speed' in df.columns:
        agg_dict['speed'] = 'mean'
    if 'curvature' in df.columns:
        agg_dict['curvature'] = 'mean'
    
    # Kernel bases
    for j in range(n_bases):
        agg_dict[f'kernel_{j+1}'] = 'mean'
    
    binned = df.groupby(group_cols, as_index=False).agg(agg_dict)
    binned.rename(columns={'reo_onset': 'Y'}, inplace=True)
    
    print(f"    {len(binned):,} bins created")
    print(f"    Y (reorientation count) distribution: min={binned['Y'].min()}, "
          f"max={binned['Y'].max()}, mean={binned['Y'].mean():.3f}")
    
    # 6. Scale covariates
    print("  Scaling covariates...")
    
    # LED1: linear scaling (divide by 250)
    if 'led1Val' in binned.columns:
        binned['LED1_scaled'] = binned['led1Val'] / 250.0
    else:
        binned['LED1_scaled'] = 0.0
    
    # LED2: divide by 15
    if 'led2Val' in binned.columns:
        binned['LED2_scaled'] = binned['led2Val'] / 15.0
    else:
        binned['LED2_scaled'] = 0.0
    
    # Interaction term
    binned['LED1xLED2'] = binned['LED1_scaled'] * binned['LED2_scaled']
    
    # Z-score speed and curvature
    for col in ['speed', 'curvature']:
        if col in binned.columns:
            m = binned[col].mean()
            s = binned[col].std()
            binned[f'{col}_z'] = (binned[col] - m) / (s + 1e-9)
            print(f"    {col}: mean={m:.4f}, std={s:.4f}")
        else:
            binned[f'{col}_z'] = 0.0
    
    # 7. Add bin width as column (for exposure offset)
    binned['bin_width'] = bin_width
    
    return binned


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare binned data for NB-GLM LNP model')
    parser.add_argument('--input', type=str, default='data/processed/consolidated_dataset.h5',
                        help='Input consolidated HDF5 file')
    parser.add_argument('--output', type=str, default='data/processed/binned_0.5s.parquet',
                        help='Output parquet file')
    parser.add_argument('--bin-width', type=float, default=0.5,
                        help='Bin width in seconds (default 0.5)')
    parser.add_argument('--n-bases', type=int, default=4,
                        help='Number of temporal kernel bases (default 4)')
    parser.add_argument('--kernel-window', type=float, nargs=2, default=[0.0, 3.0],
                        help='Kernel window range (default 0 3)')
    parser.add_argument('--kernel-width', type=float, default=0.6,
                        help='Kernel width parameter (default 0.6)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    # Load data
    df = load_consolidated_h5(input_path)
    
    # Bin data
    binned = bin_data_for_hazard(
        df,
        bin_width=args.bin_width,
        n_bases=args.n_bases,
        kernel_window=tuple(args.kernel_window),
        kernel_width=args.kernel_width
    )
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binned.to_parquet(output_path, index=False)
    print(f"\nSaved binned data to {output_path}")
    print(f"  {len(binned):,} bins")
    print(f"  {binned['Y'].sum():,} total reorientation onsets")
    print(f"  Columns: {list(binned.columns)}")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Experiments: {binned['experiment_id'].nunique()}")
    print(f"Tracks: {binned.groupby('experiment_id')['track_id'].nunique().sum()}")
    print(f"Bins per track (mean): {binned.groupby(['experiment_id', 'track_id']).size().mean():.1f}")
    print(f"Y distribution:")
    print(f"  0: {(binned['Y'] == 0).sum():,} ({100*(binned['Y'] == 0).mean():.1f}%)")
    print(f"  1: {(binned['Y'] == 1).sum():,} ({100*(binned['Y'] == 1).mean():.1f}%)")
    print(f"  2+: {(binned['Y'] >= 2).sum():,} ({100*(binned['Y'] >= 2).mean():.1f}%)")


if __name__ == '__main__':
    main()




