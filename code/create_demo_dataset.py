#!/usr/bin/env python3
"""
Create a smaller demo dataset from consolidated_dataset.h5

Reduces file size by:
1. Selecting subset of experiments (e.g., 1-2 experiments)
2. Selecting subset of tracks per experiment
3. Downsampling trajectory data (every Nth frame)
4. Keeping only essential columns
5. Using compression

Target size: < 500 MB (vs 5.5 GB full dataset)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json

def create_demo_dataset(
    input_h5: Path,
    output_h5: Path,
    max_experiments: int = 2,
    max_tracks_per_exp: int = 10,
    downsample_factor: int = 5,  # Keep every 5th frame
    compression: str = 'gzip',
    compression_opts: int = 9
):
    """
    Create a smaller demo dataset from consolidated H5 file.
    
    Parameters
    ----------
    input_h5 : Path
        Path to consolidated_dataset.h5
    output_h5 : Path
        Path to output demo dataset
    max_experiments : int
        Maximum number of experiments to include
    max_tracks_per_exp : int
        Maximum tracks per experiment
    downsample_factor : int
        Keep every Nth frame (1 = all frames, 5 = every 5th frame)
    compression : str
        HDF5 compression algorithm
    compression_opts : int
        Compression level (0-9)
    """
    
    print("=" * 70)
    print("Creating Demo Dataset")
    print("=" * 70)
    print(f"Input:  {input_h5}")
    print(f"Output: {output_h5}")
    print(f"Target: < 500 MB")
    print()
    
    if not input_h5.exists():
        raise FileNotFoundError(f"Input file not found: {input_h5}")
    
    # Load consolidated dataset
    print("Loading consolidated dataset...")
    exp_ids = []
    
    with h5py.File(input_h5, 'r') as f_in:
        # Get experiment IDs
        if 'experiments' in f_in:
            exp_group = f_in['experiments']
            if 'experiment_id' in exp_group:
                exp_ids = [x.decode('utf-8') if isinstance(x, bytes) else x 
                           for x in exp_group['experiment_id'][:]]
        
        # If no experiments group, try to infer from trajectories
        if not exp_ids and 'trajectories' in f_in:
            if 'experiment_id' in f_in['trajectories']:
                traj_exp_ids = f_in['trajectories']['experiment_id'][:]
                exp_ids = list(set([x.decode('utf-8') if isinstance(x, bytes) else x 
                                   for x in traj_exp_ids]))
        
        print(f"  Found {len(exp_ids)} experiments")
        
        # Select subset of experiments
        selected_exps = exp_ids[:max_experiments]
        print(f"  Selecting {len(selected_exps)} experiments: {selected_exps}")
        
        # Load trajectories
        print("\nLoading trajectories...")
        traj_data = {}
        if 'trajectories' in f_in:
            traj_group = f_in['trajectories']
            for col in traj_group.keys():
                data = traj_group[col][:]
                if data.dtype.kind == 'S':  # String data
                    data = np.array([x.decode('utf-8') for x in data])
                traj_data[col] = data
        
        # Convert to DataFrame
        traj_df = pd.DataFrame(traj_data)
        print(f"  Loaded {len(traj_df):,} trajectory rows")
        
        # Filter by selected experiments
        if 'experiment_id' in traj_df.columns:
            traj_df = traj_df[traj_df['experiment_id'].isin(selected_exps)].copy()
            print(f"  After filtering: {len(traj_df):,} rows")
        
        # Select subset of tracks per experiment
        print(f"\nSelecting up to {max_tracks_per_exp} tracks per experiment...")
        # Find track ID column (could be 'track_id', 'TrackID', 'trackId', etc.)
        track_col = None
        for col in ['track_id', 'TrackID', 'trackId', 'track']:
            if col in traj_df.columns:
                track_col = col
                break
        
        if track_col is None:
            print("  Warning: No track ID column found. Using all tracks.")
            selected_tracks = traj_df[traj_df['experiment_id'].isin(selected_exps)][track_col if track_col else 'experiment_id'].unique() if 'experiment_id' in traj_df.columns else []
        else:
            selected_tracks = []
            for exp_id in selected_exps:
                exp_mask = traj_df['experiment_id'] == exp_id if 'experiment_id' in traj_df.columns else traj_df.index < len(traj_df)
                exp_tracks = traj_df[exp_mask][track_col].unique()
                n_select = min(max_tracks_per_exp, len(exp_tracks))
                selected_tracks.extend(exp_tracks[:n_select])
                print(f"  {exp_id}: {n_select} tracks")
            
            traj_df = traj_df[traj_df[track_col].isin(selected_tracks)].copy()
        print(f"  After track selection: {len(traj_df):,} rows")
        
        # Downsample frames
        if downsample_factor > 1:
            print(f"\nDownsampling: keeping every {downsample_factor}th frame...")
            # Sort by track and time to maintain order
            sort_cols = [track_col if track_col else 'experiment_id', 'time'] if 'time' in traj_df.columns else [track_col if track_col else 'experiment_id']
            traj_df = traj_df.sort_values(sort_cols).reset_index(drop=True)
            # Group by track and sample
            if track_col:
                traj_df = traj_df.groupby(track_col).apply(
                    lambda x: x.iloc[::downsample_factor]
                ).reset_index(drop=True)
            else:
                # If no track column, just sample every Nth row
                traj_df = traj_df.iloc[::downsample_factor].reset_index(drop=True)
            print(f"  After downsampling: {len(traj_df):,} rows")
        
        # Select essential columns only
        essential_cols = [
            track_col if track_col else 'experiment_id', 'frame', 'time', 'x', 'y', 'heading', 
            'speed', 'curvature', 'experiment_id'
        ]
        # Remove duplicates and None, keep only columns that exist
        available_cols = [c for c in essential_cols if c and c in traj_df.columns]
        traj_df = traj_df[available_cols]
        print(f"  Keeping {len(available_cols)} essential columns")
        
        # Load events
        print("\nLoading events...")
        event_data = {}
        if 'events' in f_in:
            event_group = f_in['events']
            for col in event_group.keys():
                data = event_group[col][:]
                if data.dtype.kind == 'S':
                    data = np.array([x.decode('utf-8') for x in data])
                event_data[col] = data
        
        events_df = pd.DataFrame(event_data)
        print(f"  Loaded {len(events_df):,} event rows")
        
        # Filter events to match selected tracks/experiments
        if 'experiment_id' in events_df.columns:
            events_df = events_df[events_df['experiment_id'].isin(selected_exps)].copy()
        # Find track ID column in events
        event_track_col = None
        for col in ['track_id', 'TrackID', 'trackId', 'track']:
            if col in events_df.columns:
                event_track_col = col
                break
        if event_track_col and len(selected_tracks) > 0:
            events_df = events_df[events_df[event_track_col].isin(selected_tracks)].copy()
        print(f"  After filtering: {len(events_df):,} rows")
        
        # Load experiments metadata
        print("\nLoading experiment metadata...")
        exp_data = {}
        if 'experiments' in f_in:
            exp_group = f_in['experiments']
            for col in exp_group.keys():
                data = exp_group[col][:]
                if data.dtype.kind == 'S':
                    data = np.array([x.decode('utf-8') for x in data])
                exp_data[col] = data
        
        exp_df = pd.DataFrame(exp_data)
        if 'experiment_id' in exp_df.columns:
            exp_df = exp_df[exp_df['experiment_id'].isin(selected_exps)].copy()
        print(f"  Loaded {len(exp_df)} experiment records")
    
    # Write demo dataset
    print(f"\nWriting demo dataset to {output_h5}...")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_h5, 'w') as f_out:
        # Metadata
        f_out.attrs['created'] = pd.Timestamp.now().isoformat()
        f_out.attrs['source'] = str(input_h5)
        f_out.attrs['demo'] = True
        f_out.attrs['n_experiments'] = len(selected_exps)
        f_out.attrs['n_tracks'] = len(selected_tracks)
        f_out.attrs['downsample_factor'] = downsample_factor
        
        # Trajectories
        traj_grp = f_out.create_group('trajectories')
        print(f"  Creating {len(traj_df.columns)} trajectory datasets...")
        for i, col in enumerate(traj_df.columns):
            if col in traj_grp:
                print(f"    Warning: {col} already exists, skipping")
                continue
            try:
                data = traj_df[col].values
                if data.dtype == object or str(data.dtype).startswith('<U'):
                    # String columns
                    data = np.array([s.encode('utf-8') if isinstance(s, str) else str(s).encode('utf-8') 
                                   for s in data], dtype='S')
                    traj_grp.create_dataset(col, data=data, compression=compression, 
                                          compression_opts=compression_opts)
                else:
                    traj_grp.create_dataset(col, data=data, compression=compression,
                                          compression_opts=compression_opts)
                if (i + 1) % 3 == 0:
                    print(f"    Created {i + 1}/{len(traj_df.columns)} datasets...")
            except Exception as e:
                print(f"    Error creating {col}: {e}")
                raise
        traj_grp.attrs['n_rows'] = len(traj_df)
        traj_grp.attrs['columns'] = [c.encode('utf-8') for c in traj_df.columns]
        print(f"  trajectories: {len(traj_df):,} rows")
        
        # Events
        if len(events_df) > 0:
            if 'events' in f_out:
                del f_out['events']
            events_grp = f_out.create_group('events')
            for col in events_df.columns:
                data = events_df[col].values
                if data.dtype == object or str(data.dtype).startswith('<U'):
                    data = np.array([s.encode('utf-8') if isinstance(s, str) else str(s).encode('utf-8') 
                                   for s in data], dtype='S')
                    events_grp.create_dataset(col, data=data, compression=compression,
                                             compression_opts=compression_opts)
                else:
                    events_grp.create_dataset(col, data=data, compression=compression,
                                           compression_opts=compression_opts)
            events_grp.attrs['n_rows'] = len(events_df)
            events_grp.attrs['columns'] = [c.encode('utf-8') for c in events_df.columns]
            print(f"  events: {len(events_df):,} rows")
        
        # Experiments
        if len(exp_df) > 0:
            if 'experiments' in f_out:
                del f_out['experiments']
            exp_grp = f_out.create_group('experiments')
            for col in exp_df.columns:
                data = exp_df[col].values
                if data.dtype == object or str(data.dtype).startswith('<U'):
                    data = np.array([s.encode('utf-8') if isinstance(s, str) else str(s).encode('utf-8') 
                                   for s in data], dtype='S')
                    exp_grp.create_dataset(col, data=data, compression=compression,
                                          compression_opts=compression_opts)
                else:
                    exp_grp.create_dataset(col, data=data, compression=compression,
                                         compression_opts=compression_opts)
            exp_grp.attrs['n_rows'] = len(exp_df)
            exp_grp.attrs['columns'] = [c.encode('utf-8') for c in exp_df.columns]
            print(f"  experiments: {len(exp_df)} records")
    
    # Check file size
    size_mb = output_h5.stat().st_size / (1024 * 1024)
    print(f"\n✓ Demo dataset created: {size_mb:.1f} MB")
    
    if size_mb > 500:
        print(f"⚠ Warning: File size ({size_mb:.1f} MB) exceeds 500 MB target")
        print("  Consider reducing max_experiments, max_tracks_per_exp, or increasing downsample_factor")
    else:
        print(f"✓ File size ({size_mb:.1f} MB) is within target (< 500 MB)")
    
    return output_h5


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create a smaller demo dataset from consolidated H5 file'
    )
    parser.add_argument('input_h5', type=Path, 
                       help='Path to consolidated_dataset.h5')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path (default: input_demo.h5)')
    parser.add_argument('--max-experiments', type=int, default=2,
                       help='Maximum experiments to include (default: 2)')
    parser.add_argument('--max-tracks', type=int, default=10,
                       help='Maximum tracks per experiment (default: 10)')
    parser.add_argument('--downsample', type=int, default=5,
                       help='Downsample factor - keep every Nth frame (default: 5)')
    parser.add_argument('--compression', type=str, default='gzip',
                       choices=['gzip', 'lzf', 'szip'],
                       help='Compression algorithm (default: gzip)')
    parser.add_argument('--compression-level', type=int, default=9,
                       help='Compression level 0-9 (default: 9)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input_h5.parent / f"{args.input_h5.stem}_demo.h5"
    
    create_demo_dataset(
        args.input_h5,
        args.output,
        max_experiments=args.max_experiments,
        max_tracks_per_exp=args.max_tracks,
        downsample_factor=args.downsample,
        compression=args.compression,
        compression_opts=args.compression_level
    )


if __name__ == '__main__':
    main()

