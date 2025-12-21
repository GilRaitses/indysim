#!/usr/bin/env python3
"""Generate missing engineered CSVs from consolidated H5."""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    h5_path = Path('data/processed/consolidated_dataset.h5')
    out_dir = Path('data/engineered')
    out_dir.mkdir(exist_ok=True)
    
    # Get existing timestamps
    existing_ts = set()
    for f in out_dir.glob('*_events.csv'):
        parts = f.stem.replace('_events', '').split('_')
        ts = parts[-1]
        existing_ts.add(ts)
    
    print(f"Existing engineered files: {len(existing_ts)}")
    
    with h5py.File(h5_path, 'r') as f:
        # Get events data
        events_grp = f['events']
        columns = [c.decode() if isinstance(c, bytes) else c for c in events_grp.attrs['columns']]
        
        # Load all events
        print("Loading events from H5...")
        data = {}
        for col in columns:
            vals = events_grp[col][:]
            if vals.dtype.kind == 'S':
                vals = np.array([v.decode() if isinstance(v, bytes) else v for v in vals])
            data[col] = vals
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df):,} events with {len(columns)} columns")
        
        # Get unique experiments
        unique_exps = df['experiment_id'].unique()
        
        # Find missing experiments
        missing = []
        for exp_id in unique_exps:
            ts = exp_id.split('_')[-1]
            if ts not in existing_ts:
                missing.append(exp_id)
        
        print(f"\nMissing experiments: {len(missing)}")
        
        for exp_id in sorted(missing):
            exp_df = df[df['experiment_id'] == exp_id].copy()
            
            # Save as events CSV
            out_path = out_dir / f'{exp_id}_events.csv'
            exp_df.to_csv(out_path, index=False)
            print(f"  Created: {out_path.name} ({len(exp_df):,} rows)")
    
    # Verify final count
    final_count = len(list(out_dir.glob('*_events.csv')))
    print(f"\nTotal engineered files: {final_count}")

if __name__ == '__main__':
    main()
