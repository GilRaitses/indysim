#!/usr/bin/env python3
"""
Document Event Detection Thresholds

Documents all event detection thresholds used in the pipeline for
transparency and to support future sensitivity analysis.

Output:
- data/model/event_detection_thresholds.json

Usage:
    python scripts/document_thresholds.py
"""

import json
from pathlib import Path


def document_thresholds() -> dict:
    """Document all event detection thresholds used in the pipeline."""
    
    thresholds = {
        'turn_detection': {
            'threshold': 30,  # degrees
            'threshold_radians': 0.524,  # π/6
            'unit': 'degrees',
            'description': 'Heading change threshold for turn initiation',
            'source': 'scripts/detect_events.py line 73: turn_threshold=np.pi/6',
            'stabilization_threshold': 10,  # degrees
            'stabilization_radians': 0.175,  # π/18
            'stabilization_description': 'Heading change must drop below this to end turn'
        },
        
        'pause_detection': {
            'speed_threshold': 0.001,
            'unit': 'cm/s',
            'min_duration': 0.5,
            'duration_unit': 'seconds',
            'description': 'Speed below threshold for minimum duration triggers pause',
            'source': 'scripts/detect_events.py line 284-285',
            'note': 'Some files use 0.2s (200ms) minimum duration'
        },
        
        'reverse_crawl_detection': {
            'speed_criterion': 'SpeedRunVel < 0',
            'min_duration': 3.0,
            'duration_unit': 'seconds',
            'description': 'Backward locomotion sustained for minimum duration',
            'source': 'scripts/detect_events.py line 211-232',
            'reference': 'Klein et al. 2015'
        },
        
        'reversal_detection': {
            'threshold': 90,  # degrees
            'threshold_radians': 1.571,  # π/2
            'unit': 'degrees',
            'description': 'Large heading changes exceeding 90° marked as reversals',
            'source': 'scripts/detect_events.py line 235: reversal_threshold=np.pi/2'
        },
        
        'curvature_detection': {
            'curv_cut': 0.4,
            'unit': 'dimensionless (body curvature scale)',
            'description': 'Body curvature threshold for MAGAT segmentation',
            'source': 'scripts/magat_segmentation.py line 32',
            'note': 'Applies to sspineTheta-based body curvature, not path curvature',
            'auto_adjust': 'Optional percentile-based auto-adjustment available'
        },
        
        'reorientation_simple': {
            'threshold': 30,  # degrees
            'threshold_radians': 0.524,  # π/6
            'unit': 'degrees',
            'description': 'Simple heading change threshold for backwards compatibility',
            'source': 'scripts/engineer_dataset_from_h5.py line 554'
        }
    }
    
    # Sensitivity analysis recommendations
    sensitivity = {
        'recommended_ranges': {
            'turn_threshold': {
                'baseline': 30,
                'test_values': [24, 27, 30, 33, 36],  # ±20%
                'unit': 'degrees'
            },
            'pause_min_duration': {
                'baseline': 0.5,
                'test_values': [0.3, 0.4, 0.5, 0.6, 0.75],  # ±40%
                'unit': 'seconds'
            },
            'reverse_crawl_min_duration': {
                'baseline': 3.0,
                'test_values': [2.0, 2.5, 3.0, 3.5, 4.0],  # ±33%
                'unit': 'seconds'
            },
            'curvature_threshold': {
                'baseline': 0.4,
                'test_values': [0.3, 0.35, 0.4, 0.45, 0.5],  # ±25%
                'unit': 'dimensionless'
            }
        },
        
        'methodology': {
            'approach': 'Full model refit at each threshold setting',
            'metrics_to_track': ['tau1', 'tau2', 'amplitude', 'n_events', 'r_squared'],
            'robustness_criterion': 'Parameter changes < 10% across threshold range',
            'estimate_runtime': '~30 min per threshold × 4 thresholds × 5 values = 10 hours'
        },
        
        'status': 'NOT YET IMPLEMENTED',
        'reason': 'Computationally expensive - requires full kernel refitting at each threshold',
        'future_work': True
    }
    
    # Literature references
    references = {
        'turn_detection': 'Gepner et al. 2015 eLife used curvature-based detection',
        'reverse_crawl': 'Klein et al. 2015 PNAS defined SpeedRunVel < 0 for ≥3s',
        'pause_detection': 'Standard larval tracking uses speed threshold',
        'note': 'Exact thresholds vary across studies; 30° is commonly used for turns'
    }
    
    return {
        'thresholds': thresholds,
        'sensitivity_analysis': sensitivity,
        'literature_references': references,
        'documentation_date': '2025-12-13'
    }


def main():
    print("=" * 70)
    print("DOCUMENTING EVENT DETECTION THRESHOLDS")
    print("=" * 70)
    
    result = document_thresholds()
    
    print("\nThresholds documented:")
    for name, data in result['thresholds'].items():
        if 'threshold' in data:
            print(f"  {name}: {data.get('threshold', data.get('curv_cut', 'N/A'))} {data.get('unit', '')}")
        elif 'speed_threshold' in data:
            print(f"  {name}: speed < {data['speed_threshold']} {data['unit']} for {data['min_duration']}s")
        elif 'min_duration' in data:
            print(f"  {name}: {data.get('speed_criterion', '')} for {data['min_duration']}s")
    
    print("\nSensitivity analysis status:", result['sensitivity_analysis']['status'])
    print(f"Estimated runtime if implemented: {result['sensitivity_analysis']['methodology']['estimate_runtime']}")
    
    # Save
    output_path = Path('data/model/event_detection_thresholds.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return result


if __name__ == '__main__':
    main()
