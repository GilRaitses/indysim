#!/usr/bin/env python3
"""
Centralized path configuration for INDYsim project.

Provides common paths used across scripts to avoid hardcoded paths.
"""

from pathlib import Path

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
H5_FILES_DIR = DATA_DIR / 'h5_files'
MATLAB_DATA_DIR = DATA_DIR / 'matlab_data'
ENGINEERED_DATA_DIR = DATA_DIR / 'engineered'

# Output directories
OUTPUT_DIR = PROJECT_ROOT / 'output'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'

# Common test files (examples - can be overridden)
EXAMPLE_H5_FILE = H5_FILES_DIR / 'GMR61_tier2_complete.h5' if (H5_FILES_DIR / 'GMR61_tier2_complete.h5').exists() else None





