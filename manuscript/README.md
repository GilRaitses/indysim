# InDySim Manuscript

Manuscript files for "Temporal dynamics of mechanosensory behavior in Drosophila larvae"

## Contents

- `main.tex` - Main manuscript (modular LaTeX with section includes)
- `supplement.tex` - Supplementary materials
- `references.bib` - Bibliography file
- `sections/` - Individual section files (9 section files)
- `code/` - Analysis scripts:
  - **End-to-end pipeline**:
    - `run_analysis_pipeline.py` - Complete analysis pipeline (H5 → results)
  - **Core pipeline scripts** (required for `run_analysis_pipeline.py`):
    - `engineer_dataset_from_h5.py` - Extract trajectory and stimulus data from H5 files
    - `fit_biphasic_model.py` - Fit gamma-difference kernel hazard model
    - `simulate_trajectories.py` - Simulate trajectories using fitted models
    - `analytic_hazard.py` - Analytic hazard model implementation
    - `load_fitting_data.py` - Data loading utilities
    - `detect_events.py` - Behavioral event detection (used by engineer script)
  - **Analysis scripts**:
    - `analyze_event_durations.py`
    - `analyze_reverse_crawl_modulation.py`
    - `demo.py`
    - `fit_gamma_per_condition.py`
    - `generate_peristimulus_rate_figures.py`
- `data/` - Analysis results and figures:
  - `analysis_results/` - Model fitting results (JSON files)
  - `figures/` - All manuscript and supplementary figures

## Compilation

To compile the manuscript:

```bash
# Compile main manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Compile supplement
pdflatex supplement.tex
```

## Repository

Full code and analysis scripts: [https://github.com/GilRaitses/indysim](https://github.com/GilRaitses/indysim) (InDySim: Interface Dynamics Simulation Model)

## Data Availability

Original experimental data (HDF5 files) are not included. Data available upon request. Contact: mmihovil@syr.edu

## Citation

Raitses G, Goonawardhana D, Mihovilovic-Skanata M (2025). Temporal dynamics of mechanosensory behavior in Drosophila larvae. [Manuscript in preparation]

