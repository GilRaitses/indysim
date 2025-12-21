#!/usr/bin/env python3
"""
Package Supplementary Materials for PI Review

Collects all model outputs, figures, and processed data into a ZIP archive
and copies manuscript/supplement PDFs to a deliverables folder.

Usage:
    python scripts/package_supplement.py [--output-dir deliverables]

Output Structure:
    deliverables/
    ├── manuscript.pdf
    ├── supplement.pdf
    ├── INDYsim_supplement_YYYYMMDD.zip
    │   ├── model/
    │   │   ├── model_results.json
    │   │   ├── kernel_bootstrap_ci.json
    │   │   └── ...
    │   ├── figures/
    │   │   ├── figure1_kernel.png
    │   │   └── ...
    │   └── demo/
    │       ├── README.md
    │       └── indysim_demo.py
    └── README.md
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


def create_demo_readme() -> str:
    """Create README content for the demo folder."""
    return """# INDYsim Demo Pipeline

## Quick Start

Run the demo pipeline on a single H5 file:

```bash
python indysim_demo.py <path_to_h5_file> --output-dir demo_output
```

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, statsmodels
- h5py
- Quarto (for report rendering)

Install dependencies:
```bash
pip install numpy pandas matplotlib statsmodels h5py
```

## Pipeline Steps

1. **Engineer dataset** - Extract trajectories and events from H5
2. **Prepare binned data** - Create time bins for hazard model
3. **Fit hazard model** - NB-GLM with gamma-difference kernel
4. **Detect reverse crawls** - Mason Klein SpeedRunVel method
5. **Simulate trajectories** - Generate synthetic trajectories
6. **Generate figures** - Create summary visualizations
7. **Create report** - Render HTML/PDF report

## Output

The pipeline creates:
- `trajectories.parquet` - Full frame-level data
- `events.parquet` - Binned event data
- `model_results.json` - Fitted model parameters
- `reverse_crawl_results.json` - Reversal detection results
- `figures/` - All generated figures
- `report.html` - Interactive analysis report

## Example

```bash
# Run on reference experiment
python indysim_demo.py \\
    "data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#C_Bl_7PWM/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5" \\
    --output-dir demo_output

# Open the report
open demo_output/report.html
```

## Contact

Questions? Contact graitses@syr.edu
"""


def create_main_readme(timestamp: str) -> str:
    """Create README content for the deliverables folder."""
    return f"""# INDYsim Supplementary Materials

## Paper

"An Analytic Hazard Kernel for Optogenetically-Driven Larval Reorientation"

## Contents

- `manuscript.pdf` - Main manuscript
- `supplement.pdf` - Supplementary tables and figures
- `INDYsim_supplement_{timestamp}.zip` - Model data and figures

## ZIP Archive Contents

### model/
JSON files containing all fitted model parameters:
- `model_results.json` - Primary hazard model results
- `kernel_bootstrap_ci.json` - Bootstrap confidence intervals
- `factorial_model_results.json` - Factorial design analysis
- `turn_distributions.json` - Turn angle/duration parameters
- `reverse_crawl_modulation.json` - LED modulation of reverse crawls

### figures/
All figures from the paper and supplement:
- `figure1_kernel.png` - Kernel decomposition (Figure 1)
- `figure2_validation.png` - Model validation (Figure 2)
- `figure3_trajectories.png` - Trajectory simulation (Figure 3)
- `figure5_factorial.png` - Factorial analysis (Figure 5)
- Additional supplementary figures

### demo/
Runnable demo pipeline:
- `indysim_demo.py` - Self-contained analysis script
- `README.md` - Usage instructions

## Running the Demo

```bash
cd demo/
python indysim_demo.py <path_to_h5_file> --output-dir results
```

## Repository

Full code: https://github.com/GilRaitses/indysim (InDySim: Interface Dynamics Simulation Model)

## Generated

{timestamp}
"""


def package_supplements(output_dir: Path):
    """Package all supplementary materials."""
    
    project_root = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  Packaging Supplementary Materials")
    print("=" * 60)
    
    # Copy PDFs
    print("\n[1/4] Copying PDFs...")
    
    manuscript_src = project_root / "docs" / "paper" / "manuscript.pdf"
    supplement_src = project_root / "docs" / "paper" / "supplement.pdf"
    
    if manuscript_src.exists():
        shutil.copy2(manuscript_src, output_dir / "manuscript.pdf")
        print(f"  Copied: manuscript.pdf")
    else:
        print(f"  Warning: manuscript.pdf not found")
    
    if supplement_src.exists():
        shutil.copy2(supplement_src, output_dir / "supplement.pdf")
        print(f"  Copied: supplement.pdf")
    else:
        print(f"  Warning: supplement.pdf not found")
    
    # Create ZIP archive
    print("\n[2/4] Creating ZIP archive...")
    
    zip_name = f"INDYsim_supplement_{timestamp}.zip"
    zip_path = output_dir / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # Add model JSON files
        model_dir = project_root / "data" / "model"
        if model_dir.exists():
            for json_file in model_dir.glob("*.json"):
                zf.write(json_file, f"model/{json_file.name}")
                print(f"    Added: model/{json_file.name}")
        
        # Add figures
        figures_dir = project_root / "figures"
        if figures_dir.exists():
            for fig_file in figures_dir.glob("*.png"):
                zf.write(fig_file, f"figures/{fig_file.name}")
                print(f"    Added: figures/{fig_file.name}")
            
            # Add factorial diagnostics subfolder
            diag_dir = figures_dir / "factorial_diagnostics"
            if diag_dir.exists():
                for fig_file in diag_dir.glob("*.png"):
                    zf.write(fig_file, f"figures/factorial_diagnostics/{fig_file.name}")
        
        # Add paper figures
        paper_dir = project_root / "docs" / "paper"
        for fig_file in paper_dir.glob("figure*.png"):
            zf.write(fig_file, f"figures/{fig_file.name}")
        
        # Add demo script
        demo_script = project_root / "scripts" / "indysim_demo.py"
        if demo_script.exists():
            zf.write(demo_script, "demo/indysim_demo.py")
            print(f"    Added: demo/indysim_demo.py")
        
        # Add demo README
        zf.writestr("demo/README.md", create_demo_readme())
        print(f"    Added: demo/README.md")
    
    print(f"\n  Created: {zip_name}")
    
    # Create main README
    print("\n[3/4] Creating README...")
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(create_main_readme(timestamp))
    print(f"  Created: README.md")
    
    # Summary
    print("\n[4/4] Calculating sizes...")
    
    total_size = 0
    for f in output_dir.iterdir():
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size / 1024:.1f} KB")
    
    print(f"\n  Total: {total_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 60)
    print("  Packaging Complete!")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Package supplementary materials for PI review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--output-dir', type=str, default='deliverables',
                       help='Output directory (default: deliverables)')
    
    args = parser.parse_args()
    
    package_supplements(Path(args.output_dir))
    
    return 0


if __name__ == '__main__':
    exit(main())





