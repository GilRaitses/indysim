#!/usr/bin/env python3
"""
Generate supplementary tables for bioRxiv submission.

Creates:
1. Extended CV results table (per-experiment)
2. Model comparison table (AIC/BIC)
3. Coefficient comparison table (if GLMM available)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
VALIDATION_DIR = DATA_DIR / "validation"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "paper" / "tables"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_cv_table() -> str:
    """Generate extended CV results table in markdown."""
    cv_path = VALIDATION_DIR / "factorial_cv_results.json"
    if not cv_path.exists():
        return "CV results not found."
    
    cv = load_json(cv_path)
    
    lines = [
        "## Table S1: Leave-One-Experiment-Out Cross-Validation Results",
        "",
        "| Experiment | Condition | Empirical Events | Predicted Events | Rate Ratio | Status |",
        "|------------|-----------|------------------|------------------|------------|--------|"
    ]
    
    for exp_name, exp_data in cv["per_experiment"].items():
        # Shorten experiment name
        short_name = exp_name.split("@")[1].split("_202")[0] if "@" in exp_name else exp_name[:30]
        condition = exp_data.get("condition", "Unknown")
        empirical = exp_data["empirical_events"]
        predicted = exp_data["predicted_events"]
        ratio = exp_data["rate_ratio"]
        
        # Status based on 0.8-1.25 range
        status = "Pass" if 0.8 <= ratio <= 1.25 else "Fail"
        status_icon = "+" if status == "Pass" else "-"
        
        lines.append(
            f"| {short_name} | {condition} | {empirical:,} | {predicted:,.0f} | {ratio:.3f} | {status_icon} |"
        )
    
    # Add summary
    summary = cv["summary"]
    lines.extend([
        "",
        f"**Summary:** {summary['n_experiments']} experiments, "
        f"mean rate ratio = {summary['mean_rate_ratio']:.3f} +/- {summary['std_rate_ratio']:.3f}, "
        f"pass rate = {summary['pass_rate']:.1f}% ({summary['pass_count']}/{summary['n_experiments']})"
    ])
    
    return "\n".join(lines)


def generate_cv_table_latex() -> str:
    """Generate CV results table in LaTeX."""
    cv_path = VALIDATION_DIR / "factorial_cv_results.json"
    if not cv_path.exists():
        return "% CV results not found."
    
    cv = load_json(cv_path)
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Leave-one-experiment-out cross-validation results. Rate ratio within 0.8--1.25 indicates acceptable generalization.}",
        r"\label{tab:cv}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"\textbf{Experiment} & \textbf{Condition} & \textbf{Empirical} & \textbf{Predicted} & \textbf{Rate Ratio} & \textbf{Status} \\",
        r"\hline"
    ]
    
    for exp_name, exp_data in cv["per_experiment"].items():
        short_name = exp_name.split("@")[1].split("_202")[0] if "@" in exp_name else exp_name[:20]
        short_name = short_name.replace("_", r"\_")
        condition = exp_data.get("condition", "Unknown").replace("→", r"$\rightarrow$")
        empirical = exp_data["empirical_events"]
        predicted = exp_data["predicted_events"]
        ratio = exp_data["rate_ratio"]
        status = r"\checkmark" if 0.8 <= ratio <= 1.25 else r"$\times$"
        
        lines.append(
            f"{short_name} & {condition} & {empirical:,} & {predicted:,.0f} & {ratio:.3f} & {status} \\\\"
        )
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def generate_model_comparison_table() -> str:
    """Generate model comparison table."""
    fe_path = MODEL_DIR / "factorial_model_results.json"
    glmm_path = MODEL_DIR / "glmm_results.json"
    
    if not fe_path.exists():
        return "Fixed-effects results not found."
    
    fe = load_json(fe_path)
    
    lines = [
        "## Table S2: Model Comparison",
        "",
        "| Model | Parameters | AIC | Deviance | Notes |",
        "|-------|------------|-----|----------|-------|",
        f"| Fixed-effects NB-GLM | 8 | {fe['aic']:.1f} | {fe['deviance']:.1f} | Primary model |"
    ]
    
    if glmm_path.exists():
        glmm = load_json(glmm_path)
        # GLMM doesn't have traditional AIC, note this
        lines.append(
            f"| NB-GLMM (1|track) | 9 + 623 RE | -- | -- | Random intercepts |"
        )
    else:
        lines.append("| NB-GLMM | -- | -- | -- | Pending (run externally) |")
    
    return "\n".join(lines)


def generate_coefficient_table() -> str:
    """Generate coefficient comparison table."""
    fe_path = MODEL_DIR / "factorial_model_results.json"
    if not fe_path.exists():
        return "Fixed-effects results not found."
    
    fe = load_json(fe_path)
    
    lines = [
        "## Table S3: Factorial Model Coefficients",
        "",
        "| Parameter | Estimate | SE | 95% CI | p-value | Interpretation |",
        "|-----------|----------|----|---------|---------:|----------------|"
    ]
    
    interpretations = {
        "beta_0": "Baseline log-hazard (reference condition)",
        "beta_I": "Effect of 50->250 intensity on baseline",
        "beta_T": "Effect of cycling background on baseline",
        "beta_IT": "Intensity x Cycling interaction (baseline)",
        "alpha": "Reference suppression amplitude",
        "alpha_I": "Intensity effect on suppression (66% weaker)",
        "alpha_T": "Cycling effect on suppression (15% stronger)",
        "gamma": "LED-OFF rebound coefficient"
    }
    
    for param, vals in fe["coefficients"].items():
        interp = interpretations.get(param, "")
        ci = f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}]"
        pval = f"{vals['pvalue']:.2e}" if vals['pvalue'] < 0.001 else f"{vals['pvalue']:.4f}"
        sig = "*" if vals['significant'] else ""
        
        lines.append(
            f"| {param} | {vals['mean']:.4f} | {vals['se']:.4f} | {ci} | {pval}{sig} | {interp} |"
        )
    
    lines.extend([
        "",
        "*p < 0.05"
    ])
    
    return "\n".join(lines)


def generate_condition_amplitudes_table() -> str:
    """Generate condition-specific suppression amplitudes table."""
    fe_path = MODEL_DIR / "factorial_model_results.json"
    if not fe_path.exists():
        return "Results not found."
    
    fe = load_json(fe_path)
    
    lines = [
        "## Table S4: Condition-Specific Suppression Amplitudes",
        "",
        "| Condition | Amplitude | Events | Tracks | Interpretation |",
        "|-----------|-----------|--------|--------|----------------|"
    ]
    
    for condition, amplitude in fe["condition_amplitudes"].items():
        val_data = fe["validation"].get(condition, {})
        events = val_data.get("empirical_events", "--")
        tracks = val_data.get("n_tracks", "--")
        
        if "50" in condition:
            interp = "Reduced (partial adaptation)"
        elif "Temp" in condition or "Cycling" in condition:
            interp = "Enhanced (cycling background)"
        else:
            interp = "Reference condition"
        
        lines.append(
            f"| {condition} | {amplitude:.3f} | {events:,} | {tracks} | {interp} |"
        )
    
    return "\n".join(lines)


def main():
    """Generate all supplementary tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown tables
    tables = {
        "cv_results": generate_cv_table(),
        "model_comparison": generate_model_comparison_table(),
        "coefficients": generate_coefficient_table(),
        "condition_amplitudes": generate_condition_amplitudes_table()
    }
    
    # Save combined markdown
    combined_path = OUTPUT_DIR / "supplementary_tables.md"
    with open(combined_path, "w") as f:
        f.write("# Supplementary Tables\n\n")
        for name, content in tables.items():
            f.write(content)
            f.write("\n\n---\n\n")
    
    print(f"Saved combined tables to {combined_path}")
    
    # Save LaTeX CV table
    latex_cv = generate_cv_table_latex()
    latex_path = OUTPUT_DIR / "table_cv.tex"
    with open(latex_path, "w") as f:
        f.write(latex_cv)
    print(f"Saved LaTeX CV table to {latex_path}")
    
    # Print preview
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY TABLES PREVIEW")
    print("=" * 60)
    for name, content in tables.items():
        print(f"\n### {name.upper()} ###")
        print(content[:500] + "..." if len(content) > 500 else content)


if __name__ == "__main__":
    main()


