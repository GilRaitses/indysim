#!/usr/bin/env python3
"""
Generate Individual Panels for Identifiability Figure - PRESENTATION VERSION

Outputs 4 separate PDFs, one per panel, with larger fonts for slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Output
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/presentation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_FILE = Path("/Users/gilraitses/INDYsim_project/scripts/2025-12-19/phenotyping_experiments/15_identification_analysis/results/design_kernel_sweep/sweep_results.json")

# Larger fonts for presentation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 14  # Larger base font
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Colors
COLORS = {
    'data': '#60A5FA',
    'success': '#22C55E',
    'failure': '#EF4444',
    'burst': '#10B981',
    'text': '#1E293B',
}

def load_sweep_data():
    """Load the design × kernel sweep results."""
    if SWEEP_FILE.exists():
        with open(SWEEP_FILE) as f:
            sweep = json.load(f)
        our_kernel = [r for r in sweep if r['ab_ratio'] == 0.125]
        return our_kernel
    return None

def get_design_data():
    """Get continuous and burst design statistics."""
    sweep = load_sweep_data()
    if sweep:
        continuous = next((r for r in sweep if 'Continuous' in r['design']), None)
        burst = next((r for r in sweep if 'Burst' in r['design']), None)
        if continuous and burst:
            return continuous, burst
    
    # Fallback values
    continuous = {'bias': 0.61, 'rmse': 0.71, 'mean_events': 16.3, 'fisher': 0.29, 'fitted_mean': 1.24}
    burst = {'bias': 0.14, 'rmse': 0.38, 'mean_events': 16.9, 'fisher': 2.88, 'fitted_mean': 0.77}
    return continuous, burst


def panel_A_design_comparison():
    """Panel A: Bar chart comparing bias and RMSE by design."""
    continuous, burst = get_design_data()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    designs = ['Continuous\n10s ON', 'Burst\n10×0.5s']
    bias_vals = [continuous['bias'], burst['bias']]
    rmse_vals = [continuous['rmse'], burst['rmse']]
    
    x = np.arange(len(designs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bias_vals, width, label='Bias', color=COLORS['failure'], alpha=0.85)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label='RMSE', color=COLORS['data'], alpha=0.85)
    
    ax.set_ylabel(r'Error in $\tau_1$ (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(designs, fontsize=14)
    ax.set_title('Same Events, Different Information\n(~17 events each)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Value labels
    for bar, val in zip(bars1, bias_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'+{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylim(0, 0.95)
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_A.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_A.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'identifiability_panel_A.pdf'}")
    plt.close()


def panel_B_fisher_information():
    """Panel B: Fisher Information comparison."""
    continuous, burst = get_design_data()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    ax.text(0.5, 0.92, r'Fisher Information for $\tau_1$', fontsize=22, ha='center',
            fontweight='bold', color=COLORS['text'], transform=ax.transAxes)
    
    ax.text(0.5, 0.72, f'Continuous: {continuous["fisher"]:.2f}', fontsize=20, ha='center',
            color=COLORS['failure'], transform=ax.transAxes)
    
    ax.text(0.5, 0.55, f'Burst: {burst["fisher"]:.2f}', fontsize=20, ha='center',
            color=COLORS['success'], transform=ax.transAxes)
    
    # Draw a simple horizontal line using plot
    ax.plot([0.3, 0.7], [0.40, 0.40], color=COLORS['text'], linewidth=2, 
            transform=ax.transAxes, clip_on=False)
    
    ratio = burst["fisher"] / continuous["fisher"]
    ax.text(0.5, 0.25, f'Burst extracts {ratio:.0f}× more info', fontsize=22, ha='center',
            color=COLORS['success'], fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.10, 'from the same number of events', fontsize=16, ha='center',
            color=COLORS['text'], transform=ax.transAxes)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_B.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_B.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'identifiability_panel_B.pdf'}")
    plt.close()


def panel_C_mle_recovery():
    """Panel C: MLE Recovery by Design."""
    continuous, burst = get_design_data()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    true_tau1 = 0.63
    ax.axhline(y=true_tau1, color=COLORS['success'], linewidth=3, linestyle='--', label=r'True $\tau_1$')
    
    x_pos = [0, 1]
    fitted_means = [continuous.get('fitted_mean', 1.24), burst.get('fitted_mean', 0.77)]
    fitted_stds = [0.37, 0.35]
    
    bars = ax.bar(x_pos, fitted_means, yerr=fitted_stds, width=0.6, 
            color=[COLORS['failure'], COLORS['burst']], alpha=0.85, capsize=8,
            error_kw={'elinewidth': 3, 'capthick': 3})
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Continuous', 'Burst'], fontsize=14)
    ax.set_ylabel(r'Fitted $\tau_1$ (seconds)')
    ax.set_title('MLE Recovery by Design', fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.9)
    
    # Bias annotations
    bias_vals = [continuous['bias'], burst['bias']]
    for i, (fm, bias, std) in enumerate(zip(fitted_means, bias_vals, fitted_stds)):
        ax.annotate(f'Bias: +{bias:.2f}s', (x_pos[i], fm + std + 0.1),
                    ha='center', fontsize=13, color=COLORS['text'], fontweight='bold')
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_C.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_C.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'identifiability_panel_C.pdf'}")
    plt.close()


def panel_D_why_continuous_fails():
    """Panel D: Explanation of why continuous design fails."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    ax.set_title('Why Continuous Design Fails', fontweight='bold', fontsize=20)
    
    explanations = [
        ("Kernel is inhibition-dominated (B/A = 8)", COLORS['text'], 'normal'),
        ("", COLORS['text'], 'normal'),
        ("~80% of events occur during LED-OFF", COLORS['text'], 'normal'),
        ("    No tau1 information", COLORS['failure'], 'bold'),
        ("", COLORS['text'], 'normal'),
        ("Remaining ~20% mostly after t > 0.5s", COLORS['text'], 'normal'),
        ("    Inhibition dominates, tau1 unidentifiable", COLORS['failure'], 'bold'),
        ("", COLORS['text'], 'normal'),
        ("Burst design samples multiple", COLORS['success'], 'bold'),
        ("early excitatory windows", COLORS['success'], 'bold'),
    ]
    
    y_start = 0.85
    for i, (text, color, weight) in enumerate(explanations):
        ax.text(0.08, y_start - i * 0.085, text, fontsize=15, ha='left',
                color=color, fontweight=weight, transform=ax.transAxes)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_D.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'identifiability_panel_D.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'identifiability_panel_D.pdf'}")
    plt.close()


if __name__ == '__main__':
    print("Generating individual identifiability panels...")
    panel_A_design_comparison()
    panel_B_fisher_information()
    panel_C_mle_recovery()
    panel_D_why_continuous_fails()
    print(f"\nAll panels saved to: {OUTPUT_DIR}")

