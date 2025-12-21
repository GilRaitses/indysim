#!/usr/bin/env python3
"""
Generate Figure 6: Combined PSTH panels in a 3x2 grid.

Layout:
    A (0-to-250 Constant)    B (0-to-250 Cycling)    E (All Conditions Overlaid)
    C (50-to-250 Constant)   D (50-to-250 Cycling)   F (PSTH vs LNP Model)

Usage:
    python scripts/generate_figure6_combined.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

# Set font
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['axes.titleweight'] = 'ultralight'


def crop_4panel(img_4panel):
    """Crop the 4-panel image into 4 individual panels with minimal margin loss."""
    img_array = np.array(img_4panel)
    h, w = img_array.shape[:2]
    
    # The 4-panel is arranged as:
    # [A] [B]
    # [C] [D]
    mid_h = h // 2
    mid_w = w // 2
    
    # Very small margins to preserve axes and titles
    margin_h = int(h * 0.005)
    margin_w = int(w * 0.005)
    
    panel_a = img_array[:mid_h - margin_h, :mid_w - margin_w]
    panel_b = img_array[:mid_h - margin_h, mid_w + margin_w:]
    panel_c = img_array[mid_h + margin_h:, :mid_w - margin_w]
    panel_d = img_array[mid_h + margin_h:, mid_w + margin_w:]
    
    return panel_a, panel_b, panel_c, panel_d


def resize_to_target(img_array, target_size):
    """Resize an image array to target size (width, height)."""
    img = Image.fromarray(img_array)
    img_resized = img.resize(target_size, Image.LANCZOS)
    return np.array(img_resized)


def main():
    fig_dir = Path('figures')
    output_path = fig_dir / 'figure6_psth_kernels_combined.png'
    
    # Load the component figures
    psth_4panel_path = fig_dir / 'peristimulus_turn_rate_4panel.png'
    psth_overlaid_path = fig_dir / 'peristimulus_turn_rate_overlaid.png'
    psth_vs_kernel_path = fig_dir / 'psth_vs_kernel_verification.png'
    
    # Check all files exist
    missing = []
    for p in [psth_4panel_path, psth_overlaid_path, psth_vs_kernel_path]:
        if not p.exists():
            missing.append(str(p))
    
    if missing:
        print(f"Error: Missing files: {', '.join(missing)}")
        print("Run generate_peristimulus_rate_figures.py and generate_psth_vs_kernel_verification.py first.")
        return 1
    
    # Load images
    img_4panel = Image.open(psth_4panel_path)
    img_overlaid = Image.open(psth_overlaid_path)
    img_psth_kernel = Image.open(psth_vs_kernel_path)
    
    # Crop the 4-panel into individual panels
    panel_a, panel_b, panel_c, panel_d = crop_4panel(img_4panel)
    
    # Define target size for all panels (same aspect ratio)
    target_width = 550
    target_height = 450
    target_size = (target_width, target_height)
    
    # Resize all panels to same size
    panel_a_resized = resize_to_target(panel_a, target_size)
    panel_b_resized = resize_to_target(panel_b, target_size)
    panel_c_resized = resize_to_target(panel_c, target_size)
    panel_d_resized = resize_to_target(panel_d, target_size)
    panel_e_resized = resize_to_target(np.array(img_overlaid), target_size)
    panel_f_resized = resize_to_target(np.array(img_psth_kernel), target_size)
    
    # Create figure with extra space at top for labels
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Adjust subplot positions to leave room for labels above
    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98, 
                        wspace=0.08, hspace=0.12)
    
    # All panels in order: A, B, E (top), C, D, F (bottom)
    panels = [
        (axes[0, 0], panel_a_resized, 'A'),
        (axes[0, 1], panel_b_resized, 'B'),
        (axes[0, 2], panel_e_resized, 'E'),
        (axes[1, 0], panel_c_resized, 'C'),
        (axes[1, 1], panel_d_resized, 'D'),
        (axes[1, 2], panel_f_resized, 'F'),
    ]
    
    for ax, img, label in panels:
        ax.imshow(img)
        ax.axis('off')
        # Place label ABOVE the plot (y > 1.0 in axes coordinates)
        ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=20, 
                fontweight='bold', va='bottom', ha='left')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved Figure 6 (3x2 grid) to {output_path}")
    return 0


if __name__ == '__main__':
    exit(main())
