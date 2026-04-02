"""
fig1_z_stack_propagation.py
───────────────────────────
Generates a macro-level diagram showing the bidirectional, 
sequential Z-stack registration loop.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
C_ANCHOR  = '#00AEEF'  
C_RAW     = '#9E9E9E'  
C_ALIGNED = '#4CAF50'  
C_BG      = '#FAFAFA'

def draw_slice(ax, cx, cy, w, h, label, sublabel, facecolor, edgecolor, zorder=3):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle='round,pad=0.05',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=2, zorder=zorder
    ))
    ax.text(cx, cy + 0.15, label, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#111111', zorder=zorder+1)
    ax.text(cx, cy - 0.2, sublabel, ha='center', va='center', 
            fontsize=8, color='#333333', zorder=zorder+1)

def draw_dependency_arrow(ax, x0, y0, x1, y1, label):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5, 
                                connectionstyle="arc3,rad=-0.3", mutation_scale=15), 
                zorder=5)
    
    mx, my = (x0+x1)/2, (y0+y1)/2 + 0.6
    ax.text(mx, my, label, ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", fc="#FCE4F3", ec="#EC008C", lw=1),
            zorder=6)

def fig1_z_stack_propagation(out_path):
    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    fig.suptitle('Bidirectional Z-Stack Propagation',
                 fontsize=16, fontweight='bold', y=0.92, color='#111111')

    # Dimensions
    W = 1.8
    H = 1.2
    Y_SLICE = 4.0

    # Positions (representing Z-indices)
    centers_x = [2.0, 5.0, 8.0, 11.0, 14.0]
    labels = ["Z_{c-2}", "Z_{c-1}", "Z_{c} (Anchor)", "Z_{c+1}", "Z_{c+2}"]
    
    # 1. Draw the Anchor
    draw_slice(ax, centers_x[2], Y_SLICE, W, H, labels[2], 
               "Direct copy to\naligned_vol", '#E3F5FD', C_ANCHOR)

    # 2. Draw Forward Pass Nodes
    draw_slice(ax, centers_x[3], Y_SLICE, W, H, labels[3], "Aligned to Z_{c}", '#E8F5E9', C_ALIGNED)
    draw_slice(ax, centers_x[4], Y_SLICE, W, H, labels[4], "Aligned to Z_{c+1}", '#E8F5E9', C_ALIGNED)
    
    # Forward Pass Arrows
    draw_dependency_arrow(ax, centers_x[2] + W/2, Y_SLICE + H/2, centers_x[3] - W/2, Y_SLICE + H/2, "register_slice()")
    draw_dependency_arrow(ax, centers_x[3] + W/2, Y_SLICE + H/2, centers_x[4] - W/2, Y_SLICE + H/2, "register_slice()")

    # 3. Draw Backward Pass Nodes
    draw_slice(ax, centers_x[1], Y_SLICE, W, H, labels[1], "Aligned to Z_{c}", '#E8F5E9', C_ALIGNED)
    draw_slice(ax, centers_x[0], Y_SLICE, W, H, labels[0], "Aligned to Z_{c-1}", '#E8F5E9', C_ALIGNED)

    # Backward Pass Arrows (Fixed is right, Moving is left)
    draw_dependency_arrow(ax, centers_x[2] - W/2, Y_SLICE + H/2, centers_x[1] + W/2, Y_SLICE + H/2, "register_slice()")
    draw_dependency_arrow(ax, centers_x[1] - W/2, Y_SLICE + H/2, centers_x[0] + W/2, Y_SLICE + H/2, "register_slice()")

    # Pass Labels
    ax.text(3.5, 6.5, "← BACKWARD PASS (fixed_offset = +1)", ha='center', va='center', fontsize=10, fontweight='bold', color='#555555')
    ax.text(12.5, 6.5, "FORWARD PASS (fixed_offset = -1) →", ha='center', va='center', fontsize=10, fontweight='bold', color='#555555')

    # Final Stack Container Bracket
    ax.annotate('', xy=(1.0, 2.5), xytext=(15.0, 2.5),
                arrowprops=dict(arrowstyle='-', color='#37474F', lw=2), zorder=1)
    ax.plot([1.0, 1.0], [2.5, 3.0], color='#37474F', lw=2)
    ax.plot([15.0, 15.0], [2.5, 3.0], color='#37474F', lw=2)
    
    ax.text(8.0, 1.8, "Final Output: aligned_vol Stack (Z, C, H, W)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#37474F')
    
    ax.text(8.0, 1.2, "Each output slice serves as the 'Fixed' reference for the subsequent 'Moving' raw slice.", 
            ha='center', va='center', fontsize=9, color='#555555', style='italic')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='fig1_z_stack_propagation.png')
    args = parser.parse_args()
    fig1_z_stack_propagation(args.out)