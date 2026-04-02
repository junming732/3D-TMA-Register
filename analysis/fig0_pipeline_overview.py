"""
fig0_conceptual_pipeline.py
─────────────────────────
Generates a high-level, conceptual block diagram of the 
AKAZE Affine → RoMaV2 Dense Warp registration pipeline.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
C_FIXED   = '#00AEEF'
C_MOVING  = '#EC008C'
C_ALGO    = '#607D8B'
C_DATA    = '#E0E0E0'
C_OUT     = '#4CAF50'
C_NCC     = '#FF9800'  # Orange for NCC metrics
C_BG      = '#FAFAFA'

def draw_block(ax, cx, cy, w, h, title, subtitle=None, facecolor='white', edgecolor='#333', lw=1.5, zorder=3):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle='round,pad=0.05',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder
    ))
    ax.text(cx, cy + (0.15 if subtitle else 0), title, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#111', zorder=zorder+1)
    if subtitle:
        ax.text(cx, cy - 0.2, subtitle, ha='center', va='center', 
                fontsize=8, color='#444', zorder=zorder+1)

def draw_arrow(ax, x0, y0, x1, y1, label=None, rad=0.0, color='#555', lw=1.5):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, 
                                mutation_scale=12, connectionstyle=connectionstyle), 
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.25, label, ha='center', va='center', fontsize=8, 
                color='#333', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)

def draw_metric_badge(ax, cx, cy, text):
    """Renders the NCC evaluations as standalone metrics attached to the data flow."""
    ax.text(cx, cy, text, ha='center', va='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.3", fc=C_NCC, ec="none"), zorder=5)
    ax.plot(cx, cy, marker='o', markersize=4, color=C_NCC, zorder=4)

def fig0_conceptual_pipeline(out_path):
    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    fig.suptitle('Registration Pipeline: Global to Local Alignment',
                 fontsize=15, fontweight='bold', y=0.92, color='#111')

    # Grid Coordinates
    W, H = 2.0, 1.2
    X_RAW = 1.5
    X_PREP = 4.5
    X_AKAZE = 8.2    
    X_ROMA = 12.0    

    Y_FIXED = 6.0    
    Y_ALGO = 4.0     
    Y_MOVING = 2.0   

    # ── Stage 1: Inputs ────────────────────────────────────────────────────────
    draw_block(ax, X_RAW, Y_FIXED, W, H, "Fixed Slice", "Target Reference", '#E3F5FD', C_FIXED)
    draw_block(ax, X_RAW, Y_MOVING, W, H, "Moving Slice", "Raw Volume", '#FCE4F3', C_MOVING)

    # ── Stage 2: Pre-processing ────────────────────────────────────────────────
    draw_block(ax, X_PREP, Y_FIXED, W, H, "Pre-processing", "Norm & Tissue Mask", 'white', C_FIXED)
    draw_block(ax, X_PREP, Y_MOVING, W, H, "Pre-processing", "Norm & Tissue Mask", 'white', C_MOVING)

    draw_arrow(ax, X_RAW + W/2, Y_FIXED, X_PREP - W/2, Y_FIXED, color=C_FIXED)
    draw_arrow(ax, X_RAW + W/2, Y_MOVING, X_PREP - W/2, Y_MOVING, color=C_MOVING)

    # ── Stage 3: L0 Global Alignment ───────────────────────────────────────────
    draw_block(ax, X_AKAZE, Y_ALGO, W*1.1, H*1.1, "L0: AKAZE Affine", "Global Pre-alignment", '#F5F5F5', C_ALGO)
    draw_block(ax, X_AKAZE, Y_MOVING, W*1.1, H*1.1, "Affine Volume", "Shift & Rotate Fixed", '#FCE4F3', C_MOVING)

    # Feeds into AKAZE
    draw_arrow(ax, X_PREP + W/2, Y_FIXED, X_AKAZE - 0.2, Y_ALGO + H*0.55, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_PREP + W/2, Y_MOVING + 0.2, X_AKAZE - W*0.55, Y_ALGO - 0.2, rad=0.15, color=C_MOVING)
    
    # AKAZE computes and applies to the moving volume
    draw_arrow(ax, X_AKAZE, Y_ALGO - H*0.55, X_AKAZE, Y_MOVING + H*0.55, label="Apply Transform")
    
    # Data trunk flowing into Affine Volume
    draw_arrow(ax, X_PREP + W/2, Y_MOVING, X_AKAZE - W*0.55, Y_MOVING, color=C_MOVING)
    
    # Metric 1: Raw Correlation
    draw_metric_badge(ax, (X_PREP + X_AKAZE) / 2, Y_MOVING, "NCC Raw")

    # ── Stage 4: L1 Local Alignment ────────────────────────────────────────────
    draw_block(ax, X_ROMA, Y_ALGO, W*1.1, H*1.1, "L1: RoMaV2 Warp", "Local Dense Deformation", '#F5F5F5', C_ALGO)
    draw_block(ax, X_ROMA, Y_MOVING, W*1.1, H*1.1, "Aligned Volume", "Final Registered Data", '#E8F5E9', C_OUT, lw=2)
    
    # Feeds into RoMaV2
    draw_arrow(ax, X_PREP + W/2, Y_FIXED + 0.2, X_ROMA - W*0.55, Y_ALGO + 0.4, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_AKAZE + W*0.55, Y_MOVING + 0.2, X_ROMA - W*0.55, Y_ALGO - 0.2, rad=0.1, color=C_MOVING)

    # RoMaV2 computes and applies to the moving volume
    draw_arrow(ax, X_ROMA, Y_ALGO - H*0.55, X_ROMA, Y_MOVING + H*0.55, label="Apply Warp")
    
    # Data trunk flowing from Affine into Aligned Volume
    draw_arrow(ax, X_AKAZE + W*0.55, Y_MOVING, X_ROMA - W*0.55, Y_MOVING, color=C_MOVING)
    
    # Metric 2: Affine Correlation
    draw_metric_badge(ax, (X_AKAZE + X_ROMA) / 2, Y_MOVING, "NCC Affine")

    # Metric 3: Final Correlation
    draw_metric_badge(ax, X_ROMA + W*0.55 + 0.8, Y_MOVING, "NCC Warp\n(Final)")
    ax.plot([X_ROMA + W*0.55, X_ROMA + W*0.55 + 0.8], [Y_MOVING, Y_MOVING], 
            color=C_NCC, linestyle=':', lw=1.5, zorder=1)

    # ── Legend ─────────────────────────────────────────────────────────────────
    ax.legend(handles=[
        mpatches.Patch(facecolor='#E3F5FD', edgecolor=C_FIXED,  label='Fixed Reference Data'),
        mpatches.Patch(facecolor='#FCE4F3', edgecolor=C_MOVING, label='Moving Data Stream'),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor=C_ALGO,   label='Algorithms'),
        mpatches.Patch(facecolor=C_NCC,     edgecolor='none',   label='NCC Metrics'),
    ], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='fig0_conceptual_pipeline.png')
    args = parser.parse_args()
    fig0_conceptual_pipeline(args.out)