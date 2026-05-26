"""
fig0_conceptual_pipeline_valis.py

Generates a conceptual block diagram of the VALIS Automated Registration pipeline.
Scaled and aligned to match the unified format of Pipelines A and C.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Color Palette ---
C_FIXED   = '#00AEEF'
C_MOVING  = '#EC008C'
C_ALGO    = '#607D8B'
C_OUT     = '#4CAF50'
C_BG      = '#FAFAFA'

def draw_block(ax, cx, cy, w, h, title, subtitle=None, facecolor='white', edgecolor='#333', lw=2.5, zorder=3):
    """Draws a main architectural block with a title and optional subtitle."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle='round,pad=0.08',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder
    ))
    dy = 0.5 if subtitle else 0
    ax.text(cx, cy + dy, title, ha='center', va='center', 
            fontsize=32, fontweight='bold', color='#111', zorder=zorder+1)
    if subtitle:
        ax.text(cx, cy - 0.45, subtitle, ha='center', va='center', 
                fontsize=30, color='#444', zorder=zorder+1)

def draw_arrow(ax, x0, y0, x1, y1, label=None, rad=0.0, color='#555', lw=2.5, ls='-'):
    """Draws a connecting arrow between blocks, optionally with a text label."""
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls,
                                mutation_scale=30, connectionstyle=connectionstyle), 
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.45, label, ha='center', va='center', fontsize=28, 
                color='#333', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92), zorder=4)

def generate_pipeline_figure(out_path):
    """Constructs and saves the full conceptual pipeline diagram."""
    fig, ax = plt.subplots(figsize=(28, 13), dpi=300)
    
    # Expanded coordinate limits to fit larger boxes perfectly
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Main Title
    fig.suptitle('Pipeline B: Automated Multi Resolution Framework',
                 fontsize=48, fontweight='bold', y=0.96, color='#111')

    # Expanded block dimensions
    W_sm, H_sm = 5.6, 2.6 
    W_lg, H_lg = 7.4, 3.2 
    
    # Spatial Configuration
    X_RAW = 3.2
    X_PREP = 9.8
    X_RIGID = 17.6    
    X_NONRIGID = 26.6    

    Y_FIXED = 11.2    
    Y_ALGO = 7.0     
    Y_MOVING = 2.8   

    # --- Stage 1: Inputs ---
    draw_block(ax, X_RAW, Y_FIXED, W_sm, H_sm, "Fixed Section", "Central Reference", '#E3F5FD', C_FIXED)
    draw_block(ax, X_RAW, Y_MOVING, W_sm, H_sm, "Moving Section", "Raw Serial Slice", '#FCE4F3', C_MOVING)

    # --- Stage 2: Pre-processing ---
    draw_block(ax, X_PREP, Y_FIXED, W_sm, H_sm, "Pre-processing", "Extract DAPI Channel", 'white', C_FIXED)
    draw_block(ax, X_PREP, Y_MOVING, W_sm, H_sm, "Pre-processing", "Extract DAPI Channel", 'white', C_MOVING)

    draw_arrow(ax, X_RAW + W_sm/2, Y_FIXED, X_PREP - W_sm/2, Y_FIXED, color=C_FIXED)
    draw_arrow(ax, X_RAW + W_sm/2, Y_MOVING, X_PREP - W_sm/2, Y_MOVING, color=C_MOVING)

    # --- Stage 3: Two-Stage Rigid Alignment ---
    draw_block(ax, X_RIGID, Y_ALGO, W_lg, H_lg, "Two-Stage Rigid", "Rotation Est. -> DISK/LightGlue", '#F5F5F5', C_ALGO)
    draw_block(ax, X_RIGID, Y_MOVING, W_lg, H_lg, "Rigid Volume", "Composed Transformation", '#FCE4F3', C_MOVING)

    # Feeds into Rigid
    draw_arrow(ax, X_PREP + W_sm/2, Y_FIXED, X_RIGID - W_lg/2 + 0.4, Y_ALGO + H_lg*0.5, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_PREP + W_sm/2, Y_MOVING + 0.3, X_RIGID - W_lg/2, Y_ALGO - 0.3, rad=0.15, color=C_MOVING)
    
    # Computes and applies
    draw_arrow(ax, X_RIGID, Y_ALGO - H_lg*0.5, X_RIGID, Y_MOVING + H_lg*0.5, label="Apply Rigid")
    
    # Data trunk flowing
    draw_arrow(ax, X_PREP + W_sm/2, Y_MOVING, X_RIGID - W_lg/2, Y_MOVING, color=C_MOVING)

    # --- Stage 4: Non-Rigid Registration ---
    draw_block(ax, X_NONRIGID, Y_ALGO, W_lg, H_lg, "Automated Non-Rigid", "Internal Deformation Scaling", '#F5F5F5', C_ALGO)
    draw_block(ax, X_NONRIGID, Y_MOVING, W_lg, H_lg, "Aligned Volume", "Cropped & Stacked OME-TIFF", '#E8F5E9', C_OUT, lw=3)
    
    # Feeds into Non-Rigid
    draw_arrow(ax, X_PREP + W_sm/2, Y_FIXED + 0.4, X_NONRIGID - W_lg/2 + 0.4, Y_ALGO + H_lg*0.5, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_RIGID + W_lg/2, Y_MOVING + 0.3, X_NONRIGID - W_lg/2, Y_ALGO - 0.3, rad=0.1, color=C_MOVING)

    # Computes and applies
    draw_arrow(ax, X_NONRIGID, Y_ALGO - H_lg*0.5, X_NONRIGID, Y_MOVING + H_lg*0.5, label="Warp & Merge")
    
    # Data trunk flowing
    draw_arrow(ax, X_RIGID + W_lg/2, Y_MOVING, X_NONRIGID - W_lg/2, Y_MOVING, color=C_MOVING)

    # --- Legend ---
    ax.legend(handles=[
        mpatches.Patch(facecolor='#E3F5FD', edgecolor=C_FIXED,  label='Fixed Reference', lw=2),
        mpatches.Patch(facecolor='#FCE4F3', edgecolor=C_MOVING, label='Moving Stream', lw=2),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor=C_ALGO,   label='Algorithms', lw=2),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=26, frameon=False,
      handlelength=2.5, handleheight=1.5, columnspacing=1.8)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Diagram saved successfully to: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Conceptual Pipeline Diagram for VALIS")
    parser.add_argument('--out', type=str, default='fig0_conceptual_pipeline_valis_final.png',
                        help='Output file path for the generated figure.')
    args = parser.parse_args()
    
    generate_pipeline_figure(args.out)