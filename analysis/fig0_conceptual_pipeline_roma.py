import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
C_FIXED   = '#00AEEF'
C_MOVING  = '#EC008C'
C_ALGO    = '#607D8B'
C_OUT     = '#4CAF50'
C_FILTER  = '#9C27B0'  
C_FAIL    = '#D32F2F'  
C_BG      = '#FAFAFA'

def draw_block(ax, cx, cy, w, h, title, subtitle=None, facecolor='white', edgecolor='#333', lw=2.5, zorder=3):
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

def draw_metric_badge(ax, cx, cy, text, color=C_FILTER):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=24, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="none"), zorder=5)
    ax.plot(cx, cy, marker='o', markersize=10, color=color, zorder=4)

def fig0_conceptual_pipeline(out_path):
    fig, ax = plt.subplots(figsize=(28, 13), dpi=300)
    
    # Expanded coordinate limits to fit larger boxes perfectly
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Main Title
    fig.suptitle('Pipeline C: Learning-Based Dense Matching with RoMaV2',
                 fontsize=48, fontweight='bold', y=0.96, color='#111')

    # Expanded block dimensions
    W_sm, H_sm = 5.6, 2.6 
    W_lg, H_lg = 7.4, 3.2 
    
    # Adjusted X spacing
    X_RAW = 3.2
    X_PREP = 9.8
    X_AKAZE = 17.6    
    X_ROMA = 26.6    

    # Adjusted Y spacing
    Y_FIXED = 11.2    
    Y_ALGO = 7.0     
    Y_MOVING = 2.8   

    # ── Stage 1: Inputs ────────────────────────────────────────────────────────
    draw_block(ax, X_RAW, Y_FIXED, W_sm, H_sm, "Fixed Slice", "Target Reference", '#E3F5FD', C_FIXED)
    draw_block(ax, X_RAW, Y_MOVING, W_sm, H_sm, "Moving Slice", "Raw Unaligned Volume", '#FCE4F3', C_MOVING)

    # ── Stage 2: Pre-processing ────────────────────────────────────────────────
    draw_block(ax, X_PREP, Y_FIXED, W_sm, H_sm, "Pre-processing", "Log/Lin Norm & Mask", 'white', C_FIXED)
    draw_block(ax, X_PREP, Y_MOVING, W_sm, H_sm, "Pre-processing", "Log/Lin Norm & Mask", 'white', C_MOVING)

    draw_arrow(ax, X_RAW + W_sm/2, Y_FIXED, X_PREP - W_sm/2, Y_FIXED, color=C_FIXED)
    draw_arrow(ax, X_RAW + W_sm/2, Y_MOVING, X_PREP - W_sm/2, Y_MOVING, color=C_MOVING)

    # ── Stage 3: L0 Global Alignment ───────────────────────────────────────────
    draw_block(ax, X_AKAZE, Y_ALGO, W_lg, H_lg, "L0: AKAZE Affine", "RANSAC + Lowe Ratio Test", '#F5F5F5', C_ALGO)
    draw_block(ax, X_AKAZE, Y_MOVING, W_lg, H_lg, "Affine Volume", "Coarse Rigid Prealignment", '#FCE4F3', C_MOVING)

    # Feeds into AKAZE (Log-Norm)
    draw_arrow(ax, X_PREP + W_sm/2, Y_FIXED, X_AKAZE - W_lg/2 + 0.4, Y_ALGO + H_lg*0.5, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_PREP + W_sm/2, Y_MOVING + 0.3, X_AKAZE - W_lg/2, Y_ALGO - 0.3, rad=0.15, color=C_MOVING)
    
    # AKAZE computes and applies
    draw_arrow(ax, X_AKAZE, Y_ALGO - H_lg*0.5, X_AKAZE, Y_MOVING + H_lg*0.5, label="Apply Transform")
    
    # Data trunk flowing into Affine Volume + Fallback definition
    draw_arrow(ax, X_PREP + W_sm/2, Y_MOVING, X_AKAZE - W_lg/2, Y_MOVING, color=C_MOVING)
    ax.text((X_PREP + X_AKAZE - W_lg/2)/2 + 0.4, Y_MOVING - 1.5, "Fail L0 -> Warp Raw Image", ha='center', fontsize=26, color=C_FAIL, fontweight='bold')

    # ── Stage 4: L1 Local Alignment ────────────────────────────────────────────
    draw_block(ax, X_ROMA, Y_ALGO, W_lg, H_lg, "L1: RoMaV2 Dense Warp", "Learning-Based Pseudo-RGB", '#F5F5F5', C_ALGO)
    draw_block(ax, X_ROMA, Y_MOVING, W_lg, H_lg, "Aligned Volume", "Final Registered Output", '#E8F5E9', C_OUT, lw=3)
    
    # Feeds into RoMaV2 (Lin-Norm)
    draw_arrow(ax, X_PREP + W_sm/2, Y_FIXED + 0.4, X_ROMA - W_lg/2 + 0.4, Y_ALGO + H_lg*0.5, rad=-0.1, color=C_FIXED)
    draw_arrow(ax, X_AKAZE + W_lg/2, Y_MOVING + 0.3, X_ROMA - W_lg/2, Y_ALGO - 0.3, rad=0.1, color=C_MOVING)

    # RoMaV2 computes and applies
    draw_arrow(ax, X_ROMA, Y_ALGO - H_lg*0.5, X_ROMA, Y_MOVING + H_lg*0.5, label="Warp Channels")
    
    # Data trunk flowing from Affine into Aligned Volume + Fallback definition
    draw_arrow(ax, X_AKAZE + W_lg/2, Y_MOVING, X_ROMA - W_lg/2, Y_MOVING, color=C_MOVING)
    ax.text((X_AKAZE + W_lg/2 + X_ROMA - W_lg/2)/2, Y_MOVING - 1.5, "Fail L1 -> Retain Affine", ha='center', fontsize=26, color=C_FAIL, fontweight='bold')

    # Metric 2: Vector Filtering Gates (Restored but modified)
    draw_metric_badge(ax, (X_AKAZE + X_ROMA)/2, Y_ALGO - 1.6, "Warp Filters:\nConf >= 0.5\nCap <= 200px", color=C_FILTER)

    # Metric 3: Global Output Gate (Replaced Sanity Gate with NCC)
    draw_metric_badge(ax, X_ROMA, Y_MOVING - 1.9, "Global Gate:\n>= 5% NCC Gain", color=C_FILTER)

    # ── Legend ─────────────────────────────────────────────────────────────────
    ax.legend(handles=[
        mpatches.Patch(facecolor='#E3F5FD', edgecolor=C_FIXED,  label='Fixed Reference', lw=2),
        mpatches.Patch(facecolor='#FCE4F3', edgecolor=C_MOVING, label='Moving Stream', lw=2),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor=C_ALGO,   label='Algorithms', lw=2),
        mpatches.Patch(facecolor=C_FILTER,  edgecolor='none',   label='Filters & NCC Gates'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=26, frameon=False,
      handlelength=2.5, handleheight=1.5, columnspacing=1.8)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='fig0_conceptual_pipeline_romav2_updated.png')
    args = parser.parse_args()
    fig0_conceptual_pipeline(args.out)