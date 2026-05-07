"""
fig_pipeline_clean_ultra_fill.py
─────────────────────
Single-row pipeline diagram, 7 stages.
Optimized for MAXIMAL block fill: massive titles and nearly equal-sized subtitles.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────────
C_REG    = '#6A1B9A'   # deep purple  – registration
C_SEG    = '#AD1457'   # deep pink    – segmentation
C_DEN    = '#E65100'   # deep orange  – denoising
C_QUA    = '#37474F'   # blue-grey    – marker quantification
C_REC    = '#1565C0'   # dark blue    – 3D reconstruction
C_ASN    = '#2E7D32'   # dark green   – cell-type assignment
C_CMP    = '#00695C'   # teal         – spatial comparison
C_BG     = '#F7F9FA'

COLORS   = [C_REG, C_SEG, C_DEN, C_QUA, C_REC, C_ASN, C_CMP]

# ── Helpers ───────────────────────────────────────────────────────────────────

PAD = 0.12   # Padding for block rendering AND exact arrow boundary offsets

def draw_block(ax, cx, cy, w, h, title, subtitle=None,
               facecolor='white', edgecolor='#333', lw=3.0, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={PAD}',
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder, clip_on=False,
    ))
    
    # Symmetrical offset to tightly pack the huge text inside the box
    dy = 0.35 if subtitle else 0
    
    # Massive Title font
    ax.text(cx, cy + dy, title,
            ha='center', va='center',
            fontsize=40, fontweight='bold', color='#111',
            zorder=zorder+1, linespacing=1.1)
    
    if subtitle:
        # Massive Subtitle font (nearly as big as title)
        ax.text(cx, cy - 0.35, subtitle,
                ha='center', va='center',
                fontsize=30, color='#222', # Darkened for better contrast
                zorder=zorder+1, linespacing=1.15)


def draw_arrow(ax, x_tail, y_tail, x_tip, y_tip, color='#666', lw=3.5, label=None):
    """Fixed-size arrowhead regardless of shaft length."""
    dx, dy  = x_tip - x_tail, y_tip - y_tail
    length  = np.hypot(dx, dy)
    ux, uy  = dx / length, dy / length
    SHRINK  = 0.12          
    ax.plot([x_tail, x_tip - ux*SHRINK], [y_tail, y_tip - uy*SHRINK],
            color=color, lw=lw, solid_capstyle='butt', zorder=2)
    ax.annotate('',
                xy=(x_tip, y_tip),
                xytext=(x_tip - ux*0.001, y_tip - uy*0.001),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=28), 
                zorder=3)
    if label:
        mx, my = (x_tail+x_tip)/2, (y_tail+y_tip)/2
        ax.text(mx, my + 0.18, label,
                ha='center', va='bottom', fontsize=16,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9),
                zorder=5)


def stage_badge(ax, cx, cy, text, color):
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=36,
            color=color, fontstyle='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.30',
                      fc=color+'18', ec=color+'66', lw=1.8),
            zorder=6)


# ── Build ─────────────────────────────────────────────────────────────────────

def build(out_path):
    # Shortened canvas vertically since bottom tags are removed
    fig, ax = plt.subplots(figsize=(34, 6.2), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.axis('off')

    fig.suptitle(
        '3D-TMA Multiplex Immunofluorescence Analysis Pipeline',
        fontsize=42, fontweight='bold', y=0.95, color='#1A1A2E',
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    N      = 7
    BW     = 4.00          
    BH     = 2.40          
    MARGIN = 0.50          
    W      = 34.0          
    SPAN   = W - 2*MARGIN  

    xs = [MARGIN + BW/2 + i * (SPAN - BW) / (N-1) for i in range(N)]

    Y_BLK  = 3.20            
    Y_LBL  = Y_BLK + BH/2 + PAD + 0.65   
    Y_BOT  = Y_BLK - BH/2 - PAD          

    ax.set_xlim(0, W)
    # Tightly crop the Y axis to the components and the legend
    ax.set_ylim(0.1, Y_LBL + 0.5)

    stages = [
        ('Stage 1', C_REG,
         'Registration',
         'AKAZE affine pre-align\n+ RoMaV2 dense warp'),

        ('Stage 2', C_SEG,
         'Segmentation',
         'CellPose deep-learning\nnuclear instance detect'),

        ('Stage 3', C_DEN,
         'Artefact\nDenoising',
         'Inpainting + top-hat /\nGaussian sub. removal'),

        ('Stage 4', C_QUA,
         'Marker\nQuantification',
         'GMM/BIC intensity thresh\n→ binary positive calls'),

        ('Stage 5', C_REC,
         '3D Cell\nReconstruction',
         'Sparse connected comps\n+ Z-span severing'),

        ('Stage 6', C_ASN,
         'Cell-type\nAssignment',
         'Codebook lookup per cell\n+ majority vote in 3D'),

        ('Stage 7', C_CMP,
         '2D–3D Spatial\nComparison',
         'Cell-type density · NN\nPermutation interact scores'),
    ]

    for x, (badge, col, title, sub) in zip(xs, stages):
        stage_badge(ax, x, Y_LBL, badge, col)
        draw_block(ax, x, Y_BLK, BW, BH, title, sub,
                   facecolor=col+'12', edgecolor=col, lw=2.5)

    h_labels = [
        'deformation\nmaps\n+ masks',
        'warped\nmasks\n+ images',
        'denoised\nvolume',
        'per-slice\ncell\nmeasures',
        'identity map\n+ catalogue',
        'typed records\n+ 3D catalog',
    ]
    
    for i in range(N - 1):
        col    = stages[i][1]
        x_tail = xs[i]   + BW/2 + PAD
        x_tip  = xs[i+1] - BW/2 - PAD
        
        draw_arrow(ax, x_tail, Y_BLK, x_tip, Y_BLK,
                   color=col, lw=3.5, label=h_labels[i])

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc=c+'20', ec=c, label=lbl)
        for c, lbl in zip(COLORS, [
            'Registration', 'Segmentation', 'Denoising',
            'Marker quantification', '3D reconstruction',
            'Cell-type assignment', 'Spatial comparison',
        ])
    ]
    ax.legend(handles=handles,
              loc='lower center', bbox_to_anchor=(0.5, 0.0),
              ncol=7, fontsize=26, frameon=False,
              handlelength=1.5, handleheight=1.2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='fig_pipeline_clean_ultra_fill.png')
    args = parser.parse_args()
    build(args.out)