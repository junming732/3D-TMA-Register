"""
fig_pipeline_two_row.py
─────────────────────
Two-row pipeline diagram, 7 stages (4 top, 3 bottom).
Extremely large block and text sizes.
Row 1: Stages 1-4  →  Row 2: Stages 5-7
An S-curve connector routes between the rows to link the end of row 1 to row 2.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

# ── Palette ───────────────────────────────────────────────────────────────────
C_REG = '#6A1B9A'   # deep purple  – registration
C_SEG = '#AD1457'   # deep pink    – segmentation
C_DEN = '#E65100'   # deep orange  – denoising
C_QUA = '#37474F'   # blue-grey    – marker quantification
C_REC = '#1565C0'   # dark blue    – 3D reconstruction
C_ASN = '#2E7D32'   # dark green   – cell-type assignment
C_CMP = '#00695C'   # teal         – spatial comparison
C_BG  = '#F7F9FA'

COLORS = [C_REG, C_SEG, C_DEN, C_QUA, C_REC, C_ASN, C_CMP]

PAD = 0.12

# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_block(ax, cx, cy, w, h, title, subtitle=None,
               facecolor='white', edgecolor='#333', lw=3.0, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={PAD}',
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder, clip_on=False,
    ))
    # Spread the title and subtitle significantly for huge fonts
    dy = 0.65 if subtitle else 0
    ax.text(cx, cy + dy, title,
            ha='center', va='center',
            fontsize=28, fontweight='bold', color='#111',
            zorder=zorder+1, linespacing=1.15)
    if subtitle:
        ax.text(cx, cy - 0.50, subtitle,
                ha='center', va='center',
                fontsize=20, color='#333',
                zorder=zorder+1, linespacing=1.25)


def draw_arrow(ax, x_tail, y_tail, x_tip, y_tip, color='#666', lw=3.5, label=None):
    dx, dy = x_tip - x_tail, y_tip - y_tail
    length = np.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    SHRINK = 0.1
    ax.plot([x_tail, x_tip - ux*SHRINK], [y_tail, y_tip - uy*SHRINK],
            color=color, lw=lw, solid_capstyle='butt', zorder=2)
    ax.annotate('',
                xy=(x_tip, y_tip),
                xytext=(x_tip - ux*0.001, y_tip - uy*0.001),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=28),
                zorder=3)
    if label:
        mx, my = (x_tail + x_tip) / 2, (y_tail + y_tip) / 2
        ax.text(mx, my + 0.22, label,
                ha='center', va='bottom', fontsize=18,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.92),
                zorder=5)


def stage_badge(ax, cx, cy, text, color):
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=22,
            color=color, fontstyle='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc=color+'18', ec=color+'66', lw=2.0),
            zorder=6)


# ── Build ─────────────────────────────────────────────────────────────────────

def build(out_path):
    FW, FH = 32, 18  # Massively expanded canvas to fit the huge text proportionally
    fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.axis('off')

    fig.suptitle(
        '3D-TMA Multiplex Immunofluorescence Analysis Pipeline',
        fontsize=40, fontweight='bold', y=0.95, color='#1A1A2E',
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    BW = 6.60   # heavily expanded block width
    BH = 3.80   # heavily expanded block height
    MARGIN = 1.0

    Y_ROW1 = 12.00
    Y_ROW2 = 5.60
    BADGE_OFFSET = BH/2 + 1.0

    N1, N2 = 4, 3

    x_total = FW - 2*MARGIN
    xs1 = [MARGIN + BW/2 + i * (x_total - BW) / (N1-1) for i in range(N1)]

    xs2_span = xs1[2] - xs1[0] 
    xs2 = [xs1[0] + i * xs2_span / (N2-1) for i in range(N2)]

    ax.set_xlim(0, FW)
    ax.set_ylim(1.0, Y_ROW1 + BADGE_OFFSET + 0.8)

    stages = [
        ('Stage 1', C_REG, 'Registration', 'AKAZE affine pre-align\n+ RoMaV2 dense warp'),
        ('Stage 2', C_SEG, 'Segmentation', 'CellPose deep-learning\nnuclear instance detect'),
        ('Stage 3', C_DEN, 'Artefact\nDenoising', 'Inpainting + top-hat /\nGaussian sub. removal'),
        ('Stage 4', C_QUA, 'Marker\nQuantification', 'GMM/BIC intensity thresh\n→ binary positive calls'),
        ('Stage 5', C_REC, '3D Cell\nReconstruction', 'Sparse connected comps\n+ Z-span severing'),
        ('Stage 6', C_ASN, 'Cell-type\nAssignment', 'Codebook lookup per cell\n+ majority vote in 3D'),
        ('Stage 7', C_CMP, '2D–3D Spatial\nComparison', 'Cell-type density · NN\nPermutation interact scores'),
    ]

    # ── Row 1 blocks ──────────────────────────────────────────────────────────
    for i, x in enumerate(xs1):
        badge, col, title, sub = stages[i]
        stage_badge(ax, x, Y_ROW1 + BADGE_OFFSET, badge, col)
        draw_block(ax, x, Y_ROW1, BW, BH, title, sub,
                   facecolor=col+'14', edgecolor=col, lw=3.0)

    # ── Row 2 blocks ──────────────────────────────────────────────────────────
    for i, x in enumerate(xs2):
        badge, col, title, sub = stages[4 + i]
        stage_badge(ax, x, Y_ROW2 + BADGE_OFFSET, badge, col)
        draw_block(ax, x, Y_ROW2, BW, BH, title, sub,
                   facecolor=col+'14', edgecolor=col, lw=3.0)

    # ── Row 1 horizontal arrows ───────────────────────────────────────────────
    h_labels_r1 = [
        'deformation maps\n+ masks',
        'warped masks\n+ images',
        'denoised volume',
    ]
    for i in range(N1 - 1):
        col    = stages[i][1]
        draw_arrow(ax, xs1[i] + BW/2 + PAD, Y_ROW1, xs1[i+1] - BW/2 - PAD, Y_ROW1,
                   color=col, lw=3.5, label=h_labels_r1[i])

    # ── Re-routed Inter-row Connector ─────────────────────────────────────────
    elbow_col = stages[3][1]
    
    x0 = xs1[3] + BW/2 + PAD
    y0 = Y_ROW1
    x1 = xs1[3] + BW/2 + 0.9
    y_mid = (Y_ROW1 + Y_ROW2) / 2
    x2 = xs2[0] - BW/2 - 0.9
    y3 = Y_ROW2
    x3 = xs2[0] - BW/2 - PAD

    # 5-Segment Connector Logic
    ax.plot([x0, x1], [y0, y0], color=elbow_col, lw=3.5, ls='--', solid_capstyle='butt', zorder=2)
    ax.plot([x1, x1], [y0, y_mid], color=elbow_col, lw=3.5, ls='--', solid_capstyle='butt', zorder=2)
    ax.plot([x1, x2], [y_mid, y_mid], color=elbow_col, lw=3.5, ls='--', solid_capstyle='butt', zorder=2)
    ax.plot([x2, x2], [y_mid, y3], color=elbow_col, lw=3.5, ls='--', solid_capstyle='butt', zorder=2)
    ax.plot([x2, x3], [y3, y3], color=elbow_col, lw=3.5, ls='--', solid_capstyle='butt', zorder=2)

    ax.annotate('', xy=(x3 + 0.01, y3), xytext=(x3, y3),
                arrowprops=dict(arrowstyle='->', color=elbow_col, lw=3.5, mutation_scale=28), zorder=3)

    # Repositioned label squarely on the horizontal midline
    ax.text((x1 + x2)/2, y_mid, 'per-slice\ncell measures',
            ha='center', va='center', fontsize=18,
            color=elbow_col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.92), zorder=5)

    # ── Row 2 horizontal arrows ───────────────────────────────────────────────
    h_labels_r2 = [
        'identity map\n+ catalogue',
        'typed records\n+ 3D catalog',
    ]
    for i in range(N2 - 1):
        col = stages[4 + i][1]
        draw_arrow(ax, xs2[i] + BW/2 + PAD, Y_ROW2, xs2[i+1] - BW/2 - PAD, Y_ROW2,
                   color=col, lw=3.5, label=h_labels_r2[i])

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc=c+'22', ec=c, label=lbl)
        for c, lbl in zip(COLORS, [
            'Registration', 'Segmentation', 'Artefact denoising',
            'Marker quantification', '3D reconstruction',
            'Cell-type assignment', 'Spatial comparison',
        ])
    ]
    ax.legend(handles=handles,
              loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=20, frameon=False,
              handlelength=2.5, handleheight=1.5, columnspacing=1.8)

    fig.savefig(out_path, bbox_inches='tight', facecolor=C_BG, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='fig_pipeline_two_row_v4.png')
    args = parser.parse_args()
    build(args.out)