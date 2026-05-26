"""
aggregate_tme.py
================
Aggregate per-core TME comparison outputs produced by compare_2d_3d_tme.py
across a predefined set of TMA cores and generate multi-core summary figures.

Cores analysed
--------------
    Core_01, Core_02, Core_03, Core_04, Core_05, Core_07, Core_08, Core_09,
    Core_10, Core_11, Core_12, Core_13, Core_14, Core_15, Core_18, Core_24,
    Core_26

Input CSVs expected per core (under TME_Analysis/<CORE>/)
----------------------------------------------------------
    cell_density_2d.csv        — mean_density_per_mm2 per cell type
    cell_density_3d.csv        — density_per_mm3 per cell type
    nn_distances_2d.csv        — mean NN distances per type-pair, 2D
    nn_distances_3d.csv        — mean NN distances per type-pair, 3D
    entropy_summary.csv        — per-cell-type mean H ± std + KS/MWU tests
    summary_comparison.csv     — nn_dist_2d/3d/delta per type-pair

Outputs (under TME_Analysis/Aggregate/)
----------------------------------------
    aggregate_density.csv
    aggregate_nn_distances.csv
    aggregate_entropy.csv
    aggregate_summary.csv
    figures/
        fig_1_density_2d_vs_3d.png        — paired strip plot, 2D vs 3D density
        fig_2_nn_delta_heatmap.png         — heatmap of 3D-2D NN delta per type-pair
        fig_3_entropy_delta_boxplot.png    — boxplot of delta mean H per cell type
        fig_4_entropy_significance.png     — fraction of cores with KS p < 0.05

Usage
-----
    python aggregate_tme.py
    python aggregate_tme.py --radius_um 50
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Aggregate multi-core TME spatial comparison outputs.'
)
parser.add_argument('--radius_um', type=float, default=50.0,
                    help='Neighbourhood radius used in compare_2d_3d_tme.py (default: 50).')
args = parser.parse_args()
RADIUS_UM = args.radius_um

# ─────────────────────────────────────────────────────────────────────────────
# CORE LIST
# ─────────────────────────────────────────────────────────────────────────────
CORE_IDS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 24, 26]
CORE_NAMES = [f'Core_{str(c).zfill(2)}' for c in CORE_IDS]

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
TME_DIR = os.path.join(config.DATASPACE, 'TME_Analysis')
OUT_DIR = os.path.join(TME_DIR, 'Aggregate')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
BG = '#F7F9FA'
_ALL_TYPES_SORTED = ['Tumour', 'Macrophage', 'T_cell', 'Endothelial',
                     'Neural', 'Ambiguous', 'Unknown']
_PALETTE = {
    'Tumour':      '#C62828',
    'Macrophage':  '#EF6C00',
    'T_cell':      '#1565C0',
    'Endothelial': '#6A1B9A',
    'Neural':      '#00695C',
    'Ambiguous':   '#78909C',
    'Unknown':     '#BDBDBD',
}

def _type_color(ct):
    return _PALETTE.get(ct, '#888888')

def _finish_ax(ax):
    ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND CONCATENATE PER-CORE CSVs
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(core_name, filename):
    path = os.path.join(TME_DIR, core_name, filename)
    if not os.path.exists(path):
        logger.warning(f'  Missing: {path}')
        return None
    df = pd.read_csv(path)
    df.insert(0, 'core', core_name)
    return df


logger.info('Loading per-core CSVs ...')

density_2d_all   = []
density_3d_all   = []
nn_2d_all        = []
nn_3d_all        = []
entropy_all      = []
summary_all      = []

for core in CORE_NAMES:
    d2  = load_csv(core, 'cell_density_2d.csv')
    d3  = load_csv(core, 'cell_density_3d.csv')
    n2  = load_csv(core, 'nn_distances_2d.csv')
    n3  = load_csv(core, 'nn_distances_3d.csv')
    ent = load_csv(core, 'entropy_summary.csv')
    sm  = load_csv(core, 'summary_comparison.csv')

    if d2  is not None: density_2d_all.append(d2)
    if d3  is not None: density_3d_all.append(d3)
    if n2  is not None: nn_2d_all.append(n2)
    if n3  is not None: nn_3d_all.append(n3)
    if ent is not None: entropy_all.append(ent)
    if sm  is not None: summary_all.append(sm)

    logger.info(f'  {core}: loaded {sum(x is not None for x in [d2,d3,n2,n3,ent,sm])}/6 files')

df_density_2d = pd.concat(density_2d_all, ignore_index=True)
df_density_3d = pd.concat(density_3d_all, ignore_index=True)
df_nn_2d      = pd.concat(nn_2d_all,      ignore_index=True)
df_nn_3d      = pd.concat(nn_3d_all,      ignore_index=True)
df_entropy    = pd.concat(entropy_all,    ignore_index=True)
df_summary    = pd.concat(summary_all,    ignore_index=True)

# Save aggregate CSVs
df_density_2d.to_csv(os.path.join(OUT_DIR, 'aggregate_density_2d.csv'), index=False)
df_density_3d.to_csv(os.path.join(OUT_DIR, 'aggregate_density_3d.csv'), index=False)
df_nn_2d.to_csv(     os.path.join(OUT_DIR, 'aggregate_nn_2d.csv'),      index=False)
df_nn_3d.to_csv(     os.path.join(OUT_DIR, 'aggregate_nn_3d.csv'),      index=False)
df_entropy.to_csv(   os.path.join(OUT_DIR, 'aggregate_entropy.csv'),    index=False)
df_summary.to_csv(   os.path.join(OUT_DIR, 'aggregate_summary.csv'),    index=False)
logger.info('Aggregate CSVs saved.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Cell-type fraction: 2D vs 3D scatter plot
#
# For each cell type, one panel shows the 2D fraction (x) vs 3D fraction (y)
# across cores. Each point is one core. Points on the diagonal indicate that
# 2D and 3D composition agree; deviation indicates systematic over- or
# under-estimation of that cell type in 2D.
# Fractions are computed as proportion of total cells per core per modality,
# using only the biologically meaningful types (excluding Ambiguous/Unknown).
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating Figure 1: cell-type fraction scatter 2D vs 3D ...')

BIOLOGICAL_TYPES = ['Tumour', 'Macrophage', 'T_cell', 'Endothelial', 'Neural']

# Compute total cells per core per modality from density × area/volume proxy.
# Since we only have density (not raw counts), use density directly as a
# proportional measure and compute fractions across biological types only.
def compute_fractions(df_density, density_col, types):
    sub = df_density[df_density['cell_type'].isin(types)][
        ['core', 'cell_type', density_col]].copy()
    totals = sub.groupby('core')[density_col].sum().rename('total')
    sub    = sub.merge(totals, on='core')
    sub['fraction'] = sub[density_col] / sub['total']
    return sub[['core', 'cell_type', 'fraction']]

frac_2d = compute_fractions(df_density_2d, 'mean_density_per_mm2', BIOLOGICAL_TYPES)
frac_3d = compute_fractions(df_density_3d, 'density_per_mm3',      BIOLOGICAL_TYPES)

types_frac = [t for t in BIOLOGICAL_TYPES
              if t in frac_2d['cell_type'].values
              and t in frac_3d['cell_type'].values]

n_types = len(types_frac)
ncols   = 3
nrows   = int(np.ceil(n_types / ncols))

fig1, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows),
                           facecolor=BG)
axes = axes.flatten()

for ax in axes:
    _finish_ax(ax)

from scipy import stats as scipy_stats

for idx, ct in enumerate(types_frac):
    ax  = axes[idx]
    col = _type_color(ct)

    f2 = frac_2d[frac_2d['cell_type'] == ct][['core', 'fraction']].rename(
             columns={'fraction': 'f2d'})
    f3 = frac_3d[frac_3d['cell_type'] == ct][['core', 'fraction']].rename(
             columns={'fraction': 'f3d'})
    merged = f2.merge(f3, on='core', how='inner')

    ax.scatter(merged['f2d'], merged['f3d'],
               color=col, s=55, alpha=0.85, zorder=3,
               edgecolors='white', lw=0.5)

    # Diagonal (perfect agreement)
    lim_max = max(merged['f2d'].max(), merged['f3d'].max()) * 1.12
    ax.plot([0, lim_max], [0, lim_max],
            color='#aaa', lw=1.2, ls='--', zorder=1)

    # Pearson r annotation
    if len(merged) >= 3:
        r, p = scipy_stats.pearsonr(merged['f2d'], merged['f3d'])
        pstr = f'p={p:.2f}' if p >= 0.01 else 'p<0.01'
        ax.text(0.97, 0.05, f'r={r:.2f}  {pstr}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, color='#333')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('2D fraction', fontsize=9)
    ax.set_ylabel('3D fraction', fontsize=9)
    ax.set_title(ct, fontsize=11, fontweight='bold', color=col)

# Hide unused panels
for ax in axes[n_types:]:
    ax.set_visible(False)

fig1.suptitle(f'Cell-type composition: 2D vs 3D across {len(CORE_NAMES)} cores\n'
              f'Each point = one core   Dashed line = perfect agreement',
              fontsize=13, fontweight='bold', color='#1A1A2E', y=1.01)
fig1.tight_layout()
path1 = os.path.join(FIG_DIR, 'fig_1_celltype_fraction_2d_vs_3d.png')
fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig1)
logger.info(f'  Figure 1 saved: {path1}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — NN distance delta heatmap (3D - 2D), median across cores
#
# Rows = source type, columns = target type.
# Colour = median delta NN distance (µm) across cores.
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating Figure 2: NN distance delta heatmap ...')

nn_merged = (df_nn_2d[['core', 'src_type', 'tgt_type', 'mean_dist_um']]
             .rename(columns={'mean_dist_um': 'dist_2d'})
             .merge(
                 df_nn_3d[['core', 'src_type', 'tgt_type', 'mean_dist_um']]
                 .rename(columns={'mean_dist_um': 'dist_3d'}),
                 on=['core', 'src_type', 'tgt_type'], how='inner'))
nn_merged['delta'] = nn_merged['dist_3d'] - nn_merged['dist_2d']

nn_pivot = (nn_merged
            .groupby(['src_type', 'tgt_type'])['delta']
            .median()
            .unstack(fill_value=np.nan))

# Reorder rows/cols by canonical type order
row_order = [t for t in _ALL_TYPES_SORTED if t in nn_pivot.index]
col_order = [t for t in _ALL_TYPES_SORTED if t in nn_pivot.columns]
nn_pivot  = nn_pivot.loc[row_order, col_order]

fig2, ax2 = plt.subplots(figsize=(8, 6.5), facecolor=BG)
_finish_ax(ax2)

vmax = np.nanpercentile(np.abs(nn_pivot.values), 95)
im   = ax2.imshow(nn_pivot.values, cmap='RdBu_r', aspect='auto',
                  vmin=-vmax, vmax=vmax)

ax2.set_xticks(range(len(col_order)))
ax2.set_yticks(range(len(row_order)))
ax2.set_xticklabels(col_order, rotation=35, ha='right', fontsize=10)
ax2.set_yticklabels(row_order, fontsize=10)
ax2.set_xlabel('Target cell type', fontsize=11)
ax2.set_ylabel('Source cell type', fontsize=11)
ax2.set_title(f'Median NN distance delta (3D minus 2D, µm)\nacross {len(CORE_NAMES)} cores',
              fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

# Annotate cells
for i in range(len(row_order)):
    for j in range(len(col_order)):
        val = nn_pivot.values[i, j]
        if np.isfinite(val):
            ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=7.5, color='white' if abs(val) > vmax * 0.6 else '#333')

cbar = fig2.colorbar(im, ax=ax2, fraction=0.035, pad=0.03)
cbar.set_label('Δ NN distance (µm)', fontsize=10)

fig2.tight_layout()
path2 = os.path.join(FIG_DIR, 'fig_2_nn_delta_heatmap.png')
fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
logger.info(f'  Figure 2 saved: {path2}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Shannon entropy delta (H_3D - H_2D) boxplot per cell type
#
# One box per cell type, distribution across cores.
# Overlaid strip of individual core values.
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating Figure 3: entropy delta boxplot ...')

types_ent = [t for t in _ALL_TYPES_SORTED if t in df_entropy['cell_type'].values]

fig3, ax3 = plt.subplots(figsize=(9, 5.5), facecolor=BG)
_finish_ax(ax3)

x_positions = np.arange(len(types_ent))

bp_data = []
for ct in types_ent:
    vals = df_entropy[df_entropy['cell_type'] == ct]['delta_mean_H'].dropna().values
    bp_data.append(vals)

bp = ax3.boxplot(bp_data, positions=x_positions, widths=0.45,
                 patch_artist=True, notch=False,
                 medianprops=dict(color='white', lw=2),
                 whiskerprops=dict(color='#888'),
                 capprops=dict(color='#888'),
                 flierprops=dict(marker='o', markersize=3,
                                 markerfacecolor='#aaa', markeredgecolor='none'))

for patch, ct in zip(bp['boxes'], types_ent):
    patch.set_facecolor(_type_color(ct))
    patch.set_alpha(0.75)
    patch.set_edgecolor('none')

# Strip of individual core points
rng = np.random.default_rng(42)
for xi, (ct, vals) in enumerate(zip(types_ent, bp_data)):
    jitter = rng.uniform(-0.14, 0.14, len(vals))
    ax3.scatter(xi + jitter, vals, color=_type_color(ct),
                s=28, alpha=0.85, zorder=4, edgecolors='white', lw=0.4)

ax3.axhline(0, color='#555', lw=1.2, ls='--', zorder=1)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(types_ent, rotation=28, ha='right', fontsize=10)
ax3.set_ylabel('Δ Mean Shannon entropy H (nats)\n(3D minus 2D)', fontsize=11)
ax3.set_title(f'Neighbourhood entropy shift: 3D vs 2D\n'
              f'Distribution across {len(CORE_NAMES)} cores   '
              f'Radius = {int(RADIUS_UM)} µm',
              fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

fig3.tight_layout()
path3 = os.path.join(FIG_DIR, 'fig_3_entropy_delta_boxplot.png')
fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
logger.info(f'  Figure 3 saved: {path3}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Fraction of cores with significant entropy shift (KS p < 0.05)
#
# Horizontal bar per cell type, bar length = fraction of cores significant.
# Annotated with n significant / n total cores.
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating Figure 4: entropy significance summary ...')

sig_rows = []
for ct in types_ent:
    sub      = df_entropy[df_entropy['cell_type'] == ct]
    n_total  = len(sub)
    n_sig    = (sub['ks_p'] < 0.05).sum()
    sig_rows.append({'cell_type': ct,
                     'n_sig':     n_sig,
                     'n_total':   n_total,
                     'frac_sig':  n_sig / n_total if n_total > 0 else 0})

df_sig = pd.DataFrame(sig_rows)

fig4, ax4 = plt.subplots(figsize=(8, 5), facecolor=BG)
_finish_ax(ax4)

y_pos = np.arange(len(df_sig))
for k, row in df_sig.iterrows():
    ax4.barh(k, row['frac_sig'], color=_type_color(row['cell_type']),
             alpha=0.80, edgecolor='none', height=0.55)
    ax4.text(row['frac_sig'] + 0.01, k,
             f"{int(row['n_sig'])}/{int(row['n_total'])}",
             va='center', fontsize=9, color='#333')

ax4.axvline(0.5, color='#888', lw=1, ls='--')
ax4.set_xlim(0, 1.15)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(df_sig['cell_type'], fontsize=10)
ax4.set_xlabel('Fraction of cores with KS p < 0.05\n(2D vs 3D entropy distribution)',
               fontsize=11)
ax4.set_title(f'Significance of 2D vs 3D entropy shift per cell type\n'
              f'across {len(CORE_NAMES)} cores',
              fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)
ax4.text(0.5, -0.7, '— 50 % threshold', fontsize=8.5,
         color='#888', ha='center', style='italic')

fig4.tight_layout()
path4 = os.path.join(FIG_DIR, 'fig_4_entropy_significance.png')
fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig4)
logger.info(f'  Figure 4 saved: {path4}')


# ─────────────────────────────────────────────────────────────────────────────
# FINAL LOG
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'DONE — {len(CORE_NAMES)} cores aggregated')
logger.info(f'  Output directory : {OUT_DIR}')
logger.info('  CSVs:')
logger.info('    aggregate_density_2d/3d.csv')
logger.info('    aggregate_nn_2d/3d.csv')
logger.info('    aggregate_entropy.csv')
logger.info('    aggregate_summary.csv')
logger.info('  Figures:')
logger.info('    fig_1_celltype_fraction_2d_vs_3d.png')
logger.info('    fig_2_nn_delta_heatmap.png')
logger.info('    fig_3_entropy_delta_boxplot.png')
logger.info('    fig_4_entropy_significance.png')
logger.info('=' * 60)