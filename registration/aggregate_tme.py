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
        fig_2_nn_distances.png      — per-source-type 2D vs 3D scatter across cores
        fig_3_entropy_delta_boxplot.png    — boxplot of delta mean H per cell type

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
# TME_DIR = os.path.join(config.DATASPACE, 'TME_Analysis')
TME_DIR = os.path.join(config.DATASPACE, 'TME_Analysis_Bspline')
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
# AMBIGUOUS CELL EXCLUSION SUMMARY
#
# Quantifies how many Ambiguous cells are present per core (and as a % of all
# cells) in both 2D and 3D density CSVs.  These cells are excluded from the
# downstream figures (Figure 1 uses BIOLOGICAL_TYPES only; Figures 2-4 receive
# pre-filtered CSVs from compare_2d_3d_tme.py where Ambiguous never enters the
# valid_types lists).  This block makes that exclusion explicit and auditable.
#
# Outputs
# -------
#   ambiguous_exclusion_summary.csv   — per-core counts and percentages (2D & 3D)
#   ambiguous_exclusion_summary.txt   — human-readable report with per-core table
#                                       and cross-core averages
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info('AMBIGUOUS EXCLUSION SUMMARY')

ambig_rows = []
for core in CORE_NAMES:
    # ── 2D ───────────────────────────────────────────────────────────────────
    sub2 = df_density_2d[df_density_2d['core'] == core]
    total2  = sub2['mean_density_per_mm2'].sum()
    ambig2  = sub2.loc[sub2['cell_type'] == 'Ambiguous',
                       'mean_density_per_mm2'].sum()
    n_ambig2 = sub2.loc[sub2['cell_type'] == 'Ambiguous',
                        'n_cells'].sum() if 'n_cells' in sub2.columns else np.nan
    pct2    = 100.0 * ambig2 / total2 if total2 > 0 else np.nan

    # ── 3D ───────────────────────────────────────────────────────────────────
    sub3 = df_density_3d[df_density_3d['core'] == core]
    total3  = sub3['density_per_mm3'].sum()
    ambig3  = sub3.loc[sub3['cell_type'] == 'Ambiguous',
                       'density_per_mm3'].sum()
    n_ambig3 = sub3.loc[sub3['cell_type'] == 'Ambiguous',
                        'n_cells'].sum() if 'n_cells' in sub3.columns else np.nan
    pct3    = 100.0 * ambig3 / total3 if total3 > 0 else np.nan

    ambig_rows.append({
        'core':                    core,
        # 2D
        'n_ambiguous_2d':          int(n_ambig2) if not np.isnan(n_ambig2) else np.nan,
        'density_ambiguous_2d':    round(ambig2,  4),
        'density_total_2d':        round(total2,  4),
        'pct_ambiguous_2d':        round(pct2,    2) if not np.isnan(pct2)  else np.nan,
        # 3D
        'n_ambiguous_3d':          int(n_ambig3) if not np.isnan(n_ambig3) else np.nan,
        'density_ambiguous_3d':    round(ambig3,  4),
        'density_total_3d':        round(total3,  4),
        'pct_ambiguous_3d':        round(pct3,    2) if not np.isnan(pct3)  else np.nan,
    })

df_ambig = pd.DataFrame(ambig_rows)

# ── Cross-core averages ───────────────────────────────────────────────────────
mean_pct2  = df_ambig['pct_ambiguous_2d'].mean()
mean_pct3  = df_ambig['pct_ambiguous_3d'].mean()
mean_n2    = df_ambig['n_ambiguous_2d'].mean()
mean_n3    = df_ambig['n_ambiguous_3d'].mean()
total_n2   = df_ambig['n_ambiguous_2d'].sum()
total_n3   = df_ambig['n_ambiguous_3d'].sum()

# ── Save CSV ──────────────────────────────────────────────────────────────────
ambig_csv = os.path.join(OUT_DIR, 'ambiguous_exclusion_summary.csv')
df_ambig.to_csv(ambig_csv, index=False)
logger.info(f'  Saved: {ambig_csv}')

# ── Save human-readable text report ──────────────────────────────────────────
ambig_txt = os.path.join(OUT_DIR, 'ambiguous_exclusion_summary.txt')
col_w = 12
with open(ambig_txt, 'w') as f:
    f.write('AMBIGUOUS CELL EXCLUSION REPORT\n')
    f.write(f'Cores analysed : {len(CORE_NAMES)}\n')
    f.write('=' * 72 + '\n\n')

    # Header
    f.write(f"{'Core':<12} {'N_ambig_2D':>12} {'%_ambig_2D':>12} "
            f"{'N_ambig_3D':>12} {'%_ambig_3D':>12}\n")
    f.write('-' * 62 + '\n')

    for _, row in df_ambig.iterrows():
        n2_str  = f"{int(row['n_ambiguous_2d'])}"  if not pd.isna(row['n_ambiguous_2d'])  else 'N/A'
        n3_str  = f"{int(row['n_ambiguous_3d'])}"  if not pd.isna(row['n_ambiguous_3d'])  else 'N/A'
        pct2_str = f"{row['pct_ambiguous_2d']:.2f}%" if not pd.isna(row['pct_ambiguous_2d']) else 'N/A'
        pct3_str = f"{row['pct_ambiguous_3d']:.2f}%" if not pd.isna(row['pct_ambiguous_3d']) else 'N/A'
        f.write(f"{row['core']:<12} {n2_str:>12} {pct2_str:>12} "
                f"{n3_str:>12} {pct3_str:>12}\n")

    f.write('-' * 62 + '\n')
    f.write(f"{'MEAN':<12} {mean_n2:>11.1f} {mean_pct2:>11.2f}% "
            f"{mean_n3:>11.1f} {mean_pct3:>11.2f}%\n")
    f.write(f"{'TOTAL':<12} {total_n2:>11.0f} {'':>12} "
            f"{total_n3:>11.0f}\n\n")

    f.write('Notes\n')
    f.write('-----\n')
    f.write('N_ambig     : raw cell count from density CSV (n_cells column).\n')
    f.write('              Shown as N/A if n_cells not present in CSV.\n')
    f.write('%_ambig     : Ambiguous density / total density × 100.\n')
    f.write('              Density proxy used because raw counts may vary\n')
    f.write('              across slices / volume estimates.\n')
    f.write('Exclusion   : Ambiguous cells are excluded from Figure 1\n')
    f.write('              (BIOLOGICAL_TYPES filter) and from Figures 2-4\n')
    f.write('              (valid_types filter in compare_2d_3d_tme.py).\n')

logger.info(f'  Saved: {ambig_txt}')

# ── Log summary ───────────────────────────────────────────────────────────────
logger.info(f'  {"Core":<12} {"N_ambig_2D":>12} {"%_ambig_2D":>12} '
            f'{"N_ambig_3D":>12} {"%_ambig_3D":>12}')
for _, row in df_ambig.iterrows():
    n2_s  = f"{int(row['n_ambiguous_2d'])}"   if not pd.isna(row['n_ambiguous_2d'])  else 'N/A'
    n3_s  = f"{int(row['n_ambiguous_3d'])}"   if not pd.isna(row['n_ambiguous_3d'])  else 'N/A'
    p2_s  = f"{row['pct_ambiguous_2d']:.2f}%" if not pd.isna(row['pct_ambiguous_2d']) else 'N/A'
    p3_s  = f"{row['pct_ambiguous_3d']:.2f}%" if not pd.isna(row['pct_ambiguous_3d']) else 'N/A'
    logger.info(f'  {row["core"]:<12} {n2_s:>12} {p2_s:>12} {n3_s:>12} {p3_s:>12}')
logger.info(f'  {"MEAN":<12} {mean_n2:>11.1f} {mean_pct2:>11.2f}% '
            f'{mean_n3:>11.1f} {mean_pct3:>11.2f}%')
logger.info(f'  {"TOTAL":<12} {total_n2:>11.0f} {"":>12} {total_n3:>11.0f}')
logger.info('=' * 60)


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

fig1, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.5 * nrows),
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
               color=col, s=120, alpha=0.85, zorder=3,
               edgecolors='white', lw=0.8)

    # Diagonal (perfect agreement)
    lim_max = max(merged['f2d'].max(), merged['f3d'].max()) * 1.12
    ax.plot([0, lim_max], [0, lim_max],
            color='#aaa', lw=2.0, ls='--', zorder=1)

    # Pearson r annotation
    if len(merged) >= 3:
        r, p = scipy_stats.pearsonr(merged['f2d'], merged['f3d'])
        pstr = f'p={p:.2f}' if p >= 0.01 else 'p<0.01'
        ax.text(0.97, 0.05, f'r={r:.2f}  {pstr}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=22, color='#333')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('2D fraction', fontsize=24)
    ax.set_ylabel('3D fraction', fontsize=24)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_title(ct, fontsize=26, fontweight='bold', color=col)

# Hide unused panels
for ax in axes[n_types:]:
    ax.set_visible(False)

fig1.suptitle(f'Cell-type composition: 2D vs 3D across {len(CORE_NAMES)} cores\n'
              f'Each point = one core   Dashed line = perfect agreement',
              fontsize=28, fontweight='bold', color='#1A1A2E', y=1.01)
fig1.tight_layout()
path1 = os.path.join(FIG_DIR, 'fig_1_celltype_fraction_2d_vs_3d.png')
fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig1)
logger.info(f'  Figure 1 saved: {path1}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — NN distances: multi-panel paired bar + core dot strip
#
# One subplot per source cell type (grid layout, 3 columns).
# Within each panel, one row per target type.
# Pale bar  = median 2D distance across cores.
# Solid bar = median 3D distance across cores.
# Individual core values overlaid as dots (2D = open, 3D = filled).
# Coloured by source type throughout.
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating Figure 2: NN distance multi-panel bar + strip plot ...')

nn_merged = (df_nn_2d[['core', 'src_type', 'tgt_type', 'mean_dist_um']]
             .rename(columns={'mean_dist_um': 'dist_2d'})
             .merge(
                 df_nn_3d[['core', 'src_type', 'tgt_type', 'mean_dist_um']]
                 .rename(columns={'mean_dist_um': 'dist_3d'}),
                 on=['core', 'src_type', 'tgt_type'], how='inner'))

src_types2 = [t for t in _ALL_TYPES_SORTED if t in nn_merged['src_type'].values]
n_src      = len(src_types2)
ncols2     = 3
nrows2     = 2   # fixed 3×2 grid

# Panel height scales with the number of target types in the busiest source
max_tgts   = max(
    nn_merged[nn_merged['src_type'] == s]['tgt_type'].nunique()
    for s in src_types2
)
panel_h    = max(6.0, max_tgts * 1.1 + 2.0)

fig2, axes2 = plt.subplots(nrows2, ncols2,
                            figsize=(8.5 * ncols2, panel_h * nrows2),
                            facecolor=BG)
axes2 = axes2.flatten()
for ax in axes2:
    _finish_ax(ax)

bar_h = 0.32
gap   = 0.08
row_h = bar_h * 2 + gap + 0.18
rng2  = np.random.default_rng(0)

for idx, src in enumerate(src_types2):
    ax  = axes2[idx]
    col = _type_color(src)

    tgts = [t for t in _ALL_TYPES_SORTED
            if ((nn_merged['src_type'] == src) &
                (nn_merged['tgt_type'] == t)).any()]
    y_centers2 = np.arange(len(tgts)) * row_h

    for k, tgt in enumerate(tgts):
        sub = nn_merged[(nn_merged['src_type'] == src) &
                        (nn_merged['tgt_type'] == tgt)]
        yc   = y_centers2[k]
        med2 = sub['dist_2d'].median()
        med3 = sub['dist_3d'].median()

        # Pale bar = 2D median
        ax.barh(yc + bar_h / 2 + gap / 2, med2,
                height=bar_h, color=col, alpha=0.40, edgecolor='none', zorder=2)
        # Solid bar = 3D median
        ax.barh(yc - bar_h / 2 - gap / 2, med3,
                height=bar_h, color=col, alpha=1.00, edgecolor='none', zorder=2)

        # Core dots — 2D open, 3D filled
        jit2 = rng2.uniform(-bar_h * 0.35, bar_h * 0.35, len(sub))
        jit3 = rng2.uniform(-bar_h * 0.35, bar_h * 0.35, len(sub))
        ax.scatter(sub['dist_2d'], yc + bar_h / 2 + gap / 2 + jit2,
                   color=col, s=60, alpha=0.70, zorder=3,
                   edgecolors=col, lw=1.5, facecolors='white')
        ax.scatter(sub['dist_3d'], yc - bar_h / 2 - gap / 2 + jit3,
                   color=col, s=60, alpha=0.70, zorder=3,
                   edgecolors='none')

    ax.set_yticks(y_centers2)
    ax.set_yticklabels(tgts, fontsize=28)
    ax.tick_params(axis='x', labelsize=26)
    ax.invert_yaxis()
    ax.set_xlabel('Mean NN distance (µm)', fontsize=28)
    ax.set_title(f'Source: {src}', fontsize=32, fontweight='bold', color=col)

# Shared legend on the first panel
axes2[0].legend(handles=[
    mpatches.Patch(facecolor='#888', alpha=0.40, label='2D  (pale bar + open dots)'),
    mpatches.Patch(facecolor='#888', alpha=1.00, label='3D  (solid bar + filled dots)'),
], fontsize=20, frameon=False, loc='lower right')

for ax in axes2[n_src:]:
    ax.set_visible(False)

fig2.suptitle(
    f'Nearest-neighbour distances: 2D vs 3D across {len(CORE_NAMES)} cores\n'
    'Bars = median   Dots = individual cores   Rows = target type',
    fontsize=34, fontweight='bold', color='#1A1A2E', y=1.01)
fig2.tight_layout()
path2 = os.path.join(FIG_DIR, 'fig_2_nn_distances.png')
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

fig3, ax3 = plt.subplots(figsize=(13, 8), facecolor=BG)
_finish_ax(ax3)

x_positions = np.arange(len(types_ent))

bp_data = []
for ct in types_ent:
    sub = df_entropy[df_entropy['cell_type'] == ct][
        ['delta_mean_H', 'mean_H_2d']].dropna()
    # relative % delta: (H_3D - H_2D) / H_2D * 100
    rel = np.where(sub['mean_H_2d'] > 0,
                   sub['delta_mean_H'] / sub['mean_H_2d'] * 100,
                   np.nan)
    bp_data.append(rel[np.isfinite(rel)])

bp = ax3.boxplot(bp_data, positions=x_positions, widths=0.50,
                 patch_artist=True, notch=False,
                 medianprops=dict(color='white', lw=3.0),
                 whiskerprops=dict(color='#888', lw=2.0),
                 capprops=dict(color='#888', lw=2.0),
                 flierprops=dict(marker='o', markersize=7,
                                 markerfacecolor='#aaa', markeredgecolor='none'))

for patch, ct in zip(bp['boxes'], types_ent):
    patch.set_facecolor(_type_color(ct))
    patch.set_alpha(0.75)
    patch.set_edgecolor('none')

# Strip of individual core points
rng = np.random.default_rng(42)
for xi, (ct, vals) in enumerate(zip(types_ent, bp_data)):
    jitter = rng.uniform(-0.16, 0.16, len(vals))
    ax3.scatter(xi + jitter, vals, color=_type_color(ct),
                s=80, alpha=0.85, zorder=4, edgecolors='white', lw=0.6)

ax3.axhline(0, color='#555', lw=2.0, ls='--', zorder=1)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(types_ent, rotation=28, ha='right', fontsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax3.set_ylabel('Relative Δ Shannon entropy H (%)\n(3D − 2D) / 2D × 100', fontsize=21)
ax3.set_title(f'Neighbourhood entropy shift: 3D vs 2D\n'
              f'Distribution across {len(CORE_NAMES)} cores   '
              f'Radius = {int(RADIUS_UM)} µm',
              fontsize=23, fontweight='bold', color='#1A1A2E', pad=14)

fig3.tight_layout()
path3 = os.path.join(FIG_DIR, 'fig_3_entropy_delta_boxplot.png')
fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
logger.info(f'  Figure 3 saved: {path3}')


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
logger.info('    fig_2_nn_distances.png')
logger.info('    fig_3_entropy_delta_boxplot.png')
logger.info('=' * 60)