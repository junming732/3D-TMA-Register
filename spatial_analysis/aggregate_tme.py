"""
aggregate_tme.py
================
Aggregate per-core TME comparison outputs produced by compare_2d_3d_tme.py
across a predefined set of TMA cores and generate multi-core summary figures.

Cores analysed
--------------
    Controlled via CORE_START / CORE_END / EXCLUDED_CORES below (see "CORE
    LIST" section) — defaults to all cores in [CORE_START, CORE_END] except
    those listed in EXCLUDED_CORES.

Input CSVs expected per core (under <tme_dir_name>/<CORE>/)
----------------------------------------------------------
    cell_density_2d.csv        — mean_density_per_mm2 per cell type
    cell_density_3d.csv        — density_per_mm3 per cell type
    nn_distances_2d.csv        — mean NN distances per type-pair, 2D
    nn_distances_3d.csv        — mean NN distances per type-pair, 3D
    entropy_summary.csv        — per-cell-type mean H ± std + KS/MWU tests
    summary_comparison.csv     — nn_dist_2d/3d/delta per type-pair
    contact_summary_3d.csv     — real 3D nucleus-adjacency per type-pair
                                  (contact_graph_3d.py; optional — missing files
                                  are skipped with a warning, not fatal)
    contact_summary_2d.csv     — same metric, per-slice (2D), from the same
                                  contact_graph_3d.py run (unless --skip_2d was
                                  passed) — enables the 2D vs 3D comparison

Outputs (under <tme_dir_name>/Aggregate/)
----------------------------------------
    aggregate_density_2d.csv
    aggregate_density_3d.csv
    aggregate_nn_2d.csv
    aggregate_nn_3d.csv
    aggregate_entropy.csv
    aggregate_summary.csv
    aggregate_contact_3d.csv            — mean ± std contact metrics per type-pair
                                           across cores (from contact_summary_3d.csv)
    aggregate_contact_2d.csv            — same, from contact_summary_2d.csv
    ambiguous_exclusion_summary.csv    — per-core Ambiguous cell counts/percentages
    ambiguous_exclusion_summary.txt    — human-readable version of the above
    figures/
        fig_1_celltype_fraction_2d_vs_3d.png  — per-cell-type scatter of 2D vs 3D
                                                 composition fraction across cores,
                                                 with Pearson r annotation
        fig_2_nn_distances.png                — per-source-type 2D vs 3D median NN
                                                 distance, individual cores overlaid
        fig_3_entropy_delta_boxplot.png       — boxplot of relative entropy delta
                                                 (3D − 2D) / 2D per cell type
        fig_4_contact_heatmap.png             — mean nucleus-contacts/cell, type x
                                                 type, 2D and 3D side by side (same
                                                 color scale) when both are available,
                                                 else 3D only
        fig_4b_contact_2d_vs_3d_homotypic.png — same-type contact rate, 2D vs 3D,
                                                 grouped bars (only when both present)

Usage
-----
    python aggregate_tme.py
    python aggregate_tme.py --radius_um 50
    python aggregate_tme.py --tme_dir_name TME_Analysis_Bspline
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
parser.add_argument('--tme_dir_name', type=str, default='TME_Analysis_Bspline',
                    help='Folder under DATASPACE containing per-core TME comparison '
                         'output from compare_2d_3d_tme.py (default: TME_Analysis_Bspline).')
args = parser.parse_args()
RADIUS_UM = args.radius_um

# ─────────────────────────────────────────────────────────────────────────────
# CORE LIST
# ─────────────────────────────────────────────────────────────────────────────
CORE_START = 1          # First core number to run (inclusive)
CORE_END = 30            # Last core number to run (inclusive)

# Cores to exclude entirely (e.g. known-bad cores from analyse_unsuitable_cores.py)
EXCLUDED_CORES = (17, 21, 23, 27)

CORE_IDS = [c for c in range(CORE_START, CORE_END + 1) if c not in EXCLUDED_CORES]
CORE_NAMES = [f'Core_{str(c).zfill(2)}' for c in CORE_IDS]

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
TME_DIR = os.path.join(config.DATASPACE, args.tme_dir_name)
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
contact_3d_all   = []
contact_2d_all   = []

for core in CORE_NAMES:
    d2  = load_csv(core, 'cell_density_2d.csv')
    d3  = load_csv(core, 'cell_density_3d.csv')
    n2  = load_csv(core, 'nn_distances_2d.csv')
    n3  = load_csv(core, 'nn_distances_3d.csv')
    ent = load_csv(core, 'entropy_summary.csv')
    sm  = load_csv(core, 'summary_comparison.csv')
    ct3 = load_csv(core, 'contact_summary_3d.csv')  # optional — see docstring
    ct2 = load_csv(core, 'contact_summary_2d.csv')  # optional — needs contact_graph_3d.py NOT run with --skip_2d

    if d2  is not None: density_2d_all.append(d2)
    if d3  is not None: density_3d_all.append(d3)
    if n2  is not None: nn_2d_all.append(n2)
    if n3  is not None: nn_3d_all.append(n3)
    if ent is not None: entropy_all.append(ent)
    if sm  is not None: summary_all.append(sm)
    if ct3 is not None: contact_3d_all.append(ct3)
    if ct2 is not None: contact_2d_all.append(ct2)

    logger.info(f'  {core}: loaded '
               f'{sum(x is not None for x in [d2,d3,n2,n3,ent,sm,ct3,ct2])}/8 files')

df_density_2d = pd.concat(density_2d_all, ignore_index=True)
df_density_3d = pd.concat(density_3d_all, ignore_index=True)
df_nn_2d      = pd.concat(nn_2d_all,      ignore_index=True)
df_nn_3d      = pd.concat(nn_3d_all,      ignore_index=True)
df_entropy    = pd.concat(entropy_all,    ignore_index=True)
df_summary    = pd.concat(summary_all,    ignore_index=True)
df_contact_3d = pd.concat(contact_3d_all, ignore_index=True) if contact_3d_all else None
df_contact_2d = pd.concat(contact_2d_all, ignore_index=True) if contact_2d_all else None

# Save aggregate CSVs
df_density_2d.to_csv(os.path.join(OUT_DIR, 'aggregate_density_2d.csv'), index=False)
df_density_3d.to_csv(os.path.join(OUT_DIR, 'aggregate_density_3d.csv'), index=False)
df_nn_2d.to_csv(     os.path.join(OUT_DIR, 'aggregate_nn_2d.csv'),      index=False)
df_nn_3d.to_csv(     os.path.join(OUT_DIR, 'aggregate_nn_3d.csv'),      index=False)
df_entropy.to_csv(   os.path.join(OUT_DIR, 'aggregate_entropy.csv'),    index=False)
df_summary.to_csv(   os.path.join(OUT_DIR, 'aggregate_summary.csv'),    index=False)
if df_contact_3d is not None:
    df_contact_3d.to_csv(os.path.join(OUT_DIR, 'aggregate_contact_3d.csv'), index=False)
else:
    logger.warning('  No contact_summary_3d.csv found for any core — skipping '
                   'aggregate_contact_3d.csv and Figure 4. Run contact_graph_3d.py '
                   'per core to enable this.')
if df_contact_2d is not None:
    df_contact_2d.to_csv(os.path.join(OUT_DIR, 'aggregate_contact_2d.csv'), index=False)
else:
    logger.warning('  No contact_summary_2d.csv found for any core — Figure 4 will show '
                   '3D only. Run contact_graph_3d.py without --skip_2d to enable the '
                   '2D vs 3D comparison panel.')
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
                 showfliers=False,
                 medianprops=dict(color='white', lw=3.0),
                 whiskerprops=dict(color='#888', lw=2.0),
                 capprops=dict(color='#888', lw=2.0))

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
# FIGURE 4 — Nucleus-contact heatmap, 2D vs 3D, side by side, type x type
#
# mean_contacts_per_cell averaged across cores, per (src_type, tgt_type),
# same shared color scale in both panels so they're directly comparable by
# eye. This is the Tier 1 companion to Figure 2 (nn_distances): Figure 2
# shows centroid-proximity in 2D vs 3D, this shows actual measured nucleus
# adjacency in 2D vs 3D (see contact_graph_3d.py docstring for exactly what
# "contact" means, and why the 2D pass is structurally blind to any contact
# whose partner sits mostly in a neighboring slice). If 3D-only contacts are
# common in this tissue, the 3D panel should read visibly "hotter" than the
# 2D panel, especially on the diagonal (homotypic / same-type contact).
#
# Falls back to a single 3D-only panel (previous behavior) if
# contact_summary_2d.csv wasn't produced for any core (contact_graph_3d.py
# run with --skip_2d, or an older run predating the 2D pass).
# ─────────────────────────────────────────────────────────────────────────────
def _contact_matrix(df_contact, types):
    agg = (
        df_contact.groupby(['src_type', 'tgt_type'])
        .agg(mean_contacts_per_cell=('mean_contacts_per_cell', 'mean'),
             std_contacts_per_cell=('mean_contacts_per_cell', 'std'),
             mean_pct_cells_with_contact=('pct_cells_with_contact', 'mean'),
             n_cores=('mean_contacts_per_cell', 'count'))
        .reset_index()
    )
    n_t = len(types)
    mat = np.full((n_t, n_t), np.nan)
    for i, st in enumerate(types):
        for j, tt in enumerate(types):
            row = agg[(agg['src_type'] == st) & (agg['tgt_type'] == tt)]
            if len(row):
                mat[i, j] = row['mean_contacts_per_cell'].iloc[0]
    return mat, agg


def _draw_contact_heatmap(ax, mat, types, vmax, title):
    im = ax.imshow(mat, cmap='Reds', vmin=0, vmax=vmax, aspect='equal')
    n_t = len(types)
    for i in range(n_t):
        for j in range(n_t):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = 'white' if val > vmax * 0.55 else '#333'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=15, color=txt_color, fontweight='bold')
    ax.set_xticks(range(n_t)); ax.set_yticks(range(n_t))
    ax.set_xticklabels(types, rotation=35, ha='right', fontsize=14)
    ax.set_yticklabels(types, fontsize=14)
    ax.set_xlabel('Contacting (tgt) type', fontsize=15)
    ax.set_ylabel('Source type', fontsize=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1A1A2E', pad=10)
    return im


if df_contact_3d is None:
    logger.info('Skipping Figure 4 (contact heatmap) — no 3D contact data loaded.')
else:
    logger.info('Generating Figure 4: contact heatmap ...')

    types_contact = [t for t in _ALL_TYPES_SORTED if t in df_contact_3d['src_type'].values]
    mat_3d, agg_3d = _contact_matrix(df_contact_3d, types_contact)
    agg_3d.to_csv(os.path.join(OUT_DIR, 'aggregate_contact_3d.csv'), index=False)
    n_cores_3d = int(agg_3d['n_cores'].max()) if len(agg_3d) else 0

    if df_contact_2d is not None:
        # Restrict to types present in the 3D table too, so both panels plot
        # the same axes — a type that only clears --min_cells in one
        # dimension would otherwise misalign the two heatmaps.
        types_2d_only = [t for t in _ALL_TYPES_SORTED if t in df_contact_2d['src_type'].values]
        types_contact = [t for t in types_contact if t in types_2d_only]
        mat_3d, agg_3d = _contact_matrix(df_contact_3d, types_contact)
        mat_2d, agg_2d = _contact_matrix(df_contact_2d, types_contact)
        agg_2d.to_csv(os.path.join(OUT_DIR, 'aggregate_contact_2d.csv'), index=False)
        n_cores_2d = int(agg_2d['n_cores'].max()) if len(agg_2d) else 0

        vmax = np.nanmax([np.nanmax(mat_2d) if not np.all(np.isnan(mat_2d)) else 0,
                          np.nanmax(mat_3d) if not np.all(np.isnan(mat_3d)) else 0]) or 1.0

        n_ct = len(types_contact)
        fig4, (ax4a, ax4b) = plt.subplots(
            1, 2, figsize=(3.0 + 2.6 * n_ct, 1.8 + 1.15 * n_ct), facecolor=BG)
        _draw_contact_heatmap(ax4a, mat_2d, types_contact, vmax,
                              f'2D (per-slice, {n_cores_2d} cores)\nblind to contacts above/below the plane')
        im = _draw_contact_heatmap(ax4b, mat_3d, types_contact, vmax,
                                   f'3D (reconstructed, {n_cores_3d} cores)\nreal geometric adjacency')
        fig4.suptitle('Mean nucleus-contacts per cell, type × type — 2D vs 3D\n',
                      fontsize=18, fontweight='bold', color='#1A1A2E', y=1.06)
        cbar = fig4.colorbar(im, ax=[ax4a, ax4b], shrink=0.85)
        cbar.set_label('mean contacts / cell', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        path4 = os.path.join(FIG_DIR, 'fig_4_contact_heatmap.png')
        fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor=BG)
        plt.close(fig4)
        logger.info(f'  Figure 4 saved: {path4}')

        # ─────────────────────────────────────────────────────────────────
        # FIGURE 4b — Homotypic (same-type) contact rate, 2D vs 3D, grouped
        # bars. The single most direct version of the paper's claim: for
        # each type, does the 3D bar read higher than the 2D bar?
        # ─────────────────────────────────────────────────────────────────
        logger.info('Generating Figure 4b: 2D vs 3D homotypic contact bars ...')
        homo_rows = []
        for t in types_contact:
            v2 = mat_2d[types_contact.index(t), types_contact.index(t)]
            v3 = mat_3d[types_contact.index(t), types_contact.index(t)]
            homo_rows.append((t, v2, v3))

        fig4b, ax4c = plt.subplots(figsize=(1.5 + 1.1 * len(homo_rows), 6), facecolor=BG)
        x = np.arange(len(homo_rows))
        w = 0.35
        v2_vals = [r[1] for r in homo_rows]
        v3_vals = [r[2] for r in homo_rows]
        ax4c.bar(x - w/2, v2_vals, width=w, label='2D (per-slice)', color='#94A3B8')
        ax4c.bar(x + w/2, v3_vals, width=w, label='3D (reconstructed)', color='#DC2626')
        for xi, v2, v3 in zip(x, v2_vals, v3_vals):
            if not np.isnan(v2):
                ax4c.text(xi - w/2, v2, f'{v2:.2f}', ha='center', va='bottom', fontsize=11)
            if not np.isnan(v3):
                ax4c.text(xi + w/2, v3, f'{v3:.2f}', ha='center', va='bottom', fontsize=11)
        ax4c.set_xticks(x)
        ax4c.set_xticklabels([r[0] for r in homo_rows], rotation=30, ha='right', fontsize=13)
        ax4c.set_ylabel('mean same-type contacts / cell', fontsize=14)
        ax4c.set_title('Homotypic contact rate, 2D vs 3D\n',
                       fontsize=15, fontweight='bold', color='#1A1A2E', pad=10)
        ax4c.legend(fontsize=12, frameon=False)
        _finish_ax(ax4c)

        fig4b.tight_layout()
        path4b = os.path.join(FIG_DIR, 'fig_4b_contact_2d_vs_3d_homotypic.png')
        fig4b.savefig(path4b, dpi=200, bbox_inches='tight', facecolor=BG)
        plt.close(fig4b)
        logger.info(f'  Figure 4b saved: {path4b}')
    else:
        # Fallback: 3D-only single panel (previous behavior), when no core
        # has contact_summary_2d.csv.
        n_ct = len(types_contact)
        vmax = np.nanmax(mat_3d) if not np.all(np.isnan(mat_3d)) else 1.0
        fig4, ax4 = plt.subplots(figsize=(1.5 + 1.35 * n_ct, 1.5 + 1.15 * n_ct), facecolor=BG)
        im = _draw_contact_heatmap(
            ax4, mat_3d, types_contact, vmax,
            f'Mean real 3D nucleus-contacts per cell, type × type\n'
            f'Averaged across {n_cores_3d} cores (no 2D contact data available)')
        cbar = fig4.colorbar(im, ax=ax4, shrink=0.85)
        cbar.set_label('mean contacts / cell', fontsize=15)
        cbar.ax.tick_params(labelsize=13)
        fig4.tight_layout()
        path4 = os.path.join(FIG_DIR, 'fig_4_contact_heatmap.png')
        fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor=BG)
        plt.close(fig4)
        logger.info(f'  Figure 4 saved: {path4}  (3D only)')


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
if df_contact_3d is not None:
    logger.info('    aggregate_contact_3d.csv')
if df_contact_2d is not None:
    logger.info('    aggregate_contact_2d.csv')
logger.info('  Figures:')
logger.info('    fig_1_celltype_fraction_2d_vs_3d.png')
logger.info('    fig_2_nn_distances.png')
logger.info('    fig_3_entropy_delta_boxplot.png')
if df_contact_3d is not None:
    logger.info('    fig_4_contact_heatmap.png')
if df_contact_2d is not None:
    logger.info('    fig_4b_contact_2d_vs_3d_homotypic.png')
logger.info('=' * 60)