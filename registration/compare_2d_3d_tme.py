"""
compare_2d_3d_tme.py  (entropy edition)
========================================
Compares 2D and 3D representations of the tumour microenvironment (TME)
for a single TMA core using three spatial analysis modules:

  1. Cell densities
     - Per cell type as cells / mm²  (2D: averaged across slices)
     - Per cell type as cells / mm³  (3D: using reconstructed volume)

  2. Nearest-neighbour distances
     - For each cell, find the nearest cell of every other type
     - Summarise as mean ± std per type-pair
     - Computed in 2D (XY only, averaged across slices) and 3D (XYZ)

  3. Neighbourhood alpha diversity
     - For each cell, collect all neighbours within radius_um
     - Compute Shannon entropy of the neighbour cell-type composition
     - Summarise per-cell entropy distributions for 2D vs 3D
     - Statistical comparison: KS test + Mann-Whitney U (2D vs 3D)
     - Reference: Pentimalli et al., Cell Systems 2025 (NSCLC 3D atlas, Chao/
       Shannon neighbourhood diversity comparison); Bull & Byrne, PLOS Comp Bio
       2023 (weighted PCF / neighbourhood entropy framework)

Inputs
------
  Phenotypes/<CORE>/<CORE>_phenotypes_typed.csv   — 2D per-slice records
  Phenotypes/<CORE>/<CORE>_3d_typed.csv           — 3D reconstructed cells

Outputs
-------
  TME_Analysis/<CORE>/
    cell_density_2d.csv
    cell_density_3d.csv
    nn_distances_2d.csv
    nn_distances_3d.csv
    neighbourhood_entropy_2d.csv   ← per-cell entropy, 2D
    neighbourhood_entropy_3d.csv   ← per-cell entropy, 3D
    entropy_summary.csv            ← mean/std/median entropy + KS/MWU p-values
    summary_comparison.csv         — wide table for direct 2D vs 3D comparison
    figures/
      fig_A_cell_composition.png
      fig_B_nn_distances.png
      fig_C_entropy_comparison.png

Usage
-----
    python compare_2d_3d_tme.py --core_name Core_01
    python compare_2d_3d_tme.py --core_name Core_01 --radius_um 50
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
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import ks_2samp, mannwhitneyu

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
    description='2D vs 3D TME spatial analysis: density, NN distances, '
                'and neighbourhood entropy.'
)
parser.add_argument('--core_name',   type=str,   required=True)
parser.add_argument('--radius_um',   type=float, default=50.0,
                    help='Neighbourhood radius in µm (default: 50).')
parser.add_argument('--pixel_um',    type=float, default=0.4961,
                    help='Pixel size in µm (default: 0.4961).')
parser.add_argument('--section_um',  type=float, default=4.5,
                    help='Section thickness in µm (default: 4.5).')
parser.add_argument('--min_cells',   type=int,   default=10,
                    help='Min cells of a type required for analysis (default: 10).')
args = parser.parse_args()

TARGET_CORE  = args.core_name
RADIUS_UM    = args.radius_um
PIXEL_UM     = args.pixel_um
SECTION_UM   = args.section_um
MIN_CELLS    = args.min_cells

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PHENO_DIR   = os.path.join(config.DATASPACE, 'Phenotypes', TARGET_CORE)
OUT_DIR     = os.path.join(config.DATASPACE, 'TME_Analysis', TARGET_CORE)
FIG_DIR     = os.path.join(OUT_DIR, 'figures')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

TYPED_2D_CSV = os.path.join(PHENO_DIR, f'{TARGET_CORE}_phenotypes_typed.csv')
TYPED_3D_CSV = os.path.join(PHENO_DIR, f'{TARGET_CORE}_3d_typed.csv')

for p, label in [(TYPED_2D_CSV, '2D typed CSV'), (TYPED_3D_CSV, '3D typed CSV')]:
    if not os.path.exists(p):
        logger.error(f'{label} not found: {p}')
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Loading typed phenotype tables ...')
df2d = pd.read_csv(TYPED_2D_CSV)
df3d = pd.read_csv(TYPED_3D_CSV)

logger.info(f'  2D records : {len(df2d):,}  across {df2d["slice_id"].nunique()} slices')
logger.info(f'  3D cells   : {len(df3d):,}')

# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
df2d['x_um'] = df2d['centroid_x'] * PIXEL_UM
df2d['y_um'] = df2d['centroid_y'] * PIXEL_UM

df3d = df3d.rename(columns={
    'centroid_x_um': 'x_um',
    'centroid_y_um': 'y_um',
    'centroid_z_um': 'z_um',
})

all_types_2d = set(df2d['cell_type'].unique())
all_types_3d = set(df3d['cell_type'].unique())
all_types    = sorted(all_types_2d | all_types_3d)
logger.info(f'  Cell types : {all_types}')

valid_types_3d        = [t for t in all_types if (df3d['cell_type'] == t).sum() >= MIN_CELLS]
valid_types_2d_global = [t for t in all_types if (df2d['cell_type'] == t).sum() >= MIN_CELLS]
logger.info(f'  Valid types (3D, >= {MIN_CELLS} cells) : {valid_types_3d}')

slice_ids = df2d['slice_id'].unique()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CELL DENSITIES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info('MODULE 1: Cell densities')

def slice_area_um2(df_slice):
    coords = df_slice[['x_um', 'y_um']].values
    if len(coords) < 3:
        return np.nan
    try:
        return ConvexHull(coords).volume
    except Exception:
        return np.nan

density_2d_rows = []
for sid, grp in df2d.groupby('slice_id'):
    area = slice_area_um2(grp)
    if np.isnan(area) or area == 0:
        continue
    area_mm2 = area / 1e6
    for ct in all_types:
        n = (grp['cell_type'] == ct).sum()
        density_2d_rows.append({
            'slice_id':          sid,
            'cell_type':         ct,
            'n_cells':           n,
            'area_mm2':          round(area_mm2, 4),
            'density_per_mm2':   round(n / area_mm2, 4),
        })

df_density_2d_per_slice = pd.DataFrame(density_2d_rows)
df_density_2d = (df_density_2d_per_slice
                 .groupby('cell_type')
                 .agg(
                     mean_density_per_mm2=('density_per_mm2', 'mean'),
                     std_density_per_mm2 =('density_per_mm2', 'std'),
                     n_slices            =('slice_id',        'nunique'),
                 )
                 .reset_index())

coords_3d = df3d[['x_um', 'y_um', 'z_um']].values
try:
    hull_3d = ConvexHull(coords_3d)
    vol_mm3 = hull_3d.volume / 1e9
except Exception:
    vol_mm3 = np.nan
    logger.warning('Could not compute 3D convex hull — density will be NaN.')

density_3d_rows = []
for ct in all_types:
    n = (df3d['cell_type'] == ct).sum()
    density_3d_rows.append({
        'cell_type':       ct,
        'n_cells':         n,
        'volume_mm3':      round(vol_mm3, 6) if not np.isnan(vol_mm3) else np.nan,
        'density_per_mm3': round(n / vol_mm3, 4) if not np.isnan(vol_mm3) else np.nan,
    })
df_density_3d = pd.DataFrame(density_3d_rows)

df_density_2d.to_csv(os.path.join(OUT_DIR, 'cell_density_2d.csv'), index=False)
df_density_3d.to_csv(os.path.join(OUT_DIR, 'cell_density_3d.csv'), index=False)
logger.info('  Density CSVs saved.')
logger.info(f'  3D core volume estimate: {vol_mm3:.4f} mm³')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — NEAREST-NEIGHBOUR DISTANCES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info('MODULE 2: Nearest-neighbour distances')

def nn_distances(coords_src, coords_tgt):
    if len(coords_tgt) == 0 or len(coords_src) == 0:
        return np.array([np.nan])
    same = np.array_equal(coords_src, coords_tgt)
    k    = 2 if same else 1
    tree = cKDTree(coords_tgt)
    dists, _ = tree.query(coords_src, k=k, workers=-1)
    if same:
        dists = dists[:, 1]
    else:
        dists = dists.ravel()
    return dists[np.isfinite(dists)]

nn_2d_rows = []
for sid in slice_ids:
    grp = df2d[df2d['slice_id'] == sid]
    types_present = [t for t in valid_types_2d_global
                     if (grp['cell_type'] == t).sum() >= MIN_CELLS]
    for src_type in types_present:
        src_coords = grp[grp['cell_type'] == src_type][['x_um', 'y_um']].values
        for tgt_type in types_present:
            tgt_coords = grp[grp['cell_type'] == tgt_type][['x_um', 'y_um']].values
            dists = nn_distances(src_coords, tgt_coords)
            nn_2d_rows.append({
                'slice_id':     sid,
                'src_type':     src_type,
                'tgt_type':     tgt_type,
                'mean_dist_um': np.mean(dists),
                'std_dist_um':  np.std(dists),
                'n_src_cells':  len(src_coords),
            })

df_nn_2d = (pd.DataFrame(nn_2d_rows)
            .groupby(['src_type', 'tgt_type'])
            .agg(
                mean_dist_um     =('mean_dist_um', 'mean'),
                std_across_slices=('mean_dist_um', 'std'),
                n_slices         =('slice_id',     'nunique'),
            )
            .reset_index())

nn_3d_rows = []
for src_type in valid_types_3d:
    src_coords = df3d[df3d['cell_type'] == src_type][['x_um', 'y_um', 'z_um']].values
    for tgt_type in valid_types_3d:
        tgt_coords = df3d[df3d['cell_type'] == tgt_type][['x_um', 'y_um', 'z_um']].values
        dists = nn_distances(src_coords, tgt_coords)
        nn_3d_rows.append({
            'src_type':    src_type,
            'tgt_type':    tgt_type,
            'mean_dist_um': round(np.mean(dists), 3),
            'std_dist_um':  round(np.std(dists),  3),
            'n_src_cells':  len(src_coords),
        })
df_nn_3d = pd.DataFrame(nn_3d_rows)

df_nn_2d.to_csv(os.path.join(OUT_DIR, 'nn_distances_2d.csv'), index=False)
df_nn_3d.to_csv(os.path.join(OUT_DIR, 'nn_distances_3d.csv'), index=False)
logger.info('  NN distance CSVs saved.')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — NEIGHBOURHOOD ALPHA DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────
# Per-cell Shannon entropy of neighbour composition (alpha diversity)
#   H(cell_i) = -sum_t [ p_t * log(p_t) ]
#   where p_t = fraction of neighbours within radius_um that are type t.
#   An isolated cell or one with no neighbours gets H = NaN.
#   Following Pentimalli et al. 2025 (Cell Systems) who validated this
#   approach for 2D vs 3D neighbourhood diversity comparison in NSCLC.
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'MODULE 3: Neighbourhood entropy (radius={RADIUS_UM} µm)')


def shannon_entropy(counts_array):
    """
    Shannon entropy H = -sum(p * log(p)) in nats.
    counts_array : 1D array of counts per cell type (integers >= 0).
    Returns NaN if total == 0.
    """
    total = counts_array.sum()
    if total == 0:
        return np.nan
    p = counts_array / total
    # only non-zero terms contribute
    p_nz = p[p > 0]
    return -np.sum(p_nz * np.log(p_nz))


def neighbourhood_entropy(coords, labels, valid_types, radius_um):
    """
    Compute per-cell Shannon entropy of neighbour composition.

    Parameters
    ----------
    coords      : (N, D) array of cell positions in µm
    labels      : (N,) array of cell type strings
    valid_types : list of types to include
    radius_um   : neighbourhood radius in µm

    Returns
    -------
    df_entropy  : DataFrame with columns [cell_idx, cell_type, n_neighbours,
                                          entropy, <type>_count for each type]
    """
    labels = np.array(labels)
    N      = len(labels)
    T      = len(valid_types)
    type_set = set(valid_types)

    tree       = cKDTree(coords)
    neighbours = tree.query_ball_point(coords, r=radius_um, workers=-1)

    # ── Per-cell entropy ──────────────────────────────────────────────────────
    entropy_rows = []
    # count matrix: rows = cells, cols = valid_types
    count_matrix = np.zeros((N, T), dtype=np.int32)

    for i, nbs in enumerate(neighbours):
        nb_labels = labels[nbs]
        for k, ct in enumerate(valid_types):
            count_matrix[i, k] = np.sum(nb_labels == ct)

    # subtract self contribution for same-type
    for i in range(N):
        own_type = labels[i]
        if own_type in type_set:
            k = valid_types.index(own_type)
            count_matrix[i, k] = max(0, count_matrix[i, k] - 1)

    for i in range(N):
        if labels[i] not in type_set:
            continue
        row = count_matrix[i]
        h   = shannon_entropy(row)
        n_nb = int(row.sum())
        rec  = {'cell_idx':    i,
                'cell_type':   labels[i],
                'n_neighbours': n_nb,
                'entropy':      h}
        for k, ct in enumerate(valid_types):
            rec[f'n_{ct}'] = int(row[k])
        entropy_rows.append(rec)

    df_entropy = pd.DataFrame(entropy_rows)
    return df_entropy


# ── 2D entropy (per slice, then aggregate) ───────────────────────────────────
logger.info('  Computing 2D neighbourhood entropy ...')
entropy_2d_all = []

for sid in slice_ids:
    grp = df2d[df2d['slice_id'] == sid]
    types_here = [t for t in valid_types_2d_global
                  if (grp['cell_type'] == t).sum() >= MIN_CELLS]
    if len(types_here) < 2:
        continue
    coords = grp[['x_um', 'y_um']].values
    labels = grp['cell_type'].values
    mask   = np.isin(labels, types_here)

    df_ent_s = neighbourhood_entropy(
        coords[mask], labels[mask], types_here, RADIUS_UM
    )
    df_ent_s['slice_id'] = sid
    entropy_2d_all.append(df_ent_s)
    logger.info(f'    Slice {sid}: {mask.sum()} cells, '
                f'mean H={df_ent_s["entropy"].mean():.3f} nats')

df_entropy_2d = pd.concat(entropy_2d_all, ignore_index=True)

# ── 3D entropy ────────────────────────────────────────────────────────────────
logger.info('  Computing 3D neighbourhood entropy ...')
mask_3d = df3d['cell_type'].isin(valid_types_3d)
coords_3d_v = df3d[mask_3d][['x_um', 'y_um', 'z_um']].values
labels_3d_v = df3d[mask_3d]['cell_type'].values

df_entropy_3d = neighbourhood_entropy(
    coords_3d_v, labels_3d_v, valid_types_3d, RADIUS_UM
)
logger.info(f'    3D: {len(df_entropy_3d)} cells, '
            f'mean H={df_entropy_3d["entropy"].mean():.3f} nats')

# ── Entropy summary table with statistical tests ──────────────────────────────
logger.info('  Building entropy summary with KS and Mann-Whitney U tests ...')

entropy_summary_rows = []
all_types_entropy = sorted(
    set(df_entropy_2d['cell_type'].unique()) |
    set(df_entropy_3d['cell_type'].unique())
)

for ct in all_types_entropy:
    h2d = df_entropy_2d[df_entropy_2d['cell_type'] == ct]['entropy'].dropna().values
    h3d = df_entropy_3d[df_entropy_3d['cell_type'] == ct]['entropy'].dropna().values
    if len(h2d) < 5 or len(h3d) < 5:
        continue

    ks_stat, ks_p   = ks_2samp(h2d, h3d)
    mwu_stat, mwu_p = mannwhitneyu(h2d, h3d, alternative='two-sided')

    entropy_summary_rows.append({
        'cell_type':        ct,
        # 2D stats
        'mean_H_2d':        round(np.mean(h2d),   4),
        'std_H_2d':         round(np.std(h2d),    4),
        'median_H_2d':      round(np.median(h2d), 4),
        'n_cells_2d':       len(h2d),
        # 3D stats
        'mean_H_3d':        round(np.mean(h3d),   4),
        'std_H_3d':         round(np.std(h3d),    4),
        'median_H_3d':      round(np.median(h3d), 4),
        'n_cells_3d':       len(h3d),
        # delta
        'delta_mean_H':     round(np.mean(h3d) - np.mean(h2d), 4),
        # tests
        'ks_stat':          round(ks_stat,  4),
        'ks_p':             round(ks_p,     4),
        'mwu_stat':         round(mwu_stat, 4),
        'mwu_p':            round(mwu_p,    4),
        # interpretation flag
        'H_increased_in_3D': np.mean(h3d) > np.mean(h2d),
    })

df_entropy_summary = pd.DataFrame(entropy_summary_rows)

# Save all Module 3 outputs
df_entropy_2d.to_csv(os.path.join(OUT_DIR, 'neighbourhood_entropy_2d.csv'), index=False)
df_entropy_3d.to_csv(os.path.join(OUT_DIR, 'neighbourhood_entropy_3d.csv'), index=False)
df_entropy_summary.to_csv(os.path.join(OUT_DIR, 'entropy_summary.csv'), index=False)
logger.info('  Module 3 CSVs saved.')

# Print summary to log
for _, row in df_entropy_summary.iterrows():
    direction = '↑' if row['H_increased_in_3D'] else '↓'
    logger.info(f"    {row['cell_type']:15s}  "
                f"H_2D={row['mean_H_2d']:.3f}  "
                f"H_3D={row['mean_H_3d']:.3f}  "
                f"{direction}  KS p={row['ks_p']:.3e}  MWU p={row['mwu_p']:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Building summary comparison table ...')

nn_merge = (df_nn_2d[['src_type', 'tgt_type', 'mean_dist_um']]
            .rename(columns={'mean_dist_um': 'nn_dist_2d_um'})
            .merge(
                df_nn_3d[['src_type', 'tgt_type', 'mean_dist_um']]
                .rename(columns={'mean_dist_um': 'nn_dist_3d_um'}),
                on=['src_type', 'tgt_type'], how='outer'))
nn_merge['nn_dist_delta_um'] = (nn_merge['nn_dist_3d_um']
                                 - nn_merge['nn_dist_2d_um']).round(3)

summary = nn_merge.copy()
summary.to_csv(os.path.join(OUT_DIR, 'summary_comparison.csv'), index=False)
logger.info(f'  Summary table saved ({len(summary)} type-pairs).')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating figures ...')

import matplotlib.patches as mpatches

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
GRAY = '#90A4AE'

def _type_color(ct):
    return _PALETTE.get(ct, '#888888')

def _finish_ax(ax):
    ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)

bh = 0.35

# ── Pre-compute shared data ───────────────────────────────────────────────────
total_2d_dens = df_density_2d['mean_density_per_mm2'].sum()
total_3d_dens = df_density_3d['density_per_mm3'].sum()
pct_2d = {r['cell_type']: r['mean_density_per_mm2'] / total_2d_dens * 100
          for _, r in df_density_2d.iterrows()}
pct_3d = {r['cell_type']: r['density_per_mm3'] / total_3d_dens * 100
          for _, r in df_density_3d.iterrows()}

nn_merge_fig = (df_nn_2d[['src_type', 'tgt_type', 'mean_dist_um']]
                .rename(columns={'mean_dist_um': 'd2d'})
                .merge(df_nn_3d[['src_type', 'tgt_type', 'mean_dist_um']]
                       .rename(columns={'mean_dist_um': 'd3d'}),
                       on=['src_type', 'tgt_type']))


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Cell-type composition  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
fig_a, ax_comp = plt.subplots(figsize=(8, 5.5), facecolor=BG)
_finish_ax(ax_comp)
types_comp = [t for t in _ALL_TYPES_SORTED if t in pct_2d or t in pct_3d]
y_comp = np.arange(len(types_comp))

for k, ct in enumerate(types_comp):
    v2 = pct_2d.get(ct, 0); v3 = pct_3d.get(ct, 0)
    col = _type_color(ct)
    ax_comp.barh(y_comp[k] + bh/2, v2, height=bh, color=col, alpha=0.45, edgecolor='none')
    ax_comp.barh(y_comp[k] - bh/2, v3, height=bh, color=col, alpha=1.00, edgecolor='none')
    delta = v3 - v2
    ax_comp.text(max(v2, v3) + 0.5, y_comp[k],
                 f'{("+" if delta >= 0 else "")}{delta:.1f}%',
                 va='center', fontsize=9, color='#444')

ax_comp.set_yticks(y_comp);  ax_comp.set_yticklabels(types_comp, fontsize=11)
ax_comp.set_xlabel('% of all cells', fontsize=11)
ax_comp.set_title(f'{TARGET_CORE} — Cell-type composition: 2D vs 3D',
                  fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)
ax_comp.axvline(0, color='#ccc', lw=0.8)
ax_comp.legend(handles=[
    mpatches.Patch(facecolor='#888', alpha=0.45, label='2D (mean density/mm²)'),
    mpatches.Patch(facecolor='#888', alpha=1.0,  label='3D (density/mm³)'),
], fontsize=9, frameon=False, loc='upper right')
fig_a.tight_layout()
path_a = os.path.join(FIG_DIR, 'fig_A_cell_composition.png')
fig_a.savefig(path_a, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_a)
logger.info(f'  Figure A saved: {path_a}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — NN distances scatter
#
# Colour  = source cell type
# Marker  = target cell type
# This dual encoding resolves the overlap problem where multiple same-colour
# dots land on top of each other (one per target type per source type).
# Two separate legends are placed without overlapping data points.
# ─────────────────────────────────────────────────────────────────────────────
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']   # one per type, up to 7
_marker_for = {ct: _MARKERS[i % len(_MARKERS)]
               for i, ct in enumerate(_ALL_TYPES_SORTED)}

fig_b, ax_nn = plt.subplots(figsize=(8, 7.5), facecolor=BG)
_finish_ax(ax_nn)

for _, row in nn_merge_fig.iterrows():
    ax_nn.scatter(row['d2d'], row['d3d'],
                  color=_type_color(row['src_type']),
                  marker=_marker_for.get(row['tgt_type'], 'o'),
                  s=90, alpha=0.88, edgecolors='white', linewidths=0.5, zorder=3)

nn_lim = max(nn_merge_fig['d2d'].max(), nn_merge_fig['d3d'].max()) * 1.12
ax_nn.plot([0, nn_lim], [0, nn_lim], '--', color=GRAY, lw=1.2, zorder=1)

# Label the diagonal
mid = nn_lim * 0.52
ax_nn.text(mid * 1.07, mid * 0.90, '2D = 3D',
           fontsize=8.5, color=GRAY, rotation=42, ha='center', style='italic')

# Quadrant annotation
ax_nn.text(nn_lim * 0.05, nn_lim * 0.92,
           'Points above diagonal:\n3D distance > 2D distance',
           fontsize=8, color='#444', va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', lw=0.7))

ax_nn.set_xlim(0, nn_lim); ax_nn.set_ylim(0, nn_lim)
ax_nn.set_xlabel('2D mean NN distance (µm)', fontsize=11)
ax_nn.set_ylabel('3D mean NN distance (µm)', fontsize=11)
ax_nn.set_title(f'{TARGET_CORE} — Nearest-neighbour distances: 2D vs 3D\n'
                'Colour = source type   Marker = target type',
                fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

# Legend 1: colour = source type  (top-left, inside axes)
src_in_nn = [t for t in _ALL_TYPES_SORTED if t in nn_merge_fig['src_type'].values]
leg_src = ax_nn.legend(
    handles=[mpatches.Patch(fc=_type_color(t), label=t) for t in src_in_nn],
    title='Source type', title_fontsize=8.5,
    fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#ccc',
    loc='upper left')
ax_nn.add_artist(leg_src)

# Legend 2: marker = target type  (bottom-right, inside axes)
tgt_in_nn = [t for t in _ALL_TYPES_SORTED if t in nn_merge_fig['tgt_type'].values]
leg_tgt = ax_nn.legend(
    handles=[plt.Line2D([0], [0], marker=_marker_for.get(t, 'o'),
                        color='#555', linestyle='none',
                        markersize=7, label=t) for t in tgt_in_nn],
    title='Target type', title_fontsize=8.5,
    fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#ccc',
    loc='lower right')

fig_b.tight_layout()
path_b = os.path.join(FIG_DIR, 'fig_B_nn_distances.png')
fig_b.savefig(path_b, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_b)
logger.info(f'  Figure B saved: {path_b}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Neighbourhood entropy comparison
#
# Layout: two horizontal panels
#   Left  — paired violins per cell type (2D pale, 3D solid), vertical layout
#            so cell-type labels sit on a clean y-axis without rotation
#   Right — horizontal bars mean H ± std with delta and KS significance star
#
# Legend labels corrected to entropy units (not density units).
# ─────────────────────────────────────────────────────────────────────────────
types_ent = [t for t in _ALL_TYPES_SORTED
             if t in df_entropy_summary['cell_type'].values]
n_types   = len(types_ent)
y_ent     = np.arange(n_types)

fig_c, (ax_vio, ax_bar) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
for ax in (ax_vio, ax_bar):
    _finish_ax(ax)

# ── Left: horizontal violin pairs (y = cell type, x = entropy value) ─────────
# Using a rotated layout: violins drawn along the x-axis per cell type,
# with 2D (pale) below and 3D (solid) above the cell-type tick.
violin_offset = 0.22
for k, ct in enumerate(types_ent):
    h2d = df_entropy_2d[df_entropy_2d['cell_type'] == ct]['entropy'].dropna().values
    h3d = df_entropy_3d[df_entropy_3d['cell_type'] == ct]['entropy'].dropna().values
    col = _type_color(ct)

    for offset, hvals, alpha_val in [
        (-violin_offset, h2d, 0.38),
        (+violin_offset, h3d, 0.90),
    ]:
        if len(hvals) < 5:
            continue
        vp = ax_vio.violinplot(hvals, positions=[k + offset],
                               widths=0.36, showmedians=True,
                               showextrema=False, vert=True)
        for body in vp['bodies']:
            body.set_facecolor(col)
            body.set_alpha(alpha_val)
            body.set_edgecolor('none')
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(1.6)

ax_vio.set_xticks(y_ent)
ax_vio.set_xticklabels(types_ent, rotation=28, ha='right', fontsize=10)
ax_vio.set_ylabel('Shannon entropy H (nats)', fontsize=11)
ax_vio.set_title('Per-cell neighbourhood entropy distributions\n'
                 f'Radius = {RADIUS_UM:.0f} µm',
                 fontsize=11, fontweight='bold', color='#1A1A2E')
ax_vio.legend(handles=[
    mpatches.Patch(facecolor='#888', alpha=0.38, label='2D (per section)'),
    mpatches.Patch(facecolor='#888', alpha=0.90, label='3D (reconstructed volume)'),
], fontsize=9, frameon=False, loc='upper left')

# ── Right: mean ± std horizontal bars, y = cell type ─────────────────────────
bar_w = 0.32
for k, ct in enumerate(types_ent):
    row = df_entropy_summary[df_entropy_summary['cell_type'] == ct].iloc[0]
    col = _type_color(ct)
    sig = row['ks_p'] < 0.05

    ax_bar.barh(k + bar_w / 2, row['mean_H_2d'], height=bar_w,
                xerr=row['std_H_2d'], color=col, alpha=0.38,
                edgecolor='none', capsize=3)
    ax_bar.barh(k - bar_w / 2, row['mean_H_3d'], height=bar_w,
                xerr=row['std_H_3d'], color=col, alpha=0.90,
                edgecolor='none', capsize=3)

    x_ann = max(row['mean_H_2d'] + row['std_H_2d'],
                row['mean_H_3d'] + row['std_H_3d']) + 0.03
    label = rf"$\Delta${row['delta_mean_H']:+.2f}"
    if sig:
        label += '  ★'
    ax_bar.text(x_ann, k, label, va='center', fontsize=8.5,
                color='#1A237E' if sig else '#777')

ax_bar.set_yticks(y_ent)
ax_bar.set_yticklabels(types_ent, fontsize=10)
ax_bar.set_xlabel('Mean Shannon entropy H (nats)', fontsize=11)
ax_bar.set_title('Mean H ± std per cell type\n'
                 '★ = KS test p < 0.05',
                 fontsize=11, fontweight='bold', color='#1A1A2E')
ax_bar.legend(handles=[
    mpatches.Patch(facecolor='#888', alpha=0.38, label='2D (per section)'),
    mpatches.Patch(facecolor='#888', alpha=0.90, label='3D (reconstructed volume)'),
], fontsize=9, frameon=False, loc='lower right')

fig_c.suptitle(
    f'{TARGET_CORE}  —  Neighbourhood alpha diversity: 2D vs 3D\n'
    f'Shannon entropy H of cell-type composition within {RADIUS_UM:.0f} µm radius',
    fontsize=12, fontweight='bold', color='#1A1A2E', y=1.01)

fig_c.tight_layout()
path_c = os.path.join(FIG_DIR, 'fig_C_entropy_comparison.png')
fig_c.savefig(path_c, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_c)
logger.info(f'  Figure C saved: {path_c}')


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY LOG
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'DONE — {TARGET_CORE}')
logger.info(f'  Output directory : {OUT_DIR}')
logger.info('  CSVs written:')
logger.info('    cell_density_2d/3d.csv')
logger.info('    nn_distances_2d/3d.csv')
logger.info('    neighbourhood_entropy_2d/3d.csv')
logger.info('    entropy_summary.csv  (KS + MWU tests)')
logger.info('    summary_comparison.csv')
logger.info('  Figures written:')
logger.info('    fig_A_cell_composition.png')
logger.info('    fig_B_nn_distances.png')
logger.info('    fig_C_entropy_comparison.png   (violin + bar, 2D vs 3D)')
logger.info('=' * 60)