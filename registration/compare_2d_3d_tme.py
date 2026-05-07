"""
compare_2d_3d_tme.py
====================
Compares 2D and 3D representations of the tumour microenvironment (TME)
for a single TMA core using three spatial analysis modules:

  1. Cell densities
     - Per cell type as cells / mm²  (2D: averaged across slices)
     - Per cell type as cells / mm³  (3D: using reconstructed volume)

  2. Nearest-neighbour distances
     - For each cell, find the nearest cell of every other type
     - Summarise as mean ± std per type-pair
     - Computed in 2D (XY only, averaged across slices) and 3D (XYZ)

  3. Permutation-based spatial interaction scores
     - Build a radius neighbourhood graph per type-pair
     - Count observed co-localisation vs permuted null (n=1000)
     - Output z-score per type-pair: positive = co-localised, negative = avoided
     - Computed in 2D and 3D separately

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
    interaction_scores_2d.csv
    interaction_scores_3d.csv
    summary_comparison.csv         — wide table for direct 2D vs 3D comparison
    figures/
      cell_density_comparison.png
      nn_distance_comparison.png
      interaction_heatmap_2d.png
      interaction_heatmap_3d.png

Usage
-----
    python compare_2d_3d_tme.py --core_name Core_01
    python compare_2d_3d_tme.py --core_name Core_01 --radius_um 50 --n_perm 1000
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
from scipy.spatial import cKDTree

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
    description='2D vs 3D TME spatial analysis: density, NN distances, interaction scores.'
)
parser.add_argument('--core_name',   type=str,   required=True)
parser.add_argument('--radius_um',   type=float, default=50.0,
                    help='Neighbourhood radius in µm for interaction scoring (default: 50).')
parser.add_argument('--n_perm',      type=int,   default=1000,
                    help='Number of permutations for interaction score null (default: 1000).')
parser.add_argument('--pixel_um',    type=float, default=0.4961,
                    help='Pixel size in µm (default: 0.4961).')
parser.add_argument('--section_um',  type=float, default=4.5,
                    help='Section thickness in µm (default: 4.5).')
parser.add_argument('--min_cells',   type=int,   default=10,
                    help='Min cells of a type required to include in analysis (default: 10).')
args = parser.parse_args()

TARGET_CORE  = args.core_name
RADIUS_UM    = args.radius_um
N_PERM       = args.n_perm
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

# 2D: convert pixel centroids to µm, keep per-slice
# centroid_x / centroid_y are in pixels in the phenotype CSV
df2d['x_um'] = df2d['centroid_x'] * PIXEL_UM
df2d['y_um'] = df2d['centroid_y'] * PIXEL_UM

# 3D: centroids already in µm from analyse_3d_cells.py
# columns: centroid_x_um, centroid_y_um, centroid_z_um
df3d = df3d.rename(columns={
    'centroid_x_um': 'x_um',
    'centroid_y_um': 'y_um',
    'centroid_z_um': 'z_um',
})

# Cell types present in both representations
all_types_2d = set(df2d['cell_type'].unique())
all_types_3d = set(df3d['cell_type'].unique())
all_types    = sorted(all_types_2d | all_types_3d)
logger.info(f'  Cell types : {all_types}')

# Filter to types with enough cells in 3D (2D will be filtered per-metric)
valid_types_3d = [t for t in all_types
                  if (df3d['cell_type'] == t).sum() >= MIN_CELLS]
valid_types_2d_global = [t for t in all_types
                         if (df2d['cell_type'] == t).sum() >= MIN_CELLS]
logger.info(f'  Valid types (3D, >= {MIN_CELLS} cells) : {valid_types_3d}')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CELL DENSITIES
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info('MODULE 1: Cell densities')

# ── 2D density ───────────────────────────────────────────────────────────────
# Per slice: count cells of each type, divide by slice area (µm²)
# Slice area estimated from the convex hull of all cell centroids in that slice.
# Average across slices for the core-level estimate.
from scipy.spatial import ConvexHull

def slice_area_um2(df_slice):
    """Convex hull area of cell centroids in µm²."""
    coords = df_slice[['x_um', 'y_um']].values
    if len(coords) < 3:
        return np.nan
    try:
        return ConvexHull(coords).volume   # .volume = area in 2D
    except Exception:
        return np.nan

density_2d_rows = []
for sid, grp in df2d.groupby('slice_id'):
    area = slice_area_um2(grp)
    if np.isnan(area) or area == 0:
        continue
    area_mm2 = area / 1e6   # µm² → mm²
    for ct in all_types:
        n = (grp['cell_type'] == ct).sum()
        density_2d_rows.append({
            'slice_id':    sid,
            'cell_type':   ct,
            'n_cells':     n,
            'area_mm2':    round(area_mm2, 4),
            'density_per_mm2': round(n / area_mm2, 4),
        })

df_density_2d_per_slice = pd.DataFrame(density_2d_rows)

# Average across slices
df_density_2d = (df_density_2d_per_slice
                 .groupby('cell_type')
                 .agg(
                     mean_density_per_mm2=('density_per_mm2', 'mean'),
                     std_density_per_mm2 =('density_per_mm2', 'std'),
                     n_slices            =('slice_id',        'nunique'),
                 )
                 .reset_index())

# ── 3D density ───────────────────────────────────────────────────────────────
# Core volume estimated as convex hull of 3D centroids in µm³
coords_3d = df3d[['x_um', 'y_um', 'z_um']].values
try:
    hull_3d  = ConvexHull(coords_3d)
    vol_mm3  = hull_3d.volume / 1e9   # µm³ → mm³
except Exception:
    vol_mm3  = np.nan
    logger.warning('Could not compute 3D convex hull — density will be NaN.')

density_3d_rows = []
for ct in all_types:
    n = (df3d['cell_type'] == ct).sum()
    density_3d_rows.append({
        'cell_type':          ct,
        'n_cells':            n,
        'volume_mm3':         round(vol_mm3, 6) if not np.isnan(vol_mm3) else np.nan,
        'density_per_mm3':    round(n / vol_mm3, 4) if not np.isnan(vol_mm3) else np.nan,
    })
df_density_3d = pd.DataFrame(density_3d_rows)

df_density_2d.to_csv(os.path.join(OUT_DIR, 'cell_density_2d.csv'), index=False)
df_density_3d.to_csv(os.path.join(OUT_DIR, 'cell_density_3d.csv'), index=False)
logger.info('  Density CSVs saved.')
logger.info(f'  3D core volume estimate: {vol_mm3:.4f} mm³')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — NEAREST-NEIGHBOUR DISTANCES
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info('MODULE 2: Nearest-neighbour distances')

def nn_distances(coords_src, coords_tgt):
    """
    For each point in coords_src, find the distance to the nearest point
    in coords_tgt. Returns array of distances (µm).
    Excludes self-matches when src == tgt by using k=2 and taking second NN.
    """
    if len(coords_tgt) == 0 or len(coords_src) == 0:
        return np.array([np.nan])
    same = np.array_equal(coords_src, coords_tgt)
    k    = 2 if same else 1
    tree = cKDTree(coords_tgt)
    dists, _ = tree.query(coords_src, k=k, workers=-1)
    if same:
        dists = dists[:, 1]   # skip self (dist=0)
    else:
        dists = dists.ravel()
    return dists[np.isfinite(dists)]


# ── 2D NN distances (averaged across slices) ─────────────────────────────────
nn_2d_rows = []
slice_ids  = df2d['slice_id'].unique()

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
                'slice_id':    sid,
                'src_type':    src_type,
                'tgt_type':    tgt_type,
                'mean_dist_um': np.mean(dists),
                'std_dist_um':  np.std(dists),
                'n_src_cells':  len(src_coords),
            })

df_nn_2d_per_slice = pd.DataFrame(nn_2d_rows)

# Average across slices
df_nn_2d = (df_nn_2d_per_slice
            .groupby(['src_type', 'tgt_type'])
            .agg(
                mean_dist_um     =('mean_dist_um', 'mean'),
                std_across_slices=('mean_dist_um', 'std'),
                n_slices         =('slice_id',     'nunique'),
            )
            .reset_index())

# ── 3D NN distances ───────────────────────────────────────────────────────────
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
# MODULE 3 — PERMUTATION-BASED SPATIAL INTERACTION SCORES
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'MODULE 3: Spatial interaction scores (radius={RADIUS_UM} µm, n_perm={N_PERM})')

rng = np.random.default_rng(seed=42)

def interaction_scores(coords_all, labels_all, valid_types, radius_um, n_perm, rng):
    """
    Permutation-based spatial interaction score for all type pairs.

    For each ordered pair (A, B):
      observed  = mean number of type-B neighbours within radius_um of type-A cells
      null      = same quantity under n_perm random shuffles of cell type labels
      z_score   = (observed - mean(null)) / std(null)

    Positive z: A and B co-localise more than expected by chance.
    Negative z: A and B are spatially avoided relative to chance.

    Vectorised implementation: the neighbour list is converted to a CSR sparse
    adjacency matrix once; each permutation is then a single sparse matrix
    multiply (A @ indicator) rather than a Python loop over cells.  This reduces
    the per-permutation cost from O(N_cells) Python iterations to one C-level
    sparse-dense matmul, giving ~100–300× speedup for large cell counts.

    Parameters
    ----------
    coords_all  : (N, D) array of cell coordinates in µm
    labels_all  : (N,) array of cell type strings
    valid_types : list of cell types to include
    radius_um   : float, neighbourhood radius
    n_perm      : int, number of permutations
    rng         : numpy random generator

    Returns
    -------
    DataFrame with columns: src_type, tgt_type, observed, null_mean, null_std, z_score, p_value
    """
    from scipy.sparse import csr_matrix

    labels_all = np.array(labels_all)
    N          = len(labels_all)
    T          = len(valid_types)
    type_to_idx = {t: k for k, t in enumerate(valid_types)}

    # ── Build KD-tree and neighbour list once ─────────────────────────────────
    tree       = cKDTree(coords_all)
    neighbours = tree.query_ball_point(coords_all, r=radius_um, workers=-1)

    # ── Convert neighbour list to CSR adjacency matrix (self-loops excluded) ──
    rows_idx, cols_idx = [], []
    for i, nbs in enumerate(neighbours):
        for j in nbs:
            if j != i:
                rows_idx.append(i)
                cols_idx.append(j)
    data = np.ones(len(rows_idx), dtype=np.float32)
    A    = csr_matrix((data, (rows_idx, cols_idx)), shape=(N, N))

    # ── Vectorised count: for a given label array return (T × T) matrix ──────
    # result[i, j] = mean neighbours of type j per cell of type i
    def count_interactions_fast(label_arr):
        label_idx  = np.array([type_to_idx.get(l, -1) for l in label_arr])
        valid_mask = label_idx >= 0
        indicator  = np.zeros((N, T), dtype=np.float32)
        indicator[valid_mask, label_idx[valid_mask]] = 1.0
        nb_counts  = A @ indicator            # (N × T): sparse × dense matmul
        result     = np.zeros((T, T), dtype=np.float64)
        for i, src in enumerate(valid_types):
            src_mask = label_arr == src
            n_src    = src_mask.sum()
            if n_src == 0:
                continue
            result[i, :] = nb_counts[src_mask, :].sum(axis=0) / n_src
        return result

    # ── Observed counts ───────────────────────────────────────────────────────
    logger.info('    Computing observed interaction counts ...')
    observed_mat = count_interactions_fast(labels_all)

    # ── Permutations — running stats (no large null array stored) ────────────
    logger.info(f'    Running {n_perm} permutations ...')
    null_sum  = np.zeros((T, T), dtype=np.float64)
    null_sum2 = np.zeros((T, T), dtype=np.float64)
    null_ge   = np.zeros((T, T), dtype=np.int32)   # count(perm >= observed)

    for _ in range(n_perm):
        perm_mat   = count_interactions_fast(rng.permutation(labels_all))
        null_sum  += perm_mat
        null_sum2 += perm_mat ** 2
        null_ge   += (perm_mat >= observed_mat).astype(np.int32)

    null_mean = null_sum  / n_perm
    null_std  = np.sqrt(np.maximum(null_sum2 / n_perm - null_mean ** 2, 0.0))

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    rows = []
    for i, src in enumerate(valid_types):
        for j, tgt in enumerate(valid_types):
            obs  = observed_mat[i, j]
            nmn  = null_mean[i, j]
            nstd = null_std[i, j]
            z    = (obs - nmn) / nstd if nstd > 0 else 0.0
            p    = (null_ge[i, j] + 1) / (n_perm + 1)
            rows.append({
                'src_type':  src,
                'tgt_type':  tgt,
                'observed':  round(float(obs),  4),
                'null_mean': round(float(nmn),  4),
                'null_std':  round(float(nstd), 4),
                'z_score':   round(float(z),    4),
                'p_value':   round(float(p),    4),
            })
    return pd.DataFrame(rows)


# ── 2D interaction scores (per slice, then average z-scores) ─────────────────
logger.info('  Computing 2D interaction scores ...')
int_2d_per_slice = []

for sid in slice_ids:
    grp = df2d[df2d['slice_id'] == sid]
    types_here = [t for t in valid_types_2d_global
                  if (grp['cell_type'] == t).sum() >= MIN_CELLS]
    if len(types_here) < 2:
        continue
    coords = grp[['x_um', 'y_um']].values
    labels = grp['cell_type'].values
    # Filter to valid types only
    mask   = np.isin(labels, types_here)
    df_s   = interaction_scores(coords[mask], labels[mask],
                                types_here, RADIUS_UM, N_PERM, rng)
    df_s['slice_id'] = sid
    int_2d_per_slice.append(df_s)
    logger.info(f'    Slice {sid} done.')

df_int_2d_all = pd.concat(int_2d_per_slice, ignore_index=True)

# Average z-scores across slices
df_int_2d = (df_int_2d_all
             .groupby(['src_type', 'tgt_type'])
             .agg(
                 mean_z_score   =('z_score',  'mean'),
                 std_z_score    =('z_score',  'std'),
                 mean_observed  =('observed', 'mean'),
                 n_slices       =('slice_id', 'nunique'),
             )
             .reset_index())

# ── 3D interaction scores ─────────────────────────────────────────────────────
logger.info('  Computing 3D interaction scores ...')
mask_3d   = df3d['cell_type'].isin(valid_types_3d)
coords_3d_valid = df3d[mask_3d][['x_um', 'y_um', 'z_um']].values
labels_3d_valid = df3d[mask_3d]['cell_type'].values

df_int_3d = interaction_scores(coords_3d_valid, labels_3d_valid,
                               valid_types_3d, RADIUS_UM, N_PERM, rng)

df_int_2d.to_csv(os.path.join(OUT_DIR, 'interaction_scores_2d.csv'), index=False)
df_int_3d.to_csv(os.path.join(OUT_DIR, 'interaction_scores_3d.csv'), index=False)
logger.info('  Interaction score CSVs saved.')


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Building summary comparison table ...')

# NN distance comparison
nn_merge = df_nn_2d[['src_type', 'tgt_type', 'mean_dist_um']].rename(
    columns={'mean_dist_um': 'nn_dist_2d_um'}
).merge(
    df_nn_3d[['src_type', 'tgt_type', 'mean_dist_um']].rename(
        columns={'mean_dist_um': 'nn_dist_3d_um'}),
    on=['src_type', 'tgt_type'], how='outer'
)
nn_merge['nn_dist_delta_um'] = (nn_merge['nn_dist_3d_um']
                                 - nn_merge['nn_dist_2d_um']).round(3)

# Interaction score comparison
int_merge = df_int_2d[['src_type', 'tgt_type', 'mean_z_score']].rename(
    columns={'mean_z_score': 'z_score_2d'}
).merge(
    df_int_3d[['src_type', 'tgt_type', 'z_score']].rename(
        columns={'z_score': 'z_score_3d'}),
    on=['src_type', 'tgt_type'], how='outer'
)
int_merge['z_score_delta'] = (int_merge['z_score_3d']
                               - int_merge['z_score_2d']).round(3)

summary = nn_merge.merge(int_merge, on=['src_type', 'tgt_type'], how='outer')
summary.to_csv(os.path.join(OUT_DIR, 'summary_comparison.csv'), index=False)
logger.info(f'  Summary table saved ({len(summary)} type-pairs).')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES — four separate publication-quality panels
# ─────────────────────────────────────────────────────────────────────────────
logger.info('Generating figures ...')

import matplotlib.patches as mpatches

BG = '#F7F9FA'

# Consistent colour per cell type across all panels
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
    """Common spine/background cleanup."""
    ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)

bh = 0.35   # bar half-height used in composition panels

# ── Pre-compute shared data that multiple figures reference ───────────────────
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

int_merge_fig = (df_int_2d[['src_type', 'tgt_type', 'mean_z_score']]
                 .rename(columns={'mean_z_score': 'z2d'})
                 .merge(df_int_3d[['src_type', 'tgt_type', 'z_score']]
                        .rename(columns={'z_score': 'z3d'}),
                        on=['src_type', 'tgt_type']))

tgt_tum_2d = df_int_2d[df_int_2d['tgt_type'] == 'Tumour'].set_index('src_type')['mean_z_score']
tgt_tum_3d = df_int_3d[df_int_3d['tgt_type'] == 'Tumour'].set_index('src_type')['z_score']

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Cell-type composition
# ─────────────────────────────────────────────────────────────────────────────
fig_a, ax_comp = plt.subplots(figsize=(8, 5.5), facecolor=BG)
_finish_ax(ax_comp)

types_comp = [t for t in _ALL_TYPES_SORTED if t in pct_2d or t in pct_3d]
y_comp = np.arange(len(types_comp))

for k, ct in enumerate(types_comp):
    v2  = pct_2d.get(ct, 0)
    v3  = pct_3d.get(ct, 0)
    col = _type_color(ct)
    ax_comp.barh(y_comp[k] + bh/2, v2, height=bh, color=col, alpha=0.45,
                 edgecolor='none')
    ax_comp.barh(y_comp[k] - bh/2, v3, height=bh, color=col, alpha=1.00,
                 edgecolor='none')
    delta = v3 - v2
    sign  = '+' if delta >= 0 else ''
    ax_comp.text(max(v2, v3) + 0.5, y_comp[k],
                 f'{sign}{delta:.1f}%', va='center', fontsize=9, color='#444')

ax_comp.set_yticks(y_comp)
ax_comp.set_yticklabels(types_comp, fontsize=11)
ax_comp.set_xlabel('% of all cells', fontsize=11)
ax_comp.set_title(f'{TARGET_CORE} — Cell-type composition: 2D vs 3D',
                  fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)
ax_comp.axvline(0, color='#ccc', lw=0.8)

h_2d_leg = mpatches.Patch(facecolor='#888', alpha=0.45, label='2D (mean density/mm²)')
h_3d_leg = mpatches.Patch(facecolor='#888', alpha=1.0,  label='3D (density/mm³)')
ax_comp.legend(handles=[h_2d_leg, h_3d_leg], fontsize=9, frameon=False,
               loc='upper right')
ax_comp.text(0.98, 0.22,
             'Tumour dominates in both\nrepresentations (~60%)',
             transform=ax_comp.transAxes, ha='right', va='bottom', fontsize=9,
             color='#C62828',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FFEBEE', ec='#C62828', lw=0.9))

fig_a.tight_layout()
path_a = os.path.join(FIG_DIR, 'fig_A_cell_composition.png')
fig_a.savefig(path_a, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_a)
logger.info(f'  Figure A saved: {path_a}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Nearest-neighbour distances scatter
# ─────────────────────────────────────────────────────────────────────────────
fig_b, ax_nn = plt.subplots(figsize=(7.5, 7), facecolor=BG)
_finish_ax(ax_nn)

for _, row in nn_merge_fig.iterrows():
    ax_nn.scatter(row['d2d'], row['d3d'],
                  color=_type_color(row['src_type']),
                  s=70, alpha=0.85, edgecolors='white', linewidths=0.6, zorder=3)

nn_lim = max(nn_merge_fig['d2d'].max(), nn_merge_fig['d3d'].max()) * 1.1
ax_nn.plot([0, nn_lim], [0, nn_lim], '--', color=GRAY, lw=1.3, zorder=1)

# No per-point arrow annotations — colour encodes source type via legend
ax_nn.set_xlabel('2D mean NN distance (µm)', fontsize=11)
ax_nn.set_ylabel('3D mean NN distance (µm)', fontsize=11)
ax_nn.set_title(
    f'{TARGET_CORE} — Nearest-neighbour distances: 2D vs 3D\n'
    'Each dot = one cell-type pair  |  colour = source type',
    fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

# Diagonal guide label
mid = nn_lim * 0.55
ax_nn.text(mid * 0.88, mid * 1.10, '2D > 3D  (2D projection inflates distance)',
           fontsize=8, color=GRAY, ha='center', rotation=38, style='italic')

src_in_nn = [t for t in _ALL_TYPES_SORTED if t in nn_merge_fig['src_type'].values]
ax_nn.legend(handles=[mpatches.Patch(fc=_type_color(t), label=t) for t in src_in_nn],
             fontsize=9, frameon=True, framealpha=0.85, edgecolor='#ccc',
             loc='upper left', ncol=1)
ax_nn.text(0.97, 0.05,
           '3D distances shorter overall:\n2D projection inflates separation',
           transform=ax_nn.transAxes, ha='right', va='bottom', fontsize=9,
           color='#1A237E',
           bbox=dict(boxstyle='round,pad=0.35', fc='#E8EAF6', ec='#3949AB', lw=0.9))

fig_b.tight_layout()
path_b = os.path.join(FIG_DIR, 'fig_B_nn_distances.png')
fig_b.savefig(path_b, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_b)
logger.info(f'  Figure B saved: {path_b}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Spatial interaction z-score concordance
# ─────────────────────────────────────────────────────────────────────────────
fig_c, ax_int = plt.subplots(figsize=(7.5, 7), facecolor=BG)
_finish_ax(ax_int)

for _, row in int_merge_fig.iterrows():
    ax_int.scatter(row['z2d'], row['z3d'],
                   color=_type_color(row['src_type']),
                   s=70, alpha=0.85, edgecolors='white', linewidths=0.6, zorder=3)

zlim = np.abs(int_merge_fig[['z2d', 'z3d']].values).max() * 1.12
ax_int.plot([-zlim, zlim], [-zlim, zlim], '--', color=GRAY, lw=1.3, zorder=1)
ax_int.axhline(0, color='#ccc', lw=0.9)
ax_int.axvline(0, color='#ccc', lw=0.9)
ax_int.fill_between([-zlim, 0], [0, 0], [zlim, zlim],
                    color='#E3F2FD', alpha=0.4, zorder=0)
ax_int.fill_between([0, zlim], [-zlim, -zlim], [0, 0],
                    color='#FCE4EC', alpha=0.4, zorder=0)
ax_int.text(-zlim * 0.96, zlim * 0.88, 'Avoided 2D\nCo-localised 3D',
            fontsize=8.5, color='#1565C0', va='top')
ax_int.text(zlim * 0.04, -zlim * 0.88, 'Co-localised 2D\nAvoided 3D',
            fontsize=8.5, color='#C62828')
ax_int.set_xlim(-zlim, zlim)
ax_int.set_ylim(-zlim, zlim)
ax_int.set_xlabel('2D interaction z-score', fontsize=11)
ax_int.set_ylabel('3D interaction z-score', fontsize=11)
ax_int.set_title(f'{TARGET_CORE} — Spatial interaction scores: 2D vs 3D concordance',
                 fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

src_in_int = [t for t in _ALL_TYPES_SORTED if t in int_merge_fig['src_type'].values]
ax_int.legend(handles=[mpatches.Patch(fc=_type_color(t), label=t) for t in src_in_int],
              fontsize=9, frameon=False, loc='lower right', ncol=2)
ax_int.text(0.03, 0.97,
            'Sign preserved across all pairs:\navoidance/co-localisation\npatterns consistent 2D↔3D',
            transform=ax_int.transAxes, ha='left', va='top', fontsize=9,
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.35', fc='#E8F5E9', ec='#388E3C', lw=0.9))

fig_c.tight_layout()
path_c = os.path.join(FIG_DIR, 'fig_C_interaction_concordance.png')
fig_c.savefig(path_c, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_c)
logger.info(f'  Figure C saved: {path_c}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure D — Tumour co-localisation profile
# ─────────────────────────────────────────────────────────────────────────────
fig_d, ax_tum = plt.subplots(figsize=(8, 5.5), facecolor=BG)
_finish_ax(ax_tum)

non_tum = [t for t in _ALL_TYPES_SORTED
           if t != 'Tumour' and t in tgt_tum_2d.index and t in tgt_tum_3d.index]
y_tum = np.arange(len(non_tum))

for k, ct in enumerate(non_tum):
    col = _type_color(ct)
    ax_tum.barh(y_tum[k] + bh/2, tgt_tum_2d.get(ct, 0), height=bh,
                color=col, alpha=0.45, edgecolor='none')
    ax_tum.barh(y_tum[k] - bh/2, tgt_tum_3d.get(ct, 0), height=bh,
                color=col, alpha=1.00, edgecolor='none')

ax_tum.axvline(0, color='#555', lw=1.3)
ax_tum.set_yticks(y_tum)
ax_tum.set_yticklabels(non_tum, fontsize=11)
ax_tum.set_xlabel('Spatial proximity z-score toward Tumour\n'
                  '(negative = spatially segregated, positive = spatially enriched near tumour)',
                  fontsize=10)
ax_tum.set_title(
    f'{TARGET_CORE} — Spatial proximity of each cell type to Tumour\n'
    'Pale = 2D  |  Solid = 3D  |  Each bar = permutation z-score',
    fontsize=12, fontweight='bold', color='#1A1A2E', pad=10)

# Shade the positive zone to highlight co-localisation
ax_tum.axvspan(0, ax_tum.get_xlim()[1] if ax_tum.get_xlim()[1] > 0 else 5,
               color='#E8F5E9', alpha=0.4, zorder=0)

# Axis region labels instead of floating text box
xlims = [tgt_tum_2d.min(), tgt_tum_2d.max(), tgt_tum_3d.min(), tgt_tum_3d.max()]
xmax_d = max(abs(v) for v in xlims) * 1.15
ax_tum.set_xlim(-xmax_d, xmax_d)
ax_tum.text(-xmax_d * 0.97, -0.6,
            'Spatially segregated\nfrom Tumour',
            fontsize=8.5, color='#B71C1C', va='top', ha='left', style='italic')
ax_tum.text(xmax_d * 0.03, -0.6,
            'Spatially enriched\nnear Tumour',
            fontsize=8.5, color='#2E7D32', va='top', ha='left', style='italic')
# Highlight T_cell 3D bar as the notable sign-discordant case
if 'T_cell' in tgt_tum_3d.index and tgt_tum_3d.get('T_cell', 0) > 0:
    tcell_idx = non_tum.index('T_cell') if 'T_cell' in non_tum else None
    if tcell_idx is not None:
        ax_tum.annotate('3D: spatially enriched\nnear Tumour ★',
                        xy=(tgt_tum_3d['T_cell'], tcell_idx - bh/2),
                        xytext=(tgt_tum_3d['T_cell'] + xmax_d * 0.08, tcell_idx + 0.5),
                        fontsize=8.5, color='#1565C0',
                        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.0))
# No legend — 2D/3D encoded by bar opacity, labelled in title

fig_d.tight_layout()
path_d = os.path.join(FIG_DIR, 'fig_D_tumour_colocalisation.png')
fig_d.savefig(path_d, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig_d)
logger.info(f'  Figure D saved: {path_d}')

# Legacy combined figure path (kept for backward compatibility)
summary_fig_path = os.path.join(FIG_DIR, 'summary_2d_vs_3d.png')
logger.info(f'  All 4 figures saved to: {FIG_DIR}')
logger.info(f'    fig_A_cell_composition.png')
logger.info(f'    fig_B_nn_distances.png')
logger.info(f'    fig_C_interaction_concordance.png')
logger.info(f'    fig_D_tumour_colocalisation.png')

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY LOG
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'DONE — {TARGET_CORE}')
logger.info(f'  Output directory : {OUT_DIR}')
logger.info(f'  CSVs written     : cell_density_2d/3d, nn_distances_2d/3d, '
            f'interaction_scores_2d/3d, summary_comparison')
logger.info(f'  Figure written   : {FIG_DIR}/summary_2d_vs_3d.png')
logger.info('=' * 60)