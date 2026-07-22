"""
render_3d_cells.py
====================
Render selected 3D reconstructed cells as real, surface-rendered interactive
3D meshes (marching cubes), reading directly from link_3d_cells.py's saved
3D label volume — no need to re-run or touch link_3d_cells.py itself.

By default, renders the exact same set of cells sampled for the 2D tile QC
montages in link_3d_cells.py (same random seed, same min_confirmed filter,
same sample size), so the two QC views correspond 1:1 for direct comparison.
Use --cell_ids to render specific cells on demand instead (e.g. ones a
pathologist flags after reviewing the 2D tiles).

Output: one self-contained, interactive HTML per cell (rotate/zoom in any
browser) — meant to replace downloading a downsampled volume and opening it
in FIJI's 3D viewer for a quick per-cell sanity check.

Requires plotly (not otherwise used elsewhere in this pipeline) and
scikit-image (already a dependency via other scripts' skimage.transform use).

Usage
-----
    python render_3d_cells.py --core_name Core_01
    python render_3d_cells.py --core_name Core_01 --cell_ids 105412,88213
    python render_3d_cells.py --core_name Core_01 --cell_ids 1-50
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import marching_cubes
import plotly.graph_objects as go

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Physical units — must match link_3d_cells.py exactly, or meshes will have
# the wrong proportions (z is much thicker per-voxel than xy).
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

N_SAMPLES_DEFAULT = 50   # matches link_3d_cells.py's N_TILE_SAMPLES
PAD_VOXELS        = 3    # padding around each cell's bounding box, in voxels


def parse_cell_ids(spec: str) -> list:
    """Comma-separated cell IDs and/or 'lo-hi' ranges, e.g. '105412,88213' or '1-50'."""
    ids = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            ids.extend(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            ids.append(int(part))
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Render 3D reconstructed cells as interactive surface meshes.'
)
parser.add_argument('--core_name', type=str, required=True)
parser.add_argument('--cell_ids', type=str, default=None,
                    help="Explicit cell_id_3d values to render, comma-separated and/or "
                         "'lo-hi' ranges (e.g. '105412,88213' or '1-50'). If omitted, "
                         "renders the same random sample used by link_3d_cells.py's "
                         "2D tile QC montages (same seed, same min_confirmed filter, "
                         "same sample size), so results correspond 1:1.")
parser.add_argument('--min_confirmed', type=int, default=2,
                    help='Only used for the default (non --cell_ids) sample: minimum '
                         'z_span_slices to be eligible (default: 2, matches '
                         'link_3d_cells.py\'s --min_confirmed default).')
parser.add_argument('--n_samples', type=int, default=N_SAMPLES_DEFAULT,
                    help=f'Only used for the default sample: how many cells to render '
                         f'(default: {N_SAMPLES_DEFAULT}, matches link_3d_cells.py\'s '
                         f'N_TILE_SAMPLES).')
parser.add_argument('--input_dir_name', type=str, default='CellPose_DAPI_3D_Bspline',
                    help='Folder under DATASPACE containing link_3d_cells.py output '
                         '(default: CellPose_DAPI_3D_Bspline) — must match whatever '
                         '--output_dir_name link_3d_cells.py was run with.')
args = parser.parse_args()

TARGET_CORE = args.core_name
CH_NAME     = 'DAPI'   # matches link_3d_cells.py — CellPose only ever runs on DAPI

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FOLDER = os.path.join(config.DATASPACE, args.input_dir_name, TARGET_CORE)
QC_DIR       = os.path.join(INPUT_FOLDER, 'qc')
RENDER_DIR   = os.path.join(QC_DIR, 'render_3d_qc')
os.makedirs(RENDER_DIR, exist_ok=True)

label_path = os.path.join(INPUT_FOLDER, f'{TARGET_CORE}_{CH_NAME}_3d_labels.tif')
stats_path = os.path.join(INPUT_FOLDER, f'{TARGET_CORE}_{CH_NAME}_3d_stats.csv')

for path, label in [(label_path, '3D label volume'), (stats_path, '3D stats CSV')]:
    if not os.path.exists(path):
        logger.error(f'{label} not found: {path} — run link_3d_cells.py first.')
        sys.exit(1)

logger.info(f'Loading 3D label volume: {label_path}')
label_vol = tifffile.imread(label_path)   # (Z, H, W), uint32, voxel value = cell_id_3d
logger.info(f'  shape={label_vol.shape}  dtype={label_vol.dtype}')

df_stats = pd.read_csv(stats_path)

# ─────────────────────────────────────────────────────────────────────────────
# SELECT CELLS
# ─────────────────────────────────────────────────────────────────────────────
if args.cell_ids:
    cell_ids = parse_cell_ids(args.cell_ids)
    logger.info(f'Rendering {len(cell_ids)} explicitly requested cell(s): {cell_ids}')
else:
    eligible = df_stats[df_stats['z_span_slices'] >= args.min_confirmed]['cell_id_3d'].values
    if len(eligible) == 0:
        logger.error(f'No cells with z_span_slices >= {args.min_confirmed} found.')
        sys.exit(1)
    rng = np.random.default_rng(seed=0)   # same seed as link_3d_cells.py's tile QC sample
    cell_ids = rng.choice(eligible, size=min(args.n_samples, len(eligible)), replace=False)
    logger.info(f'No --cell_ids given — rendering the same {len(cell_ids)} cell(s) sampled '
               f'for the 2D tile QC montages (seed=0, min_confirmed={args.min_confirmed}).')

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
n_ok, n_skip = 0, 0
for cid3d in cell_ids:
    cid3d = int(cid3d)
    row = df_stats[df_stats['cell_id_3d'] == cid3d]
    if row.empty:
        logger.warning(f'  cell_id_3d={cid3d} not found in stats CSV — skipping.')
        n_skip += 1
        continue

    # Bounding box read directly from the label volume — ground truth voxels,
    # independent of how link_3d_cells.py's in-memory cell_members were built.
    binary = (label_vol == cid3d)
    if not binary.any():
        logger.warning(f'  cell_id_3d={cid3d} has no voxels in the label volume — skipping.')
        n_skip += 1
        continue

    zs, ys, xs = np.where(binary)
    true_z_span = int(zs.max() - zs.min() + 1)
    if true_z_span < 2:
        # marching_cubes needs real extent in every dimension — a cell present
        # on only one slice has no 3D shape to render, regardless of padding.
        logger.warning(f'  cell_id_3d={cid3d} spans only {true_z_span} slice — no 3D '
                       f'shape to render (2D-only / not confirmed in 3D), skipping.')
        n_skip += 1
        continue

    z0, z1 = max(0, zs.min() - PAD_VOXELS), min(binary.shape[0], zs.max() + PAD_VOXELS + 1)
    y0, y1 = max(0, ys.min() - PAD_VOXELS), min(binary.shape[1], ys.max() + PAD_VOXELS + 1)
    x0, x1 = max(0, xs.min() - PAD_VOXELS), min(binary.shape[2], xs.max() + PAD_VOXELS + 1)
    sub = binary[z0:z1, y0:y1, x0:x1]

    try:
        verts, faces, normals, _ = marching_cubes(
            sub.astype(np.float32), level=0.5,
            spacing=(SECTION_THICKNESS_UM, PIXEL_SIZE_XY_UM, PIXEL_SIZE_XY_UM),
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(f'  cell_id_3d={cid3d}: marching_cubes failed ({e}) — skipping.')
        n_skip += 1
        continue

    span    = int(row['z_span_slices'].values[0])
    vol_um3 = float(row['volume_um3'].values[0])

    # verts columns are (z, y, x) in um (spacing order above); map explicitly
    # so the depth axis is z, not whatever Plotly's default would assume.
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='mediumseagreen', opacity=1.0,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=0),
    )])
    fig.update_layout(
        title=f'3D cell {cid3d}  |  span={span} slices  |  vol={vol_um3:.0f} µm³',
        scene=dict(
            xaxis_title='x (µm)', yaxis_title='y (µm)', zaxis_title='z (µm)',
            aspectmode='data',   # true physical proportions, not equal-axis distortion
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path = os.path.join(RENDER_DIR, f'cell_{cid3d:06d}_3d.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    logger.info(f'  cell_id_3d={cid3d}  (span={span}, vol={vol_um3:.0f} µm³) -> {out_path}')
    n_ok += 1

logger.info('=' * 60)
logger.info(f'Done.  Rendered: {n_ok}  Skipped: {n_skip}')
logger.info(f'Output: {RENDER_DIR}/')
logger.info('=' * 60)