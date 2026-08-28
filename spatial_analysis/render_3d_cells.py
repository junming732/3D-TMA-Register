"""
render_3d_cells.py
====================
Render selected 3D reconstructed cells as real, surface rendered
interactive 3D meshes using marching cubes. It reads directly from the 3D
label volume saved by link_3d_cells.py, so there is no need to re-run or
touch link_3d_cells.py itself.

By default, it renders the same set of cells that link_3d_cells.py sampled
for its 2D tile QC montages. It uses the same random seed, the same
min_confirmed filter, and the same sample size, so the two QC views line up
1 to 1 for direct comparison. Use --cell_ids to render specific cells on
demand instead, for example ones a pathologist flags after reviewing the
2D tiles.

Output is one self-contained, interactive HTML file per cell, rotatable and
zoomable in any browser. This is meant to replace the workflow of
downloading a downsampled volume and opening it in FIJI's 3D viewer just to
do a quick per-cell sanity check.

This script requires plotly, which is not otherwise used elsewhere in this
pipeline, and scikit-image, which is already a dependency through other
scripts.

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
from scipy.ndimage import binary_dilation
import plotly.graph_objects as go

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config
from qc_reference import get_or_match_qc_cells

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Physical units — must match link_3d_cells.py exactly, or meshes will have
# the wrong proportions (z is much thicker per-voxel than xy).
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

N_SAMPLES_DEFAULT = 50   # matches link_3d_cells.py's N_TILE_SAMPLES
PAD_VOXELS        = 3    # padding around each cell's bounding box, in voxels (no-neighbors mode)

# Colors cycled through for surrounding/neighbor cells, so each one is visually
# distinct from the main (mediumseagreen) cell and from each other — mirrors
# the per-label coloring pathologists already see in the 2D tile QC montages.
NEIGHBOR_PALETTE = [
    '#6699CC', '#CC99CC', '#CCCC66', '#66CCCC', '#CC9966', '#9999CC',
    '#99CC99', '#CC6699', '#99CCCC', '#CCB266',
]


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


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """'#6699CC' + 0.35 -> 'rgba(102, 153, 204, 0.35)' for per-vertex Mesh3d alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


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
parser.add_argument('--no_neighbors', dest='show_neighbors', action='store_false',
                    help='Disable rendering of surrounding cells for context (they are '
                         'rendered by default, in different colors, so each HTML matches '
                         "the neighboring-cell context visible in the 2D tile QC montages).")
parser.add_argument('--context_pad_voxels', type=int, default=35,
                    help='Padding (in voxels) around the main cell\'s bounding box used to '
                         'find and render surrounding cells when --no_neighbors is not set '
                         '(default: 35, raised from the original 20 so more surrounding '
                         'cells show up). Ignored if neighbors are disabled, in which case '
                         f'the tighter default padding of {PAD_VOXELS} voxels is used instead.')
parser.add_argument('--max_neighbors', type=int, default=25,
                    help='Cap on how many surrounding cells get their own rendered mesh, to '
                         'keep HTML file size / clutter in check (default: 25). Does not '
                         'limit which neighbors count toward touching-surface highlighting.')
parser.add_argument('--no_contact_highlight', dest='contact_highlight', action='store_false',
                    help='Disable touching-surface highlighting (on by default): the patch '
                         'of each neighbor\'s own surface that touches the main cell is '
                         'rendered more opaque than the rest of that neighbor, making the '
                         'contact visible without adding a separate highlight color.')
parser.add_argument('--contact_opacity', type=float, default=0.85,
                    help='Opacity for the touching patch of a neighboring cell\'s surface '
                         '(default: 0.85, vs the base neighbor opacity of 0.35 elsewhere).')
parser.add_argument('--contact_dilation_voxels', type=int, default=1,
                    help='How far (in voxels) to grow each touching zone so it reads as a '
                         'visible patch rather than a single-voxel seam (default: 1).')
parser.add_argument('--set_qc_reference',   action='store_true',
                    help='Make this run\'s sample the shared cross-pipeline QC cell registry '
                         '(overwriting it if one exists), so other pipelines match their QC '
                         'picks to these physical cells. Off by default — every existing '
                         'call site keeps behaving exactly as before unless this is '
                         'explicitly passed.')
parser.add_argument('--qc_pipeline_kind', type=str, default=None, choices=['valis', 'romav2'],
                    help='Enables raw-space coordinate correction for cross-pipeline QC cell '
                         'matching (see qc_reference.py / raw_space_transform.py). Requires '
                         '--qc_transform_dir too. Omit both for the old behavior (direct '
                         'centroid comparison — only valid within one pipeline\'s own runs).')
parser.add_argument('--qc_transform_dir', type=str, default=None,
                    help='Directory containing this pipeline\'s per-z-slice transform .npz '
                         'files. For valis: the point_transforms/ folder produced by '
                         'export_valis_transform.py. For romav2: deformation_maps/, already '
                         'produced by registration — no extra export step needed.')
parser.set_defaults(show_neighbors=True, contact_highlight=True)
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
    # Matches against (or, when --set_qc_reference is passed, writes) the
    # shared cross-pipeline QC reference, so this lines up with
    # link_3d_cells.py's 2D tile QC montages and with other registration
    # pipelines' renders for the same core. See qc_reference.py.
    cell_ids = get_or_match_qc_cells(
        core_name=TARGET_CORE, df_cells=df_stats, min_confirmed=args.min_confirmed,
        n_samples=args.n_samples, pipeline_tag=args.input_dir_name,
        dataspace=config.DATASPACE, seed=0,
        set_reference=args.set_qc_reference, logger=logger,
        pixel_size_um=PIXEL_SIZE_XY_UM, section_thickness_um=SECTION_THICKNESS_UM,
        qc_pipeline_kind=args.qc_pipeline_kind, qc_transform_dir=args.qc_transform_dir,
    )
    if not cell_ids:
        logger.error('QC reference matching returned no cells to render.')
        sys.exit(1)
    logger.info(f'No --cell_ids given — rendering {len(cell_ids)} cell(s) from the shared '
               f'QC reference (same physical cells as the 2D tile QC, and as other '
               f'pipeline variants for this core).')

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

    pad = args.context_pad_voxels if args.show_neighbors else PAD_VOXELS
    z0, z1 = max(0, zs.min() - pad), min(binary.shape[0], zs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(binary.shape[1], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(binary.shape[2], xs.max() + pad + 1)
    sub = binary[z0:z1, y0:y1, x0:x1]
    # Same crop, but keeping the original label IDs (not just this cell's binary
    # mask) — this is how we find which other cells fall within the context window.
    label_sub = label_vol[z0:z1, y0:y1, x0:x1]

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

    # Any other cell_id_3d present in this (wider) crop is a surrounding cell.
    all_neighbor_ids = [n for n in np.unique(label_sub) if n not in (0, cid3d)]
    neighbor_ids = all_neighbor_ids
    if args.show_neighbors and len(neighbor_ids) > args.max_neighbors:
        logger.info(f'    cell_id_3d={cid3d}: {len(neighbor_ids)} neighbors found, '
                   f'capping at {args.max_neighbors}.')
        neighbor_ids = neighbor_ids[:args.max_neighbors]

    # Dilating the main cell's own mask once, up front, is what "touching"
    # means for every neighbor below: a neighbor voxel within
    # contact_dilation_voxels of this is in contact with the main cell.
    main_grown = (binary_dilation(sub, iterations=args.contact_dilation_voxels)
                  if args.contact_highlight else None)

    # Render each surrounding cell in its own color, semi-transparent — except
    # for the patch of it that touches the main cell, which is rendered far
    # more opaque (--contact_opacity) so the contact is visible without
    # recoloring anything. Per-vertex alpha via rgba() strings in vertexcolor.
    neighbor_traces, n_touching_neighbors = [], 0
    if args.show_neighbors:
        for i, nid in enumerate(neighbor_ids):
            n_mask = (label_sub == nid)
            n_zs = np.where(n_mask.any(axis=(1, 2)))[0]
            if len(n_zs) < 2:
                # Neighbor only clips the edge of the crop with no z-extent here —
                # nothing 3D to render for it within this context window.
                continue
            try:
                n_verts, n_faces, _, _ = marching_cubes(
                    n_mask.astype(np.float32), level=0.5,
                    spacing=(SECTION_THICKNESS_UM, PIXEL_SIZE_XY_UM, PIXEL_SIZE_XY_UM),
                )
            except (ValueError, RuntimeError):
                continue
            color = NEIGHBOR_PALETTE[i % len(NEIGHBOR_PALETTE)]

            touching = False
            if args.contact_highlight:
                contact_mask = main_grown & n_mask
                if contact_mask.any():
                    z_idx = np.clip(np.round(n_verts[:, 0] / SECTION_THICKNESS_UM).astype(int),
                                    0, n_mask.shape[0] - 1)
                    y_idx = np.clip(np.round(n_verts[:, 1] / PIXEL_SIZE_XY_UM).astype(int),
                                    0, n_mask.shape[1] - 1)
                    x_idx = np.clip(np.round(n_verts[:, 2] / PIXEL_SIZE_XY_UM).astype(int),
                                    0, n_mask.shape[2] - 1)
                    is_contact = contact_mask[z_idx, y_idx, x_idx]
                    if is_contact.any():
                        touching = True
                        n_touching_neighbors += 1

            mesh_kwargs = dict(
                x=n_verts[:, 2], y=n_verts[:, 1], z=n_verts[:, 0],
                i=n_faces[:, 0], j=n_faces[:, 1], k=n_faces[:, 2],
                name=f'neighbor {int(nid)}', showlegend=True,
                lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1, roughness=0.7),
            )
            if touching:
                vertex_colors = np.where(
                    is_contact,
                    hex_to_rgba(color, args.contact_opacity),
                    hex_to_rgba(color, 0.35),
                )
                mesh_kwargs['vertexcolor'] = vertex_colors
            else:
                mesh_kwargs['color']   = color
                mesh_kwargs['opacity'] = 0.35
            neighbor_traces.append(go.Mesh3d(**mesh_kwargs))
    n_neighbors_rendered = len(neighbor_traces)

    # verts columns are (z, y, x) in um (spacing order above); map explicitly
    # so the depth axis is z, not whatever Plotly's default would assume.
    mesh_traces = [go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='mediumseagreen', opacity=1.0, name=f'cell {cid3d}', showlegend=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=0),
    )] + neighbor_traces

    fig = go.Figure(data=mesh_traces)
    title = f'3D cell {cid3d}  |  span={span} slices  |  vol={vol_um3:.0f} µm³'
    if args.show_neighbors:
        title += f'  |  {n_neighbors_rendered} neighboring cell(s) shown for context'
    if args.contact_highlight and n_touching_neighbors:
        title += f'  |  {n_touching_neighbors} touching (opaque patch)'
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (µm)', yaxis_title='y (µm)', zaxis_title='z (µm)',
            aspectmode='data',   # true physical proportions, not equal-axis distortion
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path = os.path.join(RENDER_DIR, f'cell_{cid3d:06d}_3d.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    neighbor_note = f', +{n_neighbors_rendered} neighbors' if args.show_neighbors else ''
    logger.info(f'  cell_id_3d={cid3d}  (span={span}, vol={vol_um3:.0f} µm³{neighbor_note}) -> {out_path}')
    n_ok += 1

logger.info('=' * 60)
logger.info(f'Done.  Rendered: {n_ok}  Skipped: {n_skip}')
logger.info(f'Output: {RENDER_DIR}/')
logger.info('=' * 60)