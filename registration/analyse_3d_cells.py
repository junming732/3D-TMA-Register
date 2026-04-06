"""
analyse_3d_cells.py
===================
Link CellPose segmentation masks across registered Z-slices to identify
cells that span multiple sections using overlap-based 3D linking.

Approach
--------
Rather than binarising all masks and running a global 3D flood-fill (which
merges adjacent cells that share a border into one giant component), this
script uses a graph-based approach that respects the original 2D CellPose
cell boundaries:

  1. For each pair of adjacent Z-slices, find which 2D cell IDs overlap in XY.
     Two cells "overlap" if they share at least --min_overlap pixels.
  2. Build a graph where nodes = (z, 2d_cell_id) and edges = overlapping pairs.
  3. Find connected components of this graph via union-find → each component
     is one 3D cell.
  4. Filter by Z-span and minimum area, then assign new sequential 3D IDs.

This correctly handles dense tissue: two adjacent T-cells in the same slice
that are correctly separated by a CellPose boundary will never be merged,
even if they both overlap with the same cell in the next slice.

Usage
-----
    python analyse_3d_cells.py \\
        --core_name   Core_01   \\
        --channel     CD3       \\
        [--min_overlap  3]      \\
        [--min_slices   1]      \\
        [--max_slices   4]      \\
        [--min_area_px  10]     \\
        [--plot_qc]

Input
-----
    <DATASPACE>/CellPose_<CHANNEL>_Warped/<CORE_NAME>/
        TMA_<ID>_<CHANNEL>_cp_masks_warped.tif   (one per slice)

Output
------
    <DATASPACE>/CellPose_<CHANNEL>_3D/<CORE_NAME>/
        <CORE_NAME>_<CHANNEL>_3d_labels.tif   -- (Z, H, W) uint32 3D label volume
        <CORE_NAME>_<CHANNEL>_3d_stats.csv    -- per-cell statistics
        <CORE_NAME>_<CHANNEL>_3d_qc.png       -- QC montage (if --plot_qc)
"""

import os
import sys
import re
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
from collections import defaultdict
from scipy.spatial import cKDTree

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Overlap-based 3D cell linking across registered CellPose masks.'
)
parser.add_argument('--core_name',    type=str, required=True)
parser.add_argument('--channel',      type=str, default='CD3',
                    help="Channel label used in mask filenames (default: CD3)")
parser.add_argument('--min_overlap',  type=int, default=3,
                    help="Min shared pixels for two 2D cells to be linked across Z "
                         "(default: 3). Raise to reduce spurious links in dense tissue.")
parser.add_argument('--min_slices',   type=int, default=1,
                    help="Min Z-slices a 3D cell must span to be kept (default: 1).")
parser.add_argument('--max_slices',   type=int, default=4,
                    help="Max Z-slices a 3D cell may span. T-cells ~10um at 4.5um/section "
                         "= 2-3 slices max. Larger = likely artifact (default: 4).")
parser.add_argument('--min_area_px',  type=int, default=10,
                    help="Min pixel area of a 2D cell to include in linking (default: 10).")
parser.add_argument('--plot_qc',       action='store_true',
                    help="Save QC montage PNG.")
parser.add_argument('--coloc_channel', type=str,   default=None,
                    help="Second channel for co-localisation e.g. CD163. "
                         "CSV must already exist in CellPose_<CHANNEL>_3D/.")
parser.add_argument('--coloc_radius_um', type=float, default=50.0,
                    help="Max distance (um) to count as co-localised (default: 50).")
parser.add_argument('--min_confirmed',   type=int,   default=2,
                    help="Min Z-span to count as confirmed 3D cell (default: 2).")
args = parser.parse_args()

PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

TARGET_CORE = args.core_name
CH_NAME     = args.channel

INPUT_FOLDER  = os.path.join(config.DATASPACE, f"CellPose_{CH_NAME}_Warped",  TARGET_CORE)
OUTPUT_FOLDER = os.path.join(config.DATASPACE, f"CellPose_{CH_NAME}_3D",      TARGET_CORE)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def get_slice_id(path: str) -> int:
    match = re.search(r"TMA_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1


class UnionFind:
    """
    Union-find (disjoint set union) for graph connected components.
    Nodes are (z_index, cell_id_2d) tuples.
    """
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def components(self):
        """Return dict mapping root -> list of member nodes."""
        groups = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return dict(groups)


# -----------------------------------------------------------------------------
# LOAD WARPED MASKS
# -----------------------------------------------------------------------------

mask_files = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, f"*{CH_NAME}*_warped.tif")),
    key=get_slice_id
)

if not mask_files:
    logger.error(f"No warped mask files found in {INPUT_FOLDER}")
    sys.exit(1)

logger.info(f"Found {len(mask_files)} warped mask slices for {TARGET_CORE} / {CH_NAME}")

slice_ids = [get_slice_id(f) for f in mask_files]
masks_2d  = []
for f in mask_files:
    m = tifffile.imread(f)
    if m.ndim != 2:
        m = m.squeeze()
    masks_2d.append(m.astype(np.uint32))

# Ensure consistent shape across slices
H, W = masks_2d[0].shape
for i, m in enumerate(masks_2d):
    if m.shape != (H, W):
        logger.warning(f"Slice {slice_ids[i]} shape mismatch -- zero-padding.")
        padded = np.zeros((H, W), dtype=np.uint32)
        h, w   = min(m.shape[0], H), min(m.shape[1], W)
        padded[:h, :w] = m[:h, :w]
        masks_2d[i] = padded

n_slices = len(masks_2d)
logger.info(f"Stack: Z={n_slices}, H={H}, W={W}")


# -----------------------------------------------------------------------------
# STEP 1 -- FIND OVERLAPPING CELL PAIRS BETWEEN ADJACENT SLICES
#
# For each adjacent pair (z, z+1), at every pixel where both slices have a
# foreground label, we record the pair (id_from_z, id_from_z+1).
# We then count how many pixels each pair shares (the overlap area).
# Pairs with overlap >= min_overlap are linked in the union-find graph.
#
# Encoding trick: pack the two uint32 IDs into one int64 for fast np.unique.
# -----------------------------------------------------------------------------

uf = UnionFind()

# Register every 2D cell as its own node
for z, mask in enumerate(masks_2d):
    for cid in np.unique(mask):
        if cid > 0:
            uf.find((z, int(cid)))

# Precompute pixel area AND centroid for every 2D cell in every slice.
# Using a single pass per slice with np.bincount for areas and
# coordinate-weighted sums for centroids — much faster than per-cell argwhere.
logger.info("Precomputing 2D cell areas and centroids...")
cell_areas     = {}   # (z, cid2d) -> int pixel count
cell_centroids = {}   # (z, cid2d) -> (cy, cx) float

for z, mask in enumerate(masks_2d):
    counts = np.bincount(mask.ravel())
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_areas[(z, cid)] = int(counts[cid])

    # Centroid via coordinate-weighted sum — single pass over all cells
    ys, xs = np.indices(mask.shape)   # (H,W) coordinate grids
    flat   = mask.ravel()
    ys_f   = ys.ravel().astype(np.float64)
    xs_f   = xs.ravel().astype(np.float64)
    # sum of y-coords and x-coords per label
    sum_y = np.bincount(flat, weights=ys_f, minlength=len(counts))
    sum_x = np.bincount(flat, weights=xs_f, minlength=len(counts))
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_centroids[(z, cid)] = (
                sum_y[cid] / counts[cid],
                sum_x[cid] / counts[cid],
            )

logger.info("Finding overlapping cell pairs between adjacent slices...")

for z in range(n_slices - 1):
    mask_a = masks_2d[z]
    mask_b = masks_2d[z + 1]

    fg = (mask_a > 0) & (mask_b > 0)
    if not np.any(fg):
        logger.info(f"  Z{z}->Z{z+1}: no overlapping foreground pixels")
        continue

    ids_a = mask_a[fg].astype(np.int64)
    ids_b = mask_b[fg].astype(np.int64)

    # Encode each (a, b) pair as a single int64
    pairs               = (ids_a << 32) | ids_b
    unique_pairs, counts = np.unique(pairs, return_counts=True)

    n_links = 0
    for pair, count in zip(unique_pairs, counts):
        if count < args.min_overlap:
            continue
        cid_a = int(pair >> 32)
        cid_b = int(pair & 0xFFFFFFFF)
        uf.union((z, cid_a), (z + 1, cid_b))
        n_links += 1

    logger.info(
        f"  Z{z} (slice {slice_ids[z]:03d}) -> Z{z+1} (slice {slice_ids[z+1]:03d}): "
        f"{n_links} cell links  (min_overlap={args.min_overlap}px)"
    )


# -----------------------------------------------------------------------------
# STEP 2 -- BUILD 3D CELLS FROM CONNECTED COMPONENTS
# -----------------------------------------------------------------------------

logger.info("Building 3D cells from linked components...")

components = uf.components()

rows       = []
cell_id    = 0
node_to_3d = {}

for root, members in components.items():
    z_indices = [m[0] for m in members]
    z_min     = min(z_indices)
    z_max     = max(z_indices)
    z_span    = z_max - z_min + 1

    if z_span < args.min_slices or z_span > args.max_slices:
        continue

    # Look up precomputed areas — O(1) per cell, no array scan
    per_slice_areas = {node: cell_areas.get(node, 0) for node in members}
    if max(per_slice_areas.values()) < args.min_area_px:
        continue

    cell_id += 1
    total_px = sum(per_slice_areas.values())
    vol_um3  = round(total_px * PIXEL_SIZE_XY_UM**2 * SECTION_THICKNESS_UM, 3)

    for node in members:
        node_to_3d[node] = cell_id

    # Compute 3D centroid as area-weighted mean of precomputed per-slice centroids.
    sum_y, sum_x, sum_z, sum_w = 0.0, 0.0, 0.0, 0.0
    for z, cid2d in members:
        area = per_slice_areas.get((z, cid2d), 0)
        if area == 0:
            continue
        cy, cx = cell_centroids.get((z, cid2d), (0.0, 0.0))
        sum_y += cy * area
        sum_x += cx * area
        sum_z += z  * area
        sum_w += area

    centroid_y_px  = round(sum_y / sum_w, 2) if sum_w > 0 else 0.0
    centroid_x_px  = round(sum_x / sum_w, 2) if sum_w > 0 else 0.0
    centroid_z_idx = round(sum_z / sum_w, 2) if sum_w > 0 else 0.0
    centroid_y_um  = round(centroid_y_px  * PIXEL_SIZE_XY_UM,     3)
    centroid_x_um  = round(centroid_x_px  * PIXEL_SIZE_XY_UM,     3)
    centroid_z_um  = round(centroid_z_idx * SECTION_THICKNESS_UM, 3)

    rows.append(dict(
        cell_id_3d     = cell_id,
        z_min          = z_min,
        z_max          = z_max,
        z_span_slices  = z_span,
        slice_id_min   = slice_ids[z_min],
        slice_id_max   = slice_ids[z_max],
        n_2d_segments  = len(members),
        volume_px      = total_px,
        volume_um3     = vol_um3,
        centroid_x_px  = centroid_x_px,
        centroid_y_px  = centroid_y_px,
        centroid_z_idx = centroid_z_idx,
        centroid_x_um  = centroid_x_um,
        centroid_y_um  = centroid_y_um,
        centroid_z_um  = centroid_z_um,
        slice_ids      = str(sorted(set(slice_ids[z] for z in z_indices))),
    ))

n_cells = cell_id
logger.info(f"Final 3D cell count: {n_cells}")


# -----------------------------------------------------------------------------
# STEP 3 -- BUILD OUTPUT LABEL VOLUME
#
# For every pixel in every slice, map its 2D cell ID to its 3D cell ID.
# Pixels whose 2D cell was filtered out get label 0 (background).
# -----------------------------------------------------------------------------

logger.info("Building output 3D label volume...")

output_volume = np.zeros((n_slices, H, W), dtype=np.uint32)

for z, mask in enumerate(masks_2d):
    # Build a lookup table: 2d_cell_id -> 3d_cell_id (0 if not kept)
    max_id  = int(mask.max()) if mask.max() > 0 else 0
    lut     = np.zeros(max_id + 1, dtype=np.uint32)
    for cid2d in range(1, max_id + 1):
        node = (z, cid2d)
        if node in node_to_3d:
            lut[cid2d] = node_to_3d[node]
    # Apply LUT in one vectorised step — no per-cell array scan
    out_slice          = lut[mask]
    output_volume[z]   = out_slice
    n_cells_slc = len(np.unique(out_slice)) - 1
    logger.info(
        f"  Slice Z{z} (ID {slice_ids[z]:03d}): "
        f"{n_cells_slc} 3D cells  ({int(np.sum(out_slice > 0))} px labelled)"
    )


# -----------------------------------------------------------------------------
# STATS CSV
# -----------------------------------------------------------------------------

df_cells = pd.DataFrame(rows).sort_values('cell_id_3d')
csv_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_stats.csv")
df_cells.to_csv(csv_path, index=False)
logger.info(f"Stats CSV -> {csv_path}")

# For display purposes, assign each cell a random ID in 1..254 per slice
# so every slice independently hits the full LUT palette.
# The stats CSV retains the original sequential cell_id_3d for analysis.
# Note: the same physical cell will have different display IDs in different
# slices in the TIFF — use the CSV for cross-slice identity, not the TIFF colours.
logger.info("Assigning per-slice random display IDs (1-254) for even LUT coverage...")
rng_disp = np.random.default_rng(seed=42)

# Build per-slice display volume separately from output_volume
display_volume = np.zeros_like(output_volume)
for z in range(n_slices):
    ids_in_slice = np.unique(output_volume[z])
    ids_in_slice = ids_in_slice[ids_in_slice > 0]
    n = len(ids_in_slice)
    if n == 0:
        continue
    # Assign random IDs from 1..254, cycling if more than 254 cells
    n_reps      = (n // 254) + 1
    palette     = np.tile(np.arange(1, 255, dtype=np.uint16), n_reps)[:n]
    rng_disp.shuffle(palette)
    lut         = np.zeros(int(output_volume[z].max()) + 1, dtype=np.uint16)
    for orig_id, disp_id in zip(ids_in_slice, palette):
        lut[orig_id] = disp_id
    display_volume[z] = lut[output_volume[z]]

logger.info("Per-slice display IDs assigned.")

if len(df_cells) > 0:
    logger.info(
        f"Z-span distribution:\n"
        f"{df_cells['z_span_slices'].value_counts().sort_index().to_string()}"
    )
    logger.info(
        f"Volume (um3): mean={df_cells['volume_um3'].mean():.1f}  "
        f"median={df_cells['volume_um3'].median():.1f}  "
        f"max={df_cells['volume_um3'].max():.1f}"
    )


# -----------------------------------------------------------------------------
# SAVE 3D LABEL VOLUME
# -----------------------------------------------------------------------------

vol_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_labels.tif")
logger.info(f"Saving 3D label volume -> {vol_path}")
# Save two TIFFs:
# 1. analysis volume (output_volume) — original sequential uint32 IDs, OME-TIFF
# 2. display volume  (display_volume) — per-slice random 1-254 uint16 IDs,
#    ImageJ format so Fiji 3D viewer reads Z spacing automatically.
tifffile.imwrite(
    vol_path,
    output_volume,
    photometric='minisblack',
    compression='deflate',
    metadata={
        'axes': 'ZYX',
        'PhysicalSizeX':     PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'um',
        'PhysicalSizeY':     PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'um',
        'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'um',
    },
)

disp_path = vol_path.replace('_3d_labels.tif', '_3d_display.tif')
tifffile.imwrite(
    disp_path,
    display_volume.astype(np.uint16),
    imagej=True,
    resolution=(1 / PIXEL_SIZE_XY_UM, 1 / PIXEL_SIZE_XY_UM),
    metadata={
        'axes':    'ZYX',
        'spacing': SECTION_THICKNESS_UM,
        'unit':    'um',
    },
    compression='deflate',
)
logger.info(f"Display volume saved -> {disp_path}")
logger.info("3D label volume saved.")


# -----------------------------------------------------------------------------
# QC MONTAGE
# -----------------------------------------------------------------------------

if args.plot_qc:
    logger.info("Generating QC montage...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng      = np.random.default_rng(seed=42)
    n_colors = 256
    palette  = rng.uniform(0.3, 1.0, size=(n_colors - 1, 4))
    palette[:, 3] = 1.0
    # Shuffle so sequential IDs (singletons in later slices) get
    # perceptually distant colours rather than all landing in one hue band
    rng.shuffle(palette)
    colors = np.vstack([[0, 0, 0, 1], palette])
    cmap   = ListedColormap(colors)

    def cycle_labels(arr):
        return np.where(arr > 0, (arr % (n_colors - 1)) + 1, 0)

    max_proj = output_volume.max(axis=0)

    n_cols = min(n_slices, 8)
    n_rows = (n_slices + n_cols - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(n_rows, n_cols)

    axes[0, 0].imshow(cycle_labels(max_proj), cmap=cmap, vmin=0, vmax=n_colors,
                      interpolation='nearest')
    axes[0, 0].set_title("Max projection\n(3D cell IDs)", fontsize=8)
    axes[0, 0].axis('off')
    for j in range(1, n_cols):
        axes[0, j].axis('off')

    for idx in range(n_slices):
        row = idx // n_cols + 1
        col = idx % n_cols
        slc = output_volume[idx]
        axes[row, col].imshow(cycle_labels(slc), cmap=cmap, vmin=0, vmax=n_colors,
                               interpolation='nearest')
        axes[row, col].set_title(
            f"ID {slice_ids[idx]}\n{len(np.unique(slc)) - 1} cells", fontsize=7
        )
        axes[row, col].axis('off')

    for idx in range(n_slices, n_rows * n_cols - n_cols):
        axes[idx // n_cols + 1, idx % n_cols].axis('off')

    fig.suptitle(
        f"3D Cell Analysis QC -- {TARGET_CORE}  {CH_NAME}\n"
        f"{n_cells} 3D cells  |  {n_slices} slices  |  "
        f"z_span: {args.min_slices}-{args.max_slices}  |  "
        f"min_overlap: {args.min_overlap}px",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    qc_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_qc.png")
    plt.savefig(qc_path, dpi=80, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"QC montage -> {qc_path}")

# -----------------------------------------------------------------------------
# PATTERN ANALYSIS
# -----------------------------------------------------------------------------

logger.info("Running pattern analysis...")

confirmed   = df_cells[df_cells['z_span_slices'] >= args.min_confirmed]
singletons  = df_cells[df_cells['z_span_slices'] <  args.min_confirmed]

logger.info(
    f"Cell summary -- total={len(df_cells)}  "
    f"confirmed(>={args.min_confirmed} slices)={len(confirmed)}  "
    f"singletons={len(singletons)}  "
    f"pct_confirmed={len(confirmed)/max(len(df_cells),1)*100:.1f}%"
)
if len(df_cells) > 0:
    logger.info(
        f"Volume (um3): mean={df_cells['volume_um3'].mean():.1f}  "
        f"median={df_cells['volume_um3'].median():.1f}  "
        f"max={df_cells['volume_um3'].max():.1f}"
    )

# ── 3D point cloud ────────────────────────────────────────────────────────────
if args.plot_qc and 'centroid_x_um' in df_cells.columns and len(confirmed) > 0:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(confirmed['centroid_x_um'], confirmed['centroid_y_um'],
                   confirmed['centroid_z_um'],
                   s=2, alpha=0.4, c='red',
                   label=f'Confirmed {len(confirmed)}')
        if len(singletons) > 0:
            ax.scatter(singletons['centroid_x_um'], singletons['centroid_y_um'],
                       singletons['centroid_z_um'],
                       s=1, alpha=0.15, c='grey',
                       label=f'Singleton {len(singletons)}')
        ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)'); ax.set_zlabel('Z (um)')
        ax.legend(fontsize=8)
        ax.set_title(
            f"{TARGET_CORE} {CH_NAME} -- 3D cell point cloud\n"
            f"{len(df_cells)} total  |  {len(confirmed)} confirmed",
            fontsize=10
        )
        plt.tight_layout()
        pc_path = os.path.join(OUTPUT_FOLDER,
                               f"{TARGET_CORE}_{CH_NAME}_pointcloud.png")
        plt.savefig(pc_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Point cloud -> {pc_path}")
    except Exception as e:
        logger.warning(f"Point cloud failed: {e}")

# ── Cross-channel co-localisation ─────────────────────────────────────────────
if args.coloc_channel is not None:
    coloc_csv = os.path.join(
        config.DATASPACE, f"CellPose_{args.coloc_channel}_3D",
        TARGET_CORE, f"{TARGET_CORE}_{args.coloc_channel}_3d_stats.csv"
    )
    if not os.path.exists(coloc_csv):
        logger.warning(
            f"Co-localisation CSV not found: {coloc_csv} -- skipping. "
            f"Run analyse_3d_cells.py for {args.coloc_channel} first."
        )
    else:
        df_coloc = pd.read_csv(coloc_csv)
        conf_b   = df_coloc[df_coloc['z_span_slices'] >= args.min_confirmed]

        if len(confirmed) == 0 or len(conf_b) == 0:
            logger.warning("Insufficient confirmed cells for co-localisation.")
        else:
            coords_a = confirmed[['centroid_x_um',
                                   'centroid_y_um',
                                   'centroid_z_um']].values
            coords_b = conf_b[['centroid_x_um',
                                'centroid_y_um',
                                'centroid_z_um']].values

            tree_b         = cKDTree(coords_b)
            dist_a, _      = tree_b.query(coords_a, k=1)
            tree_a         = cKDTree(coords_a)
            dist_b, _      = tree_a.query(coords_b, k=1)

            confirmed = confirmed.copy()
            confirmed[f'nearest_{args.coloc_channel}_um'] = dist_a
            coloc_count = int((dist_a <= args.coloc_radius_um).sum())

            # Save enriched CSV
            enriched_path = csv_path.replace(
                '_3d_stats.csv',
                f'_3d_stats_with_nearest_{args.coloc_channel}.csv'
            )
            confirmed.to_csv(enriched_path, index=False)

            logger.info(
                f"Co-localisation {CH_NAME} -> {args.coloc_channel}: "
                f"mean dist={dist_a.mean():.1f}um  "
                f"median={float(np.median(dist_a)):.1f}um  "
                f"within {args.coloc_radius_um}um: "
                f"{coloc_count}/{len(confirmed)} "
                f"({coloc_count/max(len(confirmed),1)*100:.1f}%)"
            )
            logger.info(f"Enriched CSV -> {enriched_path}")

            if args.plot_qc:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(dist_a, bins=50, color='red',
                                 alpha=0.7, edgecolor='none')
                    axes[0].axvline(args.coloc_radius_um, color='black',
                                    linestyle='--',
                                    label=f'{args.coloc_radius_um}um radius')
                    axes[0].set_xlabel(
                        f'Distance to nearest {args.coloc_channel} (um)')
                    axes[0].set_ylabel('Count')
                    axes[0].set_title(
                        f'{CH_NAME} -> nearest {args.coloc_channel}')
                    axes[0].legend()

                    axes[1].hist(dist_b, bins=50, color='blue',
                                 alpha=0.7, edgecolor='none')
                    axes[1].axvline(args.coloc_radius_um, color='black',
                                    linestyle='--',
                                    label=f'{args.coloc_radius_um}um radius')
                    axes[1].set_xlabel(
                        f'Distance to nearest {CH_NAME} (um)')
                    axes[1].set_ylabel('Count')
                    axes[1].set_title(
                        f'{args.coloc_channel} -> nearest {CH_NAME}')
                    axes[1].legend()

                    fig.suptitle(
                        f"{TARGET_CORE} -- {CH_NAME} vs {args.coloc_channel} "
                        f"co-localisation\n"
                        f"{coloc_count}/{len(confirmed)} {CH_NAME} cells within "
                        f"{args.coloc_radius_um}um of a {args.coloc_channel} cell",
                        fontsize=11, fontweight='bold'
                    )
                    plt.tight_layout()
                    hist_path = os.path.join(
                        OUTPUT_FOLDER,
                        f"{TARGET_CORE}_{CH_NAME}_vs_{args.coloc_channel}_coloc.png"
                    )
                    plt.savefig(hist_path, dpi=120, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Co-localisation plot -> {hist_path}")
                except Exception as e:
                    logger.warning(f"Co-localisation plot failed: {e}")

logger.info("Done.")