"""
link_3d_cells.py
===================
Link CellPose segmentation masks across registered Z-slices to identify
cells that span multiple sections using overlap-based 3D linking.

Approach
--------
1. Graph Construction: For adjacent Z-slices, find overlapping 2D cell IDs.
   Record the connection and the "weight" (number of shared pixels).
2. Topological Severing: Build connected components. If a component exceeds
   the maximum allowable Z-span, recursively find and sever the weakest connection
   (minimum shared pixels) to separate vertically stacked cells.
3. Feature Extraction: Compute 3D volume, span, and area-weighted centroids.
4. Output Generation:
   - 32-bit analysis volume (true sequential IDs).
   - 16-bit randomized display volume (for Fiji/Glasbey LUT compatibility).
   - QC plots (when --plot_qc is set):
       overlap_fraction_distribution.png  — link quality
       z_span_distribution.png            — biological plausibility
       centroid_drift.png                 — per-link XY drift
       area_consistency.png               — per-link area ratio
       visual_tile_qc/cell_XXXXXX.png    — per-cell image patch montages

Usage
-----
    python link_3d_cells.py \
        --core_name   Core_01   \
        [--min_overlap      30] \
        [--max_slices        5] \
        [--plot_qc            ] \
        [--n_tile_samples   50] \
        [--patch_size_px    80] \
        [--input_dir_name    'CellPose_DAPI_Warped_Bspline'] \
        [--output_dir_name   'CellPose_DAPI_3D_Bspline']     \
        [--denoised_dir_name 'Denoised_bspline']              \
        [--coloc_channel     CD163] [--coloc_dir_name 'CellPose_CD163_3D_Bspline']

    CellPose (and therefore this script) only ever runs on DAPI, so channel
    naming is fixed rather than a CLI flag. --coloc_channel is the one place
    a second channel name is still relevant — for comparing DAPI-linked 3D
    cells against another channel's already-computed 3D stats.

    The *_dir_name flags make this script independent of which registration
    algorithm produced its inputs.
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
from qc_reference import get_or_match_qc_cells

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Overlap-based 3D cell linking across registered CellPose masks.'
)
parser.add_argument('--core_name',          type=str,   required=True)
parser.add_argument('--min_overlap',        type=int,   default=10,
                    help='Min shared pixels for two 2D cells to be linked across Z.')
parser.add_argument('--min_slices',         type=int,   default=1,
                    help='Min Z-slices a 3D cell must span to be kept (default: 1).')
parser.add_argument('--max_slices',         type=int,   default=5,
                    help='Max Z-slices a 3D cell may span (default: 5).')
parser.add_argument('--min_area_px',        type=int,   default=10,
                    help='Min pixel area of a 2D cell to include in linking (default: 10).')
parser.add_argument('--min_confirmed',      type=int,   default=2,
                    help='Min Z-span to count as a confirmed multi-slice cell (default: 2).')
parser.add_argument('--coloc_channel',      type=str,   default=None,
                    help='Second channel for co-localisation e.g. CD163.')
parser.add_argument('--coloc_radius_um',    type=float, default=50.0,
                    help='Max distance (um) to count as co-localised (default: 50).')
parser.add_argument('--min_iou',            type=float, default=0.0,
                    help='Min IoU to accept a link between adjacent slices (default: 0.0).')
parser.add_argument('--min_intensity_frac', type=float, default=0.0,
                    help='Min mean DAPI intensity as fraction of per-cell peak to accept a link. '
                         'Requires denoised OME-TIFF. (default: 0.0 = off)')
parser.add_argument('--plot_qc',            action='store_true',
                    help='Save all QC plots and visual tile montages.')
parser.add_argument('--input_dir_name',     type=str, default='CellPose_DAPI_Warped_Bspline',
                    help='Folder under DATASPACE containing warped DAPI masks '
                         '(default: CellPose_DAPI_Warped_Bspline). Independent of which '
                         'registration algorithm produced the warp.')
parser.add_argument('--output_dir_name',    type=str, default='CellPose_DAPI_3D_Bspline',
                    help='Folder under DATASPACE to write 3D linkage output into '
                         '(default: CellPose_DAPI_3D_Bspline).')
parser.add_argument('--denoised_dir_name',  type=str, default='Denoised_bspline',
                    help='Folder under DATASPACE containing <CORE>_denoised.ome.tif, '
                         'used for intensity-gated linking and QC tile montages '
                         '(default: Denoised_bspline).')
parser.add_argument('--coloc_dir_name',     type=str, default=None,
                    help='Folder under DATASPACE containing --coloc_channel\'s 3D stats CSV. '
                         'If omitted, defaults to CellPose_{coloc_channel}_3D_Bspline — this '
                         'assumes the co-localisation channel was linked with the same '
                         'registration variant as this (DAPI) run.')
parser.add_argument('--set_qc_reference',   action='store_true',
                    help='Make this run\'s tile-QC sample the shared cross-pipeline QC cell '
                         'registry (overwriting it if one exists), so other pipelines match '
                         'their QC picks to these physical cells. Off by default — every '
                         'existing call site keeps behaving exactly as before unless this is '
                         'explicitly passed.')
parser.add_argument('--qc_pipeline_kind', type=str, default=None, choices=['valis', 'romav2'],
                    help='Enables raw-space coordinate correction for cross-pipeline QC cell '
                         'matching (see qc_reference.py / raw_space_transform.py) — converts '
                         'centroids to the shared pre-registration image frame before '
                         'comparing them, instead of comparing this pipeline\'s own registered '
                         'coordinates directly. Requires --qc_transform_dir too. Omit both for '
                         'the old behavior (direct centroid comparison — only valid for QC '
                         'within one pipeline\'s own runs, not across different pipelines).')
parser.add_argument('--qc_transform_dir', type=str, default=None,
                    help='Directory containing this pipeline\'s per-z-slice transform .npz '
                         'files. For valis: the point_transforms/ folder produced by '
                         'export_valis_transform.py. For romav2: deformation_maps/, already '
                         'produced by registration — no extra export step needed.')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# INTERNAL CONSTANTS — not arguments because they are not tunable per-channel;
# they are either biological facts or fixed QC display settings.
# -----------------------------------------------------------------------------
# A clean single-nucleus chain must contribute exactly 1 mask per Z-slice.
# More than 1 means CellPose over-segmented that section → always reject.
MAX_SEGMENTS_PER_Z = 1

# QC display constants — thresholds used only for flagging/plotting, not filtering.
FLAG_DRIFT_PX   = 30.0   # centroid drift above which a link is flagged in QC
FLAG_AREA_RATIO = 3.0    # area ratio above which a link is flagged in QC
N_TILE_SAMPLES  = 50     # number of 3D cells rendered as tile montages
PATCH_SIZE_PX   = 80     # half-size of patch around each centroid in tiles

PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

TARGET_CORE = args.core_name
CH_NAME     = 'DAPI'   # CellPose is only ever run on DAPI in this pipeline

INPUT_FOLDER  = os.path.join(config.DATASPACE, args.input_dir_name,  TARGET_CORE)
OUTPUT_FOLDER = os.path.join(config.DATASPACE, args.output_dir_name, TARGET_CORE)

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

def build_3d_components(all_nodes, edges_uvw, max_slices):
    """
    Link 2D nuclei across Z-slices into 3D cells using a scipy sparse graph.
    Nodes are (z, cell_id_2d) tuples encoded as contiguous integers for scipy.
    Connected components are found via scipy.sparse.csgraph (C-backed) — orders
    of magnitude faster than pure-Python BFS for graphs with hundreds of
    thousands of edges.  Components exceeding max_slices are pruned by
    iteratively severing the weakest edge using numpy sub-matrix operations,
    with connected components re-checked via scipy after each cut.

    Performance notes
    -----------------
    * eliminate_zeros() is deferred and called in batches every
      ELIM_BATCH cuts rather than on every single cut, avoiding
      an O(nnz) scan of the full 500k-node matrix each iteration.
    * Sub-matrix extraction uses a single fancy-index slice
      (mat[comp_idx[:, None], comp_idx]) which produces a CSR view
      in one step, avoiding the intermediate COO allocation of the
      previous double-slice pattern.
    * Per-component z_vals are computed with a vectorised numpy
      take rather than a Python list comprehension.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as sp_cc

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n = len(all_nodes)
    if not edges_uvw:
        return [{node} for node in all_nodes]

    us = [node_to_idx[u] for u, v, w in edges_uvw]
    vs = [node_to_idx[v] for u, v, w in edges_uvw]
    ws = [w               for u, v, w in edges_uvw]

    # Use lil_matrix for efficient in-place edge removal, then convert to
    # CSR for fast sp_cc calls.  The toggle between formats is cheap
    # relative to the cost of repeated eliminate_zeros on a huge CSR.
    mat = sp.csr_matrix(
        (ws + ws, (us + vs, vs + us)),
        shape=(n, n), dtype=np.int32,
    )

    # Pre-extract node Z-indices as a numpy array for fast vectorised lookup
    node_z = np.array([node[0] for node in all_nodes], dtype=np.int32)

    n_comp, labels = sp_cc(mat, directed=False)
    logger.info(f"  Initial connected components: {n_comp}")

    valid_components = []
    pending = []
    for comp_id in range(n_comp):
        pending.append(np.where(labels == comp_id)[0])

    severed_total = 0

    # NOTE: each top-level connected component is disjoint from every other,
    # so once we slice it out of the global matrix we never need to touch
    # the global matrix again for that component. All cutting/splitting work
    # below happens on the small LOCAL sub-matrix only. This avoids both:
    #   (a) the old `mat[comp_idx[:, None], comp_idx]` broadcast-index bug
    #       (caused ValueError / ArrayMemoryError on large components), and
    #   (b) repeatedly re-slicing the huge global matrix on every single
    #       cut, which is what caused the script to hang/stall.
    while pending:
        comp_idx = pending.pop()

        z_vals = node_z[comp_idx]
        z_span = int(z_vals.max() - z_vals.min() + 1)

        if z_span <= max_slices:
            valid_components.append({all_nodes[i] for i in comp_idx})
            continue

        # Extract the local subgraph ONCE (row-slice then column-slice,
        # never a dense comp_idx x comp_idx broadcast).
        sub = mat[comp_idx][:, comp_idx].tocsr()

        local_pending = [(comp_idx, sub)]

        while local_pending:
            c_idx, local_mat = local_pending.pop()

            z_vals = node_z[c_idx]
            z_span = int(z_vals.max() - z_vals.min() + 1)
            if z_span <= max_slices:
                valid_components.append({all_nodes[i] for i in c_idx})
                continue

            coo = local_mat.tocoo()
            upper = coo.row < coo.col
            if coo.nnz == 0 or not np.any(upper):
                valid_components.append({all_nodes[i] for i in c_idx})
                continue

            # 1. Minimum overlap weight in this (local) component
            min_weight = coo.data[upper].min()

            # 2. All edges sharing this minimum weight
            cut_mask = coo.data[upper] == min_weight
            lu = coo.row[upper][cut_mask]
            lv = coo.col[upper][cut_mask]

            severed_total += len(lu)

            # 3. Zero out the cut edges in the LOCAL matrix only (small, fast)
            local_mat = local_mat.tolil()
            local_mat[lu, lv] = 0
            local_mat[lv, lu] = 0
            local_mat = local_mat.tocsr()
            local_mat.eliminate_zeros()

            # 4. Re-split locally and push the pieces back onto local_pending
            n_sub, sub_labels = sp_cc(local_mat, directed=False)
            for sub_id in range(n_sub):
                mask = sub_labels == sub_id
                local_pending.append((c_idx[mask], local_mat[mask][:, mask]))

    logger.info(f"  Edges severed during Z-span pruning: {severed_total}")
    logger.info(f"  Final component count: {len(valid_components)}")
    return valid_components


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
# STEP 1 -- BUILD OVERLAP GRAPH
# -----------------------------------------------------------------------------

all_nodes      = []   # ordered list of (z, cid) — index = scipy node id
edges_uvw      = []   # list of (node_a, node_b, weight)

logger.info("Precomputing 2D cell areas and centroids...")
cell_areas     = {}
cell_centroids = {}

for z, mask in enumerate(masks_2d):
    counts = np.bincount(mask.ravel())
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_areas[(z, cid)] = int(counts[cid])
            all_nodes.append((z, int(cid)))

    ys, xs = np.indices(mask.shape)
    flat   = mask.ravel()
    ys_f   = ys.ravel().astype(np.float64)
    xs_f   = xs.ravel().astype(np.float64)
    sum_y  = np.bincount(flat, weights=ys_f, minlength=len(counts))
    sum_x  = np.bincount(flat, weights=xs_f, minlength=len(counts))
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_centroids[(z, cid)] = (sum_y[cid] / counts[cid],
                                        sum_x[cid] / counts[cid])

# -----------------------------------------------------------------------------
# EARLY LOAD OF DENOISED VOLUME (needed for intensity-gated linking)
# Also used later in QC tile montages — loaded once here, reused there.
# -----------------------------------------------------------------------------
DENOISED_VOL_PATH = os.path.join(
    config.DATASPACE, args.denoised_dir_name, TARGET_CORE,
    f'{TARGET_CORE}_denoised.ome.tif',
)
dapi_vol_early = None   # memmap view (n_slices, H, W) — slices read on demand
if args.min_intensity_frac > 0.0:
    if os.path.exists(DENOISED_VOL_PATH):
        try:
            _raw = tifffile.memmap(DENOISED_VOL_PATH, mode='r')
            _raw = np.squeeze(_raw)
            if _raw.ndim == 4:
                # (Z, C, Y, X) — keep as memmap view; slice into channel 0 per Z on demand
                dapi_vol_early = _raw[:, 0, :, :]   # still memmap, no copy
            elif _raw.ndim == 3:
                dapi_vol_early = _raw               # (Z, Y, X), still memmap
            else:
                logger.warning(f"Denoised volume unexpected shape {_raw.shape}; intensity gating disabled.")
            if dapi_vol_early is not None:
                logger.info(f"Denoised DAPI mmap ready for intensity-gated linking: shape={dapi_vol_early.shape}")
        except Exception as _e:
            logger.warning(f"Could not load denoised volume for intensity gating: {_e}. Gating disabled.")
    else:
        logger.warning(
            f"--min_intensity_frac set but denoised OME-TIFF not found at {DENOISED_VOL_PATH}. "
            f"Intensity gating disabled."
        )

# Per-cell mean DAPI intensity (z, cid) -> float.
# Computed only when intensity gating is requested AND volume is available.
# Fast path: uses scipy.ndimage.mean which iterates the image only ONCE per
# slice (O(H*W)) instead of once per cell (O(H*W*n_cells)).
# The memmap is read one slice at a time so we never load the full volume
# into RAM — critical for large TMA OME-TIFFs (19+ slices, full-core FOV).
cell_mean_intensity: dict = {}
if dapi_vol_early is not None and args.min_intensity_frac > 0.0:
    logger.info("Computing per-cell mean DAPI intensity for intensity-gated linking...")
    for z, mask in enumerate(masks_2d):
        if z >= dapi_vol_early.shape[0]:
            continue
        # One-pass bincount approach: O(H*W) per slice regardless of cell count.
        # sum_intensity[cid] = sum of pixel values inside cell cid
        # count[cid]         = number of pixels (= area, already in cell_areas)
        img_slice  = np.array(dapi_vol_early[z], dtype=np.float64)
        flat_mask  = mask.ravel()
        flat_img   = img_slice.ravel()
        max_id     = int(flat_mask.max()) if flat_mask.max() > 0 else 0
        if max_id == 0:
            continue
        sum_intens = np.bincount(flat_mask, weights=flat_img,  minlength=max_id + 1)
        pix_count  = np.bincount(flat_mask,                    minlength=max_id + 1)
        for cid in range(1, max_id + 1):
            if pix_count[cid] > 0:
                cell_mean_intensity[(z, cid)] = float(sum_intens[cid] / pix_count[cid])
        logger.info(f"  Intensity Z{z} ({slice_ids[z]}): {int((pix_count[1:] > 0).sum())} cells")

logger.info("Finding overlapping cell pairs between adjacent slices...")

for z in range(n_slices - 1):
    mask_a = masks_2d[z]
    mask_b = masks_2d[z + 1]

    fg = (mask_a > 0) & (mask_b > 0)
    if not np.any(fg):
        continue

    ids_a = mask_a[fg].astype(np.int64)
    ids_b = mask_b[fg].astype(np.int64)

    pairs        = (ids_a << 32) | ids_b
    unique_pairs, counts = np.unique(pairs, return_counts=True)

    n_links = 0
    for pair, count in zip(unique_pairs, counts):
        if count < args.min_overlap:
            continue
        cid_a = int(pair >> 32)
        cid_b = int(pair & 0xFFFFFFFF)
        area_a = cell_areas.get((z,     cid_a), 1)
        area_b = cell_areas.get((z + 1, cid_b), 1)

        # ── Filter 1: IoU (Intersection over Union / Jaccard index) ───────────
        # IoU = |A∩B| / |A∪B|.  Penalises large area mismatches between adjacent
        # slices more robustly than raw overlap or overlap-fraction alone, because
        # it grows the denominator when either mask is large relative to the overlap.
        # Reference: Hirling et al. (2025) Commun. Biol. — IoU ≥ 0.5 as standard
        # link quality threshold in multiplexed-IF nuclear segmentation benchmarks.
        if args.min_iou > 0.0:
            union = area_a + area_b - count
            iou   = count / union if union > 0 else 0.0
            if iou < args.min_iou:
                continue

        # ── Filter 2: intensity gate ───────────────────────────────────────────
        # Reject a link if either nucleus slice has mean DAPI intensity below
        # min_intensity_frac × the brighter of the two slices.
        # Prevents dim / empty mid-stack CellPose masks from bridging two separate
        # nuclei (e.g. cell 86223 Z12 dark slice).
        # Reference: Mestre et al. Lab. Invest. (2022) — DAPI intensity as
        # discriminative feature for nucleus classification in serial-section IF.
        if args.min_intensity_frac > 0.0 and cell_mean_intensity:
            int_a = cell_mean_intensity.get((z,     cid_a), None)
            int_b = cell_mean_intensity.get((z + 1, cid_b), None)
            if int_a is not None and int_b is not None:
                peak = max(int_a, int_b)
                if peak > 0:
                    if (int_a / peak) < args.min_intensity_frac:
                        continue
                    if (int_b / peak) < args.min_intensity_frac:
                        continue

        edges_uvw.append(((z, cid_a), (z + 1, cid_b), count))
        n_links += 1

    logger.info(
        f"  Z{z} (slice {slice_ids[z]:03d}) -> Z{z+1}: {n_links} cell links"
    )


# -----------------------------------------------------------------------------
# STEP 2 -- BUILD 3D CELLS FROM SEVERED COMPONENTS
# -----------------------------------------------------------------------------

total_nodes = len(all_nodes)
total_edges = len(edges_uvw)
logger.info(
    f"Building 3D cells from topologically severed components "
    f"(graph: {total_nodes} nodes, {total_edges} edges) ...")

components = build_3d_components(all_nodes, edges_uvw, max_slices=args.max_slices)

rows         = []
cell_id      = 0
node_to_3d   = {}
mapping_rows = []

for members in components:
    z_indices = [m[0] for m in members]
    z_min     = min(z_indices)
    z_max     = max(z_indices)
    z_span    = z_max - z_min + 1

    if z_span < args.min_slices:
        continue

    # ── Reject over-segmented components ──────────────────────────────────────
    # A single nucleus must contribute exactly 1 CellPose mask per Z-slice.
    # More than 1 on any slice means CellPose split the nucleus → always reject.
    from collections import Counter
    z_counts = Counter(m[0] for m in members)
    if max(z_counts.values()) > MAX_SEGMENTS_PER_Z:
        continue

    per_slice_areas = {node: cell_areas.get(node, 0) for node in members}
    if max(per_slice_areas.values()) < args.min_area_px:
        continue

    cell_id  += 1
    total_px  = sum(per_slice_areas.values())
    vol_um3   = round(total_px * PIXEL_SIZE_XY_UM**2 * SECTION_THICKNESS_UM, 3)

    for node in members:
        node_to_3d[node] = cell_id
        z, cid2d = node
        mapping_rows.append({
            'cell_id_3d': cell_id,
            'slice_z':    z,
            'slice_id':   slice_ids[z],
            'cell_id_2d': cid2d,
        })

    sum_y, sum_x, sum_z, sum_w = 0.0, 0.0, 0.0, 0.0
    for z, cid2d in members:
        area = per_slice_areas.get((z, cid2d), 0)
        if area == 0:
            continue
        cy, cx  = cell_centroids.get((z, cid2d), (0.0, 0.0))
        sum_y  += cy * area
        sum_x  += cx * area
        sum_z  += z  * area
        sum_w  += area

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
# STEP 3 -- BUILD OUTPUT LABEL VOLUMES
# -----------------------------------------------------------------------------

logger.info("Building 32-bit output 3D label volume...")
output_volume = np.zeros((n_slices, H, W), dtype=np.uint32)

for z, mask in enumerate(masks_2d):
    max_id = int(mask.max()) if mask.max() > 0 else 0
    lut    = np.zeros(max_id + 1, dtype=np.uint32)
    for (nz, cid2d), c3d in node_to_3d.items():   # only assigned nodes
        if nz == z and cid2d <= max_id:
            lut[cid2d] = c3d
    output_volume[z] = lut[mask]

logger.info("Assigning global random display IDs (1-255) for Fiji Glasbey LUT compatibility...")
rng_disp      = np.random.default_rng(seed=42)
unique_3d_ids = np.unique(output_volume)
unique_3d_ids = unique_3d_ids[unique_3d_ids > 0]
max_orig_id   = int(output_volume.max()) if output_volume.size > 0 else 0
lut_display   = np.zeros(max_orig_id + 1, dtype=np.uint16)
for orig_id in unique_3d_ids:
    lut_display[orig_id] = rng_disp.integers(1, 256)
display_volume = lut_display[output_volume]
logger.info("Global display IDs assigned.")


# -----------------------------------------------------------------------------
# SAVE FILES
# -----------------------------------------------------------------------------

df_cells = pd.DataFrame(rows).sort_values('cell_id_3d')
csv_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_stats.csv")
df_cells.to_csv(csv_path, index=False)
logger.info(f"Stats CSV -> {csv_path}")

df_mapping = pd.DataFrame(mapping_rows)
map_path   = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_2d_to_3d_map.csv")
df_mapping.to_csv(map_path, index=False)
logger.info(f"2D-to-3D Mapping CSV -> {map_path}")

vol_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_labels.tif")
logger.info(f"Saving 3D analysis volume -> {vol_path}")
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


# -----------------------------------------------------------------------------
# STEP 4 -- QC PLOTS
# -----------------------------------------------------------------------------

confirmed  = df_cells[df_cells['z_span_slices'] >= args.min_confirmed]
singletons = df_cells[df_cells['z_span_slices'] <  args.min_confirmed]

if args.plot_qc:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    QC_DIR   = os.path.join(OUTPUT_FOLDER, 'qc')
    TILE_DIR = os.path.join(QC_DIR, 'visual_tile_qc')
    os.makedirs(QC_DIR,   exist_ok=True)
    os.makedirs(TILE_DIR, exist_ok=True)

    # Rebuild per-3D-cell member list (sorted by z) for QC metrics
    n_raw = len(mapping_rows)
    seen  = set()
    cell_members = defaultdict(list)
    for row in mapping_rows:
        key = (row['cell_id_3d'], row['slice_z'], row['cell_id_2d'])
        if key in seen:
            continue
        seen.add(key)
        cell_members[row['cell_id_3d']].append((row['slice_z'], row['cell_id_2d']))
    for cid3d in cell_members:
        cell_members[cid3d].sort(key=lambda x: x[0])

    n_deduped = sum(len(v) for v in cell_members.values())
    if n_deduped < n_raw:
        logger.warning(
            f"Duplicate nodes detected in mapping_rows: {n_raw} rows -> "
            f"{n_deduped} unique (dropped {n_raw - n_deduped}). "
            f"Possible upstream bug in build_3d_components."
        )

    # ── QC 1: Overlap fraction distribution ──────────────────────────────────
    logger.info("QC: computing overlap fractions for all linked pairs...")

    # Build a lookup from edges already computed in Step 1 — O(n_edges), not O(n_pairs × H × W)
    edge_overlap_px = {}
    for (z_a, id_a), (z_b, id_b), count in edges_uvw:
        edge_overlap_px[(z_a, id_a, z_b, id_b)] = count

    overlap_fractions = []
    flagged_pairs     = []

    for cid3d, members in cell_members.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            z_a, id_a = members[i]
            z_b, id_b = members[i + 1]
            if z_b != z_a + 1:
                continue
            overlap_px = edge_overlap_px.get((z_a, id_a, z_b, id_b), 0)
            area_a     = cell_areas.get((z_a, id_a), 1)
            area_b     = cell_areas.get((z_b, id_b), 1)
            frac       = overlap_px / min(area_a, area_b)
            overlap_fractions.append(frac)
            if frac < 0.10:
                flagged_pairs.append((cid3d, z_a, id_a, z_b, id_b, round(frac, 3)))

    n_pairs    = len(overlap_fractions)
    n_weak     = len(flagged_pairs)
    frac_weak  = n_weak / max(n_pairs, 1)
    logger.info(f"  Total linked pairs         : {n_pairs}")
    logger.info(f"  Weak links (overlap < 10%) : {n_weak}  ({100*frac_weak:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(overlap_fractions, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(0.10, color='red',    linestyle='--', linewidth=1.2,
               label='Weak link threshold (0.10)')
    ax.axvline(0.25, color='orange', linestyle='--', linewidth=1.2,
               label='Moderate threshold (0.25)')
    ax.set_xlabel('Overlap fraction (shared px / smaller nucleus area)', fontsize=10)
    ax.set_ylabel('Number of linked pairs', fontsize=10)
    ax.set_title(
        f'{TARGET_CORE} {CH_NAME} — Link overlap fraction\n'
        f'n={n_pairs} pairs  |  {n_weak} flagged < 0.10  ({100*frac_weak:.1f}%)',
        fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(QC_DIR, 'overlap_fraction_distribution.png'), dpi=120)
    plt.close(fig)

    if flagged_pairs:
        pd.DataFrame(flagged_pairs,
            columns=['cell_id_3d', 'z_a', 'id_2d_a', 'z_b', 'id_2d_b', 'overlap_fraction']
        ).to_csv(os.path.join(QC_DIR, 'flagged_weak_links.csv'), index=False)

    # ── QC 2: Z-span distribution ─────────────────────────────────────────────
    # Given 4.5 µm section thickness and ~10 µm nucleus diameter, almost all
    # DAPI nuclei should span 2-3 slices. A long tail beyond 4 indicates false
    # merges; a spike at 1 (singletons) indicates segmentation dropout or an
    # overly strict min_overlap.
    logger.info("QC: plotting Z-span distribution...")

    spans    = df_cells['z_span_slices'].values
    max_span = int(spans.max()) if len(spans) > 0 else 5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spans, bins=np.arange(0.5, max_span + 1.5, 1),
            color='mediumseagreen', edgecolor='white', linewidth=0.4)
    ax.axvline(4.5, color='red', linestyle='--', linewidth=1.2,
               label='Biological max for DAPI nuclei (~4 slices)')
    ax.set_xlabel('Z-span (number of slices)', fontsize=10)
    ax.set_ylabel('Number of 3D cells', fontsize=10)
    ax.set_title(
        f'{TARGET_CORE} {CH_NAME} — Z-span distribution\n'
        f'n={len(spans)} cells  |  median={np.median(spans):.1f} slices',
        fontsize=10)
    ax.set_xticks(range(1, max_span + 1))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(QC_DIR, 'z_span_distribution.png'), dpi=120)
    plt.close(fig)

    # ── QC 3: Centroid drift ──────────────────────────────────────────────────
    # For each consecutive linked pair, the Euclidean XY distance between the
    # two 2D nucleus centroids. A true nucleus drifts only a few pixels between
    # slices due to registration imprecision. Drift larger than the nucleus
    # radius (~13-18 px for typical DAPI nuclei) flags suspicious links.
    logger.info("QC: computing centroid drift...")

    drifts         = []
    flagged_drifts = []

    for cid3d, members in cell_members.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            z_a, id_a = members[i]
            z_b, id_b = members[i + 1]
            if z_b != z_a + 1:
                continue
            cy_a, cx_a = cell_centroids.get((z_a, id_a), (0, 0))
            cy_b, cx_b = cell_centroids.get((z_b, id_b), (0, 0))
            drift = float(np.sqrt((cx_b - cx_a)**2 + (cy_b - cy_a)**2))
            drifts.append(drift)
            if drift > FLAG_DRIFT_PX:
                flagged_drifts.append((cid3d, z_a, id_a, z_b, id_b, round(drift, 2)))

    logger.info(f"  Median centroid drift : {np.median(drifts):.1f} px  "
                f"({np.median(drifts)*PIXEL_SIZE_XY_UM:.2f} um)")
    logger.info(f"  High-drift pairs (> {FLAG_DRIFT_PX} px) : {len(flagged_drifts)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(drifts, bins=50, color='mediumpurple', edgecolor='white', linewidth=0.4)
    ax.axvline(FLAG_DRIFT_PX, color='red', linestyle='--', linewidth=1.2,
               label=f'Flag threshold ({FLAG_DRIFT_PX:.0f} px ≈ '
                     f'{FLAG_DRIFT_PX*PIXEL_SIZE_XY_UM:.1f} µm)')
    ax.set_xlabel('XY centroid drift between consecutive slices (px)', fontsize=10)
    ax.set_ylabel('Number of linked pairs', fontsize=10)
    ax.set_title(
        f'{TARGET_CORE} {CH_NAME} — Centroid drift\n'
        f'n={len(drifts)} pairs  |  median={np.median(drifts):.1f} px  |  '
        f'{len(flagged_drifts)} flagged',
        fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(QC_DIR, 'centroid_drift.png'), dpi=120)
    plt.close(fig)

    if flagged_drifts:
        pd.DataFrame(flagged_drifts,
            columns=['cell_id_3d', 'z_a', 'id_2d_a', 'z_b', 'id_2d_b', 'drift_px']
        ).to_csv(os.path.join(QC_DIR, 'flagged_high_drift.csv'), index=False)

    # ── QC 4: Area consistency ────────────────────────────────────────────────
    # For each consecutive linked pair, the ratio of the larger to smaller 2D
    # nucleus area. A true nucleus cross-section shrinks gradually toward the
    # top/bottom of the cell but should not change dramatically. Ratios > 3
    # suggest a fragment in one slice is linked to a full nucleus in the next.
    logger.info("QC: computing area consistency...")

    area_ratios         = []
    flagged_area_ratios = []

    for cid3d, members in cell_members.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            z_a, id_a = members[i]
            z_b, id_b = members[i + 1]
            if z_b != z_a + 1:
                continue
            area_a = cell_areas.get((z_a, id_a), 0)
            area_b = cell_areas.get((z_b, id_b), 0)
            if area_a == 0 or area_b == 0:
                continue
            ratio = max(area_a, area_b) / min(area_a, area_b)
            area_ratios.append(ratio)
            if ratio > FLAG_AREA_RATIO:
                flagged_area_ratios.append(
                    (cid3d, z_a, id_a, area_a, z_b, id_b, area_b, round(ratio, 2))
                )

    logger.info(f"  Median area ratio : {np.median(area_ratios):.2f}")
    logger.info(f"  High-ratio pairs (> {FLAG_AREA_RATIO}) : {len(flagged_area_ratios)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(area_ratios, bins=50, color='darkorange', edgecolor='white', linewidth=0.4)
    ax.axvline(FLAG_AREA_RATIO, color='red', linestyle='--', linewidth=1.2,
               label=f'Flag threshold (ratio = {FLAG_AREA_RATIO})')
    ax.set_xlabel('Area ratio between consecutive slices (larger / smaller)', fontsize=10)
    ax.set_ylabel('Number of linked pairs', fontsize=10)
    ax.set_title(
        f'{TARGET_CORE} {CH_NAME} — Area consistency\n'
        f'n={len(area_ratios)} pairs  |  median ratio={np.median(area_ratios):.2f}  |  '
        f'{len(flagged_area_ratios)} flagged',
        fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(QC_DIR, 'area_consistency.png'), dpi=120)
    plt.close(fig)

    if flagged_area_ratios:
        pd.DataFrame(flagged_area_ratios,
            columns=['cell_id_3d', 'z_a', 'id_2d_a', 'area_a_px',
                     'z_b', 'id_2d_b', 'area_b_px', 'area_ratio']
        ).to_csv(os.path.join(QC_DIR, 'flagged_area_inconsistency.csv'), index=False)

    # ── QC 4.5: Comprehensive Link Metrics Export ─────────────────────────────
    # Aggregate all link metrics into a single dataset to enable cross-pipeline 
    # statistical comparisons without requiring recalculation.
    logger.info("QC: exporting complete link metrics dataset...")

    all_metrics = []
    
    for cid3d, members in cell_members.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            z_a, id_a = members[i]
            z_b, id_b = members[i + 1]
            
            if z_b != z_a + 1:
                continue
                
            # 1. Overlap Fraction
            overlap_px = edge_overlap_px.get((z_a, id_a, z_b, id_b), 0)
            area_a     = cell_areas.get((z_a, id_a), 1)
            area_b     = cell_areas.get((z_b, id_b), 1)
            frac       = overlap_px / min(area_a, area_b)
            
            # 2. Centroid Drift
            cy_a, cx_a = cell_centroids.get((z_a, id_a), (0, 0))
            cy_b, cx_b = cell_centroids.get((z_b, id_b), (0, 0))
            drift      = float(np.sqrt((cx_b - cx_a)**2 + (cy_b - cy_a)**2))
            
            # 3. Area Ratio
            ratio = max(area_a, area_b) / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
            
            all_metrics.append({
                'cell_id_3d': cid3d,
                'z_a': z_a,
                'id_2d_a': id_a,
                'z_b': z_b,
                'id_2d_b': id_b,
                'overlap_px': overlap_px,
                'area_a_px': area_a,
                'area_b_px': area_b,
                'overlap_fraction': round(frac, 3),
                'centroid_drift_px': round(drift, 2),
                'area_ratio': round(ratio, 2)
            })

    if all_metrics:
        metrics_csv_path = os.path.join(QC_DIR, 'all_link_metrics.csv')
        pd.DataFrame(all_metrics).to_csv(metrics_csv_path, index=False)
        logger.info(f"  Complete link metrics saved -> {metrics_csv_path}")

    # ── QC 5: Visual tile montages ────────────────────────────────────────────
    # For a random sample of multi-slice 3D cells, render a row of DAPI
    # intensity patches — one per slice — centred on the 2D nucleus centroid,
    # with the CellPose mask boundary overlaid as a contour so it is easy to
    # judge whether the highlighted nucleus is the same cell across Z.
    # Source: denoised OME-TIFF (Z, C, Y, X); DAPI is channel index 0.
    # Falls back silently to binary mask patches if the OME-TIFF is missing.
    logger.info(f"QC: generating visual tile montages (n={N_TILE_SAMPLES})...")

    PATCH = PATCH_SIZE_PX

    # -- reuse denoised volume loaded for intensity gating (or load now if not done) --
    dapi_vol = dapi_vol_early   # may be None — falls back to binary mask patches below

    def extract_patch(img: np.ndarray, cy: float, cx: float, half: int) -> np.ndarray:
        """Extract a (2*half x 2*half) patch centred on (cy, cx), padding at borders."""
        h, w  = img.shape
        y0    = max(0, int(cy) - half)
        y1    = min(h, int(cy) + half)
        x0    = max(0, int(cx) - half)
        x1    = min(w, int(cx) + half)
        patch = img[y0:y1, x0:x1].astype(np.float32)
        pad_t = max(0, half - int(cy))
        pad_b = max(0, int(cy) + half - h)
        pad_l = max(0, half - int(cx))
        pad_r = max(0, int(cx) + half - w)
        return np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r)), mode='constant')

    multi_slice_ids = [c for c, m in cell_members.items() if len(m) >= args.min_confirmed]

    if not multi_slice_ids:
        logger.warning("No multi-slice cells found — skipping tile QC.")
    else:
        # Matches against (or, when --set_qc_reference is passed, writes) the
        # shared cross-pipeline QC reference, so the same physical cells show
        # up in every registration variant's QC. See qc_reference.py.
        sample_ids = get_or_match_qc_cells(
            core_name=TARGET_CORE, df_cells=df_cells, min_confirmed=args.min_confirmed,
            n_samples=N_TILE_SAMPLES, pipeline_tag=args.input_dir_name,
            dataspace=config.DATASPACE, seed=0,
            set_reference=args.set_qc_reference, logger=logger,
            pixel_size_um=PIXEL_SIZE_XY_UM, section_thickness_um=SECTION_THICKNESS_UM,
            qc_pipeline_kind=args.qc_pipeline_kind, qc_transform_dir=args.qc_transform_dir,
        )
        if not sample_ids:
            logger.warning("QC reference matching returned no cells — skipping tile QC.")

        # Build two lookups used during neighbour rendering:
        #   z_to_multislice_2d : z -> set of 2D IDs belonging to any confirmed
        #                        multi-slice 3D cell (for fast crop filtering)
        #   id2d_to_cid3d      : (z, id_2d) -> cid3d  (to map 2D crop hit back
        #                        to its 3D cell so colours stay consistent)
        z_to_multislice_2d: dict = defaultdict(set)
        id2d_to_cid3d:      dict = {}
        for cid3d_ms in multi_slice_ids:
            for z_ms, id_2d_ms in cell_members[cid3d_ms]:
                z_to_multislice_2d[z_ms].add(id_2d_ms)
                id2d_to_cid3d[(z_ms, id_2d_ms)] = cid3d_ms

        # Visually distinct colours for neighbour pairs (RGBA, alpha=0.7).
        # Green is reserved for the primary cell; these are used for neighbours.
        NEIGHBOUR_COLOURS = [
            [1.0, 0.2, 0.2, 0.7],   # red
            [1.0, 0.8, 0.0, 0.7],   # yellow
            [0.6, 0.2, 1.0, 0.7],   # purple
            [1.0, 0.5, 0.0, 0.7],   # orange
        ]
        MAX_NEIGHBOURS = len(NEIGHBOUR_COLOURS)  # show at most 4 pairs

        for cid3d in sample_ids:
            members       = cell_members[cid3d]
            n_slices_cell = len(members)
            patches, mask_patches, labels = [], [], []
            # per-slice list of dicts: {cid3d_neighbour: boundary_patch}
            neighbour_patches_per_slice = []

            # --- Pass 1: find which neighbour 3D cells are visible in ANY
            #     slice of this primary cell, then pick up to MAX_NEIGHBOURS
            #     by the order they are first encountered. ——————————————————
            neighbour_cids_seen = {}   # cid3d -> colour index, insertion-ordered
            for z, id_2d in members:
                cy, cx = cell_centroids.get((z, id_2d), (H // 2, W // 2))
                y0 = max(0, int(cy) - PATCH);  y1 = min(H, int(cy) + PATCH)
                x0 = max(0, int(cx) - PATCH);  x1 = min(W, int(cx) + PATCH)
                mask_crop = masks_2d[z][y0:y1, x0:x1]
                present_2d = (set(np.unique(mask_crop))
                              & z_to_multislice_2d.get(z, set()))
                present_2d.discard(id_2d)
                present_2d.discard(0)
                for nb_2d in present_2d:
                    nb_cid = id2d_to_cid3d.get((z, nb_2d))
                    if nb_cid is None or nb_cid == cid3d:
                        continue
                    if nb_cid not in neighbour_cids_seen:
                        if len(neighbour_cids_seen) >= MAX_NEIGHBOURS:
                            break
                        neighbour_cids_seen[nb_cid] = len(neighbour_cids_seen)

            # --- Pass 2: build patches slice by slice ————————————————————
            for z, id_2d in members:
                cy, cx = cell_centroids.get((z, id_2d), (H // 2, W // 2))
                area   = cell_areas.get((z, id_2d), 0)

                # Intensity patch ————————————————————————————————————————
                if dapi_vol is not None and z < dapi_vol.shape[0]:
                    raw_patch = extract_patch(dapi_vol[z], cy, cx, PATCH)
                    fg = raw_patch[raw_patch > 0]
                    if fg.size > 10:
                        p2, p98 = np.percentile(fg, (2, 98))
                        norm    = np.clip((raw_patch - p2) / max(p98 - p2, 1e-6), 0, 1)
                    else:
                        norm = raw_patch / max(raw_patch.max(), 1e-6)
                    source = 'intensity'
                else:
                    norm   = extract_patch(
                        (masks_2d[z] == id_2d).astype(np.float32), cy, cx, PATCH
                    )
                    source = 'mask'

                # Primary cell boundary ———————————————————————————————————
                binary     = (masks_2d[z] == id_2d)
                from scipy.ndimage import binary_erosion
                boundary   = binary & ~binary_erosion(binary)
                mask_patch = extract_patch(boundary.astype(np.float32), cy, cx, PATCH)

                # Neighbour boundary patches (one per selected 3D neighbour) —
                y0 = max(0, int(cy) - PATCH);  y1 = min(H, int(cy) + PATCH)
                x0 = max(0, int(cx) - PATCH);  x1 = min(W, int(cx) + PATCH)
                mask_crop  = masks_2d[z][y0:y1, x0:x1]
                pad_t = max(0, PATCH - int(cy));  pad_b = max(0, int(cy) + PATCH - H)
                pad_l = max(0, PATCH - int(cx));  pad_r = max(0, int(cx) + PATCH - W)

                nb_patches_this_slice = {}
                for nb_cid, col_idx in neighbour_cids_seen.items():
                    # find the 2D id for this neighbour on this Z (may be absent)
                    nb_2d = next(
                        (iid for (zz, iid) in cell_members[nb_cid] if zz == z),
                        None,
                    )
                    if nb_2d is None:
                        nb_patches_this_slice[nb_cid] = None
                        continue
                    nb_bin  = mask_crop == nb_2d
                    nb_ring = nb_bin & ~binary_erosion(nb_bin)
                    nb_pad  = np.pad(nb_ring.astype(np.float32),
                                     ((pad_t, pad_b), (pad_l, pad_r)),
                                     mode='constant')
                    nb_patches_this_slice[nb_cid] = nb_pad

                patches.append(norm)
                mask_patches.append(mask_patch)
                neighbour_patches_per_slice.append(nb_patches_this_slice)
                labels.append(f'Z{z} / s{slice_ids[z]}\narea={area}px\n({source})')

            # Pad all patches to the same size ————————————————————————————
            target_h = max(p.shape[0] for p in patches)
            target_w = max(p.shape[1] for p in patches)

            def pad_to(p, th, tw):
                return np.pad(p, ((0, th - p.shape[0]), (0, tw - p.shape[1])), mode='constant')

            patches      = [pad_to(p, target_h, target_w) for p in patches]
            mask_patches = [pad_to(p, target_h, target_w) for p in mask_patches]

            row_stats = df_cells[df_cells['cell_id_3d'] == cid3d]
            span = int(row_stats['z_span_slices'].values[0]) if len(row_stats) > 0 else '?'
            vol  = float(row_stats['volume_um3'].values[0])  if len(row_stats) > 0 else 0.0

            fig, axes = plt.subplots(1, n_slices_cell,
                                     figsize=(n_slices_cell * 2.5, 3.0))
            if n_slices_cell == 1:
                axes = [axes]

            fig.suptitle(
                f'3D cell {cid3d}  |  span={span} slices  |  vol={vol:.0f} µm³',
                fontsize=9, y=1.02,
            )
            for ax, patch, mpatch, nb_patches_slice, label in zip(
                    axes, patches, mask_patches, neighbour_patches_per_slice, labels):
                ax.imshow(patch, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                overlay = np.zeros((*patch.shape, 4), dtype=np.float32)
                # Draw neighbours first so primary cell green always wins on top
                for nb_cid, col_idx in neighbour_cids_seen.items():
                    nb_pad = nb_patches_slice.get(nb_cid)
                    if nb_pad is None:
                        continue
                    nb_pad = pad_to(nb_pad, target_h, target_w)
                    colour = NEIGHBOUR_COLOURS[col_idx]
                    overlay[nb_pad > 0] = colour
                # Primary cell boundary — always green
                overlay[mpatch > 0] = [0.0, 1.0, 0.0, 0.8]
                ax.imshow(overlay, interpolation='nearest')
                ax.set_title(label, fontsize=6)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(TILE_DIR, f'cell_{cid3d:06d}.png'),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"  Tile montages -> {TILE_DIR}/")

    # ── QC summary to log ─────────────────────────────────────────────────────
    logger.info('=' * 60)
    logger.info(f'QC SUMMARY  —  {TARGET_CORE} / {CH_NAME}')
    logger.info(f'  Total 3D cells          : {len(df_cells)}')
    logger.info(f'  Multi-slice (confirmed) : {len(confirmed)}')
    logger.info(f'  Singletons              : {len(singletons)}')
    logger.info(f'  Total linked pairs      : {n_pairs}')
    logger.info(f'  Weak links (<10% overlap): {n_weak}  ({100*frac_weak:.1f}%)')
    logger.info(f'  High centroid drift     : {len(flagged_drifts)}')
    logger.info(f'  High area ratio         : {len(flagged_area_ratios)}')
    logger.info(f'  QC plots  -> {QC_DIR}/')
    logger.info('=' * 60)

    if frac_weak > 0.15:
        logger.warning(
            f"> 15% of links have overlap fraction < 0.10. "
            f"Consider increasing --min_overlap in the batch script."
        )
    if drifts and np.median(drifts) > 15:
        logger.warning(
            f"Median centroid drift is high ({np.median(drifts):.1f} px). "
            f"Check registration quality or tighten linking constraints."
        )


# -----------------------------------------------------------------------------
# STEP 5 -- CROSS-CHANNEL CO-LOCALISATION (optional)
# -----------------------------------------------------------------------------

if args.coloc_channel is not None:
    coloc_dir_name = args.coloc_dir_name or f"CellPose_{args.coloc_channel}_3D_Bspline"
    coloc_csv = os.path.join(
        config.DATASPACE, coloc_dir_name,
        TARGET_CORE, f"{TARGET_CORE}_{args.coloc_channel}_3d_stats.csv"
    )
    if os.path.exists(coloc_csv):
        df_coloc = pd.read_csv(coloc_csv)
        conf_b   = df_coloc[df_coloc['z_span_slices'] >= args.min_confirmed]

        if len(confirmed) > 0 and len(conf_b) > 0:
            coords_a = confirmed[['centroid_x_um', 'centroid_y_um', 'centroid_z_um']].values
            coords_b = conf_b[['centroid_x_um',   'centroid_y_um', 'centroid_z_um']].values

            tree_b    = cKDTree(coords_b)
            dist_a, _ = tree_b.query(coords_a, k=1)

            confirmed = confirmed.copy()
            confirmed[f'nearest_{args.coloc_channel}_um'] = dist_a

            enriched_path = csv_path.replace(
                '_3d_stats.csv', f'_3d_stats_with_nearest_{args.coloc_channel}.csv'
            )
            confirmed.to_csv(enriched_path, index=False)
            logger.info(f"Enriched co-localisation CSV -> {enriched_path}")
    else:
        logger.warning(f"Co-localisation CSV not found: {coloc_csv}  (skipping)")

logger.info("Done.")