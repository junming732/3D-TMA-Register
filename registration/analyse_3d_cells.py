"""
analyse_3d_cells.py
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
   - 8-bit randomized display volume (for Fiji/Glasbey LUT compatibility).
   - Interactive 3D HTML point cloud for spatial QC.

Usage
-----
    python analyse_3d_cells.py \
        --core_name   Core_01   \
        --channel     CD3       \
        [--min_overlap  3]      \
        [--max_slices   3]
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

# Attempt Plotly import for interactive HTML generation; degrade gracefully if missing.
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
                    help="Min shared pixels for two 2D cells to be linked across Z.")
parser.add_argument('--min_slices',   type=int, default=1,
                    help="Min Z-slices a 3D cell must span to be kept (default: 1).")
parser.add_argument('--max_slices',   type=int, default=3,
                    help="Max Z-slices a 3D cell may span. Default 3 for standard T-cells.")
parser.add_argument('--min_area_px',  type=int, default=10,
                    help="Min pixel area of a 2D cell to include in linking (default: 10).")
parser.add_argument('--plot_qc',       action='store_true',
                    help="Save QC montage PNG and interactive HTML point cloud.")
parser.add_argument('--coloc_channel', type=str,   default=None,
                    help="Second channel for co-localisation e.g. CD163.")
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

class WeightedCellGraph:
    """
    Adjacency list graph to track connection strengths between 2D cell overlaps.
    Supports recursive bottleneck severing for components exceeding Z-span limits.
    """
    def __init__(self):
        self.adj = defaultdict(dict)
        self.isolated_nodes = set()

    def add_node(self, node):
        if node not in self.adj:
            self.isolated_nodes.add(node)

    def add_edge(self, u, v, weight):
        self.adj[u][v] = weight
        self.adj[v][u] = weight
        self.isolated_nodes.discard(u)
        self.isolated_nodes.discard(v)

    def _get_initial_components(self):
        """Standard Breadth-First Search to find all disconnected sub-graphs."""
        visited = set()
        components = []
        for node in list(self.adj.keys()) + list(self.isolated_nodes):
            if node not in visited:
                comp = set()
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in self.adj.get(curr, {}):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)
        return components

    def get_valid_components(self, max_slices):
        """Returns a list of components, iteratively severing any that exceed max_slices."""
        valid_components = []
        # Use an explicit stack instead of recursion to avoid hitting Python's recursion limit
        stack = self._get_initial_components()

        while stack:
            comp = stack.pop()

            z_indices = [node[0] for node in comp]
            z_span = max(z_indices) - min(z_indices) + 1

            if z_span <= max_slices:
                valid_components.append(comp)
                continue

            # Find the weakest edge (minimum pixel overlap) within this component
            min_weight = float('inf')
            weakest_edge = None

            for u in comp:
                for v, weight in self.adj.get(u, {}).items():
                    if v in comp and weight < min_weight:
                        min_weight = weight
                        weakest_edge = (u, v)

            if not weakest_edge:
                # Cannot sever further; accept as-is
                valid_components.append(comp)
                continue

            # Sever the weakest link
            u, v = weakest_edge
            del self.adj[u][v]
            del self.adj[v][u]

            # Re-split via localized BFS and push sub-components onto the stack
            visited = set()
            for start_node in (u, v):
                if start_node not in visited:
                    sub_comp = set()
                    queue = [start_node]
                    visited.add(start_node)
                    while queue:
                        curr = queue.pop(0)
                        sub_comp.add(curr)
                        for neighbor in self.adj.get(curr, {}):
                            if neighbor in comp and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    stack.append(sub_comp)

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

graph = WeightedCellGraph()

# Precompute 2D areas, centroids, and initialize graph nodes
logger.info("Precomputing 2D cell areas and centroids...")
cell_areas     = {}   
cell_centroids = {}   

for z, mask in enumerate(masks_2d):
    counts = np.bincount(mask.ravel())
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_areas[(z, cid)] = int(counts[cid])
            graph.add_node((z, int(cid)))

    ys, xs = np.indices(mask.shape)
    flat   = mask.ravel()
    ys_f   = ys.ravel().astype(np.float64)
    xs_f   = xs.ravel().astype(np.float64)
    sum_y = np.bincount(flat, weights=ys_f, minlength=len(counts))
    sum_x = np.bincount(flat, weights=xs_f, minlength=len(counts))
    for cid in range(1, len(counts)):
        if counts[cid] > 0:
            cell_centroids[(z, cid)] = (sum_y[cid] / counts[cid], sum_x[cid] / counts[cid])

logger.info("Finding overlapping cell pairs between adjacent slices...")

for z in range(n_slices - 1):
    mask_a = masks_2d[z]
    mask_b = masks_2d[z + 1]

    fg = (mask_a > 0) & (mask_b > 0)
    if not np.any(fg):
        continue

    ids_a = mask_a[fg].astype(np.int64)
    ids_b = mask_b[fg].astype(np.int64)

    pairs = (ids_a << 32) | ids_b
    unique_pairs, counts = np.unique(pairs, return_counts=True)

    n_links = 0
    for pair, count in zip(unique_pairs, counts):
        if count < args.min_overlap:
            continue
        cid_a = int(pair >> 32)
        cid_b = int(pair & 0xFFFFFFFF)
        
        graph.add_edge((z, cid_a), (z + 1, cid_b), weight=count)
        n_links += 1

    logger.info(
        f"  Z{z} (slice {slice_ids[z]:03d}) -> Z{z+1}: {n_links} cell links"
    )


# -----------------------------------------------------------------------------
# STEP 2 -- BUILD 3D CELLS FROM SEVERED COMPONENTS
# -----------------------------------------------------------------------------

logger.info("Building 3D cells from topologically severed components...")

# The graph automatically identifies and recursively severs overgrown clusters
components = graph.get_valid_components(max_slices=args.max_slices)

rows       = []
cell_id    = 0
node_to_3d = {}

for members in components:
    z_indices = [m[0] for m in members]
    z_min     = min(z_indices)
    z_max     = max(z_indices)
    z_span    = z_max - z_min + 1

    if z_span < args.min_slices:
        continue

    per_slice_areas = {node: cell_areas.get(node, 0) for node in members}
    if max(per_slice_areas.values()) < args.min_area_px:
        continue

    cell_id += 1
    total_px = sum(per_slice_areas.values())
    vol_um3  = round(total_px * PIXEL_SIZE_XY_UM**2 * SECTION_THICKNESS_UM, 3)

    for node in members:
        node_to_3d[node] = cell_id

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
# STEP 3 -- BUILD OUTPUT LABEL VOLUMES
# -----------------------------------------------------------------------------

logger.info("Building 32-bit output 3D label volume...")
output_volume = np.zeros((n_slices, H, W), dtype=np.uint32)

for z, mask in enumerate(masks_2d):
    max_id  = int(mask.max()) if mask.max() > 0 else 0
    lut     = np.zeros(max_id + 1, dtype=np.uint32)
    for cid2d in range(1, max_id + 1):
        node = (z, cid2d)
        if node in node_to_3d:
            lut[cid2d] = node_to_3d[node]
    out_slice = lut[mask]
    output_volume[z] = out_slice


logger.info("Assigning global random display IDs (1-255) for Fiji Glasbey LUT compatibility...")
rng_disp = np.random.default_rng(seed=42)

unique_3d_ids = np.unique(output_volume)
unique_3d_ids = unique_3d_ids[unique_3d_ids > 0]

max_orig_id = int(output_volume.max()) if output_volume.size > 0 else 0
lut_display = np.zeros(max_orig_id + 1, dtype=np.uint16)

# Generate uniform distribution across the 1-255 color range
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
# QC & PATTERN ANALYSIS
# -----------------------------------------------------------------------------

confirmed  = df_cells[df_cells['z_span_slices'] >= args.min_confirmed]
singletons = df_cells[df_cells['z_span_slices'] <  args.min_confirmed]

if args.plot_qc:
    logger.info("Generating static QC montage...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng      = np.random.default_rng(seed=42)
    n_colors = 256
    palette  = rng.uniform(0.3, 1.0, size=(n_colors - 1, 4))
    palette[:, 3] = 1.0
    rng.shuffle(palette)
    colors = np.vstack([[0, 0, 0, 1], palette])
    cmap   = ListedColormap(colors)

    def cycle_labels(arr):
        return np.where(arr > 0, (arr % (n_colors - 1)) + 1, 0)

    max_proj = display_volume.max(axis=0)

    n_cols = min(n_slices, 8)
    n_rows = (n_slices + n_cols - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(n_rows, n_cols)

    axes[0, 0].imshow(cycle_labels(max_proj), cmap=cmap, vmin=0, vmax=n_colors, interpolation='nearest')
    axes[0, 0].set_title("Max projection\n(Display IDs)", fontsize=8)
    axes[0, 0].axis('off')
    for j in range(1, n_cols):
        axes[0, j].axis('off')

    for idx in range(n_slices):
        row = idx // n_cols + 1
        col = idx % n_cols
        slc = display_volume[idx]
        axes[row, col].imshow(slc, cmap=cmap, vmin=0, vmax=n_colors, interpolation='nearest')
        axes[row, col].set_title(f"ID {slice_ids[idx]}\n{len(np.unique(output_volume[idx])) - 1} cells", fontsize=7)
        axes[row, col].axis('off')

    for idx in range(n_slices, n_rows * n_cols - n_cols):
        axes[idx // n_cols + 1, idx % n_cols].axis('off')

    plt.tight_layout()
    qc_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_3d_qc.png")
    plt.savefig(qc_path, dpi=80, bbox_inches='tight')
    plt.close(fig)

# ── Interactive 3D Point Cloud ────────────────────────────────────────────────
if args.plot_qc and 'centroid_x_um' in df_cells.columns and len(confirmed) > 0:
    if PLOTLY_AVAILABLE:
        try:
            logger.info("Generating interactive 3D point cloud (HTML)...")
            plot_df = df_cells.copy()
            plot_df['status'] = np.where(plot_df['z_span_slices'] >= args.min_confirmed, 
                                         'Confirmed', 'Singleton')
            
            fig = px.scatter_3d(
                plot_df, x='centroid_x_um', y='centroid_y_um', z='centroid_z_um',
                color='status', hover_data=['cell_id_3d', 'z_span_slices', 'volume_um3'],
                color_discrete_map={'Confirmed': 'red', 'Singleton': 'gray'},
                opacity=0.6, title=f"{TARGET_CORE} {CH_NAME} -- Spatial Cloud"
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(scene=dict(aspectmode='data')) 
            
            pc_html_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_{CH_NAME}_pointcloud.html")
            fig.write_html(pc_html_path)
            logger.info(f"Interactive Point cloud saved -> {pc_html_path}")
        except Exception as e:
            logger.warning(f"Interactive point cloud failed: {e}")
    else:
        logger.warning("Plotly not installed. Skipping interactive HTML export. Install via: pip install plotly")

# ── Cross-channel co-localisation ─────────────────────────────────────────────
if args.coloc_channel is not None:
    coloc_csv = os.path.join(
        config.DATASPACE, f"CellPose_{args.coloc_channel}_3D",
        TARGET_CORE, f"{TARGET_CORE}_{args.coloc_channel}_3d_stats.csv"
    )
    if os.path.exists(coloc_csv):
        df_coloc = pd.read_csv(coloc_csv)
        conf_b   = df_coloc[df_coloc['z_span_slices'] >= args.min_confirmed]

        if len(confirmed) > 0 and len(conf_b) > 0:
            coords_a = confirmed[['centroid_x_um', 'centroid_y_um', 'centroid_z_um']].values
            coords_b = conf_b[['centroid_x_um', 'centroid_y_um', 'centroid_z_um']].values

            tree_b         = cKDTree(coords_b)
            dist_a, _      = tree_b.query(coords_a, k=1)
            
            confirmed = confirmed.copy()
            confirmed[f'nearest_{args.coloc_channel}_um'] = dist_a
            
            enriched_path = csv_path.replace('_3d_stats.csv', f'_3d_stats_with_nearest_{args.coloc_channel}.csv')
            confirmed.to_csv(enriched_path, index=False)
            logger.info(f"Enriched Coloc CSV -> {enriched_path}")

logger.info("Done.")