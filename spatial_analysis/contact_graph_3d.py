"""Computes a 3D and 2D nucleus-adjacency contact graph from segmented volumes.

This script evaluates geometric touching between cells by expanding segmented 
nuclei by a defined gap distance. It performs a 3D pass on a reconstructed 
volume and a 2D pass on individual warped masks to quantify how much a single 
2D section undercounts cell-cell contacts compared to a 3D volume. 

The resulting outputs are structurally identical CSV summaries for both dimensions, 
allowing for direct comparison of contact frequencies.

Notes:
    * Adjacency is based on expanded nuclei, not whole-cell membrane boundaries, 
      as there is no cytoplasm segmentation in this pipeline.
    * The 3D pass expands labels anisotropically using real voxel geometry. 
      Contacts confined entirely within the z-gap (4.0 um) may be under-detected 
      if the contact gap is smaller than the slice spacing.
    * The 2D pass expands labels isotropically in the XY plane per slice, 
      ignoring cross-slice information to simulate single-section imaging.
    * Boundary voxel pair counts in the raw outputs represent relative contact 
      extent and should not be compared directly between 2D and 3D tables.

Generated Files:
    * contact_pairs_3d.csv / contact_pairs_2d.csv: Raw pairs containing 
      source/target cell IDs, types, and boundary voxel pair counts.
    * contact_summary_3d.csv / contact_summary_2d.csv: Aggregated tables 
      containing mean contacts per cell, percentage of cells with contact, 
      total source cells, and total contact events.

Example:
    Run the analysis with a default 2.0 um gap:
        $ python contact_graph_3d.py --core_name Core_01

    Run with strict geometric touching only (0 um gap):
        $ python contact_graph_3d.py --core_name Core_01 --contact_gap_um 0.0

    Skip the 2D slice processing:
        $ python contact_graph_3d.py --core_name Core_01 --skip_2d
"""
import os
import re
import sys
import glob
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import expand_labels

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Physical units — must match link_3d_cells.py / render_3d_cells.py /
# compare_2d_3d_tme.py exactly, or contact_gap_um will not mean what it says.
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.0


def get_slice_id(path: str) -> int:
    """Mirrors link_3d_cells.py's own get_slice_id exactly — same warped
    filenames, same TMA_<N>_ naming, must stay in sync with that function if
    the naming convention ever changes there."""
    match = re.search(r"TMA_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_adjacent_pairs(label_arr, spacing, contact_gap_um):
    """Expand every label by contact_gap_um (physical units, given spacing),
    then find every pair of different non-background labels that end up
    face-adjacent, via one vectorized shift-and-compare pass per axis.
    Works for a 2D slice or a 3D volume identically — spacing's length just
    needs to match label_arr.ndim. Returns (lo_ids, hi_ids, boundary_voxel_pair_counts).
    """
    if contact_gap_um > 0:
        expanded = expand_labels(label_arr, distance=contact_gap_um, spacing=spacing)
    else:
        expanded = label_arr

    max_label = int(expanded.max())
    if max_label == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    key_base = max_label + 1

    lo_parts, hi_parts = [], []
    for axis in range(expanded.ndim):
        a = np.take(expanded, indices=np.arange(expanded.shape[axis] - 1), axis=axis)
        b = np.take(expanded, indices=np.arange(1, expanded.shape[axis]),  axis=axis)
        mask = (a != b) & (a > 0) & (b > 0)
        pa, pb = a[mask], b[mask]
        lo_parts.append(np.minimum(pa, pb))
        hi_parts.append(np.maximum(pa, pb))

    lo_all = np.concatenate(lo_parts).astype(np.int64)
    hi_all = np.concatenate(hi_parts).astype(np.int64)
    if len(lo_all) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    keys = lo_all * key_base + hi_all
    keys_u, counts = np.unique(keys, return_counts=True)
    lo_u = (keys_u // key_base).astype(int)
    hi_u = (keys_u %  key_base).astype(int)
    return lo_u, hi_u, counts


def aggregate_contacts(df_pairs, id_col_a, id_col_b, type_col_a, type_col_b,
                       type_counts, min_cells):
    """Directed edge expansion + per-type-pair aggregation — shared by the 2D
    and 3D passes; only the input pair table's id/type column names and the
    type_counts denominator differ between them. id_col_a/b values just need
    to be globally unique per instance (for 2D that means callers must pass
    an id that already encodes slice_id, since raw CellPose ids repeat
    across slices — see the 2D pass below)."""
    if len(df_pairs) > 0:
        edges_fwd = df_pairs.rename(columns={
            id_col_a: 'src_id', id_col_b: 'tgt_id',
            type_col_a: 'src_type', type_col_b: 'tgt_type',
        })[['src_id', 'tgt_id', 'src_type', 'tgt_type']]
        edges_rev = df_pairs.rename(columns={
            id_col_b: 'src_id', id_col_a: 'tgt_id',
            type_col_b: 'src_type', type_col_a: 'tgt_type',
        })[['src_id', 'tgt_id', 'src_type', 'tgt_type']]
        edges = pd.concat([edges_fwd, edges_rev], ignore_index=True)
    else:
        edges = pd.DataFrame(columns=['src_id', 'tgt_id', 'src_type', 'tgt_type'])

    valid_types = sorted(t for t, n in type_counts.items() if n >= min_cells)

    # Distinct-partner count per (src_id, tgt_type) — a cell touching the same
    # partner via multiple boundary voxels still counts as ONE contact.
    per_cell_partner_counts = (
        edges.drop_duplicates(subset=['src_id', 'tgt_id'])
             .groupby(['src_id', 'src_type', 'tgt_type'])
             .size().rename('n_contacts').reset_index()
    )

    summary_rows = []
    for src_type in valid_types:
        n_src_cells = int(type_counts.get(src_type, 0))
        for tgt_type in valid_types:
            sub = per_cell_partner_counts[
                (per_cell_partner_counts['src_type'] == src_type) &
                (per_cell_partner_counts['tgt_type'] == tgt_type)
            ]
            # Cells of src_type with zero contacts of tgt_type never appear in
            # `sub` (groupby only sees cells with >=1 contact) — the mean
            # below divides by ALL src cells, not just contacting ones, so
            # those zeros are implicitly included.
            n_contacting  = len(sub)
            n_events      = int(sub['n_contacts'].sum())
            mean_contacts = n_events / n_src_cells if n_src_cells > 0 else np.nan
            pct_contact   = 100.0 * n_contacting / n_src_cells if n_src_cells > 0 else np.nan
            summary_rows.append({
                'src_type':               src_type,
                'tgt_type':               tgt_type,
                'mean_contacts_per_cell': round(mean_contacts, 4),
                'pct_cells_with_contact': round(pct_contact,   2),
                'n_src_cells':            n_src_cells,
                'n_contact_events':       n_events,
            })
    return pd.DataFrame(summary_rows), valid_types


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Real nucleus-adjacency contact graph, 2D and 3D — sibling metric '
               'to compare_2d_3d_tme.py\'s centroid-based nn_distances_2d/3d.csv.'
)
parser.add_argument('--core_name', type=str, required=True)
parser.add_argument('--contact_gap_um', type=float, default=2.0,
                    help='Physical gap (µm) within which two nuclei are called '
                         '"in contact", applied via expand_labels with the '
                         'pipeline\'s real voxel spacing (isotropic-in-plane for '
                         'the 2D pass, anisotropic z-vs-xy for the 3D pass) '
                         '(default: 2.0). Set to 0 for strict touching only. '
                         'For the 3D pass, values below SECTION_THICKNESS_UM may '
                         'under-detect contacts confined to the z-gap between '
                         'sections.')
parser.add_argument('--min_cells', type=int, default=10,
                    help='Minimum instances of a type required for it to appear '
                         'in the summary CSVs (default: 10, matches '
                         'compare_2d_3d_tme.py\'s --min_cells). For the 2D table '
                         'this counts per-slice instances, same convention as '
                         'nn_distances_2d.csv.')
parser.add_argument('--skip_2d', action='store_true',
                    help='Skip the per-slice 2D pass and only compute '
                         'contact_summary_3d.csv (default: both are computed).')
parser.add_argument('--linking_dir_name', type=str, default='CellPose_DAPI_3D_Bspline',
                    help='Folder under DATASPACE containing link_3d_cells.py output '
                         '(default: CellPose_DAPI_3D_Bspline).')
parser.add_argument('--warped_mask_dir_name', type=str, default='CellPose_DAPI_Warped_Bspline',
                    help='Folder under DATASPACE containing the per-slice warped DAPI '
                         'masks — the SAME folder link_3d_cells.py reads as its own '
                         '--input_dir_name, used here only for the 2D pass '
                         '(default: CellPose_DAPI_Warped_Bspline).')
parser.add_argument('--phenotype_dir_name', type=str, default='Phenotypes_Bspline',
                    help='Folder under DATASPACE containing <CORE>_3d_typed.csv and '
                         '<CORE>_phenotypes_typed.csv from assign_phenotypes.py '
                         '(default: Phenotypes_Bspline).')
parser.add_argument('--output_dir_name', type=str, default='TME_Analysis_Bspline',
                    help='Folder under DATASPACE to write into — same per-core folder '
                         'compare_2d_3d_tme.py writes nn_distances_2d/3d.csv into, so '
                         'this lands alongside them as a sibling metric (default: '
                         'TME_Analysis_Bspline).')
args = parser.parse_args()

TARGET_CORE    = args.core_name
CONTACT_GAP_UM = args.contact_gap_um
MIN_CELLS      = args.min_cells

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
LINK3D_DIR = os.path.join(config.DATASPACE, args.linking_dir_name,     TARGET_CORE)
WARPED_DIR = os.path.join(config.DATASPACE, args.warped_mask_dir_name, TARGET_CORE)
PHENO_DIR  = os.path.join(config.DATASPACE, args.phenotype_dir_name,   TARGET_CORE)
OUT_DIR    = os.path.join(config.DATASPACE, args.output_dir_name,      TARGET_CORE)
os.makedirs(OUT_DIR, exist_ok=True)

LABELS_TIF   = os.path.join(LINK3D_DIR, f'{TARGET_CORE}_DAPI_3d_labels.tif')
TYPED_3D_CSV = os.path.join(PHENO_DIR,  f'{TARGET_CORE}_3d_typed.csv')
TYPED_2D_CSV = os.path.join(PHENO_DIR,  f'{TARGET_CORE}_phenotypes_typed.csv')

for path, label in [(LABELS_TIF, '3D label volume'), (TYPED_3D_CSV, '3D typed CSV')]:
    if not os.path.exists(path):
        logger.error(f'{label} not found: {path} — run link_3d_cells.py / '
                     f'assign_phenotypes.py first.')
        sys.exit(1)

# =============================================================================
# 3D PASS
# =============================================================================
logger.info(f'Loading 3D label volume: {LABELS_TIF}')
label_vol = tifffile.imread(LABELS_TIF).astype(np.uint32)
logger.info(f'  shape={label_vol.shape}  dtype={label_vol.dtype}  '
           f'n_labels={len(np.unique(label_vol)) - 1}')

df_typed_3d = pd.read_csv(TYPED_3D_CSV)
type_of_3d  = dict(zip(df_typed_3d['cell_id_3d'].astype(int), df_typed_3d['cell_type']))

logger.info(f'Expanding 3D labels by {CONTACT_GAP_UM} µm '
           f'(spacing z={SECTION_THICKNESS_UM} µm, xy={PIXEL_SIZE_XY_UM} µm) ...')
t0 = time.time()
lo_u, hi_u, counts = find_adjacent_pairs(
    label_vol, spacing=(SECTION_THICKNESS_UM, PIXEL_SIZE_XY_UM, PIXEL_SIZE_XY_UM),
    contact_gap_um=CONTACT_GAP_UM,
)
logger.info(f'  {len(lo_u)} unique adjacent cell pairs found in {time.time() - t0:.1f}s.')

n_untyped_3d = 0
pair_rows_3d = []
for a, b, n in zip(lo_u, hi_u, counts):
    ta, tb = type_of_3d.get(a), type_of_3d.get(b)
    if ta is None or tb is None:
        n_untyped_3d += 1
        continue
    pair_rows_3d.append({'cell_id_3d_a': a, 'cell_id_3d_b': b,
                         'type_a': ta, 'type_b': tb,
                         'n_boundary_voxel_pairs': int(n)})
if n_untyped_3d:
    logger.warning(f'  {n_untyped_3d} adjacent 3D pairs involved a label with no '
                   f'entry in {os.path.basename(TYPED_3D_CSV)} — excluded.')

df_pairs_3d = pd.DataFrame(pair_rows_3d)
pairs_3d_csv = os.path.join(OUT_DIR, 'contact_pairs_3d.csv')
df_pairs_3d.to_csv(pairs_3d_csv, index=False)
logger.info(f'3D pair table saved -> {pairs_3d_csv}')

type_counts_3d = pd.Series(type_of_3d.values()).value_counts()
df_summary_3d, valid_types_3d = aggregate_contacts(
    df_pairs_3d, 'cell_id_3d_a', 'cell_id_3d_b', 'type_a', 'type_b',
    type_counts_3d, MIN_CELLS,
)
summary_3d_csv = os.path.join(OUT_DIR, 'contact_summary_3d.csv')
df_summary_3d.to_csv(summary_3d_csv, index=False)
logger.info(f'  Valid 3D types (>= {MIN_CELLS} cells): {valid_types_3d}')
logger.info(f'3D summary saved -> {summary_3d_csv}')

# =============================================================================
# 2D PASS (independent per slice — no cross-slice information used at all)
# =============================================================================
df_summary_2d = None
if args.skip_2d:
    logger.info('--skip_2d given — skipping the 2D pass.')
elif not os.path.exists(WARPED_DIR):
    logger.warning(f'2D pass skipped — warped mask folder not found: {WARPED_DIR}')
elif not os.path.exists(TYPED_2D_CSV):
    logger.warning(f'2D pass skipped — 2D typed CSV not found: {TYPED_2D_CSV} '
                   f'(run assign_phenotypes.py first).')
else:
    mask_files = sorted(glob.glob(os.path.join(WARPED_DIR, '*DAPI*_warped.tif')),
                        key=get_slice_id)
    if not mask_files:
        logger.warning(f'2D pass skipped — no warped DAPI masks found in {WARPED_DIR}')
    else:
        logger.info(f'2D pass: {len(mask_files)} warped mask slices found.')
        df_typed_2d = pd.read_csv(TYPED_2D_CSV)
        type_of_2d  = {(int(r.slice_id), int(r.cell_id)): r.cell_type
                       for r in df_typed_2d.itertuples()}

        t0 = time.time()
        pair_rows_2d, n_untyped_2d = [], 0
        for f in mask_files:
            slice_id = get_slice_id(f)
            mask = tifffile.imread(f)
            if mask.ndim != 2:
                mask = mask.squeeze()
            mask = mask.astype(np.uint32)

            lo_s, hi_s, counts_s = find_adjacent_pairs(
                mask, spacing=(PIXEL_SIZE_XY_UM, PIXEL_SIZE_XY_UM),
                contact_gap_um=CONTACT_GAP_UM,
            )
            for a, b, n in zip(lo_s, hi_s, counts_s):
                ta = type_of_2d.get((slice_id, int(a)))
                tb = type_of_2d.get((slice_id, int(b)))
                if ta is None or tb is None:
                    n_untyped_2d += 1
                    continue
                # Composite ids so the same numeric CellPose id in different
                # slices is never confused with itself downstream — raw
                # per-slice cell_id is only unique WITHIN one slice.
                pair_rows_2d.append({
                    'slice_id': slice_id,
                    'cell_id_2d_a': f'{slice_id}_{a}', 'cell_id_2d_b': f'{slice_id}_{b}',
                    'type_a': ta, 'type_b': tb,
                    'n_boundary_voxel_pairs': int(n),
                })
        logger.info(f'  2D pass done in {time.time() - t0:.1f}s '
                   f'({len(pair_rows_2d)} typed pairs, {n_untyped_2d} untyped excluded).')

        df_pairs_2d = pd.DataFrame(pair_rows_2d)
        pairs_2d_csv = os.path.join(OUT_DIR, 'contact_pairs_2d.csv')
        df_pairs_2d.to_csv(pairs_2d_csv, index=False)
        logger.info(f'2D pair table saved -> {pairs_2d_csv}')

        # Denominator: total 2D instances per type across ALL slices — same
        # per-instance convention nn_distances_2d.csv / cell_density_2d.csv
        # already use (no 3D identity involved here).
        type_counts_2d = df_typed_2d['cell_type'].value_counts()
        df_summary_2d, valid_types_2d = aggregate_contacts(
            df_pairs_2d, 'cell_id_2d_a', 'cell_id_2d_b', 'type_a', 'type_b',
            type_counts_2d, MIN_CELLS,
        )
        summary_2d_csv = os.path.join(OUT_DIR, 'contact_summary_2d.csv')
        df_summary_2d.to_csv(summary_2d_csv, index=False)
        logger.info(f'  Valid 2D types (>= {MIN_CELLS} instances): {valid_types_2d}')
        logger.info(f'2D summary saved -> {summary_2d_csv}')

# ─────────────────────────────────────────────────────────────────────────────
# FINAL LOG — same-type contact rate, 2D vs 3D, side by side where both exist
# ─────────────────────────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'Done.  Core: {TARGET_CORE}  |  contact_gap_um={CONTACT_GAP_UM}')
same_3d = df_summary_3d[df_summary_3d['src_type'] == df_summary_3d['tgt_type']]
same_2d = (df_summary_2d[df_summary_2d['src_type'] == df_summary_2d['tgt_type']]
          if df_summary_2d is not None else None)
for _, row in same_3d.iterrows():
    line = (f'  {row["src_type"]:<12}: 3D={row["mean_contacts_per_cell"]:.2f} '
           f'contacts/cell ({row["pct_cells_with_contact"]:.1f}% of '
           f'{row["n_src_cells"]} cells)')
    if same_2d is not None:
        r2 = same_2d[same_2d['src_type'] == row['src_type']]
        if len(r2):
            line += (f'   |   2D={r2["mean_contacts_per_cell"].iloc[0]:.2f} '
                     f'contacts/cell ({r2["pct_cells_with_contact"].iloc[0]:.1f}% of '
                     f'{int(r2["n_src_cells"].iloc[0])} instances)')
    logger.info(line)
logger.info(f'  3D pairs/summary : {pairs_3d_csv}  /  {summary_3d_csv}')
if df_summary_2d is not None:
    logger.info(f'  2D pairs/summary : {pairs_2d_csv}  /  {summary_2d_csv}')
logger.info('=' * 60)