"""
assign_phenotypes.py
====================
Assign cell types to CellPose-segmented nuclei by applying a marker codebook
to the per-cell binary positivity calls produced by phenotype_cells.py, then
propagate cell type labels into 3D cell identities produced by analyse_3d_cells.py.

Linkage
-------
The join key between the two upstream tables is the CellPose 2D nucleus ID:

    phenotypes table  : (core, slice_id, cell_id)
                                  ↕  join on slice_id + cell_id = cell_id_2d
    2D→3D map table   : (slice_id, cell_id_2d, cell_id_3d)

This produces a table where every 2D nucleus instance has both its marker
positivity profile and its 3D cell identity. A codebook lookup then assigns
a cell type to each 2D instance, and a majority vote across slices yields a
consensus cell type per 3D cell.

Codebook priority (ordered, first match wins)
---------------------------------------------
    1. Tumour cell     : CK+
    2. T cell          : CD3+  & CK-
    3. Macrophage      : CD163+ & CK-
    4. Endothelial     : CD31+  & CK-
    5. Neural/perineural: (GAP43+ | NFP+) & CK-
    6. Unknown         : none of the above

Outputs
-------
    <core>_phenotypes_typed.csv   — 2D-level table with cell_type column added
    <core>_3d_typed.csv           — 3D-level table with consensus cell_type + composition

Usage
-----
    python assign_phenotypes.py --core_name Core_01 [--dapi_channel CD3]
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from collections import Counter

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
    description='Assign cell types via codebook and link to 3D cell identities.'
)
parser.add_argument('--core_name',    type=str, required=True,
                    help='TMA core identifier, e.g. Core_01.')
parser.add_argument('--linking_channel', type=str, default='DAPI',
                    help='Channel used for 3D linking in analyse_3d_cells.py (default: DAPI). '
                         'Determines which 2D→3D map CSV is loaded.')
parser.add_argument('--min_confidence', type=float, default=0.5,
                    help='Minimum fraction of 2D slices agreeing on a cell type '
                         'for the 3D consensus to be accepted (default: 0.5).')
args = parser.parse_args()

TARGET_CORE     = args.core_name
LINK_CHANNEL    = args.linking_channel
MIN_CONFIDENCE  = args.min_confidence

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
PHENOTYPE_DIR = os.path.join(config.DATASPACE, 'Phenotypes',  TARGET_CORE)
LINKING_DIR   = os.path.join(config.DATASPACE, f'CellPose_{LINK_CHANNEL}_3D', TARGET_CORE)
OUTPUT_DIR    = os.path.join(config.DATASPACE, 'Phenotypes',  TARGET_CORE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHENOTYPE_CSV = os.path.join(PHENOTYPE_DIR, f'{TARGET_CORE}_phenotypes.csv')
MAP_CSV       = os.path.join(LINKING_DIR,   f'{TARGET_CORE}_{LINK_CHANNEL}_2d_to_3d_map.csv')
STATS_3D_CSV  = os.path.join(LINKING_DIR,   f'{TARGET_CORE}_{LINK_CHANNEL}_3d_stats.csv')

for path, label in [(PHENOTYPE_CSV, 'Phenotype CSV'),
                    (MAP_CSV,       '2D→3D map CSV'),
                    (STATS_3D_CSV,  '3D stats CSV')]:
    if not os.path.exists(path):
        logger.error(f'{label} not found: {path}')
        sys.exit(1)

# -----------------------------------------------------------------------------
# MARKER CHANNELS expected in the phenotype table
# -----------------------------------------------------------------------------
MARKER_CHANNELS = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK']

# -----------------------------------------------------------------------------
# CODEBOOK
# Ordered list of (cell_type_label, rule_function).
# Rules receive a row (pd.Series) with boolean pos_<marker> columns.
# First matching rule wins. 'Unknown' is the fallback.
# -----------------------------------------------------------------------------
CODEBOOK = [
    ('Tumour',          lambda r: r['pos_CK']   == 1),
    ('T_cell',          lambda r: r['pos_CD3']  == 1 and r['pos_CK'] == 0),
    ('Macrophage',      lambda r: r['pos_CD163']== 1 and r['pos_CK'] == 0),
    ('Endothelial',     lambda r: r['pos_CD31'] == 1 and r['pos_CK'] == 0),
    ('Neural',          lambda r: (r['pos_GAP43'] == 1 or r['pos_NFP'] == 1)
                                   and r['pos_CK'] == 0),

]
UNKNOWN_LABEL = 'Unknown'


def apply_codebook(row: pd.Series) -> str:
    """Return the first matching cell type label for a single cell row."""
    for label, rule in CODEBOOK:
        try:
            if rule(row):
                return label
        except KeyError:
            continue
    return UNKNOWN_LABEL


def consensus_cell_type(types: pd.Series, min_confidence: float) -> tuple:
    """
    Majority-vote consensus over 2D slice-level cell type calls for one 3D cell.

    Returns
    -------
    (consensus_type : str, confidence : float)
        confidence is the fraction of slices agreeing on the winning label.
        If the winning fraction is below min_confidence, returns ('Ambiguous', confidence).
    """
    if len(types) == 0:
        return UNKNOWN_LABEL, 0.0
    counts    = Counter(types)
    top_type, top_count = counts.most_common(1)[0]
    confidence = top_count / len(types)
    if confidence < min_confidence:
        return 'Ambiguous', confidence
    return top_type, confidence


# -----------------------------------------------------------------------------
# LOAD TABLES
# -----------------------------------------------------------------------------
logger.info(f'Loading phenotype table: {PHENOTYPE_CSV}')
df_pheno = pd.read_csv(PHENOTYPE_CSV)

logger.info(f'Loading 2D→3D map: {MAP_CSV}')
df_map = pd.read_csv(MAP_CSV)

logger.info(f'Loading 3D stats: {STATS_3D_CSV}')
df_3d = pd.read_csv(STATS_3D_CSV)

# Validate expected columns
required_pos = [f'pos_{ch}' for ch in MARKER_CHANNELS]
missing = [c for c in required_pos if c not in df_pheno.columns]
if missing:
    logger.error(f'Phenotype table missing positivity columns: {missing}')
    sys.exit(1)

# -----------------------------------------------------------------------------
# STEP 1 — APPLY CODEBOOK TO EACH 2D NUCLEUS
# -----------------------------------------------------------------------------
logger.info('Applying codebook to 2D nuclei ...')

# Ensure positivity columns are integer (0/1) not float
for col in required_pos:
    df_pheno[col] = df_pheno[col].fillna(0).astype(int)

df_pheno['cell_type'] = df_pheno.apply(apply_codebook, axis=1)

type_counts = df_pheno['cell_type'].value_counts()
logger.info('2D cell type distribution:')
for ct, n in type_counts.items():
    pct = 100 * n / len(df_pheno)
    logger.info(f'  {ct:<20s}: {n:6d}  ({pct:.1f} %)')

# -----------------------------------------------------------------------------
# STEP 2 — JOIN 2D PHENOTYPES TO 3D MAP
# Linkage: phenotype (slice_id, cell_id) ↔ map (slice_id, cell_id_2d)
# -----------------------------------------------------------------------------
logger.info('Joining 2D phenotypes to 3D cell map ...')

# Rename map columns for unambiguous merge
df_map_renamed = df_map.rename(columns={'cell_id_2d': 'cell_id'})

# Merge: each 2D nucleus gets its 3D identity.
# slice_z is intentionally excluded from the map columns here — the phenotype
# table already carries slice_z (assigned during denoised-volume processing)
# and including it would produce a slice_z_x / slice_z_y collision.
df_linked = df_pheno.merge(
    df_map_renamed[['slice_id', 'cell_id', 'cell_id_3d']],
    on=['slice_id', 'cell_id'],
    how='left',
)

n_matched   = df_linked['cell_id_3d'].notna().sum()
n_unmatched = df_linked['cell_id_3d'].isna().sum()
logger.info(f'  Matched to 3D identity           : {n_matched}')
logger.info(f'  No 3D match (singleton/filtered) : {n_unmatched}')

# Cells with no 3D match remain as 2D-only; assign cell_id_3d = -1 as sentinel
df_linked['cell_id_3d'] = df_linked['cell_id_3d'].fillna(-1).astype(int)

# Save full 2D typed table
typed_2d_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes_typed.csv')
df_linked.to_csv(typed_2d_path, index=False)
logger.info(f'2D typed phenotype table saved -> {typed_2d_path}')

# -----------------------------------------------------------------------------
# STEP 3 — CONSENSUS CELL TYPE PER 3D CELL
# -----------------------------------------------------------------------------
logger.info('Computing consensus cell type per 3D cell ...')

# Only cells that have a valid 3D identity
df_has_3d = df_linked[df_linked['cell_id_3d'] > 0].copy()

consensus_rows = []
for cell_id_3d, grp in df_has_3d.groupby('cell_id_3d'):
    ct, conf = consensus_cell_type(grp['cell_type'], MIN_CONFIDENCE)

    # Composition: how many slices agreed on each type
    composition = dict(Counter(grp['cell_type']))

    consensus_rows.append({
        'cell_id_3d':        cell_id_3d,
        'cell_type':         ct,
        'type_confidence':   round(conf, 3),
        'n_slices_voted':    len(grp),
        'composition':       str(composition),
    })

df_consensus = pd.DataFrame(consensus_rows)

# Merge consensus labels into the 3D stats table
df_3d_typed = df_3d.merge(df_consensus, on='cell_id_3d', how='left')
df_3d_typed['cell_type']       = df_3d_typed['cell_type'].fillna(UNKNOWN_LABEL)
df_3d_typed['type_confidence'] = df_3d_typed['type_confidence'].fillna(0.0)

# Summary
logger.info('3D consensus cell type distribution:')
type_counts_3d = df_3d_typed['cell_type'].value_counts()
for ct, n in type_counts_3d.items():
    pct = 100 * n / len(df_3d_typed)
    logger.info(f'  {ct:<20s}: {n:6d}  ({pct:.1f} %)')

ambiguous_n = (df_3d_typed['cell_type'] == 'Ambiguous').sum()
if ambiguous_n > 0:
    logger.warning(
        f'  {ambiguous_n} 3D cells labelled Ambiguous '
        f'(slice agreement < {MIN_CONFIDENCE:.0%}). '
        f'Consider reviewing thresholds or increasing min_confidence.'
    )

typed_3d_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_3d_typed.csv')
df_3d_typed.to_csv(typed_3d_path, index=False)
logger.info(f'3D typed cell table saved -> {typed_3d_path}')

logger.info('=' * 60)
logger.info(f'Done.  Core: {TARGET_CORE}')
logger.info(f'  2D typed : {typed_2d_path}')
logger.info(f'  3D typed : {typed_3d_path}')
logger.info('=' * 60)