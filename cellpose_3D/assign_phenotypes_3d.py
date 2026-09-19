"""Assigns phenotypes to 3D CellPose nuclei and flags potential spatial artifacts.

Applies a marker codebook to 3D nuclei measurements to assign cell types. 
Instead of filtering anomalous cells, this module flags nuclei exhibiting 
unexpected z-extents or area-vs-z profiles (e.g., potential merged pairs or 
elongation artifacts) for downstream review.

Example:
    $ python assign_phenotypes_3d.py \
        --core_name Core_01 \
        --phenotype3d_dir_name Phenotypes3D_Valis \
        --output_dir_name Phenotypes3D_Valis \
        --low_signal_ratio 0.5

Notes:
    - Architectural Scope: Unlike the 2D pipeline, this module evaluates single 
      3D volumetric measurements per cell. Consequently, majority voting and 
      fields like `type_confidence` or `n_slices_voted` are structurally 
      inapplicable and intentionally omitted. Do not mock `type_confidence=1.0` 
      to satisfy upstream schemas.
    - Review Flags vs. Filtering: To preserve true tissue density for downstream 
      spatial graphs, morphological outliers are flagged (`review_flag=True`) 
      rather than dropped. Outlier detection uses `EXPECTED_Z_EXTENT_SLICES` as 
      a baseline prior, accommodating natural size variations (e.g., large tumor 
      nuclei vs. small lymphocytes). Exclusions remain an explicit downstream opt-in.
    - Unknown Classifications: The `unknown_subtype` field provides a heuristic split 
      between globally uninformative (weak) signals and clean profiles that lack 
      a codebook rule. This is not a calibrated confidence score and requires 
      manual quality control validation.

Outputs:
    <core>_3d_typed.csv: Contains `cell_type`, `unknown_subtype`, `review_flag`, 
    and `review_reason`, alongside the base 3D measurement outputs.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

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
    description='Assign cell types to native 3D CellPose nuclei via codebook.'
)
parser.add_argument('--core_name', type=str, required=True,
                    help='TMA core identifier, e.g. Core_01.')
parser.add_argument('--phenotype3d_dir_name', type=str, default='Phenotypes3D_Valis',
                    help='Folder under DATASPACE containing <CORE>_phenotypes_3d.csv '
                         '(phenotype_cells_3d.py output).')
parser.add_argument('--output_dir_name', type=str, default='Phenotypes3D_Valis',
                    help='Folder under DATASPACE to write typed output into.')
parser.add_argument('--low_signal_ratio', type=float, default=0.5,
                    help="For 'Unknown' cells: max(median_ch / thresh_ch) below this ratio "
                         "across all markers → 'Unclassified_LowSignal'; at or above → "
                         "'Unclassified_Novel' (default: 0.5).")
args = parser.parse_args()

TARGET_CORE = args.core_name

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
PHENOTYPE_DIR = os.path.join(config.DATASPACE, args.phenotype3d_dir_name, TARGET_CORE)
OUTPUT_DIR    = os.path.join(config.DATASPACE, args.output_dir_name,      TARGET_CORE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHENOTYPE_CSV = os.path.join(PHENOTYPE_DIR, f'{TARGET_CORE}_phenotypes_3d.csv')
if not os.path.exists(PHENOTYPE_CSV):
    logger.error(f'Phenotype CSV not found: {PHENOTYPE_CSV}')
    sys.exit(1)

# -----------------------------------------------------------------------------
# MARKER CHANNELS + CODEBOOK
# Duplicated from assign_phenotypes.py — same reasoning as
# phenotype_cells_3d.py's duplication of phenotype_cells.py's threshold
# fitting: importing that module directly would trigger its own
# argparse.parse_args() against this script's CLI args. Keep these two
# CODEBOOKs in sync by hand.
# -----------------------------------------------------------------------------
MARKER_CHANNELS = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK']

CODEBOOK = [
    ('Tumour',      lambda r: r['pos_CK']    == 1),
    ('T_cell',      lambda r: r['pos_CD3']   == 1 and r['pos_CK'] == 0),
    ('Macrophage',  lambda r: r['pos_CD163'] == 1 and r['pos_CK'] == 0),
    ('Endothelial', lambda r: r['pos_CD31']  == 1 and r['pos_CK'] == 0),
    ('Neural',      lambda r: (r['pos_GAP43'] == 1 or r['pos_NFP'] == 1)
                               and r['pos_CK'] == 0),
]
UNKNOWN_LABEL = 'Unknown'


def apply_codebook(row: pd.Series) -> str:
    for label, rule in CODEBOOK:
        try:
            if rule(row):
                return label
        except KeyError:
            continue
    return UNKNOWN_LABEL


def unknown_subtype(row: pd.Series, low_signal_ratio: float) -> str:
    """
    Split 'Unknown' cells into low-signal (uninformative) vs novel (clean
    signal, just no codebook rule matches) — see module docstring caveat.
    Only meaningful for rows already labelled 'Unknown'; returns '' otherwise.
    """
    if row['cell_type'] != UNKNOWN_LABEL:
        return ''
    ratios = []
    for ch in MARKER_CHANNELS:
        thresh = row.get(f'thresh_{ch}', 0.0)
        median = row.get(f'median_{ch}', 0.0)
        if thresh and thresh > 0:
            ratios.append(median / thresh)
    max_ratio = max(ratios) if ratios else 0.0
    return 'Unclassified_LowSignal' if max_ratio < low_signal_ratio else 'Unclassified_Novel'


# -----------------------------------------------------------------------------
# SLICE-SPAN / MERGE REVIEW FLAG
# Priors from team discussion — refine per-core once real z_extent_slices
# distributions per type are available. (lo, hi) inclusive, in slice counts.
# -----------------------------------------------------------------------------
EXPECTED_Z_EXTENT_SLICES = {
    'Tumour':      (1, 6),   # cancer nuclei: more size/shape variability, longer tail
    'T_cell':      (1, 3),   # lymphocytes: consistently small per team discussion
    'Macrophage':  (1, 4),
    'Endothelial': (1, 4),
    'Neural':      (1, 4),
    'Unknown':     (1, 6),   # unclassified — err permissive until typed
}


def review_flags(row: pd.Series) -> tuple[bool, str]:
    reasons = []
    lo, hi = EXPECTED_Z_EXTENT_SLICES.get(row['cell_type'], (1, 6))
    if not (lo <= row['z_extent_slices'] <= hi):
        reasons.append('z_extent_outlier')
    if bool(row.get('possible_merge_profile', False)):
        reasons.append('merge_profile')
    return (len(reasons) > 0), ';'.join(reasons)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
logger.info(f'Loading 3D phenotype table: {PHENOTYPE_CSV}')
df = pd.read_csv(PHENOTYPE_CSV)

required_pos = [f'pos_{ch}' for ch in MARKER_CHANNELS]
missing = [c for c in required_pos if c not in df.columns]
if missing:
    logger.error(f'Phenotype table missing positivity columns: {missing}')
    sys.exit(1)
for req_col in ('z_extent_slices', 'possible_merge_profile'):
    if req_col not in df.columns:
        logger.error(f"Phenotype table missing '{req_col}' — was it produced by "
                     f"phenotype_cells_3d.py?")
        sys.exit(1)

for col in required_pos:
    df[col] = df[col].fillna(0).astype(int)

logger.info('Applying codebook to 3D cells ...')
df['cell_type'] = df.apply(apply_codebook, axis=1)
df['unknown_subtype'] = df.apply(lambda r: unknown_subtype(r, args.low_signal_ratio), axis=1)

type_counts = df['cell_type'].value_counts()
logger.info('3D cell type distribution:')
for ct, n in type_counts.items():
    pct = 100 * n / len(df)
    logger.info(f'  {ct:<12s}: {n:6d}  ({pct:.1f} %)')

n_unknown = int((df['cell_type'] == UNKNOWN_LABEL).sum())
if n_unknown:
    sub_counts = df.loc[df['cell_type'] == UNKNOWN_LABEL, 'unknown_subtype'].value_counts()
    logger.info(f'  Unknown breakdown ({n_unknown} cells):')
    for st, n in sub_counts.items():
        logger.info(f'    {st:<24s}: {n:6d}')

logger.info('Computing review flags (z-extent outlier / merge profile) ...')
flags = df.apply(review_flags, axis=1)
df['review_flag']   = [f[0] for f in flags]
df['review_reason'] = [f[1] for f in flags]

n_flagged = int(df['review_flag'].sum())
logger.info(f'  {n_flagged}/{len(df)} cells flagged for review '
           f'({100 * n_flagged / len(df):.1f}%). Not dropped from output — '
           f'exclude explicitly downstream if a given analysis needs it.')
if n_flagged:
    reason_counts = df.loc[df['review_flag'], 'review_reason'].str.split(';').explode().value_counts()
    for reason, n in reason_counts.items():
        logger.info(f'    {reason:<20s}: {n:6d}')

typed_3d_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_3d_typed.csv')
df.to_csv(typed_3d_path, index=False)

logger.info('=' * 60)
logger.info(f'Done.  Core: {TARGET_CORE}')
logger.info(f'  3D typed : {typed_3d_path}')
logger.info('=' * 60)