"""Executes cell phenotyping on native 3D CellPose masks using the denoised registered volume.

Extracts single morphological and marker intensity measurements per 3D cell across 
the entire Z-volume. Thresholding is applied directly to the full per-core population 
of 3D-cell medians, bypassing the cross-slice consensus step required in 2D pipelines. 
Membrane-capture dilation is strictly restricted to in-plane (XY) expansion per 
Z-slice to prevent artificial signal bleed across physically thick anisotropic sections. 

Outputs maintain the one-to-one relationship between a 3D mask and its phenotype, 
exporting morphological markers (`z_extent_slices`, `possible_merge_profile`) for 
conditional downstream review rather than applying rigid upstream filters.

Example:
    $ python phenotype_cells_3d.py \
        --core_name Core_01 \
        --min_volume_um3 65.0 \
        --bic_threshold 6.0 \
        --z_thickness_um 4.0 \
        --denoised_dir_name Denoised_Valis \
        --mask_dir_name CellPose_DAPI_3D_native_Valis \
        --output_dir_name Phenotypes3D_Valis

Args (CLI):
    --core_name (str): Identifier for the core being processed.
    --min_volume_um3 (float, optional): Minimum cell volume threshold. Defaults to 65.0.
    --bic_threshold (float, optional): Threshold for Bayesian Information Criterion. Defaults to 6.0.
    --z_thickness_um (float, optional): Physical thickness of Z-slices. Defaults to 4.0.
    --denoised_dir_name (str, optional): Target directory for denoised volume inputs. Defaults to 'Denoised_Valis'.
    --mask_dir_name (str, optional): Target directory for 3D mask inputs. Defaults to 'CellPose_DAPI_3D_native_Valis'.
    --output_dir_name (str, optional): Target directory for CSV outputs. Defaults to 'Phenotypes3D_Valis'.

Outputs:
    A CSV mapping one measurement per 3D cell, utilizing the following schema:
    `core`, `cell_id_3d`, `n_voxels`, `volume_um3`, `z_min`, `z_max`, `z_extent_slices`, 
    `centroid_x`, `centroid_y`, `centroid_z`, `possible_merge_profile`, 
    `median_<CH>`, `pos_<CH>`, `thresh_<CH>`, `thresh_type_<CH>`.

Notes:
    - The `*_dir_name` parameterization isolates this script from specific upstream 
      registration toolchains.
    - No `--reg_stats_csv` equivalent is utilized. The 3D masks are natively 
      voxel-aligned to the denoised volume from which they were segmented, eliminating 
      the slice-reordering ambiguity inherent to independently-segmented 2D pipelines.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import brentq
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from skimage.segmentation import expand_labels

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# Duplicated from phenotype_cells.py rather than imported — that script runs
# argparse.parse_args() at module import time, so importing it directly would
# consume/collide with this script's own CLI args. Keep these in sync by hand
# (or factor both into a shared module later if they drift).
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
MARKER_CHANNELS  = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK']
CHANNEL_IDX      = {name: i for i, name in enumerate(CHANNEL_NAMES)}
PIXEL_SIZE_XY_UM = 0.4961

DEFAULT_BIC_THRESHOLD    = 6.0
DEFAULT_Z_THICKNESS_UM   = 4.0
DEFAULT_MIN_VOLUME_UM3   = 65.0   # matches compare_cellpose3d_vs_linked3d.py's
                                  # MIN_NUCLEUS_VOLUME_UM3 — keep these two in sync.
MEMBRANE_EXPAND_PX       = 4      # in-plane only; matches phenotype_cells.py's 4 px (~2 um)

PREFER_OTSU = {'CK'}
MANUAL_THRESHOLDS: dict[str, float] = {
    'CD31': 200.0,   # ← same caveat as phenotype_cells.py: set from visual inspection
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Phenotype cells on native 3D CellPose masks.'
)
parser.add_argument('--core_name',       type=str, required=True)
parser.add_argument('--min_volume_um3',  type=float, default=DEFAULT_MIN_VOLUME_UM3,
                    help=f'Minimum nucleus volume in um^3 (default: {DEFAULT_MIN_VOLUME_UM3}).')
parser.add_argument('--bic_threshold',   type=float, default=DEFAULT_BIC_THRESHOLD,
                    help=f'ΔBIC required to prefer 2-component GMM (default: {DEFAULT_BIC_THRESHOLD}).')
parser.add_argument('--z_thickness_um',  type=float, default=DEFAULT_Z_THICKNESS_UM,
                    help=f'Physical thickness per z-slice, um (default: {DEFAULT_Z_THICKNESS_UM}).')
parser.add_argument('--denoised_dir_name', type=str, default='Denoised_Valis',
                    help='Folder under DATASPACE containing <CORE>_denoised.ome.tif.')
parser.add_argument('--mask_dir_name',   type=str, default='CellPose_DAPI_3D_native_Valis',
                    help='Folder under DATASPACE containing <CORE>_3D_Cellpose_Masks.tif '
                         '(test_3d_cellpose.py output).')
parser.add_argument('--output_dir_name', type=str, default='Phenotypes3D_Valis',
                    help='Folder under DATASPACE to write phenotype output into.')
args = parser.parse_args()

TARGET_CORE   = args.core_name
BIC_THRESHOLD = args.bic_threshold
Z_THICKNESS_UM = args.z_thickness_um

DENOISED_VOL = os.path.join(
    config.DATASPACE, args.denoised_dir_name, TARGET_CORE,
    f'{TARGET_CORE}_denoised.ome.tif',
)
MASK_3D_PATH = os.path.join(
    config.DATASPACE, args.mask_dir_name, TARGET_CORE,
    f'{TARGET_CORE}_3D_Cellpose_Masks.tif',
)
OUTPUT_DIR = os.path.join(config.DATASPACE, args.output_dir_name, TARGET_CORE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD FITTING — identical logic to phenotype_cells.py, duplicated (see
# module docstring for why). Operates on whatever 1-value-per-cell array it's
# given; here that's one median per 3D cell rather than one median per 2D
# per-slice cell.
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_intersection(gmm: GaussianMixture) -> float | None:
    means   = gmm.means_.flatten()
    sigmas  = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    neg_idx = int(np.argmin(means))
    pos_idx = int(np.argmax(means))
    mu_neg, sig_neg, w_neg = means[neg_idx], sigmas[neg_idx], weights[neg_idx]
    mu_pos, sig_pos, w_pos = means[pos_idx], sigmas[pos_idx], weights[pos_idx]

    separation = (mu_pos - mu_neg) / max(min(sig_neg, sig_pos), 1e-9)
    if separation < 0.5:
        return None

    def delta_pdf(x):
        g_neg = w_neg * np.exp(-0.5 * ((x - mu_neg) / sig_neg) ** 2) / sig_neg
        g_pos = w_pos * np.exp(-0.5 * ((x - mu_pos) / sig_pos) ** 2) / sig_pos
        return g_neg - g_pos

    try:
        root = float(brentq(delta_pdf, mu_neg, mu_pos, xtol=1e-6, maxiter=200))
    except (ValueError, RuntimeError):
        root = float((mu_neg + mu_pos) / 2.0)

    if root <= mu_neg + 0.5 * sig_neg:
        return None
    return root


def threshold_from_cell_medians(cell_medians: np.ndarray, marker_name: str = 'Marker') -> tuple[float, str]:
    """Same strategy/order as phenotype_cells.py: GMM bimodal -> otsu_nz -> p75_fallback."""
    cell_medians = np.asarray(cell_medians, dtype=np.float64)

    if marker_name in MANUAL_THRESHOLDS:
        thresh = float(MANUAL_THRESHOLDS[marker_name])
        logger.info(f'  [{marker_name}] Using manual threshold: {thresh:.1f}')
        return thresh, 'manual'

    if len(cell_medians) < 10:
        return float(np.percentile(cell_medians, 75)), 'p75_fallback_too_few_cells'

    if marker_name not in PREFER_OTSU:
        X = cell_medians.reshape(-1, 1)
        try:
            gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
            gmm2 = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(X)
            delta_bic = gmm1.bic(X) - gmm2.bic(X)
        except Exception as exc:
            logger.warning(f'  [{marker_name}] GMM fit failed ({exc}) — using otsu_nz.')
            delta_bic = 0.0
            gmm2 = None

        if delta_bic > BIC_THRESHOLD and gmm2 is not None:
            intersection = _gmm_intersection(gmm2)
            mu_pos = float(gmm2.means_.max())
            mu_neg = float(gmm2.means_.min())
            min_meaningful = max(mu_pos * 0.05, (mu_pos - mu_neg) * 0.10)
            if intersection is not None and intersection > min_meaningful:
                return intersection, 'gmm_bimodal'

    nz = cell_medians[cell_medians > 0]
    if len(nz) >= 10:
        n_bins = min(256, len(nz) // 2)
        counts, edges = np.histogram(nz, bins=n_bins)
        centres = (edges[:-1] + edges[1:]) / 2.0
        total = counts.sum()
        prob  = counts / total
        w0    = np.cumsum(prob)
        mu0s  = np.cumsum(prob * centres)
        w1    = 1.0 - w0
        mu_t  = float(np.sum(prob * centres))
        with np.errstate(invalid='ignore', divide='ignore'):
            mu0 = np.where(w0 > 0, mu0s / w0, 0.0)
            mu1 = np.where(w1 > 0, (mu_t - mu0s) / w1, 0.0)
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        otsu_val = float(centres[np.argmax(between_var)])
        if otsu_val > float(np.percentile(nz, 10)):
            return otsu_val, 'otsu_nz'

    thresh = float(np.percentile(cell_medians, 75))
    logger.warning(
        f'  [{marker_name}] No bimodal structure found in cell medians. '
        f'Using p75 fallback ({thresh:.1f}).'
    )
    return thresh, 'p75_fallback'


# ─────────────────────────────────────────────────────────────────────────────
# MERGE-PROFILE HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

def _detect_merge_profile(area_profile: np.ndarray) -> bool:
    """
    Crude "dumbbell" detector on a cell's cross-sectional-area-vs-z profile.

    A genuinely elongated nucleus tends to taper smoothly (one peak). Two
    touching nuclei that got fused into one 3D label typically show two area
    maxima with a pinched neck between them. Flags True if >=2 peaks are
    found and the trough between the two largest peaks dips to <=70% of the
    smaller peak's height. This is a starting heuristic, not a validated
    classifier — check flagged cells visually before trusting it broadly.
    """
    if len(area_profile) < 3:
        return False
    peaks, _ = find_peaks(area_profile)
    if len(peaks) < 2:
        return False
    # Two largest peaks by height
    peak_heights = area_profile[peaks]
    top2_idx = peaks[np.argsort(peak_heights)[-2:]]
    lo_peak, hi_peak = sorted(top2_idx)
    trough = float(area_profile[lo_peak:hi_peak + 1].min())
    smaller_peak_height = float(min(area_profile[lo_peak], area_profile[hi_peak]))
    if smaller_peak_height <= 0:
        return False
    return trough <= 0.7 * smaller_peak_height


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(f'Phenotyping (3D) — core: {TARGET_CORE}')
    logger.info(f'Denoised volume : {DENOISED_VOL}')
    logger.info(f'3D mask         : {MASK_3D_PATH}')
    logger.info(f'BIC threshold   : {BIC_THRESHOLD}')
    logger.info(f'Z thickness     : {Z_THICKNESS_UM} um')

    if not os.path.exists(DENOISED_VOL):
        logger.error(f'Denoised volume not found: {DENOISED_VOL}')
        sys.exit(1)
    if not os.path.exists(MASK_3D_PATH):
        logger.error(f'3D mask not found: {MASK_3D_PATH}')
        sys.exit(1)

    logger.info('Loading denoised volume ...')
    vol = tifffile.imread(DENOISED_VOL)
    if vol.ndim == 4:
        if vol.shape[0] == len(CHANNEL_NAMES) and vol.shape[1] != len(CHANNEL_NAMES):
            vol = np.moveaxis(vol, 0, 1)
            logger.info('  Volume reordered CZYX → ZCYX.')
    elif vol.ndim == 3:
        logger.warning('  Volume is 3-D — treating as single-slice CYX.')
        vol = vol[np.newaxis]
    n_slices, n_channels, H, W = vol.shape
    logger.info(f'Volume shape (ZCYX): Z={n_slices}  C={n_channels}  H={H}  W={W}')
    if n_channels != len(CHANNEL_NAMES):
        logger.warning(
            f'Expected {len(CHANNEL_NAMES)} channels, got {n_channels}. '
            f'Verify CHANNEL_NAMES matches your data.'
        )

    logger.info('Loading 3D mask ...')
    mask = tifffile.imread(MASK_3D_PATH).astype(np.uint32)
    if mask.ndim != 3:
        logger.error(f'Expected a 3-D (Z,H,W) mask, got shape {mask.shape}')
        sys.exit(1)
    if mask.shape != (n_slices, H, W):
        logger.error(f'Mask shape {mask.shape} != volume spatial shape {(n_slices, H, W)}. '
                     f'Confirm the mask was segmented from THIS denoised volume.')
        sys.exit(1)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        logger.error('No cells in 3D mask — no CSV written.')
        sys.exit(1)
    label_list = labels.tolist()
    logger.info(f'{len(label_list)} raw 3D labels found.')

    # ── Geometry: volume, z-extent, centroid (all from the RAW mask) ─────────
    n_voxels = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int64)
    volume_um3 = n_voxels * (PIXEL_SIZE_XY_UM ** 2) * Z_THICKNESS_UM

    centroids = ndi.center_of_mass(np.ones_like(mask), mask, label_list)  # list of (z, y, x)
    centroids = np.array(centroids)

    # Per-slice area profile for every raw label, built once (cheap: Z bincount passes).
    max_label = int(mask.max())
    area_profile = np.zeros((max_label + 1, n_slices), dtype=np.int32)
    for z in range(n_slices):
        counts = np.bincount(mask[z].ravel(), minlength=max_label + 1)
        area_profile[:len(counts), z] = counts[:max_label + 1]

    # ── Volume filter ─────────────────────────────────────────────────────────
    keep_mask = volume_um3 >= args.min_volume_um3
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        logger.info(f'Dropping {n_dropped}/{len(label_list)} cells below '
                    f'{args.min_volume_um3} um^3 (min_volume_um3).')
    label_list  = [l for l, k in zip(label_list, keep_mask) if k]
    n_voxels    = n_voxels[keep_mask]
    volume_um3  = volume_um3[keep_mask]
    centroids   = centroids[keep_mask]
    if len(label_list) == 0:
        logger.error('No cells left after min_volume_um3 filter — no CSV written.')
        sys.exit(1)

    z_min_list, z_max_list, z_extent_list, merge_flag_list = [], [], [], []
    for lbl in label_list:
        profile = area_profile[lbl]
        z_present = np.nonzero(profile)[0]
        z_min, z_max = int(z_present.min()), int(z_present.max())
        z_min_list.append(z_min)
        z_max_list.append(z_max)
        z_extent_list.append(z_max - z_min + 1)
        merge_flag_list.append(_detect_merge_profile(profile[z_min:z_max + 1]))

    df = pd.DataFrame({
        'core':        TARGET_CORE,
        'cell_id_3d':  label_list,
        'n_voxels':    n_voxels,
        'volume_um3':  np.round(volume_um3, 2),
        'z_min':       z_min_list,
        'z_max':       z_max_list,
        'z_extent_slices': z_extent_list,
        'centroid_z':  np.round(centroids[:, 0], 2),
        'centroid_y':  np.round(centroids[:, 1], 2),
        'centroid_x':  np.round(centroids[:, 2], 2),
        'possible_merge_profile': merge_flag_list,
    })
    logger.info(f'{len(df)} cells after min_volume_um3 filter. '
               f'Median z_extent_slices = {df["z_extent_slices"].median():.1f}, '
               f'possible_merge_profile = {int(df["possible_merge_profile"].sum())}.')

    # ── Per-marker intensity + threshold ──────────────────────────────────────
    # In-plane-only expansion: dilate each z-slice of the mask independently in
    # 2D (distance=MEMBRANE_EXPAND_PX), never across z. See module docstring.
    logger.info(f'Expanding mask {MEMBRANE_EXPAND_PX} px in-plane (per slice) for membrane capture ...')
    cell_mask = np.stack([expand_labels(mask[z], distance=MEMBRANE_EXPAND_PX) for z in range(n_slices)])

    label_arr = np.array(label_list)
    flat_labels = cell_mask.ravel()
    sort_idx    = np.argsort(flat_labels, kind='stable')
    sorted_lbl  = flat_labels[sort_idx]

    for ch_name in MARKER_CHANNELS:
        ch_idx = CHANNEL_IDX[ch_name]
        ch_vol = vol[:, ch_idx].astype(np.float32)          # (Z, H, W)
        flat_px = ch_vol.ravel()[sort_idx]

        boundaries = np.searchsorted(sorted_lbl, label_arr)
        end_bounds = np.searchsorted(sorted_lbl, label_arr, side='right')
        cell_medians = np.array([
            float(np.median(flat_px[s:e])) if e > s else 0.0
            for s, e in zip(boundaries, end_bounds)
        ], dtype=np.float32)

        df[f'median_{ch_name}'] = np.round(cell_medians, 2)

        thresh, t_type = threshold_from_cell_medians(cell_medians, marker_name=ch_name)
        logger.info(
            f'  [{ch_name}] threshold = {thresh:.1f}  [{t_type}]  '
            f'({int((cell_medians >= thresh).sum())}/{len(cell_medians)} positive)'
        )
        df[f'thresh_{ch_name}']      = round(thresh, 2)
        df[f'thresh_type_{ch_name}'] = t_type
        df[f'pos_{ch_name}']         = (cell_medians >= thresh).astype(np.uint8)

    csv_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes_3d.csv')
    df.to_csv(csv_path, index=False)

    logger.info('=' * 60)
    logger.info(f'Done.  Core: {TARGET_CORE}  |  Total cells: {len(df)}')
    logger.info(f'CSV: {csv_path}')
    for ch in MARKER_CHANNELS:
        n_pos = int(df[f'pos_{ch}'].sum())
        pct = 100.0 * n_pos / len(df) if len(df) > 0 else 0.0
        logger.info(f'  {ch:8s}: {n_pos:6d} positive ({pct:5.1f}%)  '
                    f'thresh = {df[f"thresh_{ch}"].iloc[0]:.1f}  [{df[f"thresh_type_{ch}"].iloc[0]}]')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()