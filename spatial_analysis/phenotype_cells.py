"""
phenotype_cells.py
==================
Cell phenotyping based on DAPI-segmented nuclei and a denoised registered volume.

Input
-----
  Denoised volume: <CORE>_denoised.ome.tif  (ZCYX float32)
    - Produced by denoise_volume.py.
    - Each channel has had a large-sigma Gaussian subtracted, so background
      pixels sit near zero.  Raw pixel histograms are therefore NOT suitable
      for thresholding — the background mode is squashed against zero and
      neither Otsu nor GMM can find a meaningful negative/positive boundary
      in pixel space.

What we do instead
------------------
  Per cell we compute one number: the median pixel intensity within the
  expanded nucleus mask (nuclear mask dilated 4 px to capture membrane signal).
  This is already the right representation for cell classification.

  Thresholding is done entirely on the distribution of per-cell medians:

    1. Fit a 2-component GMM.  If ΔBIC > BIC_THRESHOLD the distribution is
       genuinely bimodal and the threshold is the Gaussian intersection
       → type 'gmm_bimodal'.

    2. If the intersection is suspiciously close to zero (negative component
       squashed at zero after background subtraction, common for sparse/dim
       markers) → fall through to step 3.

    3. Otsu on non-zero cell medians only → type 'otsu_nz'.
       Non-zero restriction prevents the zero-median mass from dominating.

    4. If neither works (all medians ≈ zero, truly no signal) → p75 of all
       cell medians → type 'p75_fallback'.  This is a known limitation and
       is logged clearly.

  Cross-slice consensus (default on):
    After all slices are processed, one threshold per marker is derived as
    the median of per-slice thresholds across reliable slices.  This is
    especially important for sparse markers (CD31) where a single slice may
    not contain enough positive cells for a stable fit.

Known limitations
-----------------
  - CD31 is very sparse; the per-slice distribution may be effectively
    unimodal in most slices.  The consensus helps but a manually validated
    threshold is more reliable for truly sparse markers.
  - Very dim markers (GAP43, NFP) may yield a soft bimodal where the
    intersection is biologically ambiguous.  Check the QC plots.
  - Thresholds are in the denoised intensity space, not raw camera counts.
    They are not directly comparable across cores unless the denoised volume
    was globally scaled (which denoise_volume.py does per-channel).

Output columns
--------------
    core, slice_z, slice_id, cell_id,
    area_px, area_um2, centroid_x, centroid_y,
    median_<CH>,        — per-cell median denoised intensity (classification value)
    pos_<CH>,           — 1 = positive, 0 = negative
    thresh_<CH>,        — threshold used (consensus after step 4, per-slice before)
    thresh_type_<CH>,   — how the threshold was derived (see types above)
    thresh_per_slice_<CH> — original per-slice threshold before consensus override

Usage
-----
    python phenotype_cells.py --core_name Core_01 [--plot_qc] [--min_area_px 200]
                              [--no_consensus] [--bic_threshold 6.0]
                              [--denoised_dir_name Denoised_bspline]
                              [--mask_dir_name CellPose_DAPI_Warped_Bspline]
                              [--output_dir_name Phenotypes_Bspline]
                              [--reg_stats_csv /path/to/registration_stats.csv]

    The three *_dir_name flags and --reg_stats_csv make this script independent
    of which registration algorithm produced its inputs — point them at whatever
    folder/CSV your current registration run actually wrote.
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
from scipy import ndimage as ndi
from scipy.optimize import brentq
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
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
MARKER_CHANNELS  = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK']
CHANNEL_IDX      = {name: i for i, name in enumerate(CHANNEL_NAMES)}
PIXEL_SIZE_XY_UM = 0.4961

# ΔBIC threshold for bimodal vs unimodal decision.
# ΔBIC = BIC(1-component) − BIC(2-component).
# > 6 is conventionally "strong evidence" for the 2-component model.
DEFAULT_BIC_THRESHOLD = 6.0

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Phenotype cells: measure marker expression in DAPI-segmented nuclei.'
)
parser.add_argument('--core_name',     type=str,   required=True)
parser.add_argument('--min_area_px',   type=int,   default=200,
                    help='Minimum nucleus area in pixels (default: 200).')
parser.add_argument('--plot_qc',       action='store_true',
                    help='Save per-slice QC plots.')
parser.add_argument('--no_consensus',  action='store_true',
                    help='Skip cross-slice consensus threshold step.')
parser.add_argument('--bic_threshold', type=float, default=DEFAULT_BIC_THRESHOLD,
                    help='ΔBIC required to prefer 2-component GMM (default: 6.0).')
parser.add_argument('--denoised_dir_name', type=str, default='Denoised_bspline',
                    help='Folder under DATASPACE containing <CORE>_denoised.ome.tif '
                         '(default: Denoised_bspline). Independent of which '
                         'registration algorithm produced the aligned volume.')
parser.add_argument('--mask_dir_name', type=str, default='CellPose_DAPI_Warped_Bspline',
                    help='Folder under DATASPACE containing warped CellPose masks '
                         '(default: CellPose_DAPI_Warped_Bspline).')
parser.add_argument('--output_dir_name', type=str, default='Phenotypes_Bspline',
                    help='Folder under DATASPACE to write phenotype output into '
                         '(default: Phenotypes_Bspline).')
parser.add_argument('--reg_stats_csv', type=str, default=None,
                    help='Full path to a registration_stats CSV with Slice_ID/Slice_Z '
                         'columns, used for accurate slice-to-volume-index mapping. '
                         'Filename and producing folder vary by registration algorithm, '
                         'so this is a full explicit path, not assembled from a folder '
                         'name. If omitted, falls back to sorted-filename-order matching '
                         '(logged clearly — verify slice_z in the output CSV if used).')
args = parser.parse_args()

TARGET_CORE   = args.core_name
BIC_THRESHOLD = args.bic_threshold

DENOISED_VOL = os.path.join(
    config.DATASPACE, args.denoised_dir_name, TARGET_CORE,
    f'{TARGET_CORE}_denoised.ome.tif',
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DAPI_MASK_DIR = os.path.join(config.DATASPACE, args.mask_dir_name, TARGET_CORE)
OUTPUT_DIR    = os.path.join(config.DATASPACE, args.output_dir_name, TARGET_CORE)
QC_DIR        = os.path.join(OUTPUT_DIR, 'qc_plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
if args.plot_qc:
    os.makedirs(QC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id(path: str) -> int:
    m = re.search(r'TMA_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _gmm_intersection(gmm: GaussianMixture) -> float | None:
    """
    Threshold at the intersection of the two Gaussian PDFs between their means.
    Returns None if components overlap too much to give a reliable boundary.
    """
    means   = gmm.means_.flatten()
    sigmas  = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    neg_idx = int(np.argmin(means))
    pos_idx = int(np.argmax(means))
    mu_neg, sig_neg, w_neg = means[neg_idx], sigmas[neg_idx], weights[neg_idx]
    mu_pos, sig_pos, w_pos = means[pos_idx], sigmas[pos_idx], weights[pos_idx]

    # Require at least 0.5σ separation — below this the intersection is unstable
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

    # Reject if intersection sits inside the negative component
    if root <= mu_neg + 0.5 * sig_neg:
        return None

    return root


# Markers where Otsu on non-zero cell medians is preferred over GMM even
# when GMM finds bimodality.  Typically high-dynamic-range structural markers
# where the positive population is non-Gaussian (heavy right tail) and GMM
# places the intersection too conservatively.
PREFER_OTSU = {'CK'}

# Manual thresholds — override automatic fitting entirely for specified markers.
# Use this for markers that are too sparse for reliable automatic thresholding
# (e.g. CD31 in TMA cores where vessel cross-sections are very few).
# Values are in denoised intensity units (same space as cell medians).
# Set to None or remove the entry to use automatic fitting.
# Validate these values by inspecting QC plots across multiple cores.
MANUAL_THRESHOLDS: dict[str, float] = {
    'CD31': 200.0,   # ← set from visual inspection; adjust as needed
}


def threshold_from_cell_medians(
    cell_medians: np.ndarray,
    marker_name:  str = 'Marker',
) -> tuple[float, str]:
    """
    Derive a positivity threshold from the distribution of per-cell median
    intensities.

    This is the only thresholding function used.  Image-pixel histograms are
    not used because background subtraction in denoise_volume.py collapses
    the background mode to near zero, making pixel-space Otsu/GMM unreliable.

    Strategy (in order):
      1. GMM bimodal   — clear negative/positive separation  → 'gmm_bimodal'
      2. Otsu on non-zero medians — GMM intersection near zero, but signal
                                    exists above background  → 'otsu_nz'
      3. p75 fallback  — no meaningful signal detected        → 'p75_fallback'
                         (known limitation, logged as warning)

    Returns
    -------
    (threshold: float, threshold_type: str)
    """
    cell_medians = np.asarray(cell_medians, dtype=np.float64)

    # ── Manual override ───────────────────────────────────────────────────────
    if marker_name in MANUAL_THRESHOLDS:
        thresh = float(MANUAL_THRESHOLDS[marker_name])
        logger.info(f'  [{marker_name}] Using manual threshold: {thresh:.1f}')
        return thresh, 'manual'

    if len(cell_medians) < 10:
        return float(np.percentile(cell_medians, 75)), 'p75_fallback_too_few_cells'

    # ── Step 1: GMM on all cell medians ──────────────────────────────────────
    # Skipped for markers in PREFER_OTSU — go straight to Otsu.
    if marker_name not in PREFER_OTSU:
        X = cell_medians.reshape(-1, 1)
        try:
            gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
            gmm2 = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(X)
            delta_bic = gmm1.bic(X) - gmm2.bic(X)
        except Exception as exc:
            logger.warning(f'  [{marker_name}] GMM fit failed ({exc}) — using otsu_nz.')
            delta_bic = 0.0
            gmm2      = None

        if delta_bic > BIC_THRESHOLD and gmm2 is not None:
            intersection = _gmm_intersection(gmm2)
            mu_pos = float(gmm2.means_.max())
            mu_neg = float(gmm2.means_.min())
            min_meaningful = max(mu_pos * 0.05, (mu_pos - mu_neg) * 0.10)
            if intersection is not None and intersection > min_meaningful:
                return intersection, 'gmm_bimodal'

    # ── Step 2: Otsu on non-zero medians only ────────────────────────────────
    nz = cell_medians[cell_medians > 0]
    if len(nz) >= 10:
        n_bins = min(256, len(nz) // 2)
        counts, edges = np.histogram(nz, bins=n_bins)
        centres = (edges[:-1] + edges[1:]) / 2.0
        # Classical Otsu on the non-zero histogram
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
        # Sanity check: Otsu should be above the 10th percentile of non-zero
        # medians, otherwise it found the zero/non-zero boundary not the
        # negative/positive boundary
        if otsu_val > float(np.percentile(nz, 10)):
            return otsu_val, 'otsu_nz'

    # ── Step 3: p75 fallback — no signal structure detected ──────────────────
    thresh = float(np.percentile(cell_medians, 75))
    logger.warning(
        f'  [{marker_name}] No bimodal structure found in cell medians. '
        f'Using p75 fallback ({thresh:.1f}). '
        f'This marker may have very little signal in this core/slice.'
    )
    return thresh, 'p75_fallback'


# ─────────────────────────────────────────────────────────────────────────────
# MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_slice(
    mask:         np.ndarray,
    volume_slice: np.ndarray,
    min_area:     int = 200,
    slice_id:     int = 0,
) -> pd.DataFrame:
    """
    For one slice: measure per-cell median intensity for each marker,
    derive a per-slice threshold from cell medians, and classify cells.

    Parameters
    ----------
    mask         : 2-D uint32 CellPose label image (warped DAPI mask).
    volume_slice : CYX array from the denoised volume (float32).
    min_area     : Minimum nucleus area in pixels; smaller nuclei are dropped.
    slice_id     : TMA slice identifier used for logging.

    Returns
    -------
    pd.DataFrame with one row per cell.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return pd.DataFrame()

    label_list = labels.tolist()
    areas      = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int32)
    keep       = areas >= min_area
    label_list = [l for l, k in zip(label_list, keep) if k]
    areas      = areas[keep]
    if len(label_list) == 0:
        return pd.DataFrame()

    # Expand nuclear mask 4 px (~2 µm) to capture membrane signal
    cell_mask = expand_labels(mask, distance=4)

    cy = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape),
        mask, label_list,
    ))
    cx = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[1])[None, :], mask.shape),
        mask, label_list,
    ))

    df = pd.DataFrame({
        'cell_id':    label_list,
        'area_px':    areas,
        'area_um2':   np.round(areas * PIXEL_SIZE_XY_UM ** 2, 3),
        'centroid_x': np.round(cx, 1),
        'centroid_y': np.round(cy, 1),
    })

    for ch_name in MARKER_CHANNELS:
        ch_idx  = CHANNEL_IDX[ch_name]
        ch_img  = volume_slice[ch_idx].astype(np.float32)

        # ── Per-cell median (the only number used for classification) ─────────
        flat_labels  = cell_mask.ravel()
        flat_pixels  = ch_img.ravel()
        sort_idx     = np.argsort(flat_labels, kind='stable')
        sorted_lbl   = flat_labels[sort_idx]
        sorted_px    = flat_pixels[sort_idx]
        boundaries   = np.searchsorted(sorted_lbl, label_list)
        end_bounds   = np.searchsorted(sorted_lbl, label_list, side='right')
        cell_medians = np.array([
            float(np.median(sorted_px[s:e])) if e > s else 0.0
            for s, e in zip(boundaries, end_bounds)
        ], dtype=np.float32)

        df[f'median_{ch_name}'] = np.round(cell_medians, 2)

        # ── Threshold from cell-median distribution ───────────────────────────
        thresh, t_type = threshold_from_cell_medians(cell_medians, marker_name=ch_name)

        logger.info(
            f'    [{ch_name}] threshold = {thresh:.1f}  [{t_type}]  '
            f'({int((cell_medians >= thresh).sum())}/{len(cell_medians)} positive)'
        )

        df[f'thresh_{ch_name}']      = round(thresh, 2)
        df[f'thresh_type_{ch_name}'] = t_type
        df[f'pos_{ch_name}']         = (cell_medians >= thresh).astype(np.uint8)

    # Return channel images keyed by name so QC plots can draw pixel histograms
    ch_imgs = {ch: volume_slice[CHANNEL_IDX[ch]].astype(np.float32)
               for ch in MARKER_CHANNELS}
    return df, ch_imgs


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SLICE CONSENSUS
# ─────────────────────────────────────────────────────────────────────────────

# Threshold types that are unreliable for consensus — slices with these types
# are excluded from the median calculation.
_UNRELIABLE_TYPES = ('p75_fallback', 'too_few_cells')

def apply_consensus_thresholds(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Derive one threshold per marker from the median of per-slice thresholds
    across all slices, then recompute positivity globally with that consensus.

    Original per-slice thresholds are preserved in thresh_per_slice_<marker>.

    This is especially important for sparse markers (CD31) where individual
    slices may not contain enough positive cells for a stable fit.
    """
    logger.info('Applying cross-slice consensus thresholds ...')

    for ch in MARKER_CHANNELS:
        col     = f'thresh_{ch}'
        t_col   = f'thresh_type_{ch}'

        # Preserve per-slice values
        df_all[f'thresh_per_slice_{ch}'] = df_all[col]

        # Manual thresholds pass through consensus unchanged — don't overwrite
        if df_all[t_col].str.contains('manual', na=False).all():
            logger.info(f'  [{ch}] Manual threshold — skipping consensus.')
            continue

        # Only use slices where the threshold was derived cleanly
        reliable = ~df_all[t_col].str.contains(
            '|'.join(_UNRELIABLE_TYPES), na=False
        )
        reliable_vals = df_all.loc[reliable, col]

        if reliable_vals.empty:
            logger.warning(
                f'  [{ch}] No reliable per-slice thresholds found. '
                f'Keeping per-slice values. Check QC plots for this marker.'
            )
            continue

        consensus = float(reliable_vals.median())
        spread    = float(reliable_vals.std())
        n_reliable = int(reliable.sum())
        n_total    = int(len(df_all['slice_id'].unique()))
        logger.info(
            f'  [{ch}] consensus = {consensus:.1f}  '
            f'(median ± SD: {consensus:.1f} ± {spread:.1f}  '
            f'from {n_reliable}/{n_total} reliable slices)'
        )

        df_all[col]             = consensus
        df_all[t_col]           = df_all[t_col] + '_consensus'
        cell_medians            = df_all[f'median_{ch}'].values
        df_all[f'pos_{ch}']     = (cell_medians >= consensus).astype(np.uint8)

        n_pos = int(df_all[f'pos_{ch}'].sum())
        n_tot = len(df_all)
        logger.info(f'  [{ch}] {n_pos}/{n_tot} positive ({100.0 * n_pos / n_tot:.1f}%)')

    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# QC VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_label_map(mask, cell_ids, values, dtype=np.float32) -> np.ndarray:
    """Paint a per-cell scalar value onto a label-mask image."""
    max_label = int(mask.max())
    lut       = np.zeros(max_label + 1, dtype=dtype)
    cell_ids  = np.asarray(cell_ids)
    values    = np.asarray(values)
    valid     = cell_ids <= max_label
    lut[cell_ids[valid]] = values[valid]
    return lut[mask]


def _stretch(img: np.ndarray) -> np.ndarray:
    flat = img.ravel()
    lo, hi = np.percentile(flat, [1, 99])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


def _make_overlay(
    raw_img:     np.ndarray,
    mask:        np.ndarray,
    cell_ids:    np.ndarray,
    cell_meds:   np.ndarray,
    threshold:   float,
) -> np.ndarray:
    """Grayscale image with negative cells in dark red, positive cells in green."""
    gray = _stretch(raw_img)
    rgb  = np.stack([gray, gray, gray], axis=-1).copy()
    if len(cell_ids) == 0:
        return rgb
    pos_arr = (cell_meds >= threshold).astype(np.float32)
    pos_map = build_label_map(mask, cell_ids, pos_arr)
    neg_px  = (mask > 0) & (pos_map == 0)
    rgb[neg_px, 0] = np.clip(rgb[neg_px, 0] * 0.5 + 0.3, 0, 1)
    rgb[neg_px, 1] = rgb[neg_px, 1] * 0.2
    rgb[neg_px, 2] = rgb[neg_px, 2] * 0.2
    pos_px = pos_map == 1
    rgb[pos_px, 0] = rgb[pos_px, 0] * 0.2
    rgb[pos_px, 1] = np.clip(rgb[pos_px, 1] * 0.4 + 0.5, 0, 1)
    rgb[pos_px, 2] = rgb[pos_px, 2] * 0.2
    return rgb


def save_qc_plot(
    dapi_img:     np.ndarray,
    mask:         np.ndarray,
    df_slice:     pd.DataFrame,
    ch_imgs:      dict,
    slice_id:     int,
    out_path:     str,
) -> None:
    """
    Per-slice overview: one row per marker, four columns.

    Col 0  Raw image (contrast-stretched grayscale)
    Col 1  Positivity overlay (green = positive, dark red = negative)
    Col 2  Pixel intensity histogram — QuPath-style (all pixels, log y).
           Threshold is shown converted back to pixel space as a dashed line
           for visual familiarity.  Note: threshold is derived from col 3,
           not from this histogram.
    Col 3  Cell-median histogram — where the threshold is actually derived.
           Threshold line (red) sits on the distribution it was fitted to.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_markers = len(MARKER_CHANNELS)
        fig = plt.figure(figsize=(24, 4 * (1 + n_markers)))
        gs  = gridspec.GridSpec(1 + n_markers, 4, figure=fig,
                                hspace=0.40, wspace=0.18)

        n_cells = len(df_slice)
        fig.suptitle(
            f'{TARGET_CORE}  |  Slice {slice_id}  |  {n_cells} cells',
            fontsize=14, fontweight='bold', y=1.002,
        )

        # Calculate aspect ratio (Height / Width) to align plot heights with image dimensions
        img_aspect = dapi_img.shape[0] / dapi_img.shape[1]

        # ── Row 0: DAPI | score map | positivity bar chart | (empty) ─────────
        ax_dapi = fig.add_subplot(gs[0, 0])
        ax_dapi.imshow(_stretch(dapi_img), cmap='gray', interpolation='nearest')
        ax_dapi.set_title('DAPI', fontsize=10)
        ax_dapi.axis('off')

        ax_score = fig.add_subplot(gs[0, 1])
        if n_cells > 0:
            scores    = df_slice[[f'pos_{ch}' for ch in MARKER_CHANNELS]].sum(axis=1).values.astype(np.float32)
            score_map = build_label_map(mask, df_slice['cell_id'].values, scores)
        else:
            score_map = np.zeros(mask.shape, dtype=np.float32)
        im = ax_score.imshow(score_map, cmap='hot', vmin=0, vmax=n_markers,
                             interpolation='nearest')
        plt.colorbar(im, ax=ax_score, fraction=0.03, pad=0.02)
        ax_score.set_title(f'Total marker score (0–{n_markers})', fontsize=10)
        ax_score.axis('off')

        ax_bar = fig.add_subplot(gs[0, 2])
        pct_pos = [100.0 * df_slice[f'pos_{ch}'].sum() / n_cells if n_cells > 0 else 0
                   for ch in MARKER_CHANNELS]
        colors  = ['#e74c3c' if p > 50 else '#3498db' for p in pct_pos]
        bars    = ax_bar.barh(MARKER_CHANNELS, pct_pos, color=colors)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('% positive cells', fontsize=9)
        ax_bar.set_title('Positivity summary', fontsize=10)
        for bar, pct in zip(bars, pct_pos):
            ax_bar.text(min(pct + 1, 98), bar.get_y() + bar.get_height() / 2,
                        f'{pct:.0f}%', va='center', fontsize=8)
        ax_bar.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_bar.set_box_aspect(img_aspect)

        fig.add_subplot(gs[0, 3]).axis('off')   # placeholder

        # ── Rows 1+: one row per marker ──────────────────────────────────────
        for row, ch in enumerate(MARKER_CHANNELS, start=1):
            ch_img    = ch_imgs[ch]
            thresh_t  = float(df_slice[f'thresh_{ch}'].iloc[0]) if n_cells > 0 else 0.0
            t_type    = df_slice[f'thresh_type_{ch}'].iloc[0]   if n_cells > 0 else ''
            cell_meds = df_slice[f'median_{ch}'].values          if n_cells > 0 else np.array([])
            pos_arr   = df_slice[f'pos_{ch}'].values             if n_cells > 0 else np.array([])

            # ── Col 0: raw image ──────────────────────────────────────────────
            ax_raw = fig.add_subplot(gs[row, 0])
            ax_raw.imshow(_stretch(ch_img), cmap='gray', interpolation='nearest')
            ax_raw.set_title(f'{ch}  raw', fontsize=9)
            ax_raw.axis('off')

            # ── Col 1: positivity overlay ─────────────────────────────────────
            ax_pos = fig.add_subplot(gs[row, 1])
            if n_cells > 0:
                rgb   = _make_overlay(ch_img, mask, df_slice['cell_id'].values,
                                      cell_meds, thresh_t)
                n_pos = int(pos_arr.sum())
                pct   = 100.0 * n_pos / n_cells
                ax_pos.set_title(f'{ch}  {n_pos}/{n_cells} ({pct:.0f}%)', fontsize=9)
            else:
                gray = _stretch(ch_img)
                rgb  = np.stack([gray, gray, gray], axis=-1)
                ax_pos.set_title(f'{ch}  no cells', fontsize=9)
            ax_pos.imshow(rgb, interpolation='nearest')
            ax_pos.axis('off')

            # ── Col 2: QuPath-style pixel histogram ───────────────────────────
            ax_px = fig.add_subplot(gs[row, 2])
            flat  = ch_img.ravel()
            flat  = flat[flat > 0]          # skip exact-zero background pixels
            if len(flat) > 0:
                counts, edges = np.histogram(flat, bins=512)
                centres = (edges[:-1] + edges[1:]) / 2.0
                bin_w   = centres[1] - centres[0]
                ax_px.bar(centres, counts, width=bin_w,
                          color='steelblue', alpha=0.75, edgecolor='none')
                ax_px.set_yscale('log')
                ax_px.axvline(thresh_t, color='red', linewidth=1.5,
                              linestyle='--',
                              label=f'threshold = {thresh_t:.1f}')
                ax_px.legend(fontsize=7)
            ax_px.set_xlabel('Pixel intensity (denoised)', fontsize=8)
            ax_px.set_ylabel('Pixel count (log)', fontsize=8)
            ax_px.set_title(f'{ch}  pixel distribution (QuPath-style)', fontsize=9)
            ax_px.tick_params(labelsize=7)
            ax_px.set_box_aspect(img_aspect)

            # ── Col 3: cell-median histogram ──────────────────────────────────
            ax_cm = fig.add_subplot(gs[row, 3])
            if n_cells > 0 and len(cell_meds) > 0:
                ax_cm.hist(cell_meds, bins=80, color='#e67e22',
                           alpha=0.80, edgecolor='none', log=True)
                line_col = 'magenta' if 'manual' in t_type else 'red'
                ax_cm.axvline(thresh_t, color=line_col, linewidth=2.0,
                              label=f'{thresh_t:.1f}\n[{t_type}]')
                ax_cm.legend(fontsize=7)
            ax_cm.set_xlabel('Cell median intensity', fontsize=8)
            ax_cm.set_ylabel('Cell count (log)', fontsize=8)
            ax_cm.set_title(f'{ch}  cell-median distribution  ← threshold here', fontsize=9)
            ax_cm.tick_params(labelsize=7)
            ax_cm.set_box_aspect(img_aspect)

        fig.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  QC plot saved: {os.path.basename(out_path)}')

    except Exception as exc:
        logger.warning(f'  QC plot failed for slice {slice_id}: {exc}')


def save_single_channel_qc(
    mask:      np.ndarray,
    df_slice:  pd.DataFrame,
    raw_img:   np.ndarray,
    ch_name:   str,
    slice_id:  int,
    out_path:  str,
) -> None:
    """
    High-resolution 1×4 pathologist QC panel for one marker channel.

    Panel 1  Raw image (contrast-stretched grayscale)
    Panel 2  Cell-median histogram — where the threshold is derived and applied.
             Threshold shown as solid red line (magenta for manual thresholds).
    Panel 3  Full-image pixel intensity histogram (non-zero pixels, log y).
             Threshold shown as dashed red line for visual reference only.
    Panel 4  Positivity overlay (green = positive, dark red = negative)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig     = plt.figure(figsize=(32, 12))
        # Reserve the top 12% for the suptitle; panels occupy the rest
        gs      = gridspec.GridSpec(1, 4, figure=fig, wspace=0.30,
                                    left=0.04, right=0.98,
                                    top=0.78, bottom=0.12)
        n_cells = len(df_slice)

        thresh_t  = float(df_slice[f'thresh_{ch_name}'].iloc[0]) if n_cells > 0 else 0.0
        t_type    = df_slice[f'thresh_type_{ch_name}'].iloc[0]   if n_cells > 0 else ''
        cell_meds = df_slice[f'median_{ch_name}'].values          if n_cells > 0 else np.array([])
        n_pos     = int((cell_meds >= thresh_t).sum())            if n_cells > 0 else 0
        pct       = 100.0 * n_pos / n_cells                       if n_cells > 0 else 0.0

        fig.suptitle(
            f'{TARGET_CORE}  |  Slice {slice_id:03d}  |  {ch_name}  |  {n_cells} cells\n'
            f'threshold = {thresh_t:.1f}  [{t_type}]  |  {n_pos}/{n_cells} positive ({pct:.0f}%)',
            fontsize=32, fontweight='bold', y=0.97,
        )

        # Calculate aspect ratio (Height / Width) to lock histogram box heights to the image
        img_aspect = raw_img.shape[0] / raw_img.shape[1]

        # ── Panel 1: raw image ────────────────────────────────────────────────
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_raw.imshow(_stretch(raw_img), cmap='gray', interpolation='nearest')
        ax_raw.set_title(
            f'{ch_name}\nBackground-corrected',
            fontsize=30, pad=10,
        )
        ax_raw.axis('off')

        # ── Panel 2: cell-median histogram ────────────────────────────────────
        ax_cm = fig.add_subplot(gs[0, 1])
        if n_cells > 0 and len(cell_meds) > 0:
            ax_cm.hist(cell_meds, bins=100, color='#e67e22',
                       alpha=0.80, edgecolor='none', log=True)
            line_col = 'magenta' if 'manual' in t_type else 'red'
            ax_cm.axvline(thresh_t, color=line_col, linewidth=2.5,
                          label=f'threshold = {thresh_t:.1f}\n[{t_type}]')
            ax_cm.legend(fontsize=28)
        ax_cm.set_xlabel('Per-cell median intensity', fontsize=28)
        ax_cm.set_ylabel('Cell count (log scale)', fontsize=28)
        ax_cm.set_title(
            'Per-cell median distribution\n(threshold derived here)',
            fontsize=30, pad=10,
        )
        ax_cm.tick_params(labelsize=28)
        ax_cm.set_box_aspect(img_aspect)

        # ── Panel 3: full-image pixel histogram (visual reference) ────────────
        ax_px = fig.add_subplot(gs[0, 2])
        flat  = raw_img.ravel().astype(np.float32)
        flat  = flat[flat > 0]
        if len(flat) > 0:
            counts, edges = np.histogram(flat, bins=512)
            centres = (edges[:-1] + edges[1:]) / 2.0
            bin_w   = centres[1] - centres[0]
            ax_px.bar(centres, counts, width=bin_w,
                      color='steelblue', alpha=0.75, edgecolor='none')
            ax_px.set_yscale('log')
            ax_px.axvline(thresh_t, color='red', linewidth=2.0, linestyle='--',
                          label=f'threshold = {thresh_t:.1f}\n(visual reference only)')
            ax_px.legend(fontsize=28)
        ax_px.set_xlabel('Pixel intensity (background-corrected)', fontsize=28)
        ax_px.set_ylabel('Pixel count (log scale)', fontsize=28)
        ax_px.set_title(
            'Full-image pixel distribution\n(visual reference only)',
            fontsize=30, pad=10,
        )
        ax_px.tick_params(labelsize=28)
        ax_px.set_box_aspect(img_aspect)

        # ── Panel 4: positivity overlay ───────────────────────────────────────
        ax_over = fig.add_subplot(gs[0, 3])
        if n_cells > 0:
            rgb = _make_overlay(raw_img, mask, df_slice['cell_id'].values,
                                cell_meds, thresh_t)
            ax_over.set_title(
                f'Positivity overlay\n{n_pos} / {n_cells} positive  ({pct:.0f}%)',
                fontsize=30, pad=10,
            )
        else:
            gray = _stretch(raw_img)
            rgb  = np.stack([gray, gray, gray], axis=-1)
            ax_over.set_title('Positivity overlay\nNo cells', fontsize=30, pad=10)
        ax_over.imshow(rgb, interpolation='nearest')
        ax_over.axis('off')

        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as exc:
        logger.warning(f'  High-res QC failed for {ch_name} slice {slice_id}: {exc}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(f'Phenotyping — core: {TARGET_CORE}')
    logger.info(f'Denoised volume   : {DENOISED_VOL}')
    logger.info(f'DAPI mask dir     : {DAPI_MASK_DIR}')
    logger.info(f'BIC threshold     : {BIC_THRESHOLD}')

    if not os.path.exists(DENOISED_VOL):
        logger.error(f'Denoised volume not found: {DENOISED_VOL}')
        sys.exit(1)
    if not os.path.isdir(DAPI_MASK_DIR):
        logger.error(f'DAPI mask directory not found: {DAPI_MASK_DIR}')
        sys.exit(1)

    # ── Load denoised volume ──────────────────────────────────────────────────
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

    # ── Slice ID → Z-index mapping ────────────────────────────────────────────
    slice_id_to_z = {}
    if args.reg_stats_csv:
        if not os.path.exists(args.reg_stats_csv):
            logger.warning(
                f'--reg_stats_csv given but not found: {args.reg_stats_csv} — '
                f'falling back to sorted-filename-order matching. '
                f'Verify slice_z in the output CSV is correct for this core.'
            )
        else:
            reg_df = pd.read_csv(args.reg_stats_csv)
            slice_id_to_z = dict(
                zip(reg_df['Slice_ID'].astype(int), reg_df['Slice_Z'].astype(int))
            )
            all_z     = set(range(n_slices))
            missing_z = all_z - set(slice_id_to_z.values())
            if len(missing_z) == 1:
                mask_ids     = {get_slice_id(p) for p in
                                glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif'))}
                unmapped_ids = mask_ids - set(slice_id_to_z.keys())
                if len(unmapped_ids) == 1:
                    anchor_id = unmapped_ids.pop()
                    anchor_z  = missing_z.pop()
                    slice_id_to_z[anchor_id] = anchor_z
                    logger.info(f'Anchor slice inferred: ID={anchor_id} → Z={anchor_z}')
            logger.info(f'Slice ID→Z mapping: {len(slice_id_to_z)} entries')
    else:
        logger.warning(
            'No --reg_stats_csv provided — using sorted-filename-order matching. '
            'This assumes Slice_ID order matches true Z stack order; verify slice_z '
            'in the output CSV if your registration reordered slices.'
        )

    mask_files = sorted(
        glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif')),
        key=get_slice_id,
    )
    if not mask_files:
        logger.error(f'No DAPI mask files found in {DAPI_MASK_DIR}')
        sys.exit(1)
    logger.info(f'Found {len(mask_files)} DAPI mask files.')

    all_records = []

    for enum_idx, mask_path in enumerate(mask_files):
        slice_id = get_slice_id(mask_path)
        z_idx    = slice_id_to_z.get(slice_id, enum_idx)
        if slice_id_to_z and slice_id not in slice_id_to_z:
            logger.warning(f'  Slice ID {slice_id} not in CSV mapping — using position {enum_idx}.')

        logger.info(f'  Slice Z={z_idx:03d}  ID={slice_id:03d}  ({enum_idx + 1}/{len(mask_files)})')

        mask = tifffile.imread(mask_path).astype(np.uint32)
        if mask.ndim != 2:
            mask = mask.squeeze()
        if mask.shape != (H, W):
            logger.warning(f'  Mask {mask.shape} != volume ({H},{W}) — skipping.')
            continue
        if z_idx >= n_slices:
            logger.warning(f'  z_idx {z_idx} out of range — skipping.')
            continue

        df_slice, ch_imgs = measure_slice(
            mask=mask,
            volume_slice=vol[z_idx],
            min_area=args.min_area_px,
            slice_id=slice_id,
        )

        if df_slice.empty:
            logger.warning(f'  No cells in slice {slice_id} after min_area filter.')
            continue

        df_slice.insert(0, 'slice_id', slice_id)
        df_slice.insert(0, 'slice_z',  z_idx)
        df_slice.insert(0, 'core',     TARGET_CORE)

        logger.info(
            f'  {len(df_slice)} nuclei | '
            + '  '.join(f'{ch}+={int(df_slice[f"pos_{ch}"].sum())}' for ch in MARKER_CHANNELS)
        )

        all_records.append(df_slice)

        if args.plot_qc:
            qc_path  = os.path.join(QC_DIR, f'TMA_{slice_id:03d}_DAPI_phenotype_qc.png')
            dapi_img = vol[z_idx][CHANNEL_IDX['DAPI']]
            save_qc_plot(dapi_img, mask, df_slice, ch_imgs, slice_id, qc_path)

            ch_qc_dir = os.path.join(QC_DIR, 'per_channel', f'Slice_{slice_id:03d}')
            os.makedirs(ch_qc_dir, exist_ok=True)
            for ch in MARKER_CHANNELS:
                save_single_channel_qc(
                    mask=mask,
                    df_slice=df_slice,
                    raw_img=ch_imgs[ch],
                    ch_name=ch,
                    slice_id=slice_id,
                    out_path=os.path.join(ch_qc_dir, f'{ch}_highres_qc.png'),
                )

    if not all_records:
        logger.error('No cells phenotyped — no CSV written.')
        sys.exit(1)

    df_all = pd.concat(all_records, ignore_index=True)

    if not args.no_consensus:
        df_all = apply_consensus_thresholds(df_all)
    else:
        logger.info('Consensus threshold skipped (--no_consensus).')

    # ── Output columns ────────────────────────────────────────────────────────
    meta_cols      = ['core', 'slice_z', 'slice_id', 'cell_id',
                      'area_px', 'area_um2', 'centroid_x', 'centroid_y']
    median_cols    = [f'median_{ch}'           for ch in MARKER_CHANNELS]
    pos_cols       = [f'pos_{ch}'              for ch in MARKER_CHANNELS]
    thresh_cols    = [f'thresh_{ch}'           for ch in MARKER_CHANNELS]
    ps_cols        = [f'thresh_per_slice_{ch}' for ch in MARKER_CHANNELS
                      if f'thresh_per_slice_{ch}' in df_all.columns]
    t_type_cols    = [f'thresh_type_{ch}'      for ch in MARKER_CHANNELS]

    df_all = df_all[meta_cols + median_cols + pos_cols
                    + thresh_cols + ps_cols + t_type_cols]

    csv_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes.csv')
    df_all.to_csv(csv_path, index=False)

    total_cells = len(df_all)
    logger.info('=' * 60)
    logger.info(f'Done.  Core: {TARGET_CORE}  |  Total cells: {total_cells}')
    logger.info(f'CSV: {csv_path}')
    for ch in MARKER_CHANNELS:
        n_pos  = int(df_all[f'pos_{ch}'].sum())
        pct    = 100.0 * n_pos / total_cells if total_cells > 0 else 0.0
        thresh = float(df_all[f'thresh_{ch}'].iloc[0])
        t_type = df_all[f'thresh_type_{ch}'].iloc[0]
        logger.info(
            f'  {ch:8s}: {n_pos:6d} positive ({pct:5.1f}%)  '
            f'thresh = {thresh:.1f}  [{t_type}]'
        )
    logger.info('=' * 60)


if __name__ == '__main__':
    main()