"""
phenotype_cells.py
==================
Cell phenotyping based on DAPI-segmented nuclei and a denoised registered volume.

Pipeline per core:
  1. Load the denoised volume  (Denoised/<CORE>/<CORE>_denoised.ome.tif)  — ZCYX float32
  2. For each slice Z:
       a. Load the warped DAPI mask  (CellPose_DAPI_Warped/<CORE>/TMA_<ID>_DAPI_cp_masks_warped.tif)
       b. For each marker channel (CD31, GAP43, NFP, CD3, CD163, CK, AF):
            - Apply log1p normalization to the denoised channel image.
            - Extract per-nucleus mean intensity using the expanded DAPI mask labels.
            - Fit 1- and 2-component Gaussian Mixture Models; select via BIC.
            - Assign binary positivity (0/1) per cell per marker.
  3. Concatenate all slices → one CSV per core.
  4. Apply cross-slice consensus threshold (median across slices) and recompute positivity.

Thresholding strategy
---------------------
  A 2-component GMM is fit to the log1p per-cell means for each marker.
  BIC scores for the 1-component and 2-component fits are compared:
    - If ΔBIC = BIC(1C) − BIC(2C) > BIC_THRESHOLD (default 6), the distribution is
      genuinely bimodal and the threshold is placed at the intersection of the two
      Gaussian components  → type 'gmm_bimodal'.
    - Otherwise the distribution is unimodal and Otsu is used as a fallback
      → type 'otsu_unimodal'.
  A MAD-based noise floor guards against the threshold dipping below the negative
  population noise level.  A positivity ceiling (MARKER_MAX_PCT) acts as a final
  biological sanity check, mirroring the pathologist review step in Backman et al.

Output columns
--------------
    core, slice_z, slice_id, cell_id,
    mean_CD31, mean_GAP43, mean_NFP, mean_CD3, mean_CD163, mean_CK, mean_AF,
    pos_CD31,  pos_GAP43,  pos_NFP,  pos_CD3,  pos_CD163,  pos_CK,  pos_AF,
    area_px, area_um2, centroid_x, centroid_y,
    thresh_CD31, ..., otsu_CD31, ..., floor_CD31, ..., thresh_type_CD31, ...

Usage
-----
    python phenotype_cells.py --core_name Core_01 [--plot_qc] [--min_area_px 200]
                              [--no_consensus] [--bic_threshold 6.0]
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
# A value > 6 is conventionally "strong evidence" for the more complex model.
# Raise to 10–20 to prefer unimodal (more conservative positivity calls).
# Passed in at runtime via --bic_threshold; this is the default.
DEFAULT_BIC_THRESHOLD = 6.0

# Biologically plausible maximum positivity fraction per marker.
# If the threshold calls more than this fraction positive it is pushed rightward.
# Adjust for your tissue type.
MARKER_MAX_PCT = {
    'CD31':  0.85,
    'GAP43': 0.85,
    'NFP':   0.50,
    'CD3':   0.90,
    'CD163': 0.90,
    'CK':    0.98,
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Phenotype cells: measure marker expression in DAPI-segmented nuclei.'
)
parser.add_argument('--core_name',      type=str,   required=True)
parser.add_argument('--min_area_px',    type=int,   default=200,
                    help='Minimum nucleus area in pixels to include (default: 200).')
parser.add_argument('--plot_qc',        action='store_true',
                    help='Save per-slice QC overlay images and GMM density plots.')
parser.add_argument('--no_consensus',   action='store_true',
                    help='Skip cross-slice consensus threshold step.')
parser.add_argument('--bic_threshold',  type=float, default=DEFAULT_BIC_THRESHOLD,
                    help='ΔBIC required to prefer 2-component GMM over 1-component '
                         '(default: 6.0).  Increase for more conservative positivity.')
args = parser.parse_args()

TARGET_CORE   = args.core_name
BIC_THRESHOLD = args.bic_threshold

DENOISED_VOL = os.path.join(
    config.DATASPACE, 'Denoised', TARGET_CORE,
    f'{TARGET_CORE}_denoised.ome.tif',
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DAPI_MASK_DIR = os.path.join(config.DATASPACE, 'CellPose_DAPI_Warped', TARGET_CORE)
OUTPUT_DIR    = os.path.join(config.DATASPACE, 'Phenotypes', TARGET_CORE)
QC_DIR        = os.path.join(OUTPUT_DIR, 'qc_plots')
GMM_QC_DIR    = os.path.join(QC_DIR, 'gmm_validation')

os.makedirs(OUTPUT_DIR, exist_ok=True)
if args.plot_qc:
    os.makedirs(QC_DIR, exist_ok=True)
    os.makedirs(GMM_QC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id(path: str) -> int:
    m = re.search(r'TMA_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def apply_log_normalization(image_slice: np.ndarray) -> np.ndarray:
    return np.log1p(image_slice.astype(np.float32))


def otsu_threshold(values: np.ndarray) -> float:
    """Vectorized Otsu on log1p per-cell means."""
    if len(values) < 2:
        return float(values[0]) if len(values) == 1 else 0.0
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return float(vmax)
    counts, bin_edges = np.histogram(values, bins=256)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return float(np.median(values))
    prob     = counts / total
    w0       = np.cumsum(prob)
    mu0_sum  = np.cumsum(prob * bin_centres)
    w1       = 1.0 - w0
    mu_total = float(np.sum(prob * bin_centres))
    with np.errstate(invalid='ignore', divide='ignore'):
        mu0 = np.where(w0 > 0, mu0_sum / w0, 0.0)
        mu1 = np.where(w1 > 0, (mu_total - mu0_sum) / w1, 0.0)
    between_var = w0 * w1 * (mu0 - mu1) ** 2
    return float(bin_centres[np.argmax(between_var)])


def background_floor(values: np.ndarray, k: float = 3.0) -> float:
    """
    MAD-based noise floor computed on the bottom 50 % of the distribution
    to isolate the negative population.
    """
    if len(values) < 2:
        return 0.0
    negative_pop = values[values < np.percentile(values, 50)]
    if len(negative_pop) == 0:
        return 0.0
    med = float(np.median(negative_pop))
    mad = float(np.median(np.abs(negative_pop - med)))
    return med if mad == 0.0 else med + k * mad


def _gmm_intersection(gmm: GaussianMixture) -> float:
    """
    Find the threshold as the intersection of the two Gaussian PDFs between
    their means.  Falls back to the arithmetic midpoint if root-finding fails.
    """
    means  = gmm.means_.flatten()
    sigmas = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    neg_idx = int(np.argmin(means))
    pos_idx = int(np.argmax(means))

    mu_neg, sig_neg, w_neg = means[neg_idx],  sigmas[neg_idx],  weights[neg_idx]
    mu_pos, sig_pos, w_pos = means[pos_idx],  sigmas[pos_idx],  weights[pos_idx]

    def delta_pdf(x):
        g_neg = w_neg * np.exp(-0.5 * ((x - mu_neg) / sig_neg) ** 2) / sig_neg
        g_pos = w_pos * np.exp(-0.5 * ((x - mu_pos) / sig_pos) ** 2) / sig_pos
        return g_neg - g_pos

    midpoint = float((mu_neg + mu_pos) / 2.0)
    try:
        # Root exists between the two means
        root = brentq(delta_pdf, mu_neg, mu_pos, xtol=1e-6, maxiter=200)
        return float(root)
    except (ValueError, RuntimeError):
        return midpoint


def _apply_positivity_ceiling(
    threshold: float,
    values: np.ndarray,
    marker_name: str,
    t_type: str,
) -> tuple:
    """Push threshold rightward if positivity exceeds the biological ceiling."""
    max_pct  = MARKER_MAX_PCT.get(marker_name, 0.90)
    pos_frac = float((values >= threshold).mean())
    if pos_frac > max_pct:
        new_thresh = float(np.percentile(values, (1.0 - max_pct) * 100.0))
        logger.warning(
            f'    [{marker_name}] Positivity {pos_frac:.1%} > ceiling {max_pct:.0%} '
            f'— threshold raised from {threshold:.3f} to {new_thresh:.3f} (ceiling_corrected)'
        )
        return new_thresh, t_type + '_ceiling_corrected'
    return threshold, t_type


# ─────────────────────────────────────────────────────────────────────────────
# GMM THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

def gmm_threshold(
    values: np.ndarray,
    marker_name: str = 'Marker',
    slice_id: int = 0,
    plot_dir: str = None,
    raw_pixels: np.ndarray = None,
    otsu_val: float = None,
    floor_val: float = None,
) -> tuple:
    """
    GMM + BIC thresholding.

    1. Fit a 1-component and a 2-component GMM to the log1p per-cell means.
    2. Compute ΔBIC = BIC(1C) − BIC(2C).
       - ΔBIC > BIC_THRESHOLD → bimodal: threshold at the intersection of the
         two Gaussians  → type 'gmm_bimodal'.
       - Otherwise → unimodal: fall back to Otsu  → type 'otsu_unimodal'.
    3. Otsu escalation: if Otsu calls > MARKER_MAX_PCT positive, escalate to
       the 90th percentile  → type 'high_pct_fallback'.

    Returns
    -------
    (threshold: float, threshold_type: str)
    """
    if len(values) < 4:
        val = float(values[0]) if len(values) == 1 else float(np.median(values))
        return val, 'sparse'

    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        return vmax, 'uniform'

    X = values.reshape(-1, 1)

    try:
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(X)
    except Exception as exc:
        logger.warning(f'    [{marker_name}] GMM fit failed ({exc}) — using Otsu.')
        return otsu_threshold(values), 'otsu_fallback'

    delta_bic = gmm1.bic(X) - gmm2.bic(X)
    logger.debug(f'    [{marker_name}] ΔBIC = {delta_bic:.1f} (threshold={BIC_THRESHOLD})')

    if delta_bic > BIC_THRESHOLD:
        threshold      = _gmm_intersection(gmm2)
        threshold_type = 'gmm_bimodal'
        logger.debug(
            f'    [{marker_name}] gmm_bimodal → intersection at {threshold:.3f} '
            f'(means: {sorted(gmm2.means_.flatten())})'
        )
    else:
        threshold      = otsu_threshold(values)
        threshold_type = 'otsu_unimodal'

        pos_frac = float((values >= threshold).mean())
        max_pct  = MARKER_MAX_PCT.get(marker_name, 0.90)
        if pos_frac > max_pct:
            threshold      = float(np.percentile(values, 90.0))
            threshold_type = 'high_pct_fallback'
            logger.debug(
                f'    [{marker_name}] Otsu gave {pos_frac:.1%} positive '
                f'— escalating to 90th pct → {threshold:.3f}'
            )

    if plot_dir is not None:
        _save_gmm_plot(values, gmm2, delta_bic, threshold, threshold_type,
                       marker_name, slice_id, plot_dir,
                       raw_pixels=raw_pixels, otsu_val=otsu_val, floor_val=floor_val)

    return threshold, threshold_type


def _save_gmm_plot(
    values, gmm2, delta_bic, threshold, threshold_type,
    marker_name, slice_id, plot_dir,
    raw_pixels: np.ndarray = None,
    otsu_val: float = None,
    floor_val: float = None,
) -> None:
    """Save a GMM diagnostic plot with two panels:
      Left  — raw pixel-intensity histogram (like QuPath) with threshold lines.
      Right — per-nucleus log1p mean histogram with GMM fit (algorithmic view).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(
            f'GMM — {marker_name}  (slice {slice_id},  ΔBIC={delta_bic:.1f})',
            fontsize=12, fontweight='bold',
        )

        # ── Left panel: RAW pixel histogram (QuPath-style) ───────────────────
        ax_px = axes[0]
        if raw_pixels is not None and len(raw_pixels) > 0:
            px_raw = raw_pixels.astype(np.float32)
            ax_px.hist(px_raw, bins=256, color='#4a90d9', alpha=0.6,
                       label='Raw pixel intensity')
            # Thresholds are in log1p space — convert back to raw for overlay
            ax_px.axvline(np.expm1(threshold), color='red', linewidth=2.0,
                          label=f'Thresh = {np.expm1(threshold):.1f} ({threshold_type})')
            if otsu_val is not None:
                ax_px.axvline(np.expm1(otsu_val), color='orange', linewidth=1.5,
                              linestyle='--', label=f'Otsu = {np.expm1(otsu_val):.1f}')
            if floor_val is not None:
                ax_px.axvline(np.expm1(floor_val), color='green', linewidth=1.5,
                              linestyle='--', label=f'Floor = {np.expm1(floor_val):.1f}')
            ax_px.set_yscale('log')
            ax_px.set_xlabel('Raw pixel intensity', fontsize=10)
            ax_px.set_ylabel('Pixel count (log scale)', fontsize=10)
            ax_px.set_title('Pixel intensity distribution\n(tissue pixels, QuPath-style)', fontsize=10)
            ax_px.legend(fontsize=8)
        else:
            ax_px.text(0.5, 0.5, 'No raw pixels available',
                       ha='center', va='center', transform=ax_px.transAxes)
            ax_px.set_title('Pixel intensity distribution', fontsize=10)

        # ── Right panel: per-nucleus mean histogram + GMM fit ────────────────
        ax_gmm = axes[1]
        ax_gmm.hist(values, bins=120, density=True, alpha=0.45,
                    color='#4a90d9', label='Per-nucleus log1p mean')

        x_space  = np.linspace(values.min(), values.max(), 500)
        g_means  = gmm2.means_.flatten()
        sigmas   = np.sqrt(gmm2.covariances_.flatten())
        weights  = gmm2.weights_.flatten()

        total_pdf = np.zeros_like(x_space)
        for mu, sigma, w in zip(g_means, sigmas, weights):
            component = w * np.exp(-0.5 * ((x_space - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            ax_gmm.plot(x_space, component, '--', linewidth=1.2, alpha=0.7,
                        label=f'GMM component μ={mu:.2f}')
            total_pdf += component
        ax_gmm.plot(x_space, total_pdf, 'b-', linewidth=1.8, label='GMM total')
        ax_gmm.axvline(threshold, color='red', linewidth=2.0, linestyle='--',
                       label=f'Threshold = {threshold:.2f} ({threshold_type})')
        if otsu_val is not None:
            ax_gmm.axvline(otsu_val, color='orange', linewidth=1.5,
                           linestyle='--', label=f'Otsu = {otsu_val:.2f}')
        if floor_val is not None:
            ax_gmm.axvline(floor_val, color='green', linewidth=1.5,
                           linestyle='--', label=f'Floor = {floor_val:.2f}')

        ax_gmm.set_xlabel('Log1p mean intensity (per nucleus)', fontsize=10)
        ax_gmm.set_ylabel('Density', fontsize=10)
        ax_gmm.set_title('Per-nucleus mean + GMM fit\n(algorithmic view)', fontsize=10)
        ax_gmm.legend(fontsize=8)

        fig.tight_layout()
        out = os.path.join(plot_dir, f'gmm_{slice_id:03d}_{marker_name}.png')
        fig.savefig(out, dpi=90, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        logger.warning(f'  GMM plot failed for {marker_name} slice {slice_id}: {exc}')


# ─────────────────────────────────────────────────────────────────────────────
# PER-SLICE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_slice(
    mask: np.ndarray,
    volume_slice: np.ndarray,
    min_area: int,
    slice_id: int = 0,
    plot_dir: str = None,
) -> tuple:
    """
    Measure per-cell marker intensities from a single denoised volume slice
    and compute per-slice GMM thresholds.

    Parameters
    ----------
    mask         : 2-D uint32 label image from CellPose (warped DAPI mask).
    volume_slice : CYX slice from the denoised volume (float32, log1p applied here).
    min_area     : Minimum nucleus area in pixels; smaller nuclei are dropped.
    slice_id     : TMA slice identifier, used for logging and plot filenames.
    plot_dir     : Directory for GMM diagnostic plots; None disables plotting.

    Returns
    -------
    (df : pd.DataFrame, normalized_slices : dict[str, np.ndarray])
        normalized_slices maps each marker name to its log1p-normalised 2-D image,
        used by save_qc_plot for the processed-channel column.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return pd.DataFrame(), {}

    label_list = labels.tolist()
    areas      = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int32)

    keep       = areas >= min_area
    label_list = [l for l, k in zip(label_list, keep) if k]
    areas      = areas[keep]
    if len(label_list) == 0:
        return pd.DataFrame(), {}

    # Expand nuclear mask by 4 px (~2 µm) to capture membrane signal
    cell_mask = expand_labels(mask, distance=4)

    cy = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape),
        mask, label_list,
    ))
    cx = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[1])[None, :], mask.shape),
        mask, label_list,
    ))

    marker_means      = {}
    normalized_slices = {}

    for ch_name in MARKER_CHANNELS:
        ch_idx   = CHANNEL_IDX[ch_name]
        norm_img = apply_log_normalization(volume_slice[ch_idx])
        normalized_slices[ch_name] = norm_img
        marker_means[ch_name] = np.array(
            ndi.mean(norm_img, cell_mask, label_list), dtype=np.float32
        )

    df = pd.DataFrame({
        'cell_id':    label_list,
        'area_px':    areas,
        'area_um2':   np.round(areas * PIXEL_SIZE_XY_UM ** 2, 3),
        'centroid_x': np.round(cx, 1),
        'centroid_y': np.round(cy, 1),
    })

    for ch_name in MARKER_CHANNELS:
        means = marker_means[ch_name]
        df[f'mean_{ch_name}'] = np.round(means, 3)

        o_val = otsu_threshold(means)
        f_val = background_floor(means)

        # Gather tissue pixels for this channel (for pixel-level QC histogram)
        raw_px = volume_slice[CHANNEL_IDX[ch_name]][cell_mask > 0] if plot_dir else None

        gmm_thresh, t_type = gmm_threshold(
            values=means,
            marker_name=ch_name,
            slice_id=slice_id,
            plot_dir=plot_dir,
            raw_pixels=raw_px,
            otsu_val=o_val,
            floor_val=f_val,
        )

        # Floor guard
        final_thresh = max(gmm_thresh, f_val)
        if final_thresh > gmm_thresh:
            logger.debug(
                f'    [{ch_name}] GMM threshold {gmm_thresh:.3f} < floor {f_val:.3f} '
                f'— raised to floor.'
            )

        # Ceiling guard
        final_thresh, t_type = _apply_positivity_ceiling(
            final_thresh, means, ch_name, t_type
        )

        pos_array = (means >= final_thresh).astype(np.uint8)

        # Reject extreme top-end outliers (>p99.9) as likely imaging artefacts
        if len(means) > 100:
            outlier_ceiling = float(np.percentile(means, 99.9))
            outlier_mask    = means > outlier_ceiling
            if np.any(outlier_mask):
                pos_array[outlier_mask] = 0
                logger.debug(
                    f'    [{ch_name}] Rejected {outlier_mask.sum()} top-end outliers '
                    f'(> {outlier_ceiling:.3f}).'
                )

        df[f'thresh_{ch_name}']      = round(final_thresh, 3)
        df[f'otsu_{ch_name}']        = round(o_val, 3)
        df[f'floor_{ch_name}']       = round(f_val, 3)
        df[f'thresh_type_{ch_name}'] = t_type
        df[f'pos_{ch_name}']         = pos_array

    return df, normalized_slices


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SLICE CONSENSUS THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def apply_consensus_thresholds(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Derive one threshold per marker from the median across all per-slice thresholds
    and recompute positivity globally.  Mirrors the Backman et al. approach of
    applying a single threshold derived from representative samples to all images.

    Original per-slice thresholds are preserved in thresh_raw_<marker>.
    """
    logger.info('Applying cross-slice consensus thresholds ...')

    for ch in MARKER_CHANNELS:
        col = f'thresh_{ch}'
        df_all[f'thresh_raw_{ch}'] = df_all[col]

        # Exclude threshold types that were already biased by ceiling corrections
        reliable_mask = ~df_all[f'thresh_type_{ch}'].str.contains(
            'ceiling_corrected|singular_fallback|sparse|uniform', na=False
        )
        reliable_vals = df_all.loc[reliable_mask, col]

        if reliable_vals.empty:
            logger.warning(f'  [{ch}] No reliable per-slice thresholds — keeping per-slice values.')
            continue

        consensus = float(reliable_vals.median())
        spread    = float(reliable_vals.std())
        logger.info(
            f'  [{ch}] consensus threshold = {consensus:.3f}  '
            f'(per-slice median ± SD: {reliable_vals.median():.3f} ± {spread:.3f})'
        )

        df_all[col] = consensus
        means = df_all[f'mean_{ch}'].values
        pos   = (means >= consensus).astype(np.uint8)

        # Re-apply ceiling at aggregate level
        pos_frac = float(pos.mean())
        max_pct  = MARKER_MAX_PCT.get(ch, 0.90)
        if pos_frac > max_pct:
            ceiling_thresh = float(np.percentile(means, (1.0 - max_pct) * 100.0))
            logger.warning(
                f'  [{ch}] Consensus positivity {pos_frac:.1%} > ceiling {max_pct:.0%} '
                f'— raising consensus threshold to {ceiling_thresh:.3f}'
            )
            df_all[col] = ceiling_thresh
            pos = (means >= ceiling_thresh).astype(np.uint8)
            df_all[f'thresh_type_{ch}'] = df_all[f'thresh_type_{ch}'].apply(
                lambda t: t + '_consensus_ceiling' if 'ceiling' not in t else t
            )

        df_all[f'pos_{ch}'] = pos

        n_pos = int(pos.sum())
        n_tot = len(pos)
        logger.info(
            f'  [{ch}] After consensus: {n_pos}/{n_tot} positive '
            f'({100.0 * n_pos / n_tot:.1f} %)'
        )

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


def save_qc_plot(
    dapi_img: np.ndarray,
    mask: np.ndarray,
    df_slice: pd.DataFrame,
    volume_slice: np.ndarray,
    slice_id: int,
    out_path: str,
    normalized_slices: dict = None,
) -> None:
    """
    Save a per-marker QC panel: raw image | processed image | positivity overlay | histogram.

    Histogram lines:
      Red solid   — final threshold
      Red dotted  — raw per-slice threshold (if different from consensus)
      Orange dash — Otsu reference
      Green dash  — MAD noise floor
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_markers = len(MARKER_CHANNELS)
        fig = plt.figure(figsize=(24, 4 * (1 + n_markers)))
        gs  = gridspec.GridSpec(1 + n_markers, 4, figure=fig, hspace=0.35, wspace=0.15)

        n_cells = len(df_slice)
        fig.suptitle(
            f'{TARGET_CORE}  |  Slice {slice_id}  |  {n_cells} nuclei',
            fontsize=14, fontweight='bold', y=1.002,
        )

        def stretch(img):
            tissue = img[mask > 0]
            if tissue.size == 0 or tissue.max() == tissue.min():
                return img.astype(np.float32)
            lo, hi = np.percentile(tissue, [1, 99])
            return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)

        # Row 0 — DAPI overview + positivity bar chart
        ax_dapi = fig.add_subplot(gs[0, 0])
        ax_dapi.imshow(stretch(dapi_img), cmap='gray', interpolation='nearest')
        ax_dapi.set_title('DAPI', fontsize=10)
        ax_dapi.axis('off')

        fig.add_subplot(gs[0, 1]).axis('off')  # blank

        ax_score = fig.add_subplot(gs[0, 2])
        scores    = df_slice[[f'pos_{ch}' for ch in MARKER_CHANNELS]].sum(axis=1).values.astype(np.float32)
        score_map = build_label_map(mask, df_slice['cell_id'].values, scores) if n_cells > 0 else np.zeros(mask.shape, dtype=np.float32)
        im = ax_score.imshow(score_map, cmap='hot', vmin=0, vmax=n_markers, interpolation='nearest')
        plt.colorbar(im, ax=ax_score, fraction=0.03, pad=0.02)
        ax_score.set_title(f'Total marker score (0–{n_markers})', fontsize=10)
        ax_score.axis('off')

        ax_bar  = fig.add_subplot(gs[0, 3])
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

        # Rows 1+ — one row per marker
        for row, ch in enumerate(MARKER_CHANNELS, start=1):
            ch_idx   = CHANNEL_IDX[ch]
            ch_img   = volume_slice[ch_idx]
            thresh_t = float(df_slice[f'thresh_{ch}'].iloc[0]) if n_cells > 0 else 0.0
            means    = df_slice[f'mean_{ch}'].values if n_cells > 0 else np.array([])
            pos_arr  = df_slice[f'pos_{ch}'].values  if n_cells > 0 else np.array([])

            ax_raw = fig.add_subplot(gs[row, 0])
            ax_raw.imshow(stretch(ch_img), cmap='gray', interpolation='nearest')
            ax_raw.set_title(f'{ch}  raw', fontsize=9)
            ax_raw.axis('off')

            ax_proc = fig.add_subplot(gs[row, 1])
            if normalized_slices and ch in normalized_slices:
                proc_img = normalized_slices[ch]
                tissue   = proc_img[mask > 0]
                if tissue.size > 0 and tissue.max() > tissue.min():
                    lo, hi = np.percentile(tissue, [1, 99])
                    proc_disp = np.clip((proc_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
                else:
                    proc_disp = proc_img.astype(np.float32)
                ax_proc.imshow(proc_disp, cmap='gray', interpolation='nearest')
                ax_proc.set_title(f'{ch}  processed\n(clipped+log1p)', fontsize=9)
            else:
                ax_proc.text(0.5, 0.5, 'N/A', ha='center', va='center',
                             transform=ax_proc.transAxes, fontsize=10)
                ax_proc.set_title(f'{ch}  processed', fontsize=9)
            ax_proc.axis('off')

            ax_pos = fig.add_subplot(gs[row, 2])
            if n_cells > 0:
                pos_map = build_label_map(mask, df_slice['cell_id'].values, pos_arr.astype(np.float32))
                outlier_ceiling = float(np.percentile(means, 99.9)) if len(means) > 100 else np.inf
                outlier_mask    = means > outlier_ceiling
                outlier_map     = build_label_map(mask, df_slice['cell_id'].values, outlier_mask.astype(np.float32))
            else:
                pos_map = outlier_map = np.zeros(mask.shape, dtype=np.float32)
                outlier_mask = np.array([], dtype=bool)

            rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
            rgb[(mask > 0) & (pos_map == 0)] = [0.4, 0.05, 0.05]
            rgb[pos_map == 1]                = [0.2, 0.9,  0.2]
            rgb[outlier_map == 1]            = [0.9, 0.9,  0.2]
            ax_pos.imshow(rgb, interpolation='nearest')
            n_pos = int(pos_arr.sum()) if n_cells > 0 else 0
            n_out = int(outlier_mask.sum()) if len(outlier_mask) else 0
            pct   = 100.0 * n_pos / n_cells if n_cells > 0 else 0.0
            ax_pos.set_title(f'{ch} positivity ({n_pos}/{n_cells}, {pct:.0f}%) | {n_out} rejected', fontsize=9)
            ax_pos.axis('off')

            ax_hist = fig.add_subplot(gs[row, 3])
            if len(means) > 1:
                # Show raw pixel intensity distribution (tissue pixels only) — QuPath-style
                ch_raw   = volume_slice[ch_idx]
                px_vals  = ch_raw[mask > 0].astype(np.float32)
                ax_hist.hist(px_vals, bins=256, color='steelblue', alpha=0.75, edgecolor='none')
                otsu_v  = float(df_slice[f'otsu_{ch}'].iloc[0])
                floor_v = float(df_slice[f'floor_{ch}'].iloc[0])
                t_type  = df_slice[f'thresh_type_{ch}'].iloc[0]

                # Thresholds are in log1p space — convert to raw for display
                ax_hist.axvline(np.expm1(thresh_t), color='red', linewidth=2.0,
                                label=f'Threshold = {np.expm1(thresh_t):.1f} ({t_type})')

                raw_col = f'thresh_raw_{ch}'
                if raw_col in df_slice.columns:
                    raw_t = float(df_slice[raw_col].iloc[0])
                    if not np.isclose(raw_t, thresh_t):
                        ax_hist.axvline(np.expm1(raw_t), color='red', linewidth=1.2,
                                        linestyle=':', alpha=0.6,
                                        label=f'Raw slice thresh = {np.expm1(raw_t):.1f}')

                ax_hist.axvline(np.expm1(otsu_v),  color='orange', linewidth=1.2, linestyle='--',
                                label=f'Otsu = {np.expm1(otsu_v):.1f}')
                ax_hist.axvline(np.expm1(floor_v), color='green',  linewidth=1.2, linestyle='--',
                                label=f'Floor = {np.expm1(floor_v):.1f}')
                ax_hist.legend(fontsize=7)

            ax_hist.set_yscale('log')
            ax_hist.set_xlabel('Raw pixel intensity', fontsize=8)
            ax_hist.set_ylabel('Pixel count (log scale)', fontsize=8)
            ax_hist.set_title(f'{ch}  pixel intensity distribution', fontsize=9)
            ax_hist.tick_params(labelsize=7)

        fig.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  QC plot saved: {os.path.basename(out_path)}')

    except Exception as exc:
        logger.warning(f'  QC plot failed for slice {slice_id}: {exc}')


def save_single_channel_qc(
    mask: np.ndarray,
    df_slice: pd.DataFrame,
    raw_img: np.ndarray,
    proc_img: np.ndarray,
    ch_name: str,
    slice_id: int,
    out_path: str,
) -> None:
    """
    Save a high-resolution 1×5 QC panel for a single marker channel.

    Panels (left → right):
      1. Raw image (contrast-stretched)
      2. Pixel intensity histogram (QuPath-style) with threshold lines
      3. Positivity overlay — Otsu threshold
      4. Positivity overlay — Auto threshold (GMM / floor-corrected)
      5. Processed image (log1p, contrast-stretched)

    Negative cells = dark red, positive cells = green, outliers = yellow.
    This layout lets a histologist directly compare the two thresholding
    strategies on the actual tissue image.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(30, 6))
        gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.12)
        n_cells = len(df_slice)

        thresh_t = float(df_slice[f'thresh_{ch_name}'].iloc[0]) if n_cells > 0 else 0.0
        otsu_v   = float(df_slice[f'otsu_{ch_name}'].iloc[0])   if n_cells > 0 else 0.0
        floor_v  = float(df_slice[f'floor_{ch_name}'].iloc[0])  if n_cells > 0 else 0.0
        t_type   = df_slice[f'thresh_type_{ch_name}'].iloc[0]   if n_cells > 0 else ''
        means    = df_slice[f'mean_{ch_name}'].values if n_cells > 0 else np.array([])

        fig.suptitle(
            f'Slice {slice_id:03d}  |  {ch_name}  |  {n_cells} cells  '
            f'|  Auto thresh = {thresh_t:.2f} ({t_type})  |  Otsu = {otsu_v:.2f}',
            fontsize=14, fontweight='bold', y=1.03,
        )

        def stretch(img):
            tissue = img[mask > 0]
            if tissue.size == 0 or tissue.max() == tissue.min():
                return img.astype(np.float32)
            lo, hi = np.percentile(tissue, [1, 99])
            return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)

        def make_overlay(threshold_val):
            """Build an RGB positivity overlay image for a given threshold."""
            if n_cells == 0:
                return np.zeros((*mask.shape, 3), dtype=np.float32)
            pos_arr_t = (means >= threshold_val).astype(np.float32)
            pos_map   = build_label_map(mask, df_slice['cell_id'].values, pos_arr_t)
            outlier_ceiling = float(np.percentile(means, 99.9)) if len(means) > 100 else np.inf
            outlier_map = build_label_map(
                mask, df_slice['cell_id'].values,
                (means > outlier_ceiling).astype(np.float32),
            )
            rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
            rgb[(mask > 0) & (pos_map == 0)] = [0.4, 0.05, 0.05]
            rgb[pos_map == 1]                = [0.2, 0.9,  0.2]
            rgb[outlier_map == 1]            = [0.9, 0.9,  0.2]
            return rgb

        # ── Panel 1: Raw image ───────────────────────────────────────────────
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_raw.imshow(stretch(raw_img), cmap='gray', interpolation='nearest')
        ax_raw.set_title(f'{ch_name}  Raw', fontsize=12)
        ax_raw.axis('off')

        # ── Panel 2: Pixel intensity histogram (QuPath-style) ───────────────
        ax_hist = fig.add_subplot(gs[0, 1])
        px_raw = raw_img[mask > 0].astype(np.float32)
        ax_hist.hist(px_raw, bins=256, color='#4a90d9', alpha=0.75, edgecolor='none')
        ax_hist.set_yscale('log')
        # Thresholds are in log1p space — convert to raw for display
        ax_hist.axvline(np.expm1(thresh_t), color='red',    linewidth=2.0,
                        label=f'Auto = {np.expm1(thresh_t):.1f} ({t_type})')
        ax_hist.axvline(np.expm1(otsu_v),   color='orange', linewidth=1.8, linestyle='--',
                        label=f'Otsu = {np.expm1(otsu_v):.1f}')
        ax_hist.axvline(np.expm1(floor_v),  color='green',  linewidth=1.5, linestyle='--',
                        label=f'Floor = {np.expm1(floor_v):.1f}')
        raw_col = f'thresh_raw_{ch_name}'
        if raw_col in df_slice.columns and n_cells > 0:
            raw_t = float(df_slice[raw_col].iloc[0])
            if not np.isclose(raw_t, thresh_t):
                ax_hist.axvline(np.expm1(raw_t), color='red', linewidth=1.2, linestyle=':',
                                alpha=0.6, label=f'Raw slice = {np.expm1(raw_t):.1f}')
        ax_hist.set_xlabel('Raw pixel intensity', fontsize=10)
        ax_hist.set_ylabel('Pixel count (log scale)', fontsize=10)
        ax_hist.set_title(f'{ch_name}  Pixel intensity\n(QuPath-style)', fontsize=11)
        ax_hist.legend(fontsize=8)

        # ── Panel 3: Positivity overlay — Otsu ──────────────────────────────
        ax_otsu = fig.add_subplot(gs[0, 2])
        rgb_otsu = make_overlay(otsu_v)
        ax_otsu.imshow(rgb_otsu, interpolation='nearest')
        if n_cells > 0:
            n_pos_otsu = int((means >= otsu_v).sum())
            pct_otsu   = 100.0 * n_pos_otsu / n_cells
            ax_otsu.set_title(
                f'Otsu threshold = {otsu_v:.2f}\n'
                f'{n_pos_otsu}/{n_cells} positive ({pct_otsu:.0f}%)',
                fontsize=11,
            )
        else:
            ax_otsu.set_title(f'Otsu threshold = {otsu_v:.2f}', fontsize=11)
        ax_otsu.axis('off')

        # ── Panel 4: Positivity overlay — Auto (GMM / floor) ────────────────
        ax_auto = fig.add_subplot(gs[0, 3])
        rgb_auto = make_overlay(thresh_t)
        ax_auto.imshow(rgb_auto, interpolation='nearest')
        if n_cells > 0:
            n_pos_auto = int((means >= thresh_t).sum())
            pct_auto   = 100.0 * n_pos_auto / n_cells
            ax_auto.set_title(
                f'Auto threshold = {thresh_t:.2f}  [{t_type}]\n'
                f'{n_pos_auto}/{n_cells} positive ({pct_auto:.0f}%)',
                fontsize=11,
            )
        else:
            ax_auto.set_title(f'Auto threshold = {thresh_t:.2f}', fontsize=11)
        ax_auto.axis('off')

        # ── Panel 5: Processed (log1p, stretched) ───────────────────────────
        ax_proc = fig.add_subplot(gs[0, 4])
        if proc_img is not None:
            tissue = proc_img[mask > 0]
            if tissue.size > 0 and tissue.max() > tissue.min():
                lo, hi = np.percentile(tissue, [1, 99])
                proc_disp = np.clip(
                    (proc_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1
                )
            else:
                proc_disp = proc_img.astype(np.float32)
            ax_proc.imshow(proc_disp, cmap='gray', interpolation='nearest')
            ax_proc.set_title(f'{ch_name}  Processed\n(log1p)', fontsize=11)
        else:
            ax_proc.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax_proc.set_title(f'{ch_name}  Processed', fontsize=11)
        ax_proc.axis('off')

        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as exc:
        logger.warning(f'  High-res QC plot failed for {ch_name} slice {slice_id}: {exc}')


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
        logger.error('Run denoise_volume.py first to generate the denoised volume.')
        sys.exit(1)
    if not os.path.isdir(DAPI_MASK_DIR):
        logger.error(f'DAPI mask directory not found: {DAPI_MASK_DIR}')
        sys.exit(1)

    # ── Load denoised volume ──────────────────────────────────────────────────
    logger.info('Loading denoised volume ...')
    vol = tifffile.imread(DENOISED_VOL)

    if vol.ndim == 4:
        # Detect CZYX storage order and convert to ZCYX
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

    # ── Slice ID → Z-index mapping from registration stats CSV ───────────────
    reg_stats_csv = os.path.join(
        config.DATASPACE,
        'Filter_AKAZE_RoMaV2_Linear_Warp_map',
        TARGET_CORE,
        'registration_stats_AKAZE_RoMaV2_Linear.csv',
    )
    slice_id_to_z = {}
    if os.path.exists(reg_stats_csv):
        reg_df = pd.read_csv(reg_stats_csv)
        slice_id_to_z = dict(
            zip(reg_df['Slice_ID'].astype(int), reg_df['Slice_Z'].astype(int))
        )
        # Infer a single unmapped slice if exactly one Z is missing
        all_z     = set(range(n_slices))
        missing_z = all_z - set(slice_id_to_z.values())
        if len(missing_z) == 1:
            mask_ids = {
                get_slice_id(p)
                for p in glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif'))
            }
            unmapped_ids = mask_ids - set(slice_id_to_z.keys())
            if len(unmapped_ids) == 1:
                anchor_id = unmapped_ids.pop()
                anchor_z  = missing_z.pop()
                slice_id_to_z[anchor_id] = anchor_z
                logger.info(f'Anchor slice inferred: ID={anchor_id} → Z={anchor_z}')
        logger.info(f'Slice ID→Z mapping loaded from CSV: {len(slice_id_to_z)} entries')
    else:
        logger.warning(
            f'Registration stats CSV not found at {reg_stats_csv}. '
            f'Falling back to sorted-order matching.'
        )

    mask_files = sorted(
        glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif')),
        key=get_slice_id,
    )
    if not mask_files:
        logger.error(f'No DAPI mask files found in {DAPI_MASK_DIR}')
        sys.exit(1)
    logger.info(f'Found {len(mask_files)} DAPI mask files.')

    gmm_plot_dir = GMM_QC_DIR if args.plot_qc else None
    all_records  = []

    for enum_idx, mask_path in enumerate(mask_files):
        slice_id = get_slice_id(mask_path)

        if slice_id_to_z and slice_id in slice_id_to_z:
            z_idx = slice_id_to_z[slice_id]
        else:
            z_idx = enum_idx
            if slice_id_to_z:
                logger.warning(f'  Slice ID {slice_id} not in CSV mapping — using position {enum_idx}.')

        logger.info(f'  Slice Z={z_idx:03d}  ID={slice_id:03d}  ({enum_idx + 1}/{len(mask_files)})')

        mask = tifffile.imread(mask_path).astype(np.uint32)
        if mask.ndim != 2:
            logger.warning(f'  Unexpected mask shape {mask.shape} — squeezing.')
            mask = mask.squeeze()

        if mask.shape != (H, W):
            logger.warning(f'  Mask shape {mask.shape} != volume ({H}, {W}) — skipping slice.')
            continue
        if z_idx >= n_slices:
            logger.warning(f'  z_idx {z_idx} out of range for volume ({n_slices} slices) — skipping.')
            continue

        df_slice, normalized_slices = measure_slice(
            mask=mask,
            volume_slice=vol[z_idx],
            min_area=args.min_area_px,
            slice_id=slice_id,
            plot_dir=gmm_plot_dir,
        )

        if df_slice.empty:
            logger.warning(f'  No cells found in slice {slice_id} (after min_area filter).')
            continue

        df_slice.insert(0, 'slice_id', slice_id)
        df_slice.insert(0, 'slice_z',  z_idx)
        df_slice.insert(0, 'core',     TARGET_CORE)

        n_pos_summary = {ch: int(df_slice[f'pos_{ch}'].sum()) for ch in MARKER_CHANNELS}
        logger.info(
            f'  {len(df_slice)} nuclei | '
            + '  '.join(f'{ch}+={n}' for ch, n in n_pos_summary.items())
        )

        all_records.append(df_slice)

        if args.plot_qc:
            qc_path  = os.path.join(QC_DIR, f'TMA_{slice_id:03d}_DAPI_phenotype_qc.png')
            dapi_img = vol[z_idx][CHANNEL_IDX['DAPI']]
            save_qc_plot(dapi_img, mask, df_slice, vol[z_idx], slice_id, qc_path,
                         normalized_slices=normalized_slices)

            # --- NEW HIGH-RES PER-CHANNEL LOGIC ---
            # Create a dedicated slice subdirectory to avoid polluting the main QC folder
            channel_qc_dir = os.path.join(QC_DIR, 'per_channel', f'Slice_{slice_id:03d}')
            os.makedirs(channel_qc_dir, exist_ok=True)
            
            for ch in MARKER_CHANNELS:
                ch_idx = CHANNEL_IDX[ch]
                ch_out_path = os.path.join(channel_qc_dir, f'{ch}_highres_qc.png')
                
                # Fetch the log-normalized slice created during measurement
                proc_img = normalized_slices.get(ch) if normalized_slices else None
                
                save_single_channel_qc(
                    mask=mask,
                    df_slice=df_slice,
                    raw_img=vol[z_idx][ch_idx],
                    proc_img=proc_img,
                    ch_name=ch,
                    slice_id=slice_id,
                    out_path=ch_out_path
                )

    if not all_records:
        logger.error('No cells phenotyped across any slice — no CSV written.')
        sys.exit(1)

    df_all = pd.concat(all_records, ignore_index=True)

    if not args.no_consensus:
        df_all = apply_consensus_thresholds(df_all)
    else:
        logger.info('Cross-slice consensus threshold skipped (--no_consensus).')

    meta_cols   = ['core', 'slice_z', 'slice_id', 'cell_id',
                   'area_px', 'area_um2', 'centroid_x', 'centroid_y']
    mean_cols   = [f'mean_{ch}'        for ch in MARKER_CHANNELS]
    pos_cols    = [f'pos_{ch}'         for ch in MARKER_CHANNELS]
    thresh_cols = [f'thresh_{ch}'      for ch in MARKER_CHANNELS]
    raw_t_cols  = [f'thresh_raw_{ch}'  for ch in MARKER_CHANNELS
                   if f'thresh_raw_{ch}' in df_all.columns]
    otsu_cols   = [f'otsu_{ch}'        for ch in MARKER_CHANNELS]
    floor_cols  = [f'floor_{ch}'       for ch in MARKER_CHANNELS]
    t_type_cols = [f'thresh_type_{ch}' for ch in MARKER_CHANNELS]

    df_all = df_all[meta_cols + mean_cols + pos_cols + thresh_cols
                    + raw_t_cols + otsu_cols + floor_cols + t_type_cols]

    csv_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes.csv')
    df_all.to_csv(csv_path, index=False)

    total_cells = len(df_all)
    logger.info('=' * 60)
    logger.info(f'Done.  Core: {TARGET_CORE}  |  Total nuclei: {total_cells}')
    logger.info(f'Output CSV: {csv_path}')
    for ch in MARKER_CHANNELS:
        n_pos  = int(df_all[f'pos_{ch}'].sum())
        pct    = 100.0 * n_pos / total_cells if total_cells > 0 else 0.0
        thresh = float(df_all[f'thresh_{ch}'].iloc[0])
        t_type = df_all[f'thresh_type_{ch}'].iloc[0]
        logger.info(
            f'  {ch:8s}: {n_pos:6d} positive ({pct:5.1f} %)  '
            f'thresh={thresh:.3f}  [{t_type}]'
        )
    logger.info('=' * 60)


if __name__ == '__main__':
    main()