"""
phenotype_cells.py
==================
Cell phenotyping based on DAPI-segmented nuclei and registered multi-channel images.
Integrates Log1p Normalization and Unsupervised KDE Thresholding.

Pipeline per core:
  1. Load the registered volume  (<CORE>_AKAZE_RoMaV2_Linear_Aligned.ome.tif)  — ZCYX uint16
  2. For each slice Z:
       a. Load the warped DAPI mask  (CellPose_DAPI_Warped/<CORE>/TMA_<ID>_DAPI_cp_masks.tif)
       b. For each marker channel (CD31, GAP43, NFP, CD3, CD163, CK, AF):
            - Apply log1p normalization to compress extreme artifacts.
            - Extract per-nucleus mean intensity using the DAPI mask labels.
            - Compute KDE-based thresholding (valley between populations).
            - Assign binary positivity (0/1) per cell per marker.
  3. Concatenate all slices → one CSV per core
  4. Apply cross-slice consensus threshold (median across slices) and recompute positivity.

Key improvements over v1
------------------------
  - Narrow per-marker KDE bandwidth (MARKER_KDE_BW) prevents over-smoothing that
    collapsed CD163's bimodal distribution into a single valley at 1.19.
  - Prominence + height + distance filters on find_peaks() suppress micro-peaks
    from noise and ensure only biologically meaningful populations are detected.
  - Unimodal fallback uses Otsu instead of the arbitrary 95th percentile; a
    suspicious-positivity guard escalates to the 90th percentile when Otsu also
    produces an implausibly high call rate.
  - Biologically plausible positivity ceilings (MARKER_MAX_PCT) act as a final
    safety net mirroring the pathologist review step in the Backman et al. pipeline.
  - Cross-slice consensus threshold: after all slices are processed, the per-marker
    median threshold is applied globally and positivity is recomputed, matching the
    inForm approach of deriving one threshold from representative samples and
    applying it to all images.

Output columns
--------------
    core, slice_z, slice_id, cell_id,
    mean_CD31, mean_GAP43, mean_NFP, mean_CD3, mean_CD163, mean_CK, mean_AF,
    pos_CD31,  pos_GAP43,  pos_NFP,  pos_CD3,  pos_CD163,  pos_CK,  pos_AF,
    area_px, area_um2, centroid_x, centroid_y,
    thresh_CD31, thresh_GAP43, thresh_NFP, thresh_CD3, thresh_CD163, thresh_CK, thresh_AF,
    otsu_CD31, otsu_GAP43..., floor_CD31..., thresh_type_CD31...

Usage
-----
    python phenotype_cells.py --core_name Core_01 [--plot_qc] [--min_area_px 200]
                              [--no_consensus]
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
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from skimage.morphology import binary_erosion, disk
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
MARKER_CHANNELS  = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
CHANNEL_IDX      = {name: i for i, name in enumerate(CHANNEL_NAMES)}
PIXEL_SIZE_XY_UM = 0.4961

# Per-marker KDE bandwidth in log1p space.
# Narrower values (0.10-0.15) resolve tight bimodal distributions (CD163, CK).
# Wider values (0.18-0.22) smooth noisy unimodal distributions without creating
# spurious peaks.  Scott's rule (the scipy default) is equivalent to ~0.25-0.35
# for 20 k+ cells and systematically over-smoothes shoulder populations.
MARKER_KDE_BW = {
    'CD31':  0.15,   # tighten slightly
    'GAP43': 0.18,
    'NFP':   0.15,   # tighten — needs to find any shoulder that exists
    'CD3':   0.12,   # already working well, tighten slightly
    'CD163': 0.10,   # narrowest — bimodal but valley keeps being missed
    'CK':    0.10,   # already working, maintain
    'AF':    0.15,   # tighten to prevent calling bulk AF population
}

# Biologically plausible maximum positivity fraction per marker.
# If the computed threshold calls more than this fraction positive the threshold
# is pushed rightward to match the ceiling.  These values are intentionally
# conservative; adjust for your tissue type.
MARKER_MAX_PCT = {
    'CD31':  0.85,   # Endothelium can be highly dense in vascularized cores
    'GAP43': 0.85,   # Nerve bundles can dominate a small punch
    'NFP':   0.50,   
    'CD3':   0.90,   # Dense lymphoid aggregates/nodes
    'CD163': 0.90,   # Heavy macrophage clusters
    'CK':    0.98,   # Solid tumor cores can be almost entirely epithelial
    'AF':    0.20,   # Autofluorescence remains naturally lower
}

# Minimum positivity fraction below which Otsu is considered to have found
# only noise and the threshold is escalated.
MARKER_MIN_PCT = {ch: 0.001 for ch in MARKER_CHANNELS}   # 0.1 % floor


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Phenotype cells: measure marker expression in DAPI-segmented nuclei.'
)
parser.add_argument('--core_name',    type=str, required=True)
parser.add_argument('--min_area_px',  type=int, default=200,
                    help='Minimum nucleus area in pixels to include (default: 200).')
parser.add_argument('--plot_qc',      action='store_true',
                    help='Save per-slice QC overlay images and KDE density plots.')
parser.add_argument('--no_consensus', action='store_true',
                    help='Skip cross-slice consensus threshold step (keep per-slice thresholds).')
parser.add_argument('--denoised_vol', type=str, default=None,
                    help='Path to a pre-denoised OME-TIFF produced by denoise_volume.py. '
                         'If omitted, falls back to the raw registered volume with the '
                         'original per-channel clip + log1p path.')
args = parser.parse_args()

TARGET_CORE = args.core_name

# Path to the pre-denoised volume (produced by denoise_volume.py).
# Auto-resolved if --denoised_vol is not supplied on the command line.
_AUTO_DENOISED = os.path.join(
    config.DATASPACE,
    'Denoised',
    TARGET_CORE,
    f'{TARGET_CORE}_denoised.ome.tif',
)
DENOISED_VOL = args.denoised_vol if args.denoised_vol else _AUTO_DENOISED

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REGISTERED_VOL = os.path.join(
    config.DATASPACE,
    'Filter_AKAZE_RoMaV2_Linear_Warp_map',
    TARGET_CORE,
    f'{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif',
)

DAPI_MASK_DIR = os.path.join(
    config.DATASPACE,
    'CellPose_DAPI_Warped',
    TARGET_CORE,
)

OUTPUT_DIR = os.path.join(config.DATASPACE, 'Phenotypes', TARGET_CORE)
QC_DIR     = os.path.join(OUTPUT_DIR, 'qc_plots')
KDE_QC_DIR = os.path.join(QC_DIR, 'kde_validation')

os.makedirs(OUTPUT_DIR, exist_ok=True)
if args.plot_qc:
    os.makedirs(QC_DIR, exist_ok=True)
    os.makedirs(KDE_QC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES & THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id(path: str) -> int:
    m = re.search(r'TMA_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def apply_log_normalization(image_slice: np.ndarray) -> np.ndarray:
    """Applies log1p normalization to compress extreme artifacts."""
    return np.log1p(image_slice.astype(np.float32))


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method on the log1p-normalised per-cell means."""
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
    prob      = counts / total
    # Vectorized between-class variance — avoids a 256-iteration Python loop
    w0       = np.cumsum(prob)
    mu0_sum  = np.cumsum(prob * bin_centres)
    w1       = 1.0 - w0
    mu_total = float(np.sum(prob * bin_centres))
    with np.errstate(invalid='ignore', divide='ignore'):
        mu0 = np.where(w0 > 0, mu0_sum / w0, 0.0)
        mu1 = np.where(w1 > 0, (mu_total - mu0_sum) / w1, 0.0)
    between_var = w0 * w1 * (mu0 - mu1) ** 2
    best_t = float(bin_centres[np.argmax(between_var)])
    return best_t


def background_floor(values: np.ndarray, k: float = 3.0) -> float:
    """
    MAD-based noise floor computed strictly on the bottom 50% of the cellular 
    data to isolate the negative population and prevent dense positive 
    cores from artificially inflating the baseline.
    """
    if len(values) < 2:
        return 0.0
        
    # Isolate the left side of the distribution to model the negative population
    negative_pop = values[values < np.percentile(values, 50)]
    
    if len(negative_pop) == 0:
        return 0.0
        
    med = float(np.median(negative_pop))
    mad = float(np.median(np.abs(negative_pop - med)))
    
    # Fallback for highly uniform sparse data
    if mad == 0.0:
        return med
        
    return med + k * mad


def _apply_positivity_ceiling(
    threshold: float,
    values: np.ndarray,
    marker_name: str,
    t_type: str,
) -> tuple:
    """
    Push the threshold rightward if the fraction of positive cells exceeds
    MARKER_MAX_PCT for this marker.  Returns (adjusted_threshold, updated_type).
    """
    max_pct = MARKER_MAX_PCT.get(marker_name, 0.90)
    pos_frac = float((values >= threshold).mean())

    if pos_frac > max_pct:
        # Set threshold to the percentile that matches the ceiling
        target_pct = (1.0 - max_pct) * 100.0
        new_thresh = float(np.percentile(values, target_pct))
        logger.warning(
            f'    [{marker_name}] Positivity {pos_frac:.1%} > ceiling {max_pct:.0%} '
            f'— threshold raised from {threshold:.3f} to {new_thresh:.3f} '
            f'(ceiling_corrected)'
        )
        return new_thresh, t_type + '_ceiling_corrected'

    return threshold, t_type


def density_based_threshold(
    values: np.ndarray,
    marker_name: str = 'Marker',
    slice_id: int = 0,
    plot_dir: str = None,
) -> tuple:
    """
    Unsupervised KDE thresholding replicating the Backman et al. inForm approach.

    Strategy
    --------
    1.  Fit a Gaussian KDE with a narrow, marker-specific bandwidth so that
        shoulder populations (e.g. CD163+) are not merged into the dominant peak.
    2.  Detect peaks with prominence and height filters to suppress noise bumps.
    3.  If ≥ 2 prominent peaks exist, place the threshold at the deepest valley
        between the two tallest peaks  → type 'kde_valley'.
    4.  If the distribution is unimodal, fall back to Otsu  → type 'otsu_fallback'.
        A suspicious-positivity guard escalates to the 90th percentile when Otsu
        still calls too many cells positive  → type 'high_pct_fallback'.
    5.  The positivity ceiling (MARKER_MAX_PCT) is applied after floor clamping
        in measure_slice(); the ceiling guard here is an internal diagnostic only.

    Returns
    -------
    (threshold: float, threshold_type: str)
    """
    if len(values) < 2:
        val = float(values[0]) if len(values) == 1 else 0.0
        return val, 'sparse'

    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        return vmax, 'uniform'

    # ── 1. KDE with narrow, marker-specific bandwidth ──────────────────────
    bw = MARKER_KDE_BW.get(marker_name, 0.15)
    try:
        kde = gaussian_kde(values, bw_method=bw)
    except np.linalg.LinAlgError:
        # Singular matrix — data has near-zero variance
        return float(np.percentile(values, 90)), 'singular_fallback'

    x_space = np.linspace(vmin, vmax, 1000)
    density  = kde(x_space)
    max_dens = float(density.max())

    # ── 2. Prominence-filtered peak detection ──────────────────────────────
    # prominence ≥ 5 % of mode height eliminates noise ridges.
    # distance ≥ 50 grid points (~5 % of the value range) prevents
    # two peaks being called on the same population shoulder.
    peaks, peak_props = find_peaks(
        density,
        prominence=max_dens * 0.05,
        height=max_dens * 0.02,
        distance=50,
    )
    valleys, _ = find_peaks(
        -density,
        prominence=max_dens * 0.01,
        distance=20,
    )

    threshold     = None
    threshold_type = 'otsu_fallback'

    # ── 3. Bimodal path: deepest valley between the two tallest peaks ──────
    if len(peaks) >= 2:
        # Rank peaks by height (descending) and take the top two
        sorted_peaks = sorted(peaks, key=lambda idx: density[idx], reverse=True)
        peak_a, peak_b = sorted(sorted_peaks[:2])   # left, right in x-space

        valleys_between = [v for v in valleys if peak_a < v < peak_b]

        if valleys_between:
            # Deepest valley = minimum KDE density
            best_valley = min(valleys_between, key=lambda v: density[v])
            threshold   = float(x_space[best_valley])
            threshold_type = 'kde_valley'
            logger.debug(
                f'    [{marker_name}] kde_valley at {threshold:.3f} '
                f'(peaks at {x_space[peak_a]:.2f} and {x_space[peak_b]:.2f})'
            )

    # ── 4. Unimodal fallback: Otsu with suspicious-positivity escalation ───
    if threshold is None:
        threshold      = otsu_threshold(values)
        threshold_type = 'otsu_fallback'

        pos_frac = float((values >= threshold).mean())
        max_pct  = MARKER_MAX_PCT.get(marker_name, 0.90)

        if pos_frac > max_pct:
            # Otsu landed in the bulk of the distribution — use 90th percentile
            threshold      = float(np.percentile(values, 90.0))
            threshold_type = 'high_pct_fallback'
            logger.debug(
                f'    [{marker_name}] Otsu gave {pos_frac:.1%} positive '
                f'— escalating to 90th pct → {threshold:.3f}'
            )
        elif pos_frac < MARKER_MIN_PCT.get(marker_name, 0.001):
            # Otsu placed the threshold too high (almost nothing passes)
            # Fall back to a more lenient percentile
            threshold      = float(np.percentile(values, 85.0))
            threshold_type = 'low_pct_fallback'
            logger.debug(
                f'    [{marker_name}] Otsu gave {pos_frac:.3%} positive '
                f'— escalating to 85th pct → {threshold:.3f}'
            )

    # ── 5. Diagnostic KDE plot (optional) ──────────────────────────────────
    if plot_dir is not None:
        _save_kde_plot(
            values, x_space, density, peaks, valleys,
            threshold, threshold_type, marker_name, slice_id, plot_dir,
        )

    return threshold, threshold_type


def _save_kde_plot(
    values, x_space, density, peaks, valleys,
    threshold, threshold_type, marker_name, slice_id, plot_dir,
) -> None:
    """Save a standalone KDE topology diagnostic plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=120, density=True, alpha=0.45,
                color='#4a90d9', label='Log-norm data')
        ax.plot(x_space, density, 'b-', linewidth=1.8, label='KDE')

        if len(peaks) > 0:
            ax.plot(x_space[peaks], density[peaks], 'g^',
                    markersize=8, label='Peaks')
        if len(valleys) > 0:
            ax.plot(x_space[valleys], density[valleys], 'rv',
                    markersize=7, label='Valleys')

        ax.axvline(threshold, color='red', linewidth=2.0, linestyle='--',
                   label=f'Threshold = {threshold:.2f} ({threshold_type})')

        bw = MARKER_KDE_BW.get(marker_name, 0.15)
        ax.set_title(
            f'KDE topology — {marker_name}  (slice {slice_id}, bw={bw})',
            fontsize=11,
        )
        ax.set_xlabel('Log1p mean intensity (per nucleus)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()

        out = os.path.join(plot_dir, f'kde_val_{slice_id:03d}_{marker_name}.png')
        fig.savefig(out, dpi=90, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        logger.warning(f'  KDE plot failed for {marker_name} slice {slice_id}: {exc}')



def detect_spatial_artifacts(raw_img: np.ndarray, tissue_mask: np.ndarray, 
                               percentile: float = 99.5,
                               min_artifact_size: int = 5,
                               max_artifact_size: int = 500) -> np.ndarray:
    """
    Detect isolated superbright artifact pixels spatially.
    
    Artifacts are bright compact blobs that are:
      - Above a high percentile threshold
      - Small in area (not real tissue structure)
    
    Returns a boolean mask — True where artifact pixels are.
    """
    if not np.any(tissue_mask):
        return np.zeros_like(raw_img, dtype=bool)
    
    # 1. Threshold: pixels brighter than percentile of tissue
    p_thresh = float(np.percentile(raw_img[tissue_mask], percentile))
    bright_mask = raw_img > p_thresh
    
    # 2. Label connected components of bright regions
    labeled, n_components = ndi.label(bright_mask)
    component_sizes = np.array(ndi.sum(bright_mask, labeled, range(1, n_components + 1)))

    # 3. Keep only small isolated blobs (artifacts), not large bright tissue regions
    # Vectorized: find valid label indices in one shot, then use np.isin for the mask
    valid_labels = np.where(
        (component_sizes >= min_artifact_size) & (component_sizes <= max_artifact_size)
    )[0] + 1  # +1 because ndi labels start at 1
    artifact_mask = np.isin(labeled, valid_labels)
    
    return artifact_mask

# ─────────────────────────────────────────────────────────────────────────────
# PER-SLICE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_slice(
    mask: np.ndarray,
    volume_slice: np.ndarray,
    min_area: int,
    slice_id: int = 0,
    plot_dir: str = None,
    denoised_slice: np.ndarray = None,
) -> tuple:
    """
    Measure per-cell marker intensities and compute per-slice thresholds.
    Integrates pre-normalization artifact clipping and top-end outlier rejection.

    Thresholds stored in the returned DataFrame are the *raw per-slice* values.
    The cross-slice consensus step in main() replaces them before writing the
    final CSV.

    Returns
    -------
    (df : pd.DataFrame, processed_slices : dict[str, np.ndarray])
        processed_slices maps each marker name to its artifact-clipped +
        log1p-normalised 2-D image, used by save_qc_plot for the
        "processed channel" column.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]

    if len(labels) == 0:
        return pd.DataFrame(), {}

    label_list = labels.tolist()
    areas = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int32)

    keep       = areas >= min_area
    label_list = [l for l, k in zip(label_list, keep) if k]
    areas      = areas[keep]

    if len(label_list) == 0:
        return pd.DataFrame(), {}

    # A distance of 4 pixels (~2 um) captures adjacent membrane markers without 
    # diluting the signal-to-noise ratio by bleeding too far into the background.
    cell_mask = expand_labels(mask, distance=4)

    cy = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape),
        mask, label_list,
    ))
    cx = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[1])[None, :], mask.shape),
        mask, label_list,
    ))

    # ── Phase 1: Normalisation and mean extraction ─────────────────────────
    # If a pre-denoised slice (from denoise_volume.py) is supplied, use it
    # directly — it is already the cleaned top-hat in float32, so we only
    # need log1p before extracting per-cell means.
    #
    # Fallback (no denoised volume): original per-channel clip + log1p path.
    using_denoised = denoised_slice is not None
    marker_means      = {}
    normalized_slices = {}

    for ch_name in MARKER_CHANNELS:
        ch_idx = CHANNEL_IDX[ch_name]

        if using_denoised:
            # denoised_slice is uint16 (same scale as raw); convert to float32
            # and apply log1p — identical to what denoise_volume.py produces.
            cleaned = denoised_slice[ch_idx].astype(np.float32)
            norm_img = apply_log_normalization(cleaned)
        else:
            raw_img  = volume_slice[ch_idx].copy()
            tissue_mask = raw_img > 0
            if np.any(tissue_mask):
                artifact_mask = detect_spatial_artifacts(raw_img, tissue_mask,
                                                         percentile=99.5,
                                                         max_artifact_size=5000)
                if np.any(artifact_mask):
                    bg_value = float(np.percentile(raw_img[tissue_mask == 0], 50))
                    raw_img[artifact_mask] = bg_value
                p99_9 = float(np.percentile(raw_img[tissue_mask], 99.9))
                raw_img = np.clip(raw_img, 0, p99_9)
            norm_img = apply_log_normalization(raw_img)

        normalized_slices[ch_name] = norm_img
        means = np.array(ndi.mean(norm_img, cell_mask, label_list), dtype=np.float32)
        marker_means[ch_name] = means

    df = pd.DataFrame({
        'cell_id':    label_list,
        'area_px':    areas,
        'area_um2':   np.round(areas * PIXEL_SIZE_XY_UM ** 2, 3),
        'centroid_x': np.round(cx, 1),
        'centroid_y': np.round(cy, 1),
    })

    # ── Phase 2: per-marker thresholding and top-end outlier rejection
    for ch_name in MARKER_CHANNELS:
        means    = marker_means[ch_name]
        norm_img = normalized_slices[ch_name]

        df[f'mean_{ch_name}'] = np.round(means, 3)

        kde_thresh, t_type = density_based_threshold(
            values=means,
            marker_name=ch_name,
            slice_id=slice_id,
            plot_dir=plot_dir,
        )

        # Reference metrics (kept for QC and backward compatibility)
        # UPDATED: Calculate the floor using the cellular means array
        o_val = otsu_threshold(means)
        f_val = background_floor(means)

        # Floor guard: threshold must not dip below the pixel-level noise floor
        final_thresh = max(kde_thresh, f_val)
        if final_thresh > kde_thresh:
            logger.debug(
                f'    [{ch_name}] KDE threshold {kde_thresh:.3f} < floor {f_val:.3f} '
                f'— raised to floor.'
            )

        # Ceiling guard: threshold must not call an implausible fraction positive
        final_thresh, t_type = _apply_positivity_ceiling(
            final_thresh, means, ch_name, t_type,
        )

        # Base Positivity Call
        pos_array = (means >= final_thresh).astype(np.uint8)

        # Top-End Outlier Rejection: Force extreme biological impossibilities to negative
        if len(means) > 100:
            outlier_ceiling = float(np.percentile(means, 99.9))
            outlier_mask = means > outlier_ceiling
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
    Derive one threshold per marker from the median across all slices and
    recompute positivity.  This mirrors the Backman et al. step where a single
    threshold (derived from pathologist-annotated subsets) is applied globally
    rather than varying per image.

    The original per-slice thresholds are preserved in thresh_raw_<marker>
    columns for audit purposes.  The consensus value is written back into
    thresh_<marker> and pos_<marker> is recomputed.

    Ceiling and floor guards are re-applied after consensus to catch edge cases
    where the median lands outside the safe range for a specific slice.
    """
    logger.info('Applying cross-slice consensus thresholds ...')

    for ch in MARKER_CHANNELS:
        col = f'thresh_{ch}'

        # Preserve original per-slice value
        df_all[f'thresh_raw_{ch}'] = df_all[col]

        # Only use slices whose threshold was derived from actual KDE valleys or
        # Otsu — not from ceiling corrections, which are already biased.
        reliable_mask = ~df_all[f'thresh_type_{ch}'].str.contains(
            'ceiling_corrected|singular_fallback|sparse|uniform', na=False,
        )
        reliable_vals = df_all.loc[reliable_mask, col]

        if reliable_vals.empty:
            logger.warning(
                f'  [{ch}] No reliable per-slice thresholds — keeping per-slice values.'
            )
            continue

        consensus = float(reliable_vals.median())

        # Log change relative to per-slice spread
        spread = float(reliable_vals.std())
        logger.info(
            f'  [{ch}] consensus threshold = {consensus:.3f}  '
            f'(per-slice median ± SD: {reliable_vals.median():.3f} ± {spread:.3f})'
        )

        df_all[col] = consensus

        # Recompute positivity with consensus threshold
        means = df_all[f'mean_{ch}'].values
        pos   = (means >= consensus).astype(np.uint8)

        # Re-apply ceiling guard at the aggregate level
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
            # Mark all rows as ceiling-corrected at consensus stage
            df_all[f'thresh_type_{ch}'] = df_all[f'thresh_type_{ch}'].apply(
                lambda t: t + '_consensus_ceiling' if 'ceiling' not in t else t
            )

        df_all[f'pos_{ch}'] = pos

        # Final summary
        n_pos  = int(pos.sum())
        n_tot  = len(pos)
        logger.info(
            f'  [{ch}] After consensus: {n_pos}/{n_tot} positive '
            f'({100.0 * n_pos / n_tot:.1f} %)'
        )

    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# QC VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_label_map(
    mask: np.ndarray,
    cell_ids,
    values,
    dtype=np.float32,
) -> np.ndarray:
    """Paint a per-cell scalar value onto a label-mask image."""
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=dtype)
    cell_ids = np.asarray(cell_ids)
    values   = np.asarray(values)
    valid    = cell_ids <= max_label
    lut[cell_ids[valid]] = values[valid]
    return lut[mask]


def save_qc_plot(
    dapi_img: np.ndarray,
    mask: np.ndarray,
    df_slice: pd.DataFrame,
    volume_slice: np.ndarray,
    slice_id: int,
    out_path: str,
    processed_slices: dict = None,
) -> None:
    """
    Save a per-marker QC panel showing raw image, processed (artifact-clipped +
    log1p-normalised) image, positivity overlay, and log-intensity histogram
    with all three threshold references.

    Columns per marker row:
      0 – raw channel image
      1 – processed channel image (clipped + log1p)
      2 – positivity overlay
      3 – log-intensity histogram

    The histogram shows four lines:
      - Red solid   : final threshold used for positivity calls
      - Red dashed  : raw per-slice KDE/Otsu threshold (before consensus)
      - Orange dashed: Otsu reference
      - Green dashed : MAD noise floor
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_markers = len(MARKER_CHANNELS)
        n_rows    = 1 + n_markers
        n_cols    = 4                                   # raw | processed | overlay | hist
        fig = plt.figure(figsize=(24, 4 * n_rows))     # wider to fit 4 columns
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.35, wspace=0.15)

        n_cells = len(df_slice)
        fig.suptitle(
            f'{TARGET_CORE}  |  Slice {slice_id}  |  {n_cells} nuclei',
            fontsize=14, fontweight='bold', y=1.002,
        )

        def stretch(img: np.ndarray) -> np.ndarray:
            tissue = img[mask > 0]
            if tissue.size == 0 or tissue.max() == tissue.min():
                return img.astype(np.float32)
            lo, hi = np.percentile(tissue, [1, 99])
            if hi == lo:
                return np.zeros_like(img, dtype=np.float32)
            return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)

        # Row 0 — DAPI overview (col 0), empty col 1, total score (col 2), bar chart (col 3)
        ax_dapi = fig.add_subplot(gs[0, 0])
        ax_dapi.imshow(stretch(dapi_img), cmap='gray', interpolation='nearest')
        ax_dapi.set_title('DAPI', fontsize=10)
        ax_dapi.axis('off')

        # Col 1 of row 0 left blank (no "processed DAPI" — DAPI uses a dedicated
        # denoising path whose output is the mask, not a separate 2-D image)
        ax_blank = fig.add_subplot(gs[0, 1])
        ax_blank.axis('off')

        ax_score = fig.add_subplot(gs[0, 2])
        if n_cells > 0:
            pos_cols  = [f'pos_{ch}' for ch in MARKER_CHANNELS]
            scores    = df_slice[pos_cols].sum(axis=1).values.astype(np.float32)
            score_map = build_label_map(mask, df_slice['cell_id'].values, scores)
        else:
            score_map = np.zeros(mask.shape, dtype=np.float32)
        im = ax_score.imshow(score_map, cmap='hot',
                             vmin=0, vmax=n_markers, interpolation='nearest')
        plt.colorbar(im, ax=ax_score, fraction=0.03, pad=0.02)
        ax_score.set_title(f'Total marker score (0–{n_markers})', fontsize=10)
        ax_score.axis('off')

        ax_bar = fig.add_subplot(gs[0, 3])
        if n_cells > 0:
            pct_pos = [100.0 * df_slice[f'pos_{ch}'].sum() / n_cells
                       for ch in MARKER_CHANNELS]
        else:
            pct_pos = [0] * n_markers
        colors = ['#e74c3c' if p > 50 else '#3498db' for p in pct_pos]
        bars   = ax_bar.barh(MARKER_CHANNELS, pct_pos, color=colors)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('% positive cells', fontsize=9)
        ax_bar.set_title('Positivity summary', fontsize=10)
        for bar, pct in zip(bars, pct_pos):
            ax_bar.text(min(pct + 1, 98), bar.get_y() + bar.get_height() / 2,
                        f'{pct:.0f}%', va='center', fontsize=8)
        ax_bar.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # Rows 1+ — one row per marker: raw | processed | overlay | hist
        for row, ch in enumerate(MARKER_CHANNELS, start=1):
            ch_idx   = CHANNEL_IDX[ch]
            ch_img   = volume_slice[ch_idx]
            thresh_t = float(df_slice[f'thresh_{ch}'].iloc[0]) if n_cells > 0 else 0.0
            means    = df_slice[f'mean_{ch}'].values if n_cells > 0 else np.array([])
            pos_arr  = df_slice[f'pos_{ch}'].values  if n_cells > 0 else np.array([])

            # ── Col 0: Raw channel image ──────────────────────────────────────
            ax_raw = fig.add_subplot(gs[row, 0])
            ax_raw.imshow(stretch(ch_img), cmap='gray', interpolation='nearest')
            ax_raw.set_title(f'{ch}  raw', fontsize=9)
            ax_raw.axis('off')

            # ── Col 1: Processed channel (artifact-clipped + log1p) ──────────
            ax_proc = fig.add_subplot(gs[row, 1])
            if processed_slices is not None and ch in processed_slices:
                proc_img = processed_slices[ch]
                # Stretch the log1p image for display using tissue pixels
                tissue = proc_img[mask > 0]
                if tissue.size > 0 and tissue.max() > tissue.min():
                    lo, hi = np.percentile(tissue, [1, 99])
                    proc_disp = np.clip(
                        (proc_img.astype(np.float32) - lo) / max(hi - lo, 1e-6),
                        0, 1,
                    )
                else:
                    proc_disp = proc_img.astype(np.float32)
                ax_proc.imshow(proc_disp, cmap='gray', interpolation='nearest')
                ax_proc.set_title(f'{ch}  processed\n(clipped+log1p)', fontsize=9)
            else:
                ax_proc.text(0.5, 0.5, 'N/A', ha='center', va='center',
                             transform=ax_proc.transAxes, fontsize=10)
                ax_proc.set_title(f'{ch}  processed', fontsize=9)
            ax_proc.axis('off')

            # ── Col 2: Positivity overlay ─────────────────────────────────────
            ax_pos = fig.add_subplot(gs[row, 2])
            if n_cells > 0:
                pos_map = build_label_map(mask, df_slice['cell_id'].values,
                                          pos_arr.astype(np.float32))
                
                # Re-calculate outlier mask purely for visualization purposes
                outlier_mask = np.zeros(n_cells, dtype=bool)
                if n_cells > 100:
                    outlier_ceiling = float(np.percentile(means, 99.9))
                    outlier_mask = means > outlier_ceiling
                
                outlier_map = build_label_map(mask, df_slice['cell_id'].values,
                                              outlier_mask.astype(np.float32))
            else:
                pos_map = np.zeros(mask.shape, dtype=np.float32)
                outlier_map = np.zeros(mask.shape, dtype=np.float32)
                
            rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
            
            # Base states
            rgb[(mask > 0) & (pos_map == 0)] = [0.4, 0.05, 0.05] # Dark Red (Negative)
            rgb[pos_map == 1]                = [0.2, 0.9, 0.2]   # Green (Positive)
            
            # Preprocessing visualization: Override color for rejected extreme outliers
            rgb[outlier_map == 1]            = [0.9, 0.9, 0.2]   # Yellow (Rejected Artifact)

            ax_pos.imshow(rgb, interpolation='nearest')
            n_pos = int(pos_arr.sum()) if n_cells > 0 else 0
            n_out = int(outlier_mask.sum()) if n_cells > 0 else 0
            pct   = 100.0 * n_pos / n_cells if n_cells > 0 else 0.0
            
            ax_pos.set_title(f'{ch} positivity ({n_pos}/{n_cells}, {pct:.0f}%) | {n_out} rejected',
                             fontsize=9)
            ax_pos.axis('off')

            # ── Col 3: Intensity histogram with threshold references ──────────
            ax_hist = fig.add_subplot(gs[row, 3])
            if len(means) > 1:
                ax_hist.hist(means, bins=80, color='steelblue', alpha=0.75,
                             edgecolor='none')
                otsu_v  = float(df_slice[f'otsu_{ch}'].iloc[0])
                floor_v = float(df_slice[f'floor_{ch}'].iloc[0])
                t_type  = df_slice[f'thresh_type_{ch}'].iloc[0]

                # Final threshold (consensus or per-slice)
                ax_hist.axvline(thresh_t, color='red', linewidth=2.0,
                                label=f'Threshold = {thresh_t:.2f} ({t_type})')

                # Raw per-slice KDE threshold (before consensus) — if available
                raw_col = f'thresh_raw_{ch}'
                if raw_col in df_slice.columns:
                    raw_t = float(df_slice[raw_col].iloc[0])
                    if not np.isclose(raw_t, thresh_t):
                        ax_hist.axvline(raw_t, color='red', linewidth=1.2,
                                        linestyle=':', alpha=0.6,
                                        label=f'Raw slice thresh = {raw_t:.2f}')

                ax_hist.axvline(otsu_v,  color='orange', linewidth=1.2,
                                linestyle='--', label=f'Otsu = {otsu_v:.2f}')
                ax_hist.axvline(floor_v, color='green',  linewidth=1.2,
                                linestyle='--', label=f'Floor = {floor_v:.2f}')
                ax_hist.legend(fontsize=7)

            ax_hist.set_xlabel('Log1p mean intensity (per nucleus)', fontsize=8)
            ax_hist.set_ylabel('Cell count', fontsize=8)
            ax_hist.set_title(f'{ch}  log-intensity distribution', fontsize=9)
            ax_hist.tick_params(labelsize=7)

        fig.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  QC plot saved: {os.path.basename(out_path)}')

    except Exception as exc:
        logger.warning(f'  QC plot failed for slice {slice_id}: {exc}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(f'Phenotyping — core: {TARGET_CORE}')
    logger.info(f'Registered volume : {REGISTERED_VOL}')
    logger.info(f'DAPI mask dir     : {DAPI_MASK_DIR}')

    if not os.path.exists(REGISTERED_VOL):
        logger.error(f'Registered volume not found: {REGISTERED_VOL}')
        sys.exit(1)

    if not os.path.isdir(DAPI_MASK_DIR):
        logger.error(f'DAPI mask directory not found: {DAPI_MASK_DIR}')
        sys.exit(1)

    # ── Load denoised volume (preferred) or raw volume (fallback) ─────────
    if os.path.exists(DENOISED_VOL):
        logger.info(f'Denoised volume found — using it for channel intensities.')
        logger.info(f'Denoised volume : {DENOISED_VOL}')
        den_vol = tifffile.imread(DENOISED_VOL)
        if den_vol.ndim == 4 and den_vol.shape[0] == len(CHANNEL_NAMES) and den_vol.shape[1] != len(CHANNEL_NAMES):
            den_vol = np.moveaxis(den_vol, 0, 1)
    else:
        logger.warning(
            f'Denoised volume not found at {DENOISED_VOL}. '
            f'Falling back to raw volume with per-channel clip + log1p. '
            f'Run denoise_volume.py first for best results.'
        )
        den_vol = None
        sys.exit(1)

    # ── Load registered volume ─────────────────────────────────────────────
    logger.info('Loading registered volume ...')
    vol = tifffile.imread(REGISTERED_VOL)

    if vol.ndim == 3:
        vol = vol[np.newaxis, np.newaxis]
    elif vol.ndim == 4:
        # Ensure ZCYX order
        if vol.shape[0] == len(CHANNEL_NAMES) and vol.shape[1] != len(CHANNEL_NAMES):
            vol = np.moveaxis(vol, 0, 1)

    n_slices, n_channels, H, W = vol.shape
    logger.info(f'Volume shape: Z={n_slices}  C={n_channels}  H={H}  W={W}')

    if n_channels != len(CHANNEL_NAMES):
        logger.warning(
            f'Expected {len(CHANNEL_NAMES)} channels, got {n_channels}. '
            f'Proceeding — verify CHANNEL_NAMES matches your data.'
        )

    # ── Slice ID → Z-index mapping ─────────────────────────────────────────
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

        all_z    = set(range(n_slices))
        mapped_z = set(slice_id_to_z.values())
        missing_z = all_z - mapped_z
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

    # ── Discover DAPI mask files ───────────────────────────────────────────
    mask_files = sorted(
        glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif')),
        key=get_slice_id,
    )
    if not mask_files:
        logger.error(f'No DAPI mask files found in {DAPI_MASK_DIR}')
        sys.exit(1)

    logger.info(f'Found {len(mask_files)} DAPI mask files.')

    kde_plot_dir = KDE_QC_DIR if args.plot_qc else None
    all_records  = []

    # ── Per-slice processing ───────────────────────────────────────────────
    for enum_idx, mask_path in enumerate(mask_files):
        slice_id = get_slice_id(mask_path)

        if slice_id_to_z and slice_id in slice_id_to_z:
            z_idx = slice_id_to_z[slice_id]
        else:
            z_idx = enum_idx
            if slice_id_to_z:
                logger.warning(
                    f'  Slice ID {slice_id} not in CSV mapping — using position {enum_idx}.'
                )

        logger.info(
            f'  Slice Z={z_idx:03d}  ID={slice_id:03d}  '
            f'({enum_idx + 1}/{len(mask_files)})'
        )

        mask = tifffile.imread(mask_path).astype(np.uint32)
        if mask.ndim != 2:
            logger.warning(f'  Unexpected mask shape {mask.shape} — squeezing.')
            mask = mask.squeeze()

        if mask.shape != (H, W):
            logger.warning(
                f'  Mask shape {mask.shape} != volume ({H}, {W}) — skipping slice.'
            )
            continue

        if z_idx >= n_slices:
            logger.warning(
                f'  z_idx {z_idx} out of range for volume ({n_slices} slices) — skipping.'
            )
            continue

        volume_slice = vol[z_idx]
        denoised_slice = den_vol[z_idx] if den_vol is not None else None

        df_slice, processed_slices = measure_slice(
            mask=mask,
            volume_slice=volume_slice,
            min_area=args.min_area_px,
            slice_id=slice_id,
            plot_dir=kde_plot_dir,
            denoised_slice=denoised_slice,
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

        # QC plots use the per-slice thresholds (before consensus) so the
        # raw_thresh columns are not yet populated here; save_qc_plot handles
        # their absence gracefully.
        if args.plot_qc:
            qc_path  = os.path.join(QC_DIR, f'TMA_{slice_id:03d}_DAPI_phenotype_qc.png')
            dapi_img = volume_slice[CHANNEL_IDX['DAPI']]
            save_qc_plot(dapi_img, mask, df_slice, volume_slice, slice_id, qc_path,
                         processed_slices=processed_slices)

    if not all_records:
        logger.error('No cells phenotyped across any slice — no CSV written.')
        sys.exit(1)

    # ── Concatenate all slices ─────────────────────────────────────────────
    df_all = pd.concat(all_records, ignore_index=True)

    # ── Cross-slice consensus threshold ────────────────────────────────────
    if not args.no_consensus:
        df_all = apply_consensus_thresholds(df_all)
    else:
        logger.info('Cross-slice consensus threshold skipped (--no_consensus).')

    # ── Column ordering ────────────────────────────────────────────────────
    meta_cols     = ['core', 'slice_z', 'slice_id', 'cell_id',
                     'area_px', 'area_um2', 'centroid_x', 'centroid_y']
    mean_cols     = [f'mean_{ch}'        for ch in MARKER_CHANNELS]
    pos_cols      = [f'pos_{ch}'         for ch in MARKER_CHANNELS]
    thresh_cols   = [f'thresh_{ch}'      for ch in MARKER_CHANNELS]
    raw_t_cols    = [f'thresh_raw_{ch}'  for ch in MARKER_CHANNELS
                     if f'thresh_raw_{ch}' in df_all.columns]
    otsu_cols     = [f'otsu_{ch}'        for ch in MARKER_CHANNELS]
    floor_cols    = [f'floor_{ch}'       for ch in MARKER_CHANNELS]
    t_type_cols   = [f'thresh_type_{ch}' for ch in MARKER_CHANNELS]

    ordered_cols = (meta_cols + mean_cols + pos_cols + thresh_cols
                    + raw_t_cols + otsu_cols + floor_cols + t_type_cols)
    df_all = df_all[ordered_cols]

    # ── Write output CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes.csv')
    df_all.to_csv(csv_path, index=False)

    total_cells = len(df_all)
    logger.info('=' * 60)
    logger.info(f'Done.  Core: {TARGET_CORE}  |  Total nuclei: {total_cells}')
    logger.info(f'Output CSV: {csv_path}')
    for ch in MARKER_CHANNELS:
        n_pos = int(df_all[f'pos_{ch}'].sum())
        pct   = 100.0 * n_pos / total_cells if total_cells > 0 else 0.0
        thresh = float(df_all[f'thresh_{ch}'].iloc[0])
        t_type = df_all[f'thresh_type_{ch}'].iloc[0]
        logger.info(
            f'  {ch:8s}: {n_pos:6d} positive ({pct:5.1f} %)  '
            f'thresh={thresh:.3f}  [{t_type}]'
        )
    logger.info('=' * 60)


if __name__ == '__main__':
    main()