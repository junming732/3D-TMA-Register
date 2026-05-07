"""
denoise_volume.py
=================
Artifact detection and background correction for all channels of a registered
OME-TIFF volume.  Run this ONCE per core before phenotype_cells.py.

Pipeline per slice × channel (parallelised across channels):
  Stage 1 — ARTIFACT DETECTION
    DAPI:       1a. Large-blob scan — stride-sampled CC scan that catches the
                    known macroscopic bubble by raw size alone.
                1b. Intensity + circularity scan — two-gate CC analysis on the
                    post-1a image (Gate A: any bright blob; Gate B: bright AND
                    round).  Catches residual bright compact debris.
    Non-DAPI:   1c. DoG bandpass scan — Difference of Gaussians filter that
                    suppresses both fine noise and slowly-varying background,
                    boosting only intermediate-scale macro-artifacts.  Candidates
                    filtered by area, circularity, and solidity; top-N by
                    (area × peak DoG response) are masked.
    After detection (all channels): masked pixels are inpainted from local
    neighbours, then the mask is dilated by ~7.5 µm to absorb optical halos.

  Stage 2 — BACKGROUND CORRECTION  (per-channel mode)
    'tophat'   — white morphological top-hat (image − opening).
                 Use for punctate nuclear markers: DAPI, CD3, CD163.
    'gaussian' — subtract a large Gaussian blur of the inpainted image.
                 Use for fibrous/vascular markers: CD31, GAP43, NFP, CK, AF.

Output
------
  <OUTPUT_DIR>/<CORE>/<CORE>_denoised.ome.tif  — ZCYX uint16, same shape as
    input; pixel values are the background-corrected, artifact-zeroed signal
    scaled to uint16 per-channel global max (log1p applied downstream by
    phenotype_cells.py).

Usage
-----
  python denoise_volume.py --core_name Core_01
  python denoise_volume.py --core_name Core_01 --workers 8 --preview_slice 5
"""

import os
import sys
import time
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use('Agg')   # must be set before any other matplotlib import
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Top-hat denoising for all channels of a registered OME-TIFF.'
)
parser.add_argument('--core_name',     type=str, required=True)
parser.add_argument('--workers',       type=int, default=4,
                    help='Parallel threads per slice (default: 4). '
                         'Raise if you have many CPU cores; each channel is '
                         '~150 MB at float32 for a 6112² image.')
parser.add_argument('--pixel_um',      type=float, default=0.4961,
                    help='Pixel size in µm (default: 0.4961).')
parser.add_argument('--dust_pct',      type=float, default=99.0,
                    help='Percentile of positive pixels used as the tissue '
                         'signal ceiling for artifact detection (default: 99). '
                         'Raise toward 99.9 if real bright structures are being '
                         'flagged; lower toward 95 if artifacts slip through.')
parser.add_argument('--preview_slice', type=int, default=None,
                    help='If set, save a per-channel PNG diagnostic for this '
                         'Z-slice (0-indexed) and exit — useful for tuning SE size.')
parser.add_argument('--plot_qc',       action='store_true',
                    help='Save a QC panel (Raw | Top-Hat | Dust | Cleaned | Histogram) '
                         'for every channel of every slice. Written to '
                         'Denoised/<CORE>/qc/<CORE>_Z###_Ch##_<NAME>.png')
parser.add_argument('--overwrite',     action='store_true',
                    help='Re-run even if the output file already exists.')
parser.add_argument('--artifact_max_area_um2', type=float, default=5000.0,
                    help='Any connected bright blob larger than this area (µm²) is '
                         'unconditionally removed as a large artifact (fold, debris, '
                         'dust clump). Default: 5000 µm² ≈ ~20,000 px at 0.5 µm/px. '
                         'Set to 0 to disable. Raise if large real structures '
                         '(vessel cross-sections) are being removed.')
args = parser.parse_args()

TARGET_CORE = args.core_name

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VOL = os.path.join(
    config.DATASPACE,
    'Filter_AKAZE_RoMaV2_Linear_Warp_map',
    TARGET_CORE,
    f'{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif',
)

OUTPUT_DIR  = os.path.join(config.DATASPACE, 'Denoised', TARGET_CORE)
OUTPUT_VOL  = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_denoised.ome.tif')
PREVIEW_DIR = os.path.join(OUTPUT_DIR, 'preview')
QC_DIR      = os.path.join(OUTPUT_DIR, 'qc')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PER-CHANNEL DENOISING STRATEGY CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
# ── DAPI only: Passes 1a + 1b ────────────────────────────────────────────────
#   artifact_max_area_um2  — Pass 1a: blobs larger than this (µm²) are removed
#     by the fast stride-sampled size scan.  Sized to catch DAPI's macroscopic
#     bubble while leaving real nuclei untouched.
#
#   artifact_thresh_factor — Pass 1b Gate A: blob_max > factor × DUST_PCT-pct
#     ceiling → flagged regardless of shape.  High value = extreme outliers only.
#   artifact_circ_factor   — Pass 1b Gate B intensity multiplier.  Must be ≤
#     thresh_factor.  Blobs above this AND with circularity ≥ circ_thresh are
#     flagged.  Catches compact round debris that sits below Gate A.
#   artifact_circ_thresh   — Minimum circularity (0–1) for Gate B.
#
# ── Non-DAPI channels: Pass 1c (DoG) only ────────────────────────────────────
#   Intensity-based detection (Passes 1a/1b) does not reliably separate macro-
#   artifacts from real tissue in these channels.  The DoG bandpass scanner is
#   the sole artifact detector.
#
#   dog_sigma_high_um  — High-sigma Gaussian for the bandpass (µm).
#     Controls the maximum artifact radius the scanner responds to.  Raise if
#     large macro-artifacts are missed; lower to avoid catching real tissue.
#   dog_min_area_um2   — Minimum blob area to be considered a macro-artifact.
#   dog_max_area_um2   — Maximum blob area.  Structures larger than this are
#     likely connected tissue and are ignored.
#   dog_min_circ       — Minimum circularity for DoG candidates (kept loose,
#     ~0.20, because macro-artifact edges are often irregular).
#   dog_min_solid      — Minimum solidity.
#   dog_top_n          — Maximum artifacts removed per slice.  Conservative to
#     avoid over-masking.
#
# ── Stage 2: Background correction (all channels) ─────────────────────────────
#   mode = 'tophat'   — white morphological top-hat (image − opening).
#     SE must be ≥ 1.5× the largest real structure.  Use for punctate markers.
#   mode = 'gaussian' — subtract large Gaussian blur of the inpainted image.
#     Cannot strip structures smaller than sigma.  Use for fibrous/vascular
#     markers where top-hat would destroy real signal.
#   se_radius_um  — SE radius for tophat mode (µm).
#   bg_sigma_um   — Gaussian sigma for gaussian mode (µm).
#     Rule: ≥ 3× the largest real tissue structure you want to preserve.

CHANNEL_CONFIG = {
    # ── DAPI ─────────────────────────────────────────────────────────────────
    # Has a giant macroscopic bubble.  Pass 1a catches it by size; Pass 1b mops
    # up any residual compact bright debris.  DoG pass not used for DAPI.
    'DAPI': dict(
        mode='tophat', se_radius_um=8.0, bg_sigma_um=None,
        artifact_max_area_um2=5000.0,
        artifact_thresh_factor=6.0, artifact_circ_factor=6.0, artifact_circ_thresh=0.75,
    ),

    # ── CD31 ──────────────────────────────────────────────────────────────────
    # Vascular marker with fine capillary network.  Intensity-based detection
    # cannot distinguish dye precipitates from real vessels — DoG is the only
    # viable scanner.
    'CD31': dict(
        mode='gaussian', se_radius_um=None, bg_sigma_um=80.0,
        dog_sigma_high_um=60.0, dog_min_area_um2=5000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── GAP43 ─────────────────────────────────────────────────────────────────
    # Fibrous axonal marker.  Real signal is elongated; DoG catches macro-scale
    # fold-edge and debris artifacts by area + shape rather than intensity.
    'GAP43': dict(
        mode='gaussian', se_radius_um=None, bg_sigma_um=60.0,
        dog_sigma_high_um=60.0, dog_min_area_um2=10_000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── NFP ───────────────────────────────────────────────────────────────────
    # Neurofilament; same approach as GAP43.
    'NFP': dict(
        mode='gaussian', se_radius_um=None, bg_sigma_um=60.0,
        dog_sigma_high_um=60.0, dog_min_area_um2=5000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── CD3 ───────────────────────────────────────────────────────────────────
    # T-cell marker; real cells are small and round so intensity gates produce
    # false positives.  DoG handles macro-artifacts; top-hat corrects background.
    'CD3': dict(
        mode='tophat', se_radius_um=8.0, bg_sigma_um=None,
        dog_sigma_high_um=60.0, dog_min_area_um2=10_000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── CD163 ─────────────────────────────────────────────────────────────────
    # Macrophage marker; same approach as CD3.
    'CD163': dict(
        mode='tophat', se_radius_um=8.0, bg_sigma_um=None,
        dog_sigma_high_um=60.0, dog_min_area_um2=10_000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── CK ────────────────────────────────────────────────────────────────────
    # Cytokeratin; large epithelial sheets with variable intensity.  Large
    # bg_sigma preserves sheet-scale signal.
    'CK': dict(
        mode='gaussian', se_radius_um=None, bg_sigma_um=150.0,
        dog_sigma_high_um=30.0, dog_min_area_um2=10_000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),

    # ── AF (autofluorescence) ─────────────────────────────────────────────────
    # Broad background channel.  Artifacts are typically fold-edge halos or
    # lipofuscin deposits caught well by DoG.
    'AF': dict(
        mode='gaussian', se_radius_um=None, bg_sigma_um=40.0,
        dog_sigma_high_um=60.0, dog_min_area_um2=10_000.0, dog_max_area_um2=35_000.0,
        dog_min_circ=0.20, dog_min_solid=0.40, dog_top_n=3,
    ),
}

# Ordered channel list — index must match the volume's channel axis
CHANNEL_NAMES_ORDERED = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

PIXEL_UM = args.pixel_um
DUST_PCT = args.dust_pct

# Minimum artifact blob area: anything smaller is a hot pixel caught by median
# filter anyway; only blobs larger than this are zeroed in artifact masking.
# Fixed in physical units so it doesn't scale with SE / background sigma.
ARTIFACT_MIN_AREA_UM2 = 50.0   # ~50 µm² ≈ 200 px at 0.5 µm/px — tweak if needed

def _um_to_px(um: float) -> int:
    return max(1, int(round(um / PIXEL_UM)))

def _build_se(radius_um: float):
    r_px = _um_to_px(radius_um)
    diam = 2 * r_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam)), r_px, diam

def _gaussian_sigma_px(sigma_um: float) -> float:
    return sigma_um / PIXEL_UM

ARTIFACT_MIN_AREA_PX = max(10, int(ARTIFACT_MIN_AREA_UM2 / (PIXEL_UM ** 2)))


logger.info('Per-channel denoising configuration:')
for _ch, _cfg in CHANNEL_CONFIG.items():
    _mode = _cfg['mode']
    _bg_str = (f"SE={_cfg['se_radius_um']:.0f}µm ({_um_to_px(_cfg['se_radius_um'])}px)"
               if _mode == 'tophat' else f"σ={_cfg['bg_sigma_um']:.0f}µm")
    if _cfg.get('artifact_max_area_um2'):
        _det_str = (f"1a: largeBlob≤{_cfg['artifact_max_area_um2']:.0f}µm²  "
                    f"1b: A={_cfg['artifact_thresh_factor']}×  "
                    f"B={_cfg['artifact_circ_factor']}×@circ≥{_cfg['artifact_circ_thresh']}")
    else:
        _det_str = (f"1c DoG: σ_hi={_cfg['dog_sigma_high_um']}µm  "
                    f"area=[{_cfg['dog_min_area_um2']:.0f}–{_cfg['dog_max_area_um2']:.0f}]µm²  "
                    f"top_n={_cfg['dog_top_n']}")
    logger.info(f"  {_ch:8s}  mode={_mode:8s}  bg={_bg_str:28s}  {_det_str}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-CHANNEL DENOISE
# ─────────────────────────────────────────────────────────────────────────────

# Stride for the large-blob size scan (must be >1).
# We subsample the image with this stride (zero-copy view) instead of
# allocating a downscaled copy — reading only (H/ds)*(W/ds) values.
_LARGE_BLOB_STRIDE = 32


def _large_blob_mask(med: np.ndarray, max_area_px: int) -> np.ndarray:
    """
    Pass 1a — Fast stride-sampled size scan for implausibly large bright blobs.

    Operates on a 1/32-downsampled view of the image to avoid allocating a full
    copy.  Blobs whose area in the downsampled image exceeds max_area_px/(ds²)
    are projected back to full resolution and dilated to absorb ragged edges.

    Only enabled for channels where artifact_max_area_um2 > 0 (currently DAPI).
    Returns an all-False mask immediately when max_area_px <= 0.
    """
    if max_area_px <= 0:
        return np.zeros(med.shape, dtype=bool)
    if med.max() == 0:
        return np.zeros(med.shape, dtype=bool)

    H, W = med.shape
    ds = _LARGE_BLOB_STRIDE

    # Zero-copy stride subsample
    small = med[::ds, ::ds]
    Hd, Wd = small.shape

    # Threshold at the DUST_PCT tissue ceiling — same anchor as Pass 1b so
    # the two passes are consistently calibrated.
    pos_ds = small[small > 0]
    if pos_ds.size == 0:
        return np.zeros(med.shape, dtype=bool)

    ceiling_ds = float(np.percentile(pos_ds, DUST_PCT))
    bright_ds  = (small >= ceiling_ds).astype(np.uint8)

    # CC on downsampled image
    _, label_ds, stats_ds, _ = cv2.connectedComponentsWithStats(
        bright_ds, connectivity=8
    )

    # Project oversized blobs back to full resolution with a small dilation pad
    max_area_ds = max_area_px / (ds * ds)
    mask_full   = np.zeros((H, W), dtype=bool)
    pad = 2

    dilate_radius = pad * ds
    kernel_size   = (dilate_radius * 2) + 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i in range(stats_ds.shape[0] - 1):
        lbl  = i + 1
        area = int(stats_ds[lbl, cv2.CC_STAT_AREA])
        
        if area <= max_area_ds:
            continue

        x0_ds = int(stats_ds[lbl, cv2.CC_STAT_LEFT])
        y0_ds = int(stats_ds[lbl, cv2.CC_STAT_TOP])
        bw_ds = int(stats_ds[lbl, cv2.CC_STAT_WIDTH])
        bh_ds = int(stats_ds[lbl, cv2.CC_STAT_HEIGHT])

        x0_pad_ds = max(0, x0_ds - pad)
        y0_pad_ds = max(0, y0_ds - pad)
        x1_pad_ds = min(Wd, x0_ds + bw_ds + pad)
        y1_pad_ds = min(Hd, y0_ds + bh_ds + pad)

        x0 = x0_pad_ds * ds
        y0 = y0_pad_ds * ds
        x1 = x1_pad_ds * ds
        y1 = y1_pad_ds * ds

        patch_ds = (label_ds[y0_pad_ds:y1_pad_ds, x0_pad_ds:x1_pad_ds] == lbl).astype(np.uint8)
        patch_full = cv2.resize(patch_ds, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        patch_full_dilated = cv2.dilate(patch_full, dilate_kernel).astype(bool)

        mask_full[y0:y1, x0:x1] |= patch_full_dilated

    return mask_full


def _detect_artifacts(med: np.ndarray, artifact_thresh_factor: float,
                       circularity_thresh: float = 0.82,
                       circularity_intensity_factor: float = 3.5) -> np.ndarray:
    """
    PASS 2 — Two-gate intensity + circularity scan for compact artifacts.

    Gate A (intensity-only): blob_max > ceiling × artifact_thresh_factor
        → flagged unconditionally regardless of shape.  Catches extreme hot-spots.

    Gate B (intensity + shape): blob_max > ceiling × artifact_circ_factor
        AND circularity >= circularity_thresh
        → catches modestly-bright but perfectly round debris (dye precipitates,
          dust) that would slip below Gate A.  circularity_intensity_factor must
          be < artifact_thresh_factor for this branch to do any extra work.

    Both thresholds are anchored to the DUST_PCT percentile tissue ceiling so
    real biological structures are never flagged.
    """
    pos = med[med > 0]
    if pos.size == 0:
        return np.zeros(med.shape, dtype=bool)

    ceiling = float(np.percentile(pos, DUST_PCT))
    if ceiling <= 0:
        return np.zeros(med.shape, dtype=bool)

    # Gate A: unconditional intensity tripwire (high bar)
    thresh_a = ceiling * artifact_thresh_factor
    # Gate B: lower intensity bar — only flags if shape is also round
    thresh_b = ceiling * circularity_intensity_factor

    # Seed the CC analysis from the lower of the two thresholds so Gate B
    # candidates are included in the connected-component image.
    seed_thresh = min(thresh_a, thresh_b)
    bright_bin  = (med >= seed_thresh).astype(np.uint8)

    _, label_img, stats, _ = cv2.connectedComponentsWithStats(
        bright_bin, connectivity=8
    )

    artifact_ids = []
    n_components = stats.shape[0] - 1

    for i in range(n_components):
        lbl  = i + 1
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < ARTIFACT_MIN_AREA_PX:
            continue

        blob_pixels = med[label_img == lbl]
        blob_max    = float(blob_pixels.max())

        # Gate A: bright enough to flag without any shape check
        if blob_max >= thresh_a:
            artifact_ids.append(lbl)
            continue

        # Gate B: only evaluate if this blob exceeds the lower circularity bar
        if blob_max < thresh_b:
            continue

        # Gate B — shape check: only round blobs are artifacts at this intensity level
        blob_mask_u8 = (label_img == lbl).astype(np.uint8)
        contours, _  = cv2.findContours(blob_mask_u8,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], closed=True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity >= circularity_thresh:
                    artifact_ids.append(lbl)

    if not artifact_ids:
        return np.zeros(med.shape, dtype=bool)

    return np.isin(label_img, artifact_ids)


def _dog_artifact_mask(med: np.ndarray, pixel_um: float, cfg: dict, ch_name: str) -> np.ndarray:
    """
    Pass 1c — DoG (Difference of Gaussians) bandpass macro-artifact scanner.
    """
    try:
        from skimage.measure import regionprops
    except ImportError:
        logger.warning('scikit-image not available — DoG artifact pass skipped.')
        return np.zeros(med.shape, dtype=bool)

    sigma_high_um = cfg['dog_sigma_high_um']
    min_area_um2  = cfg['dog_min_area_um2']
    max_area_um2  = cfg['dog_max_area_um2']
    min_circ      = cfg['dog_min_circ']
    min_solid     = cfg['dog_min_solid']
    top_n         = cfg['dog_top_n']

    SIGMA_LOW_UM  = 2.0    # fixed: removes hot-pixel / single-cell scale noise
    DOG_THRESHOLD = 1.0    # minimum DoG response to seed CC analysis

    sigma_low_px  = SIGMA_LOW_UM  / pixel_um
    sigma_high_px = sigma_high_um / pixel_um
    min_area_px   = min_area_um2  / (pixel_um ** 2)
    max_area_px   = max_area_um2  / (pixel_um ** 2)

    # DoG response map
    img_f      = med.astype(np.float32)
    blur_low   = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_low_px,  sigmaY=sigma_low_px)
    blur_high  = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_high_px, sigmaY=sigma_high_px)
    dog_map    = blur_low - blur_high
    dog_map[dog_map < 0] = 0.0  # bright-blob only

    # Connected components on thresholded DoG map
    binary   = (dog_map > DOG_THRESHOLD).astype(np.uint8)
    n_labels, label_img = cv2.connectedComponents(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros(med.shape, dtype=bool)

    regions = regionprops(label_img)

    candidates = []
    for reg in regions:
        area = reg.area
        if not (min_area_px <= area <= max_area_px):
            continue
        perimeter = reg.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circ:
            continue
        if reg.solidity < min_solid:
            continue

        blob_pixels  = dog_map[label_img == reg.label]
        max_response = float(blob_pixels.max())
        candidates.append({
            'label':    reg.label,
            'centroid': reg.centroid,
            'area_um2': area * (pixel_um ** 2),
            'score':    max_response,
        })

    if not candidates:
        return np.zeros(med.shape, dtype=bool)

    # Rank by (area × DoG peak) and keep top_n
    candidates.sort(key=lambda x: x['area_um2'] * x['score'], reverse=True)
    top = candidates[:top_n]

    final_mask = np.zeros(med.shape, dtype=bool)
    for t in top:
        final_mask[label_img == t['label']] = True
        logger.debug(f"    [{ch_name}] DoG artifact: center=({t['centroid'][0]:.0f},{t['centroid'][1]:.0f})  "
                     f"area={t['area_um2']:.0f}µm²  score={t['score']:.1f}")

    logger.info(f'[{ch_name}] Stage 1c (DoG): {len(top)} macro-artifact(s) detected '
                f'({int(final_mask.sum()):,} px)')
    return final_mask


def _inpaint_artifact(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fast matrix-based inpainting. 
    Replaces cv2.inpaint (Telea algorithm) which is computationally prohibitive 
    for high-resolution arrays with scattered masks.
    """
    if not mask.any():
        return img
    
    # 1. Determine a safe baseline to prevent artifacts from spiking the blur
    #    and zeros from sinking it.
    valid_pixels = img[img > 0]
    baseline_val = float(np.median(valid_pixels)) if valid_pixels.size > 0 else 0.0
    
    # 2. Create a base image where artifacts are neutralized
    clean_base = img.copy()
    clean_base[mask] = baseline_val
    
    # 3. Generate a fast local topography estimate
    #    31x31 is large enough to blend edges without massive CPU overhead
    local_topo = cv2.GaussianBlur(clean_base, (31, 31), 0, borderType=cv2.BORDER_REFLECT)
    
    # 4. Inject only the local topography into the masked regions
    inpainted = img.copy()
    inpainted[mask] = local_topo[mask]
    
    return inpainted

def _gaussian_background(img_clean: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Estimate slow-varying background via large Gaussian blur.
    Uses a kernel size that covers ±3σ, rounded to nearest odd integer.
    """
    ksize = int(sigma_px * 6)
    if ksize % 2 == 0:
        ksize += 1
    # cv2.GaussianBlur is faster than scipy for large kernels on float32
    return cv2.GaussianBlur(img_clean, (ksize, ksize), sigma_px,
                            borderType=cv2.BORDER_REFLECT)




def denoise_channel(raw: np.ndarray, cfg: dict, ch_name: str) -> dict:
    """
    Artifact detection + background correction for one 2-D float32 channel.
    """
    mode    = cfg['mode']
    is_dapi = 'artifact_max_area_um2' in cfg   # DAPI has 1a/1b keys; others do not

    raw_max = float(raw.max())
    if raw_max == 0:
        empty = np.zeros_like(raw)
        return dict(cleaned=empty, dust_mask=empty.astype(bool),
                    large_mask=empty.astype(bool), dog_mask=empty.astype(bool),
                    tophat=empty, bg=empty, removed=empty)

    # ── Hot-pixel removal ─────────────────────────────────────────────────
    scale   = 65535.0 / raw_max
    u16     = (raw * scale).clip(0, 65535).astype(np.uint16)
    med_u16 = cv2.medianBlur(u16, 3)
    med     = med_u16.astype(np.float32) / scale

    # ── Stage 1: Artifact detection ───────────────────────────────────────
    if is_dapi:
        # Pass 1a: large-blob size scan
        max_area_px = int(cfg['artifact_max_area_um2'] / (PIXEL_UM ** 2))
        large_mask  = _large_blob_mask(med, max_area_px)
        n_1a = int(large_mask.sum())

        # Pass 1b: intensity + circularity scan on the post-1a image
        med_no_large = med.copy()
        med_no_large[large_mask] = 0.0
        small_mask = _detect_artifacts(
            med_no_large,
            artifact_thresh_factor       = cfg['artifact_thresh_factor'],
            circularity_thresh           = cfg['artifact_circ_thresh'],
            circularity_intensity_factor = cfg['artifact_circ_factor'],
        )
        n_1b = int(small_mask.sum())

        artifact_mask = large_mask | small_mask
        dog_mask      = np.zeros(med.shape, dtype=bool)

        if n_1a > 0 or n_1b > 0:
            logger.info(f"[{ch_name}] Stage 1a (Large-blob): {n_1a:,} px removed | Stage 1b (Intensity/Circ): {n_1b:,} px removed")

    else:
        # Pass 1c: DoG bandpass scan — sole detector for non-DAPI channels
        large_mask    = np.zeros(med.shape, dtype=bool)
        dog_mask      = _dog_artifact_mask(med, PIXEL_UM, cfg, ch_name)
        artifact_mask = dog_mask

    # Dilate the combined mask by ~7.5 µm to absorb optical halos
    if artifact_mask.any():
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        artifact_mask = cv2.dilate(artifact_mask.astype(np.uint8), dilate_kernel).astype(bool)

    # Inpaint artifact regions using fast local topography
    med_inpainted = _inpaint_artifact(med, artifact_mask)
    if mode == 'gaussian':
        sigma_px = _gaussian_sigma_px(cfg['bg_sigma_um'])
        bg       = _gaussian_background(med_inpainted, sigma_px)
        corrected = np.clip(med_inpainted - bg, 0.0, None)
        tophat_display = corrected

    elif mode == 'tophat':
        se, r_px, _ = _build_se(cfg['se_radius_um'])
        inp_u16    = (med_inpainted * scale).clip(0, 65535).astype(np.uint16)
        th_u16     = cv2.morphologyEx(inp_u16, cv2.MORPH_TOPHAT, se, borderType=cv2.BORDER_REFLECT)
        corrected  = th_u16.astype(np.float32) / scale
        bg         = med_inpainted - corrected
        tophat_display = corrected

    else: 
        corrected  = med_inpainted
        bg         = np.zeros_like(raw)
        tophat_display = corrected

    # ── Zero confirmed artifact pixels in the corrected image ────────────
    cleaned = corrected.copy()
    cleaned[artifact_mask] = 0.0

    return dict(
        cleaned=cleaned,
        dust_mask=artifact_mask,
        large_mask=large_mask,
        dog_mask=dog_mask,
        tophat=tophat_display,
        bg=bg,
        removed=raw - cleaned
    )


def denoise_slice(raw_slice: np.ndarray, n_channels: int, max_workers: int) -> list:
    """
    Denoise all channels of a single CYX slice in true parallel using multi-processing.
    """
    results = [None] * n_channels
    workers = min(max_workers, n_channels)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for c in range(n_channels):
            ch_name = CHANNEL_NAMES_ORDERED[c] if c < len(CHANNEL_NAMES_ORDERED) else f'Ch{c:02d}'
            cfg     = CHANNEL_CONFIG.get(ch_name, dict(
                mode='gaussian', bg_sigma_um=40.0, se_radius_um=None,
                artifact_thresh_factor=8.0,
            ))
            # Added ch_name to the arguments submitted to the pool
            fut = pool.submit(denoise_channel, raw_slice[c].astype(np.float32), cfg, ch_name)
            futures[fut] = c
            
        for fut in as_completed(futures):
            c = futures[fut]
            results[c] = fut.result()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QC PLOT  — one panel per channel, saved per slice
# ─────────────────────────────────────────────────────────────────────────────

# Channel names in order — index matches volume channel axis.
# Adjust if your volume has different channels.
_CH_LABELS = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']


def _stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.9) -> np.ndarray:
    """Percentile stretch to [0, 1] for display."""
    flat = img[img > 0]
    if flat.size == 0 or flat.max() == flat.min():
        return np.zeros_like(img, dtype=np.float32)
    lo = float(np.percentile(flat, lo_pct))
    hi = float(np.percentile(flat, hi_pct))
    if hi == lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def save_qc_slice_plot(
    raw_slice: np.ndarray,
    results: list,
    z_idx: int,
    n_channels: int,
    out_dir: str,
) -> None:
    """
    Save one PNG per channel for a given Z-slice.
    Layout (5 columns): Raw | BG-corrected | Dust mask | Cleaned | Histogram
    """
    os.makedirs(out_dir, exist_ok=True)

    for c in range(n_channels):
        res          = results[c]
        raw          = raw_slice[c].astype(np.float32)
        tophat       = res['tophat']
        cleaned      = res['cleaned']
        dust         = res['dust_mask']
        removed      = res['removed']

        ch_label  = _CH_LABELS[c] if c < len(_CH_LABELS) else f'Ch{c:02d}'
        cfg       = CHANNEL_CONFIG.get(ch_label, {})
        mode      = cfg.get('mode', '?')
        n_dust_px  = int(dust.sum())
        large_mask = res.get('large_mask', np.zeros_like(dust))
        dog_mask   = res.get('dog_mask',   np.zeros_like(dust))
        n_large_px = int(large_mask.sum())
        n_dog_px   = int(dog_mask.sum())
        n_small_px = n_dust_px - n_large_px - n_dog_px

        if mode == 'tophat':
            mode_str = f"tophat  SE={cfg.get('se_radius_um', 0):.0f} µm"
        elif mode == 'gaussian':
            mode_str = f"gaussian  σ={cfg.get('bg_sigma_um', 0):.0f} µm"
        else:
            mode_str = 'median only'

        fig, axes = plt.subplots(1, 5, figsize=(32, 7))
        fig.suptitle(
            f'{TARGET_CORE}  |  Z={z_idx:03d}  |  Ch {c:02d}: {ch_label}  '
            f'|  mode: {mode_str}  '
            f'|  artifact px removed: {n_dust_px:,}  '
            f'(large-blob: {n_large_px:,}  DoG: {n_dog_px:,}  small: {n_small_px:,})',
            fontsize=12, fontweight='bold',
        )

        # ── Col 0: Raw ────────────────────────────────────────────────────
        axes[0].imshow(_stretch(raw), cmap='gray', interpolation='nearest')
        raw_max = float(raw.max())
        axes[0].set_title(f'Raw\n(max={raw_max:.0f})', fontsize=10)
        axes[0].axis('off')

        # ── Col 1: Background-corrected ───────────────────────────────────
        axes[1].imshow(_stretch(tophat), cmap='gray', interpolation='nearest')
        axes[1].set_title(f'BG-corrected\n({mode_str})', fontsize=10)
        axes[1].axis('off')

        # ── Col 2: Dust mask overlaid ─────────────────────────────────────
        rgb_dust = np.stack([_stretch(tophat)] * 3, axis=-1)
        if n_dust_px > 0:
            rgb_dust[dust, 0] = 1.0   
            rgb_dust[dust, 1] = 0.0
            rgb_dust[dust, 2] = 0.0
        axes[2].imshow(rgb_dust, interpolation='nearest')
        dust_pct_px = 100.0 * n_dust_px / raw.size
        axes[2].set_title(f'Dust mask\n({n_dust_px:,} px, {dust_pct_px:.2f}% of image)', fontsize=10)
        axes[2].axis('off')

        # ── Col 3: Cleaned ────────────────────────────────────────────────
        axes[3].imshow(_stretch(cleaned), cmap='gray', interpolation='nearest')
        signal_px = int((cleaned > 0).sum())
        axes[3].set_title(f'Cleaned\n({signal_px:,} signal px retained)', fontsize=10)
        axes[3].axis('off')

        # ── Col 4: Histogram ──────────────────────────────────────────────
        ax_h = axes[4]
        raw_pos     = raw[raw > 0].ravel()
        cleaned_pos = cleaned[cleaned > 0].ravel()

        bins = np.linspace(0, float(np.percentile(raw_pos, 99.9)) if raw_pos.size else 1, 120)

        if raw_pos.size:
            ax_h.hist(raw_pos,     bins=bins, alpha=0.55, color='tomato',
                      label=f'Raw  (n={raw_pos.size:,})',     density=True)
        if cleaned_pos.size:
            ax_h.hist(cleaned_pos, bins=bins, alpha=0.55, color='steelblue',
                      label=f'Cleaned  (n={cleaned_pos.size:,})', density=True)

        ax_h.set_yscale('log')
        ax_h.set_xlabel('Pixel intensity', fontsize=9)
        ax_h.set_ylabel('Density (log)', fontsize=9)
        ax_h.set_title('Intensity distribution\nRaw vs Cleaned', fontsize=10)
        ax_h.legend(fontsize=8)
        ax_h.tick_params(labelsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f'{TARGET_CORE}_Z{z_idx:03d}_Ch{c:02d}_{ch_label}.png'
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'  QC plots saved: Z={z_idx:03d}  ({n_channels} channels → {out_dir})')


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW MODE
# ─────────────────────────────────────────────────────────────────────────────

def save_preview(raw_slice: np.ndarray, n_channels: int, z_idx: int) -> None:
    """Save QC panels for one slice using the full pipeline (same as --plot_qc)."""
    results = denoise_slice(raw_slice, n_channels, args.workers)
    save_qc_slice_plot(raw_slice, results, z_idx, n_channels, PREVIEW_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(f'denoise_volume.py — core: {TARGET_CORE}')
    logger.info(f'Input  : {INPUT_VOL}')
    logger.info(f'Output : {OUTPUT_VOL}')

    if not os.path.exists(INPUT_VOL):
        logger.error(f'Input volume not found: {INPUT_VOL}')
        sys.exit(1)

    if os.path.exists(OUTPUT_VOL) and not args.overwrite:
        logger.info('Output already exists — skipping (use --overwrite to force).')
        sys.exit(0)

    # ── Load volume ────────────────────────────────────────────────────────
    logger.info('Loading registered volume ...')
    vol = tifffile.imread(INPUT_VOL)

    if vol.ndim == 3:
        vol = vol[np.newaxis, np.newaxis]   # (1, 1, H, W) — single slice, single ch
    elif vol.ndim == 4:
        # Ensure ZCYX order
        if vol.shape[0] != 1 and vol.shape[1] == 1:
            vol = np.moveaxis(vol, 0, 1)    # was CZYX → ZCYX

    n_slices, n_channels, H, W = vol.shape
    logger.info(f'Volume shape: Z={n_slices}  C={n_channels}  H={H}  W={W}')
    logger.info(f'Parallel workers per slice: {args.workers}')

    # ── Preview mode ───────────────────────────────────────────────────────
    if args.preview_slice is not None:
        z = args.preview_slice
        if z >= n_slices:
            logger.error(f'--preview_slice {z} out of range (volume has {n_slices} slices).')
            sys.exit(1)
        save_preview(vol[z].astype(np.float32), n_channels, z)
        logger.info('Preview complete — exiting.')
        sys.exit(0)

    # ── Determine a GLOBAL scale factor across all channels ───────────────
    # We need to know the per-channel maximum across the whole volume so we
    # can map to uint16 consistently.  Load raw channel maxima first.
    logger.info('Computing per-channel global maxima for consistent uint16 scaling ...')
    ch_global_max = np.zeros(n_channels, dtype=np.float64)
    for z in range(n_slices):
        for c in range(n_channels):
            ch_global_max[c] = max(ch_global_max[c], float(vol[z, c].max()))
    logger.info(f'  Per-channel raw maxima: {ch_global_max}')

    # ── Allocate output volume in uint16 ──────────────────────────────────
    out_vol = np.zeros((n_slices, n_channels, H, W), dtype=np.uint16)

    if args.plot_qc:
        os.makedirs(QC_DIR, exist_ok=True)
        logger.info(f'QC plots enabled — writing to {QC_DIR}')

    t_total = time.perf_counter()

    for z in range(n_slices):
        t0  = time.perf_counter()
        raw = vol[z].astype(np.float32)           # (C, H, W)

        results = denoise_slice(raw, n_channels, args.workers)  # list[dict] len C

        # Pack cleaned arrays back into the output volume using the
        # global per-channel max so intensities are comparable across slices.
        for c, res in enumerate(results):
            cleaned = res['cleaned']
            ch_max  = ch_global_max[c]
            if ch_max > 0:
                out_vol[z, c] = (cleaned * (65535.0 / ch_max)).clip(0, 65535).astype(np.uint16)

        elapsed = time.perf_counter() - t0

        # Dust summary per slice
        total_dust_px = sum(int(r['dust_mask'].sum()) for r in results)
        logger.info(
            f'  Z={z:03d}/{n_slices-1}  denoised in {elapsed:.1f}s  '
            f'| dust removed: {total_dust_px:,} px across {n_channels} channels'
        )

        # QC plots — one PNG per channel for this slice
        if args.plot_qc:
            try:
                save_qc_slice_plot(raw, results, z, n_channels, QC_DIR)
            except Exception as exc:
                import traceback
                logger.error(f'  QC plot FAILED for Z={z}: {exc}\n{traceback.format_exc()}')

    total = time.perf_counter() - t_total
    logger.info(f'All slices done in {total:.1f}s  ({total/n_slices:.1f}s per slice)')

    # ── Write output OME-TIFF ──────────────────────────────────────────────
    logger.info(f'Writing denoised volume → {OUTPUT_VOL}')
    tifffile.imwrite(
        OUTPUT_VOL,
        out_vol,
        imagej=False,
        photometric='minisblack',
        metadata={'axes': 'ZCYX'},
    )
    file_gb = os.path.getsize(OUTPUT_VOL) / 1e9
    logger.info(f'Written: {OUTPUT_VOL}  ({file_gb:.2f} GB)')
    logger.info('Done.')


if __name__ == '__main__':
    main()
