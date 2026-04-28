"""
denoise_volume.py
=================
Dust-aware white top-hat denoising for all channels of a registered OME-TIFF
volume.  Run this ONCE per core before phenotype_cells.py.

Pipeline per slice × channel (parallelised across channels via threads):
  1. Median filter (ksize=3)  — removes isolated hot pixels
  2. White top-hat            — isolates foreground structures smaller than the
                                SE (nucleus radius), suppressing slowly-varying
                                background illumination
  3. Dust-mask (connected-component)  — zeros large bright blobs in the top-hat
                                image (dust, fold artefacts)
  4. log1p                    — compresses the cleaned signal for downstream KDE

Output
------
  <OUTPUT_DIR>/<CORE>/<CORE>_denoised.ome.tif   — ZCYX uint16, same shape as
                                                    the input volume; pixel
                                                    values are the cleaned
                                                    top-hat in uint16 scale
                                                    (before log1p so that
                                                    phenotype_cells.py can
                                                    apply log1p itself and
                                                    keep its existing QC path)

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
# Two-stage pipeline per channel:
#
#   Stage 1 — ARTIFACT DETECTION (on the raw image, before any background math)
#       Finds physically implausible bright blobs: high intensity AND compact
#       shape AND above a per-channel "impossible" threshold. These are fibers,
#       dust, debris on the slide — not tissue.  Masked pixels are interpolated
#       from local neighbours so downstream steps see clean signal.
#
#   Stage 2 — BACKGROUND CORRECTION (preserves all real tissue signal)
#       Two modes, chosen per channel:
#
#       'gaussian'  — subtract a large Gaussian blur of the masked image.
#                     Models slow illumination falloff / out-of-focus haze.
#                     Safe for ANY channel because it cannot strip structures
#                     that are smaller than the blur sigma; real tissue
#                     intensity relationships are preserved.
#                     Use for: CD31, GAP43, NFP, CK — anything with structures
#                     spanning many µm that top-hat would destroy.
#
#       'tophat'    — white morphological top-hat (image − opening).
#                     Best when signal is truly punctate (single cells) and
#                     background is smoothly varying at the SE scale.
#                     Use for: DAPI, CD3, CD163 — nuclear-scale markers.
#                     SE radius should be ≥ 1.5× the largest real structure
#                     you expect so the opening models background correctly.
#
#       None        — skip background correction (median hot-pixel removal only).
#
# ARTIFACT_THRESH_FACTOR: a pixel whose value exceeds
#   artifact_thresh_factor × (99th-percentile of the masked tissue signal)
#   AND belongs to a compact connected component is flagged as an artifact.
#   Increase if real bright structures (vessel lumens, glands) get masked.
#   Decrease if bright artifacts are slipping through.
#
# BG_SIGMA_UM (gaussian mode): Gaussian sigma for background estimation.
#   Rule: set to ≥ 3× the largest real tissue structure you want to keep.
#   Larger = more conservative (safer for intensity preservation).
#
# SE_RADIUS_UM (tophat mode): morphological SE radius.
#   Rule: set to ≥ 1.5× the largest nucleus/cell you expect.

# ARTIFACT_THRESH_FACTOR: Criterion A — blob mean > factor × 99th-pct ceiling.
#   High value = only extreme outliers. Raise if real bright tissue gets masked.
#
# ARTIFACT_CIRC_FACTOR: Criterion B intensity multiplier — lower than A.
#   Blobs above this AND with circularity > ARTIFACT_CIRC_THRESH are flagged.
#   For channels where real signal is always elongated (CD31, GAP43, NFP),
#   set this low (1.5–2×) to catch modestly-bright round artifacts.
#   For channels where real cells are round (DAPI, CD3, CD163), set higher
#   or rely on Criterion A only.
#
# ARTIFACT_CIRC_THRESH: minimum circularity (0–1) for Criterion B.
#   0.55 comfortably separates round debris from elongated vessels/axons.
#   Raise toward 0.75 to be more conservative (fewer false positives).

CHANNEL_CONFIG = {
    'DAPI':  dict(mode='tophat',   bg_sigma_um=None,  se_radius_um=8.0,
                  artifact_thresh_factor=6.0, artifact_circ_factor=6.0,  artifact_circ_thresh=0.75,
                  artifact_max_area_um2=0.0),
    'CD31':  dict(mode='gaussian', bg_sigma_um=80.0,  se_radius_um=None,
                  # Reverted to safe local parameters
                  artifact_thresh_factor=3.5, artifact_circ_factor=3.5,  artifact_circ_thresh=0.60,
                  # ACTIVATED: Catch artifacts by sheer mass (1000 µm²)
                  artifact_max_area_um2=1000.0),

    'GAP43': dict(mode='gaussian', bg_sigma_um=60.0,  se_radius_um=None,
                  artifact_thresh_factor=6.0, artifact_circ_factor=1.5,  artifact_circ_thresh=0.55,
                  artifact_max_area_um2=0.0),
    'NFP':   dict(mode='gaussian', bg_sigma_um=60.0,  se_radius_um=None,
                  # Lowered intensity threshold to catch dim peaks
                  artifact_thresh_factor=2.0, artifact_circ_factor=2.0,  
                  # Relies on shape to protect squiggly nerve fibers
                  artifact_circ_thresh=0.65,
                  artifact_max_area_um2=0.0),
    'CD3':   dict(mode='tophat',   bg_sigma_um=None,  se_radius_um=8.0,
                  artifact_thresh_factor=6.0, artifact_circ_factor=6.0,  artifact_circ_thresh=0.75,
                  artifact_max_area_um2=0.0),
    'CD163': dict(mode='tophat',   bg_sigma_um=None,  se_radius_um=8.0,
                  artifact_thresh_factor=6.0, artifact_circ_factor=6.0,  artifact_circ_thresh=0.75,
                  artifact_max_area_um2=0.0),
    'CK':    dict(mode='gaussian', bg_sigma_um=150.0, se_radius_um=None,
                  artifact_thresh_factor=8.0, artifact_circ_factor=3.0,  artifact_circ_thresh=0.65,
                  artifact_max_area_um2=0.0),
    'AF':    dict(mode='gaussian', bg_sigma_um=40.0,  se_radius_um=None,
                  artifact_thresh_factor=8.0, artifact_circ_factor=3.0,  artifact_circ_thresh=0.65,
                  artifact_max_area_um2=0.0),
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
    _illum = _cfg.get('illum_sigma_um')
    _illum_str = f'illum_σ={_illum:.0f}µm' if _illum else 'illum=off'
    if _cfg['mode'] == 'tophat':
        _r_px = _um_to_px(_cfg['se_radius_um'])
        logger.info(f"  {_ch:8s}  mode=tophat     SE radius: {_cfg['se_radius_um']:.0f} µm "
                    f"({_r_px} px)  A_factor={_cfg['artifact_thresh_factor']}  "
                    f"B_factor={_cfg['artifact_circ_factor']}  B_circ={_cfg['artifact_circ_thresh']}  {_illum_str}")
    elif _cfg['mode'] == 'gaussian':
        logger.info(f"  {_ch:8s}  mode=gaussian   bg_sigma: {_cfg['bg_sigma_um']:.0f} µm  "
                    f"A_factor={_cfg['artifact_thresh_factor']}  "
                    f"B_factor={_cfg['artifact_circ_factor']}  B_circ={_cfg['artifact_circ_thresh']}  {_illum_str}")
    else:
        logger.info(f"  {_ch:8s}  mode=None       (median only)  {_illum_str}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-CHANNEL DENOISE
# ─────────────────────────────────────────────────────────────────────────────

# Stride for the large-blob size scan (must be >1).
# We subsample the image with this stride (zero-copy view) instead of
# allocating a downscaled copy — reading only (H/ds)*(W/ds) values.
_LARGE_BLOB_STRIDE = 32


def _large_blob_mask(med: np.ndarray, max_area_px: int) -> np.ndarray:
    """
    PASS 1 — Fast size-only scan for implausibly large bright structures.
    UPDATED: Uses the 99th percentile (DUST_PCT) instead of the 80th percentile 
    to prevent dense capillary beds from connecting into false-positive webs.
    """
    if max_area_px <= 0:
        return np.zeros(med.shape, dtype=bool)
    if med.max() == 0:
        return np.zeros(med.shape, dtype=bool)

    H, W = med.shape
    ds = _LARGE_BLOB_STRIDE

    # 1. Zero-copy stride subsample
    small = med[::ds, ::ds]
    Hd, Wd = small.shape

    # 2. Threshold using the extreme tissue ceiling (Top 1%)
    pos_ds = small[small > 0]
    if pos_ds.size == 0:
        return np.zeros(med.shape, dtype=bool)
    
    # REPLACED '80' with 'DUST_PCT' to isolate dye precipitates from real tissue
    ceiling_ds = float(np.percentile(pos_ds, DUST_PCT))
    bright_ds = (small >= ceiling_ds).astype(np.uint8)

    # 3. CC on tiny image
    _, label_ds, stats_ds, _ = cv2.connectedComponentsWithStats(
        bright_ds, connectivity=8
    )

    # 4. Project oversized blobs to full-res
    max_area_ds = max_area_px / (ds * ds)
    mask_full   = np.zeros((H, W), dtype=bool)
    pad = 2 

    # Pre-compute the morphological kernel
    dilate_radius = pad * ds
    kernel_size = (dilate_radius * 2) + 1
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
    PASS 2 — Intensity AND circularity scan for artifacts (hot spots, round debris).

    Both gates must pass (AND logic):
      Gate 1 — blob peak > artifact_thresh_factor × tissue ceiling (intensity)
      Gate 2 — circularity >= circularity_thresh (shape)

    This prevents real tissue structures that are bright but irregular (vessel
    clusters, glands) from being falsely flagged. circularity_intensity_factor
    is retained in the signature for config compatibility but is no longer used
    as an independent OR criterion — artifact_thresh_factor drives segmentation.
    """
    pos = med[med > 0]
    if pos.size == 0:
        return np.zeros(med.shape, dtype=bool)

    ceiling = float(np.percentile(pos, DUST_PCT))
    if ceiling <= 0:
        return np.zeros(med.shape, dtype=bool)

    # Segment at the intensity gate threshold — only pixels bright enough to
    # potentially be artifacts are candidates. This is intentionally the same
    # factor as artifact_thresh_factor so Gate 1 and segmentation are consistent.
    base_thresh = ceiling * artifact_thresh_factor

    bright_bin = (med >= base_thresh).astype(np.uint8)

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

        # ── Gate 1 (intensity): blob peak must be implausibly bright.
        # This is the primary discriminator. Real tissue (vessels, dense
        # staining) is filtered out here before shape is even considered.
        # For CD31: real vessels peak ~117, real artifacts peak ~564 → ~5×
        # gap; a factor of 3.5 is conservative and safe.
        if blob_max <= ceiling * artifact_thresh_factor:
            continue   # not bright enough to be an artifact — skip regardless of shape

        # ── Gate 2 (circularity): intensity passed; now confirm compact shape.
        # Dust/debris is round (circularity ~0.97). Real structures that happen
        # to be very bright (large vessels, glands) are elongated or irregular.
        # Both gates must pass — AND logic, not OR.
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




def denoise_channel(raw: np.ndarray, cfg: dict) -> dict:
    """
    Two-stage tissue-preserving denoise for one 2-D float32 channel.
    """
    mode                  = cfg['mode']
    artifact_thresh_factor = cfg['artifact_thresh_factor']

    raw_max = float(raw.max())
    if raw_max == 0:
        empty = np.zeros_like(raw)
        return dict(cleaned=empty, dust_mask=empty.astype(bool),
                    tophat=empty, bg=empty, removed=empty)

    # ── Hot-pixel removal ─────────────────────────────────────────────────
    scale   = 65535.0 / raw_max
    u16     = (raw * scale).clip(0, 65535).astype(np.uint16)
    med_u16 = cv2.medianBlur(u16, 3)
    med     = med_u16.astype(np.float32) / scale

    # ── Stage 1a: Large-blob scan ─────────────────────────────────────────
    max_area_um2 = cfg.get('artifact_max_area_um2', 0.0)
    max_area_px = int(max_area_um2 / (PIXEL_UM ** 2)) if max_area_um2 > 0 else 0
    large_mask = _large_blob_mask(med, max_area_px)
    
    # ── Stage 1b: Intensity + circularity scan ────────────────────────────
    med_no_large = med.copy()
    med_no_large[large_mask] = 0.0

    small_mask = _detect_artifacts(
        med_no_large,
        artifact_thresh_factor       = cfg['artifact_thresh_factor'],
        circularity_thresh           = cfg.get('artifact_circ_thresh', 0.82),
        circularity_intensity_factor = cfg.get('artifact_circ_factor', 3.5),
    )

    artifact_mask = large_mask | small_mask

    # ── NEW: Smart Optical Halo Dilation (Vessel Preserving) ─────────────
    # Expands the mask to catch the halo, but protects entangled vessels 
    # using a local high-frequency shield.
    if artifact_mask.any():
        # 1. Create the Wrecking Ball (Blind Dilation)
        # 31x31 diameter = 15px radius (~7.5 µm expansion)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        expanded_mask = cv2.dilate(artifact_mask.astype(np.uint8), dilate_kernel).astype(bool)

        # 2. Build the Shield (Isolate sharp vessels from smooth halos)
        # We calculate the local background using a medium Gaussian blur.
        local_bg = cv2.GaussianBlur(med, (21, 21), 0, borderType=cv2.BORDER_REFLECT)
        
        # A pixel is a vessel if it is >50% brighter than its immediate local blur.
        # Halos are smooth, so halo pixels will fail this test and remain unshielded.
        vessel_shield = med > (local_bg * 1.5) 

        # 3. Apply the Shield
        # The final mask is the expanded mask, MINUS the shielded vessels, 
        # PLUS the original core artifact (to guarantee the core stays dead).
        artifact_mask = (expanded_mask & ~vessel_shield) | artifact_mask
    # ─────────────────────────────────────────────────────────────────────

    # Inpaint artifact regions using fast local topography
    med_inpainted = _inpaint_artifact(med, artifact_mask)

    # ── Stage 2: Background correction ────────────────────────────────────
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

    # Replaced ThreadPoolExecutor with ProcessPoolExecutor to bypass the GIL
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for c in range(n_channels):
            ch_name = CHANNEL_NAMES_ORDERED[c] if c < len(CHANNEL_NAMES_ORDERED) else f'Ch{c:02d}'
            cfg     = CHANNEL_CONFIG.get(ch_name, dict(
                mode='gaussian', bg_sigma_um=40.0, se_radius_um=None,
                artifact_thresh_factor=8.0,
            ))
            fut = pool.submit(denoise_channel, raw_slice[c].astype(np.float32), cfg)
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
        n_large_px = int(large_mask.sum())
        n_small_px = n_dust_px - n_large_px

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
            f'(large-blob: {n_large_px:,}  small: {n_small_px:,})',
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