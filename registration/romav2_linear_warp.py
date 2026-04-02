"""
Feature registration — RoMaV2 native dense warp field, linear-normalised input.

Pipeline per slice pair:
  RoMaV2 dense warp → confidence filtering → upsample to full resolution → cv2.remap

Difference from romav2_warp_registration.py (log-normalised input):
  RoMaV2 is a deep learning model pretrained on natural RGB images with a standard
  linear 0–255 intensity distribution. Feeding log-normalised images changes the
  local contrast patterns the model's feature extraction layers learned to respond
  to. This script feeds linear percentile-stretched uint8 images to RoMaV2 instead.
  Log-normalised images are still used for NCC measurement and tissue masking, where
  the log stretch helps discriminate tissue from background.

Use output folder Filter_RoMaV2_Linear_Warp to distinguish from the log version.

Fallback:
  If the confidence-weighted warp produces a blank CK output, reverts to
  identity (raw moving slice) to protect the chain.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
import glob
import re
import cv2
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

# Torch setup — must happen before any torch import
os.environ['TORCH_HOME']           = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # force CPU; remove if GPU available

import torch
torch._dynamo.config.disable = True

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='RoMaV2 native warp registration.')
parser.add_argument('--core_name', type=str, required=True)
args = parser.parse_args()

TARGET_CORE = args.core_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_RoMaV2_Linear_Warp")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# ROMAV2 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# ROMAV2_H / W           — matching resolution (divisible by 14).
#                          Effective spatial resolution of the warp field.
#                          448 → ~13px per cell on a 6000px image.
#                          560 → ~11px per cell (needs more VRAM).
# ROMAV2_H_HR / W_HR     — high-res pass. Set None to disable (saves ~4GB VRAM).
# WARP_CONFIDENCE_THRESH — overlap confidence threshold [0,1].
#                          Pixels below this are treated as uncertain and
#                          their displacement is interpolated from neighbours.
#                          0.0 = use all pixels; 0.5 = high-confidence only.
# WARP_MAX_DISPLACEMENT_PX — hard cap on per-pixel displacement magnitude.
#                          Prevents catastrophic wrap-around on failed matches.
ROMAV2_DEVICE            = 'cpu'     # 'cuda' when GPU available
ROMAV2_H                 = 448
ROMAV2_W                 = 448
ROMAV2_H_HR              = None
ROMAV2_W_HR              = None
WARP_CONFIDENCE_THRESH   = 0.0       # start at 0 — raise if noisy results
WARP_MAX_DISPLACEMENT_PX = 200.0     # pixels at full resolution

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL / METADATA
# ─────────────────────────────────────────────────────────────────────────────
CK_CHANNEL_IDX       = 6
CHANNEL_NAMES        = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5
MIN_CK_NONZERO_FRAC  = 0.01

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_slice_filter(yaml_path, core_name):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get(core_name)
    if raw is None:
        return None
    allowed = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            allowed.add(int(part))
    return allowed


def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def conform_slice(arr, target_h, target_w):
    c, h, w = arr.shape
    out    = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0 = max(0, (h - target_h) // 2)
    dst_y0 = max(0, (target_h - h) // 2)
    copy_h = min(h - src_y0, target_h - dst_y0)
    src_x0 = max(0, (w - target_w) // 2)
    dst_x0 = max(0, (target_w - w) // 2)
    copy_w = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
        arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out


def prepare_ck(img_arr):
    """
    Returns (norm_lin, norm_log) — both uint8.

    norm_lin : linear percentile stretch (0.1–99.9%) → fed to RoMaV2.
               Preserves the natural intensity relationships the model was
               pretrained on (standard 0–255 linear RGB distribution).
    norm_log : log-stretch then percentile normalise → used for tissue
               masking and NCC measurement, where log boosts weak signal.
    """
    img_float  = img_arr.astype(np.float32)

    # Linear normalisation for RoMaV2
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Log normalisation for tissue mask / NCC
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return norm_lin, norm_log


def ck_to_rgb_pil(ck_log):
    from PIL import Image
    return Image.fromarray(np.stack([ck_log, ck_log, ck_log], axis=-1))


# ─────────────────────────────────────────────────────────────────────────────
# NCC HELPERS  (same logic as AKAZE script for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """
    Binary tissue mask from the log-normalised CK channel.
    Returns uint8 (H, W): 255 = tissue, 0 = background.
    Otsu threshold on non-zero pixels + 20px dilation to cover tissue edges.
    """
    img = ck_log.astype(np.uint8)
    nonzero = img[img > 0]
    if len(nonzero) == 0:
        return np.zeros_like(img)
    thresh, _ = cv2.threshold(
        nonzero.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = (img > thresh).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 20px radius
    mask = cv2.dilate(mask, kern)
    return mask


def _measure_ncc_masked(
    fixed_f32: np.ndarray,
    moving_f32: np.ndarray,
    mask_uint8: np.ndarray,
) -> float:
    """
    Measure NCC between two float32 images restricted to mask pixels.
    Uses a 0-iteration SimpleITK LBFGSB pass — evaluation only, no optimisation.
    Returns 0.0 on failure.  NCC is negative; more-negative = better alignment.
    """
    try:
        sitk_f = sitk.GetImageFromArray(fixed_f32)
        sitk_m = sitk.GetImageFromArray(moving_f32)
        reg    = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.10)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(numberOfIterations=0)
        reg.SetInitialTransform(sitk.TranslationTransform(2), inPlace=False)
        if mask_uint8 is not None and mask_uint8.max() > 0:
            reg.SetMetricFixedMask(sitk.GetImageFromArray(mask_uint8))
        reg.Execute(sitk_f, sitk_m)
        return reg.GetMetricValue()
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ROMAV2 MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        device = ROMAV2_DEVICE if (ROMAV2_DEVICE == 'cuda' and torch.cuda.is_available()) else 'cpu'
        if device == 'cpu':
            logger.warning("CUDA not available — RoMaV2 on CPU. Expect slow matching.")
        logger.info(f"Loading RoMaV2 on {device}...")
        _romav2_model = RoMaV2().to(device)
        _romav2_model.eval()
        if device == 'cpu':
            _romav2_model = torch._dynamo.disable(_romav2_model)
        _romav2_model.H_lr = ROMAV2_H
        _romav2_model.W_lr = ROMAV2_W
        _romav2_model.H_hr = ROMAV2_H_HR
        _romav2_model.W_hr = ROMAV2_W_HR
        logger.info("RoMaV2 model loaded.")
    return _romav2_model


# ─────────────────────────────────────────────────────────────────────────────
# CORE: RoMaV2 dense warp → full-resolution remap maps
# ─────────────────────────────────────────────────────────────────────────────

def romav2_dense_warp(fixed_lin, moving_lin, slice_id, orig_h, orig_w,
                      tissue_mask_full=None):
    """
    Run RoMaV2 and return (map_x, map_y, n_confident, coverage_pct, mean_confidence).

    Strategy
    --------
    1.  Run model.match() → warp_AB (B-side coords in [-1,1]) + overlap_AB (confidence).
    2.  Convert B-side coords to full-resolution pixel space.
    3.  Apply confidence threshold — low-confidence cells revert to identity.
    4.  Apply displacement cap — protects against catastrophic wrap-around.
    5.  Apply tissue mask — background cells outside the tissue are forced to
        identity BEFORE upsampling.  This prevents RoMaV2's featureless-canvas
        vectors from bleeding into tissue edges through bilinear upsampling and
        creating hard seam artefacts at the tissue boundary.
    6.  Upsample to full resolution.

    Parameters
    ----------
    fixed_lin, moving_lin : uint8 linear-normalised CK channel (fed to RoMaV2)
    slice_id              : logging label
    orig_h, orig_w        : full-resolution image dimensions
    tissue_mask_full      : uint8 (orig_h, orig_w) tissue mask — 255=tissue, 0=background.
                            Downsampled to (H_lr, W_lr) internally.  Pass None to skip.

    Returns
    -------
    map_x, map_y    : float32 (orig_h, orig_w) — source coords for cv2.remap
    n_confident     : int   — number of high-confidence cells in the low-res grid
    coverage_pct    : float — n_confident / total_cells * 100
    mean_confidence : float — mean overlap_AB value across the whole grid
    """
    try:
        model = get_romav2_model()
        img_A = ck_to_rgb_pil(fixed_lin)
        img_B = ck_to_rgb_pil(moving_lin)

        with torch.no_grad():
            preds = model.match(img_A, img_B)

        # warp_AB: (1, H_lr, W_lr, 2) — for each location in A, the
        # corresponding location in B, in [-1,1] normalised coordinates.
        # Channels 0=x, 1=y (B-side coords).
        warp_AB    = preds['warp_AB'].squeeze(0).cpu().numpy()           # (H_lr, W_lr, 2)
        overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy()          # (H_lr, W_lr) — squeeze ALL size-1 dims

        H_lr, W_lr = warp_AB.shape[:2]
        logger.info(f"[{slice_id}] warp_AB: {warp_AB.shape}, overlap_AB: {overlap_AB.shape}")

        # Convert B-side coords from [-1,1] to full-resolution pixel space
        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)  # (H_lr, W_lr)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

        # Confidence mask — ensure exactly 2D (H_lr, W_lr)
        confident_2d = overlap_AB.reshape(H_lr, W_lr) >= WARP_CONFIDENCE_THRESH
        n_confident  = int(confident_2d.sum())

        # Build identity coordinates at lr resolution for fallback
        grid_x_lr = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
        grid_y_lr = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
        identity_x, identity_y = np.meshgrid(grid_x_lr, grid_y_lr)

        # Apply confidence mask: use model prediction where confident, else identity
        map_x_lr = np.where(confident_2d, b_coords_x, identity_x).astype(np.float32)
        map_y_lr = np.where(confident_2d, b_coords_y, identity_y).astype(np.float32)

        # Cap displacement magnitude before upsampling
        disp_x = map_x_lr - identity_x
        disp_y = map_y_lr - identity_y
        mag    = np.sqrt(disp_x**2 + disp_y**2)
        excess = mag > WARP_MAX_DISPLACEMENT_PX
        if np.any(excess):
            scale    = np.where(excess, WARP_MAX_DISPLACEMENT_PX / (mag + 1e-8), 1.0)
            disp_x  *= scale
            disp_y  *= scale
            map_x_lr = (identity_x + disp_x).astype(np.float32)
            map_y_lr = (identity_y + disp_y).astype(np.float32)
            logger.info(
                f"[{slice_id}] Clipped {int(excess.sum())} warp vectors "
                f"> {WARP_MAX_DISPLACEMENT_PX}px."
            )

        # Zero out displacement in background regions BEFORE upsampling.
        # RoMaV2 produces high-confidence but meaningless vectors in featureless
        # canvas areas outside the tissue.  If left in, bilinear upsampling
        # propagates these into the tissue boundary creating hard seam artefacts.
        # Downsampling the mask to lr grid resolution and forcing background
        # cells to identity eliminates the problem at its source.
        if tissue_mask_full is not None:
            mask_lr = cv2.resize(
                tissue_mask_full, (W_lr, H_lr), interpolation=cv2.INTER_NEAREST
            ).astype(bool)   # True = tissue
            background = ~mask_lr
            if np.any(background):
                map_x_lr[background] = identity_x[background]
                map_y_lr[background] = identity_y[background]
                logger.info(
                    f"[{slice_id}] Tissue mask: zeroed {int(background.sum())} "
                    f"background warp cells ({background.sum()/(H_lr*W_lr)*100:.1f}%)."
                )

        # Upsample to full resolution
        map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # After upsampling, the coordinates were computed at lr scale — rescale
        # the coordinate values to full resolution (resize only stretches the grid,
        # not the coordinate values themselves).
        # Actually cv2.resize on a coordinate map IS correct without rescaling
        # because each value IS already in full-res pixel space (we converted
        # from [-1,1] using orig_w/orig_h before upsampling).
        # So map_x/map_y are already in full-res pixel coordinates. ✓

        coverage_pct    = n_confident / (H_lr * W_lr) * 100
        mean_confidence = float(overlap_AB.mean())
        logger.info(
            f"[{slice_id}] RoMaV2 warp: {H_lr}×{W_lr} grid, "
            f"{coverage_pct:.1f}% confident (thresh={WARP_CONFIDENCE_THRESH}), "
            f"mean confidence={mean_confidence:.3f}"
        )
        return map_x, map_y, n_confident, coverage_pct, mean_confidence

    except Exception as exc:
        logger.error(f"[{slice_id}] RoMaV2 dense warp failed: {exc}")
        return None, None, 0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ONE SLICE PAIR
# ─────────────────────────────────────────────────────────────────────────────

def register_slice(fixed_np, moving_np, slice_id=None):
    """
    Warp moving to fixed using RoMaV2 native dense warp field.
    Returns (aligned_np, elapsed, stats, success).
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # Tissue mask and NCC use log-normalised images (better tissue/background separation)
    tissue_mask = build_tissue_mask(fixed_log)

    # NCC before warp — log space, masked
    ncc_before = _measure_ncc_masked(
        fixed_log.astype(np.float32),
        moving_log.astype(np.float32),
        tissue_mask,
    )

    # RoMaV2 receives linear-normalised images — matches its pretraining distribution
    # tissue_mask passed so background vectors are zeroed before upsampling
    map_x, map_y, n_confident, coverage_pct, mean_confidence = romav2_dense_warp(
        fixed_lin, moving_lin, sid, h, w,
        tissue_mask_full=tissue_mask,
    )

    success = map_x is not None
    ncc_after = 0.0

    if success:
        # Apply warp field to all channels
        aligned_channels = []
        for ch in range(fixed_np.shape[0]):
            warped = cv2.remap(
                moving_np[ch].astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            aligned_channels.append(warped)
        aligned_np = np.stack(aligned_channels, axis=0).astype(np.uint16)

        # Sanity: blank output check
        ck_out = aligned_np[CK_CHANNEL_IDX]
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(
                f"[{sid}] CK output nearly blank — warp diverged. "
                "Reverting to raw."
            )
            success    = False
            aligned_np = moving_np.copy()
        else:
            # NCC after warp — warp the log image for NCC measurement
            # (log space consistent with ncc_before)
            warped_ck_f32  = aligned_np[CK_CHANNEL_IDX].astype(np.float32)
            _, warped_log  = prepare_ck(warped_ck_f32)
            ncc_after = _measure_ncc_masked(
                fixed_log.astype(np.float32),
                warped_log.astype(np.float32),
                tissue_mask,
            )
    else:
        aligned_np = moving_np.copy()

    # Relative NCC improvement (same formula as AKAZE script)
    if abs(ncc_before) > 1e-9:
        ncc_improvement_pct = (ncc_before - ncc_after) / abs(ncc_before) * 100.0
    else:
        ncc_improvement_pct = 0.0

    logger.info(
        f"[{sid}] NCC before={ncc_before:.4f} after={ncc_after:.4f} "
        f"improvement={ncc_improvement_pct:+.1f}% | "
        f"Confident cells: {n_confident} ({coverage_pct:.1f}%) | "
        f"Mean confidence: {mean_confidence:.3f}"
    )

    stats = dict(
        n_confident         = n_confident,
        coverage_pct        = round(coverage_pct, 2),
        mean_confidence     = round(mean_confidence, 4),
        ncc_before          = round(float(ncc_before), 6),
        ncc_after           = round(float(ncc_after),  6),
        ncc_improvement_pct = round(float(ncc_improvement_pct), 2),
        success             = success,
    )

    # Save warp plot — overlays use log images for visual contrast;
    # title notes that RoMaV2 received linear-normalised input
    try:
        save_warp_plot(fixed_log, moving_log, map_x, map_y, sid, OUTPUT_FOLDER,
                       success, ncc_before=ncc_before, ncc_after=ncc_after,
                       fixed_lin=fixed_lin, moving_lin=moving_lin)
    except Exception as exc:
        logger.warning(f"[{sid}] Warp plot failed: {exc}")

    return aligned_np, time.time() - start, stats, success


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def save_warp_plot(fixed_log, moving_log, map_x, map_y,
                   slice_id, output_folder, success,
                   ncc_before=None, ncc_after=None,
                   fixed_lin=None, moving_lin=None):
    """
    3-panel interim plot:
      Panel 1 (R/G): fixed vs moving — before warp
                     Uses linear images if supplied, log otherwise.
      Panel 2 (R/G): fixed vs warped moving — after RoMaV2
                     Uses linear images if supplied (shows what RoMaV2 saw),
                     log otherwise.
      Panel 3:       displacement magnitude heatmap
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    def norm(x):
        p = np.percentile(x, 99.5)
        return np.clip(x.astype(np.float32) / (p if p > 0 else 1), 0, 1)

    # Use linear if available (shows what RoMaV2 actually received),
    # fall back to log for scripts that don't supply linear images
    f_display = norm(fixed_lin  if fixed_lin  is not None else fixed_log)
    m_display = norm(moving_lin if moving_lin is not None else moving_log)
    input_label = "linear input" if fixed_lin is not None else "log input"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — before warp
    ncc_before_str = f"NCC={ncc_before:.4f}" if ncc_before is not None else ""
    axes[0].imshow(np.dstack((f_display, m_display, np.zeros_like(f_display))))
    axes[0].set_title(
        f"Fixed (R) vs Moving (G) — before warp [{input_label}]\n{ncc_before_str}",
        fontsize=11,
    )
    axes[0].axis('off')

    if map_x is not None:
        # Remap the same image that was fed to RoMaV2 for the warped panel
        src_for_remap = (moving_lin if moving_lin is not None else moving_log).astype(np.float32)
        warped_display = cv2.remap(
            src_for_remap, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        w_norm = norm(warped_display)

        # Panel 2 — after warp
        ncc_after_str = f"NCC={ncc_after:.4f}" if ncc_after is not None else ""
        axes[1].imshow(np.dstack((f_display, w_norm, np.zeros_like(f_display))))
        axes[1].set_title(
            f"Fixed (R) vs Warped (G) — after RoMaV2 [{input_label}]\n{ncc_after_str}",
            fontsize=11,
        )

        # Panel 3 — displacement magnitude
        h, w_img = map_x.shape
        id_x = np.arange(w_img, dtype=np.float32)[None, :]
        id_y = np.arange(h,     dtype=np.float32)[:, None]
        disp_mag     = np.sqrt((map_x - id_x)**2 + (map_y - id_y)**2)
        disp_display = cv2.resize(disp_mag, (512, 512), interpolation=cv2.INTER_AREA)
        im = axes[2].imshow(disp_display, cmap='hot', vmin=0,
                            vmax=np.percentile(disp_mag, 99))
        axes[2].set_title("Displacement magnitude (px)", fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)
    else:
        axes[1].imshow(np.dstack((f_display, m_display, np.zeros_like(f_display))))
        axes[1].set_title("Warp FAILED — identity", fontsize=11)
        axes[2].axis('off')

    axes[1].axis('off')

    # Suptitle
    status = "SUCCESS" if success else "FAILED"
    if ncc_before is not None and ncc_after is not None and ncc_after != 0.0:
        if abs(ncc_before) > 1e-9:
            delta_pct = (ncc_before - ncc_after) / abs(ncc_before) * 100.0
        else:
            delta_pct = 0.0
        title = (
            f"{slice_id}  "
            f"NCC before={ncc_before:.4f} → after={ncc_after:.4f}  "
            f"Δ={delta_pct:+.1f}%  [{status}]"
        )
    else:
        title = f"{slice_id}  [{status}]"

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{slice_id}_warp.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_qc_montage(vol, output_folder, slice_ids=None,
                        channel_idx=6, channel_name="CK",
                        title_suffix="RoMaV2_Warp"):
    n_slices = vol.shape[0]
    if n_slices < 2:
        return
    logger.info(f"Generating QC montage [{title_suffix}]...")
    all_pairs = [(i, i+1) for i in range(n_slices-1)]
    n_cols    = 5
    n_rows    = (len(all_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1:               axes = axes.reshape(1, -1)
    elif n_cols == 1:               axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()
    for idx, (z1, z2) in enumerate(all_pairs):
        s1 = vol[z1, channel_idx].astype(np.float32)
        s2 = vol[z2, channel_idx].astype(np.float32)
        def norm(x):
            p = np.percentile(x, 99.5)
            return np.clip(x / (p if p > 0 else 1), 0, 1)
        overlay = np.dstack((norm(s1), norm(s2), np.zeros_like(s1)))
        axes_flat[idx].imshow(overlay)
        lbl1 = slice_ids[z1] if slice_ids else z1
        lbl2 = slice_ids[z2] if slice_ids else z2
        axes_flat[idx].set_title(f"ID{lbl1} to ID{lbl2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')
    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')
    core = os.path.basename(output_folder)
    fig.suptitle(f"Registration QC {title_suffix}: {core}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{core}_QC_Montage_{channel_name}_{title_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"RoMaV2 Linear-Input Warp Registration — {TARGET_CORE}")
    logger.info(
        f"Resolution: {ROMAV2_H}×{ROMAV2_W} | "
        f"Input to RoMaV2: LINEAR normalised | "
        f"NCC/mask: LOG normalised | "
        f"Confidence thresh: {WARP_CONFIDENCE_THRESH} | "
        f"Max displacement: {WARP_MAX_DISPLACEMENT_PX}px"
    )

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    allowed_positions = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)
    if allowed_positions is not None:
        original_count = len(file_list)
        file_list = [f for i, f in enumerate(file_list) if i in allowed_positions]
        n_slices  = len(file_list)
        excluded  = original_count - n_slices
        logger.info(
            f"Slice filter active: keeping {n_slices}/{original_count} slices "
            f"(positions {sorted(allowed_positions)}), {excluded} excluded."
        )
        if n_slices == 0:
            logger.error("Slice filter excluded all slices.")
            sys.exit(1)
    else:
        logger.info(f"No slice filter — using all {n_slices} slices.")

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in = tifffile.imread(file_list[0])
        tifffile.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Linear_Warp_Aligned.ome.tif"),
            vol_in[np.newaxis], photometric='minisblack',
            metadata={
                'axes': 'ZCYX', 'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        sys.exit(0)

    get_romav2_model()  # preload

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)
    c, target_h, target_w = _center_arr.shape
    logger.info(f"Shape: C={c}, H={target_h}, W={target_w}")

    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
        arr = tifffile.imread(file_list[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            logger.warning(f"Shape mismatch in {os.path.basename(file_list[idx])} — conforming.")
            arr = conform_slice(arr, target_h, target_w)
        return arr

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    del center_raw
    logger.info(f"Anchor: slice index {center_idx} (ID {slice_ids[center_idx]})")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")
        for i in indices:
            real_id   = slice_ids[i]
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = load_slice(i)
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            try:
                aligned_np, runtime, stats, success = register_slice(
                    fixed_np, moving_np, slice_id=sid
                )
            except Exception as exc:
                logger.error(f"[{sid}] register_slice crashed: {exc} — raw fallback.")
                aligned_np = moving_np
                runtime, stats, success = 0.0, {'n_confident': 0, 'success': False}, False

            aligned_vol[i] = aligned_np
            del moving_np

            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK_RAW"
            if not success:
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str}")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | "
                f"Confident cells: {stats['n_confident']} ({stats['coverage_pct']:.1f}%) | "
                f"Mean conf: {stats['mean_confidence']:.3f} | "
                f"NCC before: {stats['ncc_before']:.4f} | "
                f"NCC after: {stats['ncc_after']:.4f} | "
                f"NCC improvement: {stats['ncc_improvement_pct']:+.1f}% | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )
            registration_stats.append({
                "Direction":            direction,
                "Slice_Z":              i,
                "Slice_ID":             real_id,
                "N_Confident":          stats['n_confident'],
                "Coverage_Pct":         stats['coverage_pct'],
                "Mean_Confidence":      stats['mean_confidence'],
                "NCC_Before":           stats['ncc_before'],
                "NCC_After":            stats['ncc_after'],
                "NCC_Improvement_Pct":  stats['ncc_improvement_pct'],
                "Success":              success,
                "Status":               status_str,
                "Runtime_s":            round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    df.to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_RoMaV2_Linear_Warp.csv"), index=False
    )
    n_ok = int((df["Status"] == "SUCCESS").sum())
    n_fb = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(f"Complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fb}")

    # Write volume first
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Linear_Warp_Aligned.ome.tif")
    logger.info(f"Writing volume to {out_tiff}")
    try:
        tifffile.imwrite(
            out_tiff, aligned_vol,
            photometric='minisblack',
            metadata={
                'axes': 'ZCYX', 'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        logger.info("Volume written.")
    except Exception as exc:
        logger.error(f"Volume write failed: {exc}")

    # Montage
    try:
        generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="RoMaV2_Linear_Warp")
    except Exception as exc:
        logger.error(f"Montage failed: {exc}")
    del aligned_vol

    logger.info("Done.")


if __name__ == "__main__":
    main()