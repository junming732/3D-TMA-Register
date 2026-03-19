"""
Experimental registration pipeline — RoMaV2 dense matching as L0 affine replacement.

Pipeline per slice pair:
  L0  RoMaV2 dense matches → RANSAC affine (replaces AKAZE)
  L1  SimpleITK NCC B-spline FFD → elastic residual correction (same as production)

Fallback chain (identical to production):
  If L1 fails or does not improve, reverts to L0 affine result.
  If L0 fails (too few matches), raw moving slice is written.

Why RoMaV2 might help:
  - Dense DINOv3-based matcher — does not rely on salient keypoints, so it can
    match tissue regions that are too uniform or too damaged for AKAZE.
  - Produces pixel-dense correspondences with a confidence score (overlap), so
    we can filter to high-certainty matches before RANSAC.
  - Particularly useful for slices with weak CK signal or large tissue deformation
    where AKAZE struggles to find enough inliers.

Differences vs production (akaze_bspline_adaptive.py):
  - L0 uses RoMaV2 instead of AKAZE + ANMS.
  - Images are converted to RGB PIL for RoMaV2 input (it expects natural images;
    we use the log-normalised CK channel replicated across R/G/B).
  - The RANSAC step uses cv2.estimateAffine2D on RoMaV2 keypoints, same as production.
  - All L1 (B-spline), tissue mask, per-tile acceptance, and QC plot code is
    shared 1:1 with the production pipeline.

Install (once, in your environment):
  pip install romav2[fused-local-corr]
  # requires torch ≥ 2.0 and a CUDA-capable GPU for practical speed

Usage:
  python akaze_romav2_experiment.py --core_name Z008
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
matplotlib.use('Agg')   # must be before pyplot import — headless server, no display
import matplotlib.pyplot as plt
import yaml
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# Redirect torch hub cache to the project workspace so model weights are
# stored with the data rather than in the user's home directory.
# Must be set before torch is imported anywhere.
os.environ['TORCH_HOME'] = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# When running on CPU, hide the GPU entirely so no tensor accidentally lands on CUDA.
# Remove this line if you switch back to ROMAV2_DEVICE = 'cuda'.
if True:  # set to False when switching to cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description='Experimental RoMaV2 + B-spline registration.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_RoMaV2_BSpline_Experiment")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# --- ROMAV2 CONFIGURATION ---
#
# ROMAV2_N_MATCHES    — number of matches sampled from the dense warp.
#                       More matches → more robust RANSAC, but slower.
#                       5000 is a good starting point.
# ROMAV2_MIN_OVERLAP  — minimum overlap confidence score [0,1] for a match
#                       to be included.  0.0 = keep all; 0.5 = high confidence
#                       only.  Start at 0.0 and raise if RANSAC gets noisy.
# ROMAV2_DEVICE       — 'cuda' for GPU (strongly recommended), 'cpu' for fallback.
# ROMAV2_H / W        — resolution passed to RoMaV2 for matching.  The model
#                       internally resizes to this.  Larger → finer matches but
#                       more VRAM.  560×560 is the training resolution but needs
#                       ~3-4 GB free VRAM.  Use 448 or 392 if GPU is constrained.
#                       Note: must be divisible by 14 (DINOv3 patch size).
ROMAV2_N_MATCHES   = 5000
ROMAV2_MIN_OVERLAP = 0.0
ROMAV2_DEVICE      = 'cpu'
ROMAV2_H           = 448
ROMAV2_W           = 448
ROMAV2_H_HR        = None
ROMAV2_W_HR        = None

# --- RANSAC (applied to RoMaV2 matches) ---
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS  = 5000
RANSAC_THRESH     = 8.0    # pixels at full resolution

# --- TRANSFORM CONSTRAINTS (same as production) ---
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0
MIN_INLIERS         = 6

# --- L1: B-SPLINE (identical to production) ---
BSPLINE_GRID_NODES       = 8
BSPLINE_ITERATIONS       = 100
BSPLINE_SHRINK_FACTORS   = [4, 2, 1]
BSPLINE_SMOOTHING_SIGMAS = [4.0, 2.0, 0.0]
BSPLINE_USE_GRID_TILES   = True
BSPLINE_GRID_ROWS        = 2
BSPLINE_GRID_COLS        = 2
BSPLINE_TILE_OVERLAP     = 0.25
BSPLINE_USE_TISSUE_MASK  = True
BSPLINE_MASK_OTSU        = True
BSPLINE_MASK_DILATE_PX   = 20
BSPLINE_MIN_TISSUE_FRAC  = 0.05
BSPLINE_PER_TILE_ACCEPT  = True
BSPLINE_NCC_MIN_IMPROVEMENT = 0.01

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- LAZY MODEL LOAD ---
# RoMaV2 is heavy — load once and reuse across all slice pairs.
_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        try:
            from romav2 import RoMaV2
            device = ROMAV2_DEVICE if (ROMAV2_DEVICE == 'cuda' and torch.cuda.is_available()) else 'cpu'
            if device == 'cpu':
                logger.warning("CUDA not available — RoMaV2 running on CPU. Expect slow matching.")
            logger.info(f"Loading RoMaV2 model on {device}...")
            _romav2_model = RoMaV2()
            _romav2_model = _romav2_model.to(device)
            _romav2_model.eval()
            # Disable torch.compile on CPU — compilation overhead with no benefit,
            # and dynamo can trigger errors on CPU inference paths.
            if device == 'cpu':
                torch._dynamo.config.disable = True
                # Also disable compile at the module level in case RoMaV2
                # internally compiled itself during __init__ on a CUDA-visible system
                _romav2_model = torch._dynamo.disable(_romav2_model)
            # Set matching resolution for the low-res pass
            _romav2_model.H_lr = ROMAV2_H
            _romav2_model.W_lr = ROMAV2_W
            _romav2_model.H_hr = ROMAV2_H_HR
            _romav2_model.W_hr = ROMAV2_W_HR
            logger.info("RoMaV2 model loaded.")
        except ImportError:
            logger.error(
                "romav2 is not installed. Run: pip install romav2[fused-local-corr]"
            )
            raise
    return _romav2_model


# --- UTILITIES (identical to production) ---
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
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

def prepare_ck(img_arr: np.ndarray):
    img_float  = img_arr.astype(np.float32)
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    p_lo_lin, p_hi_lin = np.percentile(img_arr[::4, ::4], (1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_arr, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_lin, norm_log

def constrain_affine(M: np.ndarray) -> np.ndarray:
    if M is None: return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S        = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out

def transform_is_sane(M: np.ndarray) -> bool:
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG

def load_slice_filter(yaml_path: str, core_name: str):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r") as fh:
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


# --- L0: ROMAV2 AFFINE ---
def ck_to_rgb_pil(ck_log: np.ndarray) -> Image.Image:
    """
    Convert a log-normalised uint8 CK channel to an RGB PIL image.
    RoMaV2 expects RGB inputs — we replicate the single channel across R/G/B.
    This is a reasonable proxy: the model's DINOv3 backbone will still produce
    meaningful patch features from the tissue texture, even without true colour.
    """
    rgb = np.stack([ck_log, ck_log, ck_log], axis=-1)
    return Image.fromarray(rgb)


def romav2_affine(
    fixed_log: np.ndarray,
    moving_log: np.ndarray,
    slice_id: str,
    orig_h: int,
    orig_w: int,
) -> tuple:
    """
    Use RoMaV2 dense matching to produce an affine transform.

    RoMaV2 matches at (ROMAV2_H × ROMAV2_W); keypoints are then scaled back
    to the original image resolution before RANSAC so the resulting affine
    matrix operates in full-resolution pixel space.

    Parameters
    ----------
    fixed_log  : uint8 log-normalised CK (reference)
    moving_log : uint8 log-normalised CK (moving)
    slice_id   : logging identifier
    orig_h/w   : full-resolution image dimensions (for coordinate rescaling)

    Returns
    -------
    (M, n_matches, n_inliers, kptsA_full, kptsB_full, inlier_mask)
    M is None on failure.
    """
    model  = get_romav2_model()
    device = next(model.parameters()).device

    img_A = ck_to_rgb_pil(fixed_log)
    img_B = ck_to_rgb_pil(moving_log)

    with torch.no_grad():
        preds = model.match(img_A, img_B)
        matches, certainties, precision_AB, precision_BA = model.sample(
            preds, ROMAV2_N_MATCHES
        )

    # certainties here is the "overlap" confidence in [0,1]
    if ROMAV2_MIN_OVERLAP > 0.0:
        keep     = certainties >= ROMAV2_MIN_OVERLAP
        matches  = matches[keep]
        certainties = certainties[keep]

    n_matches = int(matches.shape[0])
    logger.info(f"[{slice_id}] RoMaV2: {n_matches} matches after overlap filter.")

    if n_matches < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Too few RoMaV2 matches ({n_matches}).")
        return None, n_matches, 0, None, None, np.array([])

    # Convert from [-1,1] normalised coords to full-resolution pixel coords.
    # RoMaV2 produces matches in [-1,1]×[-1,1] relative to ROMAV2_H/W,
    # but to_pixel_coordinates scales to the H/W you supply.
    kptsA, kptsB = model.to_pixel_coordinates(matches, orig_h, orig_w, orig_h, orig_w)
    kptsA = kptsA.cpu().numpy()   # (N, 2) — (x, y) in full-res pixels
    kptsB = kptsB.cpu().numpy()

    src_pts = kptsB.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kptsA.reshape(-1, 1, 2).astype(np.float32)

    M, mask = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RoMaV2 RANSAC diverged.")
        return None, n_matches, 0, kptsA, kptsB, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] RoMaV2 inlier count too low ({n_inliers}).")
        return None, n_matches, n_inliers, kptsA, kptsB, mask

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] RoMaV2 transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, n_matches, n_inliers, kptsA, kptsB, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] RoMaV2: matches={n_matches} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, n_matches, n_inliers, kptsA, kptsB, mask


# --- L1: B-SPLINE (copy from production — tissue mask + per-tile acceptance) ---

def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    img = ck_log.astype(np.uint8)
    if BSPLINE_MASK_OTSU:
        nonzero = img[img > 0]
        if len(nonzero) == 0:
            return np.zeros_like(img)
        thresh, _ = cv2.threshold(
            nonzero.reshape(-1, 1), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        mask = (img > thresh).astype(np.uint8) * 255
    else:
        mask = (img > 0).astype(np.uint8) * 255
    if BSPLINE_MASK_DILATE_PX > 0:
        r    = BSPLINE_MASK_DILATE_PX
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        mask = cv2.dilate(mask, kern)
    return mask


def _measure_ncc_masked(fixed_f32, moving_f32, mask_uint8) -> float:
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


def _run_bspline_on_patch(fixed_patch, moving_patch, slice_id, tile_label, mask_patch=None):
    try:
        sitk_fixed  = sitk.GetImageFromArray(fixed_patch)
        sitk_moving = sitk.GetImageFromArray(moving_patch)
        tx_init = sitk.BSplineTransformInitializer(
            sitk_fixed, transformDomainMeshSize=[BSPLINE_GRID_NODES]*2, order=3,
        )
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.05)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=BSPLINE_ITERATIONS,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetInitialTransform(tx_init, inPlace=True)
        reg.SetShrinkFactorsPerLevel(BSPLINE_SHRINK_FACTORS)
        reg.SetSmoothingSigmasPerLevel(BSPLINE_SMOOTHING_SIGMAS)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        if mask_patch is not None and mask_patch.max() > 0:
            reg.SetMetricFixedMask(sitk.GetImageFromArray(mask_patch))
        final_tx  = reg.Execute(sitk_fixed, sitk_moving)
        ncc_value = reg.GetMetricValue()
        logger.info(
            f"[{slice_id}][{tile_label}] B-spline NCC: "
            f"stop={reg.GetOptimizerStopConditionDescription()} "
            f"metric={ncc_value:.6f} iters={reg.GetOptimizerIteration()}"
        )
        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(sitk_fixed)
        disp_np = sitk.GetArrayFromImage(disp_filter.Execute(final_tx))
        return disp_np, ncc_value, True
    except Exception as exc:
        logger.warning(f"[{slice_id}][{tile_label}] B-spline patch failed ({exc}).")
        return None, 0.0, False


def apply_bspline_l1(fixed_log, moving_log_affine, moving_affine_vol, slice_id, tissue_mask=None):
    try:
        c, h, w    = moving_affine_vol.shape
        fixed_f32  = fixed_log.astype(np.float32)
        moving_f32 = moving_log_affine.astype(np.float32)
        mask_f32   = tissue_mask if (BSPLINE_USE_TISSUE_MASK and tissue_mask is not None) else None

        if not BSPLINE_USE_GRID_TILES or (BSPLINE_GRID_ROWS == 1 and BSPLINE_GRID_COLS == 1):
            disp_np, ncc_value, ok = _run_bspline_on_patch(
                fixed_f32, moving_f32, slice_id, "whole", mask_patch=mask_f32,
            )
            if not ok:
                return None, 0.0, False, []
            tile_stats = []
        else:
            rows, cols = BSPLINE_GRID_ROWS, BSPLINE_GRID_COLS
            ovlp       = BSPLINE_TILE_OVERLAP
            tile_h, tile_w = h / rows, w / cols
            disp_acc   = np.zeros((h, w, 2), dtype=np.float64)
            weight_acc = np.zeros((h, w),    dtype=np.float64)
            ncc_values, tile_stats = [], []
            n_failed = 0

            for ri in range(rows):
                for ci in range(cols):
                    y0  = max(0, int(round(ri * tile_h - ovlp * tile_h)))
                    y1  = min(h, int(round((ri+1) * tile_h + ovlp * tile_h)))
                    x0  = max(0, int(round(ci * tile_w - ovlp * tile_w)))
                    x1  = min(w, int(round((ci+1) * tile_w + ovlp * tile_w)))
                    cy0 = int(round(ri * tile_h))
                    cy1 = min(h, int(round((ri+1) * tile_h)))
                    cx0 = int(round(ci * tile_w))
                    cx1 = min(w, int(round((ci+1) * tile_w)))
                    ph, pw = y1-y0, x1-x0
                    label  = f"r{ri}c{ci}"
                    fp         = fixed_f32[y0:y1, x0:x1]
                    mp         = moving_f32[y0:y1, x0:x1]
                    mask_tile  = mask_f32[y0:y1, x0:x1] if mask_f32 is not None else None
                    ncc_base   = _measure_ncc_masked(fp, mp, mask_tile) if BSPLINE_PER_TILE_ACCEPT else 0.0
                    disp_tile, ncc_t, ok_t = _run_bspline_on_patch(fp, mp, slice_id, label, mask_patch=mask_tile)
                    accepted = False
                    if not ok_t or disp_tile is None:
                        n_failed += 1
                        disp_tile = np.zeros((ph, pw, 2), dtype=np.float64)
                        ncc_t     = float('nan')
                    elif BSPLINE_PER_TILE_ACCEPT:
                        gain = (ncc_base - ncc_t) / abs(ncc_base) if abs(ncc_base) > 1e-9 else 0.0
                        if gain < BSPLINE_NCC_MIN_IMPROVEMENT:
                            logger.info(f"[{slice_id}][{label}] Tile gain {gain*100:.2f}% — zeroing.")
                            disp_tile = np.zeros((ph, pw, 2), dtype=np.float64)
                        else:
                            accepted = True
                            ncc_values.append(ncc_t)
                    else:
                        accepted = True
                        ncc_values.append(ncc_t)
                    tile_stats.append(dict(
                        ri=ri, ci=ci, label=label, y0=cy0, y1=cy1, x0=cx0, x1=cx1,
                        ncc=ncc_t, ncc_baseline=ncc_base, ok=ok_t, accepted=accepted,
                    ))
                    win_2d = np.outer(np.hanning(ph), np.hanning(pw)).astype(np.float64)
                    disp_acc[y0:y1, x0:x1, :] += disp_tile * win_2d[..., np.newaxis]
                    weight_acc[y0:y1, x0:x1]  += win_2d

            if n_failed == rows * cols:
                return None, 0.0, False, []
            disp_np   = disp_acc / (weight_acc[..., np.newaxis] + 1e-9)
            mean_tile_ncc = float(np.mean(ncc_values)) if ncc_values else 0.0
            n_acc = sum(1 for t in tile_stats if t['accepted'])
            logger.info(f"[{slice_id}] Grid B-spline: {rows}×{cols}, {n_failed} failed, {n_acc} accepted, mean tile NCC={mean_tile_ncc:.6f}")

        map_x = np.arange(w, dtype=np.float32)[None, :] + disp_np[..., 0].astype(np.float32)
        map_y = np.arange(h, dtype=np.float32)[:, None] + disp_np[..., 1].astype(np.float32)
        aligned_channels = [
            cv2.remap(moving_affine_vol[i].astype(np.float32), map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for i in range(c)
        ]
        aligned_vol = np.stack(aligned_channels, axis=0).astype(np.uint16)

        # Measure whole-image NCC on the warped CK channel so the acceptance
        # check in register_slice is comparing like-for-like with ncc_affine
        # (also whole-image).  Mean tile NCC is NOT comparable — tiles are
        # sub-patches and have different signal distributions.
        warped_ck_log = aligned_vol[CK_CHANNEL_IDX].astype(np.float32)
        _, warped_log_u8 = prepare_ck(warped_ck_log)
        ncc_value = _measure_ncc_masked(
            fixed_log.astype(np.float32),
            warped_log_u8.astype(np.float32),
            mask_f32,
        )
        logger.info(f"[{slice_id}] B-spline complete (whole-image NCC={ncc_value:.6f}).")
        return aligned_vol, ncc_value, True, tile_stats

    except Exception as exc:
        logger.warning(f"[{slice_id}] B-spline failed ({exc}).")
        return None, 0.0, False, []


# --- PLOTTING ---
def save_romav2_match_plot(fixed_log, moving_log, kptsA, kptsB, inlier_mask,
                           n_inliers, slice_id, output_folder, romav2_ok):
    """Saves an inlier match overlay plot for RoMaV2 (analogous to AKAZE inlier plot)."""
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)
    h, w    = fixed_log.shape
    gap     = 6
    canvas  = np.zeros((h, w*2+gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_log,  cv2.COLOR_GRAY2BGR)
    canvas[:, w+gap:]   = cv2.cvtColor(moving_log, cv2.COLOR_GRAY2BGR)

    if kptsA is not None and inlier_mask is not None and len(inlier_mask) > 0:
        inlier_idx = np.where(inlier_mask.ravel())[0][:200]
        for i, idx in enumerate(inlier_idx):
            hue   = int(i / max(len(inlier_idx)-1, 1) * 179)
            color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0,0])
            ptA   = (int(kptsA[idx, 0]), int(kptsA[idx, 1]))
            ptB   = (int(kptsB[idx, 0]) + w + gap, int(kptsB[idx, 1]))
            cv2.line(canvas, ptA, ptB, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, ptA, 6, (0,200,0), -1, cv2.LINE_AA)
            cv2.circle(canvas, ptB, 6, (0,0,200), -1, cv2.LINE_AA)

    status      = "SUCCESS" if romav2_ok else "FAILED"
    title_color = (0, 230, 0) if romav2_ok else (0, 0, 220)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, canvas.shape[1] / 3000)
    thick = max(1, int(scale*2))
    title = f"{slice_id}  RoMaV2 inliers={n_inliers}  [{status}]"
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    cv2.putText(canvas, title, ((canvas.shape[1]-tw)//2, th+10),
                font, scale, title_color, thick, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_romav2_inliers.png"), canvas)


def save_bspline_plot(fixed_log, moving_log_affine, aligned_ck,
                      ncc_affine, ncc_value, elastic_ok, slice_id, output_folder,
                      tile_stats=None):
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    def norm(x):
        p = np.percentile(x, 99.5)
        return np.clip(x.astype(np.float32) / (p if p > 0 else 1), 0, 1)

    f = norm(fixed_log.astype(np.float32))
    a = norm(moving_log_affine.astype(np.float32))
    _, aligned_log = prepare_ck(aligned_ck.astype(np.float32))
    b = norm(aligned_log.astype(np.float32))

    overlay_affine  = np.dstack((f, a, np.zeros_like(f)))
    overlay_bspline = np.dstack((f, b, np.zeros_like(f)))
    diff            = np.abs(a - b)
    diff_disp       = np.dstack([diff / (diff.max() + 1e-8)] * 3)

    use_grid  = tile_stats is not None and len(tile_stats) > 0
    n_panels  = 4 if use_grid else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 6))

    axes[0].imshow(overlay_affine);  axes[0].set_title("Fixed (R) vs RoMaV2-Affine (G)", fontsize=11); axes[0].axis('off')
    axes[1].imshow(overlay_bspline); axes[1].set_title("Fixed (R) vs B-spline (G)",       fontsize=11); axes[1].axis('off')
    axes[2].imshow(diff_disp);       axes[2].set_title("|Affine − B-spline| (CK)",         fontsize=11); axes[2].axis('off')

    if use_grid:
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle
        ax = axes[3]
        ax.imshow(f, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Grid tiles — NCC per tile", fontsize=11)
        ax.axis('off')
        ncc_vals = [t['ncc'] for t in tile_stats if t['ok'] and np.isfinite(t['ncc'])]
        ncc_min  = min(ncc_vals) if ncc_vals else -1.0
        ncc_max  = max(ncc_vals) if ncc_vals else 0.0
        for t in tile_stats:
            y0, y1, x0, x1 = t['y0'], t['y1'], t['x0'], t['x1']
            tw, th = x1-x0, y1-y0
            if not t['ok']:
                r, g, lbl = 1.0, 0.0, "FAIL"
            elif not t.get('accepted', True):
                r, g, lbl = 0.9, 0.35, f"{t['ncc']:.3f}\n✗"
            elif np.isfinite(t['ncc']) and (ncc_max-ncc_min) > 1e-6:
                s = (t['ncc']-ncc_max)/(ncc_min-ncc_max)
                r, g, lbl = 1.0-s, s, f"{t['ncc']:.3f}"
            else:
                r, g, lbl = 0.5, 0.5, f"{t['ncc']:.3f}"
            if BSPLINE_PER_TILE_ACCEPT and t['ok']:
                base = t.get('ncc_baseline', float('nan'))
                if np.isfinite(base) and abs(base) > 1e-9:
                    lbl += f"\n{(base-t['ncc'])/abs(base)*100:+.1f}%"
            ax.add_patch(Rectangle((x0,y0), tw, th, linewidth=2,
                edgecolor=(r,g,0,0.9), facecolor=(r,g,0,0.35)))
            fs = max(7, min(14, int(min(tw,th)/8)))
            ax.text(x0+tw/2, y0+th/2, f"{t['label']}\n{lbl}",
                    color='white', fontsize=fs, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.45, linewidth=0))
        cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["red","yellow","green"])
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=ncc_max, vmax=ncc_min))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02).set_label("NCC (more −ve = better)", fontsize=9)

    status   = "ACCEPTED" if elastic_ok else "REVERTED"
    rel_gain = ""
    if abs(ncc_affine) > 1e-9:
        rel_gain = f"  Δ={(ncc_affine-ncc_value)/abs(ncc_affine)*100:+.2f}%"
    fig.suptitle(
        f"{slice_id}  NCC affine={ncc_affine:.4f} → bspline={ncc_value:.4f}{rel_gain}  [{status}]",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{slice_id}_romav2_bspline.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)


# --- MAIN REGISTRATION PIPELINE ---
def register_slice(fixed_np, moving_np, slice_id=None):
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)
    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # L0: RoMaV2 dense matching → RANSAC affine
    M_affine, n_matches, n_inliers, kptsA, kptsB, inlier_mask = \
        romav2_affine(fixed_log, moving_log, sid, h, w)
    romav2_ok = M_affine is not None
    if not romav2_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # Match plot
    try:
        save_romav2_match_plot(
            fixed_log, moving_log, kptsA, kptsB, inlier_mask,
            n_inliers, sid, OUTPUT_FOLDER, romav2_ok,
        )
    except Exception as exc:
        logger.warning(f"[{sid}] Match plot failed: {exc}")

    # Pre-align all channels with affine
    c = moving_np.shape[0]
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # L1: B-spline elastic refinement
    aligned_np  = None
    elastic_ok  = False
    ncc_value   = 0.0
    ncc_affine  = 0.0
    tile_stats  = []
    tissue_mask = None

    if romav2_ok:
        moving_log_affine = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

        if BSPLINE_USE_TISSUE_MASK:
            tissue_mask = build_tissue_mask(fixed_log)
            tissue_frac = float(np.count_nonzero(tissue_mask)) / tissue_mask.size
            logger.info(f"[{sid}] Tissue mask: {tissue_frac*100:.1f}% of canvas.")
            if tissue_frac < BSPLINE_MIN_TISSUE_FRAC:
                logger.warning(f"[{sid}] Tissue fraction too low — skipping B-spline.")
                aligned_np = moving_affine_vol.astype(np.uint16)
                romav2_ok  = False

        if romav2_ok:
            ncc_affine = _measure_ncc_masked(
                fixed_log.astype(np.float32),
                moving_log_affine.astype(np.float32),
                tissue_mask,
            )
            logger.info(f"[{sid}] Affine NCC baseline: {ncc_affine:.4f}")

            aligned_np, ncc_value, elastic_ok, tile_stats = apply_bspline_l1(
                fixed_log, moving_log_affine, moving_affine_vol, sid,
                tissue_mask=tissue_mask,
            )

    # Acceptance / fallback
    if elastic_ok:
        ck_out = aligned_np[CK_CHANNEL_IDX]
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(f"[{sid}] CK output nearly blank — reverting.")
            elastic_ok = False
            aligned_np = moving_affine_vol.astype(np.uint16)
        else:
            if abs(ncc_affine) > 1e-9:
                rel = (ncc_affine - ncc_value) / abs(ncc_affine)
            else:
                rel = 0.0
            if rel < BSPLINE_NCC_MIN_IMPROVEMENT:
                logger.warning(
                    f"[{sid}] B-spline gain {rel*100:.2f}% < {BSPLINE_NCC_MIN_IMPROVEMENT*100:.0f}% — reverting."
                )
                elastic_ok = False
                aligned_np = moving_affine_vol.astype(np.uint16)
    else:
        aligned_np = moving_affine_vol.astype(np.uint16)

    # B-spline interim plot
    if romav2_ok:
        try:
            save_bspline_plot(
                fixed_log, moving_log_affine,
                aligned_np[CK_CHANNEL_IDX],
                ncc_affine, ncc_value, elastic_ok, sid, OUTPUT_FOLDER,
                tile_stats=tile_stats,
            )
        except Exception as exc:
            logger.warning(f"[{sid}] B-spline plot failed: {exc}")

    _, warped_log = prepare_ck(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_lin.astype(np.float32) - warped_log.astype(np.float32)) ** 2))

    U, S, Vt  = np.linalg.svd(M_affine[:2, :2])
    R         = U @ Vt
    rot       = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale_pct = (float(np.mean(S)) - 1.0) * 100.0
    shear_pct = (float(S[0] / S[1]) - 1.0) * 100.0 if S[1] > 1e-6 else 0.0

    stats = dict(
        detector     = "RoMaV2" if romav2_ok else "Identity",
        n_matches    = n_matches,
        n_inliers    = n_inliers,
        rotation_deg = round(rot, 3),
        tx           = round(float(M_affine[0, 2]), 3),
        ty           = round(float(M_affine[1, 2]), 3),
        scale_pct    = round(scale_pct, 3),
        shear_pct    = round(shear_pct, 3),
        bspline_ok   = elastic_ok,
        ncc_value    = round(float(ncc_value), 6),
        ncc_affine   = round(float(ncc_affine), 6),
    )

    affine_np = np.clip(moving_affine_vol, 0, 65535).astype(np.uint16)
    return aligned_np, affine_np, mse, time.time() - start, stats, romav2_ok, M_affine


def generate_qc_montage(vol, output_folder, slice_ids=None, channel_idx=6,
                        channel_name="CK", title_suffix="RoMaV2+BSpline"):
    n_slices = vol.shape[0]
    if n_slices < 2:
        return
    logger.info(f"  Montage [{title_suffix}]: rendering {n_slices-1} pairs...")
    all_pairs = [(i, i+1) for i in range(n_slices-1)]
    n_cols    = 5
    n_rows    = (len(all_pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
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

    core     = os.path.basename(output_folder)
    out_path = os.path.join(output_folder, f"{core}_QC_Montage_{channel_name}_{title_suffix}.png")
    fig.suptitle(f"Registration QC {title_suffix}: {core}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Montage [{title_suffix}] saved.")


# --- MAIN ---
def main():
    logger.info(f"RoMaV2 + B-spline Registration (Experiment) — {TARGET_CORE}")

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    if not file_list:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    n_slices = len(file_list)

    allowed_positions = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)
    if allowed_positions is not None:
        original_count = len(file_list)
        file_list = [f for i, f in enumerate(file_list) if i in allowed_positions]
        n_slices  = len(file_list)
        excluded  = original_count - n_slices
        logger.info(
            f"Slice filter active for {TARGET_CORE}: "
            f"keeping {n_slices}/{original_count} slices "
            f"(positions {sorted(allowed_positions)}), "
            f"{excluded} excluded."
        )
        if n_slices == 0:
            logger.error("Slice filter excluded all slices — check slice_filter.yaml.")
            sys.exit(1)
    else:
        logger.info(f"No slice filter for {TARGET_CORE} — using all {n_slices} slices.")

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Aligned.ome.tif")
        tifffile.imwrite(
            out_path, vol_in[np.newaxis],
            photometric='minisblack',
            metadata={
                'axes': 'ZCYX', 'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        sys.exit(0)

    # Pre-load RoMaV2 model so the first slice pair doesn't pay the load cost
    get_romav2_model()

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)
    c, target_h, target_w = _center_arr.shape

    # Only collect slice IDs — do NOT load all slices into memory.
    # Loading all slices simultaneously causes OOM on large cores
    # (20 slices × 8ch × 6112² × uint16 ≈ 23GB just for raw_slices).
    # Instead, read each slice from disk on demand during registration.
    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
        """Load and conform a single slice from disk."""
        arr = tifffile.imread(file_list[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            arr = conform_slice(arr, target_h, target_w)
        return arr

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    affine_vol              = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    affine_vol[center_idx]  = center_raw
    del center_raw

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = load_slice(i)   # read from disk, discard after use
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            try:
                aligned_np, affine_np, mse, runtime, stats, success, M_final = \
                    register_slice(fixed_np, moving_np, slice_id=sid)
            except Exception as exc:
                logger.error(f"[{sid}] register_slice crashed: {exc} — using raw fallback.")
                aligned_vol[i] = moving_np
                affine_vol[i]  = moving_np
                registration_stats.append({
                    "Direction": direction, "Slice_Z": i, "Slice_ID": real_id,
                    "Detector": "CRASH", "N_Matches": 0, "N_Inliers": 0,
                    "BSpline_OK": False, "NCC_Value": 0.0, "NCC_Affine": 0.0,
                    "Success": False, "Status": "CRASH",
                    "MSE_After": 0.0, "Rotation_Deg": 0.0,
                    "Shift_X_px": 0.0, "Shift_Y_px": 0.0,
                    "Scale_Pct": 0.0, "Shear_Pct": 0.0, "Runtime_s": 0.0,
                })
                del moving_np
                continue

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                affine_vol[i]  = affine_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = moving_np
                affine_vol[i]  = moving_np
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str}")

            del moving_np   # free immediately — no longer needed

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"BSpline: {stats['bspline_ok']} | NCC: {stats['ncc_value']:.4f} | "
                f"MSE: {mse:.2f} | t: {runtime:.2f}s | Status: {status_str}"
            )
            registration_stats.append({
                "Direction": direction, "Slice_Z": i, "Slice_ID": real_id,
                "Detector": stats["detector"], "N_Matches": stats["n_matches"],
                "N_Inliers": stats["n_inliers"], "BSpline_OK": stats["bspline_ok"],
                "NCC_Value": stats["ncc_value"], "NCC_Affine": stats["ncc_affine"],
                "Success": success, "Status": status_str, "MSE_After": round(mse, 4),
                "Rotation_Deg": stats["rotation_deg"], "Shift_X_px": stats["tx"],
                "Shift_Y_px": stats["ty"], "Scale_Pct": stats["scale_pct"],
                "Shear_Pct": stats["shear_pct"], "Runtime_s": round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx-1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx+1, n_slices), "Forward")

    logger.info(f"All passes complete. {len(registration_stats)} slices registered.")
    if registration_stats:
        df = pd.DataFrame(registration_stats).sort_values("Slice_Z")
        df.to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats_RoMaV2_BSpline.csv"), index=False)
        logger.info("CSV stats written.")
    else:
        logger.warning("No registration stats to write — registration_stats is empty.")

    logger.info("Writing registered volume...")
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Aligned.ome.tif")
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
        logger.info(f"Registered volume written to {out_tiff}")
    except Exception as exc:
        logger.error(f"Volume write failed: {exc}")

    # Generate montages one at a time, freeing each before the next.
    # Raw montage reads from disk to avoid holding another full volume in RAM.
    logger.info("Generating QC montages...")
    try:
        generate_qc_montage(affine_vol, OUTPUT_FOLDER, slice_ids=slice_ids, title_suffix="RoMaV2_Affine")
    except Exception as exc:
        logger.error(f"Affine montage failed: {exc}")
    del affine_vol

    try:
        generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids, title_suffix="RoMaV2+BSpline")
    except Exception as exc:
        logger.error(f"BSpline montage failed: {exc}")
    del aligned_vol

    logger.info("Done.")

if __name__ == "__main__":
    main()