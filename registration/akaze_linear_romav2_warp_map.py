"""
Feature registration — AKAZE affine pre-alignment → RoMaV2 dense warp residual.

Pipeline per slice pair:
  L0  AKAZE tissue-masked detection → RANSAC affine pre-alignment
      Reduces global translation/rotation so RoMaV2 only sees residual deformation.
  L1  RoMaV2 dense warp on affine-prealigned CK images
      Handles local non-rigid tissue deformation that the affine cannot model.

Why this order:
  RoMaV2 has a finite receptive field at 448×448 internal resolution (~13px/cell
  on a 6000px image). Large global shifts (e.g. 387px) push correspondences far
  outside any cell's receptive field, causing mass displacement capping and poor
  confidence maps. Affine pre-alignment collapses the global offset so RoMaV2
  only needs to recover the small residual deformation it is actually designed for.

Fallback hierarchy:
  L0 fails  → run RoMaV2 on raw images (better than identity for moderate shifts)
  L0 succeeds but affine NCC ≤ raw NCC → revert affine to identity; RoMaV2 on raw
  L1 fails  → use affine-only result (L0 output)
  L1 does not improve NCC by WARP_NCC_MIN_IMPROVEMENT over affine → revert to affine
  Both fail → identity (raw moving slice)

NCC is measured at three points for comparison:
  ncc_raw    — raw moving vs fixed (before any alignment)
  ncc_affine — after AKAZE affine (L0 output vs fixed)
  ncc_warp   — after RoMaV2 warp  (L1 output vs fixed)
All three use the same tissue mask and log-normalised CK channel.

Deformation maps:
  For each registered slice, a .npz file is saved containing:
    M_affine : (2, 3) float64 — affine matrix applied at L0
    map_x    : (H, W) float32 — full-resolution X remap (L1, or None if warp failed)
    map_y    : (H, W) float32 — full-resolution Y remap (L1, or None if warp failed)
    warp_ok  : bool — whether the RoMaV2 warp succeeded
    akaze_ok : bool — whether AKAZE affine succeeded
    orig_h   : int  — image height
    orig_w   : int  — image width

  These maps can be applied to CellPose segmentation masks (run on the original,
  unregistered images) using apply_deformation_to_mask() or the companion script
  warp_cellpose_masks.py.

  Warping semantics — the maps describe a FORWARD warp (moving→fixed):
    map_x[y, x] = x-coordinate in the source (moving) image that maps to pixel (x, y)
    map_y[y, x] = y-coordinate in the source (moving) image that maps to pixel (x, y)
  Apply with cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST) using INTER_NEAREST
  for label masks to avoid interpolating integer cell IDs across boundaries.
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
parser = argparse.ArgumentParser(
    description='Registration: AKAZE affine pre-alignment → RoMaV2 dense warp.'
)
parser.add_argument('--core_name', type=str, required=True)
args = parser.parse_args()

TARGET_CORE = args.core_name

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_AKAZE_RoMaV2_Linear_Warp_map")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# Sub-folder where deformation .npz files are written, one per slice pair.
# Filename convention:  <core>_Z<idx>_ID<slice_id>_deformation.npz
DEFORM_FOLDER = os.path.join(OUTPUT_FOLDER, "deformation_maps")

# ─────────────────────────────────────────────────────────────────────────────
# L0: AKAZE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AKAZE_THRESHOLD     = 0.0001  # low threshold — tissue mask is the quality gate
AKAZE_MAX_KEYPOINTS = 20_000  # hard cap after tissue-masked detection to avoid
                               # BFMatcher IMGIDX_ONE assertion on large descriptor sets
LOWE_RATIO          = 0.80
MIN_MATCHES         = 20
MIN_INLIERS         = 6
RANSAC_CONFIDENCE   = 0.995
RANSAC_MAX_ITERS    = 5000
RANSAC_THRESH       = 8.0     # pixels at full resolution
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# L1: ROMAV2 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# After affine pre-alignment the residual shift is small, so the displacement
# cap can be much tighter than in the raw-image RoMaV2 script.
# 100px is generous for a residual deformation after affine alignment.
ROMAV2_DEVICE            = 'cpu'   # 'cuda' when GPU available
ROMAV2_H                 = 448
ROMAV2_W                 = 448
ROMAV2_H_HR              = None
ROMAV2_W_HR              = None
WARP_CONFIDENCE_THRESH   = 0.5     # 0.0 = no filtering; 0.5 is a safe default
WARP_MAX_DISPLACEMENT_PX = 200.0   # tighter cap — residual after affine should be small
# RoMaV2 warp acceptance criterion — mirrors BSPLINE_NCC_MIN_IMPROVEMENT in the B-spline
# script.  The warp result is accepted only if it improves NCC (relative to the affine
# baseline) by at least this fraction.
# NCC is negative — improvement means becoming more negative.
# Relative improvement = (ncc_affine − ncc_warp) / |ncc_affine|
# e.g. ncc_affine=−0.72, ncc_warp=−0.76 → gain=(−0.72−−0.76)/0.72 = +0.056 → accepted.
# Set to 0.0 to accept any non-negative improvement; raise to be stricter.
WARP_NCC_MIN_IMPROVEMENT = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE MASK
# ─────────────────────────────────────────────────────────────────────────────
MASK_DILATE_PX   = 20    # morphological dilation radius (px) to cover tissue edges
MASK_MIN_FRAC    = 0.05  # skip RoMaV2 if tissue covers < 5% of canvas

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
os.makedirs(DEFORM_FOLDER, exist_ok=True)

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
    norm_log : log-stretch then percentile normalise → used for AKAZE
               detection, tissue masking, and NCC measurement.
    """
    img_float = img_arr.astype(np.float32)

    # Linear normalisation — for RoMaV2
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Log normalisation — for AKAZE, tissue mask, NCC
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
# TISSUE MASK
# ─────────────────────────────────────────────────────────────────────────────

def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """
    Binary tissue mask from log-normalised CK channel.
    Otsu threshold on non-zero pixels + morphological dilation.
    Returns uint8 (H, W): 255 = tissue, 0 = background.
    """
    img     = ck_log.astype(np.uint8)
    nonzero = img[img > 0]
    if len(nonzero) == 0:
        return np.zeros_like(img)
    thresh, _ = cv2.threshold(
        nonzero.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = (img > thresh).astype(np.uint8) * 255
    if MASK_DILATE_PX > 0:
        r    = MASK_DILATE_PX
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        mask = cv2.dilate(mask, kern)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# NCC MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_ncc(fixed_f32: np.ndarray,
                moving_f32: np.ndarray,
                mask_uint8: np.ndarray) -> float:
    """
    Masked NCC via 0-iteration SimpleITK LBFGSB (evaluation only).
    NCC is negative — more negative = better alignment.
    Returns 0.0 on failure.
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
# DEFORMATION MAP I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_deformation_maps(slice_id: str,
                          M_affine: np.ndarray,
                          map_x: np.ndarray | None,
                          map_y: np.ndarray | None,
                          akaze_ok: bool,
                          warp_ok: bool,
                          orig_h: int,
                          orig_w: int) -> str:
    """
    Save the composite deformation for one slice as a compressed .npz.

    The file holds everything needed to warp a CellPose mask (or any label
    image) from the original moving-slice space into the registered fixed space:

        M_affine : (2, 3) float64  — AKAZE affine (identity if AKAZE failed)
        map_x    : (H, W) float32  — full-res X remap from RoMaV2 stage
                                     (identity grid if warp failed)
        map_y    : (H, W) float32  — full-res Y remap from RoMaV2 stage
        akaze_ok : bool
        warp_ok  : bool
        orig_h   : int
        orig_w   : int

    Applying to a mask (see also apply_deformation_to_mask):
        Step 1 — affine:  mask_affine = cv2.warpAffine(mask, M_affine, (W, H),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Step 2 — remap:   mask_final  = cv2.remap(mask_affine, map_x, map_y,
                              cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    Returns the path of the saved .npz file.
    """
    # If warp failed, store identity remap so the loading code never has to
    # branch — it always applies both steps in sequence.
    if map_x is None or map_y is None:
        grid_x = np.arange(orig_w, dtype=np.float32)[None, :]
        grid_y = np.arange(orig_h, dtype=np.float32)[:, None]
        map_x, map_y = np.broadcast_to(grid_x, (orig_h, orig_w)).copy(), \
                       np.broadcast_to(grid_y, (orig_h, orig_w)).copy()

    out_path = os.path.join(DEFORM_FOLDER, f"{TARGET_CORE}_{slice_id}_deformation.npz")
    np.savez_compressed(
        out_path,
        M_affine = M_affine.astype(np.float64),
        map_x    = map_x.astype(np.float32),
        map_y    = map_y.astype(np.float32),
        akaze_ok = np.bool_(akaze_ok),
        warp_ok  = np.bool_(warp_ok),
        orig_h   = np.int32(orig_h),
        orig_w   = np.int32(orig_w),
    )
    logger.info(f"[{slice_id}] Deformation map saved → {out_path}")
    return out_path


def apply_deformation_to_mask(mask: np.ndarray,
                               npz_path: str,
                               interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """
    Utility: apply a saved deformation .npz to a CellPose label mask (or any
    2-D array) and return the warped result.

    Parameters
    ----------
    mask          : (H, W) integer label array from CellPose (or float image).
    npz_path      : path to the .npz saved by save_deformation_maps().
    interpolation : cv2 interpolation flag.  Use cv2.INTER_NEAREST for integer
                    label masks (preserves cell IDs); cv2.INTER_LINEAR for
                    probability / intensity images.

    Returns
    -------
    warped : same dtype as mask, same shape.

    Notes
    -----
    The mask must come from running CellPose on the *original, unregistered*
    moving slice.  The deformation was computed relative to that slice.
    """
    d        = np.load(npz_path)
    M_affine = d['M_affine']                    # (2, 3) float64
    map_x    = d['map_x'].astype(np.float32)    # (H, W)
    map_y    = d['map_y'].astype(np.float32)    # (H, W)
    h, w     = int(d['orig_h']), int(d['orig_w'])

    mask_f32 = mask.astype(np.float32)

    # Step 1: affine
    mask_affine = cv2.warpAffine(
        mask_f32, M_affine, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Step 2: dense remap
    warped = cv2.remap(
        mask_affine, map_x, map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    return warped.astype(mask.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# L0: AKAZE AFFINE
# ─────────────────────────────────────────────────────────────────────────────

def constrain_affine(M: np.ndarray) -> np.ndarray:
    if M is None:
        return None
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


def akaze_affine(fixed_log, moving_log, slice_id,
                 fixed_mask=None, moving_mask=None):
    """
    Tissue-masked AKAZE detection → BFMatcher → RANSAC affine.
    Returns (M, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask).
    M is None on failure.
    """
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    if fixed_mask is None:
        fixed_mask  = build_tissue_mask(fixed_log)
    if moving_mask is None:
        moving_mask = build_tissue_mask(moving_log)

    kp1_raw, des1 = detector.detectAndCompute(fixed_log,  fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_log, moving_mask)

    n1, n2 = len(kp1_raw) if kp1_raw else 0, len(kp2_raw) if kp2_raw else 0
    if n1 > 0:
        coords = np.array([kp.pt for kp in kp1_raw])
        logger.info(f"[{slice_id}] AKAZE (tissue-masked): n={n1}, "
                    f"y_range=[{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]")
    else:
        logger.info(f"[{slice_id}] AKAZE (tissue-masked): n=0")

    if des1 is None or des2 is None or n1 < 4 or n2 < 4:
        logger.warning(f"[{slice_id}] Feature starvation (fixed={n1}, moving={n2}).")
        return None, 0, 0, [], [], [], np.array([])

    # Cap by response to avoid BFMatcher IMGIDX_ONE overflow
    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)
    logger.info(f"[{slice_id}] After cap: fixed={len(kp1)}, moving={len(kp2)}")

    matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw      = matcher.knnMatch(des1, des2, k=2)
    good     = [m for m, n in raw
                if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected (rot={rot:.1f}°).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, kp1, kp2, good, mask


# ─────────────────────────────────────────────────────────────────────────────
# L1: ROMAV2 MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        device = ROMAV2_DEVICE if (ROMAV2_DEVICE == 'cuda'
                                   and torch.cuda.is_available()) else 'cpu'
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


def romav2_dense_warp(fixed_lin, moving_lin, slice_id, orig_h, orig_w,
                      tissue_mask_full=None):
    """
    Run RoMaV2 on a (fixed_lin, moving_lin) pair and return remap maps.

    Inputs must be linear-normalised uint8 — matches RoMaV2's pretraining
    distribution (natural RGB images with standard 0–255 linear intensities).
    Inputs are expected to be affine-prealigned so the residual displacement
    is small — the cap WARP_MAX_DISPLACEMENT_PX can therefore be tight.

    tissue_mask_full : uint8 (orig_h, orig_w) — 255=tissue, 0=background.
                       Background cells are forced to identity BEFORE upsampling
                       so featureless-canvas vectors cannot bleed into tissue
                       edges and create hard seam artefacts.

    Returns (map_x, map_y, n_confident, coverage_pct, mean_confidence)
    or      (None,  None,  0,           0.0,          0.0) on failure.
    """
    try:
        model = get_romav2_model()
        img_A = ck_to_rgb_pil(fixed_lin)
        img_B = ck_to_rgb_pil(moving_lin)

        with torch.no_grad():
            preds = model.match(img_A, img_B)

        warp_AB    = preds['warp_AB'].squeeze(0).cpu().numpy()    # (H_lr, W_lr, 2)
        overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy()  # (H_lr, W_lr)

        H_lr, W_lr = warp_AB.shape[:2]

        # Convert B-side coords from [-1,1] to full-resolution pixel space
        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

        confident_2d    = overlap_AB.reshape(H_lr, W_lr) >= WARP_CONFIDENCE_THRESH
        n_confident     = int(confident_2d.sum())
        coverage_pct    = n_confident / (H_lr * W_lr) * 100
        mean_confidence = float(overlap_AB.mean())

        # Identity coordinates at lr resolution
        grid_x_lr  = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
        grid_y_lr  = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
        identity_x, identity_y = np.meshgrid(grid_x_lr, grid_y_lr)

        # Apply confidence mask
        map_x_lr = np.where(confident_2d, b_coords_x, identity_x).astype(np.float32)
        map_y_lr = np.where(confident_2d, b_coords_y, identity_y).astype(np.float32)

        # Cap displacement magnitude
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
        if tissue_mask_full is not None:
            mask_lr    = cv2.resize(
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

        # Upsample to full resolution — INTER_CUBIC reduces ringing at the
        # hard tissue/background boundary compared to bilinear.
        map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        logger.info(
            f"[{slice_id}] RoMaV2 warp: {H_lr}×{W_lr} grid, "
            f"{coverage_pct:.1f}% confident, mean confidence={mean_confidence:.3f}"
        )
        return map_x, map_y, n_confident, coverage_pct, mean_confidence

    except Exception as exc:
        logger.error(f"[{slice_id}] RoMaV2 dense warp failed: {exc}")
        return None, None, 0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CORE: REGISTER ONE SLICE PAIR
# ─────────────────────────────────────────────────────────────────────────────

def register_slice(fixed_np, moving_np, slice_id=None):
    """
    Full pipeline: AKAZE affine → RoMaV2 dense warp.

    Returns (aligned_np, affine_np, elapsed, stats, success, M_affine,
             map_x, map_y).
    affine_np is the L0-only result (useful for montage comparison).
    map_x / map_y are the full-resolution RoMaV2 remap arrays (or None).
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape
    c    = fixed_np.shape[0]

    # Tissue mask and NCC use log images — better tissue/background separation
    fixed_tissue_mask  = build_tissue_mask(fixed_log)
    moving_tissue_mask = build_tissue_mask(moving_log)
    tissue_frac        = float(np.count_nonzero(fixed_tissue_mask)) / fixed_tissue_mask.size
    logger.info(f"[{sid}] Tissue mask: {tissue_frac*100:.1f}% of canvas covered.")

    # ── NCC raw (before any alignment) — log space, masked ───────────────────
    ncc_raw = measure_ncc(
        fixed_log.astype(np.float32),
        moving_log.astype(np.float32),
        fixed_tissue_mask,
    )
    logger.info(f"[{sid}] NCC raw (before alignment): {ncc_raw:.4f}")

    # ── L0: AKAZE affine — uses log images for detection ─────────────────────
    M_affine, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_log, moving_log, sid,
                     fixed_mask=fixed_tissue_mask,
                     moving_mask=moving_tissue_mask)

    akaze_ok = M_affine is not None
    if not akaze_ok:
        logger.warning(f"[{sid}] AKAZE failed — RoMaV2 will run on raw images.")
        M_affine = np.eye(2, 3, dtype=np.float64)

    # Save AKAZE inlier plot
    if len(kp1) > 0:
        try:
            save_inlier_plot(fixed_log, moving_log, kp1, kp2,
                             good_matches, inlier_mask, sid, akaze_ok)
        except Exception as exc:
            logger.warning(f"[{sid}] Inlier plot failed: {exc}")

    # Apply affine to all channels
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    affine_np = np.clip(moving_affine_vol, 0, 65535).astype(np.uint16)

    # Affine-prealigned CK — both lin (for RoMaV2) and log (for NCC)
    moving_lin_affine = cv2.warpAffine(
        moving_lin, M_affine, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    moving_log_affine = cv2.warpAffine(
        moving_log, M_affine, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # ── NCC affine — log space, masked ───────────────────────────────────────
    ncc_affine = measure_ncc(
        fixed_log.astype(np.float32),
        moving_log_affine.astype(np.float32),
        fixed_tissue_mask,
    )
    logger.info(f"[{sid}] NCC after affine: {ncc_affine:.4f}")

    # ── Affine NCC acceptance — revert to identity if affine made things worse ─
    # AKAZE succeeded geometrically (enough inliers, sane matrix) but the affine
    # transform could still hurt NCC — e.g. a slightly wrong rotation on a low-
    # texture core.  If the affine NCC is worse than raw, fall back to identity
    # so RoMaV2 runs on the unmodified images rather than a degraded baseline.
    # "Worse" = ncc_affine > ncc_raw (less negative), i.e. no improvement at all.
    # We use a strict threshold of 0.0 (any regression reverts) because even a
    # small affine degradation compounds once RoMaV2's acceptance is relative to
    # ncc_affine — a worse baseline makes the warp easier to accept spuriously.
    if akaze_ok and ncc_affine > ncc_raw:
        logger.warning(
            f"[{sid}] Affine NCC ({ncc_affine:.4f}) is worse than raw "
            f"({ncc_raw:.4f}) — reverting affine to identity so RoMaV2 "
            "sees the unmodified images."
        )
        akaze_ok          = False
        M_affine          = np.eye(2, 3, dtype=np.float64)
        moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
        for ch in range(moving_np.shape[0]):
            moving_affine_vol[ch] = moving_np[ch].astype(np.float32)
        affine_np         = moving_np.copy()
        moving_lin_affine = moving_lin.copy()
        moving_log_affine = moving_log.copy()
        ncc_affine        = ncc_raw   # RoMaV2 acceptance is now relative to raw NCC

    # ── L1: RoMaV2 warp on affine-prealigned LINEAR images ───────────────────
    warp_ok         = False
    ncc_warp        = 0.0
    n_confident     = 0
    coverage_pct    = 0.0
    mean_confidence = 0.0
    aligned_np      = affine_np.copy()   # default: affine-only
    map_x = map_y = None                 # returned for deformation saving

    if tissue_frac >= MASK_MIN_FRAC:
        map_x, map_y, n_confident, coverage_pct, mean_confidence = romav2_dense_warp(
            fixed_lin, moving_lin_affine, sid, h, w,
            tissue_mask_full=fixed_tissue_mask,
        )

        if map_x is not None:
            # Apply warp to affine-prealigned volume
            warped_channels = []
            for ch in range(c):
                warped_channels.append(cv2.remap(
                    moving_affine_vol[ch],
                    map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ))
            warp_candidate = np.stack(warped_channels, axis=0).astype(np.uint16)

            # Blank-output sanity check
            ck_out = warp_candidate[CK_CHANNEL_IDX]
            if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
                logger.warning(
                    f"[{sid}] RoMaV2 output nearly blank — reverting to affine."
                )
                map_x = map_y = None   # mark as failed so identity is stored
            else:
                # NCC after warp — log space for consistency with ncc_raw / ncc_affine
                _, warped_log = prepare_ck(ck_out.astype(np.float32))
                ncc_warp      = measure_ncc(
                    fixed_log.astype(np.float32),
                    warped_log.astype(np.float32),
                    fixed_tissue_mask,
                )
                logger.info(f"[{sid}] NCC after RoMaV2 warp: {ncc_warp:.4f}")

                # ── NCC monotonic acceptance — mirrors bspline strategy ───────
                # Accept the warp only if it improves NCC over the affine
                # baseline by at least WARP_NCC_MIN_IMPROVEMENT (relative).
                # NCC is negative — improvement means becoming more negative.
                # This guards against RoMaV2 producing a plausible-looking warp
                # that nonetheless degrades alignment (e.g. on low-texture
                # regions where the confidence filter alone is insufficient).
                if abs(ncc_affine) > 1e-9:
                    relative_improvement = (ncc_affine - ncc_warp) / abs(ncc_affine)
                else:
                    relative_improvement = 0.0

                if relative_improvement < WARP_NCC_MIN_IMPROVEMENT:
                    logger.warning(
                        f"[{sid}] RoMaV2 NCC ({ncc_warp:.4f}) did not improve enough "
                        f"over affine baseline ({ncc_affine:.4f}): "
                        f"relative gain={relative_improvement*100:.2f}% < required "
                        f"{WARP_NCC_MIN_IMPROVEMENT*100:.0f}% "
                        "— reverting to affine-only result."
                    )
                    map_x = map_y = None   # store identity so downstream never branches
                else:
                    aligned_np = warp_candidate
                    warp_ok    = True
                    logger.info(
                        f"[{sid}] RoMaV2 warp accepted "
                        f"(NCC gain={relative_improvement*100:.2f}%)."
                    )
        else:
            logger.warning(f"[{sid}] RoMaV2 warp failed — using affine-only result.")
    else:
        logger.warning(
            f"[{sid}] Tissue fraction {tissue_frac*100:.1f}% < "
            f"{MASK_MIN_FRAC*100:.0f}% — skipping RoMaV2."
        )

    # ── Save deformation maps ─────────────────────────────────────────────────
    # Always saved — even on failure an identity warp is stored so downstream
    # CellPose mask warping code can load without branching.
    try:
        save_deformation_maps(
            slice_id  = sid,
            M_affine  = M_affine,
            map_x     = map_x,
            map_y     = map_y,
            akaze_ok  = akaze_ok,
            warp_ok   = warp_ok,
            orig_h    = h,
            orig_w    = w,
        )
    except Exception as exc:
        logger.error(f"[{sid}] Failed to save deformation maps: {exc}")

    # Overall success: at minimum affine must have worked or warp improved things
    success = akaze_ok or warp_ok

    # NCC improvement at each stage
    def pct_improvement(before, after):
        if abs(before) > 1e-9:
            return (before - after) / abs(before) * 100.0
        return 0.0

    ncc_affine_improvement = pct_improvement(ncc_raw,    ncc_affine)
    ncc_warp_improvement   = pct_improvement(ncc_affine, ncc_warp) if warp_ok else 0.0
    ncc_total_improvement  = pct_improvement(ncc_raw,    ncc_warp if warp_ok else ncc_affine)

    logger.info(
        f"[{sid}] NCC summary: raw={ncc_raw:.4f} → "
        f"affine={ncc_affine:.4f} ({ncc_affine_improvement:+.1f}%) → "
        f"warp={ncc_warp:.4f} ({ncc_warp_improvement:+.1f}%) | "
        f"total={ncc_total_improvement:+.1f}%"
    )

    # Decompose affine for stats
    U, S, Vt  = np.linalg.svd(M_affine[:2, :2])
    R         = U @ Vt
    rot       = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale_pct = (float(np.mean(S)) - 1.0) * 100.0
    shear_pct = (float(S[0] / S[1]) - 1.0) * 100.0 if S[1] > 1e-6 else 0.0

    stats = dict(
        akaze_ok            = akaze_ok,
        warp_ok             = warp_ok,
        n_matches           = n_matches,
        n_inliers           = n_inliers,
        rotation_deg        = round(rot, 3),
        tx                  = round(float(M_affine[0, 2]), 3),
        ty                  = round(float(M_affine[1, 2]), 3),
        scale_pct           = round(scale_pct, 3),
        shear_pct           = round(shear_pct, 3),
        n_confident         = n_confident,
        coverage_pct        = round(coverage_pct, 2),
        mean_confidence     = round(mean_confidence, 4),
        ncc_raw             = round(float(ncc_raw),    6),
        ncc_affine          = round(float(ncc_affine), 6),
        ncc_warp            = round(float(ncc_warp),   6),
        ncc_affine_improv   = round(float(ncc_affine_improvement), 2),
        ncc_warp_improv     = round(float(ncc_warp_improvement),   2),
        ncc_total_improv    = round(float(ncc_total_improvement),  2),
    )

    # Save interim plot
    try:
        map_x_plot = map_x if warp_ok and map_x is not None else None
        map_y_plot = map_y if warp_ok and map_y is not None else None
        save_registration_plot(
            fixed_log, moving_log, moving_log_affine,
            map_x_plot, map_y_plot,
            ncc_raw, ncc_affine, ncc_warp,
            akaze_ok, warp_ok, sid,
            fixed_lin=fixed_lin,
            moving_lin=moving_lin,
            moving_lin_affine=moving_lin_affine,
        )
    except Exception as exc:
        logger.warning(f"[{sid}] Registration plot failed: {exc}")

    # Save vector field plot if warp succeeded
    try:
        if warp_ok and map_x is not None and map_y is not None:
            save_deformation_quiver_plot(map_x, map_y, sid, step=150)
    except Exception as exc:
        logger.warning(f"[{sid}] Vector field plot failed: {exc}")

    return aligned_np, affine_np, time.time() - start, stats, success, M_affine, map_x, map_y


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def save_inlier_plot(fixed_log, moving_log, kp1, kp2,
                     good_matches, inlier_mask, slice_id, akaze_ok):
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    h, w    = fixed_log.shape
    gap     = 6
    canvas  = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_log,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_log, cv2.COLOR_GRAY2BGR)

    inlier_matches = ([m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep]
                      if len(inlier_mask) > 0 else [])

    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in
                          cv2.cvtColor(np.uint8([[[hue, 220, 220]]]),
                                       cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), color_bgr, 1, cv2.LINE_AA)

    status      = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)
    title       = f"{slice_id}  inliers={len(inlier_matches)}  [{status}]"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    scale       = max(0.8, canvas.shape[1] / 3000)
    thickness   = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    cv2.putText(canvas, title, ((canvas.shape[1] - tw) // 2, th + 10),
                font, scale, title_color, thickness, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_inliers.png"), canvas)
    logger.info(f"[{slice_id}] Inlier plot saved ({len(inlier_matches)} inliers).")


def save_registration_plot(fixed_log, moving_log, moving_log_affine,
                           map_x, map_y,
                           ncc_raw, ncc_affine, ncc_warp,
                           akaze_ok, warp_ok, slice_id,
                           fixed_lin=None, moving_lin=None,
                           moving_lin_affine=None):
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    def norm(x):
        p = np.percentile(x, 99.5)
        return np.clip(x.astype(np.float32) / (p if p > 0 else 1), 0, 1)

    use_lin   = fixed_lin is not None
    label     = "linear" if use_lin else "log"

    f         = norm(fixed_lin        if use_lin else fixed_log)
    m_raw     = norm(moving_lin       if use_lin else moving_log)
    m_affine  = norm(moving_lin_affine if use_lin else moving_log_affine)
    src_remap = (moving_lin_affine if use_lin else moving_log_affine).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(np.dstack((f, m_raw, np.zeros_like(f))))
    axes[0].set_title(f"Fixed (R) vs Raw (G) [{label}]\nNCC={ncc_raw:.4f}", fontsize=10)
    axes[0].axis('off')

    affine_status = "AKAZE OK" if akaze_ok else "identity"
    axes[1].imshow(np.dstack((f, m_affine, np.zeros_like(f))))
    axes[1].set_title(
        f"Fixed (R) vs Affine (G) [{affine_status}]\nNCC={ncc_affine:.4f}", fontsize=10
    )
    axes[1].axis('off')

    if map_x is not None and warp_ok:
        warped = cv2.remap(
            src_remap, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        axes[2].imshow(np.dstack((f, norm(warped), np.zeros_like(f))))
        axes[2].set_title(
            f"Fixed (R) vs RoMaV2 warp (G) [{label} input]\nNCC={ncc_warp:.4f}", fontsize=10
        )
    else:
        axes[2].imshow(np.dstack((f, m_affine, np.zeros_like(f))))
        axes[2].set_title("RoMaV2 FAILED\n(affine shown)", fontsize=10)
    axes[2].axis('off')

    if map_x is not None and warp_ok:
        h_img, w_img = map_x.shape
        id_x     = np.arange(w_img, dtype=np.float32)[None, :]
        id_y     = np.arange(h_img, dtype=np.float32)[:, None]
        disp_mag = np.sqrt((map_x - id_x)**2 + (map_y - id_y)**2)
        disp_display = cv2.resize(disp_mag, (512, 512), interpolation=cv2.INTER_AREA)
        im = axes[3].imshow(disp_display, cmap='hot', vmin=0,
                            vmax=np.percentile(disp_mag, 99))
        axes[3].set_title("Displacement magnitude (px)\n(residual after affine)", fontsize=10)
        plt.colorbar(im, ax=axes[3], fraction=0.03, pad=0.02)
    else:
        axes[3].axis('off')
        axes[3].set_title("No warp field", fontsize=10)
    axes[3].axis('off')

    status = "SUCCESS" if (akaze_ok or warp_ok) else "FAILED"
    if abs(ncc_raw) > 1e-9:
        total_improv = (ncc_raw - (ncc_warp if warp_ok else ncc_affine)) / abs(ncc_raw) * 100
    else:
        total_improv = 0.0
    fig.suptitle(
        f"{slice_id}  "
        f"NCC: raw={ncc_raw:.4f} → affine={ncc_affine:.4f} → warp={ncc_warp:.4f}  "
        f"total Δ={total_improv:+.1f}%  [{status}]",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{slice_id}_registration.png"),
        dpi=100, bbox_inches='tight',
    )
    plt.close(fig)
    logger.info(f"[{slice_id}] Registration plot saved.")


def save_deformation_quiver_plot(map_x, map_y, slice_id, step=150):
    """
    Renders a vector field of the dense displacement map.
    Subsampled to prevent memory exhaustion on gigapixel mIF images.
    """
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    h, w = map_x.shape
    Y, X = np.mgrid[0:h, 0:w]
    
    # Subsample the grids and maps
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    map_x_sub = map_x[::step, ::step]
    map_y_sub = map_y[::step, ::step]
    
    # Calculate relative displacement
    dX = map_x_sub - X_sub
    dY = map_y_sub - Y_sub
    
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"{slice_id} Deformation Vector Field\n(Subsampled 1:{step} px)")
    
    plt.quiver(
        X_sub, Y_sub, dX, dY, 
        angles='xy', scale_units='xy', scale=1, 
        color='red', alpha=0.6, width=0.002
    )
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"{slice_id}_vector_field.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[{slice_id}] Vector field plot saved.")


def generate_qc_montage(vol, output_folder, slice_ids=None,
                        channel_idx=6, channel_name="CK",
                        title_suffix="AKAZE_RoMaV2_Linear"):
    n_slices = vol.shape[0]
    if n_slices < 2:
        return
    logger.info(f"Generating QC montage [{title_suffix}]...")
    all_pairs = [(i, i + 1) for i in range(n_slices - 1)]
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
    core = os.path.basename(output_folder)
    fig.suptitle(f"Registration QC {title_suffix}: {core}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f"{core}_QC_Montage_{channel_name}_{title_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"AKAZE → RoMaV2 (Linear) Registration — {TARGET_CORE}")
    logger.info(
        f"AKAZE threshold={AKAZE_THRESHOLD} | Max keypoints={AKAZE_MAX_KEYPOINTS} | "
        f"RoMaV2 {ROMAV2_H}×{ROMAV2_W} | "
        f"Warp cap={WARP_MAX_DISPLACEMENT_PX}px | "
        f"Confidence thresh={WARP_CONFIDENCE_THRESH}"
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
        file_list      = [f for i, f in enumerate(file_list) if i in allowed_positions]
        n_slices       = len(file_list)
        excluded       = original_count - n_slices
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
            os.path.join(OUTPUT_FOLDER,
                         f"{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif"),
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
            logger.warning(
                f"Shape mismatch in {os.path.basename(file_list[idx])} — conforming."
            )
            arr = conform_slice(arr, target_h, target_w)
        return arr

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    affine_vol              = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    affine_vol[center_idx]  = center_raw
    del center_raw
    logger.info(f"Anchor: slice index {center_idx} (ID {slice_ids[center_idx]})")

    # Save identity deformation for the anchor slice so warp_cellpose_masks.py
    # never skips it — applying a no-op transform is the correct behaviour.
    center_sid = f"Z{center_idx:03d}_ID{slice_ids[center_idx]:03d}"
    try:
        save_deformation_maps(
            slice_id = center_sid,
            M_affine = np.eye(2, 3, dtype=np.float64),
            map_x    = None,   # triggers identity grid inside save_deformation_maps
            map_y    = None,
            akaze_ok = False,
            warp_ok  = False,
            orig_h   = target_h,
            orig_w   = target_w,
        )
        logger.info(f"Anchor [{center_sid}] identity deformation map saved.")
    except Exception as exc:
        logger.warning(f"Anchor [{center_sid}] Failed to save identity map: {exc}")

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
                aligned_np, affine_np, runtime, stats, success, M_final, map_x, map_y = \
                    register_slice(fixed_np, moving_np, slice_id=sid)
            except Exception as exc:
                logger.error(f"[{sid}] register_slice crashed: {exc} — raw fallback.")
                aligned_np = moving_np.copy()
                affine_np  = moving_np.copy()
                runtime    = 0.0
                stats      = dict(
                    akaze_ok=False, warp_ok=False,
                    n_matches=0, n_inliers=0,
                    rotation_deg=0, tx=0, ty=0, scale_pct=0, shear_pct=0,
                    n_confident=0, coverage_pct=0.0, mean_confidence=0.0,
                    ncc_raw=0, ncc_affine=0, ncc_warp=0,
                    ncc_affine_improv=0, ncc_warp_improv=0, ncc_total_improv=0,
                )
                success = False
                # Save identity deformation so downstream code can still load
                try:
                    save_deformation_maps(
                        slice_id=sid,
                        M_affine=np.eye(2, 3, dtype=np.float64),
                        map_x=None, map_y=None,
                        akaze_ok=False, warp_ok=False,
                        orig_h=target_h, orig_w=target_w,
                    )
                except Exception as save_exc:
                    logger.warning(f"[{sid}] Failed to save identity map: {save_exc}")


            aligned_vol[i] = aligned_np
            affine_vol[i]  = affine_np
            del moving_np

            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK_RAW"
            if not success:
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str}")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | "
                f"AKAZE: {stats['n_inliers']} inliers | "
                f"NCC: raw={stats['ncc_raw']:.4f} "
                f"affine={stats['ncc_affine']:.4f} ({stats['ncc_affine_improv']:+.1f}%) "
                f"warp={stats['ncc_warp']:.4f} ({stats['ncc_warp_improv']:+.1f}%) | "
                f"Conf: {stats['coverage_pct']:.1f}% cells | "
                f"t: {runtime:.2f}s | {status_str}"
            )

            registration_stats.append({
                "Direction":          direction,
                "Slice_Z":            i,
                "Slice_ID":           real_id,
                "AKAZE_OK":           stats["akaze_ok"],
                "Warp_OK":            stats["warp_ok"],
                "N_Matches":          stats["n_matches"],
                "N_Inliers":          stats["n_inliers"],
                "Rotation_Deg":       stats["rotation_deg"],
                "Shift_X_px":         stats["tx"],
                "Shift_Y_px":         stats["ty"],
                "Scale_Pct":          stats["scale_pct"],
                "Shear_Pct":          stats["shear_pct"],
                "N_Confident":        stats["n_confident"],
                "Coverage_Pct":       stats["coverage_pct"],
                "Mean_Confidence":    stats["mean_confidence"],
                "NCC_Raw":            stats["ncc_raw"],
                "NCC_Affine":         stats["ncc_affine"],
                "NCC_Warp":           stats["ncc_warp"],
                "NCC_Affine_Improv":  stats["ncc_affine_improv"],
                "NCC_Warp_Improv":    stats["ncc_warp_improv"],
                "NCC_Total_Improv":   stats["ncc_total_improv"],
                "Success":            success,
                "Status":             status_str,
                "Runtime_s":          round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    df.to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_RoMaV2_Linear.csv"),
        index=False,
    )
    n_ok = int((df["Status"] == "SUCCESS").sum())
    n_fb = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(f"Complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fb}")

    # Write final volume
    out_tiff = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif")
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

    # QC montages
    try:
        generate_qc_montage(affine_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="AKAZE_Affine")
    except Exception as exc:
        logger.error(f"Affine montage failed: {exc}")
    del affine_vol

    try:
        generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="AKAZE_RoMaV2_Linear")
    except Exception as exc:
        logger.error(f"Final montage failed: {exc}")
    del aligned_vol

    logger.info("Done.")


if __name__ == "__main__":
    main()