"""
Feature registration as the FIRST stage in the pipeline.
Reads raw .ome.tif slices at FULL resolution.

Detector/Matcher: SuperPoint + LightGlue via kornia (CUDA GPU)
Pyramid:          3-level coarse-to-fine (0.25x → 0.50x → 1.00x)

Key differences vs AKAZE version:
  - SuperPoint replaces AKAZE for keypoint detection and description
  - LightGlue replaces BFMatcher + Lowe ratio — it reasons about global geometric
    context via attention, dramatically reducing false matches
  - RANSAC is still applied on top of LightGlue matches as a final geometric filter
  - ANMS is removed — LightGlue's attention mechanism handles spatial distribution
  - Per-level inlier ratio gate, sanity gate, and chain-safe fallback retained
  - Log transform preprocessing retained (no CLAHE)

Requirements:
    pip install git+https://github.com/cvg/LightGlue.git
    CUDA GPU required for practical runtime
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
import torch
import matplotlib
import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Feature registration — SuperPoint+LightGlue pyramidal.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Feature_pyramidal_superpoint_lightblue")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# --- PYRAMID ---
PYRAMID_SCALES = [0.25, 0.50, 1.0]

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- SUPERPOINT ---
# max_num_keypoints: upper budget per image per level; scaled per level below
# detection_threshold: lower = more keypoints on weak signal
SUPERPOINT_MAX_IMAGE_SIZE = 1024
SUPERPOINT_MAX_KEYPOINTS     = 2048
SUPERPOINT_DETECTION_THRESH  = 0.005   # default 0.005; lower if starvation persists

# --- LIGHTGLLUE ---
# depth_confidence / width_confidence: early-exit thresholds.
# -1 disables early exit (full depth, more accurate, slower).
# 0.95 / 0.99 are good starting points for speed/accuracy balance.
LIGHTGLUE_DEPTH_CONFIDENCE  = 0.95
LIGHTGLUE_WIDTH_CONFIDENCE  = 0.99

# --- CHANNEL ---
CK_CHANNEL_IDX = 6

# --- MATCHING — per pyramid level [L0=0.25x, L1=0.50x, L2=1.00x] ---
# Min matches: minimum LightGlue matches before RANSAC is attempted
# Min inlier ratio: relaxed thresholds calibrated to IF imaging characteristics
MIN_MATCHES_BY_LEVEL      = [6,    8,    10   ]
MIN_INLIER_RATIO_BY_LEVEL = [0.04, 0.04, 0.02]

# --- RANSAC ---
# Applied on top of LightGlue matches as a final geometric consistency filter
RANSAC_CONFIDENCE     = 0.995
RANSAC_MAX_ITERS      = 5000
RANSAC_THRESH_FULLRES = 5.0

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.05

# --- SANITY GATE ---
MAX_ROTATION_DEG   = 15.0
MAX_TRANSLATION_PX = 300.0

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model initialisation — loaded once, reused across all slices and levels
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    """
    Initialises SuperPoint and LightGlue on DEVICE using the official cvg/LightGlue API.
    Called once at startup; returned objects are reused across all slices.
    """
    logger.info("Loading SuperPoint and LightGlue models...")
    sp = SuperPoint(max_num_keypoints=SUPERPOINT_MAX_KEYPOINTS).eval().to(DEVICE)
    lg = LightGlue(
        features="superpoint",
        depth_confidence=LIGHTGLUE_DEPTH_CONFIDENCE,
        width_confidence=LIGHTGLUE_WIDTH_CONFIDENCE,
    ).eval().to(DEVICE)
    logger.info(f"Models loaded on {DEVICE}.")
    return sp, lg


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    c, h, w = arr.shape
    out = np.zeros((c, target_h, target_w), dtype=arr.dtype)

    if h >= target_h:
        src_y0 = (h - target_h) // 2;  src_y1 = src_y0 + target_h
        dst_y0, dst_y1 = 0, target_h
    else:
        src_y0, src_y1 = 0, h
        dst_y0 = (target_h - h) // 2;  dst_y1 = dst_y0 + h

    if w >= target_w:
        src_x0 = (w - target_w) // 2;  src_x1 = src_x0 + target_w
        dst_x0, dst_x1 = 0, target_w
    else:
        src_x0, src_x1 = 0, w
        dst_x0 = (target_w - w) // 2;  dst_x1 = dst_x0 + w

    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = arr[:, src_y0:src_y1, src_x0:src_x1]
    return out


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_tensor(img_uint8: np.ndarray) -> tuple:
    """
    Converts a (H, W) uint8 image to a (1, 1, H, W) float32 tensor in [0, 1]
    on DEVICE, capped at SUPERPOINT_MAX_IMAGE_SIZE on the longest side.
    Returns (tensor, scale_factor) so keypoints can be mapped back to original coords.
    """
    h, w   = img_uint8.shape
    scale  = min(1.0, SUPERPOINT_MAX_IMAGE_SIZE / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_uint8 = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img_uint8).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE), scale


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — log transform only
# ─────────────────────────────────────────────────────────────────────────────

def prepare_for_features_with_diagnostics(img_arr: np.ndarray) -> tuple:
    """
    Normalizes dynamic range using a log transform to boost low-intensity signals.
    Returns (baseline_norm_uint8, log_norm_uint8).
    Linear image is used for MSE calculation; log image is used for detection.
    """
    p_low_lin, p_high_lin = np.percentile(img_arr[::4, ::4], (1, 99.9))
    clipped_lin = np.clip(img_arr, p_low_lin, p_high_lin)
    norm_linear = cv2.normalize(clipped_lin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_float   = img_arr.astype(np.float32)
    log_img     = np.log1p(img_float)
    p_low_log, p_high_log = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    clipped_log = np.clip(log_img, p_low_log, p_high_log)
    norm_log    = cv2.normalize(clipped_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return norm_linear, norm_log


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def save_level_diagnostics(
    output_dir, slice_id, level_label, level_idx,
    img_fixed, img_moving, pts_fixed, pts_moving, inlier_mask,
):
    """
    Saves per-pyramid-level diagnostic plots.
    pts_fixed / pts_moving are (N, 2) numpy arrays of matched point coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot a: keypoint locations as scatter on side-by-side images
    h1, w1 = img_fixed.shape[:2]
    h2, w2 = img_moving.shape[:2]
    max_h  = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1]      = img_fixed
    canvas[:h2, w1:w1+w2] = img_moving
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for pt in pts_fixed:
        cv2.circle(canvas_bgr, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    for pt in pts_moving:
        cv2.circle(canvas_bgr, (int(pt[0]) + w1, int(pt[1])), 2, (255, 0, 0), -1)

    cv2.imwrite(
        os.path.join(output_dir, f"{slice_id}_{level_idx+1:02d}a_Keypoints_{level_label}.png"),
        canvas_bgr,
    )

    # Plot b: inlier match lines
    if inlier_mask is not None and inlier_mask.any():
        match_canvas = canvas_bgr.copy()
        inlier_fixed  = pts_fixed[inlier_mask]
        inlier_moving = pts_moving[inlier_mask]
        for pf, pm in zip(inlier_fixed, inlier_moving):
            pt1 = (int(pf[0]),        int(pf[1]))
            pt2 = (int(pm[0]) + w1,   int(pm[1]))
            cv2.line(match_canvas, pt1, pt2, (0, 0, 255), 1)
        cv2.imwrite(
            os.path.join(output_dir,
                f"{slice_id}_{level_idx+1:02d}b_Inliers_{inlier_mask.sum()}_{level_label}.png"),
            match_canvas,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Transform constraints
# ─────────────────────────────────────────────────────────────────────────────

def constrain_affine(M_partial: np.ndarray, max_scale_deviation: float = 0.0) -> np.ndarray:
    if M_partial is None:
        return None
    M_out    = M_partial.copy()
    U, S, Vt = np.linalg.svd(M_out[0:2, 0:2])
    S_clamped       = np.clip(S, 1.0 - max_scale_deviation, 1.0 + max_scale_deviation)
    M_out[0:2, 0:2] = U @ np.diag(S_clamped) @ Vt
    return M_out


def constrain_full_affine(M: np.ndarray, max_scale_deviation: float = 0.08, max_shear: float = 0.05) -> np.ndarray:
    if M is None:
        return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S_clamped = np.clip(S, 1.0 - max_scale_deviation, 1.0 + max_scale_deviation)
    max_ratio = 1.0 + max_shear
    if S_clamped[1] > 1e-6 and S_clamped[0] / S_clamped[1] > max_ratio:
        S_clamped[0] = S_clamped[1] * max_ratio
    M_out[:2, :2] = U @ np.diag(S_clamped) @ Vt
    return M_out


def compose_transforms(M_residual: np.ndarray, M_accum: np.ndarray) -> np.ndarray:
    R1, t1       = M_residual[:, :2], M_residual[:, 2]
    R2, t2       = M_accum[:, :2],    M_accum[:, 2]
    M_out        = np.zeros((2, 3), dtype=np.float64)
    M_out[:, :2] = R1 @ R2
    M_out[:, 2]  = R1 @ t2 + t1
    return M_out


def transform_is_sane(M: np.ndarray) -> bool:
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx, ty   = abs(M[0, 2]), abs(M[1, 2])
    return rot_deg <= MAX_ROTATION_DEG and tx <= MAX_TRANSLATION_PX and ty <= MAX_TRANSLATION_PX


# ─────────────────────────────────────────────────────────────────────────────
# Per-level detection and estimation
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_estimate_at_level(
    fixed_8bit:  np.ndarray,
    moving_8bit: np.ndarray,
    level_scale: float,
    level_idx:   int,
    sp_model:    SuperPoint,
    lg_model:    LightGlue,
):
    """
    Runs SuperPoint + LightGlue + RANSAC on one pyramid level.

    SuperPoint detects keypoints and computes 256-dim descriptors.
    LightGlue matches them using attention over global geometry — replacing
    BFMatcher + Lowe ratio entirely.
    RANSAC provides a final geometric consistency filter.

    Returns (M_2x3_fullres | None, n_matches, inlier_ratio, diagnostics | None).
    M is always in full-resolution coordinate space.
    """
    name          = f"SP+LG_L{level_idx}_s{level_scale:.2f}"
    ransac_thresh = RANSAC_THRESH_FULLRES * level_scale
    min_m         = MIN_MATCHES_BY_LEVEL[level_idx]
    min_r         = MIN_INLIER_RATIO_BY_LEVEL[level_idx]

    t_fixed,  sp_scale_fixed  = to_tensor(fixed_8bit)
    t_moving, sp_scale_moving = to_tensor(moving_8bit)

    with torch.no_grad():
        feats_fixed  = sp_model.extract(t_fixed,  resize=None)
        feats_moving = sp_model.extract(t_moving, resize=None)

    n_kp_fixed  = feats_fixed["keypoints"].shape[1]
    n_kp_moving = feats_moving["keypoints"].shape[1]

    if n_kp_fixed < 4 or n_kp_moving < 4:
        logger.warning(f"[{name}] Feature starvation (kp_fixed={n_kp_fixed}, kp_moving={n_kp_moving}).")
        return None, 0, 0.0, None

    with torch.no_grad():
        matches_out = lg_model({"image0": feats_fixed, "image1": feats_moving})

    feats_fixed_u, feats_moving_u, matches_out_u = [
        rbd(x) for x in [feats_fixed, feats_moving, matches_out]
    ]

    match_indices = matches_out_u["matches"]
    # Scale keypoints back from SuperPoint resolution to level (pyramid) resolution
    kp0 = feats_fixed_u["keypoints"].cpu().numpy()  / sp_scale_fixed
    kp1 = feats_moving_u["keypoints"].cpu().numpy() / sp_scale_moving

    n_matches = match_indices.shape[0]
    if n_matches < min_m:
        logger.warning(f"[{name}] Insufficient LightGlue matches ({n_matches} < {min_m}).")
        return None, n_matches, 0.0, None

    idx0    = match_indices[:, 0].cpu().numpy()
    idx1    = match_indices[:, 1].cpu().numpy()
    src_pts = kp0[idx0].reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kp1[idx1].reshape(-1, 1, 2).astype(np.float32)

    # RANSAC — geometry model per level
    if level_idx == 0:
        M_partial, mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
        M_constrained = constrain_affine(M_partial, max_scale_deviation=0.0)

    elif level_idx == 1:
        M_partial, mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
        M_constrained = constrain_affine(M_partial, max_scale_deviation=0.05)

    else:
        M_partial, mask = cv2.estimateAffine2D(
            dst_pts, src_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
        M_constrained = constrain_full_affine(M_partial, max_scale_deviation=0.08, max_shear=0.05)

    if M_partial is None or mask is None or M_constrained is None:
        logger.warning(f"[{name}] RANSAC diverged.")
        return None, n_matches, 0.0, None

    inlier_ratio = float(mask.sum()) / len(mask)

    if inlier_ratio < min_r:
        logger.warning(f"[{name}] Inlier ratio too low ({inlier_ratio:.3f} < {min_r}) — rejecting level.")
        return None, n_matches, inlier_ratio, None

    M_fullres        = M_constrained.copy()
    M_fullres[:, 2] /= level_scale

    if not transform_is_sane(M_fullres):
        U, _, Vt = np.linalg.svd(M_fullres[:2, :2])
        R        = U @ Vt
        rot      = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(
            f"[{name}] Transform failed sanity check "
            f"(rot={rot:.1f}° tx={M_fullres[0,2]:.1f}px ty={M_fullres[1,2]:.1f}px) — rejecting."
        )
        return None, n_matches, inlier_ratio, None

    inlier_mask = mask.ravel().astype(bool)
    diag = {
        "pts_fixed":   kp0[idx0],
        "pts_moving":  kp1[idx1],
        "inlier_mask": inlier_mask,
    }

    logger.info(
        f"[{name}] lg_matches={n_matches} ransac_inliers={inlier_ratio:.2f} "
        f"tx={M_fullres[0,2]:.1f}px ty={M_fullres[1,2]:.1f}px"
    )

    return M_fullres, n_matches, inlier_ratio, diag


# ─────────────────────────────────────────────────────────────────────────────
# Pyramidal registration
# ─────────────────────────────────────────────────────────────────────────────

def register_slice_pyramidal(fixed_np, moving_np, sp_model, lg_model, slice_id=None, diag_dir=None):
    """
    3-level coarse-to-fine pyramidal registration using SuperPoint + LightGlue.
    Returns (aligned_np, mse, elapsed, stats, success, M_final).
    """
    start = time.time()

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_norm,  fixed_8bit_full  = prepare_for_features_with_diagnostics(fixed_ck)
    moving_norm, moving_8bit_full = prepare_for_features_with_diagnostics(moving_ck)

    h_full, w_full = fixed_8bit_full.shape

    M_accum = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]], dtype=np.float64)

    level_stats         = []
    any_level_succeeded = False

    for lvl_idx, scale in enumerate(PYRAMID_SCALES):
        level_label = f"L{lvl_idx}_s{scale:.2f}"

        fixed_lvl  = resize_image(fixed_8bit_full,  scale)
        moving_lvl = resize_image(moving_8bit_full, scale)
        lh, lw     = fixed_lvl.shape

        M_lvl        = M_accum.copy()
        M_lvl[:, 2] *= scale

        moving_warped = cv2.warpAffine(
            moving_lvl, M_lvl, (lw, lh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        M_residual_fullres, n_matches, inlier_ratio, diag = detect_and_estimate_at_level(
            fixed_lvl, moving_warped, scale, lvl_idx, sp_model, lg_model,
        )

        level_stats.append((n_matches, inlier_ratio))

        if diag is not None and slice_id is not None and diag_dir is not None:
            save_level_diagnostics(
                diag_dir, slice_id, level_label, lvl_idx,
                fixed_lvl, moving_warped,
                diag["pts_fixed"], diag["pts_moving"], diag["inlier_mask"],
            )

        if M_residual_fullres is not None:
            M_candidate = compose_transforms(M_residual_fullres, M_accum)
            if transform_is_sane(M_candidate):
                M_accum = M_candidate
                any_level_succeeded = True
            else:
                U, _, Vt = np.linalg.svd(M_candidate[:2, :2])
                R        = U @ Vt
                rot      = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                logger.warning(
                    f"[{slice_id}] Level {lvl_idx}: composed transform rejected "
                    f"(rot={rot:.1f}° tx={M_candidate[0,2]:.1f}px ty={M_candidate[1,2]:.1f}px) — carrying previous."
                )
        else:
            logger.warning(f"[{slice_id}] Level {lvl_idx} ({level_label}): no residual — carrying previous transform.")

    if not any_level_succeeded:
        logger.warning(f"[{slice_id}] All levels failed — falling back to identity.")
        M_accum = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)

    aligned_channels = []
    for c in range(fixed_np.shape[0]):
        ch     = moving_np[c].astype(np.float32)
        warped = cv2.warpAffine(
            ch, M_accum, (w_full, h_full),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_channels.append(warped)

    aligned_np = np.stack(aligned_channels, axis=0).astype(np.uint16)

    _, warped_8bit = prepare_for_features_with_diagnostics(
        aligned_np[CK_CHANNEL_IDX].astype(np.float32)
    )
    mse = float(np.mean((fixed_norm.astype(np.float32) - warped_8bit.astype(np.float32)) ** 2))

    U, _, Vt = np.linalg.svd(M_accum[:2, :2])
    R   = U @ Vt
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx  = float(M_accum[0, 2])
    ty  = float(M_accum[1, 2])

    total_matches     = sum(n for n, _ in level_stats)
    valid_ratios      = [r for _, r in level_stats if r > 0.0]
    mean_inlier_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    elapsed           = time.time() - start

    stats = dict(
        detector            = "SuperPoint_LightGlue_Pyramid_3L",
        n_good              = total_matches,
        inlier_ratio        = round(mean_inlier_ratio, 3),
        rotation_deg        = round(rot, 3),
        tx                  = round(tx, 3),
        ty                  = round(ty, 3),
        level_matches       = [n for n, _ in level_stats],
        level_inlier_ratios = [round(r, 3) for _, r in level_stats],
        success             = any_level_succeeded,
    )

    return aligned_np, mse, elapsed, stats, any_level_succeeded, M_accum


# ─────────────────────────────────────────────────────────────────────────────
# QC montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_montage(vol, output_folder, channel_idx=6, channel_name="CK"):
    logger.info(f"Generating QC montage for {channel_name} channel.")
    n_slices = vol.shape[0]
    if n_slices < 2:
        return

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
            p99 = np.percentile(x, 99.5)
            return np.clip(x / (p99 if p99 > 0 else 1), 0, 1)
        overlay = np.dstack((norm(s1), norm(s2), np.zeros_like(s1)))
        axes_flat[idx].imshow(overlay)
        axes_flat[idx].set_title(f"Z{z1} to Z{z2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Registration QC SuperPoint+LightGlue Pyramid (Stage 1): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_SP_LG_Pyramid.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Pyramidal Feature Registration (Stage 1 / Full Resolution) — {TARGET_CORE}")
    logger.info(f"Detector/Matcher: SuperPoint + LightGlue | Device: {DEVICE} | Pyramid: {PYRAMID_SCALES}")

    # Load models once — reused across all slices
    sp_model, lg_model = build_models()

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    if n_slices < 2:
        logger.warning("Insufficient slice depth. Writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol_in[np.newaxis], photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='zlib')
        sys.exit(0)

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)

    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical full-res shape: C={c}, H={target_h}, W={target_w}")

    logger.info(f"Loading and conforming {n_slices} slices at full resolution.")
    raw_slices = []
    for f in file_list:
        arr = tifffile.imread(f)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            logger.warning(f"Shape mismatch in {os.path.basename(f)} — conforming.")
            arr = conform_slice(arr, target_h, target_w)
        raw_slices.append(arr)

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    aligned_vol[center_idx] = raw_slices[center_idx]
    anchor_id               = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} spatial pass.")
        diag_out_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_Pyramid_SP_LG")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]

            aligned_np, mse, runtime, stats, success, M_final = register_slice_pyramidal(
                fixed_np, moving_np, sp_model, lg_model,
                slice_id = f"Z{i:03d}_ID{real_id:03d}",
                diag_dir = diag_out_dir,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = raw_slices[i]
                status_str     = "SANITY_FAIL_RAW" if success else "IDENTITY_FALLBACK_RAW"
                logger.warning(
                    f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice to protect chain."
                )

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"TotalMatch: {stats['n_good']} | MeanInliers: {stats['inlier_ratio']:.2f} | "
                f"MSE: {mse:.2f} | Rot: {stats['rotation_deg']:.2f} | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
                f"LvlMatches: {stats['level_matches']} | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":          direction,
                "Slice_Z":            i,
                "Slice_ID":           real_id,
                "Detector":           stats["detector"],
                "N_Matches_Total":    stats["n_good"],
                "Mean_Inlier_Ratio":  stats["inlier_ratio"],
                "L0_Matches":         stats["level_matches"][0],
                "L1_Matches":         stats["level_matches"][1],
                "L2_Matches":         stats["level_matches"][2],
                "L0_Inlier_Ratio":    stats["level_inlier_ratios"][0],
                "L1_Inlier_Ratio":    stats["level_inlier_ratios"][1],
                "L2_Inlier_Ratio":    stats["level_inlier_ratios"][2],
                "Success":            success,
                "Status":             status_str,
                "MSE_After":          round(mse, 4),
                "Rotation_Deg":       stats["rotation_deg"],
                "Shift_X_px":         stats["tx"],
                "Shift_Y_px":         stats["ty"],
                "Runtime_s":          round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = [
        "Direction", "Slice_Z", "Slice_ID", "Detector",
        "N_Matches_Total", "Mean_Inlier_Ratio",
        "L0_Matches", "L1_Matches", "L2_Matches",
        "L0_Inlier_Ratio", "L1_Inlier_Ratio", "L2_Inlier_Ratio",
        "Success", "Status", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "MSE_After", "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_SP_LG_pyramid.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_sanity   = int((df["Status"] == "SANITY_FAIL_RAW").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. SUCCESS: {n_ok} | SANITY_FAIL_RAW: {n_sanity} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Committing registered tensor to disk at {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    logger.info("Done.")


if __name__ == "__main__":
    main()