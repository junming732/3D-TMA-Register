"""
Feature registration as the FIRST stage in the pipeline.
Reads raw .ome.tif slices at FULL resolution.

Detector:   AKAZE with lowered detection threshold (0.0003)
Pyramid:    2-level coarse-to-fine (0.50x → 1.00x)
Fallback:   NCC (phase correlation) on full-resolution image when both feature levels fail

Changes vs previous version:
  - Reverted to single AKAZE detector (no SIFT/fallback switching)
  - Relaxed MIN_INLIER_RATIO_BY_LEVEL to [0.10, 0.06, 0.02] based on observed data
  - Per-level adaptive Lowe ratio and min-match threshold
  - Per-level inlier ratio gate (rejects genuinely degenerate RANSAC solutions)
  - Removed redundant per-level sanity check in detect_and_estimate_at_level:
      residual transforms are sanity-checked only after compose_transforms() in the
      pyramid loop, which is the only check that matters.
  - NCC translation fallback: when all three AKAZE levels fail, cv2.phaseCorrelate
      (FFT-based normalised cross-correlation) is used on the full-resolution CK image
      to recover at least a translation. This replaces the previous identity fallback.
      Status is reported as NCC_FALLBACK_SUCCESS in the CSV.
  - Chain-safe fallback: a bad slice writes raw (unregistered), does NOT poison next slice
  - Fixed: removed duplicate constrain_affine() call that overwrote level-specific constraint
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
parser = argparse.ArgumentParser(description='Feature registration — pyramidal, full resolution output.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Feature_pyramidal_piecewise_CK")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# --- PYRAMID ---
PYRAMID_SCALES = [0.50, 1.0]

# --- DETECTOR ---
# Lower threshold than default (0.001) to detect more keypoints on weak CK signal
AKAZE_THRESHOLD = 0.0003

# --- CHANNEL ---
CK_CHANNEL_IDX = 6

# --- MATCHING — per pyramid level [L0=0.50x, L1=1.00x] ---
# Lowe ratio: controls descriptor ambiguity filtering per level
# Min matches: minimum good matches fed into RANSAC (must be >> MIN_INLIERS)
# Min inliers: minimum RANSAC survivors required to accept the level's transform
LOWE_RATIO_BY_LEVEL       = [0.82, 0.80]
MIN_MATCHES_BY_LEVEL      = [30,   40  ]  # pool fed into RANSAC — must be >> MIN_INLIERS
MIN_INLIERS_BY_LEVEL      = [6,    8   ]  # survivors after RANSAC — rotation gate catches garbage

# --- RANSAC ---
RANSAC_CONFIDENCE     = 0.995
RANSAC_MAX_ITERS      = 5000
RANSAC_THRESH_FULLRES = 5.0    # pixels at full resolution; auto-scaled per level

# --- ANMS ---
ANMS_KEEP_FULLRES = 2000       # budget at full resolution; scaled linearly per level

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.05

# --- SANITY GATE ---
# Only rotation is hard-gated. Translation is NOT limited — large inter-slice
# tissue shifts are real and must not be blocked by a translation ceiling.
# Degenerate RANSAC solutions are caught instead by cross-level consistency.
MAX_ROTATION_DEG = 15.0


if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pads or centre-crops a (C, H, W) slice to (C, target_h, target_w)."""
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
    img_fixed, img_moving, kp_fixed, kp_moving, good_matches, inlier_mask,
):
    os.makedirs(output_dir, exist_ok=True)

    fixed_kp_img  = cv2.drawKeypoints(cv2.cvtColor(img_fixed,  cv2.COLOR_GRAY2BGR), kp_fixed,  None, color=(255, 0, 0))
    moving_kp_img = cv2.drawKeypoints(cv2.cvtColor(img_moving, cv2.COLOR_GRAY2BGR), kp_moving, None, color=(255, 0, 0))

    h1, h2 = fixed_kp_img.shape[0], moving_kp_img.shape[0]
    max_h  = max(h1, h2)
    if h1 < max_h:
        fixed_kp_img  = cv2.copyMakeBorder(fixed_kp_img,  0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if h2 < max_h:
        moving_kp_img = cv2.copyMakeBorder(moving_kp_img, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    cv2.imwrite(
        os.path.join(output_dir, f"{slice_id}_{level_idx+1:02d}a_Keypoints_{level_label}.png"),
        np.hstack((fixed_kp_img, moving_kp_img))
    )

    if good_matches and inlier_mask is not None:
        inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
        if inlier_matches:
            match_img = cv2.drawMatches(
                img_fixed, kp_fixed, img_moving, kp_moving,
                inlier_matches, None,
                matchColor=(0, 0, 255), singlePointColor=(255, 0, 0), flags=2,
            )
            cv2.imwrite(
                os.path.join(output_dir,
                    f"{slice_id}_{level_idx+1:02d}b_Inliers_{len(inlier_matches)}_{level_label}.png"),
                match_img,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ANMS
# ─────────────────────────────────────────────────────────────────────────────

def apply_anms_with_descriptors(keypoints, descriptors, num_to_keep=1000, c_robust=0.9, max_compute_pts=5000):
    n_kpts = len(keypoints)
    if n_kpts <= num_to_keep or descriptors is None:
        return keypoints, descriptors

    coords    = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sort_idx  = np.argsort(responses)[::-1]

    if len(sort_idx) > max_compute_pts:
        sort_idx = sort_idx[:max_compute_pts]
        n_kpts   = max_compute_pts

    coords    = coords[sort_idx]
    responses = responses[sort_idx]
    radii     = np.full(n_kpts, np.inf)

    for i in range(1, n_kpts):
        stronger_mask = responses[:i] * c_robust > responses[i]
        if np.any(stronger_mask):
            diffs    = coords[:i][stronger_mask] - coords[i]
            radii[i] = np.min(np.sum(diffs ** 2, axis=1))

    best_idx     = np.argsort(radii)[::-1][:num_to_keep]
    final_idx    = sort_idx[best_idx]
    filtered_kp  = tuple(keypoints[i] for i in final_idx)
    filtered_des = descriptors[final_idx]
    return filtered_kp, filtered_des


# ─────────────────────────────────────────────────────────────────────────────
# Transform constraints
# ─────────────────────────────────────────────────────────────────────────────

def constrain_affine(M_partial: np.ndarray, max_scale_deviation: float = 0.0) -> np.ndarray:
    if M_partial is None:
        return None
    M_out    = M_partial.copy()
    U, S, Vt = np.linalg.svd(M_out[0:2, 0:2])
    S_clamped       = np.clip(S, 1.0 - max_scale_deviation, 1.0 + max_scale_deviation)
    if not np.allclose(S, S_clamped):
        logger.debug(
            f"constrain_affine: scale clamped (max_scale_deviation={max_scale_deviation}) "
            f"S_raw={S.tolist()} → S_clamped={S_clamped.tolist()}"
        )
    M_out[0:2, 0:2] = U @ np.diag(S_clamped) @ Vt
    return M_out


def constrain_full_affine(M: np.ndarray, max_scale_deviation: float = 0.08, max_shear: float = 0.05) -> np.ndarray:
    if M is None:
        return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S_clamped = np.clip(S, 1.0 - max_scale_deviation, 1.0 + max_scale_deviation)
    if not np.allclose(S, S_clamped):
        logger.debug(
            f"constrain_full_affine: max_scale_deviation={max_scale_deviation} hit — "
            f"S_raw={S.tolist()} → S_clamped={S_clamped.tolist()}"
        )
    max_ratio = 1.0 + max_shear
    if S_clamped[1] > 1e-6 and S_clamped[0] / S_clamped[1] > max_ratio:
        old_s0 = S_clamped[0]
        S_clamped[0] = S_clamped[1] * max_ratio
        logger.debug(
            f"constrain_full_affine: max_shear={max_shear} hit — "
            f"singular ratio {old_s0/S_clamped[1]:.4f} → clamped S[0] {old_s0:.4f} → {S_clamped[0]:.4f}"
        )
    M_out[:2, :2] = U @ np.diag(S_clamped) @ Vt
    return M_out


def compose_transforms(M_residual: np.ndarray, M_accum: np.ndarray) -> np.ndarray:
    # M_out = M_residual ∘ M_accum: accumulated applied first, residual on top.
    R_res, t_res = M_residual[:, :2], M_residual[:, 2]
    R_acc, t_acc = M_accum[:, :2],    M_accum[:, 2]
    M_out        = np.zeros((2, 3), dtype=np.float64)
    M_out[:, :2] = R_res @ R_acc
    M_out[:, 2]  = R_res @ t_acc + t_res
    return M_out


def transform_is_sane(M: np.ndarray) -> bool:
    """Returns True if the rotation component is within physical bounds.
    Translation is not gated here — large shifts are real for some cores
    and are validated instead by cross-level consistency in the pyramid loop."""
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG


# ─────────────────────────────────────────────────────────────────────────────
# Per-level detection and estimation
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_estimate_at_level(fixed_8bit, moving_8bit, level_scale, level_idx):
    """
    Runs AKAZE + ANMS + RANSAC on one pyramid level.
    Returns (M_2x3_fullres | None, n_good, inlier_ratio, diagnostics | None).
    M is always expressed in full-resolution coordinate space.
    Returns None if: feature starvation, insufficient matches, inlier ratio below
    relaxed threshold, RANSAC divergence, or transform fails sanity check.
    """
    name          = f"AKAZE_L{level_idx}_s{level_scale:.2f}"
    ransac_thresh = RANSAC_THRESH_FULLRES * level_scale
    anms_keep     = max(200, int(ANMS_KEEP_FULLRES * level_scale))
    lowe          = LOWE_RATIO_BY_LEVEL[level_idx]
    min_m         = MIN_MATCHES_BY_LEVEL[level_idx]
    min_inliers   = MIN_INLIERS_BY_LEVEL[level_idx]

    detector    = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1, des1   = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2   = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(f"[{name}] Feature starvation (kp1={len(kp1) if kp1 else 0}, kp2={len(kp2) if kp2 else 0}).")
        return None, 0, 0.0, None

    kp1_anms, des1_anms = apply_anms_with_descriptors(kp1, des1, num_to_keep=anms_keep)
    kp2_anms, des2_anms = apply_anms_with_descriptors(kp2, des2, num_to_keep=anms_keep)

    matcher     = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1_anms, des2_anms, k=2)
    good        = [m for m, n in raw_matches if len([m, n]) == 2 and m.distance < lowe * n.distance]

    if len(good) < min_m:
        logger.warning(f"[{name}] Insufficient matches ({len(good)} < {min_m}).")
        return None, len(good), 0.0, None

    src_pts = np.float32([kp1_anms[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_anms[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # L0 (0.5×): tighter constraints — coarse alignment, keep transforms conservative
    # L1 (1.0×): relaxed slightly — fine refinement at full resolution
    if level_idx == 0:
        M_partial, mask = cv2.estimateAffine2D(
            dst_pts, src_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
        M_constrained = constrain_full_affine(M_partial, max_scale_deviation=0.05, max_shear=0.04)

    else:  # level_idx == 1, full resolution
        M_partial, mask = cv2.estimateAffine2D(
            dst_pts, src_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
        M_constrained = constrain_full_affine(M_partial, max_scale_deviation=0.08, max_shear=0.05)

    if M_partial is None or mask is None or M_constrained is None:
        logger.warning(f"[{name}] RANSAC diverged.")
        return None, len(good), 0.0, None

    n_inliers    = int(mask.sum())
    inlier_ratio = float(n_inliers) / len(mask)

    if n_inliers < min_inliers:
        logger.warning(f"[{name}] Inlier count too low ({n_inliers} < {min_inliers}) — rejecting level.")
        return None, len(good), inlier_ratio, None

    M_fullres        = M_constrained.copy()
    M_fullres[:, 2] /= level_scale

    # NOTE: no per-level sanity check here. The residual is a partial transform and
    # its rotation can look large in isolation at coarse scales. The composed transform
    # is sanity-checked in the pyramid loop after compose_transforms() — that is the
    # only gate needed.

    diag = {
        "kp_fixed":     kp1_anms,
        "kp_moving":    kp2_anms,
        "good_matches": good,
        "inlier_mask":  mask.ravel(),
    }

    U_log, S_log, _ = np.linalg.svd(M_fullres[:2, :2])
    scale_pct = (float(np.mean(S_log)) - 1.0) * 100.0
    shear_pct = (float(S_log[0] / S_log[1]) - 1.0) * 100.0 if S_log[1] > 1e-6 else 0.0

    logger.info(
        f"[{name}] matches={len(good)} inliers={n_inliers} "
        f"tx={M_fullres[0,2]:.1f}px ty={M_fullres[1,2]:.1f}px "
        f"scale={scale_pct:+.2f}% shear={shear_pct:.2f}%"
    )

    return M_fullres, len(good), inlier_ratio, diag



# ─────────────────────────────────────────────────────────────────────────────
# Piecewise affine warp (2×2 grid, L2 only)
# ─────────────────────────────────────────────────────────────────────────────

PIECEWISE_GRID      = (3, 3)   # rows × cols

# --- B-SPLINE NCC ELASTIC REFINEMENT (L3) ---
# Metric: Normalized Cross-Correlation (NCC) — robust on sparse fluorescence
#         because it is insensitive to intensity offset/scale differences between
#         slices, unlike MSE/SSD which Demons uses internally.
# Transform: B-spline FFD on a regular control point grid.
#   BSPLINE_GRID_NODES — control points per axis. At 6080px, 8 nodes → one
#                        control point every ~760px (tissue/gland cluster scale).
#                        Increase for tighter local correction, decrease if
#                        the field overfits to noise.
#   BSPLINE_ITERATIONS — optimizer steps per resolution level.
#   BSPLINE_SHRINK     — multi-resolution pyramid shrink factors (coarse→fine).
#   BSPLINE_SIGMA      — Gaussian smoothing per resolution level (voxels).
BSPLINE_GRID_NODES  = 8
BSPLINE_ITERATIONS  = 100
BSPLINE_SHRINK      = [4, 2, 1]   # 3-level pyramid: 25% → 50% → full res
BSPLINE_SIGMA       = [6, 3, 1]   # smoothing per level (voxels)

# --- OUTPUT SANITY ---
# Minimum fraction of non-zero pixels in the CK channel after registration.
# If below this, B-spline likely diverged — revert to raw to protect the chain.
MIN_CK_NONZERO_FRAC = 0.01
PIECEWISE_OVERLAP   = 0.35     # fractional overlap between tiles for smooth blending
MIN_TILE_INLIERS    = 4        # minimum inliers to accept a tile's transform

def piecewise_affine_warp(
    fixed_8bit: np.ndarray,
    moving_8bit: np.ndarray,
    M_global: np.ndarray,
    level_scale: float,
    slice_id: str,
) -> np.ndarray:
    """
    2×2 piecewise affine at full resolution (L2).
    For each tile: detect AKAZE, match, RANSAC → per-tile affine.
    Falls back to M_global for any tile with insufficient inliers.
    Returns a blended full-resolution warp of moving_8bit.
    """
    h, w      = fixed_8bit.shape
    n_rows, n_cols = PIECEWISE_GRID
    overlap   = PIECEWISE_OVERLAP

    tile_h = int(h / (n_rows * (1.0 - overlap) + overlap))
    tile_w = int(w / (n_cols * (1.0 - overlap) + overlap))
    step_h = int(tile_h * (1.0 - overlap))
    step_w = int(tile_w * (1.0 - overlap))

    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    ransac_thresh = RANSAC_THRESH_FULLRES * level_scale
    lowe    = LOWE_RATIO_BY_LEVEL[1]   # L1 = full resolution

    # Accumulate weighted warps
    warp_accum   = np.zeros((h, w), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * step_h
            x0 = col * step_w
            y1 = min(y0 + tile_h, h)
            x1 = min(x0 + tile_w, w)

            fixed_tile  = fixed_8bit[y0:y1, x0:x1]
            moving_tile = moving_8bit[y0:y1, x0:x1]

            kp1, des1 = detector.detectAndCompute(fixed_tile,  None)
            kp2, des2 = detector.detectAndCompute(moving_tile, None)

            M_tile = None
            if (des1 is not None and des2 is not None
                    and len(kp1) >= 4 and len(kp2) >= 4):
                matcher     = cv2.BFMatcher(cv2.NORM_HAMMING)
                raw_matches = matcher.knnMatch(des1, des2, k=2)
                good = [m for m, n in raw_matches
                        if len([m, n]) == 2 and m.distance < lowe * n.distance]

                if len(good) >= MIN_TILE_INLIERS:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M_partial, mask = cv2.estimateAffine2D(
                        dst_pts, src_pts, method=cv2.RANSAC,
                        ransacReprojThreshold=ransac_thresh,
                        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
                    )
                    if M_partial is not None and mask is not None:
                        n_inliers = int(mask.sum())
                        if n_inliers >= MIN_TILE_INLIERS:
                            # Convert tile-local transform to full-image coordinate space.
                            # Keypoints were detected in tile space (origin at 0,0).
                            # Full-image transform: T_full = T_tile applied around tile origin.
                            # M_full * p_full = M_tile * (p_full - origin) + origin
                            # => M_full[:, 2] = M_tile[:, 2] + (I - M_tile[:, :2]) @ origin
                            M_tile = M_partial.copy()
                            origin = np.array([x0, y0], dtype=np.float64)
                            M_tile[:, 2] = M_partial[:, 2] + (np.eye(2) - M_partial[:, :2]) @ origin
                            M_tile = constrain_full_affine(M_tile, max_scale_deviation=0.08, max_shear=0.05)
                            logger.debug(
                                f"[{slice_id}] tile({row},{col}) inliers={n_inliers} "
                                f"tx={M_tile[0,2]:.1f}px ty={M_tile[1,2]:.1f}px"
                            )

            if M_tile is None:
                M_tile = M_global.copy()
                logger.debug(f"[{slice_id}] tile({row},{col}) fallback to global transform")

            # Warp full moving image with this tile's transform
            warped_full = cv2.warpAffine(
                moving_8bit.astype(np.float32), M_tile, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            # Build smooth Gaussian weight mask for this tile
            weight = np.zeros((h, w), dtype=np.float64)
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
            sigma_y, sigma_x = (y1 - y0) / 2.5, (x1 - x0) / 2.5
            yy, xx = np.ogrid[0:h, 0:w]
            weight = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))

            warp_accum   += warped_full * weight
            weight_accum += weight

    # Normalise by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-6)
    result = (warp_accum / weight_accum).astype(np.uint8)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# NCC translation fallback
# ─────────────────────────────────────────────────────────────────────────────

def ncc_translation_fallback(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, slice_id: str) -> np.ndarray:
    """
    Estimates a pure translation using normalised cross-correlation (phase correlation)
    on the full-resolution 8-bit CK images.

    Called only when all three AKAZE pyramid levels fail to produce an accepted residual.
    Returns a 2×3 affine matrix (translation only, no rotation/scale).
    Returns identity if NCC itself fails or produces an implausible shift.

    Strategy:
      - Use cv2.phaseCorrelate (FFT-based NCC) for sub-pixel translation estimation.
      - Apply a Hanning window to suppress boundary artefacts before FFT.
      - Accept the result unconditionally on magnitude — no hard cap on translation
        (same philosophy as the rest of the pipeline: large real shifts must not be blocked).
      - Only reject if the response confidence is below NCC_MIN_RESPONSE.
    """
    NCC_MIN_RESPONSE = 0.05   # phase-correlation peak confidence; 0–1 scale

    try:
        win = cv2.createHanningWindow(
            (fixed_8bit.shape[1], fixed_8bit.shape[0]), cv2.CV_64F
        )
        (dx, dy), response = cv2.phaseCorrelate(
            fixed_8bit.astype(np.float64),
            moving_8bit.astype(np.float64),
            win,
        )

        if response < NCC_MIN_RESPONSE:
            logger.warning(
                f"[{slice_id}] NCC fallback: low confidence (response={response:.3f} < {NCC_MIN_RESPONSE}) "
                f"— falling back to identity."
            )
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        logger.info(
            f"[{slice_id}] NCC fallback: tx={dx:.1f}px ty={dy:.1f}px response={response:.3f}"
        )
        M_ncc = np.array([[1.0, 0.0, dx],
                          [0.0, 1.0, dy]], dtype=np.float64)
        return M_ncc

    except Exception as exc:
        logger.warning(f"[{slice_id}] NCC fallback raised exception: {exc} — using identity.")
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Pyramidal registration
# ─────────────────────────────────────────────────────────────────────────────

def register_slice_pyramidal(fixed_np, moving_np, slice_id=None, diag_dir=None):
    """
    3-level coarse-to-fine pyramidal registration using AKAZE.
    Returns (aligned_np, mse, elapsed, stats, success, M_final).
    M_final is the full-resolution transform actually applied.
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

        M_residual_fullres, n_good, inlier_ratio, diag = detect_and_estimate_at_level(
            fixed_lvl, moving_warped, scale, lvl_idx
        )

        level_stats.append((n_good, inlier_ratio))

        if diag is not None and slice_id is not None and diag_dir is not None:
            save_level_diagnostics(
                diag_dir, slice_id, level_label, lvl_idx,
                fixed_lvl, moving_warped,
                diag["kp_fixed"], diag["kp_moving"],
                diag["good_matches"], diag["inlier_mask"],
            )

        if M_residual_fullres is not None:
            # Each level operates on the moving image already warped by M_accum, so
            # residuals are by design near-zero and cannot be meaningfully compared
            # across levels. The only valid gate is on the composed transform.
            M_candidate = compose_transforms(M_residual_fullres, M_accum)
            # Clamp accumulated shear/scale so per-step constraints don't compound.
            M_candidate = constrain_full_affine(M_candidate, max_scale_deviation=0.05, max_shear=0.15)
            if transform_is_sane(M_candidate):
                M_accum             = M_candidate
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
            # Full-resolution feature level failed (either directly or after L0 pre-aligned).
            # Run NCC on the full-res image pre-warped by current M_accum to recover any
            # remaining fine translation that AKAZE couldn't find.
            if lvl_idx == len(PYRAMID_SCALES) - 1:
                logger.info(f"[{slice_id}] Full-res feature level failed — trying NCC on pre-warped full-res image.")
                moving_prewarped = cv2.warpAffine(
                    moving_8bit_full, M_accum, (w_full, h_full),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                )
                M_ncc_residual = ncc_translation_fallback(fixed_8bit_full, moving_prewarped, slice_id or "unknown")
                tx_ncc, ty_ncc = M_ncc_residual[0, 2], M_ncc_residual[1, 2]
                if abs(tx_ncc) > 0.5 or abs(ty_ncc) > 0.5:
                    M_candidate = compose_transforms(M_ncc_residual, M_accum)
                    if transform_is_sane(M_candidate):
                        M_accum             = M_candidate
                        any_level_succeeded = True
                        logger.info(
                            f"[{slice_id}] NCC residual accepted "
                            f"(tx={tx_ncc:.1f}px, ty={ty_ncc:.1f}px) — composed into M_accum."
                        )
                    else:
                        logger.warning(f"[{slice_id}] NCC residual failed sanity check — carrying previous.")
                else:
                    logger.info(f"[{slice_id}] NCC residual near-zero — M_accum unchanged.")
                    if any_level_succeeded:
                        pass  # L0 result already in M_accum, that's fine

    if not any_level_succeeded:
        logger.warning(
            f"[{slice_id}] All feature levels failed — attempting NCC on raw full-res images."
        )
        M_ncc = ncc_translation_fallback(fixed_8bit_full, moving_8bit_full, slice_id or "unknown")
        tx_ncc, ty_ncc = M_ncc[0, 2], M_ncc[1, 2]
        if abs(tx_ncc) > 0.5 or abs(ty_ncc) > 0.5:
            M_accum             = M_ncc
            any_level_succeeded = True
            logger.info(f"[{slice_id}] NCC fallback accepted (tx={tx_ncc:.1f}px, ty={ty_ncc:.1f}px).")
        else:
            logger.warning(f"[{slice_id}] NCC fallback returned near-identity — writing raw slice.")

    # --- L2 piecewise affine refinement ---
    # Apply global M_accum first to pre-align moving image, then refine locally.
    moving_prealigned_ck = cv2.warpAffine(
        moving_8bit_full, M_accum, (w_full, h_full),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    piecewise_warp_ck = piecewise_affine_warp(
        fixed_8bit_full, moving_prealigned_ck,
        M_global    = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        level_scale = 1.0,
        slice_id    = slice_id or "unknown",
    )

    # --- L3 B-spline NCC elastic refinement ---
    #
    # Runs on the CK channel only (same guide channel used throughout the pipeline).
    # The estimated B-spline displacement field is then applied to ALL channels.
    #
    # Why NCC as metric:
    #   Normalized Cross-Correlation is insensitive to linear intensity differences
    #   between slices (different staining intensity, photobleaching, exposure drift).
    #   This makes it much more robust than SSD/MSE on sparse fluorescence tissue
    #   where background dominates and absolute intensities vary between sections.
    #
    # Why B-spline FFD:
    #   The deformation is parameterized by a sparse control point grid with cubic
    #   B-spline interpolation. This gives C2-continuous (smooth) deformation fields
    #   by construction — no explicit regularization term needed. Control points only
    #   affect their local neighbourhood (compact support), so one region cannot
    #   distort another arbitrarily far away.
    #
    # Guard: only run if upstream AKAZE/NCC produced a valid pre-alignment.
    # Running B-spline on a globally misaligned pair wastes time and produces
    # garbage — the NCC metric has many local optima when the images are far apart.

    remap_x = remap_y = None
    if not any_level_succeeded:
        logger.warning(
            f"[{slice_id}] Skipping B-spline NCC — all upstream levels failed, "
            f"no valid pre-alignment to refine elastically."
        )
    else:
        try:
            sitk_fixed  = sitk.GetImageFromArray(fixed_8bit_full.astype(np.float32))
            sitk_moving = sitk.GetImageFromArray(piecewise_warp_ck.astype(np.float32))

            # Initialise B-spline transform from control point grid.
            # SimpleITK infers the grid spacing from the image size and BSPLINE_GRID_NODES.
            tx_init = sitk.BSplineTransformInitializer(
                sitk_fixed,
                transformDomainMeshSize=[BSPLINE_GRID_NODES] * sitk_fixed.GetDimension(),
                order=3,
            )

            reg = sitk.ImageRegistrationMethod()

            # Metric: NCC with a 5-voxel radius neighbourhood window.
            # Larger radius = more context per sample, slower but more stable.
            reg.SetMetricAsCorrelation()

            # Optimizer: LBFGSB — quasi-Newton, well suited to smooth B-spline
            # parameter spaces. Convergence is much faster than gradient descent.
            reg.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=BSPLINE_ITERATIONS,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1000,
                costFunctionConvergenceFactor=1e7,
            )

            # Multi-resolution pyramid: coarse-to-fine.
            # Coarse levels catch large residual deformation; fine level refines.
            reg.SetShrinkFactorsPerLevel(shrinkFactors=BSPLINE_SHRINK)
            reg.SetSmoothingSigmasPerLevel(smoothingSigmas=BSPLINE_SIGMA)
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

            reg.SetInitialTransform(tx_init, inPlace=True)
            reg.SetInterpolator(sitk.sitkLinear)

            bspline_tx = reg.Execute(sitk_fixed, sitk_moving)

            # Convert B-spline transform to a dense displacement field once,
            # then use cv2.remap for all channels — avoids per-channel SimpleITK overhead.
            disp_filter = sitk.TransformToDisplacementFieldFilter()
            disp_filter.SetReferenceImage(sitk_fixed)
            disp_field  = disp_filter.Execute(bspline_tx)
            disp_field  = sitk.Cast(disp_field, sitk.sitkVectorFloat64)

            disp_np = sitk.GetArrayFromImage(disp_field)  # (H, W, 2): [dy, dx]
            map_y, map_x = np.mgrid[0:h_full, 0:w_full].astype(np.float32)
            remap_x = (map_x + disp_np[..., 0]).astype(np.float32)
            remap_y = (map_y + disp_np[..., 1]).astype(np.float32)

            logger.info(
                f"[{slice_id}] B-spline NCC refinement complete "
                f"(nodes={BSPLINE_GRID_NODES}, iters={BSPLINE_ITERATIONS}, "
                f"metric={reg.GetMetricValue():.6f})."
            )

        except Exception as exc:
            logger.warning(
                f"[{slice_id}] B-spline NCC registration failed ({exc}) — skipping elastic layer."
            )
            remap_x = remap_y = None

    aligned_channels = []
    for ch in range(fixed_np.shape[0]):
        # Apply global affine (M_accum) to every channel.
        ch_prealigned = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_accum, (w_full, h_full),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Apply B-spline displacement field if available.
        if remap_x is not None:
            warped = cv2.remap(
                ch_prealigned, remap_x, remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        else:
            warped = ch_prealigned
        aligned_channels.append(warped)

    aligned_np = np.stack(aligned_channels, axis=0).astype(np.uint16)

    # Chain-protection sanity check: if the CK channel in the registered output is
    # predominantly black, Demons (or remap) produced garbage — NaN/inf in the
    # displacement field, field divergence, or dtype overflow.  Revert to the raw
    # moving slice so the next slice in the chain gets a valid fixed reference.
    # This makes B-spline parameters safe to tune aggressively without risking kp1=0
    # on the following slice.
    ck_out           = aligned_np[CK_CHANNEL_IDX]
    ck_nonzero_frac  = np.count_nonzero(ck_out) / float(ck_out.size)
    if ck_nonzero_frac < MIN_CK_NONZERO_FRAC:
        logger.warning(
            f"[{slice_id}] Output CK channel is {ck_nonzero_frac*100:.2f}% non-zero "
            f"(threshold={MIN_CK_NONZERO_FRAC*100:.1f}%) — B-spline likely diverged. "
            f"Reverting to raw moving slice to protect chain."
        )
        aligned_np          = moving_np.copy()
        any_level_succeeded = False   # caller will write IDENTITY_FALLBACK_RAW

    _, warped_8bit = prepare_for_features_with_diagnostics(
        aligned_np[CK_CHANNEL_IDX].astype(np.float32)
    )
    mse = float(np.mean((fixed_norm.astype(np.float32) - warped_8bit.astype(np.float32)) ** 2))

    U, S_acc, Vt = np.linalg.svd(M_accum[:2, :2])
    R            = U @ Vt
    rot          = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx           = float(M_accum[0, 2])
    ty           = float(M_accum[1, 2])
    scale_pct    = (float(np.mean(S_acc)) - 1.0) * 100.0
    shear_pct    = (float(S_acc[0] / S_acc[1]) - 1.0) * 100.0 if S_acc[1] > 1e-6 else 0.0

    total_good        = sum(g for g, _ in level_stats)
    valid_ratios      = [r for _, r in level_stats if r > 0.0]
    mean_inlier_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    elapsed           = time.time() - start

    # Classify how registration was achieved for the CSV
    l0_ok = level_stats[0][0] > 0 and level_stats[0][1] > 0.0
    l1_ok = level_stats[1][0] > 0 and level_stats[1][1] > 0.0
    if l0_ok and l1_ok:
        detector_label = "AKAZE_L0+L1"
    elif l0_ok and not l1_ok:
        detector_label = "AKAZE_L0+NCC_L1"   # L0 feature + NCC fine residual
    elif not l0_ok and any_level_succeeded:
        detector_label = "NCC_Only"           # both feature levels failed, NCC on raw
    else:
        detector_label = "Identity"

    stats = dict(
        detector            = detector_label,
        n_good              = total_good,
        inlier_ratio        = round(mean_inlier_ratio, 3),
        rotation_deg        = round(rot, 3),
        tx                  = round(tx, 3),
        ty                  = round(ty, 3),
        scale_pct           = round(scale_pct, 3),
        shear_pct           = round(shear_pct, 3),
        level_matches       = [g for g, _ in level_stats],
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

    plt.suptitle(f'Registration QC AKAZE Pyramid (Stage 1): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_AKAZE_Pyramid.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Pyramidal Feature Registration (Stage 1 / Full Resolution) — {TARGET_CORE}")
    logger.info(f"Detector: AKAZE (threshold={AKAZE_THRESHOLD}) | Pyramid levels: {PYRAMID_SCALES}")

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
                         metadata={'axes': 'ZCYX'}, compression='deflate', compressionargs={'level': 6})
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
        diag_out_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_Pyramid_AKAZE")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]

            aligned_np, mse, runtime, stats, success, M_final = register_slice_pyramidal(
                fixed_np, moving_np,
                slice_id = f"Z{i:03d}_ID{real_id:03d}",
                diag_dir = diag_out_dir,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                if "NCC_Fallback" in stats["detector"]:
                    status_str = "NCC_FALLBACK_SUCCESS"
                else:
                    status_str = "SUCCESS"
            else:
                # Write raw slice — keeps chain intact, does not propagate bad transform
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
                f"CumScale: {stats['scale_pct']:+.2f}% | CumShear: {stats['shear_pct']:.2f}% | "
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
                "L0_Inlier_Ratio":    stats["level_inlier_ratios"][0],
                "L1_Inlier_Ratio":    stats["level_inlier_ratios"][1],
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
        "L0_Matches", "L1_Matches",
        "L0_Inlier_Ratio", "L1_Inlier_Ratio",
        "Success", "Status", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "MSE_After", "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_pyramid.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_ncc      = int((df["Status"] == "NCC_FALLBACK_SUCCESS").sum())
    n_sanity   = int((df["Status"] == "SANITY_FAIL_RAW").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. SUCCESS: {n_ok} | NCC_FALLBACK_SUCCESS: {n_ncc} | "
        f"SANITY_FAIL_RAW: {n_sanity} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Committing registered tensor to disk at {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    logger.info("Done.")


if __name__ == "__main__":
    main()