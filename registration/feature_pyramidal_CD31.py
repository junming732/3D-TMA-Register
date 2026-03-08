"""
Feature registration as the FIRST stage in the pipeline.
Reads raw .ome.tif slices at FULL resolution.

Detector:   AKAZE with lowered detection threshold (0.0003)
Pyramid:    3-level coarse-to-fine (0.25x → 0.50x → 1.00x)

Changes vs previous version:
  - Reverted to single AKAZE detector (no SIFT/fallback switching)
  - Relaxed MIN_INLIER_RATIO_BY_LEVEL to [0.10, 0.06, 0.02] based on observed data
  - Per-level adaptive Lowe ratio and min-match threshold
  - Per-level inlier ratio gate (rejects genuinely degenerate RANSAC solutions)
  - Per-step and cumulative transform sanity gate (rotation + translation bounds)
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
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Feature_pyramidal_CD31")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# --- PYRAMID ---
PYRAMID_SCALES = [0.25, 0.50, 1.0]

# --- DETECTOR ---
# Lower threshold than default (0.001) to detect more keypoints on weak CK signal
AKAZE_THRESHOLD = 0.0003

# --- CHANNEL ---
CK_CHANNEL_IDX = 1

# --- MATCHING — per pyramid level [L0=0.25x, L1=0.50x, L2=1.00x] ---
# Lowe ratio: controls descriptor ambiguity filtering per level
# Min matches: minimum good matches before RANSAC is attempted
# Min inlier ratio: relaxed based on observed data — rejects only genuinely degenerate results
#   L0 typical healthy range: 0.15–0.21  → threshold 0.10
#   L1 typical healthy range: 0.08–0.10  → threshold 0.06
#   L2 typical healthy range: 0.01–0.05  → threshold 0.02
LOWE_RATIO_BY_LEVEL       = [0.80, 0.82, 0.80]
MIN_MATCHES_BY_LEVEL      = [6,    8,    10   ]
MIN_INLIER_RATIO_BY_LEVEL = [0.10, 0.06, 0.02]

# --- RANSAC ---
RANSAC_CONFIDENCE     = 0.995
RANSAC_MAX_ITERS      = 5000
RANSAC_THRESH_FULLRES = 5.0    # pixels at full resolution; auto-scaled per level

# --- ANMS ---
ANMS_KEEP_FULLRES = 2000       # budget at full resolution; scaled linearly per level

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.05

# --- SANITY GATE ---
# Reject any per-step or cumulative transform exceeding these physical bounds.
# Tune MAX_TRANSLATION_PX to roughly 10–15% of your core's pixel width.
MAX_ROTATION_DEG   = 15.0
MAX_TRANSLATION_PX = 300.0

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
    """Returns True if rotation and translation are within physical bounds."""
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx, ty   = abs(M[0, 2]), abs(M[1, 2])
    return rot_deg <= MAX_ROTATION_DEG and tx <= MAX_TRANSLATION_PX and ty <= MAX_TRANSLATION_PX


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
    min_r         = MIN_INLIER_RATIO_BY_LEVEL[level_idx]

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
        return None, len(good), 0.0, None

    inlier_ratio = float(mask.sum()) / len(mask)

    if inlier_ratio < min_r:
        logger.warning(f"[{name}] Inlier ratio too low ({inlier_ratio:.3f} < {min_r}) — rejecting level.")
        return None, len(good), inlier_ratio, None

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
        return None, len(good), inlier_ratio, None

    diag = {
        "kp_fixed":     kp1_anms,
        "kp_moving":    kp2_anms,
        "good_matches": good,
        "inlier_mask":  mask.ravel(),
    }

    logger.info(
        f"[{name}] matches={len(good)} inliers={inlier_ratio:.2f} "
        f"tx={M_fullres[0,2]:.1f}px ty={M_fullres[1,2]:.1f}px"
    )

    return M_fullres, len(good), inlier_ratio, diag


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

    total_good        = sum(g for g, _ in level_stats)
    valid_ratios      = [r for _, r in level_stats if r > 0.0]
    mean_inlier_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    elapsed           = time.time() - start

    stats = dict(
        detector            = "AKAZE_Pyramid_3L",
        n_good              = total_good,
        inlier_ratio        = round(mean_inlier_ratio, 3),
        rotation_deg        = round(rot, 3),
        tx                  = round(tx, 3),
        ty                  = round(ty, 3),
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
                status_str     = "SUCCESS"
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
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_pyramid.csv"), index=False
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