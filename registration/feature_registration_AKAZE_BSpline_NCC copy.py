"""
Feature registration — single full-resolution AKAZE affine + B-spline NCC elastic.

Pipeline per slice pair:
  L0  AKAZE + ANMS + RANSAC at full resolution → affine transform
  L1  B-spline FFD with NCC metric (SimpleITK) → elastic residual correction
      Skipped if L0 failed (no valid pre-alignment to refine).

Fallback:
  If AKAZE produces insufficient inliers or fails sanity check, the raw
  moving slice is written to protect the chain.

Output sanity:
  After L1, the CK channel is checked for blank output (B-spline divergence).
  If >99% zero, reverts to affine-only output to protect the chain.
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
parser = argparse.ArgumentParser(description='Feature registration — single level full resolution.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Feature_AKAZE_BSpline_NCC")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# --- CHANNEL ---
CK_CHANNEL_IDX = 6

# --- DETECTOR ---
AKAZE_THRESHOLD = 0.0003   # lower than default (0.001) for weak CK signal

# --- ANMS ---
ANMS_KEEP = 2000           # keypoints to keep after spatial suppression

# --- MATCHING ---
LOWE_RATIO  = 0.80         # descriptor ambiguity filter
MIN_MATCHES = 40           # minimum matches fed into RANSAC
MIN_INLIERS = 8            # minimum RANSAC survivors to accept the transform

# --- RANSAC ---
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS  = 5000
RANSAC_THRESH     = 5.0    # pixels at full resolution

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.08  # max isotropic scale deviation from 1.0
MAX_SHEAR           = 0.15  # max singular value ratio − 1 (real tissue shear can be large)
MAX_ROTATION_DEG    = 15.0  # hard rotation gate

# --- B-SPLINE NCC ELASTIC REFINEMENT (L1) ---
# Metric:    NCC — insensitive to inter-slice intensity differences.
# Transform: B-spline FFD — C2-smooth deformation by construction.
# BSPLINE_GRID_NODES — control points per axis.
#   At 6080px, 8 nodes → ~760px spacing (tissue/gland cluster scale).
# BSPLINE_ITERATIONS — LBFGSB optimizer steps per resolution level.
# BSPLINE_SHRINK / BSPLINE_SIGMA — multi-resolution pyramid (coarse→fine).
BSPLINE_GRID_NODES = 8
BSPLINE_ITERATIONS = 100
BSPLINE_SHRINK     = [4, 2, 1]
BSPLINE_SIGMA      = [6, 3, 1]

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01   # revert to affine-only if CK output is >99% black


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
    out     = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0  = max(0, (h - target_h) // 2)
    dst_y0  = max(0, (target_h - h) // 2)
    copy_h  = min(h - src_y0, target_h - dst_y0)
    src_x0  = max(0, (w - target_w) // 2)
    dst_x0  = max(0, (target_w - w) // 2)
    copy_w  = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
        arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ck(img_arr: np.ndarray):
    """
    Returns (linear_uint8, log_uint8).
    Log image is used for AKAZE detection (boosts weak signal).
    Linear image is used for MSE calculation.
    """
    img_float   = img_arr.astype(np.float32)
    log_img     = np.log1p(img_float)
    p_lo, p_hi  = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log    = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    p_lo_lin, p_hi_lin = np.percentile(img_arr[::4, ::4], (1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_arr, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return norm_lin, norm_log


# ─────────────────────────────────────────────────────────────────────────────
# ANMS
# ─────────────────────────────────────────────────────────────────────────────

def apply_anms(keypoints, descriptors, num_to_keep=2000, c_robust=0.9, max_compute=5000):
    """Adaptive Non-Maximum Suppression for spatially uniform keypoint distribution."""
    n = len(keypoints)
    if n <= num_to_keep or descriptors is None:
        return keypoints, descriptors

    coords    = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sort_idx  = np.argsort(responses)[::-1][:max_compute]
    n         = len(sort_idx)
    coords    = coords[sort_idx]
    responses = responses[sort_idx]
    radii     = np.full(n, np.inf)

    for i in range(1, n):
        stronger = responses[:i] * c_robust > responses[i]
        if np.any(stronger):
            diffs    = coords[:i][stronger] - coords[i]
            radii[i] = np.min(np.sum(diffs ** 2, axis=1))

    best  = np.argsort(radii)[::-1][:num_to_keep]
    final = sort_idx[best]
    return tuple(keypoints[i] for i in final), descriptors[final]


# ─────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def constrain_affine(M: np.ndarray) -> np.ndarray:
    """Clamp scale deviation and shear via SVD. Translation is untouched."""
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
    """Reject transforms with implausibly large rotation."""
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG


# ─────────────────────────────────────────────────────────────────────────────
# L0 — AKAZE affine at full resolution
# ─────────────────────────────────────────────────────────────────────────────

def akaze_affine(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, slice_id: str):
    """
    AKAZE + ANMS + BFMatcher + RANSAC affine.
    Returns (M_2x3 | None, n_matches, n_inliers).
    """
    detector  = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1) if kp1 else 0}, moving={len(kp2) if kp2 else 0})."
        )
        return None, 0, 0

    kp1, des1 = apply_anms(kp1, des1, num_to_keep=ANMS_KEEP)
    kp2, des2 = apply_anms(kp2, des2, num_to_keep=ANMS_KEEP)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, len(good), n_inliers

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers


# ─────────────────────────────────────────────────────────────────────────────
# L1 — B-spline NCC elastic refinement
# ─────────────────────────────────────────────────────────────────────────────

def bspline_ncc(fixed_8bit: np.ndarray, moving_8bit: np.ndarray,
                h: int, w: int, slice_id: str):
    """
    B-spline FFD registration with NCC metric on the affine-prealigned CK channel.
    Returns (remap_x, remap_y) float32 for cv2.remap, or (None, None) on failure.
    """
    try:
        sitk_fixed  = sitk.GetImageFromArray(fixed_8bit.astype(np.float32))
        sitk_moving = sitk.GetImageFromArray(moving_8bit.astype(np.float32))

        tx_init = sitk.BSplineTransformInitializer(
            sitk_fixed,
            transformDomainMeshSize=[BSPLINE_GRID_NODES] * sitk_fixed.GetDimension(),
            order=3,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=BSPLINE_ITERATIONS,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetShrinkFactorsPerLevel(shrinkFactors=BSPLINE_SHRINK)
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=BSPLINE_SIGMA)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        reg.SetInitialTransform(tx_init, inPlace=True)
        reg.SetInterpolator(sitk.sitkLinear)

        bspline_tx = reg.Execute(sitk_fixed, sitk_moving)

        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(sitk_fixed)
        disp_field  = disp_filter.Execute(bspline_tx)
        disp_field  = sitk.Cast(disp_field, sitk.sitkVectorFloat64)

        disp_np      = sitk.GetArrayFromImage(disp_field)  # (H, W, 2)
        map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
        remap_x      = (map_x + disp_np[..., 0]).astype(np.float32)
        remap_y      = (map_y + disp_np[..., 1]).astype(np.float32)

        logger.info(
            f"[{slice_id}] B-spline NCC complete "
            f"(nodes={BSPLINE_GRID_NODES}, iters={BSPLINE_ITERATIONS}, "
            f"metric={reg.GetMetricValue():.6f})."
        )
        return remap_x, remap_y

    except Exception as exc:
        logger.warning(f"[{slice_id}] B-spline NCC failed ({exc}) — skipping elastic layer.")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main registration function
# ─────────────────────────────────────────────────────────────────────────────

def register_slice(fixed_np, moving_np, slice_id=None):
    """
    Register one moving slice to its fixed neighbour.
    Returns (aligned_np, mse, elapsed, stats, akaze_ok, M_affine).
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # ── L0: AKAZE affine ─────────────────────────────────────────────────────
    M_affine, n_matches, n_inliers = akaze_affine(fixed_log, moving_log, sid)
    akaze_ok = M_affine is not None
    if not akaze_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # ── L1: B-spline NCC (only when L0 gave a valid pre-alignment) ───────────
    remap_x = remap_y = None
    if akaze_ok:
        moving_prealigned = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        remap_x, remap_y = bspline_ncc(fixed_log, moving_prealigned, h, w, sid)

    # ── Apply to all channels ─────────────────────────────────────────────────
    def warp_channel(ch_data):
        ch_affine = cv2.warpAffine(
            ch_data.astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if remap_x is not None:
            return cv2.remap(ch_affine, remap_x, remap_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return ch_affine

    aligned_np = np.stack(
        [warp_channel(moving_np[ch]) for ch in range(fixed_np.shape[0])], axis=0
    ).astype(np.uint16)

    # ── Output sanity: revert to affine-only if B-spline output is blank ─────
    ck_out = aligned_np[CK_CHANNEL_IDX]
    if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
        logger.warning(
            f"[{sid}] CK output nearly blank — B-spline diverged. "
            f"Reverting to affine-only."
        )
        aligned_np = np.stack(
            [cv2.warpAffine(moving_np[ch].astype(np.float32), M_affine, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
             for ch in range(fixed_np.shape[0])], axis=0
        ).astype(np.uint16)
        remap_x = remap_y = None  # mark bspline as not applied

    # ── MSE ──────────────────────────────────────────────────────────────────
    _, warped_log = prepare_ck(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_lin.astype(np.float32) - warped_log.astype(np.float32)) ** 2))

    # ── Stats ─────────────────────────────────────────────────────────────────
    U, S, Vt  = np.linalg.svd(M_affine[:2, :2])
    R         = U @ Vt
    rot       = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale_pct = (float(np.mean(S)) - 1.0) * 100.0
    shear_pct = (float(S[0] / S[1]) - 1.0) * 100.0 if S[1] > 1e-6 else 0.0

    stats = dict(
        detector     = "AKAZE" if akaze_ok else "Identity",
        n_matches    = n_matches,
        n_inliers    = n_inliers,
        rotation_deg = round(rot, 3),
        tx           = round(float(M_affine[0, 2]), 3),
        ty           = round(float(M_affine[1, 2]), 3),
        scale_pct    = round(scale_pct, 3),
        shear_pct    = round(shear_pct, 3),
        bspline_ok   = remap_x is not None,
    )

    return aligned_np, mse, time.time() - start, stats, akaze_ok, M_affine


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

    plt.suptitle(f'Registration QC AKAZE+BSpline: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_AKAZE_BSpline.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Feature Registration (AKAZE + B-spline NCC) — {TARGET_CORE}")
    logger.info(f"AKAZE threshold={AKAZE_THRESHOLD} | ANMS keep={ANMS_KEEP} | "
                f"B-spline nodes={BSPLINE_GRID_NODES}")

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol_in[np.newaxis], photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='deflate',
                         compressionargs={'level': 6})
        sys.exit(0)

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)

    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical full-res shape: C={c}, H={target_h}, W={target_w}")

    logger.info(f"Loading and conforming {n_slices} slices.")
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
        logger.info(f"Executing {direction} pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            aligned_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = raw_slices[i]
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"BSpline: {stats['bspline_ok']} | MSE: {mse:.2f} | "
                f"Rot: {stats['rotation_deg']:.2f} | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
                f"Scale: {stats['scale_pct']:+.2f}% | Shear: {stats['shear_pct']:.2f}% | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":    direction,
                "Slice_Z":      i,
                "Slice_ID":     real_id,
                "Detector":     stats["detector"],
                "N_Matches":    stats["n_matches"],
                "N_Inliers":    stats["n_inliers"],
                "BSpline_OK":   stats["bspline_ok"],
                "Success":      success,
                "Status":       status_str,
                "MSE_After":    round(mse, 4),
                "Rotation_Deg": stats["rotation_deg"],
                "Shift_X_px":   stats["tx"],
                "Shift_Y_px":   stats["ty"],
                "Scale_Pct":    stats["scale_pct"],
                "Shear_Pct":    stats["shear_pct"],
                "Runtime_s":    round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = [
        "Direction", "Slice_Z", "Slice_ID", "Detector",
        "N_Matches", "N_Inliers", "BSpline_OK",
        "Success", "Status", "Rotation_Deg",
        "Shift_X_px", "Shift_Y_px", "Scale_Pct", "Shear_Pct",
        "MSE_After", "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_BSpline.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Writing registered volume to {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='deflate', compressionargs={'level': 6})
    logger.info("Done.")


if __name__ == "__main__":
    main()