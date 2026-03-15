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
import yaml

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
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_AKAZE_BSpline_NCC_Zspace")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
# Optional YAML file that restricts which 0-based slice positions are registered.
# If the file does not exist, or the current core has no entry, all slices are used.
# Format example:
#   core_1: "1-16"      # keep positions 1–16 inclusive
#   core_2: "0-14"
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM      = 0.4961
SECTION_THICKNESS_UM  = 4.5    # physical Z spacing between serial sections (µm)

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
#   Increase to 6 nodes (~1013px spacing) to reduce degrees of freedom and
#   prevent over-fitting to local noise or damage artefacts.
# BSPLINE_MAX_DISPLACEMENT_PX — hard cap on per-control-point displacement.
#   Displacement fields exceeding this are clipped before remap is applied.
# BSPLINE_ITERATIONS — LBFGSB optimizer steps per resolution level.
# BSPLINE_SHRINK / BSPLINE_SIGMA — multi-resolution pyramid (coarse→fine).
BSPLINE_GRID_NODES          = 6     # was 8 — coarser grid = less local deformation
BSPLINE_ITERATIONS          = 75    # was 100 — fewer iterations = faster + less over-fit
BSPLINE_SHRINK              = [4, 2, 1]
BSPLINE_SIGMA               = [6, 3, 1]
BSPLINE_MAX_DISPLACEMENT_PX = 50.0  # hard cap: displacement vectors beyond this are clipped

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01   # revert to affine-only if CK output is >99% black


if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Slice filter
# ─────────────────────────────────────────────────────────────────────────────

def load_slice_filter(yaml_path: str, core_name: str):
    """
    Read slice_filter.yaml and return the set of allowed 0-based position indices
    for *core_name*, or None if no restriction is defined.

    YAML format:
        core_1: "1-16"     # inclusive range
        core_2: "0-5,8-12" # comma-separated ranges are also supported
    """
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    raw = data.get(core_name)
    if raw is None:
        return None                       # core not listed → use all slices

    allowed = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            allowed.add(int(part))
    return allowed


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
    Returns (M_2x3|None, n_matches, n_inliers, kp1_raw, kp2_raw, kp1, kp2, good_matches, inlier_mask).
    kp1_raw/kp2_raw are pre-ANMS keypoints; kp1/kp2 are post-ANMS.
    """
    detector  = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2_raw, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1_raw) < 4 or len(kp2_raw) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1_raw) if kp1_raw else 0}, moving={len(kp2_raw) if kp2_raw else 0})."
        )
        return None, 0, 0, [], [], [], [], [], np.array([])

    kp1, des1 = apply_anms(kp1_raw, des1, num_to_keep=ANMS_KEEP)
    kp2, des2 = apply_anms(kp2_raw, des2, num_to_keep=ANMS_KEEP)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1_raw, kp2_raw, kp1, kp2, good, np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0, kp1_raw, kp2_raw, kp1, kp2, good, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask


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

        # Clip displacement vectors exceeding the hard cap (regularization)
        disp_magnitude = np.sqrt(disp_np[..., 0]**2 + disp_np[..., 1]**2)
        excess         = disp_magnitude > BSPLINE_MAX_DISPLACEMENT_PX
        if np.any(excess):
            scale              = np.where(excess, BSPLINE_MAX_DISPLACEMENT_PX / (disp_magnitude + 1e-8), 1.0)
            disp_np[..., 0]   *= scale
            disp_np[..., 1]   *= scale
            logger.info(f"[{slice_id}] Clipped {int(excess.sum())} displacement vectors > {BSPLINE_MAX_DISPLACEMENT_PX}px.")

        remap_x = (map_x + disp_np[..., 0]).astype(np.float32)
        remap_y = (map_y + disp_np[..., 1]).astype(np.float32)

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
    M_affine, n_matches, n_inliers, kp1_raw, kp2_raw, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_log, moving_log, sid)
    akaze_ok = M_affine is not None
    if not akaze_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # ── Matching-pairs plots (before ANMS and after ANMS/RANSAC inliers) ──────
    if len(kp1_raw) > 0:
        save_matching_pairs_plot(
            fixed_log, moving_log,
            kp1_raw, kp2_raw, kp1, kp2,
            good_matches, inlier_mask,
            sid, OUTPUT_FOLDER, akaze_ok=akaze_ok,
        )

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
# Interim plots: before-ANMS and after-ANMS matching pairs
# ─────────────────────────────────────────────────────────────────────────────

def _draw_keypoint_marker(canvas, pt, color, radius=10):
    """Filled circle with a crosshair (+) drawn on top."""
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)
    arm = radius + 4
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1, cv2.LINE_AA)


def _make_side_by_side_canvas(fixed_8bit, moving_8bit):
    """Return a 3-channel black canvas with the two images placed side by side."""
    h, w   = fixed_8bit.shape[:2]
    gap    = 6
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_8bit,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_8bit, cv2.COLOR_GRAY2BGR)
    return canvas, h, w, gap


def _burn_title(canvas, title, color=(0, 230, 0)):
    """Burn a centred title string into the top of canvas in-place."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = max(0.8, canvas.shape[1] / 3000)
    thickness  = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = (canvas.shape[1] - tw) // 2
    ty = th + 10
    cv2.putText(canvas, title, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


def save_matching_pairs_plot(fixed_8bit, moving_8bit,
                              kp1_raw, kp2_raw, kp1, kp2,
                              good_matches, inlier_mask,
                              slice_id, output_folder, akaze_ok=True):
    """
    Saves two PNGs per slice pair into output_folder/interim_plots/:
      1. _before_anms.png  — all raw AKAZE keypoints, no lines (before ANMS)
      2. _after_anms.png   — RANSAC inlier matches with rainbow lines (after ANMS)
    Keypoints are drawn as filled circles with a crosshair (+) in the centre.
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    h, w   = fixed_8bit.shape[:2]
    gap    = 6
    status = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)

    # ── Plot 1: before ANMS — random sample of raw keypoints, no matches ────────
    canvas_pre, _, _, _ = _make_side_by_side_canvas(fixed_8bit, moving_8bit)
    rng         = np.random.default_rng(seed=0)
    kp1_sample  = list(kp1_raw)
    kp2_sample  = list(kp2_raw)
    if len(kp1_sample) > 3000:
        kp1_sample = [kp1_sample[i] for i in rng.choice(len(kp1_sample), 3000, replace=False)]
    if len(kp2_sample) > 3000:
        kp2_sample = [kp2_sample[i] for i in rng.choice(len(kp2_sample), 3000, replace=False)]
    for kp in kp1_sample:
        _draw_keypoint_marker(canvas_pre, kp.pt, (0, 200, 0), radius=5)
    for kp in kp2_sample:
        _draw_keypoint_marker(canvas_pre, (kp.pt[0] + w + gap, kp.pt[1]), (0, 0, 200), radius=5)
    _burn_title(
        canvas_pre,
        f"{slice_id}  kp_fixed={len(kp1_raw)}  kp_moving={len(kp2_raw)}  [BEFORE ANMS]",
        color=(180, 180, 0),
    )
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_before_anms.png"), canvas_pre)
    logger.info(f"[{slice_id}] Before-ANMS plot saved.")

    # ── Plot 2: after ANMS — RANSAC inlier matches with rainbow lines ─────────
    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep] \
                     if len(inlier_mask) > 0 else []
    n_inliers = len(inlier_matches)

    canvas_post, _, _, _ = _make_side_by_side_canvas(fixed_8bit, moving_8bit)
    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in
                          cv2.cvtColor(np.uint8([[[hue, 220, 220]]]),
                                       cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas_post,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color_bgr, 1, cv2.LINE_AA)
        _draw_keypoint_marker(canvas_post, pt1, (0, 200, 0))
        _draw_keypoint_marker(canvas_post, pt2, (0, 0, 200))

    _burn_title(
        canvas_post,
        f"{slice_id}  inliers={n_inliers}  [{status}]",
        color=title_color,
    )
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_after_anms.png"), canvas_post)
    logger.info(f"[{slice_id}] After-ANMS plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# QC montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_montage(vol, output_folder, slice_ids=None,
                        channel_idx=6, channel_name="CK", title_suffix="AKAZE+BSpline"):
    """
    Generate a QC montage showing R/G overlays for consecutive slice pairs.
    slice_ids — list of original TMA slice IDs (one per Z in vol).
                If provided, tile titles show the real IDs instead of 0-based indices.
    title_suffix — appended to the suptitle and output filename.
    """
    logger.info(f"Generating QC montage for {channel_name} channel ({title_suffix}).")
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

        # Use original slice IDs if provided, otherwise fall back to z-index
        if slice_ids is not None:
            lbl1, lbl2 = slice_ids[z1], slice_ids[z2]
        else:
            lbl1, lbl2 = z1, z2
        axes_flat[idx].set_title(f"ID{lbl1} to ID{lbl2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Registration QC {title_suffix}: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f"{TARGET_CORE}_QC_Montage_{channel_name}_{title_suffix}.png")
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

    # --- Apply slice position filter (from slice_filter.yaml if present) -----
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
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(
            out_path, vol_in[np.newaxis],
            photometric='minisblack',
            metadata={
                'axes': 'ZCYX',
                'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM,
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM,
                'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
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
    slice_ids  = []           # original TMA slice IDs, one per kept file
    for f in file_list:
        slice_ids.append(get_slice_number(f))
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

    # AKAZE montage — raw (pre-registration) volume for before/after comparison
    raw_vol = np.stack(raw_slices, axis=0)   # (N, C, H, W)
    generate_qc_montage(raw_vol,     OUTPUT_FOLDER, slice_ids=slice_ids,
                        channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                        title_suffix="AKAZE_Raw")
    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                        channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                        title_suffix="AKAZE+BSpline")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Writing registered volume to {out_tiff}")
    tifffile.imwrite(
        out_tiff, aligned_vol,
        photometric='minisblack',
        metadata={
            'axes': 'ZCYX',
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': PIXEL_SIZE_XY_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': PIXEL_SIZE_XY_UM,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': SECTION_THICKNESS_UM,
            'PhysicalSizeZUnit': 'µm',
        },
        compression='deflate', compressionargs={'level': 6},
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()