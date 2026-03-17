"""
Feature registration — single full-resolution AKAZE affine + NCC B-spline elastic.

Pipeline per slice pair:
  L0  AKAZE + ANMS + RANSAC at full resolution -> affine transform
  L1  SimpleITK NCC B-spline FFD -> elastic residual correction
      Skipped if L0 failed (no valid pre-alignment to refine).

Fallback:
  If L1 fails or does not improve on affine, reverts to the L0 affine result.
  If L0 fails, the raw moving slice is written to protect the chain.
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

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description='Feature registration — single level full resolution.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_AKAZE_BSpline_Adaptive")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# --- L1: SIMPLEITK NCC B-SPLINE CONFIGURATION ---
#
# Pure intensity-based elastic refinement — no compiled binaries, fully in Python.
# Operates on the affine-prealigned log-normalised CK channel.
# NCC metric is scale-invariant so it handles uint8 input correctly without tuning.
#
# BSPLINE_GRID_NODES       — control points per axis at the finest scale.
#                            8 is appropriate for ~6000px images (~750px spacing).
# BSPLINE_ITERATIONS       — optimizer iterations per resolution level.
# BSPLINE_SHRINK_FACTORS   — multi-resolution pyramid (coarse → fine).
# BSPLINE_SMOOTHING_SIGMAS — Gaussian smoothing at each level (pixels).
BSPLINE_GRID_NODES       = 8
BSPLINE_ITERATIONS       = 100
BSPLINE_SHRINK_FACTORS   = [4, 2, 1]
BSPLINE_SMOOTHING_SIGMAS = [4.0, 2.0, 0.0]

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

# --- DETECTOR ---
AKAZE_THRESHOLD = 0.0001   # lower threshold — detect more features on weak CK signal
                            # (was 0.0003 in older versions; RANSAC is the quality gate)

# --- ANMS ---
ANMS_KEEP = 6000           # keep more keypoints before matching
                            # More keypoints → more matches → more inliers.
                            # ANMS still enforces spatial spread.

# --- MATCHING ---
LOWE_RATIO  = 0.75         # tighter ratio (compensates for more raw keypoints)
MIN_MATCHES = 20           # minimum matches fed into RANSAC (relaxed for sparse cores)
MIN_INLIERS = 6            # minimum RANSAC survivors to accept the transform

# --- RANSAC ---
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS  = 5000
RANSAC_THRESH     = 8.0    # pixels at full resolution

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01

# B-spline acceptance criterion — relative NCC improvement over the affine baseline.
# Before the B-spline runs, NCC is measured on the affine-prealigned images.
# The B-spline result is accepted only if it improves NCC by at least this fraction.
# e.g. 0.05 = B-spline must achieve NCC at least 5% more negative than affine NCC.
# This adapts automatically to signal level: sparse-tissue cores with NCC ~ -0.55
# after affine need the same relative gain as dense-tissue cores with NCC ~ -0.85.
# Set to 0.0 to accept any improvement at all; raise to 0.10 to be stricter.
BSPLINE_NCC_MIN_IMPROVEMENT = 0.05

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- SLICE FILTER ---
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

# --- UTILITIES ---
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
    """
    Returns (linear_uint8, log_uint8).
    Log image is used for AKAZE detection (boosts weak signal).
    Linear image is used for MSE calculation.
    """
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

def apply_anms(keypoints, descriptors, num_to_keep=2000, c_robust=0.9, max_compute=5000):
    n = len(keypoints)
    if n <= num_to_keep or descriptors is None:
        return keypoints, descriptors
    # max_compute must be at least num_to_keep — otherwise the candidate pool is
    # smaller than the requested output and sort_idx[best] silently wraps/truncates.
    max_compute = max(max_compute, num_to_keep)
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

# --- L0: AKAZE AFFINE ---
# Uses fixed_log / moving_log (log-normalised uint8) for feature detection —
# log stretch boosts weak CK signal, matching the original design intent.
def akaze_affine(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, slice_id: str):
    detector      = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2_raw, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1_raw) < 4 or len(kp2_raw) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1_raw) if kp1_raw else 0}, moving={len(kp2_raw) if kp2_raw else 0})."
        )
        return None, 0, 0, [], [], [], np.array([])

    kp1, des1 = apply_anms(kp1_raw, des1, num_to_keep=ANMS_KEEP)
    kp2, des2 = apply_anms(kp2_raw, des2, num_to_keep=ANMS_KEEP)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH, maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
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
        logger.warning(f"[{slice_id}] Transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, kp1, kp2, good, mask

# --- L1: SIMPLEITK NCC B-SPLINE ELASTIC REFINEMENT ---
def apply_bspline_l1(
    fixed_log: np.ndarray,
    moving_log_affine: np.ndarray,
    moving_affine_vol: np.ndarray,
    slice_id: str,
) -> tuple:
    """
    Intensity-based elastic refinement using SimpleITK NCC B-spline FFD.

    Parameters
    ----------
    fixed_log          : uint8 log-normalised fixed CK channel (reference)
    moving_log_affine  : uint8 log-normalised moving CK channel, affine-prealigned
    moving_affine_vol  : float32 full multi-channel volume, affine-prealigned
                         (all channels warped by the B-spline displacement field)

    Returns
    -------
    (aligned_vol : np.ndarray uint16, ncc_value : float, success : bool)
    ncc_value is the final NCC metric (negative; more negative = better alignment).
    Returns (None, 0.0, False) on failure.

    Design notes
    ------------
    - NCC metric is scale-invariant — no normalization parameter to tune.
    - Multi-resolution pyramid (coarse→fine) stabilises large residuals.
    - The displacement field is extracted and applied via cv2.remap so that
      all 8 channels are warped with a single in-memory field (no disk I/O).
    - LBFGSB optimizer converges in ~100 iterations at each level and is
      more stable than gradient descent for this problem size.
    """
    try:
        c, h, w = moving_affine_vol.shape

        sitk_fixed  = sitk.GetImageFromArray(fixed_log.astype(np.float32))
        sitk_moving = sitk.GetImageFromArray(moving_log_affine.astype(np.float32))

        # Initialise B-spline transform on the fixed image domain
        tx_init = sitk.BSplineTransformInitializer(
            sitk_fixed,
            transformDomainMeshSize=[BSPLINE_GRID_NODES] * 2,
            order=3,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()                         # NCC — scale-invariant
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.05)                # 5% of pixels per iteration
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

        final_tx = reg.Execute(sitk_fixed, sitk_moving)

        ncc_value = reg.GetMetricValue()
        logger.info(
            f"[{slice_id}] B-spline NCC: stop={reg.GetOptimizerStopConditionDescription()} "
            f"metric={ncc_value:.6f} iters={reg.GetOptimizerIteration()}"
        )

        # Extract displacement field → cv2 remap arrays (no disk I/O)
        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(sitk_fixed)
        disp_field  = disp_filter.Execute(final_tx)
        disp_np     = sitk.GetArrayFromImage(disp_field)    # (H, W, 2): [dy, dx]

        map_y = (np.arange(h, dtype=np.float32)[:, None] + disp_np[..., 1])
        map_x = (np.arange(w, dtype=np.float32)[None, :] + disp_np[..., 0])

        # Apply to all channels of the full 16-bit volume
        aligned_channels = []
        for i in range(c):
            ch_warped = cv2.remap(
                moving_affine_vol[i].astype(np.float32),
                map_x.astype(np.float32),
                map_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            aligned_channels.append(ch_warped)

        aligned_vol = np.stack(aligned_channels, axis=0).astype(np.uint16)
        logger.info(f"[{slice_id}] B-spline elastic refinement complete.")
        return aligned_vol, ncc_value, True

    except Exception as exc:
        logger.warning(f"[{slice_id}] B-spline elastic failed ({exc}) — skipping elastic layer.")
        return None, 0.0, False

# --- MAIN REGISTRATION PIPELINE ---
def register_slice(fixed_np, moving_np, slice_id=None):
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # L0: AKAZE affine — log images (boosts weak signal, original design)
    M_affine, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_log, moving_log, sid)
    akaze_ok = M_affine is not None
    if not akaze_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # AKAZE inlier match plot
    if len(kp1) > 0:
        save_matching_pairs_plot(
            fixed_log, moving_log, kp1, kp2,
            good_matches, inlier_mask, sid, OUTPUT_FOLDER, akaze_ok=akaze_ok,
        )

    # Pre-align every channel with the affine transform
    c = moving_np.shape[0]
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # L1: NCC B-spline elastic — log images, affine-prealigned, only when L0 succeeded
    aligned_np    = None
    elastic_ok    = False
    ncc_value     = 0.0
    ncc_affine    = 0.0   # NCC of affine-prealigned images (baseline for acceptance)
    if akaze_ok:
        moving_log_affine = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

        # Measure NCC on affine-prealigned images before B-spline so we have
        # a per-slice baseline that reflects this tissue's actual signal level.
        sitk_fixed_tmp  = sitk.GetImageFromArray(fixed_log.astype(np.float32))
        sitk_moving_tmp = sitk.GetImageFromArray(moving_log_affine.astype(np.float32))
        ncc_filter      = sitk.StatisticsImageFilter()
        # Use SimpleITK's NCC via a quick single-resolution registration with 0 iterations
        _reg_tmp = sitk.ImageRegistrationMethod()
        _reg_tmp.SetMetricAsCorrelation()
        _reg_tmp.SetMetricSamplingStrategy(_reg_tmp.RANDOM)
        _reg_tmp.SetMetricSamplingPercentage(0.05)
        _reg_tmp.SetInterpolator(sitk.sitkLinear)
        _reg_tmp.SetOptimizerAsLBFGSB(numberOfIterations=0)
        _reg_tmp.SetInitialTransform(sitk.TranslationTransform(2), inPlace=False)
        _reg_tmp.Execute(sitk_fixed_tmp, sitk_moving_tmp)
        ncc_affine = _reg_tmp.GetMetricValue()
        logger.info(f"[{sid}] Affine NCC baseline: {ncc_affine:.4f}")

        aligned_np, ncc_value, elastic_ok = apply_bspline_l1(
            fixed_log, moving_log_affine, moving_affine_vol, sid,
        )

    # Output sanity & fallback
    if elastic_ok:
        ck_out = aligned_np[CK_CHANNEL_IDX]

        # Check 1: blank output — displacement field folded the image on itself
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(
                f"[{sid}] CK output nearly blank — B-spline diverged. "
                "Reverting to affine-only result."
            )
            elastic_ok = False
            aligned_np = moving_affine_vol.astype(np.uint16)

        # Check 2: B-spline must improve NCC by at least BSPLINE_NCC_MIN_IMPROVEMENT
        # over the affine baseline, measured on this specific slice pair.
        # NCC is negative — improvement means becoming more negative.
        # required = ncc_affine * (1 + BSPLINE_NCC_MIN_IMPROVEMENT)
        # e.g. ncc_affine=-0.55, improvement=0.05 → required=-0.5775
        else:
            ncc_required = ncc_affine * (1.0 + BSPLINE_NCC_MIN_IMPROVEMENT)
            if ncc_value > ncc_required:
                logger.warning(
                    f"[{sid}] B-spline NCC ({ncc_value:.4f}) did not improve enough "
                    f"over affine baseline ({ncc_affine:.4f}, need <{ncc_required:.4f}) "
                    "— reverting to affine-only result."
                )
                elastic_ok = False
                aligned_np = moving_affine_vol.astype(np.uint16)
    else:
        aligned_np = moving_affine_vol.astype(np.uint16)

    # B-spline interim plot — always saved when L0 succeeded so you can see
    # what the elastic layer did (or didn't do if it was reverted).
    if akaze_ok:
        save_bspline_plot(
            fixed_log, moving_log_affine,
            aligned_np[CK_CHANNEL_IDX],
            ncc_affine, ncc_value, elastic_ok, sid, OUTPUT_FOLDER,
        )

    # MSE -- linear fixed vs log warped (original metric: cross-domain perceptual measure)
    _, warped_log = prepare_ck(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_lin.astype(np.float32) - warped_log.astype(np.float32)) ** 2))

    # Stats
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
        bspline_ok   = elastic_ok,
        ncc_value    = round(float(ncc_value), 6),
        ncc_affine   = round(float(ncc_affine), 6),
    )

    # affine_np: the affine-only result (before B-spline), always uint16.
    # Used by main() to build the AKAZE-affine montage.
    affine_np = np.clip(moving_affine_vol, 0, 65535).astype(np.uint16)
    return aligned_np, affine_np, mse, time.time() - start, stats, akaze_ok, M_affine

# --- PLOTTING ---
def _draw_keypoint_marker(canvas, pt, color, radius=10):
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)
    arm = radius + 4
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1, cv2.LINE_AA)

def _make_side_by_side_canvas(fixed_8bit, moving_8bit):
    h, w   = fixed_8bit.shape[:2]
    gap    = 6
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_8bit,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_8bit, cv2.COLOR_GRAY2BGR)
    return canvas, h, w, gap

def _burn_title(canvas, title, color=(0, 230, 0)):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = max(0.8, canvas.shape[1] / 3000)
    thickness  = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = (canvas.shape[1] - tw) // 2
    ty = th + 10
    cv2.putText(canvas, title, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

def save_matching_pairs_plot(
    fixed_8bit, moving_8bit,
    kp1, kp2, good_matches, inlier_mask,
    slice_id, output_folder, akaze_ok=True,
):
    """
    Saves one PNG per slice pair: RANSAC inlier matches with rainbow lines.
    Fixed image on the left (green markers), moving on the right (blue markers).
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)
    h, w        = fixed_8bit.shape[:2]
    gap         = 6
    status      = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)

    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep] \
                     if len(inlier_mask) > 0 else []
    n_inliers = len(inlier_matches)

    canvas, _, _, _ = _make_side_by_side_canvas(fixed_8bit, moving_8bit)
    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color_bgr, 1, cv2.LINE_AA)
        _draw_keypoint_marker(canvas, pt1, (0, 200, 0))
        _draw_keypoint_marker(canvas, pt2, (0, 0, 200))

    _burn_title(canvas, f"{slice_id}  inliers={n_inliers}  [{status}]", color=title_color)
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_inliers.png"), canvas)
    logger.info(f"[{slice_id}] Inlier match plot saved ({n_inliers} inliers).")


def save_bspline_plot(
    fixed_log: np.ndarray,
    moving_log_affine: np.ndarray,
    aligned_ck: np.ndarray,
    ncc_affine: float,
    ncc_value: float,
    elastic_ok: bool,
    slice_id: str,
    output_folder: str,
):
    """
    Saves a 3-panel interim plot for the B-spline stage:
      Panel 1 (R/G): fixed vs affine-only  — shows residual before elastic
      Panel 2 (R/G): fixed vs bspline      — shows residual after elastic
      Panel 3:       difference map        — pixel-wise |affine - bspline| on CK

    This makes it immediately visible what the elastic layer added.
    If elastic_ok=False (reverted to affine), panels 1 and 2 are identical
    and the difference map is blank — clearly marking the revert.
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    def norm(x):
        p = np.percentile(x, 99.5)
        return np.clip(x.astype(np.float32) / (p if p > 0 else 1), 0, 1)

    f   = norm(fixed_log.astype(np.float32))
    a   = norm(moving_log_affine.astype(np.float32))

    # aligned_ck is uint16 — renormalise to float for display
    _, aligned_log = prepare_ck(aligned_ck.astype(np.float32))
    b = norm(aligned_log.astype(np.float32))

    overlay_affine  = np.dstack((f, a, np.zeros_like(f)))   # red=fixed, green=affine
    overlay_bspline = np.dstack((f, b, np.zeros_like(f)))   # red=fixed, green=bspline
    diff            = np.abs(a - b)                          # what the elastic layer changed
    diff_disp       = np.dstack([diff / (diff.max() + 1e-8)] * 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(overlay_affine);  axes[0].set_title("Fixed (R) vs Affine (G)",    fontsize=11); axes[0].axis('off')
    axes[1].imshow(overlay_bspline); axes[1].set_title("Fixed (R) vs B-spline (G)",  fontsize=11); axes[1].axis('off')
    axes[2].imshow(diff_disp);       axes[2].set_title("|Affine − B-spline| (CK)",   fontsize=11); axes[2].axis('off')

    status = "ACCEPTED" if elastic_ok else "REVERTED"
    fig.suptitle(
        f"{slice_id}  NCC affine={ncc_affine:.4f} → bspline={ncc_value:.4f}  [{status}]",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{slice_id}_bspline.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[{slice_id}] B-spline plot saved ({status}, NCC affine={ncc_affine:.4f} → bspline={ncc_value:.4f}).")

def generate_qc_montage(
    vol, output_folder,
    slice_ids=None, channel_idx=6,
    channel_name="CK", title_suffix="AKAZE+BSpline",
):
    logger.info(f"Generating QC montage for {channel_name} channel ({title_suffix}).")
    n_slices = vol.shape[0]
    if n_slices < 2:
        logger.warning("Fewer than 2 slices — QC montage skipped.")
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

        lbl1 = slice_ids[z1] if slice_ids is not None else z1
        lbl2 = slice_ids[z2] if slice_ids is not None else z2
        axes_flat[idx].set_title(f"ID{lbl1} to ID{lbl2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Registration QC {title_suffix}: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(
        output_folder,
        f"{TARGET_CORE}_QC_Montage_{channel_name}_{title_suffix}.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")

# --- MAIN EXECUTION ---
def main():
    logger.info(f"Feature Registration (AKAZE + NCC B-spline) — {TARGET_CORE}")
    logger.info(
        f"AKAZE threshold={AKAZE_THRESHOLD} | ANMS keep={ANMS_KEEP} | "
        f"Lowe ratio={LOWE_RATIO} | Min matches={MIN_MATCHES} | "
        f"Min inliers={MIN_INLIERS} | RANSAC thresh={RANSAC_THRESH}px | NCC min improvement={BSPLINE_NCC_MIN_IMPROVEMENT*100:.0f}%"
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
    slice_ids  = []
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
    affine_vol              = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    aligned_vol[center_idx] = raw_slices[center_idx]
    affine_vol[center_idx]  = raw_slices[center_idx]   # center is its own anchor
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

            aligned_np, affine_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                affine_vol[i]  = affine_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = raw_slices[i]
                affine_vol[i]  = raw_slices[i]   # fallback: raw = no affine either
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"BSpline: {stats['bspline_ok']} | NCC: {stats['ncc_value']:.4f} | MSE: {mse:.2f} | "
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
                "NCC_Value":    stats["ncc_value"],
                "NCC_Affine":   stats["ncc_affine"],
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
        "N_Matches", "N_Inliers", "BSpline_OK", "NCC_Value", "NCC_Affine",
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

    # --- QC MONTAGES (generated before writing the volume) ---
    # Three montages:
    #   Raw         — unregistered input (baseline)
    #   AKAZE_Affine — after AKAZE affine only (judges AKAZE contribution)
    #   AKAZE+BSpline — final output after elastic refinement
    logger.info("Generating QC montages...")
    raw_vol = np.stack(raw_slices, axis=0)
    generate_qc_montage(
        raw_vol, OUTPUT_FOLDER,
        slice_ids=slice_ids,
        channel_idx=CK_CHANNEL_IDX,
        channel_name="CK",
        title_suffix="Raw",
    )
    generate_qc_montage(
        affine_vol, OUTPUT_FOLDER,
        slice_ids=slice_ids,
        channel_idx=CK_CHANNEL_IDX,
        channel_name="CK",
        title_suffix="AKAZE_Affine",
    )
    generate_qc_montage(
        aligned_vol, OUTPUT_FOLDER,
        slice_ids=slice_ids,
        channel_idx=CK_CHANNEL_IDX,
        channel_name="CK",
        title_suffix="AKAZE+BSpline",
    )

    # --- WRITE REGISTERED VOLUME ---
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