"""
Feature registration — single full-resolution AKAZE affine + NCC B-spline elastic.

Pipeline per slice pair:
  L0  AKAZE (tissue-masked detection) + RANSAC at full resolution -> affine transform
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
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description='Feature registration — tissue-masked keypoint detection + NCC B-spline elastic.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_AKAZE_TissueMask_BSpline")
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

# --- L1: GRID-TILED B-SPLINE CONFIGURATION ---
#
# When BSPLINE_USE_GRID_TILES is True the image is divided into a regular
# grid of (BSPLINE_GRID_ROWS x BSPLINE_GRID_COLS) overlapping tiles before
# B-spline registration.  Each tile is solved independently and the resulting
# displacement fields are blended back with a raised-cosine (Hanning) weight
# so there are no seam artefacts at tile boundaries.
#
# Benefits:
#   - Localises the optimizer: each tile sees its own signal patch → less
#     chance of a poorly-textured region pulling control-points globally.
#   - Allows finer effective grid spacing without increasing memory footprint.
#   - Problematic tiles (e.g. folded displacement) revert independently so
#     they cannot corrupt the rest of the image.
#
# BSPLINE_GRID_ROWS / COLS — number of tile divisions along each axis.
#                             2×2 = 4 tiles; 3×3 = 9 tiles.
#                             Use 1×1 to disable tiling (equivalent to the
#                             original whole-image approach).
# BSPLINE_TILE_OVERLAP      — fractional overlap between adjacent tiles
#                             (0.25 = 25% overlap on each side).  Overlap is
#                             required for smooth blending — do not set to 0.
BSPLINE_USE_GRID_TILES  = True
BSPLINE_GRID_ROWS       = 2
BSPLINE_GRID_COLS       = 2
BSPLINE_TILE_OVERLAP    = 0.25   # fraction of tile size on each edge

# --- L1: TISSUE MASK CONFIGURATION ---
#
# When enabled, a binary mask of the tissue ROI is derived from the fixed CK
# channel and passed to SimpleITK so the NCC metric only samples pixels that
# contain real tissue signal.  Background pixels (zero-padded canvas outside
# the core) are excluded entirely.
#
# Benefits:
#   - Prevents background pixels from dominating the gradient, which causes
#     B-spline control points anchored in featureless regions to drift and
#     over-correct tissue areas (the small-core / ID13 problem).
#   - Makes per-tile NCC values more meaningful — a tile that is mostly
#     background no longer reports artificially weak NCC.
#   - The whole-image acceptance NCC is also measured masked, so it is
#     directly comparable to the masked tile NCCs.
#
# BSPLINE_MASK_OTSU          — use Otsu thresholding to separate tissue from
#                              background.  Set False to use a simple > 0
#                              threshold (sufficient when background is truly
#                              zero after conformation).
# BSPLINE_MASK_DILATE_PX     — morphological dilation radius (pixels) applied
#                              to the mask before use.  Ensures the mask
#                              slightly over-covers the tissue edge so boundary
#                              pixels are not penalised.
# BSPLINE_MIN_TISSUE_FRAC    — if the tissue mask covers less than this
#                              fraction of the canvas, skip B-spline entirely
#                              for this slice pair.  Protects very small or
#                              edge-clipped cores where the optimizer has too
#                              little signal to work with.
BSPLINE_USE_TISSUE_MASK   = True
BSPLINE_MASK_OTSU         = True
BSPLINE_MASK_DILATE_PX    = 20
BSPLINE_MIN_TISSUE_FRAC   = 0.05   # skip B-spline if tissue < 5% of canvas

# --- L1: PER-TILE ACCEPTANCE CONFIGURATION ---
#
# When enabled, each tile measures its own NCC baseline (before B-spline) on
# the same patch so the comparison is apples-to-apples.  Tiles whose B-spline
# result does not improve their local NCC by at least BSPLINE_NCC_MIN_IMPROVEMENT
# have their displacement zeroed out before blending.  This prevents a poorly-
# constrained tile (e.g. mostly background) from corrupting adjacent well-
# aligned regions through the Hanning blend.
BSPLINE_PER_TILE_ACCEPT = True

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

# --- DETECTOR ---
AKAZE_THRESHOLD    = 0.0001   # lower threshold — detect more features on weak CK signal
                               # (was 0.0003 in older versions; RANSAC is the quality gate)
AKAZE_MAX_KEYPOINTS = 20_000  # hard cap per image after tissue-masked detection.
                               # BFMatcher knnMatch has an internal 2^29 index limit that
                               # fires as an assertion error on very large descriptor sets.
                               # 20k×20k Hamming matching is fast (~1s); 100k×100k is not.
                               # Keypoints are ranked by AKAZE response before capping so
                               # only the strongest tissue features survive.

# --- MATCHING ---
LOWE_RATIO  = 0.80         # tighter ratio (compensates for more raw keypoints)
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


def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """
    Derive a uint8 binary tissue mask from the log-normalised CK channel.

    Returns a (H, W) uint8 array: 255 = tissue, 0 = background.

    Strategy
    --------
    - If BSPLINE_MASK_OTSU is True: Otsu threshold on non-zero pixels only,
      so the background class does not bias the threshold estimate.
    - Otherwise: simple > 0 threshold (works when canvas background is truly
      zero after conformation).
    - A morphological dilation by BSPLINE_MASK_DILATE_PX is applied so the
      mask slightly over-covers the tissue boundary.
    """
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
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kern)

    return mask


# --- L0: AKAZE AFFINE ---
# Uses fixed_log / moving_log (log-normalised uint8) for feature detection.
# Keypoint detection is restricted to tissue pixels via the tissue mask so
# that boundary-ring and background keypoints are excluded before matching.
# This replaces the previous ANMS step: rather than detecting everywhere and
# then spatially sub-sampling, we never detect outside tissue in the first place.
def akaze_affine(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, slice_id: str,
                 fixed_mask: np.ndarray = None, moving_mask: np.ndarray = None):
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    # Build tissue masks if not supplied externally
    if fixed_mask is None:
        fixed_mask  = build_tissue_mask(fixed_8bit)
    if moving_mask is None:
        moving_mask = build_tissue_mask(moving_8bit)

    kp1_raw, des1 = detector.detectAndCompute(fixed_8bit,  fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_8bit, moving_mask)

    if len(kp1_raw) > 0:
        coords_raw = np.array([kp.pt for kp in kp1_raw])
        logger.info(f"[{slice_id}] AKAZE keypoints (tissue-masked): n={len(kp1_raw)}, "
                    f"y_range=[{coords_raw[:,1].min():.0f}, {coords_raw[:,1].max():.0f}]")
    else:
        logger.info(f"[{slice_id}] AKAZE keypoints (tissue-masked): n=0")

    if des1 is None or des2 is None or len(kp1_raw) < 4 or len(kp2_raw) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1_raw) if kp1_raw else 0}, moving={len(kp2_raw) if kp2_raw else 0})."
        )
        return None, 0, 0, [], [], [], np.array([])

    # Cap to AKAZE_MAX_KEYPOINTS by response strength.
    # Tissue masking already removes background/boundary junk; this cap prevents
    # the BFMatcher internal index overflow (IMGIDX_ONE assertion) that fires
    # when descriptor count approaches 2^29, and keeps matching time tractable.
    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        responses = np.array([kp.response for kp in kps])
        keep      = np.argsort(responses)[::-1][:max_kp]
        return tuple(kps[i] for i in keep), des[keep]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)
    logger.info(
        f"[{slice_id}] After cap: fixed={len(kp1)}, moving={len(kp2)} "
        f"(cap={AKAZE_MAX_KEYPOINTS})"
    )

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

def _measure_ncc_masked(
    fixed_f32: np.ndarray,
    moving_f32: np.ndarray,
    mask_uint8: np.ndarray,
) -> float:
    """
    Measure NCC between two float32 patches restricted to mask pixels.
    Uses 0-iteration LBFGSB so it only evaluates the metric, no optimisation.
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


def _run_bspline_on_patch(
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
    slice_id: str,
    tile_label: str,
    mask_patch: np.ndarray = None,
) -> tuple:
    """
    Run a single B-spline registration on a (fixed_patch, moving_patch) pair.

    Parameters
    ----------
    fixed_patch  : float32 2-D array (the reference tile)
    moving_patch : float32 2-D array (the moving tile, already affine-aligned)
    slice_id     : string identifier for logging
    tile_label   : e.g. "whole" or "r0c1" for grid tiles

    Returns
    -------
    (disp_np : ndarray (H,W,2), ncc_value : float, success : bool)
    disp_np[..., 0] = dx,  disp_np[..., 1] = dy  (in pixels)
    """
    try:
        ph, pw = fixed_patch.shape
        sitk_fixed  = sitk.GetImageFromArray(fixed_patch)
        sitk_moving = sitk.GetImageFromArray(moving_patch)

        tx_init = sitk.BSplineTransformInitializer(
            sitk_fixed,
            transformDomainMeshSize=[BSPLINE_GRID_NODES] * 2,
            order=3,
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
        disp_field = disp_filter.Execute(final_tx)
        disp_np    = sitk.GetArrayFromImage(disp_field)   # (ph, pw, 2): [dx, dy] in SimpleITK
        return disp_np, ncc_value, True

    except Exception as exc:
        logger.warning(
            f"[{slice_id}][{tile_label}] B-spline patch failed ({exc})."
        )
        return None, 0.0, False


def apply_bspline_l1(
    fixed_log: np.ndarray,
    moving_log_affine: np.ndarray,
    moving_affine_vol: np.ndarray,
    slice_id: str,
    tissue_mask: np.ndarray = None,
) -> tuple:
    """
    Intensity-based elastic refinement using SimpleITK NCC B-spline FFD.

    When BSPLINE_USE_GRID_TILES is True, the image is split into
    (BSPLINE_GRID_ROWS × BSPLINE_GRID_COLS) overlapping tiles.  Each tile is
    registered independently and the displacement fields are blended back with
    a raised-cosine (Hanning) weight to eliminate seam artefacts.

    When BSPLINE_USE_TISSUE_MASK is True, the tissue mask restricts NCC
    sampling to real tissue pixels only — background is ignored.

    When BSPLINE_PER_TILE_ACCEPT is True, each tile measures its own NCC
    baseline before running B-spline.  Tiles that do not improve by at least
    BSPLINE_NCC_MIN_IMPROVEMENT have their displacement zeroed before blending.

    Parameters
    ----------
    fixed_log          : uint8 log-normalised fixed CK channel (reference)
    moving_log_affine  : uint8 log-normalised moving CK channel, affine-prealigned
    moving_affine_vol  : float32 full multi-channel volume, affine-prealigned
    slice_id           : string identifier for logging
    tissue_mask        : uint8 binary mask (255=tissue) or None

    Returns
    -------
    (aligned_vol : np.ndarray uint16, ncc_value : float, success : bool, tile_stats : list)
    """
    try:
        c, h, w    = moving_affine_vol.shape
        fixed_f32  = fixed_log.astype(np.float32)
        moving_f32 = moving_log_affine.astype(np.float32)
        mask_f32   = tissue_mask if (BSPLINE_USE_TISSUE_MASK and tissue_mask is not None) else None

        # ------------------------------------------------------------------
        # Build the full displacement field (H, W, 2)
        # ------------------------------------------------------------------
        if not BSPLINE_USE_GRID_TILES or (BSPLINE_GRID_ROWS == 1 and BSPLINE_GRID_COLS == 1):
            # ---- whole-image mode ----
            disp_np, ncc_value, ok = _run_bspline_on_patch(
                fixed_f32, moving_f32, slice_id, "whole",
                mask_patch=mask_f32,
            )
            if not ok:
                return None, 0.0, False, []
            tile_stats = []
            # ncc_value from _run_bspline_on_patch is already whole-image — no fix needed
        else:
            # ---- grid-tile mode ----
            rows = BSPLINE_GRID_ROWS
            cols = BSPLINE_GRID_COLS
            ovlp = BSPLINE_TILE_OVERLAP

            tile_h = h / rows
            tile_w = w / cols

            disp_acc   = np.zeros((h, w, 2), dtype=np.float64)
            weight_acc = np.zeros((h, w),    dtype=np.float64)
            ncc_values = []
            tile_stats = []
            n_failed   = 0

            for ri in range(rows):
                for ci in range(cols):
                    # Tile extents with overlap
                    y0 = max(0, int(round(ri * tile_h - ovlp * tile_h)))
                    y1 = min(h, int(round((ri + 1) * tile_h + ovlp * tile_h)))
                    x0 = max(0, int(round(ci * tile_w - ovlp * tile_w)))
                    x1 = min(w, int(round((ci + 1) * tile_w + ovlp * tile_w)))
                    # Core (non-overlapping) extent for the grid overlay rect
                    cy0 = int(round(ri * tile_h))
                    cy1 = min(h, int(round((ri + 1) * tile_h)))
                    cx0 = int(round(ci * tile_w))
                    cx1 = min(w, int(round((ci + 1) * tile_w)))
                    ph, pw = y1 - y0, x1 - x0
                    label  = f"r{ri}c{ci}"

                    fp        = fixed_f32[y0:y1, x0:x1]
                    mp        = moving_f32[y0:y1, x0:x1]
                    mask_tile = mask_f32[y0:y1, x0:x1] if mask_f32 is not None else None

                    # Per-tile NCC baseline (measured masked if mask available)
                    ncc_tile_baseline = 0.0
                    if BSPLINE_PER_TILE_ACCEPT:
                        ncc_tile_baseline = _measure_ncc_masked(fp, mp, mask_tile)

                    disp_tile, ncc_t, ok_t = _run_bspline_on_patch(
                        fp, mp, slice_id, label, mask_patch=mask_tile,
                    )

                    accepted = False
                    if not ok_t or disp_tile is None:
                        n_failed += 1
                        logger.warning(
                            f"[{slice_id}][{label}] Tile failed — zeroing displacement."
                        )
                        disp_tile = np.zeros((ph, pw, 2), dtype=np.float64)
                        ncc_t     = float('nan')
                    elif BSPLINE_PER_TILE_ACCEPT:
                        if abs(ncc_tile_baseline) > 1e-9:
                            tile_gain = (ncc_tile_baseline - ncc_t) / abs(ncc_tile_baseline)
                        else:
                            tile_gain = 0.0
                        if tile_gain < BSPLINE_NCC_MIN_IMPROVEMENT:
                            logger.info(
                                f"[{slice_id}][{label}] Per-tile gain {tile_gain*100:.2f}% "
                                f"< {BSPLINE_NCC_MIN_IMPROVEMENT*100:.0f}% — zeroing displacement."
                            )
                            disp_tile = np.zeros((ph, pw, 2), dtype=np.float64)
                        else:
                            accepted = True
                            ncc_values.append(ncc_t)
                            logger.info(
                                f"[{slice_id}][{label}] Per-tile gain {tile_gain*100:.2f}% — accepted."
                            )
                    else:
                        accepted = True
                        ncc_values.append(ncc_t)

                    tile_stats.append(dict(
                        ri=ri, ci=ci, label=label,
                        y0=cy0, y1=cy1, x0=cx0, x1=cx1,
                        ncc=ncc_t,
                        ncc_baseline=ncc_tile_baseline,
                        ok=ok_t,
                        accepted=accepted,
                    ))

                    # Hanning blend — zeroed tiles contribute zero displacement
                    win_y  = np.hanning(ph).astype(np.float64)
                    win_x  = np.hanning(pw).astype(np.float64)
                    win_2d = np.outer(win_y, win_x)

                    disp_acc[y0:y1, x0:x1, :] += disp_tile * win_2d[..., np.newaxis]
                    weight_acc[y0:y1, x0:x1]  += win_2d

            if n_failed == rows * cols:
                logger.warning(f"[{slice_id}] All grid tiles failed — skipping elastic layer.")
                return None, 0.0, False, []

            eps       = 1e-9
            disp_np   = disp_acc / (weight_acc[..., np.newaxis] + eps)
            mean_tile_ncc = float(np.mean(ncc_values)) if ncc_values else 0.0
            n_accepted = sum(1 for t in tile_stats if t['accepted'])
            logger.info(
                f"[{slice_id}] Grid B-spline: {rows}×{cols} tiles, "
                f"{n_failed} failed, {n_accepted} accepted, mean tile NCC={mean_tile_ncc:.6f}"
            )

        # ------------------------------------------------------------------
        # Convert displacement field to cv2 remap coordinates and apply
        # ------------------------------------------------------------------
        map_x = (np.arange(w, dtype=np.float32)[None, :] + disp_np[..., 0].astype(np.float32))
        map_y = (np.arange(h, dtype=np.float32)[:, None] + disp_np[..., 1].astype(np.float32))

        aligned_channels = []
        for i in range(c):
            ch_warped = cv2.remap(
                moving_affine_vol[i].astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            aligned_channels.append(ch_warped)

        aligned_vol = np.stack(aligned_channels, axis=0).astype(np.uint16)

        # Measure whole-image NCC on the warped CK channel — comparable to
        # ncc_affine (also whole-image). Mean tile NCC is NOT comparable.
        warped_ck_log = aligned_vol[CK_CHANNEL_IDX].astype(np.float32)
        _, warped_log_u8 = prepare_ck(warped_ck_log)
        ncc_value = _measure_ncc_masked(
            fixed_log.astype(np.float32),
            warped_log_u8.astype(np.float32),
            mask_f32,
        )
        logger.info(f"[{slice_id}] B-spline elastic refinement complete (whole-image NCC={ncc_value:.6f}).")
        return aligned_vol, ncc_value, True, tile_stats

    except Exception as exc:
        logger.warning(f"[{slice_id}] B-spline elastic failed ({exc}) — skipping elastic layer.")
        return None, 0.0, False, []

# --- MAIN REGISTRATION PIPELINE ---
def register_slice(fixed_np, moving_np, slice_id=None):
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # Build tissue masks once — reused by AKAZE detection, NCC measurement,
    # and B-spline registration so all three operate on the same tissue region.
    fixed_tissue_mask  = build_tissue_mask(fixed_log)
    moving_tissue_mask = build_tissue_mask(moving_log)
    tissue_frac        = float(np.count_nonzero(fixed_tissue_mask)) / fixed_tissue_mask.size
    logger.info(f"[{sid}] Tissue mask: {tissue_frac*100:.1f}% of canvas covered.")

    # L0: AKAZE affine — tissue-masked keypoint detection
    M_affine, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_log, moving_log, sid,
                     fixed_mask=fixed_tissue_mask, moving_mask=moving_tissue_mask)
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
    tile_stats    = []
    tissue_mask   = fixed_tissue_mask if BSPLINE_USE_TISSUE_MASK else None
    if akaze_ok:
        moving_log_affine = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        if tissue_frac < BSPLINE_MIN_TISSUE_FRAC:
            logger.warning(
                f"[{sid}] Tissue fraction {tissue_frac*100:.1f}% < "
                f"{BSPLINE_MIN_TISSUE_FRAC*100:.0f}% minimum — skipping B-spline."
            )
            aligned_np = moving_affine_vol.astype(np.uint16)
            akaze_ok   = False   # suppress B-spline plot for this slice

        if akaze_ok:
            # Measure NCC on affine-prealigned images (masked if available)
            ncc_affine = _measure_ncc_masked(
                fixed_log.astype(np.float32),
                moving_log_affine.astype(np.float32),
                tissue_mask,
            )
            logger.info(f"[{sid}] Affine NCC baseline (masked={BSPLINE_USE_TISSUE_MASK}): {ncc_affine:.4f}")

            aligned_np, ncc_value, elastic_ok, tile_stats = apply_bspline_l1(
                fixed_log, moving_log_affine, moving_affine_vol, sid,
                tissue_mask=tissue_mask,
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
        # NCC is negative — improvement means becoming more negative (lower value).
        # Relative improvement = (ncc_value - ncc_affine) / |ncc_affine|
        # e.g. ncc_affine=-0.7236, ncc_value=-0.7465 → improvement=(−0.7465−−0.7236)/0.7236=+0.031
        # Accept if improvement >= BSPLINE_NCC_MIN_IMPROVEMENT (i.e. at least 5% gain).
        else:
            if abs(ncc_affine) > 1e-9:
                relative_improvement = (ncc_affine - ncc_value) / abs(ncc_affine)
            else:
                relative_improvement = 0.0
            if relative_improvement < BSPLINE_NCC_MIN_IMPROVEMENT:
                logger.warning(
                    f"[{sid}] B-spline NCC ({ncc_value:.4f}) did not improve enough "
                    f"over affine baseline ({ncc_affine:.4f}): "
                    f"relative gain={relative_improvement*100:.2f}% < required {BSPLINE_NCC_MIN_IMPROVEMENT*100:.0f}% "
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
            tile_stats=tile_stats,
        )

    # MSE — both images in log space so the comparison is within the same domain.
    # (previously compared fixed_lin vs warped_log which mixed intensity spaces)
    _, warped_log = prepare_ck(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_log.astype(np.float32) - warped_log.astype(np.float32)) ** 2))

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
    tile_stats: list = None,
):
    """
    Saves an interim plot for the B-spline stage.

    When grid tiling is active (tile_stats is non-empty) the figure has 4 panels:
      Panel 1 (R/G): fixed vs affine-only      — residual before elastic
      Panel 2 (R/G): fixed vs bspline result   — residual after elastic
      Panel 3:       |affine − bspline| diff   — what the elastic layer changed
      Panel 4:       grid overlay on fixed     — tile boundaries + per-tile NCC heatmap

    Without tiling (tile_stats empty) only panels 1–3 are shown (original layout).

    The grid panel colour-codes each tile by its NCC value:
      green  = strong improvement (most negative NCC)
      yellow = moderate
      red    = weak or failed tile
    The per-tile NCC is printed in the centre of each cell.
    """
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

    use_grid_panel = tile_stats is not None and len(tile_stats) > 0
    n_panels = 4 if use_grid_panel else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    axes[0].imshow(overlay_affine);  axes[0].set_title("Fixed (R) vs Affine (G)",   fontsize=11); axes[0].axis('off')
    axes[1].imshow(overlay_bspline); axes[1].set_title("Fixed (R) vs B-spline (G)", fontsize=11); axes[1].axis('off')
    axes[2].imshow(diff_disp);       axes[2].set_title("|Affine − B-spline| (CK)",  fontsize=11); axes[2].axis('off')

    if use_grid_panel:
        ax = axes[3]
        ax.imshow(f, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Grid tiles — NCC per tile", fontsize=11)
        ax.axis('off')

        # Collect finite NCC values to drive the colour scale
        ncc_vals = [t['ncc'] for t in tile_stats if t['ok'] and np.isfinite(t['ncc'])]
        ncc_min  = min(ncc_vals) if ncc_vals else -1.0   # most negative = best
        ncc_max  = max(ncc_vals) if ncc_vals else 0.0

        h_img, w_img = f.shape

        for t in tile_stats:
            y0, y1, x0, x1 = t['y0'], t['y1'], t['x0'], t['x1']
            tw, th = x1 - x0, y1 - y0

            # Colour coding:
            #   green  = accepted, good NCC
            #   orange = ran but rejected by per-tile acceptance
            #   red    = failed outright
            if not t['ok']:
                r, g       = 1.0, 0.0
                label_text = "FAIL"
            elif not t.get('accepted', True):
                r, g       = 0.9, 0.35   # orange
                label_text = f"{t['ncc']:.3f}\n✗"
            elif np.isfinite(t['ncc']) and (ncc_max - ncc_min) > 1e-6:
                strength   = (t['ncc'] - ncc_max) / (ncc_min - ncc_max)
                r, g       = 1.0 - strength, strength
                label_text = f"{t['ncc']:.3f}"
            else:
                r, g       = 0.5, 0.5
                label_text = f"{t['ncc']:.3f}" if np.isfinite(t['ncc']) else "n/a"

            # Append per-tile gain if available
            if BSPLINE_PER_TILE_ACCEPT and t['ok']:
                base = t.get('ncc_baseline', float('nan'))
                if np.isfinite(base) and abs(base) > 1e-9:
                    gain        = (base - t['ncc']) / abs(base) * 100
                    label_text += f"\n{gain:+.1f}%"

            rect_color = (r, g, 0.0, 0.35)

            rect = Rectangle(
                (x0, y0), tw, th,
                linewidth=2, edgecolor=(r, g, 0.0, 0.9),
                facecolor=rect_color,
            )
            ax.add_patch(rect)

            font_size = max(7, min(14, int(min(tw, th) / 8)))
            ax.text(
                x0 + tw / 2, y0 + th / 2,
                f"{t['label']}\n{label_text}",
                color='white', fontsize=font_size, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.45, linewidth=0),
            )

        # Colour bar legend (green=best, red=worst)
        cmap   = mcolors.LinearSegmentedColormap.from_list("rg", ["red", "yellow", "green"])
        sm     = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=ncc_max, vmax=ncc_min))
        sm.set_array([])
        cbar   = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("NCC (more −ve = better)", fontsize=9)

    status = "ACCEPTED" if elastic_ok else "REVERTED"
    if abs(ncc_affine) > 1e-9:
        rel_gain = (ncc_affine - ncc_value) / abs(ncc_affine) * 100
        gain_str = f"  Δ={rel_gain:+.2f}%"
    else:
        gain_str = ""
    fig.suptitle(
        f"{slice_id}  NCC affine={ncc_affine:.4f} → bspline={ncc_value:.4f}{gain_str}  [{status}]",
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
        f"AKAZE threshold={AKAZE_THRESHOLD} | Tissue-masked detection | Max keypoints={AKAZE_MAX_KEYPOINTS} | "
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
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_TissueMask_Aligned.ome.tif")
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
    logger.info(f"Processing {n_slices} slices (on-demand disk reads — no full preload).")

    # Only collect slice IDs — do NOT load all slices into memory simultaneously.
    # Loading all slices at once causes OOM (20 slices × 8ch × 6112² × uint16 ≈ 23GB).
    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
        """Load and conform a single slice from disk."""
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
    affine_vol              = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    affine_vol[center_idx]  = center_raw   # center is its own anchor
    del center_raw
    anchor_id               = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = load_slice(i)   # read from disk, discard after use
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            aligned_np, affine_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                affine_vol[i]  = affine_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = moving_np
                affine_vol[i]  = moving_np   # fallback: raw = no affine either
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            del moving_np   # free immediately

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
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_TissueMask_BSpline.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    # --- WRITE REGISTERED VOLUME FIRST ---
    # Write before montages so the result is always saved even if montage fails.
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_TissueMask_Aligned.ome.tif")
    logger.info(f"Writing registered volume to {out_tiff}")
    try:
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
        logger.info("Registered volume written.")
    except Exception as exc:
        logger.error(f"Volume write failed: {exc}")

    # --- QC MONTAGES ---
    logger.info("Generating QC montages...")
    try:
        generate_qc_montage(
            affine_vol, OUTPUT_FOLDER,
            slice_ids=slice_ids,
            channel_idx=CK_CHANNEL_IDX,
            channel_name="CK",
            title_suffix="AKAZE_TissueMask_Affine",
        )
    except Exception as exc:
        logger.error(f"Affine montage failed: {exc}")
    del affine_vol

    try:
        generate_qc_montage(
            aligned_vol, OUTPUT_FOLDER,
            slice_ids=slice_ids,
            channel_idx=CK_CHANNEL_IDX,
            channel_name="CK",
            title_suffix="AKAZE_TissueMask+BSpline",
        )
    except Exception as exc:
        logger.error(f"BSpline montage failed: {exc}")
    del aligned_vol
    logger.info("Done.")

if __name__ == "__main__":
    main()