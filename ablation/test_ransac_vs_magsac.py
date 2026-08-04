"""
Ablation harness — Step 1 (RANSAC vs MAGSAC) + Step 2 (Tukey) + Step 3 (downsample).
======================================================================
Step 1 tests ONLY the robust-estimation method used inside
cv2.estimateAffine2D. Everything else (AKAZE detection, Lowe-ratio
matching, thresholds, constrain_affine/transform_is_sane sanity checks,
NCC gate) is copied verbatim from akaze_romav2_multi_channel_warp.py so
this is a true single-variable test, not a re-implementation.

Key design choice: AKAZE keypoint detection + BFMatcher + Lowe-ratio
filtering is run ONCE per slice pair. The resulting (src_pts, dst_pts)
are then fit with EACH base solver (cv2.RANSAC, cv2.USAC_MAGSAC), so any
difference is attributable to the solver alone.

Step 2 layers a Tukey residual filter on top of EACH base solver's
inlier set, independently: compute residual reprojection distance for
the accepted inliers, drop anything beyond the high Tukey fence
(Q3 + 1.5*IQR), and refit via ordinary least squares on the survivors.
This is a pure post-filter — it does not touch detection, matching, or
the base solver call — so it isolates "does pruning borderline inliers
help" as its own variable, layered independently on top of whichever
base solver wins Step 1. If too few points survive to refit, or the
refit fails the sanity check, it falls back to the base solver's
transform unchanged rather than risking a worse result.

Step 3 tests downsampled AKAZE detection: detect/match/RANSAC on a
downsampled image (cheaper, less exposed to fine-texture noise), then
rescale the surviving inlier point coordinates back to full resolution
and refit precisely there via ordinary least squares — not the lossier
approach of rescaling the low-res affine matrix. Tested against the
plain full-res RANSAC baseline (current production config), since
steps 1-2's winners aren't decided yet — swap DOWNSAMPLE_BASE_METHOD
once they are. Reports both NCC delta and detection wall-time speedup,
plus which pairs succeeded at one resolution but failed at the other.

Pairs used: adjacent RAW slices (i, i+1) walked across the whole
filtered stack for the given core — deliberately NOT the anchor-outward
affine-chain used in production, so every pair here is judged on raw
CK content only, with no compounding from earlier registration steps.
This keeps the comparison clean for a pure L0 test.

Usage:
    python test_ransac_vs_magsac.py --start 1 --end 30
    python test_ransac_vs_magsac.py --core_name coreA --core_name coreB
    python test_ransac_vs_magsac.py --start 1 --end 30 --downsample_max_dim 850

Output:
    ransac_vs_magsac_ablation.csv, written next to this script (one row
    per pair, with base + Tukey-filtered columns for each solver, plus
    downsampled-detection columns).
    Summary printed to stdout: Step 1 (solver comparison), Step 2
    (Tukey-vs-base per solver), Step 3 (downsampled-vs-full-res NCC
    delta, which pairs each resolution rescued/lost, detection speedup).
"""

import os
import sys
import glob
import re
import argparse
import logging
import time

import numpy as np
import pandas as pd
import tifffile
import cv2
import SimpleITK as sitk
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
import config
from held_out_tre import split_correspondences_for_tre, compute_tre_affine

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants copied verbatim from akaze_romav2_multi_channel_warp.py — keep
# in sync manually if you retune the production script's L0 config.
# ─────────────────────────────────────────────────────────────────────────────
AKAZE_THRESHOLD     = 0.0001
AKAZE_MAX_KEYPOINTS = 20_000
LOWE_RATIO          = 0.80
MIN_MATCHES         = 20
MIN_INLIERS         = 6
RANSAC_CONFIDENCE   = 0.995
RANSAC_MAX_ITERS    = 5000
RANSAC_THRESH       = 8.0
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0
CK_CHANNEL_IDX      = 6

# Step 3 — downsampled AKAZE detection. Max dimension (px) the image is
# resized to before detection; VALIS's own default for its rigid-registration
# stage is 850px (range 500-2000px commonly used). Tested against plain
# full-res RANSAC (current production baseline) — not yet layered onto
# whichever solver/Tukey combo wins steps 1-2, since those results aren't in
# yet. Once they are, swap DOWNSAMPLE_BASE_METHOD below.
DOWNSAMPLE_MAX_DIM     = 1200
DOWNSAMPLE_BASE_METHOD = cv2.RANSAC

DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
OUTPUT_CSV        = os.path.join(current_dir, "ransac_vs_magsac_ablation_tre.csv")

# Held-out TRE (see held_out_tre.py). This is INDEPENDENT of the existing
# ncc_affine columns below — those still fit on the full match set exactly
# as before, so nothing about the NCC comparison changes. TRE is computed
# from a SEPARATE fit, done only on a subset of matches, with the remainder
# held out and never touched by fitting — avoiding VALIS's own fallback
# (which reuses fitting-stage correspondences and is therefore optimistic).
# The SAME fit/holdout split is reused for every solver on a given pair, so
# the TRE comparison stays a fair, single-variable test just like the NCC one.
TRE_HOLDOUT_FRAC = 0.2
TRE_MIN_HOLDOUT  = 15
# CHANGE THIS to your actual pixel size — TRE in microns is meaningless
# until it reflects your real imaging resolution, not a placeholder.
PIXEL_SIZE_UM = 0.497

METHODS = {
    "RANSAC":      cv2.RANSAC,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
}


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim (or trivially adapted) from the production script
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


def prepare_ck(img_arr):
    """Returns (norm_lin, norm_log) uint8 — identical to production prepare_ck."""
    img_float = img_arr.astype(np.float32)
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_lin, norm_log


def get_mask_sibling_path(tif_path):
    if tif_path.endswith(".ome.tif"):
        stem = tif_path[:-len(".ome.tif")]
    else:
        stem = os.path.splitext(tif_path)[0]
    return stem + "_tissue_mask.png"


def load_mask_or_none(tif_path, shape_hw):
    mask_path = get_mask_sibling_path(tif_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape_hw:
            return mask
        logger.warning(f"Mask at {mask_path} missing/shape-mismatched — matching unmasked.")
    else:
        logger.warning(f"No precomputed mask for {os.path.basename(tif_path)} — matching unmasked.")
    return None


def constrain_affine(M):
    if M is None:
        return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S        = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out


def transform_is_sane(M):
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG


def measure_ncc(fixed_f32, moving_f32, mask_uint8):
    """Masked NCC via 0-iteration SimpleITK LBFGSB — identical to production."""
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
# Detection/matching run ONCE — the shared input to both solver fits
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_match(fixed_log, moving_log, fixed_mask, moving_mask, slice_id):
    """
    Identical detection/matching logic to production akaze_affine, split out
    so both solvers reuse the exact same (src_pts, dst_pts).
    Returns (src_pts, dst_pts, n_matches) or (None, None, n_matches) on
    feature starvation / insufficient matches.
    """
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_log,  fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_log, moving_mask)

    n1 = len(kp1_raw) if kp1_raw else 0
    n2 = len(kp2_raw) if kp2_raw else 0
    if des1 is None or des2 is None or n1 < 4 or n2 < 4:
        logger.warning(f"[{slice_id}] Feature starvation (fixed={n1}, moving={n2}).")
        return None, None, 0

    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw
              if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, None, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts, len(good)


def fit_affine(src_pts, dst_pts, method, slice_id, method_name):
    """Identical fit/constrain/sanity logic to production akaze_affine's tail."""
    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=method,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    if M is None or mask is None:
        logger.warning(f"[{slice_id}] {method_name} diverged.")
        return None, 0, None

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] {method_name} inlier count too low ({n_inliers}).")
        return None, n_inliers, None

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        logger.warning(f"[{slice_id}] {method_name} transform rejected (rotation out of range).")
        return None, n_inliers, None

    return M, n_inliers, mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Tukey residual filter, layered on top of any base solver's inlier
# set. Pure post-filter: does not touch AKAZE detection, matching, or the
# base solver call above — it only decides which of the ALREADY-ACCEPTED
# inliers get kept for a refit, using the classic Tukey fence on residual
# reprojection distance (Q3 + 1.5*IQR). Only the high fence matters since
# residual distances are non-negative.
# ─────────────────────────────────────────────────────────────────────────────

def fit_affine_lstsq(from_pts, to_pts):
    """
    Ordinary (non-robust) least-squares affine fit: to ≈ M @ from.
    Used for the post-Tukey refit since the input set has already been
    outlier-pruned — an unweighted LS fit here isolates "did pruning help"
    from "did re-running RANSAC on fewer points help", which would muddy
    the comparison.
    """
    from_flat = from_pts.reshape(-1, 2).astype(np.float64)
    to_flat   = to_pts.reshape(-1, 2).astype(np.float64)
    n = from_flat.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    A[0::2, 0] = from_flat[:, 0]
    A[0::2, 1] = from_flat[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 3] = from_flat[:, 0]
    A[1::2, 4] = from_flat[:, 1]
    A[1::2, 5] = 1.0
    b[0::2] = to_flat[:, 0]
    b[1::2] = to_flat[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[sol[0], sol[1], sol[2]],
                     [sol[3], sol[4], sol[5]]], dtype=np.float64)


def fit_affine_tukey(src_pts, dst_pts, base_mask, base_M, slice_id, method_name):
    """
    Given the base solver's M and inlier mask, compute residuals for the
    accepted inliers, drop anything beyond the Tukey high fence
    (Q3 + 1.5*IQR), and refit via ordinary least squares on the survivors.
    Falls back to the base M unchanged if too few points survive to refit
    (< 3, or < MIN_INLIERS) — a failed filter should never leave you worse
    off than not filtering at all.
    Returns (M, n_inliers_after_filter, applied: bool).
    """
    inlier_idx = np.where(base_mask.ravel() == 1)[0]
    src_in = src_pts[inlier_idx].reshape(-1, 2)
    dst_in = dst_pts[inlier_idx].reshape(-1, 2)

    # Residual: predicted src (from dst via base_M) vs actual src — same
    # from/to convention as the estimateAffine2D(dst_pts, src_pts, ...) call.
    pred_src = (base_M[:, :2] @ dst_in.T).T + base_M[:, 2]
    residuals = np.linalg.norm(pred_src - src_in, axis=1)

    q1, q3 = np.percentile(residuals, [25, 75])
    iqr    = q3 - q1
    fence_hi = q3 + 1.5 * iqr
    keep = residuals <= fence_hi
    n_kept = int(keep.sum())

    if n_kept < max(3, MIN_INLIERS):
        logger.info(
            f"[{slice_id}] {method_name}+Tukey: only {n_kept} points survive "
            f"fence — falling back to base {method_name} transform unchanged."
        )
        return base_M, int(base_mask.sum()), False

    src_kept = src_in[keep]
    dst_kept = dst_in[keep]
    M_refined = fit_affine_lstsq(dst_kept, src_kept)

    M_refined = constrain_affine(M_refined)
    if M_refined is None or not transform_is_sane(M_refined):
        logger.info(
            f"[{slice_id}] {method_name}+Tukey: refit failed sanity check — "
            "falling back to base transform unchanged."
        )
        return base_M, int(base_mask.sum()), False

    dropped = int(base_mask.sum()) - n_kept
    logger.info(f"[{slice_id}] {method_name}+Tukey: dropped {dropped} borderline "
               f"inlier(s), refit on {n_kept}.")
    return M_refined, n_kept, True


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — downsampled AKAZE detection, full-resolution affine refit.
# Detect/match/RANSAC on a downsampled image (cheaper, less exposed to
# fine-texture noise), then rescale the surviving inlier point coordinates
# back to full resolution and refit precisely there via ordinary least
# squares — rather than the lossier approach of rescaling the low-res
# affine matrix itself.
# ─────────────────────────────────────────────────────────────────────────────

def downsample_for_detection(img, mask, max_dim):
    """
    Resize img (and its mask, nearest-neighbor to stay binary) so its
    longer side is at most max_dim. Returns (img_ds, mask_ds, scale), where
    scale = downsampled_size / original_size (so full_res_coord = ds_coord
    / scale). scale == 1.0 (no-op) if the image is already <= max_dim.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img, mask, 1.0
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_ds  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_ds = None
    if mask is not None:
        mask_ds = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_ds, mask_ds, scale


def fit_affine_downsampled(fixed_log, moving_log, fixed_mask, moving_mask,
                           slice_id, max_dim=None, base_method=None):
    """
    Full step-3 pipeline for one pair: downsample -> detect/match -> RANSAC
    at a scale-adjusted threshold -> rescale inlier coords to full res ->
    ordinary-least-squares refit -> constrain/sanity-check.

    max_dim/base_method default to the module-level DOWNSAMPLE_MAX_DIM /
    DOWNSAMPLE_BASE_METHOD, read at call time (not frozen at def time), so
    the --downsample_max_dim CLI flag can override them.

    Returns dict with: M (or None), n_matches_ds, n_inliers_ds, scale,
    detect_time_s (detection+matching wall time at this resolution only,
    excludes the solver fit — comparable directly to the full-res
    detect_and_match call's timing).
    """
    if max_dim is None:
        max_dim = DOWNSAMPLE_MAX_DIM
    if base_method is None:
        base_method = DOWNSAMPLE_BASE_METHOD

    fixed_ds,  fixed_mask_ds,  scale_f = downsample_for_detection(fixed_log, fixed_mask, max_dim)
    moving_ds, moving_mask_ds, scale_m = downsample_for_detection(moving_log, moving_mask, max_dim)
    # Both slices in a core share the same canvas shape in this pipeline, so
    # scale_f == scale_m in practice; guard against silent divergence anyway.
    scale = scale_f
    if abs(scale_f - scale_m) > 1e-6:
        logger.warning(f"[{slice_id}] Downsample scale mismatch fixed={scale_f} "
                       f"moving={scale_m} — using fixed-side scale.")

    t0 = time.time()
    src_pts_ds, dst_pts_ds, n_matches_ds = detect_and_match(
        fixed_ds, moving_ds, fixed_mask_ds, moving_mask_ds, slice_id + "_ds")
    detect_time_s = time.time() - t0

    result = dict(M=None, n_matches_ds=n_matches_ds, n_inliers_ds=0,
                  scale=scale, detect_time_s=detect_time_s)

    if src_pts_ds is None:
        return result

    ransac_thresh_ds = RANSAC_THRESH * scale  # threshold in downsampled-pixel units
    M_ds, mask_ds = cv2.estimateAffine2D(
        dst_pts_ds, src_pts_ds, method=base_method,
        ransacReprojThreshold=ransac_thresh_ds,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    if M_ds is None or mask_ds is None:
        logger.warning(f"[{slice_id}] Downsampled RANSAC diverged.")
        return result

    n_inliers_ds = int(mask_ds.sum())
    result["n_inliers_ds"] = n_inliers_ds
    if n_inliers_ds < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Downsampled inlier count too low ({n_inliers_ds}).")
        return result

    inlier_idx = np.where(mask_ds.ravel() == 1)[0]
    src_in_ds  = src_pts_ds[inlier_idx].reshape(-1, 2)
    dst_in_ds  = dst_pts_ds[inlier_idx].reshape(-1, 2)

    # Rescale surviving inlier coordinates to full-resolution pixel space,
    # then refit precisely there — this is the step that recovers
    # sub-pixel accuracy despite detecting on a coarser image.
    src_in_full = src_in_ds / scale
    dst_in_full = dst_in_ds / scale
    M_refit = fit_affine_lstsq(dst_in_full, src_in_full)
    M_refit = constrain_affine(M_refit)
    if M_refit is None or not transform_is_sane(M_refit):
        logger.warning(f"[{slice_id}] Downsampled-detect refit failed sanity check.")
        return result

    result["M"] = M_refit
    logger.info(f"[{slice_id}] Downsampled detect (scale={scale:.3f}, "
               f"{fixed_ds.shape[1]}x{fixed_ds.shape[0]}): "
               f"matches={n_matches_ds} inliers={n_inliers_ds} "
               f"detect_time={detect_time_s:.2f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_held_out_tre(src_pts, dst_pts, image_diag_px, sid):
    """
    Independent of the ncc_affine columns elsewhere in this script — those
    still fit on the FULL match set exactly as before. This does a SEPARATE
    fit, on a subset of matches only, with the rest held out and never
    touched by fitting, then measures error on the held-out points alone.
    The same fit/holdout split is reused for every solver in METHODS, so
    whichever one comes out ahead on TRE, it's a fair single-variable
    comparison — same points fit each candidate, same points evaluate it.
    """
    result = {}
    if src_pts is None or len(src_pts) == 0:
        result["tre_n_fit"] = 0
        result["tre_n_holdout"] = 0
        for m in METHODS:
            result[f"{m}_tre_mean_px"] = np.nan
            result[f"{m}_tre_median_px"] = np.nan
            result[f"{m}_tre_p90_px"] = np.nan
        return result

    n = len(src_pts)
    # Deterministic per-pair seed so re-running the script reproduces the
    # same split (important for comparing across script versions/runs).
    seed = abs(hash(sid)) % (2**32)
    fit_idx, holdout_idx = split_correspondences_for_tre(
        n, holdout_frac=TRE_HOLDOUT_FRAC, min_holdout=TRE_MIN_HOLDOUT,
        min_fit=MIN_MATCHES, seed=seed)

    if fit_idx is None:
        logger.info(f"[{sid}] Too few matches ({n}) for a held-out TRE split — skipping TRE.")
        result["tre_n_fit"] = 0
        result["tre_n_holdout"] = 0
        for m in METHODS:
            result[f"{m}_tre_mean_px"] = np.nan
            result[f"{m}_tre_median_px"] = np.nan
            result[f"{m}_tre_p90_px"] = np.nan
        return result

    result["tre_n_fit"] = len(fit_idx)
    result["tre_n_holdout"] = len(holdout_idx)
    src_fit, dst_fit = src_pts[fit_idx], dst_pts[fit_idx]
    src_ho,  dst_ho  = src_pts[holdout_idx], dst_pts[holdout_idx]

    for method_name, method_flag in METHODS.items():
        M, n_inliers_fit, _ = fit_affine(src_fit, dst_fit, method_flag, f"{sid}_tre", method_name)
        if M is None:
            result[f"{method_name}_tre_mean_px"] = np.nan
            result[f"{method_name}_tre_median_px"] = np.nan
            result[f"{method_name}_tre_p90_px"] = np.nan
            continue
        tre = compute_tre_affine(M, src_ho, dst_ho, px_size_um=PIXEL_SIZE_UM,
                                 image_diag_px=image_diag_px)
        result[f"{method_name}_tre_mean_px"]   = tre["mean_px"]
        result[f"{method_name}_tre_median_px"] = tre["median_px"]
        result[f"{method_name}_tre_p90_px"]    = tre["p90_px"]

    return result


def load_ck_pair(file_a, file_b):
    ck_a = tifffile.imread(file_a, key=CK_CHANNEL_IDX).astype(np.float32)
    ck_b = tifffile.imread(file_b, key=CK_CHANNEL_IDX).astype(np.float32)
    return ck_a, ck_b


def process_core(core_name, rows):
    input_dir = os.path.join(DATA_BASE_PATH, core_name)
    if not os.path.exists(input_dir):
        logger.error(f"[{core_name}] Input folder not found: {input_dir}")
        return

    sample_files = sorted(
        glob.glob(os.path.join(input_dir, "*.tif")) +
        glob.glob(os.path.join(input_dir, "*.tiff")),
        key=get_slice_number
    )
    file_list = [f for f in sample_files if "_thumb" not in os.path.basename(f)]

    allowed = load_slice_filter(SLICE_FILTER_YAML, core_name)
    if allowed is not None:
        file_list = [f for f in file_list if get_slice_number(f) in allowed]

    if len(file_list) < 2:
        logger.error(f"[{core_name}] Fewer than 2 slices after filtering — skipping.")
        return

    logger.info(f"[{core_name}] {len(file_list)} slices — {len(file_list)-1} adjacent pairs.")

    for i in range(len(file_list) - 1):
        file_a, file_b = file_list[i], file_list[i + 1]
        sid = f"{core_name}_Z{i:03d}-Z{i+1:03d}"

        ck_fixed, ck_moving = load_ck_pair(file_a, file_b)
        h, w = ck_fixed.shape
        fixed_mask  = load_mask_or_none(file_a, (h, w))
        moving_mask = load_mask_or_none(file_b, (h, w))

        _, fixed_log  = prepare_ck(ck_fixed)
        _, moving_log = prepare_ck(ck_moving)

        ncc_raw = measure_ncc(fixed_log.astype(np.float32),
                              moving_log.astype(np.float32), fixed_mask)

        t0 = time.time()
        src_pts, dst_pts, n_matches = detect_and_match(
            fixed_log, moving_log, fixed_mask, moving_mask, sid)
        fullres_detect_time_s = time.time() - t0

        row = dict(core=core_name, pair=sid, n_matches=n_matches, ncc_raw=ncc_raw,
                   fullres_detect_time_s=fullres_detect_time_s)

        image_diag_px = float(np.hypot(h, w))
        
        # Populate dummy counts for logging compatibility
        row["tre_n_fit"] = n_matches
        row["tre_n_holdout"] = 0

        # Step 3 — downsampled detection, full-res refit.
        ds_result = fit_affine_downsampled(
            fixed_log, moving_log, fixed_mask, moving_mask, sid)
        row["downsampled_scale"]          = ds_result["scale"]
        row["downsampled_n_matches"]      = ds_result["n_matches_ds"]
        row["downsampled_n_inliers"]      = ds_result["n_inliers_ds"]
        row["downsampled_detect_time_s"]  = ds_result["detect_time_s"]
        if ds_result["M"] is not None:
            moving_affine_ds = cv2.warpAffine(
                moving_log.astype(np.float32), ds_result["M"], (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            row["downsampled_ncc_affine"] = measure_ncc(
                fixed_log.astype(np.float32), moving_affine_ds, fixed_mask)
        else:
            row["downsampled_ncc_affine"] = np.nan

        if src_pts is None:
            for name in METHODS:
                row[f"{name}_n_inliers"] = 0
                row[f"{name}_ncc_affine"] = np.nan
                row[f"{name}_tre_mean_px"] = np.nan
                row[f"{name}_tre_median_px"] = np.nan
                row[f"{name}_tre_p90_px"] = np.nan
                row[f"{name}_tukey_n_inliers"] = 0
                row[f"{name}_tukey_ncc_affine"] = np.nan
                row[f"{name}_tukey_applied"] = False
            rows.append(row)
            continue

        for method_name, method_flag in METHODS.items():
            M, n_inliers, mask = fit_affine(src_pts, dst_pts, method_flag, sid, method_name)
            row[f"{method_name}_n_inliers"] = n_inliers
            row[f"{method_name}_tukey_n_inliers"] = np.nan
            row[f"{method_name}_tukey_ncc_affine"] = np.nan
            row[f"{method_name}_tukey_applied"] = False

            if M is None:
                row[f"{method_name}_ncc_affine"] = np.nan
                row[f"{method_name}_tre_mean_px"] = np.nan
                row[f"{method_name}_tre_median_px"] = np.nan
                row[f"{method_name}_tre_p90_px"] = np.nan
                continue

            # -----------------------------------------------------------------
            # PATH 1: Circular Inlier TRE Evaluation (Matches debug script)
            # Filter matches using this specific solver's surviving inlier mask
            # -----------------------------------------------------------------
            landmarks_fixed  = src_pts[mask.ravel() == 1].reshape(-1, 2)
            landmarks_moving = dst_pts[mask.ravel() == 1].reshape(-1, 2)
            
            tre = compute_tre_affine(M, landmarks_fixed, landmarks_moving,
                                     px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
            row[f"{method_name}_tre_mean_px"]   = tre["mean_px"]
            row[f"{method_name}_tre_median_px"] = tre["median_px"]
            row[f"{method_name}_tre_p90_px"]    = tre["p90_px"]
            # -----------------------------------------------------------------

            moving_affine = cv2.warpAffine(
                moving_log.astype(np.float32), M, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            ncc_affine = measure_ncc(fixed_log.astype(np.float32), moving_affine, fixed_mask)
            row[f"{method_name}_ncc_affine"] = ncc_affine

            # Step 2 — Tukey filter layered on top of this method's inliers
            M_tukey, n_kept, applied = fit_affine_tukey(
                src_pts, dst_pts, mask, M, sid, method_name)
            row[f"{method_name}_tukey_n_inliers"] = n_kept
            row[f"{method_name}_tukey_applied"] = applied
            moving_affine_tukey = cv2.warpAffine(
                moving_log.astype(np.float32), M_tukey, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            row[f"{method_name}_tukey_ncc_affine"] = measure_ncc(
                fixed_log.astype(np.float32), moving_affine_tukey, fixed_mask)

        rows.append(row)
        logger.info(
            f"[{sid}] matches={n_matches} | "
            f"RANSAC: {row['RANSAC_n_inliers']}→{row['RANSAC_tukey_n_inliers']} inliers "
            f"ncc={row.get('RANSAC_ncc_affine', float('nan')):.4f}→"
            f"{row.get('RANSAC_tukey_ncc_affine', float('nan')):.4f} "
            f"tre={row.get('RANSAC_tre_mean_px', float('nan')):.2f}px | "
            f"MAGSAC: {row['USAC_MAGSAC_n_inliers']}→{row['USAC_MAGSAC_tukey_n_inliers']} inliers "
            f"ncc={row.get('USAC_MAGSAC_ncc_affine', float('nan')):.4f}→"
            f"{row.get('USAC_MAGSAC_tukey_ncc_affine', float('nan')):.4f} "
            f"tre={row.get('USAC_MAGSAC_tre_mean_px', float('nan')):.2f}px | "
            f"Downsampled(scale={row['downsampled_scale']:.2f}): "
            f"inliers={row['downsampled_n_inliers']} ncc={row['downsampled_ncc_affine']:.4f} "
            f"[{row['fullres_detect_time_s']:.2f}s→{row['downsampled_detect_time_s']:.2f}s detect] "
            f"(TRE fit/holdout: {row['tre_n_fit']}/{row['tre_n_holdout']})"
        )


def summarize(df):
    print("\n" + "=" * 70)
    print("SUMMARY — RANSAC vs USAC_MAGSAC (ncc_affine, more negative = better)")
    print("=" * 70)
    valid = df.dropna(subset=["RANSAC_ncc_affine", "USAC_MAGSAC_ncc_affine"])
    print(f"Pairs total: {len(df)} | Pairs with both methods succeeding: {len(valid)}")

    if len(valid) == 0:
        print("No pairs with both methods succeeding — cannot compare.")
        return

    delta = valid["USAC_MAGSAC_ncc_affine"] - valid["RANSAC_ncc_affine"]  # negative = MAGSAC better
    n_better = int((delta < -1e-6).sum())
    n_worse  = int((delta > 1e-6).sum())
    n_tied   = len(valid) - n_better - n_worse

    print(f"MAGSAC better: {n_better} | worse: {n_worse} | tied: {n_tied}")
    print(f"Mean delta (MAGSAC - RANSAC): {delta.mean():.5f} (negative = MAGSAC better)")
    print(f"Median delta:                 {delta.median():.5f}")

    inlier_delta = valid["USAC_MAGSAC_n_inliers"] - valid["RANSAC_n_inliers"]
    print(f"Mean inlier count delta (MAGSAC - RANSAC): {inlier_delta.mean():.2f}")

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(valid["RANSAC_ncc_affine"], valid["USAC_MAGSAC_ncc_affine"])
        print(f"Wilcoxon signed-rank p-value: {p:.4f}")
    except ImportError:
        print("(scipy not available — install to get a Wilcoxon signed-rank p-value)")

    print("\n--- Marginal cases (bottom-quartile RANSAC inlier count) ---")
    q25 = valid["RANSAC_n_inliers"].quantile(0.25)
    tail = valid[valid["RANSAC_n_inliers"] <= q25]
    if len(tail) > 0:
        tail_delta = tail["USAC_MAGSAC_ncc_affine"] - tail["RANSAC_ncc_affine"]
        print(f"n={len(tail)} | mean delta: {tail_delta.mean():.5f} | "
              f"MAGSAC better: {int((tail_delta < -1e-6).sum())}/{len(tail)}")
    else:
        print("No low-inlier tail pairs found.")

    print("\n" + "-" * 70)
    print("SAME COMPARISON, BUT BY HELD-OUT TRE INSTEAD OF NCC (lower = better)")
    print("Same pairs, same solvers — fit on a held-out subset of matches,")
    print("evaluated on the REMAINING matches that were never used to fit.")
    print("This is the check for whether the NCC-based verdict above actually")
    print("reflects true geometric alignment accuracy, or just intensity")
    print("correlation on the CK channel used for gating.")
    print("-" * 70)
    tre_valid = df.dropna(subset=["RANSAC_tre_mean_px", "USAC_MAGSAC_tre_mean_px"])
    print(f"Pairs with both solvers' TRE available: {len(tre_valid)}")
    if len(tre_valid) > 0:
        tre_delta = tre_valid["USAC_MAGSAC_tre_mean_px"] - tre_valid["RANSAC_tre_mean_px"]
        tre_better = int((tre_delta < -1e-6).sum())
        tre_worse  = int((tre_delta > 1e-6).sum())
        tre_tied   = len(tre_valid) - tre_better - tre_worse
        print(f"MAGSAC lower TRE (better): {tre_better} | higher TRE (worse): {tre_worse} | tied: {tre_tied}")
        print(f"Mean TRE — RANSAC: {tre_valid['RANSAC_tre_mean_px'].mean():.2f}px | "
              f"MAGSAC: {tre_valid['USAC_MAGSAC_tre_mean_px'].mean():.2f}px")
        try:
            from scipy.stats import wilcoxon
            _, p_tre = wilcoxon(tre_valid["RANSAC_tre_mean_px"], tre_valid["USAC_MAGSAC_tre_mean_px"])
            print(f"Wilcoxon signed-rank p-value (TRE): {p_tre:.4f}")
        except ImportError:
            pass

        ncc_says_magsac_better = n_better > n_worse
        tre_says_magsac_better = tre_better > tre_worse
        if ncc_says_magsac_better == tre_says_magsac_better:
            print(">>> NCC and held-out TRE AGREE on direction — the NCC-based verdict "
                  "above is corroborated by independent geometric evidence.")
        else:
            print(">>> NCC and held-out TRE DISAGREE on direction — do not trust the "
                  "NCC-based verdict alone here. This is exactly the kind of case where "
                  "CK-channel intensity correlation and true landmark accuracy diverge.")
    else:
        print("No pairs with usable held-out TRE for both solvers (too few matches per pair, "
              "or MIN_MATCHES too close to the total match count — check TRE_HOLDOUT_FRAC/"
              "TRE_MIN_HOLDOUT if this is empty more often than expected).")

    print("\n" + "=" * 70)
    print("STEP 2 — Tukey residual filter, layered on each base solver")
    print("=" * 70)
    for method_name in METHODS:
        base_col  = f"{method_name}_ncc_affine"
        tukey_col = f"{method_name}_tukey_ncc_affine"
        applied_col = f"{method_name}_tukey_applied"
        sub = df.dropna(subset=[base_col, tukey_col])
        n_applied = int(sub[applied_col].sum()) if len(sub) else 0
        print(f"\n[{method_name}] pairs with both base+Tukey succeeding: {len(sub)} "
              f"(filter actually changed the fit on {n_applied} of them)")
        if len(sub) == 0:
            continue
        tdelta = sub[tukey_col] - sub[base_col]  # negative = Tukey better
        n_better = int((tdelta < -1e-6).sum())
        n_worse  = int((tdelta > 1e-6).sum())
        n_tied   = len(sub) - n_better - n_worse
        print(f"  Tukey better: {n_better} | worse: {n_worse} | tied: {n_tied}")
        print(f"  Mean delta (Tukey - base): {tdelta.mean():.5f} (negative = Tukey better)")
        applied_only = sub[sub[applied_col]]
        if len(applied_only) > 0:
            adelta = applied_only[tukey_col] - applied_only[base_col]
            print(f"  Restricted to pairs where filter actually fired (n={len(applied_only)}): "
                  f"mean delta {adelta.mean():.5f}")
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(sub[base_col], sub[tukey_col])
            print(f"  Wilcoxon signed-rank p-value: {p:.4f}")
        except ImportError:
            pass
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"STEP 3 — Downsampled AKAZE detection (max_dim={DOWNSAMPLE_MAX_DIM}px) "
          f"vs full-res RANSAC baseline")
    print("=" * 70)
    base_col = "RANSAC_ncc_affine"
    ds_col   = "downsampled_ncc_affine"
    sub = df.dropna(subset=[base_col, ds_col])
    print(f"Pairs with both full-res RANSAC and downsampled succeeding: {len(sub)} / {len(df)}")
    if len(sub) > 0:
        ddelta = sub[ds_col] - sub[base_col]  # negative = downsampled better
        n_better = int((ddelta < -1e-6).sum())
        n_worse  = int((ddelta > 1e-6).sum())
        n_tied   = len(sub) - n_better - n_worse
        print(f"Downsampled better: {n_better} | worse: {n_worse} | tied: {n_tied}")
        print(f"Mean delta (downsampled - full-res): {ddelta.mean():.5f} "
              f"(negative = downsampled better)")
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(sub[base_col], sub[ds_col])
            print(f"Wilcoxon signed-rank p-value: {p:.4f}")
        except ImportError:
            pass

    # Cases where full-res detection starved but downsampled still worked
    # (or vice versa) — worth knowing about even outside the NCC comparison.
    fullres_failed = df[df["RANSAC_ncc_affine"].isna()]
    if len(fullres_failed) > 0:
        rescued = fullres_failed["downsampled_ncc_affine"].notna().sum()
        print(f"\nFull-res RANSAC failed on {len(fullres_failed)} pair(s); "
              f"downsampled succeeded on {rescued} of those.")
    ds_failed = df[df["downsampled_ncc_affine"].isna()]
    if len(ds_failed) > 0:
        rescued_by_fullres = ds_failed["RANSAC_ncc_affine"].notna().sum()
        print(f"Downsampled failed on {len(ds_failed)} pair(s); "
              f"full-res RANSAC succeeded on {rescued_by_fullres} of those.")

    valid_time = df.dropna(subset=["fullres_detect_time_s", "downsampled_detect_time_s"])
    if len(valid_time) > 0:
        mean_full = valid_time["fullres_detect_time_s"].mean()
        mean_ds   = valid_time["downsampled_detect_time_s"].mean()
        speedup   = mean_full / mean_ds if mean_ds > 0 else float("inf")
        print(f"\nMean detection wall-time — full-res: {mean_full:.2f}s | "
              f"downsampled: {mean_ds:.2f}s | speedup: {speedup:.1f}x")
    print("=" * 70)


def main():
    global DOWNSAMPLE_MAX_DIM
    parser = argparse.ArgumentParser(description="Isolated RANSAC vs USAC_MAGSAC ablation test.")
    parser.add_argument("--core_name", type=str, action="append", default=[],
                        help="Explicit core folder name. Repeat flag for multiple cores, "
                             "e.g. --core_name coreA --core_name coreB. Combines with "
                             "--start/--end if both are given.")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index for Core_NN-style batch range (matches "
                             "run_all_akaze_roma_multi_channel_map.sh convention).")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (inclusive) for Core_NN-style batch range.")
    parser.add_argument("--downsample_max_dim", type=int, default=None,
                        help=f"Step-3 downsample target (px, longer side). "
                             f"Default: {DOWNSAMPLE_MAX_DIM}. Try a few values "
                             f"(e.g. 850, 1200, 1600) across separate runs to see "
                             f"how the NCC/speed tradeoff moves.")
    args = parser.parse_args()

    if args.downsample_max_dim is not None:
        DOWNSAMPLE_MAX_DIM = args.downsample_max_dim
        logger.info(f"Step 3 downsample max_dim overridden to {DOWNSAMPLE_MAX_DIM}px")

    core_names = list(args.core_name)
    if args.start is not None and args.end is not None:
        core_names += [f"Core_{i:02d}" for i in range(args.start, args.end + 1)]

    if not core_names:
        parser.error("Provide at least one --core_name, or both --start and --end.")

    total = len(core_names)
    rows  = []
    status = {}

    logger.info("=" * 60)
    logger.info(f"RANSAC vs USAC_MAGSAC ablation — {total} core(s)")
    logger.info("=" * 60)

    for idx, core_name in enumerate(core_names, start=1):
        logger.info(f"[{idx}/{total}] {core_name}")
        n_before = len(rows)
        try:
            process_core(core_name, rows)
            status[core_name] = "OK" if len(rows) > n_before else "NO_PAIRS"
        except Exception as exc:
            logger.error(f"[{core_name}] Crashed — skipping, continuing batch: {exc}")
            status[core_name] = "CRASHED"

    logger.info("-" * 60)
    logger.info("Per-core status:")
    for core_name in core_names:
        logger.info(f"    {core_name:<14} {status.get(core_name, 'UNKNOWN')}")
    n_ok = sum(1 for s in status.values() if s == "OK")
    logger.info(f"Cores with usable pairs: {n_ok}/{total}")
    logger.info("-" * 60)

    if not rows:
        logger.error("No pairs processed across any core — nothing to write.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Wrote {len(df)} rows → {OUTPUT_CSV}")

    summarize(df)


if __name__ == "__main__":
    main()