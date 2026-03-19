"""
Feature registration — RoMaV2 dense matching + Landmark-Constrained B-spline elastic.

Pipeline per slice pair:
  L0  RoMaV2 dense matching → RANSAC affine transform
  L1  Landmark-constrained B-spline FFD (SimpleITK) → elastic residual correction.
      RoMaV2 inlier correspondences (mapped to the affine-prealigned frame) are
      passed to SimpleITK as PointSetMetric landmarks that anchor control points
      near matched features.  The deformation is interpolated between anchors by
      the B-spline basis — no intensity optimizer is involved.

Why RoMaV2 over AKAZE for landmarks:
  - AKAZE requires salient keypoints (corners, blobs) — fails on uniform or
    weakly-textured tissue.
  - RoMaV2 produces dense DINOv3-based correspondences across the whole image,
    including low-texture regions where AKAZE finds nothing.
  - More landmarks → better-constrained B-spline → more accurate elastic correction.

Fallback:
  If RoMaV2 produces insufficient inliers or fails sanity check, the raw
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

# Redirect torch hub cache and hide GPU (CPU-only mode)
os.environ['TORCH_HOME']    = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # force CPU — remove if GPU is available

import torch
torch._dynamo.config.disable = True  # disable compile on CPU

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
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_RoMaV2_Landmark_BSpline")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM      = 0.4961
SECTION_THICKNESS_UM  = 4.5

# --- ROMAV2 CONFIGURATION ---
#
# ROMAV2_N_MATCHES   — dense matches sampled from the warp field.
#                      5000 gives a well-distributed set of landmarks.
# ROMAV2_MIN_OVERLAP — overlap confidence threshold [0,1].
#                      0.0 = keep all; raise to filter low-confidence matches.
# ROMAV2_DEVICE      — 'cuda' or 'cpu'.
# ROMAV2_H / W       — matching resolution (must be divisible by 14).
#                      448 needs ~2.5GB VRAM; use 392 on constrained GPUs.
# ROMAV2_H_HR / W_HR — high-res pass resolution. Set to None to disable
#                      (saves ~4GB VRAM; recommended unless GPU has headroom).
ROMAV2_N_MATCHES   = 5000
ROMAV2_MIN_OVERLAP = 0.0
ROMAV2_DEVICE      = 'cpu'    # change to 'cuda' when GPU is available
ROMAV2_H           = 448
ROMAV2_W           = 448
ROMAV2_H_HR        = None
ROMAV2_W_HR        = None

# --- RANSAC (applied to RoMaV2 matches) ---
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS  = 5000
RANSAC_THRESH     = 8.0

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0
MIN_INLIERS         = 6

# --- LANDMARK B-SPLINE (L1) ---
BSPLINE_GRID_NODES          = 8
BSPLINE_MAX_DISPLACEMENT_PX = 80.0
LANDMARK_MAX_PTS            = 1000   # max landmarks passed to B-spline fitter

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01


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
# RoMaV2 model (lazy load, singleton)
# ─────────────────────────────────────────────────────────────────────────────

_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        device = ROMAV2_DEVICE if (ROMAV2_DEVICE == 'cuda' and torch.cuda.is_available()) else 'cpu'
        if device == 'cpu':
            logger.warning("CUDA not available — RoMaV2 running on CPU. Expect slow matching.")
        logger.info(f"Loading RoMaV2 model on {device}...")
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


def ck_to_rgb_pil(ck_log: np.ndarray):
    """Convert log-normalised uint8 CK channel to RGB PIL image for RoMaV2."""
    from PIL import Image
    rgb = np.stack([ck_log, ck_log, ck_log], axis=-1)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────────────────
# L0 — RoMaV2 dense matching → RANSAC affine
# ─────────────────────────────────────────────────────────────────────────────

def romav2_affine(fixed_log: np.ndarray, moving_log: np.ndarray,
                  slice_id: str, orig_h: int, orig_w: int) -> tuple:
    """
    RoMaV2 dense matching + RANSAC affine.
    Returns (M_2x3|None, n_matches, n_inliers, kptsA, kptsB, inlier_mask).
    kptsA/kptsB are (N,2) float32 arrays in full-resolution pixel coordinates.
    """
    model  = get_romav2_model()
    img_A  = ck_to_rgb_pil(fixed_log)
    img_B  = ck_to_rgb_pil(moving_log)

    with torch.no_grad():
        preds = model.match(img_A, img_B)
        matches, certainties, _, _ = model.sample(preds, ROMAV2_N_MATCHES)

    if ROMAV2_MIN_OVERLAP > 0.0:
        keep        = certainties >= ROMAV2_MIN_OVERLAP
        matches     = matches[keep]
        certainties = certainties[keep]

    n_matches = int(matches.shape[0])
    logger.info(f"[{slice_id}] RoMaV2: {n_matches} matches after overlap filter.")

    if n_matches < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Too few RoMaV2 matches ({n_matches}).")
        return None, n_matches, 0, None, None, np.array([])

    kptsA, kptsB = model.to_pixel_coordinates(matches, orig_h, orig_w, orig_h, orig_w)
    kptsA = kptsA.cpu().numpy()   # (N,2) x,y
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
# L1 — Landmark-constrained B-spline elastic refinement
# ─────────────────────────────────────────────────────────────────────────────

def _thin_landmarks(fixed_pts: np.ndarray, moving_pts: np.ndarray,
                    max_pts: int) -> tuple:
    """
    Spatially thin landmark correspondences using ANMS-style suppression so
    that selected anchors are spread across the image rather than clustered.

    Parameters
    ----------
    fixed_pts  : (N, 2) float32  — destination points in fixed frame  (x, y)
    moving_pts : (N, 2) float32  — source points in (affine-prealigned) moving frame
    max_pts    : int             — max landmarks to keep

    Returns
    -------
    fixed_pts_thin, moving_pts_thin — each (M, 2), M <= max_pts
    """
    n = len(fixed_pts)
    if n <= max_pts:
        return fixed_pts, moving_pts

    # Use residual displacement magnitude as a proxy for "response" so that
    # points with larger residuals (where correction matters most) survive.
    residuals = np.linalg.norm(fixed_pts - moving_pts, axis=1)
    sort_idx  = np.argsort(residuals)[::-1]          # highest residual first

    pts_sorted = fixed_pts[sort_idx]
    radii      = np.full(n, np.inf)
    for i in range(1, n):
        diffs    = pts_sorted[:i] - pts_sorted[i]
        radii[i] = np.min(np.sum(diffs ** 2, axis=1))

    best = np.argsort(radii)[::-1][:max_pts]
    sel  = sort_idx[best]
    return fixed_pts[sel], moving_pts[sel]


def bspline_landmark(fixed_8bit: np.ndarray, moving_8bit: np.ndarray,
                     h: int, w: int, slice_id: str,
                     fixed_pts: np.ndarray | None,
                     moving_pts: np.ndarray | None):
    """
    Landmark-only B-spline FFD registration (no intensity optimizer).

    Uses sitk.LandmarkBasedTransformInitializer to fit a B-spline purely from
    AKAZE inlier correspondences in the affine-prealigned frame.  The deformation
    is interpolated between anchors by the B-spline basis — no NCC or any other
    intensity metric is involved.

    Well-aligned cores get near-identity fields; heavily deformed cores get
    large-but-valid fields wherever inliers exist.  Quality is bounded by the
    feature matches, not by intensity optimizer freedom.

    Fallback:
        If fewer than 4 landmarks are available, returns (None, None) so the
        caller falls back to affine-only output.

    Parameters
    ----------
    fixed_8bit  : uint8 CK image — used only to define the image domain.
    moving_8bit : uint8 CK image — accepted for API consistency, not consumed.
    h, w        : image dimensions
    slice_id    : logging label
    fixed_pts   : (N, 2) float32 matched points in fixed frame (x, y)
    moving_pts  : (N, 2) float32 matched points in affine-prealigned moving frame

    Returns
    -------
    (remap_x, remap_y) float32 for cv2.remap, or (None, None) on failure.
    """
    if fixed_pts is None or moving_pts is None or len(fixed_pts) < 4:
        logger.warning(f"[{slice_id}] No landmarks available — skipping landmark B-spline.")
        return None, None

    try:
        sitk_fixed  = sitk.GetImageFromArray(fixed_8bit.astype(np.float32))

        # ── Landmark-only B-spline fit ────────────────────────────────────────
        fp, mp = _thin_landmarks(fixed_pts, moving_pts, LANDMARK_MAX_PTS)
        n_lm   = len(fp)

        # SimpleITK expects landmarks as flat [x0,y0, x1,y1, ...] lists.
        # Coordinates are in pixel space (spacing = 1,1 → physical = pixel).
        fixed_lm_flat  = [float(v) for pt in fp for v in pt]
        moving_lm_flat = [float(v) for pt in mp for v in pt]

        # Per-landmark weight: scale by residual magnitude so that larger
        # mismatches are pulled harder during the spline fit.
        residuals  = np.linalg.norm(fp - mp, axis=1).tolist()
        r_max      = max(residuals) if max(residuals) > 0 else 1.0
        lm_weights = [r / r_max for r in residuals]

        # numberOfControlPoints in LandmarkBasedTransformInitializer is the
        # number of control points per dimension.
        bspline_proto = sitk.BSplineTransformInitializer(
            sitk_fixed,
            transformDomainMeshSize=[BSPLINE_GRID_NODES] * 2,
            order=3,
        )
        tx_init = sitk.LandmarkBasedTransformInitializer(
            bspline_proto,
            fixedLandmarks=fixed_lm_flat,
            movingLandmarks=moving_lm_flat,
            landmarkWeight=lm_weights,
            referenceImage=sitk_fixed,
            numberOfControlPoints=BSPLINE_GRID_NODES,
        )
        logger.info(
            f"[{slice_id}] Landmark B-spline fitted "
            f"({n_lm} anchors, nodes={BSPLINE_GRID_NODES})."
        )

        # ── Displacement field → cv2 remap arrays ─────────────────────────────
        # Stage A is the final transform — no intensity optimizer.
        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(sitk_fixed)
        disp_field  = disp_filter.Execute(tx_init)
        disp_field  = sitk.Cast(disp_field, sitk.sitkVectorFloat64)

        disp_np      = sitk.GetArrayFromImage(disp_field)  # (H, W, 2)
        map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Hard cap on displacement magnitude (post-hoc regularisation)
        disp_magnitude = np.sqrt(disp_np[..., 0]**2 + disp_np[..., 1]**2)
        excess         = disp_magnitude > BSPLINE_MAX_DISPLACEMENT_PX
        if np.any(excess):
            scale             = np.where(
                excess, BSPLINE_MAX_DISPLACEMENT_PX / (disp_magnitude + 1e-8), 1.0
            )
            disp_np[..., 0]  *= scale
            disp_np[..., 1]  *= scale
            logger.info(
                f"[{slice_id}] Clipped {int(excess.sum())} displacement vectors "
                f"> {BSPLINE_MAX_DISPLACEMENT_PX}px."
            )

        remap_x = (map_x + disp_np[..., 0]).astype(np.float32)
        remap_y = (map_y + disp_np[..., 1]).astype(np.float32)

        return remap_x, remap_y

    except Exception as exc:
        logger.warning(f"[{slice_id}] Landmark B-spline failed ({exc}) — skipping elastic layer.")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main registration function
# ─────────────────────────────────────────────────────────────────────────────

def register_slice(fixed_np, moving_np, slice_id=None):
    """
    Register one moving slice to its fixed neighbour.
    Returns (aligned_np, mse, elapsed, stats, romav2_ok, M_affine).
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # ── L0: RoMaV2 dense matching → RANSAC affine ────────────────────────────
    M_affine, n_matches, n_inliers, kptsA, kptsB, inlier_mask = \
        romav2_affine(fixed_log, moving_log, sid, h, w)
    romav2_ok = M_affine is not None
    if not romav2_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # ── Match plot ────────────────────────────────────────────────────────────
    try:
        save_matching_pairs_plot(
            fixed_log, moving_log, kptsA, kptsB, inlier_mask,
            n_inliers, sid, OUTPUT_FOLDER, romav2_ok,
        )
    except Exception as exc:
        logger.warning(f"[{sid}] Match plot failed: {exc}")

    # ── L1: Landmark-constrained B-spline (only when L0 gave valid pre-alignment) ──
    remap_x = remap_y = None
    if romav2_ok:
        moving_prealigned = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

        # Build residual landmark correspondences in the affine-prealigned frame.
        # RoMaV2 inliers are already in full-resolution pixel coords (x, y).
        fixed_pts_lm  = None
        moving_pts_lm = None

        if inlier_mask is not None and len(inlier_mask) > 0 and n_inliers >= 4:
            inlier_idx = np.where(inlier_mask.ravel())[0]
            fp       = kptsA[inlier_idx].astype(np.float32)   # fixed frame (x,y)
            mp_orig  = kptsB[inlier_idx].astype(np.float32)   # original moving frame

            # Apply affine to moving points → affine-prealigned frame
            ones      = np.ones((len(mp_orig), 1), dtype=np.float32)
            mp_h      = np.hstack([mp_orig, ones])
            mp_warped = (M_affine @ mp_h.T).T   # (N,2) x,y

            fixed_pts_lm  = fp
            moving_pts_lm = mp_warped
            logger.info(
                f"[{sid}] Passing {len(fp)} RoMaV2 residual landmarks to B-spline "
                f"(max={LANDMARK_MAX_PTS})."
            )

        remap_x, remap_y = bspline_landmark(
            fixed_log, moving_prealigned, h, w, sid,
            fixed_pts_lm, moving_pts_lm,
        )

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
        detector     = "RoMaV2" if romav2_ok else "Identity",
        n_matches    = n_matches,
        n_inliers    = n_inliers,
        rotation_deg = round(rot, 3),
        tx           = round(float(M_affine[0, 2]), 3),
        ty           = round(float(M_affine[1, 2]), 3),
        scale_pct    = round(scale_pct, 3),
        shear_pct    = round(shear_pct, 3),
        bspline_ok   = remap_x is not None,
    )

    return aligned_np, mse, time.time() - start, stats, romav2_ok, M_affine


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


def save_matching_pairs_plot(fixed_log, moving_log, kptsA, kptsB,
                              inlier_mask, n_inliers,
                              slice_id, output_folder, romav2_ok=True):
    """
    Saves one PNG per slice pair: RoMaV2 RANSAC inlier matches with rainbow lines.
    Fixed image on the left (green markers), moving on the right (blue markers).
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    h, w = fixed_log.shape
    gap  = 6
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_log,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_log, cv2.COLOR_GRAY2BGR)

    status      = "SUCCESS" if romav2_ok else "FAILED"
    title_color = (0, 230, 0) if romav2_ok else (0, 0, 220)

    if kptsA is not None and inlier_mask is not None and len(inlier_mask) > 0:
        inlier_idx = np.where(inlier_mask.ravel())[0][:200]
        for i, idx in enumerate(inlier_idx):
            hue   = int(i / max(len(inlier_idx) - 1, 1) * 179)
            color = tuple(int(c) for c in
                          cv2.cvtColor(np.uint8([[[hue, 220, 220]]]),
                                       cv2.COLOR_HSV2BGR)[0, 0])
            ptA = (int(kptsA[idx, 0]), int(kptsA[idx, 1]))
            ptB = (int(kptsB[idx, 0]) + w + gap, int(kptsB[idx, 1]))
            cv2.line(canvas, ptA, ptB, color, 1, cv2.LINE_AA)
            _draw_keypoint_marker(canvas, ptA, (0, 200, 0))
            _draw_keypoint_marker(canvas, ptB, (0, 0, 200))

    _burn_title(canvas, f"{slice_id}  RoMaV2 inliers={n_inliers}  [{status}]",
                color=title_color)
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_romav2_inliers.png"), canvas)
    logger.info(f"[{slice_id}] RoMaV2 inlier plot saved ({n_inliers} inliers).")


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
    logger.info(f"Feature Registration (RoMaV2 + Landmark-constrained B-spline) — {TARGET_CORE}")
    logger.info(f"RoMaV2 H={ROMAV2_H} W={ROMAV2_W} | B-spline nodes={BSPLINE_GRID_NODES} | "
                f"Landmark max pts={LANDMARK_MAX_PTS}")

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
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Landmark_Aligned.ome.tif")
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

    # Pre-load RoMaV2 model before registration starts
    get_romav2_model()

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)

    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical full-res shape: C={c}, H={target_h}, W={target_w}")

    # Collect slice IDs only — load slices on demand to avoid OOM
    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
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
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
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
            moving_np = load_slice(i)
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            aligned_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = moving_np
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            del moving_np

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
        os.path.join(OUTPUT_FOLDER, "registration_stats_RoMaV2_Landmark_BSpline.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(f"Execution complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fallback}")

    # Write volume first, then montage
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_RoMaV2_Landmark_Aligned.ome.tif")
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

    logger.info("Generating QC montage...")
    try:
        generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="RoMaV2+Landmark_BSpline")
    except Exception as exc:
        logger.error(f"Montage failed: {exc}")
    del aligned_vol

    logger.info("Done.")


if __name__ == "__main__":
    main()