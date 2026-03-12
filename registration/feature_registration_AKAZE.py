"""
Feature registration — Stage 1: AKAZE affine only.

Pipeline per slice pair:
  L0  AKAZE + ANMS + RANSAC at full resolution → affine transform

Fallback:
  If AKAZE produces insufficient inliers or fails sanity check, the raw
  moving slice is written to protect the chain.

Output:
  - Interim AKAZE-aligned stack  : <OUTPUT_FOLDER>/<CORE>_AKAZE_Aligned.ome.tif
  - QC montage (CK channel)      : <OUTPUT_FOLDER>/<CORE>_QC_Montage_CK_AKAZE.png
  - CSV stats                    : <OUTPUT_FOLDER>/registration_stats_AKAZE.csv
      Includes per-slice tile NCC diagnostic columns:
        NCC_Mean  — mean NCC across 4×4 tiles (higher = better global alignment)
        NCC_Min   — worst-tile NCC (low value flags a locally misaligned region)
        NCC_Std   — std of tile NCC (high value = spatially non-uniform residual,
                    primary indicator that elastic refinement may help)
        NCC_Tiles — per-tile values in row-major order (NaN for blank tiles)

Run this script first, then pass its output to feature_registration_BSpline_NCC.py.
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
parser = argparse.ArgumentParser(description='Feature registration — Stage 1: AKAZE affine.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Feature_AKAZE_BSpline_NCC_2Stage")
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
MAX_SHEAR           = 0.15  # max singular value ratio − 1
MAX_ROTATION_DEG    = 15.0  # hard rotation gate


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
    Returns (M_2x3 | None, n_matches, n_inliers,
             inlier_pts_fixed, inlier_pts_moving)
    where inlier_pts_* are (N,2) float32 arrays of inlier coordinates,
    or (None, None) when registration fails.
    """
    detector  = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1) if kp1 else 0}, moving={len(kp2) if kp2 else 0})."
        )
        return None, 0, 0, None, None

    kp1, des1 = apply_anms(kp1, des1, num_to_keep=ANMS_KEEP)
    kp2, des2 = apply_anms(kp2, des2, num_to_keep=ANMS_KEEP)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0, None, None

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, None, None

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, len(good), n_inliers, None, None

    inlier_mask        = mask.ravel().astype(bool)
    inlier_pts_fixed   = src_pts[inlier_mask].reshape(-1, 2)   # coords on fixed image
    inlier_pts_moving  = dst_pts[inlier_mask].reshape(-1, 2)   # coords on moving image

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, inlier_pts_fixed, inlier_pts_moving


# ─────────────────────────────────────────────────────────────────────────────
# Register one slice (AKAZE affine only)
# ─────────────────────────────────────────────────────────────────────────────

def register_slice_akaze(fixed_np, moving_np, slice_id=None):
    """
    Register one moving slice to its fixed neighbour using AKAZE affine only.
    Returns (aligned_np, mse, elapsed, stats, akaze_ok, M_affine,
             inlier_pts_fixed, inlier_pts_moving, fixed_log, moving_log).
    inlier_pts_* and *_log are used downstream for match visualisation.
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # ── L0: AKAZE affine ─────────────────────────────────────────────────────
    M_affine, n_matches, n_inliers, ipts_fixed, ipts_moving = akaze_affine(
        fixed_log, moving_log, sid
    )
    akaze_ok = M_affine is not None
    if not akaze_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # ── Apply affine to all channels ──────────────────────────────────────────
    aligned_np = np.stack(
        [cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
         ) for ch in range(fixed_np.shape[0])],
        axis=0,
    ).astype(np.uint16)

    # ── MSE on CK channel ─────────────────────────────────────────────────────
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
    )

    return (aligned_np, mse, time.time() - start, stats, akaze_ok, M_affine,
            ipts_fixed, ipts_moving, fixed_log, moving_log)


# ─────────────────────────────────────────────────────────────────────────────
# QC montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_montage(vol, output_folder, channel_idx=6, channel_name="CK", stage="AKAZE"):
    logger.info(f"Generating QC montage ({stage}) for {channel_name} channel.")
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

    plt.suptitle(f'Registration QC {stage}: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_{stage}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Deformation diagnostic — tile-wise NCC on aligned pair
# ─────────────────────────────────────────────────────────────────────────────

def tile_ncc_diagnostic(fixed_np, aligned_np, channel_idx=6, grid=(4, 4)):
    """
    Divide the CK channel of an aligned slice pair into a grid of tiles and
    compute NCC per tile.  Returns a dict of summary statistics:
        ncc_mean, ncc_min, ncc_std, ncc_tiles (flat list, row-major)

    High ncc_std or low ncc_min indicates spatially non-uniform residual
    misalignment — a signal that elastic refinement may be warranted.
    NCC is computed on log-normalised uint8 (same space used for AKAZE).
    """
    fixed_ck   = fixed_np[channel_idx].astype(np.float32)
    aligned_ck = aligned_np[channel_idx].astype(np.float32)

    _, fixed_log   = prepare_ck(fixed_ck)
    _, aligned_log = prepare_ck(aligned_ck)

    h, w     = fixed_log.shape
    rows, cols = grid
    tile_h   = h // rows
    tile_w   = w // cols
    ncc_vals = []

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            t_fix = fixed_log[y0:y1, x0:x1].astype(np.float64)
            t_mov = aligned_log[y0:y1, x0:x1].astype(np.float64)

            # Skip tiles that are mostly blank in either image
            if t_fix.std() < 1e-3 or t_mov.std() < 1e-3:
                ncc_vals.append(float('nan'))
                continue

            t_fix -= t_fix.mean(); t_mov -= t_mov.mean()
            denom  = (np.linalg.norm(t_fix) * np.linalg.norm(t_mov))
            ncc    = float(np.sum(t_fix * t_mov) / denom) if denom > 0 else float('nan')
            ncc_vals.append(round(ncc, 4))

    valid = [v for v in ncc_vals if not np.isnan(v)]
    return {
        "ncc_mean":  round(float(np.mean(valid)),  4) if valid else float('nan'),
        "ncc_min":   round(float(np.min(valid)),   4) if valid else float('nan'),
        "ncc_std":   round(float(np.std(valid)),   4) if valid else float('nan'),
        "ncc_tiles": ncc_vals,          # full grid, NaN for blank tiles
    }


# ─────────────────────────────────────────────────────────────────────────────
# Match pairs montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_match_montage(match_records, output_folder):
    """
    One panel per consecutive slice pair showing the fixed image (left) and
    moving image (right) side-by-side with inlier match lines drawn between them.
    match_records: list of dicts with keys:
        z_fixed, z_moving, fixed_log, moving_log,
        ipts_fixed, ipts_moving, n_matches, n_inliers, status
    """
    logger.info("Generating match pairs montage.")
    n       = len(match_records)
    n_cols  = 3
    n_rows  = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1:               axes = axes.reshape(1, -1)
    elif n_cols == 1:               axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    # Subsample drawn lines so the plot stays readable
    MAX_LINES = 80

    for idx, rec in enumerate(match_records):
        ax = axes_flat[idx]

        f_img = rec["fixed_log"]
        m_img = rec["moving_log"]
        h, w  = f_img.shape

        # Stack side-by-side
        canvas = np.zeros((h, w * 2), dtype=np.uint8)
        canvas[:, :w]  = f_img
        canvas[:, w:]  = m_img
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=255, aspect="auto")

        ipts_f = rec["ipts_fixed"]
        ipts_m = rec["ipts_moving"]

        if ipts_f is not None and len(ipts_f) > 0:
            n_draw = min(MAX_LINES, len(ipts_f))
            idx_s  = np.linspace(0, len(ipts_f) - 1, n_draw, dtype=int)
            cmap   = plt.cm.plasma(np.linspace(0, 1, n_draw))
            for k, ci in zip(idx_s, cmap):
                x1, y1 = ipts_f[k]
                x2, y2 = ipts_m[k]
                ax.plot([x1, x2 + w], [y1, y2],
                        color=ci, linewidth=0.6, alpha=0.7)
            ax.scatter(ipts_f[idx_s, 0], ipts_f[idx_s, 1],
                       c="lime", s=6, linewidths=0)
            ax.scatter(ipts_m[idx_s, 0] + w, ipts_m[idx_s, 1],
                       c="red",  s=6, linewidths=0)

        # Dividing line between the two images
        ax.axvline(x=w, color="white", linewidth=0.8, alpha=0.6)

        color  = "lime" if rec["status"] == "SUCCESS" else "tomato"
        title  = (f"Z{rec['z_fixed']}→Z{rec['z_moving']}  "
                  f"inliers={rec['n_inliers']}  [{rec['status']}]")
        ax.set_title(title, fontsize=8, fontweight='bold', color=color)
        ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle(f"AKAZE Inlier Matches: {TARGET_CORE}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f"{TARGET_CORE}_Match_Pairs_AKAZE.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Match pairs montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# NCC tile heatmap montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_ncc_heatmap_montage(ncc_records, output_folder, grid=(4, 4)):
    """
    One heatmap panel per consecutive slice pair showing the 4×4 NCC grid
    colour-coded from red (poor alignment) through yellow to green (good).
    ncc_records: list of dicts with keys:
        z_fixed, z_moving, ncc_tiles, ncc_mean, ncc_min, ncc_std, status
    """
    logger.info("Generating NCC tile heatmap montage.")
    n       = len(ncc_records)
    n_cols  = 5
    n_rows  = (n + n_cols - 1) // n_cols
    rows, cols = grid

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1:               axes = axes.reshape(1, -1)
    elif n_cols == 1:               axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    for idx, rec in enumerate(ncc_records):
        ax     = axes_flat[idx]
        tiles  = rec["ncc_tiles"]                          # flat list, NaN for blank
        grid_2d = np.array(
            [t if t is not None and not (isinstance(t, float) and np.isnan(t)) else np.nan
             for t in tiles],
            dtype=np.float64,
        ).reshape(rows, cols)

        # Mask NaN (blank) tiles
        masked = np.ma.masked_invalid(grid_2d)
        cmap   = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="#333333")   # dark grey for blank tiles

        im = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")

        # Annotate each cell with its NCC value
        for r in range(rows):
            for c in range(cols):
                val = grid_2d[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black", fontweight="bold")
                else:
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=6, color="white")

        ax.set_xticks([]); ax.set_yticks([])

        color = "lime" if rec["status"] == "SUCCESS" else "tomato"
        title = (f"Z{rec['z_fixed']}→Z{rec['z_moving']}\n"
                 f"μ={rec['ncc_mean']:.3f}  min={rec['ncc_min']:.3f}  "
                 f"σ={rec['ncc_std']:.3f}")
        ax.set_title(title, fontsize=7, color=color)

    # Shared colourbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=0.5, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Tile NCC")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle(f"Post-AKAZE Tile NCC Heatmap: {TARGET_CORE}",
                 fontsize=14, fontweight='bold')
    out_path = os.path.join(output_folder,
                            f"{TARGET_CORE}_NCC_Heatmap_AKAZE.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"NCC heatmap montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Feature Registration Stage 1: AKAZE affine — {TARGET_CORE}")
    logger.info(f"AKAZE threshold={AKAZE_THRESHOLD} | ANMS keep={ANMS_KEEP}")

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_Aligned.ome.tif")
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
    match_records      = []   # for match pairs montage
    ncc_records        = []   # for NCC heatmap montage

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            (aligned_np, mse, runtime, stats, success, M_final,
             ipts_fixed, ipts_moving, fixed_log, moving_log) = register_slice_akaze(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = raw_slices[i]
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            # ── Deformation diagnostic on the aligned pair ────────────────────
            diag = tile_ncc_diagnostic(
                aligned_vol[i + fixed_offset], aligned_vol[i],
                channel_idx=CK_CHANNEL_IDX, grid=(4, 4),
            )

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"MSE: {mse:.2f} | Rot: {stats['rotation_deg']:.2f} | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
                f"Scale: {stats['scale_pct']:+.2f}% | Shear: {stats['shear_pct']:.2f}% | "
                f"NCC_mean: {diag['ncc_mean']:.4f} | NCC_min: {diag['ncc_min']:.4f} | "
                f"NCC_std: {diag['ncc_std']:.4f} | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":    direction,
                "Slice_Z":      i,
                "Slice_ID":     real_id,
                "Detector":     stats["detector"],
                "N_Matches":    stats["n_matches"],
                "N_Inliers":    stats["n_inliers"],
                "Success":      success,
                "Status":       status_str,
                "MSE_After":    round(mse, 4),
                "Rotation_Deg": stats["rotation_deg"],
                "Shift_X_px":   stats["tx"],
                "Shift_Y_px":   stats["ty"],
                "Scale_Pct":    stats["scale_pct"],
                "Shear_Pct":    stats["shear_pct"],
                "NCC_Mean":     diag["ncc_mean"],
                "NCC_Min":      diag["ncc_min"],
                "NCC_Std":      diag["ncc_std"],
                "NCC_Tiles":    str(diag["ncc_tiles"]),
                "Runtime_s":    round(runtime, 3),
            })

            # ── Accumulate visualisation records ─────────────────────────────
            z_fixed = i + fixed_offset
            match_records.append({
                "z_fixed":    z_fixed,
                "z_moving":   i,
                "fixed_log":  fixed_log,
                "moving_log": moving_log,
                "ipts_fixed":  ipts_fixed,
                "ipts_moving": ipts_moving,
                "n_matches":  stats["n_matches"],
                "n_inliers":  stats["n_inliers"],
                "status":     status_str,
            })
            ncc_records.append({
                "z_fixed":  z_fixed,
                "z_moving": i,
                "ncc_tiles": diag["ncc_tiles"],
                "ncc_mean":  diag["ncc_mean"],
                "ncc_min":   diag["ncc_min"],
                "ncc_std":   diag["ncc_std"],
                "status":    status_str,
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = [
        "Direction", "Slice_Z", "Slice_ID", "Detector",
        "N_Matches", "N_Inliers",
        "Success", "Status", "Rotation_Deg",
        "Shift_X_px", "Shift_Y_px", "Scale_Pct", "Shear_Pct",
        "MSE_After",
        "NCC_Mean", "NCC_Min", "NCC_Std", "NCC_Tiles",
        "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"AKAZE pass complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    # ── QC montage for AKAZE-only result ─────────────────────────────────────
    generate_qc_montage(aligned_vol, OUTPUT_FOLDER,
                        channel_idx=CK_CHANNEL_IDX, channel_name="CK", stage="AKAZE")

    # ── Match pairs montage ───────────────────────────────────────────────────
    match_records_sorted = sorted(match_records, key=lambda r: r["z_moving"])
    generate_match_montage(match_records_sorted, OUTPUT_FOLDER)

    # ── NCC tile heatmap montage ──────────────────────────────────────────────
    ncc_records_sorted = sorted(ncc_records, key=lambda r: r["z_moving"])
    generate_ncc_heatmap_montage(ncc_records_sorted, OUTPUT_FOLDER, grid=(4, 4))

    # ── Write interim AKAZE-aligned stack ────────────────────────────────────
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_Aligned.ome.tif")
    logger.info(f"Writing AKAZE-aligned volume to {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='deflate',
                     compressionargs={'level': 6})
    logger.info(f"Stage 1 complete. Pass this stack to feature_registration_BSpline_NCC.py.")


if __name__ == "__main__":
    main()