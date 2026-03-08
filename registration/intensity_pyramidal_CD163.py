"""
Intensity-based registration as an ALTERNATIVE to feature_pyramidal_ck.py.
Reads raw .ome.tif slices at FULL resolution.

Strategy — coarse-to-fine, 3 pyramid levels:
  L0 (0.25x) — Phase correlation (FFT-based):
                Estimates global translation quickly and robustly.
                Dense signal like DAPI makes this very reliable.
  L1 (0.50x) — ECC (Enhanced Correlation Coefficient, cv2.findTransformECC):
                Refines with a full affine model. Maximises normalised
                cross-correlation; robust to intensity differences between slices.
  L2 (1.00x) — ECC again at full resolution for sub-pixel accuracy.

Why intensity vs feature for DAPI:
  - DAPI fills the entire core — no sparse signal problem
  - Phase correlation captures the dominant global shift in a single FFT
  - ECC is inherently sub-pixel and doesn't need keypoints or descriptors
  - No RANSAC needed: the signal is dense enough that outlier rejection is
    handled by the iterative ECC optimisation itself

Output is identical in format to feature_pyramidal_ck.py:
  <TARGET_CORE>_Intensity_Aligned.ome.tif  (ZCYX, uint16, zlib)
  registration_stats_Intensity_pyramid.csv
  QC montage (DAPI channel overlay)
  Diagnostics_Pyramid_Intensity/  (overlay PNGs per slice per level)

Usage:
  python intensity_pyramidal_dapi.py --core_name TMA_Core_01
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
parser = argparse.ArgumentParser(description='Intensity registration — pyramidal, full resolution output.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Intensity_pyramidal_CD163")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# --- PYRAMID ---
PYRAMID_SCALES = [0.25, 0.50, 1.0]

# --- CHANNEL ---
# DAPI is the best channel for intensity registration — dense, nuclear, always present.
# Change to 1–5 to experiment with other markers.
REG_CHANNEL_IDX  = 5
REG_CHANNEL_NAME = "CD163"

# --- PHASE CORRELATION (L0 only) ---
# Upsampling factor for sub-pixel peak localisation.
# 10 = 0.1px precision; higher = slower but more accurate.
PHASE_UPSAMPLE = 10

# --- ECC (L1 and L2) ---
# Motion model: cv2.MOTION_AFFINE allows rotation + translation + scale + shear.
# Use cv2.MOTION_EUCLIDEAN if you only expect rotation + translation.
ECC_MOTION_MODEL = cv2.MOTION_AFFINE
ECC_MAX_ITERS    = 200     # iterations per ECC call
ECC_EPS          = 1e-5    # convergence criterion
ECC_GAUSS_FILT   = 5       # Gaussian smoothing kernel size before ECC (0 = off)
                            # Smoothing reduces noise sensitivity — recommended for IF data

# --- SANITY GATE — same bounds as feature script ---
MAX_ROTATION_DEG   = 15.0
MAX_TRANSLATION_PX = 300.0   # at full resolution
MAX_SCALE_DEVIATION = 0.08   # max abs deviation from 1.0 in any singular value

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (identical to feature script)
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
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def prepare_intensity(img_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (norm_linear_uint8, norm_float32).
    norm_linear_uint8 : for MSE / QC overlay
    norm_float32      : for ECC (needs float input; CLAHE for better contrast)
    """
    # Linear uint8 — for MSE / overlay
    p_lo, p_hi  = np.percentile(img_arr[::4, ::4], (1, 99.9))
    clipped     = np.clip(img_arr, p_lo, p_hi)
    norm_u8     = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE float32 — for ECC; improves convergence on low-contrast regions
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_clahe  = clahe.apply(norm_u8).astype(np.float32) / 255.0

    return norm_u8, norm_clahe


# ─────────────────────────────────────────────────────────────────────────────
# Sanity gate  (identical logic to feature script)
# ─────────────────────────────────────────────────────────────────────────────

def transform_is_sane(M: np.ndarray) -> bool:
    """Returns True if rotation, translation, and scale are within physical bounds."""
    U, S, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx, ty   = abs(M[0, 2]), abs(M[1, 2])
    scale_ok = np.all(np.abs(S - 1.0) <= MAX_SCALE_DEVIATION)
    return rot_deg <= MAX_ROTATION_DEG and tx <= MAX_TRANSLATION_PX \
           and ty <= MAX_TRANSLATION_PX and scale_ok


def decompose_M(M: np.ndarray) -> tuple[float, float, float]:
    """Returns (rotation_deg, tx, ty) for logging."""
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot      = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot, float(M[0, 2]), float(M[1, 2])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — overlay PNG per level
# ─────────────────────────────────────────────────────────────────────────────

def save_intensity_diagnostics(output_dir, slice_id, level_label, level_idx,
                                fixed_u8, warped_u8):
    """Saves a red/green overlay of fixed vs warped moving image."""
    os.makedirs(output_dir, exist_ok=True)
    h = min(fixed_u8.shape[0], warped_u8.shape[0])
    w = min(fixed_u8.shape[1], warped_u8.shape[1])
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = fixed_u8[:h, :w]   # green = fixed
    overlay[:, :, 2] = warped_u8[:h, :w]  # blue  = moving (warped)
    overlay[:, :, 0] = 0
    # Blend so misalignment shows as colour fringe
    cv2.imwrite(
        os.path.join(output_dir,
                     f"{slice_id}_{level_idx+1:02d}_Overlay_{level_label}.png"),
        overlay,
    )


# ─────────────────────────────────────────────────────────────────────────────
# L0 — Phase correlation (translation only)
# ─────────────────────────────────────────────────────────────────────────────

def phase_correlation_translation(fixed_f32: np.ndarray,
                                   moving_f32: np.ndarray,
                                   level_scale: float) -> np.ndarray | None:
    """
    Estimates a pure translation using FFT phase correlation.
    Returns a 2x3 affine matrix in FULL-RESOLUTION coordinates, or None on failure.
    """
    # cv2.phaseCorrelate expects float32
    (dx, dy), response = cv2.phaseCorrelate(fixed_f32, moving_f32)

    logger.info(f"  [PhaseCorr L0] dx={dx:.2f}px dy={dy:.2f}px response={response:.4f}")

    if response < 0.01:
        logger.warning(f"  [PhaseCorr L0] Very low response ({response:.4f}) — likely noise.")
        return None

    # Scale translation back to full resolution
    M = np.array([[1.0, 0.0, dx / level_scale],
                  [0.0, 1.0, dy / level_scale]], dtype=np.float64)
    return M


# ─────────────────────────────────────────────────────────────────────────────
# L1 / L2 — ECC refinement
# ─────────────────────────────────────────────────────────────────────────────

def ecc_refinement(fixed_f32: np.ndarray, moving_f32: np.ndarray,
                   M_init_fullres: np.ndarray, level_scale: float,
                   level_idx: int) -> np.ndarray | None:
    """
    Refines M_init using ECC optimisation.
    M_init_fullres is the current accumulated transform in full-res coordinates.
    Returns refined 2x3 affine in FULL-RESOLUTION coordinates, or None on failure.
    """
    name = f"ECC_L{level_idx}_s{level_scale:.2f}"

    # Scale the accumulated transform to this level's coordinate space
    M_lvl        = M_init_fullres.copy()
    M_lvl[:, 2] *= level_scale

    # Gaussian pre-smoothing (reduces noise sensitivity)
    if ECC_GAUSS_FILT > 0:
        k = ECC_GAUSS_FILT if ECC_GAUSS_FILT % 2 == 1 else ECC_GAUSS_FILT + 1
        fixed_f32  = cv2.GaussianBlur(fixed_f32,  (k, k), 0)
        moving_f32 = cv2.GaussianBlur(moving_f32, (k, k), 0)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_MAX_ITERS, ECC_EPS)

    try:
        ecc_val, M_out = cv2.findTransformECC(
            fixed_f32, moving_f32,
            M_lvl.astype(np.float32),
            ECC_MOTION_MODEL,
            criteria,
            inputMask=None,
            gaussFiltSize=1,   # already pre-smoothed above
        )
    except cv2.error as e:
        logger.warning(f"  [{name}] ECC failed: {e}")
        return None

    logger.info(f"  [{name}] ECC correlation = {ecc_val:.6f}")

    if ecc_val < 0.3:
        logger.warning(f"  [{name}] ECC correlation too low ({ecc_val:.4f}) — rejecting.")
        return None

    # Scale translation back to full resolution
    M_fullres        = M_out.astype(np.float64)
    M_fullres[:, 2] /= level_scale
    return M_fullres


# ─────────────────────────────────────────────────────────────────────────────
# Pyramidal registration — one slice
# ─────────────────────────────────────────────────────────────────────────────

def register_slice_pyramidal(fixed_np, moving_np, slice_id=None, diag_dir=None):
    """
    3-level coarse-to-fine intensity registration.
      L0 (0.25x) : phase correlation → pure translation seed
      L1 (0.50x) : ECC affine refinement
      L2 (1.00x) : ECC affine refinement (sub-pixel)

    Returns (aligned_np, mse, elapsed, stats, success, M_final).
    Same return signature as the feature script for drop-in comparison.
    """
    start = time.time()

    fixed_raw  = fixed_np[REG_CHANNEL_IDX].astype(np.float32)
    moving_raw = moving_np[REG_CHANNEL_IDX].astype(np.float32)

    fixed_u8,  fixed_f32  = prepare_intensity(fixed_raw)
    moving_u8, moving_f32 = prepare_intensity(moving_raw)

    h_full, w_full = fixed_f32.shape

    # Identity seed
    M_accum = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]], dtype=np.float64)

    level_stats         = []   # (method, ecc_val_or_response, tx, ty) per level
    any_level_succeeded = False

    for lvl_idx, scale in enumerate(PYRAMID_SCALES):
        level_label = f"L{lvl_idx}_s{scale:.2f}"

        fixed_lvl  = resize_image(fixed_f32,  scale)
        moving_lvl = resize_image(moving_f32, scale)
        lh, lw     = fixed_lvl.shape

        # Apply current accumulated transform to moving at this level before estimation
        M_lvl        = M_accum.copy()
        M_lvl[:, 2] *= scale
        moving_warped_lvl = cv2.warpAffine(
            moving_lvl, M_lvl, (lw, lh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # ── Choose method per level ──
        if lvl_idx == 0:
            M_new_fullres = phase_correlation_translation(
                fixed_lvl, moving_warped_lvl, scale)
            method = "PhaseCorr"
        else:
            M_new_fullres = ecc_refinement(
                fixed_lvl, moving_warped_lvl, M_accum, scale, lvl_idx)
            method = "ECC"

        # ── Candidate composition and sanity check ──
        if M_new_fullres is not None and transform_is_sane(M_new_fullres):
            # Compose: new residual on top of accumulated
            # For phase corr (pure translation) just add translation
            if lvl_idx == 0:
                M_candidate = M_accum.copy()
                M_candidate[0, 2] += M_new_fullres[0, 2]
                M_candidate[1, 2] += M_new_fullres[1, 2]
            else:
                # ECC returns absolute transform at this level — use directly
                M_candidate = M_new_fullres

            if transform_is_sane(M_candidate):
                rot, tx, ty = decompose_M(M_candidate)
                logger.info(
                    f"  [{level_label}] {method} accepted: "
                    f"rot={rot:.2f}° tx={tx:.1f}px ty={ty:.1f}px"
                )
                M_accum = M_candidate
                any_level_succeeded = True
                level_stats.append((method, True, tx, ty))
            else:
                rot, tx, ty = decompose_M(M_candidate)
                logger.warning(
                    f"  [{level_label}] Composed transform failed sanity "
                    f"(rot={rot:.1f}° tx={tx:.1f}px ty={ty:.1f}px) — carrying previous."
                )
                level_stats.append((method, False, tx, ty))
        else:
            if M_new_fullres is None:
                logger.warning(f"  [{level_label}] {method} returned None — carrying previous.")
            else:
                rot, tx, ty = decompose_M(M_new_fullres)
                logger.warning(
                    f"  [{level_label}] {method} failed sanity gate "
                    f"(rot={rot:.1f}° tx={tx:.1f}px ty={ty:.1f}px) — carrying previous."
                )
            level_stats.append((method, False, 0.0, 0.0))

        # ── Diagnostics overlay ──
        if diag_dir is not None and slice_id is not None:
            M_diag        = M_accum.copy()
            M_diag[:, 2] *= scale
            warped_diag = cv2.warpAffine(
                moving_lvl, M_diag, (lw, lh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            fixed_u8_lvl  = cv2.normalize(
                fixed_lvl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            warped_u8_lvl = cv2.normalize(
                warped_diag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            save_intensity_diagnostics(
                diag_dir, slice_id, level_label, lvl_idx,
                fixed_u8_lvl, warped_u8_lvl,
            )

    # ── Fallback to identity if all levels failed ──
    if not any_level_succeeded:
        logger.warning(f"[{slice_id}] All levels failed — falling back to identity.")
        M_accum = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)

    # ── Apply final transform to ALL channels ──
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

    # ── MSE on registration channel ──
    warped_u8_final, _ = prepare_intensity(
        aligned_np[REG_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean(
        (fixed_u8.astype(np.float32) - warped_u8_final.astype(np.float32)) ** 2))

    rot, tx, ty = decompose_M(M_accum)
    elapsed     = time.time() - start

    stats = dict(
        detector            = f"Intensity_Pyramid_3L_{REG_CHANNEL_NAME}",
        rotation_deg        = round(rot, 3),
        tx                  = round(tx, 3),
        ty                  = round(ty, 3),
        level_methods       = [s[0] for s in level_stats],
        level_success       = [s[1] for s in level_stats],
        level_tx            = [round(s[2], 2) for s in level_stats],
        level_ty            = [round(s[3], 2) for s in level_stats],
        success             = any_level_succeeded,
    )

    return aligned_np, mse, elapsed, stats, any_level_succeeded, M_accum


# ─────────────────────────────────────────────────────────────────────────────
# QC montage  (same structure as feature script)
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_montage(vol, output_folder,
                        channel_idx=REG_CHANNEL_IDX,
                        channel_name=REG_CHANNEL_NAME):
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
        overlay = np.dstack((np.zeros_like(s1), norm(s1), norm(s2)))
        axes_flat[idx].imshow(overlay)
        axes_flat[idx].set_title(f"Z{z1}→Z{z2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(
        f'Registration QC Intensity Pyramid (Stage 1): {TARGET_CORE}',
        fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(
        output_folder,
        f"{TARGET_CORE}_QC_Montage_{channel_name}_Intensity_Pyramid.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main  (identical flow to feature script)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(
        f"Pyramidal Intensity Registration (Stage 1 / Full Resolution) — {TARGET_CORE}")
    logger.info(
        f"Channel: {REG_CHANNEL_NAME} (idx {REG_CHANNEL_IDX}) | "
        f"Levels: {PYRAMID_SCALES} | "
        f"L0=PhaseCorr, L1/L2=ECC({ECC_MOTION_MODEL})")

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    if n_slices < 2:
        logger.warning("Insufficient slice depth. Writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER,
                                f"{TARGET_CORE}_Intensity_Aligned.ome.tif")
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
        diag_out_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_Pyramid_Intensity")

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
                aligned_vol[i] = raw_slices[i]
                status_str     = "SANITY_FAIL_RAW" if success else "IDENTITY_FALLBACK_RAW"
                logger.warning(
                    f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | "
                f"MSE: {mse:.2f} | Rot: {stats['rotation_deg']:.2f}° | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
                f"LvlMethods: {stats['level_methods']} | "
                f"LvlOK: {stats['level_success']} | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":     direction,
                "Slice_Z":       i,
                "Slice_ID":      real_id,
                "Detector":      stats["detector"],
                "L0_Method":     stats["level_methods"][0],
                "L1_Method":     stats["level_methods"][1],
                "L2_Method":     stats["level_methods"][2],
                "L0_Success":    stats["level_success"][0],
                "L1_Success":    stats["level_success"][1],
                "L2_Success":    stats["level_success"][2],
                "L0_tx":         stats["level_tx"][0],
                "L1_tx":         stats["level_tx"][1],
                "L2_tx":         stats["level_tx"][2],
                "L0_ty":         stats["level_ty"][0],
                "L1_ty":         stats["level_ty"][1],
                "L2_ty":         stats["level_ty"][2],
                "Success":       success,
                "Status":        status_str,
                "MSE_After":     round(mse, 4),
                "Rotation_Deg":  stats["rotation_deg"],
                "Shift_X_px":    stats["tx"],
                "Shift_Y_px":    stats["ty"],
                "Runtime_s":     round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = [
        "Direction", "Slice_Z", "Slice_ID", "Detector",
        "L0_Method", "L1_Method", "L2_Method",
        "L0_Success", "L1_Success", "L2_Success",
        "L0_tx", "L1_tx", "L2_tx",
        "L0_ty", "L1_ty", "L2_ty",
        "Success", "Status",
        "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "MSE_After", "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_Intensity_pyramid.csv"),
        index=False)

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_sanity   = int((df["Status"] == "SANITY_FAIL_RAW").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. "
        f"SUCCESS: {n_ok} | SANITY_FAIL_RAW: {n_sanity} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER)

    out_tiff = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}_Intensity_Aligned.ome.tif")
    logger.info(f"Committing registered tensor to disk at {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    logger.info("Done.")


if __name__ == "__main__":
    main()