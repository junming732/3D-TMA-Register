"""
Feature registration — Stage 2: B-spline NCC elastic refinement.

Reads the AKAZE-aligned interim stack produced by feature_registration_AKAZE.py
and applies a B-spline FFD with NCC metric to each consecutive slice pair.

Pipeline per slice pair:
  L1  B-spline FFD with NCC metric (SimpleITK) → elastic residual correction
      Applied to every slice where the prior AKAZE alignment succeeded.

Output sanity:
  After L1, the CK channel is checked for blank output (B-spline divergence).
  If >99% zero, reverts to the AKAZE-only slice to protect the chain.

Output:
  - Final BSpline-refined stack  : <OUTPUT_FOLDER>/<CORE>_AKAZE_BSpline_Aligned.ome.tif
  - QC montage (CK channel)      : <OUTPUT_FOLDER>/<CORE>_QC_Montage_CK_BSpline.png
  - CSV stats                    : <OUTPUT_FOLDER>/registration_stats_BSpline.csv

Prerequisites:
  Run feature_registration_AKAZE.py first to generate the interim stack.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
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
parser = argparse.ArgumentParser(description='Feature registration — Stage 2: B-spline NCC elastic.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
WORK_OUTPUT   = os.path.join(config.DATASPACE, "Feature_AKAZE_BSpline_NCC_2Stage")
STAGE1_FOLDER = os.path.join(WORK_OUTPUT, TARGET_CORE)                     # output of Stage 1
OUTPUT_FOLDER = STAGE1_FOLDER                                               # write to same folder

INTERIM_STACK = os.path.join(STAGE1_FOLDER, f"{TARGET_CORE}_AKAZE_Aligned.ome.tif")

# --- CHANNEL ---
CK_CHANNEL_IDX = 6

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
MIN_CK_NONZERO_FRAC = 0.01   # revert to AKAZE-only slice if CK output is >99% black


if not os.path.exists(INTERIM_STACK):
    logger.error(
        f"Interim AKAZE stack not found: {INTERIM_STACK}\n"
        f"Run feature_registration_AKAZE.py --core_name {TARGET_CORE} first."
    )
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ck(img_arr: np.ndarray):
    """
    Returns (linear_uint8, log_uint8).
    Log image is used for B-spline registration (boosts weak signal).
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
# L1 — B-spline NCC elastic refinement
# ─────────────────────────────────────────────────────────────────────────────

def bspline_ncc(fixed_8bit: np.ndarray, moving_8bit: np.ndarray,
                h: int, w: int, slice_id: str):
    """
    B-spline FFD registration with NCC metric on the AKAZE-prealigned CK channel.
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
# Refine one slice with B-spline (fixed and moving are AKAZE-aligned slices)
# ─────────────────────────────────────────────────────────────────────────────

def refine_slice_bspline(fixed_np, moving_np, slice_id=None):
    """
    Apply B-spline NCC refinement to a slice that has already been AKAZE-aligned.
    fixed_np  — reference slice (AKAZE-aligned, from interim stack)
    moving_np — slice to refine  (AKAZE-aligned, from interim stack)
    Returns (refined_np, mse, elapsed, bspline_ok).
    """
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # ── L1: B-spline NCC ─────────────────────────────────────────────────────
    # The input is already AKAZE-aligned, so we apply B-spline directly
    # (no additional affine pre-warp needed).
    remap_x, remap_y = bspline_ncc(fixed_log, moving_log, h, w, sid)
    bspline_ok = remap_x is not None

    if bspline_ok:
        refined_np = np.stack(
            [cv2.remap(
                moving_np[ch].astype(np.float32), remap_x, remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
             ) for ch in range(fixed_np.shape[0])],
            axis=0,
        ).astype(np.uint16)

        # ── Output sanity: revert to AKAZE-only if B-spline output is blank ──
        ck_out = refined_np[CK_CHANNEL_IDX]
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(
                f"[{sid}] CK output nearly blank — B-spline diverged. "
                f"Reverting to AKAZE-only slice."
            )
            refined_np = moving_np.copy()
            bspline_ok = False
    else:
        # B-spline failed entirely — keep AKAZE-aligned slice unchanged
        refined_np = moving_np.copy()

    # ── MSE on CK channel ─────────────────────────────────────────────────────
    _, refined_log = prepare_ck(refined_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_lin.astype(np.float32) - refined_log.astype(np.float32)) ** 2))

    return refined_np, mse, time.time() - start, bspline_ok


# ─────────────────────────────────────────────────────────────────────────────
# QC montage
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_montage(vol, output_folder, channel_idx=6, channel_name="CK", stage="BSpline"):
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Feature Registration Stage 2: B-spline NCC — {TARGET_CORE}")
    logger.info(f"Reading interim AKAZE stack from {INTERIM_STACK}")
    logger.info(f"B-spline nodes={BSPLINE_GRID_NODES} | iters={BSPLINE_ITERATIONS}")

    # ── Load the AKAZE-aligned interim stack ──────────────────────────────────
    akaze_vol = tifffile.imread(INTERIM_STACK)  # expected shape: (Z, C, H, W)

    if akaze_vol.ndim == 3:
        # Single slice: (C, H, W) → (1, C, H, W)
        akaze_vol = akaze_vol[np.newaxis]
    if akaze_vol.ndim != 4:
        logger.error(f"Unexpected stack shape {akaze_vol.shape}. Expected (Z, C, H, W).")
        sys.exit(1)

    n_slices, c, target_h, target_w = akaze_vol.shape
    logger.info(f"Loaded stack: Z={n_slices}, C={c}, H={target_h}, W={target_w}")

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_BSpline_Aligned.ome.tif")
        tifffile.imwrite(out_path, akaze_vol, photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='deflate',
                         compressionargs={'level': 6})
        sys.exit(0)

    # ── Refine each slice in-order (forward pass only; stack is already ordered) ──
    # We copy the anchor (center slice stays unchanged) and refine all others
    # relative to their already-aligned neighbour.
    refined_vol  = akaze_vol.copy()
    center_idx   = n_slices // 2  # mirror the same anchor used in Stage 1
    bspline_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} B-spline pass.")

        for i in indices:
            sid = f"Z{i:03d}"

            # fixed  = already-refined neighbour (propagated chain)
            # moving = AKAZE-aligned slice that still needs elastic correction
            fixed_np  = refined_vol[i + fixed_offset]
            moving_np = akaze_vol[i]

            refined_np, mse, runtime, bspline_ok = refine_slice_bspline(
                fixed_np, moving_np, slice_id=sid,
            )

            refined_vol[i] = refined_np
            status_str     = "BSPLINE_OK" if bspline_ok else "AKAZE_FALLBACK"

            logger.info(
                f"Z{i:02d} | BSpline: {bspline_ok} | "
                f"MSE: {mse:.2f} | t: {runtime:.2f}s | Status: {status_str}"
            )

            bspline_stats.append({
                "Direction":  direction,
                "Slice_Z":    i,
                "BSpline_OK": bspline_ok,
                "Status":     status_str,
                "MSE_After":  round(mse, 4),
                "Runtime_s":  round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    # ── CSV ───────────────────────────────────────────────────────────────────
    df   = pd.DataFrame(bspline_stats).sort_values("Slice_Z")
    cols = ["Direction", "Slice_Z", "BSpline_OK", "Status", "MSE_After", "Runtime_s"]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_BSpline.csv"), index=False
    )

    n_ok       = int((df["Status"] == "BSPLINE_OK").sum())
    n_fallback = int((df["Status"] == "AKAZE_FALLBACK").sum())
    logger.info(
        f"B-spline pass complete. BSPLINE_OK: {n_ok} | AKAZE_FALLBACK: {n_fallback}"
    )

    # ── QC montage for B-spline-refined result ────────────────────────────────
    generate_qc_montage(refined_vol, OUTPUT_FOLDER,
                        channel_idx=CK_CHANNEL_IDX, channel_name="CK", stage="BSpline")

    # ── Write final refined stack ─────────────────────────────────────────────
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_AKAZE_BSpline_Aligned.ome.tif")
    logger.info(f"Writing B-spline-refined volume to {out_tiff}")
    tifffile.imwrite(out_tiff, refined_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='deflate',
                     compressionargs={'level': 6})
    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    main()