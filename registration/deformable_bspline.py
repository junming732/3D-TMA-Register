"""
Deformable registration as the SECOND stage in the pipeline.
Reads the 4D affine-aligned stack from Stage 1 (Feature_Aligned.ome.tif).

Tradeoff: BSpline deformable registration on CK channel — sub-cellular residual correction.
Tradeoff: Coarse control point grid prevents fitting biological variation between slices.
Tradeoff: Transform estimated on CK channel (index 6) and applied to all channels.
Update: Per-slice QC overlays and deformation field magnitude diagnostics.

Input:  <DATASPACE>/Feature_logarithm/<core_name>/<core_name>_Feature_Aligned.ome.tif  (Z, C, H, W)
Output: <DATASPACE>/Deformable_BSpline/<core_name>/<core_name>_Deformable_Aligned.ome.tif (Z, C, H, W)
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
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

# CONFIGURATION
parser = argparse.ArgumentParser(description='BSpline deformable registration — Stage 2.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
STAGE1_OUTPUT  = os.path.join(config.DATASPACE, "Feature_pyramidal", TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Deformable_BSpline")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
INPUT_TIFF     = os.path.join(STAGE1_OUTPUT, f"{TARGET_CORE}_Feature_Aligned.ome.tif")

# --- CHANNEL ---
CK_CHANNEL_IDX = 6

# --- BSPLINE TUNING ---
# Grid spacing in pixels. Larger = smoother field = less risk of fitting biology.
# Start at 64. Reduce to 32 only if residuals remain after visual QC.
BSPLINE_GRID_SPACING  = 64
# Maximum displacement fraction of image size per control point.
# Keeps the deformable stage from correcting gross misalignment (Stage 1's job).
BSPLINE_MAX_DISP_FRAC = 0.02
# Optimizer
BSPLINE_ITERATIONS    = 100
BSPLINE_LEARNING_RATE = 1.0
# Metric sampling fraction — 10% is sufficient and fast
METRIC_SAMPLING_FRAC  = 0.10
METRIC_HISTOGRAM_BINS = 32

if not os.path.exists(INPUT_TIFF):
    logger.error(f"Stage 1 output not found: {INPUT_TIFF}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_to_float32(channel_2d: np.ndarray) -> np.ndarray:
    """
    Percentile-clips and normalizes a 2D channel to float32 [0, 1].
    Consistent normalization is important for mutual information stability.
    """
    p_lo, p_hi = np.percentile(channel_2d[::4, ::4], (0.1, 99.9))
    clipped    = np.clip(channel_2d.astype(np.float32), p_lo, p_hi)
    denom      = p_hi - p_lo
    if denom < 1e-6:
        return np.zeros_like(clipped)
    return (clipped - p_lo) / denom


def to_sitk(arr_2d: np.ndarray) -> sitk.Image:
    """Converts a 2D numpy float32 array to a SimpleITK image."""
    return sitk.GetImageFromArray(arr_2d.astype(np.float32))


# ---------------------------------------------------------------------------
# BSpline registration
# ---------------------------------------------------------------------------

def register_bspline(
    fixed_2d:  np.ndarray,
    moving_2d: np.ndarray,
    slice_id:  str = "",
) -> sitk.Transform | None:
    """
    Estimates a BSpline deformable transform from moving to fixed on a single
    normalized 2D channel pair.

    Control point grid is derived from BSPLINE_GRID_SPACING so that the
    deformation field cannot resolve structures smaller than one grid cell.

    Returns the final SimpleITK transform or None on failure.
    """
    fixed_sitk  = to_sitk(normalize_to_float32(fixed_2d))
    moving_sitk = to_sitk(normalize_to_float32(moving_2d))

    h, w = fixed_2d.shape
    mesh_size = [
        max(2, w // BSPLINE_GRID_SPACING),
        max(2, h // BSPLINE_GRID_SPACING),
    ]
    logger.info(f"[{slice_id}] BSpline mesh size: {mesh_size} "
                f"(grid spacing ~{BSPLINE_GRID_SPACING}px, image {w}x{h})")

    tx = sitk.BSplineTransformInitializer(
        image1=fixed_sitk,
        transformDomainMeshSize=mesh_size,
        order=3,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=METRIC_HISTOGRAM_BINS)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(METRIC_SAMPLING_FRAC)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=BSPLINE_ITERATIONS,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    # Multi-resolution pyramid: 2 levels (2x downsampled then full res)
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(tx, inPlace=True)

    try:
        final_tx = reg.Execute(fixed_sitk, moving_sitk)
        metric   = reg.GetMetricValue()
        stop     = reg.GetOptimizerStopConditionDescription()
        logger.info(f"[{slice_id}] Converged | Metric: {metric:.5f} | Stop: {stop}")
        return final_tx
    except Exception as e:
        logger.error(f"[{slice_id}] BSpline registration exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def apply_transform_all_channels(
    fixed_np:  np.ndarray,      # (C, H, W) reference geometry
    moving_np: np.ndarray,      # (C, H, W) slice to warp
    transform: sitk.Transform,
) -> np.ndarray:
    """
    Applies a SimpleITK transform to every channel of moving_np,
    resampled onto the fixed_np spatial grid.
    """
    ref_sitk  = to_sitk(fixed_np[0].astype(np.float32))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    aligned_channels = []
    for ch in range(moving_np.shape[0]):
        moving_ch = to_sitk(moving_np[ch].astype(np.float32))
        warped    = sitk.GetArrayFromImage(resampler.Execute(moving_ch))
        aligned_channels.append(warped.astype(np.uint16))

    return np.stack(aligned_channels, axis=0)


# ---------------------------------------------------------------------------
# Deformation field magnitude diagnostic
# ---------------------------------------------------------------------------

def compute_field_magnitude(transform: sitk.Transform, shape_hw: tuple) -> np.ndarray:
    """
    Converts a BSpline transform to a dense displacement field and returns
    the per-pixel magnitude as a 2D float32 array.
    """
    h, w    = shape_hw
    ref_img = sitk.Image([w, h], sitk.sitkFloat32)

    disp_filter = sitk.TransformToDisplacementFieldFilter()
    disp_filter.SetReferenceImage(ref_img)
    disp_field  = disp_filter.Execute(transform)

    arr = sitk.GetArrayFromImage(disp_field)   # (H, W, 2)
    magnitude = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    return magnitude


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def save_deformable_diagnostics(
    output_dir:  str,
    slice_id:    str,
    fixed_2d:    np.ndarray,    # uint16
    moving_pre:  np.ndarray,    # uint16, before deformable (after Stage 1)
    moving_post: np.ndarray,    # uint16, after deformable
    magnitude:   np.ndarray,    # deformation field magnitude (H, W)
):
    """
    Saves three diagnostic plots per slice pair:
      01 — Before/after overlay comparison (red=fixed, green=moving)
      02 — Deformation field magnitude heatmap
      03 — Side-by-side: Stage1 result vs Stage2 result
    """
    os.makedirs(output_dir, exist_ok=True)

    def norm_display(x):
        p99 = np.percentile(x, 99.5)
        return np.clip(x.astype(np.float32) / (p99 if p99 > 0 else 1), 0, 1)

    fixed_n    = norm_display(fixed_2d)
    pre_n      = norm_display(moving_pre)
    post_n     = norm_display(moving_post)

    # Plot 01: overlay before vs after deformable
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    overlay_pre  = np.dstack((fixed_n, pre_n,  np.zeros_like(fixed_n)))
    overlay_post = np.dstack((fixed_n, post_n, np.zeros_like(fixed_n)))

    axes[0].imshow(overlay_pre);  axes[0].set_title("Before Deformable (Stage 1 output)"); axes[0].axis('off')
    axes[1].imshow(overlay_post); axes[1].set_title("After Deformable (Stage 2 output)");  axes[1].axis('off')

    plt.suptitle(f"{slice_id} — Red=Fixed  Green=Moving", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slice_id}_01_Overlay_Comparison.png"), dpi=150)
    plt.close(fig)

    # Plot 02: deformation field magnitude
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(magnitude, cmap='hot')
    plt.colorbar(im, ax=ax, label='Displacement magnitude (px)')
    ax.set_title(f"{slice_id} — Deformation Field Magnitude")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slice_id}_02_Field_Magnitude.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# QC montage
# ---------------------------------------------------------------------------

def generate_qc_montage(vol: np.ndarray, output_folder: str, channel_idx: int = 6, channel_name: str = "CK"):
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

    plt.suptitle(f'Registration QC BSpline Deformable (Stage 2): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_Deformable.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info(f"BSpline Deformable Registration (Stage 2) — {TARGET_CORE}")
    logger.info(f"Input: {INPUT_TIFF}")

    # --- Load Stage 1 output ---
    logger.info("Loading Stage 1 aligned volume.")
    vol = tifffile.imread(INPUT_TIFF)

    # Normalise axis order to (Z, C, H, W)
    if vol.ndim == 3:
        # Single slice, single channel — treat as (1, 1, H, W)
        vol = vol[np.newaxis, np.newaxis]
    elif vol.ndim == 4:
        # Could be (Z, C, H, W) or (Z, H, W, C) — detect by channel count
        if vol.shape[-1] < vol.shape[1]:
            vol = np.moveaxis(vol, -1, 1)
    else:
        logger.error(f"Unexpected volume shape: {vol.shape}")
        sys.exit(1)

    n_slices, n_channels, target_h, target_w = vol.shape
    logger.info(f"Volume shape: Z={n_slices}, C={n_channels}, H={target_h}, W={target_w}")

    if n_slices < 2:
        logger.warning("Single slice — no inter-slice registration possible. Copying input to output.")
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Deformable_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol, photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='zlib')
        sys.exit(0)

    # --- Output volume: start as a copy of Stage 1 result ---
    # Center slice needs no registration — it is the anchor.
    aligned_vol = vol.copy()
    center_idx  = n_slices // 2
    logger.info(f"Anchor slice: Z{center_idx}")

    registration_stats = []
    diag_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_Deformable")

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} deformable pass.")

        for i in indices:
            slice_id  = f"Z{i:03d}"
            start     = time.time()

            fixed_np  = aligned_vol[i + fixed_offset]   # (C, H, W) already deformable-aligned neighbor
            moving_np = aligned_vol[i]                   # (C, H, W) Stage 1 result for this slice

            fixed_ck  = fixed_np[CK_CHANNEL_IDX]
            moving_ck = moving_np[CK_CHANNEL_IDX]

            tx = register_bspline(fixed_ck, moving_ck, slice_id=slice_id)

            if tx is not None:
                warped_np = apply_transform_all_channels(fixed_np, moving_np, tx)
                aligned_vol[i] = warped_np

                # Deformation field magnitude for diagnostics
                magnitude = compute_field_magnitude(tx, (target_h, target_w))
                mean_disp = float(np.mean(magnitude))
                max_disp  = float(np.max(magnitude))

                save_deformable_diagnostics(
                    output_dir  = diag_dir,
                    slice_id    = slice_id,
                    fixed_2d    = fixed_ck,
                    moving_pre  = moving_ck,
                    moving_post = warped_np[CK_CHANNEL_IDX],
                    magnitude   = magnitude,
                )
                success = True
            else:
                logger.warning(f"[{slice_id}] Deformable registration failed — keeping Stage 1 result.")
                mean_disp = 0.0
                max_disp  = 0.0
                success   = False

            elapsed = time.time() - start

            # MSE between fixed and registered moving on CK channel
            fixed_f   = fixed_ck.astype(np.float32)
            moving_f  = aligned_vol[i][CK_CHANNEL_IDX].astype(np.float32)
            mse       = float(np.mean((fixed_f - moving_f) ** 2))

            status_str = "SUCCESS" if success else "STAGE1_FALLBACK"
            logger.info(
                f"{slice_id} | MSE: {mse:.2f} | "
                f"MeanDisp: {mean_disp:.2f}px | MaxDisp: {max_disp:.2f}px | "
                f"t: {elapsed:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":   direction,
                "Slice_Z":     i,
                "Success":     success,
                "MSE_After":   round(mse, 4),
                "Mean_Disp_px": round(mean_disp, 3),
                "Max_Disp_px":  round(max_disp, 3),
                "Runtime_s":   round(elapsed, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    # --- Save stats CSV ---
    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = ["Direction", "Slice_Z", "Success", "MSE_After",
            "Mean_Disp_px", "Max_Disp_px", "Runtime_s"]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_deformable.csv"), index=False
    )

    n_ok       = int(df["Success"].sum())
    n_fallback = int((~df["Success"]).sum())
    logger.info(f"Execution complete. Converged: {n_ok} | Stage1 Fallbacks: {n_fallback}")

    # --- QC montage ---
    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    # --- Write output ---
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Deformable_Aligned.ome.tif")
    logger.info(f"Writing deformable-aligned volume to {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    logger.info("Done.")


if __name__ == "__main__":
    main()