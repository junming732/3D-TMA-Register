"""
Intensity registration (Amoeba + MeanSquares) as the SECOND stage in the pipeline.
Reads Feature-registered per-slice .ome.tif files from Feature_Registered_rigid_geometry
at FULL resolution.

Tradeoff: Replaced downsampled (16x) approach with full-resolution processing for
          higher precision intensity alignment, at the cost of increased runtime.
Tradeoff: Input is individual per-slice .ome.tif files from Stage 1 Feature output,
          preserving the same per-slice sitk.ReadImage pattern as the original script.
"""

import SimpleITK as sitk
import numpy as np
import os
import tifffile
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Intensity registration (Amoeba/MeanSquares) — second pipeline stage, full resolution.')
parser.add_argument('--core_name', type=str, required=True, help='Name of the core (e.g., Core_01)')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- CONFIGURATION ---
# Input: per-slice Feature-registered .ome.tif files from Stage 1
FEATURE_OUTPUT = os.path.join(config.DATASPACE, "Feature_Registered_AKAZE_First")
INPUT_FOLDER   = os.path.join(FEATURE_OUTPUT, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Intensity_after_AKAZE")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)

# Tuning parameters
# SHRINK_FACTOR removed — registration runs at full resolution
SAMPLING_PERC  = 0.4
CK_CHANNEL_IDX = 6

if not os.path.exists(INPUT_FOLDER):
    print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
    print("        Ensure Stage 1 (feature_akaze_rigid.py) has completed successfully.")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- HELPER FUNCTIONS ---
def generate_qc_montage(vol, output_folder, channel_idx=6, channel_name="CK"):
    print(f"\n>>> GENERATING QC MONTAGE ({channel_name} Channel)")
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

    print(f"Rendering {len(all_pairs)} pairs...", end="", flush=True)
    for idx, (z1, z2) in enumerate(all_pairs):
        s1 = vol[z1, channel_idx, :, :]
        s2 = vol[z2, channel_idx, :, :]
        def norm(x):
            p99 = np.percentile(x, 99.5)
            if p99 == 0: p99 = 1
            return np.clip(x / p99, 0, 1)
        overlay = np.dstack((norm(s1), norm(s2), np.zeros_like(s1)))
        ax = axes_flat[idx]
        ax.imshow(overlay)
        ax.set_title(f"Z{z1} -> Z{z2}", fontsize=10, fontweight='bold')
        ax.axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    output_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_Amoeba.png")
    plt.suptitle(f'Registration QC (Amoeba Stage 2): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" Done.\nSaved to: {output_path}")


# --- AMOEBA REGISTRATION LOGIC ---
def register_slice_amoeba(fixed_np: np.ndarray, moving_np: np.ndarray):
    """
    Implements Amoeba + MeanSquares intensity registration at full resolution.
    fixed_np / moving_np: numpy arrays (C, H, W), dtype uint16.
    Returns: aligned_np (C, H, W), metric, runtime, (rotation_deg, shift_x, shift_y)
    """
    start_time = time.time()

    # 1. Extract CK channel as 2D sitk images for registration
    fixed_ck  = sitk.GetImageFromArray(fixed_np[CK_CHANNEL_IDX].astype(np.float32))
    moving_ck = sitk.GetImageFromArray(moving_np[CK_CHANNEL_IDX].astype(np.float32))

    # 2. Full resolution — no downsampling applied (SHRINK_FACTOR removed)

    # 3. Initialize Transform (Geometry-based centering)
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_ck, moving_ck,
        sitk.Euler2DTransform()
        #sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 4. Registration Setup
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(SAMPLING_PERC)
    reg.SetOptimizerAsAmoeba(simplexDelta=2.0, numberOfIterations=300)
    reg.SetShrinkFactorsPerLevel([1])
    reg.SetSmoothingSigmasPerLevel([0])
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
    reg.SetInitialTransform(initial_tx)
    reg.SetInterpolator(sitk.sitkLinear)

    # 5. Execute Registration
    final_metric = 0.0
    rotation_deg = 0.0
    shift_x      = 0.0
    shift_y      = 0.0

    try:
        final_tx     = reg.Execute(fixed_ck, moving_ck)
        final_metric = reg.GetMetricValue()
        params       = final_tx.GetParameters()
        rotation_deg = float(np.degrees(params[0]))
        shift_x      = float(params[3]) if len(params) > 3 else 0.0
        shift_y      = float(params[4]) if len(params) > 4 else 0.0

    except Exception as e:
        print(f" [Registration Warning] {e}")
        final_tx = initial_tx

    # 6. Apply Transform to All Channels
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_ck)
    resampler.SetTransform(final_tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    aligned_channels = []
    for c in range(moving_np.shape[0]):
        chan_sitk  = sitk.GetImageFromArray(moving_np[c].astype(np.float32))
        warped     = resampler.Execute(chan_sitk)
        aligned_channels.append(sitk.GetArrayFromImage(warped))

    aligned_np = np.stack(aligned_channels, axis=0).astype(np.uint16)
    elapsed    = time.time() - start_time
    return aligned_np, final_metric, elapsed, (rotation_deg, shift_x, shift_y)


# --- MAIN PROCESSING ---
def main():
    print(f"--- Starting Processing: {TARGET_CORE} (Amoeba/MeanSquares — Stage 2, Full Resolution) ---")

    input_path = os.path.join(INPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    if not os.path.exists(input_path):
        print(f"[ERROR] Feature-registered volume not found: {input_path}")
        print("        Ensure Stage 1 (feature_akaze_rigid.py) has completed successfully.")
        sys.exit(1)

    # Load stacked Feature volume: shape (Z, C, H, W)
    print(f"Loading Feature-registered volume from {input_path} ...")
    vol_in = tifffile.imread(input_path)

    if vol_in.ndim != 4:
        print(f"[ERROR] Expected 4D volume (Z, C, H, W), got shape {vol_in.shape}")
        sys.exit(1)

    total_slices = vol_in.shape[0]
    print(f"Volume loaded: {total_slices} slices, shape {vol_in.shape}")

    if total_slices < 2:
        print("Insufficient slice depth. Writing identity output.")
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Amoeba_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol_in, photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='zlib')
        sys.exit(0)

    center_idx  = total_slices // 2
    aligned_vol = np.zeros_like(vol_in)
    aligned_vol[center_idx] = vol_in[center_idx]
    print(f"Anchor set at Slice {center_idx}")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        print(f"\n>>> STARTING {direction} PASS")

        for i in indices:
            print(f"Processing Slice {i}...", end="", flush=True)

            fixed_np  = aligned_vol[i + fixed_offset]  # (C, H, W)
            moving_np = vol_in[i]                       # (C, H, W)

            aligned_np, metric, runtime, transform = register_slice_amoeba(fixed_np, moving_np)
            aligned_vol[i] = aligned_np

            registration_stats.append({
                "Direction":    direction,
                "Moving_Slice": i,
                "Final_Metric": metric,
                "Runtime(s)":   round(runtime, 2),
                "Rotation_Deg": round(transform[0], 2),
                "Shift_X_px":   round(transform[1], 2),
                "Shift_Y_px":   round(transform[2], 2),
            })
            print(f" Done. (Metric: {metric:.4f}, Time: {runtime:.2f}s)")

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < total_slices - 1:
        process_pass(range(center_idx + 1, total_slices), "Forward")

    # Save Stats
    df   = pd.DataFrame(registration_stats).sort_values(by="Moving_Slice")
    cols = ["Direction", "Moving_Slice", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "Final_Metric", "Runtime(s)"]
    if not df.empty:
        df[cols].to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats_amoeba.csv"), index=False)

    # Generate QC
    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    # Save Full OME-TIFF
    output_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Amoeba_Aligned.ome.tif")
    print(f"Saving OME-TIFF to {output_path}...")
    tifffile.imwrite(output_path, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    print("Done.")


if __name__ == "__main__":
    main()