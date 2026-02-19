import SimpleITK as sitk
import numpy as np
import glob
import os
import tifffile
import time
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import argparse 
import sys
# Get the absolute path of the directory containing this script (registration/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (3D-TMA-Register/)
parent_dir = os.path.dirname(current_dir)
# Add parent directory to sys.path so we can import config
sys.path.append(parent_dir)
import config 

# Force headless backend
matplotlib.use('Agg')

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Register TMA core using Amoeba + MeanSquares.')
parser.add_argument('--core_name', type=str, required=True, help='Name of the core (e.g., Core_01)')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- CONFIGURATION ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
WORK_OUTPUT = os.path.join(config.DATASPACE, "Amoeba_Registered_rigid_geometry") 
INPUT_FOLDER = os.path.join(DATA_BASE_PATH, TARGET_CORE)
OUTPUT_FOLDER = os.path.join(WORK_OUTPUT, TARGET_CORE)

# Tuning parameters (Specific to Amoeba Experiment)
SHRINK_FACTOR = 16
SAMPLING_PERC = 0.4
CK_CHANNEL_IDX = 6

if not os.path.exists(INPUT_FOLDER):
    print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- HELPER FUNCTIONS ---
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0 

def generate_qc_montage(vol, output_folder, channel_idx=6, channel_name="CK"):
    print(f"\n>>> GENERATING QC MONTAGE ({channel_name} Channel)")
    n_slices = vol.shape[0]
    if n_slices < 2: return

    all_pairs = [(i, i+1) for i in range(n_slices - 1)]
    n_pairs = len(all_pairs)
    n_cols = 5 
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes.reshape(1, -1)
    elif n_cols == 1: axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    print(f"Rendering {n_pairs} pairs...", end="", flush=True)
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

    for idx in range(len(all_pairs), len(axes_flat)): axes_flat[idx].axis('off')
    output_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}.png")
    plt.suptitle(f'Registration QC (Amoeba): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    print(f" Done.\nSaved to: {output_path}")

# --- AMOEBA REGISTRATION LOGIC ---
def register_slice_amoeba(fixed_full_image, moving_full_image):
    """
    Implements specific Amoeba + MeanSquares logic.
    Returns: aligned_full_image, metric, runtime, (rotation, shift_x, shift_y)
    """
    start_time = time.time()
    
    # 1. Prepare Data (Cast to Float32 for registration)
    fixed_ck = sitk.Cast(fixed_full_image[:, :, CK_CHANNEL_IDX], sitk.sitkFloat32)
    moving_ck = sitk.Cast(moving_full_image[:, :, CK_CHANNEL_IDX], sitk.sitkFloat32)

    # 2. Downsample for speed/robustness (Amoeba Strategy)
    fixed_low = sitk.Shrink(fixed_ck, [SHRINK_FACTOR] * 2)
    moving_low = sitk.Shrink(moving_ck, [SHRINK_FACTOR] * 2)

    # 3. Initialize Transform (Center -> Moments)
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_low, moving_low, 
        sitk.Euler2DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 4. Registration Setup
    reg = sitk.ImageRegistrationMethod()

    # Metric: MeanSquares
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(SAMPLING_PERC)

    # Optimizer: Amoeba (Simplex)
    reg.SetOptimizerAsAmoeba(simplexDelta=2.0, numberOfIterations=200)

    # Pyramid: Single level (pre-shrunk)
    reg.SetShrinkFactorsPerLevel([1])
    reg.SetSmoothingSigmasPerLevel([0])
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)

    reg.SetInitialTransform(initial_tx)
    reg.SetInterpolator(sitk.sitkLinear)

    # 5. Execute Registration
    final_metric = 0.0
    rotation_deg = 0.0
    shift_x = 0.0
    shift_y = 0.0
    
    try:
        final_tx = reg.Execute(fixed_low, moving_low)
        final_metric = reg.GetMetricValue()
        
        # Extract params for logging (AffineTransform has 6 parameters)
        # We approximate rotation/shift for CSV readability, though Affine is more complex
        params = final_tx.GetParameters()
        rotation_deg = np.degrees(params[0])
        shift_x = params[1]
        shift_y = params[2]
        # Rotation approximation (from first element of matrix)
        rotation_deg = np.degrees(np.arccos(params[0])) if abs(params[0]) <= 1 else 0

    except Exception as e:
        print(f" [Registration Warning] {e}")
        final_tx = initial_tx # Fallback
    
    # 6. Apply Transform to Full Image (All Channels)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_ck) # Use full res reference
    resampler.SetTransform(final_tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    rotated_channels = []
    # Iterate over channels to preserve data integrity
    for c in range(fixed_full_image.GetSize()[2]): 
        chan = moving_full_image[:, :, c]
        # Note: final_tx works on physical coordinates, so it applies correctly 
        # to the full image even though calculated on downsampled data.
        rotated_channels.append(resampler.Execute(chan))
        
    aligned_full_image = sitk.JoinSeries(rotated_channels)
    aligned_full_image = sitk.Cast(aligned_full_image, sitk.sitkUInt16)
    
    elapsed = time.time() - start_time
    return aligned_full_image, final_metric, elapsed, (rotation_deg, shift_x, shift_y)

# --- MAIN PROCESSING ---
def main():
    print(f"--- Starting Processing: {TARGET_CORE} (Amoeba/MeanSquares) ---")
    
    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    total_slices = len(file_list)
    
    if total_slices == 0:
        print("No files found.")
        sys.exit(0)

    aligned_results = [None] * total_slices
    registration_stats = [] 

    # Anchor (Center Slice)
    center_idx = total_slices // 2
    center_full = sitk.ReadImage(file_list[center_idx])
    aligned_results[center_idx] = center_full
    anchor_slice_id = get_slice_number(file_list[center_idx])
    print(f"Anchor set at Slice {anchor_slice_id}")

    # Propagation Logic
    def process_pass(indices, direction):
        # We need to maintain the "current fixed" image as we propagate
        # In the original amoeba script, it looked up aligned_results[i +/- offset]
        # Here we make that explicit
        
        print(f"\n>>> STARTING {direction} PASS")
        
        if direction == "Backward":
            fixed_offset = 1 # Fixed is at i + 1
        else:
            fixed_offset = -1 # Fixed is at i - 1

        for i in indices:
            real_id = get_slice_number(file_list[i])
            print(f"Processing Slice {real_id}...", end="", flush=True)
            
            # The fixed image is the result of the PREVIOUS step
            fixed_full = aligned_results[i + fixed_offset]
            moving_full = sitk.ReadImage(file_list[i])
            
            aligned, metric, runtime, transform = register_slice_amoeba(fixed_full, moving_full)
            aligned_results[i] = aligned
            
            registration_stats.append({
                "Direction": direction, 
                "Moving_Slice": real_id, 
                "Final_Metric": metric,
                "Runtime(s)": round(runtime, 2), 
                "Rotation_Deg": round(transform[0], 2),
                "Shift_X_px": round(transform[1], 2), 
                "Shift_Y_px": round(transform[2], 2)
            })
            print(f" Done. (Metric: {metric:.4f}, Time: {runtime:.2f}s)")

    # Run Passes
    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < total_slices - 1:
        process_pass(range(center_idx + 1, total_slices), "Forward")

    # 4. Save Stats
    df = pd.DataFrame(registration_stats).sort_values(by="Moving_Slice")
    cols = ["Direction", "Moving_Slice", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "Final_Metric", "Runtime(s)"]
    if not df.empty:
        df[cols].to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats_amoeba.csv"), index=False)

    # 5. Save Volume & Generate QC
    valid_slices = [img for img in aligned_results if img is not None]
    if not valid_slices: return
    
    print("Stacking volume...")
    # Convert list of images to numpy array (Z, C, Y, X)
    vol = np.array([sitk.GetArrayFromImage(img) for img in valid_slices])
    
    # Generate Montage
    generate_qc_montage(vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    # Save Full OME-TIFF
    output_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Amoeba_Aligned.ome.tif")
    print(f"Saving OME-TIFF to {output_path}...")
    tifffile.imwrite(output_path, vol, photometric='minisblack', metadata={'axes': 'ZCYX'}, compression='zlib')
    print("Done.")

if __name__ == "__main__":
    main()