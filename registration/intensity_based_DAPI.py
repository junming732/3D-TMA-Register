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
import config 

# Force headless backend
matplotlib.use('Agg')

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Register a single TMA core.')
parser.add_argument('--core_name', type=str, required=True, help='Name of the core (e.g., Core_01)')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- CONFIGURATION ---
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
WORK_OUTPUT = os.path.join(config.DATASPACE, "Intensity_Registered_DAPI")
INPUT_FOLDER = os.path.join(DATA_BASE_PATH, TARGET_CORE)
OUTPUT_FOLDER = os.path.join(WORK_OUTPUT, TARGET_CORE)

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
    plt.suptitle(f'Registration QC: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    print(f" Done.\nSaved to: {output_path}")

# --- REGISTRATION LOGIC ---
def register_slice_with_metrics(fixed_full_image, moving_full_image):
    #DAPI
    fixed_ck = sitk.Cast(fixed_full_image[:, :, 0], sitk.sitkFloat32)
    moving_ck = sitk.Cast(moving_full_image[:, :, 0], sitk.sitkFloat32)
    
    # GEOMETRY CENTER INITIALIZATION (Fixes "C" shape issues)
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_ck, moving_ck, 
        sitk.Euler2DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY 
    )
    
    reg = sitk.ImageRegistrationMethod()
    reg.SetShrinkFactorsPerLevel([16]) 
    reg.SetSmoothingSigmasPerLevel([2]) 
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=300, epsilon=1e-6, initialRadius=1.5, growthFactor=1.1, shrinkFactor=0.9)
    reg.SetInitialTransform(initial_tx)
    
    metric_values = []
    reg.AddCommand(sitk.sitkIterationEvent, lambda: metric_values.append(reg.GetMetricValue()))
    
    start_time = time.time()
    try:
        final_tx = reg.Execute(fixed_ck, moving_ck)
        params = final_tx.GetParameters()
        rotation_deg = np.degrees(params[0])
        shift_x = params[1]
        shift_y = params[2]
    except Exception as e:
        print(f"  Warning: Registration failed: {e}")
        return moving_full_image, [], 0.0, (0,0,0)
    end_time = time.time()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_ck)
    resampler.SetTransform(final_tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    rotated_channels = []
    for c in range(fixed_full_image.GetSize()[2]): 
        chan = moving_full_image[:, :, c]
        rotated_channels.append(resampler.Execute(chan))
        
    aligned_full_image = sitk.JoinSeries(rotated_channels)
    aligned_full_image = sitk.Cast(aligned_full_image, sitk.sitkUInt16)
    
    return aligned_full_image, metric_values, end_time - start_time, (rotation_deg, shift_x, shift_y)

# --- MAIN ---
def main():
    print(f"--- Starting Processing: {TARGET_CORE} ---")
    
    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    total_slices = len(file_list)
    
    if total_slices == 0:
        print("No files found.")
        sys.exit(0)

    aligned_results = [None] * total_slices
    registration_stats = [] 

    # Anchor
    center_idx = total_slices // 2
    center_full = sitk.ReadImage(file_list[center_idx])
    aligned_results[center_idx] = center_full
    anchor_slice_id = get_slice_number(file_list[center_idx])
    print(f"Anchor set at Slice {anchor_slice_id}")

    # Propagate
    def process_pass(indices, direction):
        nonlocal center_full
        current_fixed = center_full
        print(f"\n>>> STARTING {direction} PASS")
        for i in indices:
            real_id = get_slice_number(file_list[i])
            print(f"Processing Slice {real_id}...", end="", flush=True)
            moving = sitk.ReadImage(file_list[i])
            
            aligned, metrics, runtime, transform = register_slice_with_metrics(current_fixed, moving)
            aligned_results[i] = aligned
            
            registration_stats.append({
                "Direction": direction, "Moving_Slice": real_id, "Final_Metric": metrics[-1] if metrics else 0,
                "Runtime(s)": round(runtime, 2), "Rotation_Deg": round(transform[0], 2),
                "Shift_X_px": round(transform[1], 2), "Shift_Y_px": round(transform[2], 2)
            })
            print(f" Done. (Rot: {transform[0]:.1f}°, X: {transform[1]:.1f}, Y: {transform[2]:.1f})")
            current_fixed = aligned

    process_pass(range(center_idx - 1, -1, -1), "Backward")
    process_pass(range(center_idx + 1, total_slices), "Forward")

    # Save Stats
    df = pd.DataFrame(registration_stats).sort_values(by="Moving_Slice")
    cols = ["Direction", "Moving_Slice", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "Final_Metric", "Runtime(s)"]
    if not df.empty:
        df[cols].to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats.csv"), index=False)
    
    # Save Volume
    valid_slices = [img for img in aligned_results if img is not None]
    if not valid_slices: return
    vol = np.array([sitk.GetArrayFromImage(img) for img in valid_slices])
    
    generate_qc_montage(vol, OUTPUT_FOLDER, channel_idx=0, channel_name="DAPI")

    output_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_CenterOut_Aligned_CK.ome.tif")
    tifffile.imwrite(output_path, vol, photometric='minisblack', metadata={'axes': 'ZCYX'}, compression='zlib')
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()