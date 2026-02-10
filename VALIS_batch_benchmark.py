"""
VALIS Batch Benchmark
=====================
Description: 
    Scans all registered cores and generates a unified ranking table 
    based on NMI (Texture), SSIM (Structure), and Dice (Shape).
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import tifffile
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import config

# --- Tuning ---
DOWNSAMPLE = 8  # Higher = Faster, less precise. 8 is a good balance for WSI.

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def calc_dice(im1, im2):
    """Calculates overlap of tissue area (requires background segmentation)."""
    # Simple Otsu threshold to separate tissue from background
    try:
        t1 = threshold_otsu(im1)
        t2 = threshold_otsu(im2)
        m1 = im1 > t1
        m2 = im2 > t2
        
        intersection = np.logical_and(m1, m2).sum()
        total = m1.sum() + m2.sum()
        
        return (2.0 * intersection) / total if total > 0 else 0
    except:
        return 0

def process_core(core_path):
    """Audits a single core and returns stats."""
    slides_dir = os.path.join(core_path, "registered_slides")
    if not os.path.exists(slides_dir):
        return None

    files = sorted(glob.glob(os.path.join(slides_dir, "*.ome.tiff")), key=get_slice_number)
    if len(files) < 2:
        return None

    nmi_scores = []
    ssim_scores = []
    dice_scores = []

    for i in range(len(files) - 1):
        try:
            # Read and Downsample
            img1 = tifffile.imread(files[i], key=0)[::DOWNSAMPLE, ::DOWNSAMPLE]
            img2 = tifffile.imread(files[i+1], key=0)[::DOWNSAMPLE, ::DOWNSAMPLE]

            # 1. NMI (Texture)
            nmi_val = nmi(img1, img2, bins=100)
            nmi_scores.append(nmi_val)

            # 2. SSIM (Structure) - Data range is important!
            data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
            ssim_val = ssim(img1, img2, data_range=data_range)
            ssim_scores.append(ssim_val)

            # 3. Dice (Shape)
            dice_val = calc_dice(img1, img2)
            dice_scores.append(dice_val)
            
        except Exception as e:
            print(f"[WARN] Error in {os.path.basename(core_path)} slice {i}: {e}")

    return {
        "NMI_Mean": np.mean(nmi_scores),
        "NMI_Min": np.min(nmi_scores), # The "Weakest Link"
        "SSIM_Mean": np.mean(ssim_scores),
        "Dice_Mean": np.mean(dice_scores),
        "Slice_Count": len(files)
    }

def main():
    # Define where all your cores live
    ROOT_DIR = os.path.join(config.DATASPACE, "Registered")
    all_cores = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    
    results = []
    print(f"--- Starting Benchmark of {len(all_cores)} Cores ---")

    for core in all_cores:
        print(f"Processing {core}...", end="\r")
        core_path = os.path.join(ROOT_DIR, core)
        stats = process_core(core_path)
        
        if stats:
            stats["Core_ID"] = core
            results.append(stats)
        else:
            print(f"\n[SKIP] {core} (Missing data or <2 slides)")

    # Create Master Table
    df = pd.DataFrame(results)
    
    # Calculate a "Composite Score" for easy sorting
    # Weighting: 40% NMI, 40% SSIM, 20% Dice
    df["Composite_Score"] = (0.4 * df["NMI_Mean"]) + (0.4 * df["SSIM_Mean"]) + (0.2 * df["Dice_Mean"])
    
    # Sort by Composite Score (Best first)
    df = df.sort_values(by="Composite_Score", ascending=False)
    
    # Save
    out_path = os.path.join(config.DATASPACE, "Batch_Registration_Benchmark.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n\n[DONE] Benchmark saved to: {out_path}")
    print("\nTop 3 Cores:")
    print(df[["Core_ID", "Composite_Score", "NMI_Min"]].head(3))
    print("\nBottom 3 Cores (Inspect These!):")
    print(df[["Core_ID", "Composite_Score", "NMI_Min"]].tail(3))

if __name__ == "__main__":
    main()