"""
TMA Anchor Slice Detection Module
=================================

This module implements the "Minimum Valid Tissue" selection strategy for the 
3D TMA reconstruction pipeline. Its primary purpose is to mathematically identify 
the optimal "Anchor Slice" for manual landmark acquisition.

---------------------------------
The script isolates the Cytokeratin (CK) channel—chosen for its contiguous epithelial 
contrast—and applies a robust segmentation pipeline:
1. **Channel Extraction:** Isolates Index 6 (CK) from the 8-channel OME-TIFF.
2. **Contrast Enhancement:** Applies a Linear Stretch (0.5% - 99.5% percentile) to 
   normalize staining variations.
3. **Noise Suppression:** Applies a Gaussian Blur (sigma=15) to merge high-frequency 
   texture into cohesive tissue blobs.
4. **Segmentation:** Utilizes Triangle Thresholding (Zack's Algorithm) to generate 
   a binary tissue mask.
5. **Selection Logic:** Iterates through all Z-slices per core and selects the 
   slice with the *minimum* positive tissue area (>1000 pixels).

Input / Output
--------------
* **Input:** A root directory containing 4x Downsampled OME-TIFFs organized by Core ID 
  (e.g., `downsampled_4x/Core_01/*.ome.tif`).
* **Output:** A CSV Manifest (`anchor_slices_manifest.csv`) listing the specific filename 
  and tissue area for the selected anchor slice of each core.
"""
import os
import cv2
import numpy as np
import tifffile
import pandas as pd
from glob import glob
from tqdm import tqdm
import config

# --- CONFIGURATION ---
INPUT_ROOT = os.path.join(config.DATASPACE, "downsampled_4x")
OUTPUT_CSV = os.path.join(INPUT_ROOT, "anchor_slices_manifest.csv")

# From your cropping script
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
CK_INDEX = CHANNEL_NAMES.index('CK')  # Automatically finds index 6

def apply_linear_stretch(image, low_p=0.5, high_p=99.5):
    """
    Stretches histogram to full 0-255 range.
    Adapted strictly from your cropping logic.
    """
    # Safety check for empty images
    if image.max() == 0: return image
    
    p_min, p_max = np.percentile(image[image > 0], (low_p, high_p))
    # Clip and stretch
    stretched = np.clip((image - p_min) / (p_max - p_min + 1e-5) * 255, 0, 255).astype(np.uint8)
    return stretched

def get_tissue_area(filepath):
    """
    Returns the tissue area (in pixels) for the CK channel.
    """
    try:
        # 1. Load OME-TIFF
        # Expected shape: (8, H, W) based on your downsampling script
        img = tifffile.imread(filepath)
        
        # 2. Extract CK Channel
        ck_channel = img[CK_INDEX, :, :]
        
        # 3. Preprocessing (Your verified pipeline)
        # A. Linear Stretch
        stretched = apply_linear_stretch(ck_channel)
        
        # B. Gaussian Blur (Essential for clean thresholding)
        blur = cv2.GaussianBlur(stretched, (15, 15), 0)
        
        # C. Triangle Thresholding
        # returns (threshold_value, binary_image)
        thresh_val, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        
        # D. Optional: Quick Morphological Cleanup (Remove dust)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. Count Pixels
        area = np.count_nonzero(binary)
        return area, None

    except Exception as e:
        return -1, str(e)

def scan_dataset():
    print("="*60)
    print("  ANCHOR SLICE DETECTOR (Criteria: Min Area in CK)")
    print("="*60)
    
    # 1. Find all Core folders
    core_folders = sorted(glob(os.path.join(INPUT_ROOT, "Core_*")))
    
    results = []
    
    for folder in tqdm(core_folders, desc="Scanning Cores"):
        core_id = os.path.basename(folder)
        
        # Get all slices in this core folder
        slice_files = glob(os.path.join(folder, "*.ome.tif"))
        
        if not slice_files:
            continue
            
        # Track min area for this core
        min_area = float('inf')
        best_slice = None
        
        # Iterate through slices
        for f in slice_files:
            area, err = get_tissue_area(f)
            
            if err:
                print(f"Error reading {os.path.basename(f)}: {err}")
                continue
            
            # Logic: We want the SMALLEST area that is NOT effectively empty
            # (assuming < 1000 pixels is junk/dust)
            if area > 1000 and area < min_area:
                min_area = area
                best_slice = f
        
        if best_slice:
            results.append({
                "Core_ID": core_id,
                "Anchor_Slice_Path": best_slice,
                "Anchor_Filename": os.path.basename(best_slice),
                "Tissue_Area_Px": min_area
            })
            
    # 2. Save Manifest
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print(f"Scanning Complete.")
    print(f"Manifest saved to: {OUTPUT_CSV}")
    print(f"Found anchors for {len(df)} cores.")
    
    # Preview
    print("\nSample Results:")
    print(df[['Core_ID', 'Anchor_Filename', 'Tissue_Area_Px']].head())

if __name__ == "__main__":
    scan_dataset()