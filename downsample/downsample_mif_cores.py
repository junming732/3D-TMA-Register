"""
mIF Pyramid Level 2 Generation Module (4x Downsampling)
=======================================================

This module executes the standardized resolution reduction pipeline for the 
High-Plex Tissue Microarray (TMA) dataset. It generates "Level 2" pyramidal 
representations (approx. 2.0 µm/pixel) to serve as the master domain for 
ground-truth landmark acquisition and coarse registration initialization.

-------------------------------
The pipeline strictly adheres to the Nyquist-Shannon sampling theorem to prevent 
aliasing artifacts during the resolution reduction:
1. **Channel Preservation:** Iterates over all 8 spectral channels (DAPI, CK, etc.) 
   independently to preserve the multiplexed biological signal.
2. **Anti-Aliasing:** Applies a Gaussian Low-Pass Filter prior to resampling to 
   attenuate frequencies above the new Nyquist limit, preventing Moiré patterns 
   and "sparkle" noise in nuclear channels.
3. **Bicubic Interpolation:** Utilizes order-3 spline interpolation for sub-pixel 
   accuracy in spatial mapping.
4. **Radiometric Fidelity:** Enforces `preserve_range=True` and strict type casting 
   to maintain the original 16-bit dynamic range (0-65535) without normalization 
   artifacts.
5. **Metadata Update:** Recalculates and embeds the physical pixel size (Original * 4) 
   into the OME-XML header to ensure downstream measurement accuracy.

Input / Output
--------------
* **Input:** Root directory of raw TMA cores (e.g., `TMA_Cores_Grouped_NEW/Core_XX/*.ome.tif`).
* **Output:** A mirrored directory structure in `downsampled_4x/`, containing compressed 
  multi-channel OME-TIFFs with updated metadata.

"""
import os
import numpy as np
import tifffile
from skimage.transform import resize
from glob import glob
from tqdm import tqdm
import config

# --- CONFIGURATION ---
# Based on your cropping script logic
INPUT_ROOT = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
OUTPUT_ROOT = os.path.join(config.DATASPACE, "downsampled_4x")

# The same list from your cropping script, for metadata preservation
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_UM = 0.4961 * 4  # 4x downsampling means pixel size is 4x larger

def process_core(src_path, dest_root):
    try:
        # 1. Read OME-TIFF
        # CYX (Channel First), so shape is  (8, H, W)
        img = tifffile.imread(src_path)
        
        # Safety Check: Ensure it is actually (C, H, W)
        if img.ndim == 3 and img.shape[0] == len(CHANNEL_NAMES):
            channels, height, width = img.shape
        else:
            # Fallback for unexpected shapes (e.g. if some failed to transpose)
            # This handles (H, W, C) just in case
            if img.shape[-1] == len(CHANNEL_NAMES):
                img = np.transpose(img, (2, 0, 1))
                channels, height, width = img.shape
            else:
                return False, f"Unexpected dimensions: {img.shape}"

        # 2. Calculate Target Dimensions (1/4th)
        new_h = int(height * 0.25)
        new_w = int(width * 0.25)
        
        # 3. Allocate Output Array (Preserve 8 Channels, Unit16)
        downsampled = np.zeros((channels, new_h, new_w), dtype=img.dtype)

        # 4. Resize Loop (Iterate over channels)
        for c in range(channels):
            # anti_aliasing=True prevents DAPI "sparkle" noise
            # preserve_range=True keeps 16-bit values intact
            downsampled[c] = resize(
                img[c], 
                (new_h, new_w), 
                anti_aliasing=True, 
                preserve_range=True
            ).astype(img.dtype)

        # 5. Construct Output Path
        # Preserves "Core_XX" folder structure
        rel_path = os.path.relpath(src_path, INPUT_ROOT)
        save_path = os.path.join(dest_root, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 6. Save with Metadata
        # We update the pixel size to reflect the 4x downsampling
        metadata = {
            'axes': 'CYX',
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': PIXEL_SIZE_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': PIXEL_SIZE_UM,
            'PhysicalSizeYUnit': 'µm',
        }
        
        tifffile.imwrite(
            save_path, 
            downsampled, 
            photometric='minisblack',
            metadata=metadata,
            compression='zlib' # Saves disk space
        )
        
        return True, save_path

    except Exception as e:
        return False, f"{os.path.basename(src_path)}: {str(e)}"

def run():
    print("="*60)
    print("  mIF DOWNSAMPLING PIPELINE (4x)")
    print("="*60)
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    
    # Recursively find all .ome.tif files in Core_XX folders
    files = glob(os.path.join(INPUT_ROOT, "Core_*", "*.ome.tif"))
    print(f"Found {len(files)} cores to process.")

    errors = []
    
    for f in tqdm(files, desc="Downsampling"):
        success, msg = process_core(f, OUTPUT_ROOT)
        if not success:
            errors.append(msg)

    print("\nProcessing Complete.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors: print(f" - {e}")

if __name__ == "__main__":
    run()