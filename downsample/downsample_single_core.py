"""
mIF Single-Core Downsampling Utility (4D Support)
================================================
Handles (T, C, H, W) or (C, H, W) input formats.

Usage:
    python downsample_single_core.py --input /path/to/in.ome.tif --output /path/to/out_dir
"""

import os
import argparse
import numpy as np
import tifffile
from skimage.transform import resize

# --- CONSTANTS ---
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_UM = 0.4961 * 4 

def downsample_file(src_path, output_dir):
    if not os.path.exists(src_path):
        print(f"Error: Input file missing: {src_path}")
        return

    try:
        # 1. Load Image
        img = tifffile.imread(src_path)
        print(f"Input Shape: {img.shape}")

        # 2. Handle 4D (T, C, H, W) or 3D (C, H, W)
        if img.ndim == 4:
            t, c, h, w = img.shape
            new_h, new_w = int(h * 0.25), int(w * 0.25)
            downsampled = np.zeros((t, c, new_h, new_w), dtype=img.dtype)
            
            for i in range(t):
                for j in range(c):
                    downsampled[i, j] = resize(
                        img[i, j], 
                        (new_h, new_w), 
                        anti_aliasing=True, 
                        preserve_range=True
                    ).astype(img.dtype)
            axes = 'TCYX'
        elif img.ndim == 3:
            c, h, w = img.shape
            new_h, new_w = int(h * 0.25), int(w * 0.25)
            downsampled = np.zeros((c, new_h, new_w), dtype=img.dtype)
            
            for j in range(c):
                downsampled[j] = resize(
                    img[j], 
                    (new_h, new_w), 
                    anti_aliasing=True, 
                    preserve_range=True
                ).astype(img.dtype)
            axes = 'CYX'
        else:
            print(f"Error: Unsupported dimensions {img.shape}")
            return

        # 3. Save
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(src_path))
        
        metadata = {
            'axes': axes,
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': PIXEL_SIZE_UM,
            'PhysicalSizeY': PIXEL_SIZE_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm',
        }
        
        tifffile.imwrite(
            save_path, 
            downsampled, 
            photometric='minisblack',
            metadata=metadata,
            compression='zlib'
        )
        print(f"Done: {save_path} (Output Shape: {downsampled.shape})")

    except Exception as e:
        print(f"Process failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source .ome.tif path")
    parser.add_argument("--output", required=True, help="Destination directory path")
    
    args = parser.parse_args()
    downsample_file(args.input, args.output)