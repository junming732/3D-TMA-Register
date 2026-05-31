"""
mIF Single-Core Downsampling Utility (4D Support)
================================================
Handles (Z, C, H, W) or (C, H, W) input formats.

Usage:
    python downsample_single_core.py --input /path/to/in.ome.tif --output /path/to/out_dir --scale 0.25
"""

import os
import argparse
import numpy as np
import tifffile
from skimage.transform import resize

# --- CONSTANTS ---
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
BASE_PIXEL_SIZE_UM = 0.4961
SECTION_THICKNESS_UM = 4.5

def downsample_file(src_path, output_dir, scale_factor=0.25):
    if not os.path.exists(src_path):
        print(f"Error: Input file missing: {src_path}")
        return

    try:
        # 1. Load Image
        img = tifffile.imread(src_path)
        print(f"Input Shape: {img.shape}")

        # 2. Handle 4D (Z, C, H, W) or 3D (C, H, W)
        if img.ndim == 4:
            z, c, h, w = img.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            downsampled = np.zeros((z, c, new_h, new_w), dtype=img.dtype)
            
            for i in range(z):
                for j in range(c):
                    downsampled[i, j] = resize(
                        img[i, j], 
                        (new_h, new_w), 
                        anti_aliasing=True, 
                        preserve_range=True
                    ).astype(img.dtype)
            axes = 'ZCYX'
            
        elif img.ndim == 3:
            c, h, w = img.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
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
        
        current_pixel_size = BASE_PIXEL_SIZE_UM / scale_factor

        # Strict parity with registration script metadata structure
        metadata = {
            'axes': axes,
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': current_pixel_size,
            'PhysicalSizeY': current_pixel_size,
            'PhysicalSizeZ': SECTION_THICKNESS_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZUnit': 'µm',
        }
        
        tifffile.imwrite(
            save_path, 
            downsampled, 
            photometric='minisblack',
            metadata=metadata,
            compression='deflate',
            compressionargs={'level': 6}
        )
        print(f"Done: {save_path} (Output Shape: {downsampled.shape})")

    except Exception as e:
        print(f"Process failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source .ome.tif path")
    parser.add_argument("--output", required=True, help="Destination directory path")
    parser.add_argument(
        "--scale", 
        type=float, 
        default=0.25, 
        help="Scaling factor (e.g., 0.25 for 4x downsample, 0.125 for 8x downsample)"
    )
    
    args = parser.parse_args()
    downsample_file(args.input, args.output, args.scale)