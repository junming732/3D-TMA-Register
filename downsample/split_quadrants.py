"""
mIF Single-Core Quadrant Splitter (4D Support)
===============================================
Splits a large (Z, C, H, W) OME-TIFF into 4 quadrants at ORIGINAL resolution.

Usage:
    python split_quadrants.py --input /path/to/in.ome.tif --output /path/to/out_dir
"""

import os
import zlib
import argparse
import numpy as np

# --- Patch imagecodecs BEFORE importing tifffile ---
# When imagecodecs is installed, tifffile uses imagecodecs.deflate_decode
# (backed by strict libdeflate) instead of the pure-python fallback.
# We replace it with Python's zlib which handles the same streams correctly.
import imagecodecs

def _zlib_decode(data, out=None):
    return zlib.decompress(bytes(data), 15)

imagecodecs.deflate_decode = _zlib_decode
imagecodecs.zlib_decode    = _zlib_decode
print("Patched imagecodecs.deflate_decode + zlib_decode with Python zlib (wbits=15).")

import tifffile  # import AFTER patch so tifffile picks up patched imagecodecs

# --- CONSTANTS ---
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_UM = 0.4961

QUADRANTS = {
    'TL': (0, 0),
    'TR': (0, 1),
    'BL': (1, 0),
    'BR': (1, 1),
}

def split_file(src_path, output_dir):
    if not os.path.exists(src_path):
        print(f"Error: Input file missing: {src_path}")
        return

    try:
        print(f"Loading: {src_path}")
        img = tifffile.imread(src_path)
        print(f"Loaded shape: {img.shape}, dtype: {img.dtype}")

        if img.ndim == 4:
            z, c, h, w = img.shape
            axes = 'ZCYX'
            print(f"  Z-slices={z}, Channels={c}, H={h}, W={w}")
        elif img.ndim == 3:
            c, h, w = img.shape
            z = None
            axes = 'CYX'
            print(f"  Channels={c}, H={h}, W={w}")
        else:
            print(f"Error: Unsupported dimensions {img.shape}")
            return

        mid_h = h // 2
        mid_w = w // 2
        row_slices = [slice(0, mid_h), slice(mid_h, h)]
        col_slices = [slice(0, mid_w), slice(mid_w, w)]

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        if base_name.endswith('.ome'):
            base_name = base_name[:-4]

        metadata_base = {
            'axes': axes,
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': PIXEL_SIZE_UM,
            'PhysicalSizeY': PIXEL_SIZE_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm',
        }

        for label, (ri, ci_) in QUADRANTS.items():
            rs = row_slices[ri]
            cs = col_slices[ci_]
            quadrant = img[:, :, rs, cs] if z is not None else img[:, rs, cs]

            save_path = os.path.join(output_dir, f"{base_name}_{label}.ome.tif")
            tifffile.imwrite(
                save_path,
                quadrant,
                photometric='minisblack',
                metadata=metadata_base,
                compression='zlib'
            )
            print(f"  Saved {label}: {save_path}  (shape: {quadrant.shape})")

        print("\nAll quadrants saved successfully.")

    except Exception as e:
        print(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Source .ome.tif path")
    parser.add_argument("--output", required=True, help="Destination directory path")
    args = parser.parse_args()
    split_file(args.input, args.output)