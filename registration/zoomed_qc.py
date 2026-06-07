"""
zoomed_qc.py
============
Wrapper for denoise_volume.py that generates high-resolution, cropped QC 
visualizations of artifact regions for a specific core, slice, and channel.

Usage:
  python zoomed_qc.py --core_name Core_01 --z_idx 5 --channel DAPI --crop_size 500
"""

import os
import sys
import argparse
import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# IMPORT INTERCEPTION
# denoise_volume.py parses sys.argv at the global module level. 
# We temporarily mask sys.argv to only pass what it requires (--core_name)
# so it initializes its global variables without throwing an ArgumentError.
# -------------------------------------------------------------------------
_original_argv = sys.argv.copy()

_core_name_val = "Unknown"
if '--core_name' in _original_argv:
    _idx = _original_argv.index('--core_name')
    if _idx + 1 < len(_original_argv):
        _core_name_val = _original_argv[_idx + 1]

sys.argv = [_original_argv[0], '--core_name', _core_name_val]

# It is now safe to import the source-of-truth logic
import denoise_volume
import config 

# Restore the original command line arguments for zoomed_qc's own parser
sys.argv = _original_argv

def get_largest_artifact_center(mask: np.ndarray) -> tuple:
    """
    Locate the centroid of the largest contiguous artifact in the mask.
    Returns (y, x) coordinates. If no artifact is found, returns the image center.
    """
    if not mask.any():
        return mask.shape[0] // 2, mask.shape[1] // 2
        
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    
    if n_labels <= 1:
        return mask.shape[0] // 2, mask.shape[1] // 2
        
    # Index 0 is the background. Find the largest component among the rest.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    
    cx, cy = centroids[largest_idx]
    return int(cy), int(cx)

def get_crop_slices(cy: int, cx: int, h: int, w: int, crop_size: int) -> tuple:
    """
    Calculate safe numpy slice indices for a crop_size square centered at (cy, cx).
    """
    half = crop_size // 2
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    
    # Adjust if hitting boundaries to ensure consistent crop size where possible
    if y1 - y0 < crop_size:
        if y0 == 0: y1 = min(h, crop_size)
        else:       y0 = max(0, h - crop_size)
    if x1 - x0 < crop_size:
        if x0 == 0: x1 = min(w, crop_size)
        else:       x0 = max(0, w - crop_size)
        
    return slice(y0, y1), slice(x0, x1)

def plot_zoomed_artifact(
    raw_slice: np.ndarray,
    results: dict,
    y_slice: slice,
    x_slice: slice,
    core_name: str,
    z_idx: int,
    ch_name: str,
    out_dir: str
) -> None:
    """
    Generates a 2x2 high-resolution panel of the target bounding box
    with dynamically scaled typography relative to the figure size.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract crops
    raw_crop       = raw_slice[y_slice, x_slice]
    dust_crop      = results['dust_mask'][y_slice, x_slice]
    inpainted_crop = results['inpainted'][y_slice, x_slice]
    cleaned_crop   = results['cleaned'][y_slice, x_slice]

    n_dust_px = int(dust_crop.sum())
    
    # 1. Parameterize figure dimensions
    fig_w, fig_h = 16, 16  # Increased base size for better high-res viewing
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axs = axes.flat

    # 2. Calculate dynamic typography scales based on figure size
    base_font = int(min(fig_w, fig_h) * 1.5)
    title_font = base_font + 8
    subtitle_font = base_font + 2

    fig.suptitle(
        f'{core_name} | Z={z_idx:03d} | Channel: {ch_name}\n'
        f'Zoomed Region: Y[{y_slice.start}:{y_slice.stop}], X[{x_slice.start}:{x_slice.stop}]',
        fontsize=title_font, fontweight='bold', fontfamily='monospace'
    )

    # 1. Raw Crop
    axs[0].imshow(denoise_volume._stretch(raw_crop), cmap='gray', interpolation='nearest')
    axs[0].set_title(f'Raw Input (Zoomed)\nMax: {float(raw_crop.max()):.0f}', fontsize=subtitle_font)
    axs[0].axis('off')

    # 2. Artifact Detection Mask Overlay
    rgb_raw = np.stack([denoise_volume._stretch(raw_crop)] * 3, axis=-1)
    if n_dust_px > 0:
        rgb_raw[dust_crop, 0] = 1.0
        rgb_raw[dust_crop, 1] = 0.0
        rgb_raw[dust_crop, 2] = 0.0
    axs[1].imshow(rgb_raw, interpolation='nearest')
    axs[1].set_title(f'Stage 1: Detection Mask\n({n_dust_px:,} local px masked)', fontsize=subtitle_font)
    axs[1].axis('off')

    # 3. Inpainted / Topography Fill
    axs[2].imshow(denoise_volume._stretch(inpainted_crop), cmap='gray', interpolation='nearest')
    axs[2].set_title('Intermediate: Inpainted Base', fontsize=subtitle_font)
    axs[2].axis('off')

    # 4. Final Cleaned
    axs[3].imshow(denoise_volume._stretch(cleaned_crop), cmap='gray', interpolation='nearest')
    axs[3].set_title('Stage 2: Final Cleaned & BG-Corrected', fontsize=subtitle_font)
    axs[3].axis('off')

    # 4. Adjust top boundary to accommodate the dynamically larger suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_name = f"{core_name}_Z{z_idx:03d}_{ch_name}_zoomed_qc.png"
    out_path = os.path.join(out_dir, out_name)
    
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Zoomed QC plot saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate zoomed QC plot for artifact inspection.')
    parser.add_argument('--core_name', type=str, required=True)
    parser.add_argument('--z_idx', type=int, required=True, help='Z-slice index (0-based)')
    parser.add_argument('--channel', type=str, required=True, help='Channel name (e.g. DAPI) or index')
    parser.add_argument('--crop_size', type=int, default=500, help='Width/Height of the zoomed region in pixels')
    args = parser.parse_args()

    input_vol_path = os.path.join(
        config.DATASPACE,
        'Filter_AKAZE_RoMaV2_Linear_Warp_map',
        args.core_name,
        f'{args.core_name}_AKAZE_RoMaV2_Linear_Aligned.ome.tif',
    )

    if not os.path.exists(input_vol_path):
        print(f"Error: Input volume not found: {input_vol_path}")
        sys.exit(1)

    # Resolve channel index
    ch_idx = None
    if args.channel.isdigit():
        ch_idx = int(args.channel)
        ch_name = denoise_volume.CHANNEL_NAMES_ORDERED[ch_idx] if ch_idx < len(denoise_volume.CHANNEL_NAMES_ORDERED) else f"Ch{ch_idx:02d}"
    else:
        ch_name = args.channel
        if ch_name in denoise_volume.CHANNEL_NAMES_ORDERED:
            ch_idx = denoise_volume.CHANNEL_NAMES_ORDERED.index(ch_name)
        else:
            print(f"Error: Channel {ch_name} not found in configuration.")
            sys.exit(1)

    print(f"Loading slice Z={args.z_idx}, Channel={ch_name} ({ch_idx})...")
    
    # Efficient loading using memory map (assumes standard ZCYX order)
    vol = tifffile.imread(input_vol_path)
    
    if vol.ndim == 3:
        vol = vol[np.newaxis, np.newaxis]   # (1, 1, H, W)
    elif vol.ndim == 4:
        # Ensure ZCYX order
        if vol.shape[0] != 1 and vol.shape[1] == 1:
            vol = np.moveaxis(vol, 0, 1)    # CZYX -> ZCYX
            
    raw_slice = vol[args.z_idx, ch_idx].astype(np.float32)

    cfg = denoise_volume.CHANNEL_CONFIG.get(ch_name, dict(
        mode='gaussian', bg_sigma_um=40.0, artifact_thresh_factor=8.0
    ))

    print(f"Running core denoise logic on {ch_name}...")
    results = denoise_volume.denoise_channel(raw_slice, cfg, ch_name)

    print("Locating optimal zoom region...")
    cy, cx = get_largest_artifact_center(results['dust_mask'])
    y_slice, x_slice = get_crop_slices(cy, cx, raw_slice.shape[0], raw_slice.shape[1], args.crop_size)
    
    print(f"Target located at Y={cy}, X={cx}. Slicing bounds: Y[{y_slice.start}:{y_slice.stop}], X[{x_slice.start}:{x_slice.stop}]")
    
    # Define output directory relative to the existing denoise structure
    out_dir = os.path.join(config.DATASPACE, 'Denoised', args.core_name, 'qc_zoomed')
    
    plot_zoomed_artifact(
        raw_slice=raw_slice,
        results=results,
        y_slice=y_slice,
        x_slice=x_slice,
        core_name=args.core_name,
        z_idx=args.z_idx,
        ch_name=ch_name,
        out_dir=out_dir
    )
    
if __name__ == '__main__':
    main()