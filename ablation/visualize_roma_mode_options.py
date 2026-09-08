"""
Preview ROMA_MODE ablation input options for a single core/slice, without
running RoMaV2 or AKAZE.

Grid generator and image-builder functions are copied verbatim from
test_roma_mode_ablation_3.py, so output here is pixel-for-pixel identical to
what the ablation fed RoMaV2.

Usage:
    # Montage of every mode for one slice (downsampled overview)
    python visualize_roma_mode_options.py --core_name Core_09 --slice_id 5

    # Single mode, saved as a native-resolution crop
    python visualize_roma_mode_options.py --core_name Core_09 --slice_id 5 \
        --mode clahe_visual_additive_rgb_3ch_fusion --crop_size 768

    # Crop centered on a specific pixel instead of the image center
    python visualize_roma_mode_options.py --core_name Core_09 --slice_id 5 \
        --mode clahe_visual_additive_rgb_3ch_fusion --crop_size 768 --crop_center 2200,1400

    # List all available mode names
    python visualize_roma_mode_options.py --list_modes
"""
import os
import sys
import glob
import re
import argparse
import itertools

import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import exposure

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6
AF_CHANNEL_IDX   = 7


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim from test_roma_mode_ablation_3.py so previews match exactly
# what the ablation measured.
# ─────────────────────────────────────────────────────────────────────────────

def generate_ablation_grid():
    norm_methods    = ['clahe', 'log1p']
    output_formats  = ['direct_rgb', 'registration_color_lut', 'visual_additive_rgb']
    weight_combinations = [
        {'name': 'DAPI_only',  'weights': {0: 1.0}},
        {'name': 'CK_only',    'weights': {6: 1.0}},
        {'name': '2ch_fusion', 'weights': {0: 0.5,  6: 0.5}},
        {'name': '3ch_fusion', 'weights': {0: 0.33, 6: 0.33, 7: 0.33}},
        {'name': '7ch_fusion', 'weights': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}},
    ]
    grid = []
    for norm, fmt, combo in itertools.product(norm_methods, output_formats, weight_combinations):
        if fmt == 'direct_rgb' and combo['name'] == '7ch_fusion':
            continue
        if fmt in ['registration_color_lut', 'visual_additive_rgb'] and len(combo['weights']) == 1:
            continue
        config_name = f"{norm}_{fmt}_{combo['name']}"
        grid.append({
            'mode_name':   config_name,
            'norm_method': norm,
            'output_type': fmt,
            'weights':     combo['weights'],
        })
    return grid


def normalize_channel(img_arr: np.ndarray, method: str = 'clahe') -> np.ndarray:
    img_float = img_arr.astype(np.float32)
    if method == 'clahe':
        img01 = exposure.rescale_intensity(img_float, out_range=(0, 1))
        eq    = exposure.equalize_adapthist(img01)
        return exposure.rescale_intensity(eq, out_range=(0, 255)).astype(np.uint8)
    elif method == 'log1p':
        log_img    = np.log1p(img_float)
        p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
        return cv2.normalize(np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif method == 'linear':
        p_lo, p_hi = np.percentile(img_float[::4, ::4], (1.0, 99.0))
        return cv2.normalize(np.clip(img_float, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    raise ValueError(f"Unknown method: {method}")


def build_parametric_roma_input(vol: np.ndarray, weights: dict, norm_method: str, output_type: str) -> np.ndarray:
    h, w = vol.shape[1], vol.shape[2]
    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0

    if output_type == 'grayscale_duplicate':
        accumulator = np.zeros((h, w), dtype=np.float32)
        for ch_idx, weight in weights.items():
            if weight > 0:
                norm_img = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32)
                accumulator += norm_img * (weight / total_weight)
        gray_uint8 = np.clip(accumulator, 0, 255).astype(np.uint8)
        return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    elif output_type == 'direct_rgb':
        sorted_channels = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i, (ch_idx, _) in enumerate(sorted_channels):
            if i < 3:
                rgb[..., i] = normalize_channel(vol[ch_idx], method=norm_method)
        return rgb

    elif output_type == 'registration_color_lut':
        COLOR_LUT = {
            0: (0, 128, 255), 1: (51, 255, 51), 2: (255, 51, 51),
            3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0),
            6: (255, 128, 0), 7: (160, 160, 160),
        }
        acc = np.zeros((h, w, 3), dtype=np.float32)
        n_channels = 0
        for ch_idx, weight in weights.items():
            if weight > 0 and ch_idx in COLOR_LUT:
                norm_img  = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32) / 255.0
                color_arr = np.array(COLOR_LUT[ch_idx], dtype=np.float32) / 255.0
                acc += norm_img[..., None] * color_arr[None, None, :]
                n_channels += 1
        return np.clip(acc / max(n_channels, 1) * 255.0, 0, 255).astype(np.uint8)

    elif output_type == 'visual_additive_rgb':
        # Additive-blend mechanism and colors 0-6 match convert_tiff_RGB_script.py
        # exactly (that script's `colors` array, channels 0-6 only — its loop is
        # `for channel in range(7)`, so index 7 is defined but never rendered).
        #
        # AF (channel 7) below uses that same unused index-7 value, orange, rather
        # than the ablation's original choice of white. White was a poor choice:
        # DAPI (channel 0, above) is also white, just at higher intensity, so AF
        # and DAPI were chromatically indistinguishable — AF's contribution was
        # only visible via pixel-diffing, not by eye. Orange fixes that.
        #
        # IMPORTANT: this diverges from test_roma_mode_ablation_3.py, which still
        # uses white for AF (`np.array([1, 1, 1]) / 4`) — that's what actually
        # generated the reported TRE/NCC numbers. Swapping the tint changes the
        # real pixel values fed to RoMaV2, not just how it looks (orange has a
        # different total intensity than white, [1,0.5,0]/4 vs [1,1,1]/4), so
        # this script's visual_additive_rgb output no longer matches what the
        # ablation measured until test_roma_mode_ablation_3.py's VISUAL_COLORS
        # is updated to match and re-run. Use this for illustrating AF's spatial
        # pattern to a reviewer; don't present it as "what RoMaV2 saw" until the
        # ablation script is synced.
        VISUAL_COLORS = {
            0: np.array([1, 1, 1]) / 2,
            1: np.array([0, 1, 0]) / 4,
            2: np.array([1, 1, 0]) / 4,
            3: np.array([1, 0, 1]) / 4,
            4: np.array([0, 1, 1]) / 4,
            5: np.array([1, 0, 0]) / 4,
            6: np.array([0.5, 0, 1]) / 4,
            7: np.array([1, 0.5, 0]) / 4,  # AF — orange, matches convert_tiff_RGB_script.py's unused index-7 entry
        }
        rgb_acc = np.zeros((h, w, 3), dtype=np.float32)
        for ch_idx, weight in weights.items():
            if weight > 0 and ch_idx in VISUAL_COLORS:
                norm_img = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32) / 255.0
                rgb_acc += norm_img[..., None] * VISUAL_COLORS[ch_idx]
        rgb_acc = np.clip(rgb_acc, 0, 1)
        return (rgb_acc * 255).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown output type: {output_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Slice lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def find_slice_file(core_name: str, slice_id: int) -> str:
    input_dir = os.path.join(DATA_BASE_PATH, core_name)
    if not os.path.exists(input_dir):
        sys.exit(f"Input folder not found: {input_dir}")
    candidates = sorted(
        glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff")),
        key=get_slice_number,
    )
    candidates = [f for f in candidates if "_thumb" not in os.path.basename(f)]
    for f in candidates:
        if get_slice_number(f) == slice_id:
            return f
    found_ids = sorted({get_slice_number(f) for f in candidates})
    sys.exit(f"No slice with id {slice_id} in {input_dir}. Available ids: {found_ids}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preview ROMA_MODE ablation input options for one slice.")
    parser.add_argument('--core_name', type=str, help="e.g. Core_09")
    parser.add_argument('--slice_id',  type=int, help="Slice TMA id, e.g. 5")
    parser.add_argument('--mode',      type=str, default=None,
                        help="Single ablation mode_name to render full-size. "
                             "Omit to render a montage of every mode.")
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--crop_size', type=int, default=768,
                        help="Side length (px) of the square crop saved at native "
                             "resolution for --mode renders. Only applies with --mode; "
                             "the montage view is unaffected. Default 768.")
    parser.add_argument('--crop_center', type=str, default=None,
                        help="Optional 'x,y' pixel coordinates for the crop center. "
                             "Defaults to the image center if omitted.")
    parser.add_argument('--list_modes', action='store_true',
                        help="Print all available mode_name values and exit.")
    args = parser.parse_args()

    grid = generate_ablation_grid()

    if args.list_modes:
        for cfg in grid:
            print(cfg['mode_name'])
        return

    if not args.core_name or args.slice_id is None:
        parser.error("--core_name and --slice_id are required unless --list_modes is passed.")

    if args.mode:
        grid = [g for g in grid if g['mode_name'] == args.mode]
        if not grid:
            sys.exit(f"Unknown mode '{args.mode}'. Run with --list_modes to see valid names.")

    file_path = find_slice_file(args.core_name, args.slice_id)
    print(f"Loading {file_path} ...")
    vol = tifffile.imread(file_path).astype(np.float32)
    if vol.ndim == 3 and vol.shape[-1] < vol.shape[0]:
        vol = np.moveaxis(vol, -1, 0)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode:
        cfg = grid[0]
        img = build_parametric_roma_input(vol, cfg['weights'], cfg['norm_method'], cfg['output_type'])

        # Saved directly at native resolution (cv2.imwrite), deliberately bypassing
        # matplotlib's imshow/savefig path: imshow resamples the array to fit the
        # figure's pixel grid, which can smooth out exactly the fine noise texture
        # a full-res comparison is meant to show. A crop (rather than the whole
        # core) keeps file size sane while still being pixel-for-pixel real.
        h, w = img.shape[:2]
        crop = max(1, min(args.crop_size, h, w))
        if args.crop_center:
            cx, cy = (int(v) for v in args.crop_center.split(','))
        else:
            cx, cy = w // 2, h // 2
        x0 = max(0, min(w - crop, cx - crop // 2))
        y0 = max(0, min(h - crop, cy - crop // 2))
        crop_img = img[y0:y0 + crop, x0:x0 + crop]

        out_path = os.path.join(
            args.output_dir,
            f"{args.core_name}_slice{args.slice_id}_{cfg['mode_name']}"
            f"_fullres_crop{crop}_x{x0}_y{y0}.png"
        )
        cv2.imwrite(out_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        print(f"Saved native-resolution {crop}x{crop} crop (top-left at x={x0}, y={y0}, "
              f"full image was {w}x{h}) -> {out_path}")
        return
    else:
        n_cols = 4
        n_rows = (len(grid) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, cfg in zip(axes, grid):
            img = build_parametric_roma_input(vol, cfg['weights'], cfg['norm_method'], cfg['output_type'])
            ax.imshow(img)
            ax.set_title(cfg['mode_name'], fontsize=8)
            ax.axis('off')
        for ax in axes[len(grid):]:
            ax.axis('off')
        fig.suptitle(f"{args.core_name} slice {args.slice_id} — ROMA_MODE input previews", fontsize=14)
        out_path = os.path.join(
            args.output_dir, f"{args.core_name}_slice{args.slice_id}_roma_mode_preview.png"
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()