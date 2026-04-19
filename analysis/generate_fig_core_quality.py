"""
Core Quality Representative Figure
====================================
Generates a 3-panel figure showing one representative slice from each
morphological category:

  Panel 1 — Usable       : Core 03, slice index 5
  Panel 2 — Scattered    : Core 19, slice index 5
  Panel 3 — Severely Damaged : Core 17, slice index 5

Preprocessing matches the registration pipeline exactly:
  log1p -> percentile normalise -> uint8 (norm_log)

Output saved to config.DATASPACE/Figures/fig_core_quality_examples.png

Usage
-----
    python generate_fig_core_quality.py
"""

import os
import sys
import glob
import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFORMED_ROOT = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
OUTPUT_DIR     = os.path.join(config.DATASPACE, "Figures")

CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# Representative selections: (core_folder, slice_index, category_label)
SELECTIONS = [
    ("Core_03", 5, "Usable"),
    ("Core_19", 5, "Scattered Structure"),
    ("Core_17", 5, "Severely Damaged"),
]
# ── END CONFIG ────────────────────────────────────────────────────────────────


def prepare_ck(img_arr: np.ndarray) -> np.ndarray:
    """
    Identical to prepare_ck() in akaze_linear_romav2_warp_map.py.
    Returns norm_log (uint8) for display.
    """
    img_float  = img_arr.astype(np.float32)
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_log


def load_slice(core_folder: str, slice_idx: int) -> np.ndarray:
    """
    Loads the OME-TIFF at position slice_idx (alphabetical sort order)
    from the given core folder, extracts and returns the CK channel.
    """
    tiffs = sorted(glob.glob(os.path.join(CONFORMED_ROOT, core_folder, "*.ome.tif")))
    if not tiffs:
        tiffs = sorted(glob.glob(os.path.join(CONFORMED_ROOT, core_folder, "*.tif")))

    if not tiffs:
        raise FileNotFoundError(
            f"No OME-TIFF files found in: "
            f"{os.path.join(CONFORMED_ROOT, core_folder)}"
        )
    if slice_idx >= len(tiffs):
        raise IndexError(
            f"Slice index {slice_idx} out of range — "
            f"{core_folder} has only {len(tiffs)} slices."
        )

    fpath = tiffs[slice_idx]
    print(f"  Loading: {os.path.basename(fpath)}")
    arr = tifffile.imread(fpath)
    if arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    return arr[CK_CHANNEL_IDX]


def generate_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("black")

    for ax, (core_folder, slice_idx, label) in zip(axes, SELECTIONS):
        print(f"\n{core_folder} ({label}), slice index {slice_idx}")
        ck_raw   = load_slice(core_folder, slice_idx)
        norm_log = prepare_ck(ck_raw)

        ax.imshow(norm_log, cmap="gray", interpolation="lanczos")
        ax.set_title(label, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.5, -0.04,
            f"{core_folder.replace('_', ' ')} — slice {slice_idx}",
            transform=ax.transAxes,
            ha="center", va="top",
            color="grey", fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout(pad=1.5)

    out = os.path.join(OUTPUT_DIR, "fig_core_quality_examples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"\n  Saved -> {out}")
    return out


def main():
    print("=" * 60)
    print("  CORE QUALITY REPRESENTATIVE FIGURE")
    print("=" * 60)
    print(f"  Data root  : {CONFORMED_ROOT}")
    print(f"  Channel    : [{CK_CHANNEL_IDX}] {CHANNEL_NAMES[CK_CHANNEL_IDX]}")
    print(f"  Output dir : {OUTPUT_DIR}\n")

    for core_folder, slice_idx, label in SELECTIONS:
        print(f"  {label:20s} : {core_folder}, slice index {slice_idx}")

    generate_figure()
    print("\nDone.")


if __name__ == "__main__":
    main()