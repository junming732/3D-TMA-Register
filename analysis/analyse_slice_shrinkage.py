"""
Slice 1 Shrinkage Analysis
==========================
For each core, compares the tissue mask area of slice 1 (the first file
alphabetically) against all remaining slices of the SAME core.

Preprocessing matches the registration pipeline exactly:
  prepare_ck()       -> log1p -> percentile normalise -> uint8 (norm_log)
  build_tissue_mask()-> Otsu on non-zero pixels of norm_log -> dilate
  tissue area        -> np.sum(mask > 0)

This gives an absolute tissue footprint per slice, so a shrunken slice
genuinely returns a smaller count than a full-coverage slice.

Produces two figures saved to config.DATASPACE/Shrinkage_Analysis/:
  1. slice1_shrinkage_per_core.png  -- grouped bar: slice 1 mask area vs
                                       mean mask area of remaining slices.
  2. slice1_shrinkage_summary.png   -- boxplot: normalised slice 1 area vs
                                       all other slices, across all cores.

Usage
-----
    python analyse_slice_shrinkage.py
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
import matplotlib.ticker as mticker

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ── CONFIG — matches akaze_linear_romav2_warp_map.py exactly ─────────────────
CONFORMED_ROOT   = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
OUTPUT_DIR       = os.path.join(config.DATASPACE, "Shrinkage_Analysis")

CK_CHANNEL_IDX   = 6
MASK_DILATE_PX   = 20
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# Cores where slice 1 (index 0) was excluded from your analysis
EXCLUDED_SLICE1  = [1, 3, 4, 5, 11, 12, 13, 14, 16, 18, 19, 20, 24, 26]
# ── END CONFIG ────────────────────────────────────────────────────────────────


def prepare_ck(img_arr: np.ndarray):
    """
    Identical to prepare_ck() in akaze_linear_romav2_warp_map.py.
    Returns norm_log (uint8) used for tissue masking.
    """
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log  = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_log


def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """
    Identical to build_tissue_mask() in akaze_linear_romav2_warp_map.py.
    Otsu threshold on non-zero pixels + morphological dilation.
    Returns uint8 (H, W): 255 = tissue, 0 = background.
    """
    img     = ck_log.astype(np.uint8)
    nonzero = img[img > 0]
    if len(nonzero) == 0:
        return np.zeros_like(img)
    thresh, _ = cv2.threshold(
        nonzero.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = (img > thresh).astype(np.uint8) * 255
    if MASK_DILATE_PX > 0:
        r    = MASK_DILATE_PX
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kern)
    return mask


def tissue_area(ck_raw: np.ndarray) -> int:
    """
    Runs the full pipeline preprocessing and returns the number of tissue
    pixels in the mask — an absolute measure of tissue footprint.
    """
    norm_log = prepare_ck(ck_raw)
    mask     = build_tissue_mask(norm_log)
    return int(np.sum(mask > 0))


def load_slices_from_core(core_dir: str) -> list:
    """
    Loads all OME-TIFFs sorted alphabetically, extracts CK channel,
    returns [tissue_area_slice0, tissue_area_slice1, ...].
    """
    tiffs = sorted(glob.glob(os.path.join(core_dir, "*.ome.tif")))
    if not tiffs:
        tiffs = sorted(glob.glob(os.path.join(core_dir, "*.tif")))

    areas = []
    for fpath in tiffs:
        try:
            arr = tifffile.imread(fpath)
            if arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
                arr = np.moveaxis(arr, -1, 0)
            ck = arr[CK_CHANNEL_IDX]
            areas.append(tissue_area(ck))
        except Exception as e:
            print(f"  Warning: could not read {os.path.basename(fpath)}: {e}")
    return areas


def collect_all_cores() -> dict:
    """Returns {core_id: [area_slice0, area_slice1, ...]} for every Core_XX."""
    core_dirs = sorted(glob.glob(os.path.join(CONFORMED_ROOT, "Core_*")))
    if not core_dirs:
        raise FileNotFoundError(
            f"No Core_XX folders found under:\n  {CONFORMED_ROOT}"
        )
    print(f"Found {len(core_dirs)} core folders.\n")
    results = {}
    for cd in core_dirs:
        core_name = os.path.basename(cd)
        core_id   = int(core_name.split("_")[-1])
        print(f"  Processing {core_name} ...", end=" ", flush=True)
        areas = load_slices_from_core(cd)
        if areas:
            print(f"{len(areas)} slices  |  "
                  f"Slice 1: {areas[0]:,}  |  Mean rest: {np.mean(areas[1:]):,.0f}")
        else:
            print("no slices found.")
        results[core_id] = areas
    return results


# ── FIGURE 1: grouped bar chart ───────────────────────────────────────────────

def plot_bar(results: dict):
    core_ids    = sorted(results.keys())
    slice1_vals = []
    rest_means  = []
    labels      = []
    excluded    = []

    for cid in core_ids:
        areas = results[cid]
        if len(areas) < 2:
            continue
        slice1_vals.append(areas[0])
        rest_means.append(np.mean(areas[1:]))
        labels.append(f"Core {cid:02d}")
        excluded.append(cid in EXCLUDED_SLICE1)

    x, width = np.arange(len(labels)), 0.38

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width / 2, slice1_vals, width,
           label="Slice 1 (index 0)", color="#d62728", alpha=0.85)
    ax.bar(x + width / 2, rest_means, width,
           label="Mean of remaining slices (same core)", color="#1f77b4", alpha=0.85)

    for i, excl in enumerate(excluded):
        if excl:
            ax.bar(x[i] - width / 2, slice1_vals[i], width,
                   color="#d62728", alpha=0.3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tissue Mask Area (pixels)")
    ax.set_title("Slice 1 vs Mean of Remaining Slices — Per Core (Within-Core Comparison)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.01,
             f"Tissue area = pixels inside Otsu tissue mask (log-normalised CK channel, "
             f"index {CK_CHANNEL_IDX}). Hatched: slice 1 excluded from analysis.",
             fontsize=7, color="grey")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUTPUT_DIR, "slice1_shrinkage_per_core.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {out}")


# ── FIGURE 2: summary boxplot ─────────────────────────────────────────────────

def plot_boxplot(results: dict):
    """
    Normalise each slice's mask area by its core's median, then pool
    slice 1 vs all other slices across all cores for the boxplot.
    """
    slice1_norm, rest_norm = [], []

    for cid in sorted(results):
        areas = results[cid]
        if len(areas) < 2:
            continue
        core_median = np.median(areas)
        if core_median == 0:
            continue
        slice1_norm.append(areas[0] / core_median)
        rest_norm.extend(a / core_median for a in areas[1:])

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(
        [slice1_norm, rest_norm],
        labels=["Slice 1\n(index 0)", "Slices 2+\n(index 1–19)"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("#d62728"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#1f77b4"); bp["boxes"][1].set_alpha(0.7)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Core median")
    ax.set_ylabel("Normalised Tissue Mask Area\n(relative to per-core median)")
    ax.set_title("Within-Core Comparison:\nSlice 1 vs Remaining Slices")
    ax.legend(fontsize=8)

    for i, data in enumerate([slice1_norm, rest_norm], start=1):
        med = np.median(data)
        ax.text(i, med + 0.02, f"{med:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "slice1_shrinkage_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  SLICE 1 SHRINKAGE ANALYSIS  (within-core comparison)")
    print("=" * 60)
    print(f"  Data root  : {CONFORMED_ROOT}")
    print(f"  Channel    : [{CK_CHANNEL_IDX}] {CHANNEL_NAMES[CK_CHANNEL_IDX]}")
    print(f"  Masking    : log1p -> Otsu (non-zero) -> dilate {MASK_DILATE_PX}px")
    print(f"  Output dir : {OUTPUT_DIR}\n")

    results = collect_all_cores()

    print("\nGenerating figures...")
    plot_bar(results)
    plot_boxplot(results)

    print("\n-- Within-core ratio: slice 1 / mean of rest ----------------")
    ratios = []
    for cid in sorted(results):
        areas = results[cid]
        if len(areas) < 2:
            continue
        ratio = areas[0] / np.mean(areas[1:])
        ratios.append(ratio)
        flag = " <- reduced" if ratio < 0.75 else ""
        print(f"  Core {cid:02d}: {ratio:.2f}{flag}")

    if ratios:
        print(f"\n  Median ratio across all cores : {np.median(ratios):.2f}")
        print(f"  Cores with slice 1 < 75%      : "
              f"{sum(r < 0.75 for r in ratios)}/{len(ratios)}")
    print("=" * 60)


if __name__ == "__main__":
    main()