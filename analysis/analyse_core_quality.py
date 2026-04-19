"""
Core Quality Analysis
=====================
Measures mean tissue mask area per core (averaged across all its slices)
using the same preprocessing as the registration pipeline, then visualises
the result coloured by morphological category: Usable / Scattered / Damaged.

Preprocessing matches akaze_linear_romav2_warp_map.py exactly:
  prepare_ck()       -> log1p -> percentile normalise -> uint8 (norm_log)
  build_tissue_mask()-> Otsu on non-zero pixels of norm_log -> dilate
  tissue area        -> np.sum(mask > 0)

Produces two figures saved to config.DATASPACE/Core_Quality_Analysis/:
  1. core_quality_bar.png  -- bar chart: mean tissue area per core,
                              coloured by category, grouped by category.
  2. core_quality_box.png  -- boxplot: tissue area distribution per
                              category with individual points overlaid.

Usage
-----
    python analyse_core_quality.py
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
import matplotlib.patches as mpatches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ── CONFIG — matches akaze_linear_romav2_warp_map.py exactly ─────────────────
CONFORMED_ROOT = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
OUTPUT_DIR     = os.path.join(config.DATASPACE, "Core_Quality_Analysis")

CK_CHANNEL_IDX = 6
MASK_DILATE_PX = 20
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# Morphological classification (from your manual review)
USABLE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 24, 26]
SCATTERED = [6, 19, 20, 22, 25, 28, 29, 30]
DAMAGED   = [16, 17, 21, 23, 27]

CAT_COLORS = {
    "Usable":    "#1f77b4",
    "Scattered": "#ff7f0e",
    "Damaged":   "#d62728",
}

def category(core_id: int) -> str:
    if core_id in USABLE:    return "Usable"
    if core_id in SCATTERED: return "Scattered"
    if core_id in DAMAGED:   return "Damaged"
    return "Unknown"
# ── END CONFIG ────────────────────────────────────────────────────────────────


def prepare_ck(img_arr: np.ndarray) -> np.ndarray:
    """Identical to prepare_ck() in akaze_linear_romav2_warp_map.py."""
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log  = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_log


def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """Identical to build_tissue_mask() in akaze_linear_romav2_warp_map.py."""
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
    norm_log = prepare_ck(ck_raw)
    mask     = build_tissue_mask(norm_log)
    return int(np.sum(mask > 0))


def mean_tissue_area_for_core(core_dir: str) -> float:
    """Mean tissue mask area across all slices of one core."""
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

    return float(np.mean(areas)) if areas else 0.0


def collect_all_cores() -> dict:
    """Returns {core_id: mean_tissue_area} for every Core_XX folder."""
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
        mean_area = mean_tissue_area_for_core(cd)
        results[core_id] = mean_area
        print(f"mean mask area = {mean_area:,.0f} px  [{category(core_id)}]")
    return results


# ── FIGURE 1: bar chart grouped by category ───────────────────────────────────

def plot_bar(results: dict):
    order = (
        sorted(c for c in results if c in USABLE) +
        sorted(c for c in results if c in SCATTERED) +
        sorted(c for c in results if c in DAMAGED)
    )
    labels = [f"Core {c:02d}" for c in order]
    values = [results[c] for c in order]
    colors = [CAT_COLORS[category(c)] for c in order]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(order)), values, color=colors, alpha=0.85, width=0.7)

    n_usable    = len([c for c in order if c in USABLE])
    n_scattered = len([c for c in order if c in SCATTERED])
    for xd in [n_usable - 0.5, n_usable + n_scattered - 0.5]:
        ax.axvline(xd, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Tissue Mask Area per Core (pixels)")
    ax.set_title("Mean Tissue Mask Area per Core — Grouped by Morphological Category")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    legend_patches = [
        mpatches.Patch(color=CAT_COLORS[cat], alpha=0.85,
                       label=f"{cat} (n={sum(1 for c in results if category(c) == cat)})")
        for cat in ["Usable", "Scattered", "Damaged"]
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.text(0.01, 0.01,
             f"Tissue area = pixels inside Otsu tissue mask (log-normalised CK channel, "
             f"index {CK_CHANNEL_IDX}).",
             fontsize=7, color="grey")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUTPUT_DIR, "core_quality_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {out}")


# ── FIGURE 2: boxplot per category ────────────────────────────────────────────

def plot_boxplot(results: dict):
    groups = {
        "Usable":    [results[c] for c in results if c in USABLE],
        "Scattered": [results[c] for c in results if c in SCATTERED],
        "Damaged":   [results[c] for c in results if c in DAMAGED],
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(
        list(groups.values()),
        labels=list(groups.keys()),
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, cat in zip(bp["boxes"], groups.keys()):
        patch.set_facecolor(CAT_COLORS[cat])
        patch.set_alpha(0.75)

    for i, (cat, vals) in enumerate(groups.items(), start=1):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=CAT_COLORS[cat], edgecolors="black",
                   linewidths=0.5, s=40, zorder=3, alpha=0.9)

    ax.set_ylabel("Mean Tissue Mask Area per Core (pixels)")
    ax.set_title("Tissue Area Distribution\nby Morphological Category")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "core_quality_box.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  CORE QUALITY ANALYSIS  (cross-core comparison)")
    print("=" * 60)
    print(f"  Data root  : {CONFORMED_ROOT}")
    print(f"  Channel    : [{CK_CHANNEL_IDX}] {CHANNEL_NAMES[CK_CHANNEL_IDX]}")
    print(f"  Masking    : log1p -> Otsu (non-zero) -> dilate {MASK_DILATE_PX}px")
    print(f"  Output dir : {OUTPUT_DIR}\n")

    results = collect_all_cores()

    print("\nGenerating figures...")
    plot_bar(results)
    plot_boxplot(results)

    print("\n-- Mean tissue area by category ------------------------------")
    for cat in ["Usable", "Scattered", "Damaged"]:
        vals = [results[c] for c in results if category(c) == cat]
        if vals:
            print(f"  {cat:10s} (n={len(vals):2d}):  "
                  f"mean={np.mean(vals):>10,.0f}  "
                  f"median={np.median(vals):>10,.0f}  "
                  f"min={np.min(vals):>10,.0f}  "
                  f"max={np.max(vals):>10,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()