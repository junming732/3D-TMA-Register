"""
Preprocessing — conform all TMA core slices to a canonical (H, W) shape.

Runs once across the entire dataset before any registration pipeline.
For each core folder, the canonical shape is determined by the most common
(H, W) among all slices in that core. Mismatched slices are centre-cropped
or symmetrically zero-padded to match, then written to the output folder.

Slices that already match the canonical shape are copied as-is (no rewrite).

Usage:
    python preprocess_conform_tma.py

Output structure mirrors the input:
    OUTPUT_BASE/
        Core_01/
            240919_3D_BL_TMA_1_Core01.ome.tif   ← conformed
            240919_3D_BL_TMA_2_Core01.ome.tif   ← conformed
            ...
        Core_02/
            ...
"""

import os
import sys
import logging
import shutil
from collections import Counter

import numpy as np
import tifffile
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_BASE  = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
OUTPUT_BASE = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")

# If True, log a summary of what would be done without writing any files.
DRY_RUN = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_as_chw(path: str) -> np.ndarray:
    """Load a .ome.tif and return it as (C, H, W) uint16."""
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        # (H, W, C) → (C, H, W)
        arr = np.moveaxis(arr, -1, 0)
    return arr


def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Centre-crop or symmetrically zero-pad (C, H, W) to (C, target_h, target_w).
    Identical logic to the registration pipeline so outputs are guaranteed compatible.
    """
    c, h, w = arr.shape
    out    = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0 = max(0, (h - target_h) // 2)
    dst_y0 = max(0, (target_h - h) // 2)
    copy_h = min(h - src_y0, target_h - dst_y0)
    src_x0 = max(0, (w - target_w) // 2)
    dst_x0 = max(0, (target_w - w) // 2)
    copy_w = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
        arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out


def canonical_shape(tif_paths: list) -> tuple:
    """
    Return the (H, W) that appears most frequently across all slices in a core.
    In case of a tie, the larger shape wins (prefer crop over pad for data integrity).
    """
    shapes = []
    for p in tif_paths:
        arr = load_as_chw(p)
        shapes.append((arr.shape[1], arr.shape[2]))  # (H, W)

    counts = Counter(shapes)
    # Sort by count descending, then by total pixels descending to break ties
    ranked = sorted(counts.items(), key=lambda x: (x[1], x[0][0] * x[0][1]), reverse=True)
    best_shape, best_count = ranked[0]

    logger.info(
        f"  Shape distribution: "
        + ", ".join(f"{h}×{w}×{cnt}" for (h, w), cnt in ranked)
    )
    logger.info(f"  Canonical shape selected: {best_shape[0]}×{best_shape[1]} "
                f"({best_count}/{len(tif_paths)} slices already match)")
    return best_shape


# ─────────────────────────────────────────────────────────────────────────────
# Per-core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_core(core_name: str, input_dir: str, output_dir: str) -> dict:
    """
    Conform all slices in one core folder to the canonical shape.
    Returns a summary dict for the final report.
    """
    tif_paths = sorted(glob.glob(os.path.join(input_dir, "*.ome.tif")))
    if not tif_paths:
        logger.warning(f"[{core_name}] No .ome.tif files found — skipping.")
        return {"core": core_name, "n_slices": 0, "n_conformed": 0, "n_copied": 0,
                "canonical_h": None, "canonical_w": None}

    logger.info(f"[{core_name}] {len(tif_paths)} slices found.")

    target_h, target_w = canonical_shape(tif_paths)

    if not DRY_RUN:
        os.makedirs(output_dir, exist_ok=True)

    n_conformed = 0
    n_copied    = 0

    for src_path in tif_paths:
        fname    = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)

        arr = load_as_chw(src_path)
        h, w = arr.shape[1], arr.shape[2]

        if h == target_h and w == target_w:
            # Shape already matches — copy without touching pixel data
            if not DRY_RUN:
                shutil.copy2(src_path, dst_path)
            n_copied += 1
        else:
            logger.info(
                f"  {fname}: {h}×{w} → {target_h}×{target_w} "
                f"({'crop' if h > target_h or w > target_w else 'pad'})"
            )
            if not DRY_RUN:
                conformed = conform_slice(arr, target_h, target_w)
                tifffile.imwrite(
                    dst_path,
                    conformed,
                    photometric='minisblack',
                    metadata={'axes': 'CYX'},
                    compression='deflate',
                    compressionargs={'level': 6},
                )
            n_conformed += 1

    logger.info(
        f"[{core_name}] Done — {n_copied} copied, {n_conformed} conformed."
    )
    return {
        "core":        core_name,
        "n_slices":    len(tif_paths),
        "n_conformed": n_conformed,
        "n_copied":    n_copied,
        "canonical_h": target_h,
        "canonical_w": target_w,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if DRY_RUN:
        logger.info("DRY RUN — no files will be written.")

    # Discover all core folders
    core_dirs = sorted([
        d for d in glob.glob(os.path.join(INPUT_BASE, "*"))
        if os.path.isdir(d)
    ])

    if not core_dirs:
        logger.error(f"No core folders found under {INPUT_BASE}")
        sys.exit(1)

    logger.info(f"Found {len(core_dirs)} core folders under {INPUT_BASE}")
    logger.info(f"Output base: {OUTPUT_BASE}")

    results = []
    for core_dir in core_dirs:
        core_name  = os.path.basename(core_dir)
        output_dir = os.path.join(OUTPUT_BASE, core_name)
        summary    = process_core(core_name, core_dir, output_dir)
        results.append(summary)

    # Final report
    total_slices    = sum(r["n_slices"]    for r in results)
    total_conformed = sum(r["n_conformed"] for r in results)
    total_copied    = sum(r["n_copied"]    for r in results)

    logger.info("─" * 60)
    logger.info(f"COMPLETE — {len(results)} cores processed")
    logger.info(f"  Total slices : {total_slices}")
    logger.info(f"  Conformed    : {total_conformed}")
    logger.info(f"  Copied as-is : {total_copied}")

    if total_conformed > 0:
        logger.info("Cores with conformed slices:")
        for r in results:
            if r["n_conformed"] > 0:
                logger.info(
                    f"  {r['core']:15s} — {r['n_conformed']} conformed "
                    f"to {r['canonical_h']}×{r['canonical_w']}"
                )

    logger.info(f"Output written to: {OUTPUT_BASE}")
    logger.info(
        "Update DATA_BASE_PATH in your registration script to point to "
        f"{OUTPUT_BASE} and the shape mismatch warnings will never appear again."
    )


if __name__ == "__main__":
    main()