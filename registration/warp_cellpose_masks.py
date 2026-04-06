"""
warp_cellpose_masks.py
======================
Apply saved deformation maps (from akaze_linear_romav2_warp.py) to CellPose
segmentation masks that were run on the original, unregistered images.

Workflow
--------
1. Run CellPose on the original (unregistered) CD3 channel images → label masks.
2. Run the updated registration script → deformation_maps/*.npz are saved.
3. Run this script → warped masks aligned to the registered space.

Usage
-----
    python warp_cellpose_masks.py \
        --core_name  <CORE_NAME> \
        --mask_dir   <path/to/cellpose_masks>   \
        --deform_dir <path/to/deformation_maps> \
        --out_dir    <path/to/warped_masks>      \
        [--mask_suffix _cp_masks.tif]            \
        [--channel CD3]                          \
        [--plot_qc]

Mask filename convention expected
----------------------------------
The script matches each .npz deformation file to a mask file by slice ID.
Masks must contain the slice ID in their filename in the same TMA_<ID>_ format
as the original ome.tif slices, e.g.:
    TMA_007_CD3_cp_masks.tif

Output
------
For each slice, a warped mask TIFF is written to --out_dir with the suffix
_warped.tif.  The label IDs are preserved (INTER_NEAREST remap).

Notes on interpolation
----------------------
- Label masks      → always cv2.INTER_NEAREST  (integer cell IDs, no blending)
- Probability maps → use cv2.INTER_LINEAR       (pass --interp linear)
"""

import os
import re
import glob
import argparse
import logging
import numpy as np
import cv2
import tifffile

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Warp CellPose masks using saved registration deformation maps."
)
parser.add_argument('--core_name',    required=True,
                    help="TMA core name, e.g. Core_A1")
parser.add_argument('--mask_dir',     required=True,
                    help="Folder containing CellPose output masks (.tif)")
parser.add_argument('--deform_dir',   required=True,
                    help="Folder containing deformation .npz files "
                         "(deformation_maps/ sub-folder from registration output)")
parser.add_argument('--out_dir',      required=True,
                    help="Output folder for warped masks")
parser.add_argument('--mask_suffix',  default='_cp_masks.tif',
                    help="Filename suffix to identify mask files (default: _cp_masks.tif)")
parser.add_argument('--channel',      default='CD3',
                    help="Channel label used in mask filenames (default: CD3)")
parser.add_argument('--interp',       default='nearest',
                    choices=['nearest', 'linear'],
                    help="Interpolation: 'nearest' for label masks, "
                         "'linear' for probability maps (default: nearest)")
parser.add_argument('--plot_qc',      action='store_true',
                    help="Save side-by-side QC images (original vs warped mask)")
args = parser.parse_args()

INTERP_FLAG = cv2.INTER_NEAREST if args.interp == 'nearest' else cv2.INTER_LINEAR

os.makedirs(args.out_dir, exist_ok=True)
if args.plot_qc:
    qc_dir = os.path.join(args.out_dir, "qc_plots")
    os.makedirs(qc_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id_from_path(path: str) -> int | None:
    name = os.path.basename(path)
    # Mask files: TMA_016_CD3_cp_masks.tif
    match = re.search(r"TMA_(\d+)_", name)
    if match:
        return int(match.group(1))
    # Deformation maps: Core_01_Z014_ID016_deformation.npz
    match = re.search(r"ID(\d+)_", name)
    if match:
        return int(match.group(1))
    return None


def apply_deformation(mask: np.ndarray, npz_path: str,
                      interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """
    Apply a saved deformation .npz to a 2-D label mask.

    Step 1: affine warpAffine  (M_affine from AKAZE)
    Step 2: dense remap        (map_x, map_y from RoMaV2)

    Both steps are always applied — if a stage failed during registration
    the stored matrix/map is the identity, so it is a no-op.
    """
    d        = np.load(npz_path)
    M_affine = d['M_affine'].astype(np.float64)   # (2, 3)
    map_x    = d['map_x'].astype(np.float32)       # (H, W)
    map_y    = d['map_y'].astype(np.float32)       # (H, W)
    h, w     = int(d['orig_h']), int(d['orig_w'])
    akaze_ok = bool(d['akaze_ok'])
    warp_ok  = bool(d['warp_ok'])

    logger.debug(f"  AKAZE_OK={akaze_ok}  WARP_OK={warp_ok}  shape=({h},{w})")

    mask_f32 = mask.astype(np.float32)

    # Step 1: affine
    mask_affine = cv2.warpAffine(
        mask_f32, M_affine, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Step 2: dense remap
    warped = cv2.remap(
        mask_affine, map_x, map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    return warped.astype(mask.dtype)


def save_qc_plot(original_mask: np.ndarray,
                 warped_mask: np.ndarray,
                 slice_id: int,
                 out_path: str):
    """Side-by-side colour overlay of original vs warped cell boundaries."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def label_to_rgb(lbl):
            """Random colour per cell label for visualisation."""
            rng  = np.random.default_rng(seed=0)
            rgb  = np.zeros((*lbl.shape, 3), dtype=np.uint8)
            ids  = np.unique(lbl)
            ids  = ids[ids > 0]
            for cid in ids:
                col = rng.integers(80, 256, size=3, dtype=np.uint8)
                rgb[lbl == cid] = col
            return rgb

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(label_to_rgb(original_mask))
        axes[0].set_title(f"Original mask  ID={slice_id}\n"
                          f"({int((original_mask > 0).sum())} px labelled)", fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(label_to_rgb(warped_mask))
        axes[1].set_title(f"Warped mask  ID={slice_id}\n"
                          f"({int((warped_mask > 0).sum())} px labelled)", fontsize=11)
        axes[1].axis('off')

        fig.suptitle(f"CellPose mask warp QC — slice ID {slice_id}", fontsize=13,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"QC plot failed for slice {slice_id}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Collect mask files
    mask_pattern = os.path.join(args.mask_dir, f"*{args.channel}*{args.mask_suffix}")
    mask_files   = sorted(glob.glob(mask_pattern))

    if not mask_files:
        # Try without channel filter
        mask_pattern = os.path.join(args.mask_dir, f"*{args.mask_suffix}")
        mask_files   = sorted(glob.glob(mask_pattern))

    if not mask_files:
        logger.error(f"No mask files found matching: {mask_pattern}")
        return

    logger.info(f"Found {len(mask_files)} mask file(s) in {args.mask_dir}")

    # Build ID → npz lookup
    npz_files = glob.glob(os.path.join(args.deform_dir, "*.npz"))
    id_to_npz = {}
    for npz in npz_files:
        sid = get_slice_id_from_path(npz)
        if sid is not None:
            id_to_npz[sid] = npz
    logger.info(f"Found {len(id_to_npz)} deformation map(s) in {args.deform_dir}")

    n_ok   = 0
    n_skip = 0

    for mask_path in mask_files:
        sid = get_slice_id_from_path(mask_path)
        if sid is None:
            logger.warning(f"Cannot parse slice ID from: {os.path.basename(mask_path)} — skipping")
            n_skip += 1
            continue

        if sid not in id_to_npz:
            logger.warning(f"No deformation map found for slice ID {sid:03d} — skipping")
            n_skip += 1
            continue

        npz_path = id_to_npz[sid]
        logger.info(f"Warping slice ID {sid:03d}: {os.path.basename(mask_path)}")

        try:
            mask = tifffile.imread(mask_path)
        except Exception as exc:
            logger.error(f"Cannot read mask {mask_path}: {exc} — skipping")
            n_skip += 1
            continue

        if mask.ndim != 2:
            logger.warning(f"Mask is {mask.ndim}-D, expected 2-D; squeezing.")
            mask = mask.squeeze()
        if mask.ndim != 2:
            logger.error(f"Cannot reduce mask to 2-D: {mask.shape} — skipping")
            n_skip += 1
            continue

        try:
            warped = apply_deformation(mask, npz_path, interpolation=INTERP_FLAG)
        except Exception as exc:
            logger.error(f"Warp failed for slice {sid}: {exc} — skipping")
            n_skip += 1
            continue

        # Write warped mask
        base     = os.path.splitext(os.path.basename(mask_path))[0]
        out_name = f"{base}_warped.tif"
        out_path = os.path.join(args.out_dir, out_name)
        tifffile.imwrite(out_path, warped, compression='deflate')
        logger.info(f"  → {out_path}  (labels: {len(np.unique(warped)) - 1} cells)")

        if args.plot_qc:
            qc_path = os.path.join(qc_dir, f"{base}_warp_qc.png")
            save_qc_plot(mask, warped, sid, qc_path)

        n_ok += 1

    logger.info(f"Done. Warped: {n_ok} | Skipped: {n_skip}")


if __name__ == "__main__":
    main()