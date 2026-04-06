"""
verify_3d_linking.py
====================
Visual verification of 3D cell linking by annotating each slice with
the 3D cell ID as a number at the cell centroid location.

Produces one PNG per slice showing:
  - Raw warped mask (cells as white blobs)
  - 3D cell ID number printed at each cell centroid
  - Cells that appear in multiple slices highlighted in a different colour

This lets you visually confirm that the same physical cell gets the same
3D ID across adjacent slices without relying on colours.

Usage
-----
    python verify_3d_linking.py \\
        --core_name  Core_01  \\
        --channel    CD3      \\
        [--n_slices  4]       \\   # how many slices to render (default: all)
        [--max_cells 200]         # limit labels per slice for readability
"""

import os
import sys
import re
import glob
import argparse
import logging
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--core_name',  required=True)
parser.add_argument('--channel',    default='CD3')
parser.add_argument('--n_slices',   type=int, default=None,
                    help="Number of slices to verify (default: all)")
parser.add_argument('--max_cells',  type=int, default=200,
                    help="Max cell labels to draw per slice for readability (default: 200)")
args = parser.parse_args()

WARPED_DIR = os.path.join(config.DATASPACE,
                          f"CellPose_{args.channel}_Warped", args.core_name)
LABEL_DIR  = os.path.join(config.DATASPACE,
                          f"CellPose_{args.channel}_3D",     args.core_name)
OUT_DIR    = os.path.join(LABEL_DIR, "linking_verification")
os.makedirs(OUT_DIR, exist_ok=True)


def get_slice_id(path):
    m = re.search(r"TMA_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


# Load warped masks and 3D label volume
warped_files = sorted(
    glob.glob(os.path.join(WARPED_DIR, f"*{args.channel}*_warped.tif")),
    key=get_slice_id
)
label_file = os.path.join(LABEL_DIR, f"{args.core_name}_{args.channel}_3d_labels.tif")

if not warped_files:
    logger.error(f"No warped masks found in {WARPED_DIR}")
    sys.exit(1)
if not os.path.exists(label_file):
    logger.error(f"3D label volume not found: {label_file}")
    sys.exit(1)

slice_ids    = [get_slice_id(f) for f in warped_files]
label_volume = tifffile.imread(label_file)   # (Z, H, W)
n_slices     = len(warped_files)

if args.n_slices is not None:
    n_slices = min(args.n_slices, n_slices)

logger.info(f"Verifying {n_slices} slices, max {args.max_cells} labels each")

# Find which 3D IDs appear in more than one slice (truly linked cells)
id_slice_count = {}
for z in range(label_volume.shape[0]):
    for cid in np.unique(label_volume[z]):
        if cid == 0:
            continue
        id_slice_count[int(cid)] = id_slice_count.get(int(cid), 0) + 1
multi_slice_ids = {cid for cid, count in id_slice_count.items() if count > 1}
logger.info(f"3D cells spanning >1 slice: {len(multi_slice_ids)}")

# -----------------------------------------------------------------------------
# One PNG per slice
# -----------------------------------------------------------------------------
for z in range(n_slices):
    sid        = slice_ids[z]
    raw_mask   = tifffile.imread(warped_files[z]).astype(np.uint32)
    label_slc  = label_volume[z]   # (H, W) with 3D IDs

    H, W = raw_mask.shape

    # Get unique 3D IDs in this slice (excluding background)
    unique_ids = np.unique(label_slc)
    unique_ids = unique_ids[unique_ids > 0]

    # Subsample for readability if too many
    if len(unique_ids) > args.max_cells:
        rng = np.random.default_rng(seed=z)
        unique_ids = rng.choice(unique_ids, size=args.max_cells, replace=False)
        unique_ids = np.sort(unique_ids)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Background: binary mask in grey
    ax.imshow((raw_mask > 0).astype(np.uint8), cmap='gray',
              vmin=0, vmax=1, interpolation='nearest')

    # Draw ID number at centroid of each cell
    for cid in unique_ids:
        cell_pixels = (label_slc == cid)
        if not np.any(cell_pixels):
            continue
        cy, cx = center_of_mass(cell_pixels)

        # Multi-slice cells in red, singletons in yellow
        is_multi = int(cid) in multi_slice_ids
        color    = 'red' if is_multi else 'yellow'
        fontsize = 5

        ax.text(cx, cy, str(int(cid)),
                fontsize=fontsize, color=color,
                ha='center', va='center',
                fontweight='bold' if is_multi else 'normal')

    ax.set_title(
        f"{args.core_name} {args.channel} — Slice ID {sid}  (Z{z})\n"
        f"{len(unique_ids)} cells shown  |  "
        f"RED = spans multiple slices  |  YELLOW = singleton\n"
        f"Total 3D cells in slice: {len(np.unique(label_slc)) - 1}",
        fontsize=10
    )
    ax.axis('off')

    out_path = os.path.join(OUT_DIR, f"verify_Z{z:02d}_ID{sid:03d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")

# -----------------------------------------------------------------------------
# Cross-slice comparison: show same cell across adjacent slices
# -----------------------------------------------------------------------------
logger.info("Generating cross-slice comparison for multi-slice cells...")

# Pick a sample of multi-slice cells to show side by side
sample_multi = sorted(list(multi_slice_ids))[:20]

for cid in sample_multi:
    # Find which slices this cell appears in
    slices_with_cell = [z for z in range(label_volume.shape[0])
                        if np.any(label_volume[z] == cid)]

    n_panels = len(slices_with_cell)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, z in zip(axes, slices_with_cell):
        label_slc = label_volume[z]
        cell_mask = (label_slc == cid)

        # Crop a 200x200 window around the cell centroid
        cy, cx = center_of_mass(cell_mask)
        cy, cx = int(cy), int(cx)
        r      = 100
        y0, y1 = max(0, cy - r), min(label_slc.shape[0], cy + r)
        x0, x1 = max(0, cx - r), min(label_slc.shape[1], cx + r)

        crop_mask  = (tifffile.imread(warped_files[z])[y0:y1, x0:x1] > 0).astype(np.uint8)
        crop_label = label_slc[y0:y1, x0:x1]

        ax.imshow(crop_mask, cmap='gray', vmin=0, vmax=1)
        # Highlight this cell in red
        highlight = np.zeros((*crop_mask.shape, 4), dtype=np.float32)
        highlight[crop_label == cid] = [1, 0, 0, 0.5]
        ax.imshow(highlight)
        ax.set_title(f"Z{z} (ID {slice_ids[z]})\n3D cell #{cid}", fontsize=8)
        ax.axis('off')

    fig.suptitle(f"Cross-slice verification — 3D cell #{cid}", fontsize=10,
                 fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"crossslice_cell{cid:06d}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

logger.info(f"Done. Verification images -> {OUT_DIR}")