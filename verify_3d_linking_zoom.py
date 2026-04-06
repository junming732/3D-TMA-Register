"""
verify_3d_linking_zoom.py
=========================
Zoomed verification of 3D cell linking.
Picks a small ROI and shows it side-by-side across all slices
with large readable 3D cell ID numbers at each cell centroid.

Usage
-----
    python verify_3d_linking_zoom.py \\
        --core_name  Core_01  \\
        --channel    CD3      \\
        [--roi_x     2000]    \\  # centre X of ROI in pixels
        [--roi_y     2000]    \\  # centre Y of ROI in pixels
        [--roi_size  400]     \\  # width/height of ROI in pixels
        [--n_slices  4]           # how many adjacent slices to compare
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

parser = argparse.ArgumentParser()
parser.add_argument('--core_name', required=True)
parser.add_argument('--channel',   default='CD3')
parser.add_argument('--roi_x',     type=int, default=None,
                    help="Centre X of ROI (default: image centre)")
parser.add_argument('--roi_y',     type=int, default=None,
                    help="Centre Y of ROI (default: image centre)")
parser.add_argument('--roi_size',  type=int, default=400,
                    help="ROI width and height in pixels (default: 400)")
parser.add_argument('--n_slices',  type=int, default=4,
                    help="Number of adjacent slices to compare (default: 4)")
parser.add_argument('--z_start',   type=int, default=0,
                    help="Starting Z index (default: 0)")
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


warped_files = sorted(
    glob.glob(os.path.join(WARPED_DIR, f"*{args.channel}*_warped.tif")),
    key=get_slice_id
)
label_file = os.path.join(LABEL_DIR,
                          f"{args.core_name}_{args.channel}_3d_labels.tif")

if not warped_files:
    logger.error(f"No warped masks in {WARPED_DIR}")
    sys.exit(1)
if not os.path.exists(label_file):
    logger.error(f"Label volume not found: {label_file}")
    sys.exit(1)

slice_ids    = [get_slice_id(f) for f in warped_files]
label_volume = tifffile.imread(label_file)   # (Z, H, W)

H, W      = label_volume.shape[1], label_volume.shape[2]
z_start   = args.z_start
n_slices  = min(args.n_slices, len(warped_files) - z_start)

# Default ROI to image centre
cx = args.roi_x if args.roi_x is not None else W // 2
cy = args.roi_y if args.roi_y is not None else H // 2
r  = args.roi_size // 2

x0, x1 = max(0, cx - r), min(W, cx + r)
y0, y1 = max(0, cy - r), min(H, cy + r)

logger.info(f"ROI: x={x0}:{x1}  y={y0}:{y1}  ({x1-x0}x{y1-y0}px)")

# Which 3D IDs appear in more than one slice
id_slice_count = {}
for z in range(label_volume.shape[0]):
    for cid in np.unique(label_volume[z]):
        if cid > 0:
            id_slice_count[int(cid)] = id_slice_count.get(int(cid), 0) + 1
multi_slice_ids = {cid for cid, c in id_slice_count.items() if c > 1}

# -----------------------------------------------------------------------------
# Side-by-side ROI comparison across n_slices
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 6))
if n_slices == 1:
    axes = [axes]

for zi in range(n_slices):
    z   = z_start + zi
    ax  = axes[zi]
    sid = slice_ids[z]

    raw   = tifffile.imread(warped_files[z]).astype(np.uint32)
    label = label_volume[z]

    # Crop ROI
    raw_crop   = (raw[y0:y1, x0:x1] > 0).astype(np.uint8)
    label_crop = label[y0:y1, x0:x1]

    ax.imshow(raw_crop, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    # Draw each cell ID in the ROI
    unique_ids = np.unique(label_crop)
    unique_ids = unique_ids[unique_ids > 0]

    for cid in unique_ids:
        cell_pix = (label_crop == cid)
        if not np.any(cell_pix):
            continue
        pcy, pcx = center_of_mass(cell_pix)
        is_multi = int(cid) in multi_slice_ids
        color    = 'red' if is_multi else 'yellow'
        ax.text(pcx, pcy, str(int(cid)),
                fontsize=7, color=color,
                ha='center', va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='black', alpha=0.4, linewidth=0))

    n_multi_here = sum(1 for cid in unique_ids if int(cid) in multi_slice_ids)
    ax.set_title(
        f"Slice ID {sid}  (Z{z})\n"
        f"{len(unique_ids)} cells in ROI\n"
        f"RED={n_multi_here} multi-slice  YELLOW={len(unique_ids)-n_multi_here} singleton",
        fontsize=9
    )
    ax.set_xlabel(f"x: {x0}–{x1}", fontsize=7)
    ax.set_ylabel(f"y: {y0}–{y1}", fontsize=7)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

fig.suptitle(
    f"{args.core_name} {args.channel} — ROI zoom comparison\n"
    f"ROI centre: ({cx}, {cy})  size: {args.roi_size}px\n"
    f"Same RED number in adjacent slices = correctly linked 3D cell",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out_path = os.path.join(OUT_DIR,
    f"zoom_x{cx}_y{cy}_size{args.roi_size}_Z{z_start}to{z_start+n_slices-1}.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
logger.info(f"Saved: {out_path}")
logger.info("To verify: find a RED number in Z0 and check it appears at the "
            "same location in Z1, Z2 etc.")