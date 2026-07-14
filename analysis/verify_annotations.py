"""
verify_annotations.py
────────────────────────────
Overlay rough annotations onto raw and registered images to confirm
x/y/z coordinates are correct after registration.

All folder paths derived from config.py — same as akaze_linear_romav2_warp_map.py.

Usage
─────
    python verify_annotations_core09.py \
        --core_name core_09 \
        --annotation_json /path/to/rough_annotation_core_09.json \
        [--channel_idx 6]       # CK channel (default 6)
        [--mclass all]          # 'all' or comma-separated ints, e.g. '0,1,2'
        [--crop_half 300]       # half-size of crop window in pixels
        [--skip_registered]     # skip registered overlay if OME-TIFF not ready

z-index mapping (from annotator spec):  slice_idx = z_json + 10
  z_json=-10 → slice 0,  z_json=9 → slice 19
"""

import os, re, sys, json, glob, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TABLEAU_COLORS
from collections import defaultdict

# ── config.py — same path resolution as the registration script ───────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--core_name',       type=str, required=True)
parser.add_argument('--annotation_json', type=str, required=True)
parser.add_argument('--channel_idx',     type=int, default=6)
parser.add_argument('--mclass',          type=str, default='all')
parser.add_argument('--crop_half',       type=int, default=300)
parser.add_argument('--skip_registered', action='store_true')
args = parser.parse_args()

TARGET_CORE = args.core_name

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS — mirrors akaze_linear_romav2_warp_map.py exactly ─────────────────
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)

WORK_OUTPUT    = os.path.join(config.DATASPACE, "Filter_AKAZE_RoMaV2_Linear_Warp_map")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
DEFORM_FOLDER  = os.path.join(OUTPUT_FOLDER, "deformation_maps")
REGISTERED_TIF = os.path.join(OUTPUT_FOLDER,
                               f"{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif")
VERIFY_OUTPUT  = os.path.join(OUTPUT_FOLDER, "annotation_verification")
os.makedirs(VERIFY_OUTPUT, exist_ok=True)

CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
CHANNEL_NAME  = (CHANNEL_NAMES[args.channel_idx]
                 if args.channel_idx < len(CHANNEL_NAMES) else str(args.channel_idx))

logger.info(f"Core          : {TARGET_CORE}")
logger.info(f"Input folder  : {INPUT_FOLDER}")
logger.info(f"Deform folder : {DEFORM_FOLDER}")
logger.info(f"Registered TIF: {REGISTERED_TIF}")
logger.info(f"Verify output : {VERIFY_OUTPUT}")

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

# ─── Z MAPPING ────────────────────────────────────────────────────────────────
def z_json_to_slice_idx(z_json):
    return z_json + 10

# ─── FILE LIST — mirrors main() in registration script exactly ────────────────
# Registration script: raw_files = glob(INPUT_FOLDER, "*.ome.tif"), sorted by TMA number
import tifffile

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

FILE_LIST = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")), key=get_slice_number)

if not FILE_LIST:
    logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
    sys.exit(1)

logger.info(f"Found {len(FILE_LIST)} .ome.tif slices")

# Determine canonical shape from centre slice — same logic as registration script
_ctr = tifffile.imread(FILE_LIST[len(FILE_LIST) // 2])
if _ctr.ndim == 2:
    _ctr = _ctr[np.newaxis]
elif _ctr.ndim == 3 and _ctr.shape[-1] < _ctr.shape[0]:
    _ctr = np.moveaxis(_ctr, -1, 0)   # (H,W,C) -> (C,H,W)
N_CH, TARGET_H, TARGET_W = _ctr.shape
logger.info(f"Canonical shape: C={N_CH}, H={TARGET_H}, W={TARGET_W}")
del _ctr

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_raw_slice(slice_idx, channel_idx):
    """
    Load one channel — layout normalised to (C,H,W) exactly as load_slice()
    in the registration script. Returns (H,W) float32, or (None, None).
    """
    if slice_idx >= len(FILE_LIST):
        logger.warning(f"slice_idx={slice_idx} out of range ({len(FILE_LIST)} files).")
        return None, None
    path = FILE_LIST[slice_idx]
    arr  = tifffile.imread(path)
    # --- same normalisation as registration script's load_slice() ---
    if arr.ndim == 2:
        arr = arr[np.newaxis]                     # (1, H, W)
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)             # (H, W, C) -> (C, H, W)
    # else already (C, H, W)
    ch  = min(channel_idx, arr.shape[0] - 1)
    return arr[ch].astype(np.float32), path


def load_registered_slice(slice_idx, channel_idx, reg_vol=None):
    arr = reg_vol if reg_vol is not None else tifffile.imread(REGISTERED_TIF)
    if arr.ndim == 4:    # ZCYX
        return arr[slice_idx, channel_idx].astype(np.float32)
    elif arr.ndim == 3:  # ZHW
        return arr[slice_idx].astype(np.float32)
    return arr.astype(np.float32)


def find_deform_npz(slice_idx):
    pattern = os.path.join(DEFORM_FOLDER,
                           f"{TARGET_CORE}_Z{slice_idx:03d}_*_deformation.npz")
    m = glob.glob(pattern)
    if m:
        return m[0]
    pattern2 = os.path.join(DEFORM_FOLDER,
                            f"{TARGET_CORE}_Z{slice_idx}_*_deformation.npz")
    m2 = glob.glob(pattern2)
    return m2[0] if m2 else None


def apply_deformation_to_point(x, y, npz_path):
    """
    Forward-warp (x,y) from moving-slice space into registered space.
    Mirrors the two-step logic of apply_deformation_to_mask():
      Step 1 — affine:  warpAffine with M_affine
      Step 2 — remap:   cv2.remap with map_x / map_y  (inverse remap)
    For a single point we invert step 2 numerically via nearest-neighbour search.
    """
    d        = np.load(npz_path)
    M_affine = d['M_affine'].astype(np.float64)   # (2, 3)
    map_x    = d['map_x'].astype(np.float32)       # (H, W)
    map_y    = d['map_y'].astype(np.float32)

    # Step 1: forward affine
    pt     = np.array([x, y, 1.0], dtype=np.float64)
    xy_aff = M_affine @ pt
    ax, ay = float(xy_aff[0]), float(xy_aff[1])

    # Step 2: invert the dense remap — find dst pixel whose source is closest to (ax,ay)
    H, W     = map_x.shape
    search_r = 150
    x0 = max(0, int(ax) - search_r);  x1 = min(W, int(ax) + search_r)
    y0 = max(0, int(ay) - search_r);  y1 = min(H, int(ay) + search_r)
    patch_x = map_x[y0:y1, x0:x1]
    patch_y = map_y[y0:y1, x0:x1]
    dist2   = (patch_x - ax)**2 + (patch_y - ay)**2
    ry, rx  = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(x0 + rx), float(y0 + ry)


def log_norm_uint8(img):
    log        = np.log1p(img)
    p_lo, p_hi = np.percentile(log[::4, ::4], (0.5, 99.5))
    clipped    = np.clip(log, p_lo, p_hi)
    norm       = (clipped - p_lo) / (p_hi - p_lo) * 255 if p_hi > p_lo else np.zeros_like(clipped)
    return norm.astype(np.uint8)


def make_crop(img_u8, cx, cy, half):
    H, W  = img_u8.shape
    x0 = max(0, cx - half);  x1 = min(W, cx + half)
    y0 = max(0, cy - half);  y1 = min(H, cy + half)
    return img_u8[y0:y1, x0:x1], cx - x0, cy - y0

# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data['annotations']
logger.info(f"Loaded {len(annotations)} annotations.")

if args.mclass.lower() != 'all':
    target_classes = set(int(m) for m in args.mclass.split(','))
    annotations    = [a for a in annotations if a['mclass'] in target_classes]
    logger.info(f"After mclass filter {target_classes}: {len(annotations)} annotations.")

all_mclasses  = sorted(set(a['mclass'] for a in annotations))
colour_list   = list(TABLEAU_COLORS.values())
mclass_colour = {mc: colour_list[i % len(colour_list)] for i, mc in enumerate(all_mclasses)}

by_z = defaultdict(list)
for ann in annotations:
    by_z[ann['z']].append(ann)
logger.info(f"z values: {sorted(by_z.keys())}")

# ─── OPTIONALLY PRE-LOAD REGISTERED VOLUME ────────────────────────────────────
reg_vol = None
if not args.skip_registered:
    if os.path.isfile(REGISTERED_TIF):
        logger.info("Loading registered volume …")
        reg_vol = tifffile.imread(REGISTERED_TIF)
        logger.info(f"  Shape: {reg_vol.shape}")
    else:
        logger.warning(f"Registered TIF not found: {REGISTERED_TIF}")
        logger.warning("  Only RAW overlays will be produced.")

# ─── PER-Z CROP PANELS ────────────────────────────────────────────────────────
for z_json, anns_at_z in sorted(by_z.items()):
    slice_idx = z_json_to_slice_idx(z_json)
    logger.info(f"z_json={z_json}  slice_idx={slice_idx}  ({len(anns_at_z)} annotations)")

    raw_img, raw_path = load_raw_slice(slice_idx, args.channel_idx)
    if raw_img is None:
        logger.warning(f"  Skipping: could not load raw slice.")
        continue
    raw_u8 = log_norm_uint8(raw_img)
    logger.info(f"  Raw: {os.path.basename(raw_path)}  shape={raw_img.shape}")

    npz_path = find_deform_npz(slice_idx)
    if npz_path:
        logger.info(f"  Deform: {os.path.basename(npz_path)}")
    else:
        logger.warning(f"  No deformation .npz for slice_idx={slice_idx}")

    reg_img = None
    if not args.skip_registered and reg_vol is not None:
        try:
            reg_img = load_registered_slice(slice_idx, args.channel_idx, reg_vol)
            reg_u8  = log_norm_uint8(reg_img)
        except Exception as e:
            logger.warning(f"  Could not load registered slice: {e}")

    do_reg = reg_img is not None and npz_path is not None
    n_cols = 2 if do_reg else 1
    half   = args.crop_half

    fig, axes = plt.subplots(len(anns_at_z), n_cols,
                             figsize=(5 * n_cols, 5 * len(anns_at_z)),
                             squeeze=False)
    fig.suptitle(f"{TARGET_CORE}  |  {CHANNEL_NAME}  |  z_json={z_json}  slice_idx={slice_idx}",
                 fontsize=12, fontweight='bold')

    for row_i, ann in enumerate(anns_at_z):
        x_raw  = ann['points'][0]['x']
        y_raw  = ann['points'][0]['y']
        mc     = ann['mclass']
        ann_id = ann['id']
        colour = mclass_colour[mc]

        # RAW panel
        ax_raw     = axes[row_i, 0]
        cx_r, cy_r = int(round(x_raw)), int(round(y_raw))
        if 0 <= cy_r < raw_u8.shape[0] and 0 <= cx_r < raw_u8.shape[1]:
            crop, dot_x, dot_y = make_crop(raw_u8, cx_r, cy_r, half)
            ax_raw.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax_raw.plot(dot_x, dot_y, 'o', color=colour,
                        markersize=12, markeredgecolor='white', markeredgewidth=1.5)
            ax_raw.set_title(f"RAW  id={ann_id}  mclass={mc}\nx={x_raw:.1f}  y={y_raw:.1f}",
                             fontsize=9)
        else:
            ax_raw.text(0.5, 0.5,
                        f"Out of bounds\nx={cx_r}, y={cy_r}\nshape={raw_u8.shape}",
                        ha='center', va='center', transform=ax_raw.transAxes, color='red')
            ax_raw.set_title(f"RAW id={ann_id} mclass={mc} — OUT OF BOUNDS", fontsize=9)
        ax_raw.axis('off')

        # REGISTERED panel
        if do_reg:
            ax_reg = axes[row_i, 1]
            try:
                x_reg, y_reg = apply_deformation_to_point(x_raw, y_raw, npz_path)
                cx_w, cy_w   = int(round(x_reg)), int(round(y_reg))
                if 0 <= cy_w < reg_u8.shape[0] and 0 <= cx_w < reg_u8.shape[1]:
                    crop_r, dot_xr, dot_yr = make_crop(reg_u8, cx_w, cy_w, half)
                    ax_reg.imshow(crop_r, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    ax_reg.plot(dot_xr, dot_yr, 'o', color=colour,
                                markersize=12, markeredgecolor='white', markeredgewidth=1.5)
                    ax_reg.set_title(
                        f"REGISTERED  id={ann_id}  mclass={mc}\nx={x_reg:.1f}  y={y_reg:.1f}",
                        fontsize=9)
                else:
                    ax_reg.text(0.5, 0.5,
                                f"Warped point out of bounds\nx={cx_w}, y={cy_w}",
                                ha='center', va='center', transform=ax_reg.transAxes, color='red')
                    ax_reg.set_title(f"REGISTERED id={ann_id} — OUT OF BOUNDS", fontsize=9)
            except Exception as e:
                ax_reg.text(0.5, 0.5, f"Deform error:\n{e}",
                            ha='center', va='center', transform=ax_reg.transAxes, color='red')
                ax_reg.set_title(f"REGISTERED id={ann_id} — ERROR", fontsize=9)
            ax_reg.axis('off')

    legend_patches = [mpatches.Patch(color=mclass_colour[mc], label=f"mclass {mc}")
                      for mc in all_mclasses]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=len(all_mclasses), fontsize=9, title='mclass')
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = os.path.join(VERIFY_OUTPUT,
                            f"{TARGET_CORE}_z{z_json:+03d}_slice{slice_idx:02d}_annotation_check.png")
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")

# ─── OVERVIEW THUMBNAILS ──────────────────────────────────────────────────────
logger.info("Generating overview thumbnails …")
THUMB_SCALE = 0.07

for z_json, anns_at_z in sorted(by_z.items()):
    slice_idx         = z_json_to_slice_idx(z_json)
    raw_img, raw_path = load_raw_slice(slice_idx, args.channel_idx)
    if raw_img is None:
        continue
    raw_u8 = log_norm_uint8(raw_img)
    H, W   = raw_u8.shape
    th, tw = max(1, int(H * THUMB_SCALE)), max(1, int(W * THUMB_SCALE))
    thumb  = cv2.resize(raw_u8, (tw, th))

    fig, ax = plt.subplots(figsize=(8, 8 * th / tw))
    ax.imshow(thumb, cmap='gray', vmin=0, vmax=255)
    for ann in anns_at_z:
        tx = ann['points'][0]['x'] * THUMB_SCALE
        ty = ann['points'][0]['y'] * THUMB_SCALE
        mc = ann['mclass']
        ax.plot(tx, ty, 'o', color=mclass_colour[mc],
                markersize=8, markeredgecolor='white', markeredgewidth=1)
        ax.text(tx + 3, ty - 3, str(ann['id']),
                color='white', fontsize=6, clip_on=True)

    legend_patches = [mpatches.Patch(color=mclass_colour[mc], label=f"mclass {mc}")
                      for mc in all_mclasses]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7)
    ax.set_title(f"{TARGET_CORE}  |  {CHANNEL_NAME}  |  z_json={z_json}  slice_idx={slice_idx}"
                 f"  —  {len(anns_at_z)} annotation(s)  [RAW overview]", fontsize=10)
    ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(VERIFY_OUTPUT,
                            f"{TARGET_CORE}_z{z_json:+03d}_slice{slice_idx:02d}_overview.png")
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Overview: {out_path}")

logger.info(f"Done. All outputs in: {VERIFY_OUTPUT}")