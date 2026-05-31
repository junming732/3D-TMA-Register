"""
visualize_raw_landmarks.py
──────────────────────────
Visualise annotation landmarks on the **original, unregistered** input images
from TMA_Cores_Grouped_Rotate_Conformed.

Two outputs per mclass
───────────────────────
1. landmark_raw_overview_mclassN.png
   All annotated slices tiled side-by-side.  Each tile shows the DAPI crop
   around the raw (x, y) point — no warping, no registration.  Use this to
   see whether a nucleus *already looks stretched* in the raw input before
   any registration is applied.

2. landmark_raw_adjacent_mclassN.png   (≥2 slices only)
   For each consecutive pair of slices, a two-panel figure:
     Left  — DAPI crops side-by-side (same physical location, two slices)
     Right — Red/Green overlay of the two crops at the landmark position
   This lets you compare the same nucleus across adjacent z-levels directly
   in the raw (pre-registration) image space.

Usage
─────
    python visualize_raw_landmarks.py \\
        --core_name  core_09 \\
        --annotation_json  /path/to/rough_annotation_core_09.json \\
        [--dapi_crop_half  80]          # half-width of crop window (pixels)
        [--channel         0]           # channel index for "DAPI" (default 0)
        [--mclass          all]         # 'all' or comma-separated ints
        [--output_dir      .]           # where to save PNGs (default: cwd)
        [--dpi             120]
"""

import os, re, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── optional config.py (same parent-dir pattern as the other scripts) ─────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
try:
    import config
    _DATASPACE = config.DATASPACE
except ImportError:
    _DATASPACE = None   # will fall back to --data_base_path

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Visualise raw (unregistered) landmarks on DAPI channel")
parser.add_argument('--core_name',       required=True,
                    help="e.g. core_09")
parser.add_argument('--annotation_json', required=True,
                    help="Path to rough_annotation_core_09.json")
parser.add_argument('--data_base_path',  default=None,
                    help="Override: root that contains TMA_Cores_Grouped_Rotate_Conformed/<core_name>/")
parser.add_argument('--dapi_crop_half',  type=int,   default=50,
                    help="Half-width of crop window in pixels (default 50)")
parser.add_argument('--channel',         type=int,   default=0,
                    help="Channel index to display as 'DAPI' (default 0)")
parser.add_argument('--mclass',          default='all',
                    help="'all' or comma-separated ints")
parser.add_argument('--output_dir',      default=None,
                    help="Directory to write PNG files "
                         "(default: <DATASPACE>/Raw_Landmark_Visualization/<core_name>/)")
parser.add_argument('--dpi',             type=int,   default=150,
                    help="Output DPI (default 150)")
args = parser.parse_args()

TARGET_CORE   = args.core_name
CROP_HALF     = args.dapi_crop_half
CHANNEL_IDX   = args.channel

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS ────────────────────────────────────────────────────────────────────
if args.data_base_path:
    DATA_BASE_PATH = args.data_base_path
elif _DATASPACE:
    DATA_BASE_PATH = os.path.join(_DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
else:
    logger.error("No data path: set --data_base_path or ensure config.DATASPACE is importable.")
    sys.exit(1)

# ─── OUTPUT DIR ───────────────────────────────────────────────────────────────
if args.output_dir:
    OUTPUT_DIR = args.output_dir
elif _DATASPACE:
    OUTPUT_DIR = os.path.join(_DATASPACE, "Raw_Landmark_Visualization", TARGET_CORE)
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(DATA_BASE_PATH),
                              "Raw_Landmark_Visualization", TARGET_CORE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FOLDER = os.path.join(DATA_BASE_PATH, TARGET_CORE)
logger.info(f"Input folder : {INPUT_FOLDER}")
logger.info(f"Output dir   : {OUTPUT_DIR}")

# ─── Z MAPPING (same as all other scripts) ────────────────────────────────────
def z_json_to_slice_idx(z_json):
    return z_json + 10

# ─── FILE LIST ────────────────────────────────────────────────────────────────
def get_slice_number(filename):
    m = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(m.group(1)) if m else 0

FILE_LIST = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")),
                   key=get_slice_number)
if not FILE_LIST:
    logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
    sys.exit(1)
logger.info(f"Found {len(FILE_LIST)} raw slices")

# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data['annotations']
logger.info(f"Loaded {len(annotations)} annotations")

if args.mclass.lower() != 'all':
    keep        = set(int(m) for m in args.mclass.split(','))
    annotations = [a for a in annotations if a['mclass'] in keep]
    logger.info(f"After mclass filter: {len(annotations)} annotations")

# ─── TIFF LOADING HELPER ──────────────────────────────────────────────────────
try:
    import tifffile
except ImportError:
    logger.error("tifffile is required: pip install tifffile")
    sys.exit(1)


def load_raw_channel(slice_idx, channel_idx):
    """
    Load a single channel from the *raw* unregistered .ome.tif at slice_idx.
    slice_idx is a 0-based position into FILE_LIST (= z_json + 10).
    Returns a float32 array normalised to [0, 1], or None on failure.
    """
    if slice_idx < 0 or slice_idx >= len(FILE_LIST):
        logger.warning(f"slice_idx={slice_idx} out of range ({len(FILE_LIST)} files)")
        return None, None
    path = FILE_LIST[slice_idx]
    try:
        img = tifffile.imread(path)
        # Possible shapes: (C, H, W), (H, W), (Z, C, H, W) — handle gracefully
        if img.ndim == 2:
            ch = img
        elif img.ndim == 3:
            ch = img[channel_idx] if channel_idx < img.shape[0] else img[0]
        elif img.ndim == 4:
            # (Z, C, H, W) — take first Z
            ch = img[0, channel_idx] if channel_idx < img.shape[1] else img[0, 0]
        else:
            ch = img.reshape(-1, img.shape[-2], img.shape[-1])[channel_idx]
        ch = ch.astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        return ch, path
    except Exception as e:
        logger.warning(f"Could not load slice_idx={slice_idx} ({path}): {e}")
        return None, path


def crop_around(img, cx, cy, half):
    H, W = img.shape
    x0, x1 = max(0, int(cx) - half), min(W, int(cx) + half)
    y0, y1 = max(0, int(cy) - half), min(H, int(cy) + half)
    return img[y0:y1, x0:x1], x0, y0


def make_rg_overlay(img_a, img_b):
    """Red/green overlay: green = slice A (lower z), red = slice B (upper z)."""
    def safe(x):
        return x if x is not None else np.zeros((1, 1), np.float32)
    a, b = safe(img_a), safe(img_b)
    if a.shape != b.shape:
        from skimage.transform import resize as _r
        b = _r(b, a.shape, anti_aliasing=True).astype(np.float32)
    return np.stack([np.clip(b, 0, 1),
                     np.clip(a, 0, 1),
                     np.zeros_like(a)], axis=2)


# ─── BUILD PER-MCLASS RECORDS ─────────────────────────────────────────────────
from collections import defaultdict
mc_records = defaultdict(list)

for ann in annotations:
    z_json    = ann['z']
    slice_idx = z_json_to_slice_idx(z_json)
    x_raw     = ann['points'][0]['x']
    y_raw     = ann['points'][0]['y']
    mc        = ann['mclass']
    mc_records[mc].append({
        'z_json':    z_json,
        'slice_idx': slice_idx,
        'x':         x_raw,
        'y':         y_raw,
        'id':        ann['id'],
    })

# Sort each mclass by z
for mc in mc_records:
    mc_records[mc].sort(key=lambda r: r['z_json'])

logger.info(f"mclasses found: {sorted(mc_records.keys())}")

# ─── OUTPUT 1: TILED OVERVIEW — one tile per annotated slice ──────────────────
for mc, records in mc_records.items():
    n = len(records)
    logger.info(f"mclass {mc}: {n} annotation(s)")

    tile_px   = CROP_HALF * 2          
    tile_in   = max(3.0, tile_px / 80) 
    fig_w     = tile_in * n + 0.5
    
    # Reduced vertical buffer from 1.2 to 0.8 to remove the blank canvas
    fig_h     = tile_in + 0.8          
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    
    # Removed the second line containing the directory string
    fig.suptitle(
        f"{TARGET_CORE}  —  RAW landmarks  |  mclass {mc}  |  ch {CHANNEL_IDX}",
        fontsize=11, fontweight='bold'
    )

    for col, rec in enumerate(records):
        ax = axes[0][col]
        img, fpath = load_raw_channel(rec['slice_idx'], CHANNEL_IDX)

        if img is None:
            ax.set_title(f"z={rec['z_json']}  slice={rec['slice_idx']}\n(load failed)")
            ax.axis('off')
            continue

        crop, x0, y0 = crop_around(img, rec['x'], rec['y'], CROP_HALF)

        ax.imshow(crop, cmap='gray', origin='upper',
                  extent=[x0, x0 + crop.shape[1],
                          y0 + crop.shape[0], y0],
                  vmin=0, vmax=1)

        # Mark the raw landmark
        ax.scatter(rec['x'], rec['y'],
                   c='#ff4444', s=60, marker='+', linewidths=1.5, zorder=5,
                   label='raw landmark')
        ax.set_xlim(x0, x0 + crop.shape[1])
        ax.set_ylim(y0 + crop.shape[0], y0)

        ax.set_title(
            f"slice {rec['slice_idx'] + 1}",
            fontsize=9
        )
        ax.set_xlabel("x (px)", fontsize=8)
        if col == 0:
            ax.set_ylabel("y (px)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Add a shared legend
    handles = [mpatches.Patch(color='#ff4444', label='raw landmark')]
    fig.legend(handles=handles, loc='lower center', fontsize=9, ncol=1,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR,
                            f"{TARGET_CORE}_landmark_raw_overview_mclass{mc}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Overview → {out_path}")


# ─── OUTPUT 2: ADJACENT-PAIR COMPARISON ───────────────────────────────────────
# For each consecutive pair within a mclass: side-by-side crops + R/G overlay
for mc, records in mc_records.items():
    if len(records) < 2:
        continue

    pairs   = [(i, i + 1) for i in range(len(records) - 1)]
    n_pairs = len(pairs)

    # 3 rows × n_pairs columns:
    #   Row 0 — slice A crop alone
    #   Row 1 — slice B crop alone
    #   Row 2 — red/green overlay
    tile_px   = CROP_HALF * 2
    tile_in   = max(3.0, tile_px / 80)
    fig_w     = tile_in * n_pairs + 0.5
    fig_h     = tile_in * 3 + 1.5     # 3 rows + title
    fig, axes = plt.subplots(3, n_pairs,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.suptitle(
        f"{TARGET_CORE}  —  RAW adjacent-pair comparison  |  mclass {mc}  |  ch {CHANNEL_IDX}",
        fontsize=11, fontweight='bold'
    )

    for col, (ia, ib) in enumerate(pairs):
        rec_a, rec_b = records[ia], records[ib]
        img_a, _ = load_raw_channel(rec_a['slice_idx'], CHANNEL_IDX)
        img_b, _ = load_raw_channel(rec_b['slice_idx'], CHANNEL_IDX)

        # Use the mean landmark position as crop centre
        mid_x = (rec_a['x'] + rec_b['x']) / 2
        mid_y = (rec_a['y'] + rec_b['y']) / 2

        def _crop_at(img, cx, cy):
            if img is None:
                return None, 0, 0
            return crop_around(img, cx, cy, CROP_HALF)

        crop_a, x0_a, y0_a = _crop_at(img_a, mid_x, mid_y)
        crop_b, x0_b, y0_b = _crop_at(img_b, mid_x, mid_y)

        # Row 0: slice A
        ax = axes[0][col]
        if crop_a is not None:
            ax.imshow(crop_a, cmap='gray', origin='upper',
                      extent=[x0_a, x0_a + crop_a.shape[1],
                               y0_a + crop_a.shape[0], y0_a],
                      vmin=0, vmax=1)
            ax.scatter(rec_a['x'], rec_a['y'],
                       c='#00ff00', s=60, marker='+', linewidths=1.5, zorder=5)
            ax.set_xlim(x0_a, x0_a + crop_a.shape[1])
            ax.set_ylim(y0_a + crop_a.shape[0], y0_a)
        ax.set_title(f"Slice A (slice {rec_a['slice_idx'] + 1})", fontsize=9)
        ax.tick_params(labelsize=7)

        # Row 1: slice B
        ax = axes[1][col]
        if crop_b is not None:
            ax.imshow(crop_b, cmap='gray', origin='upper',
                      extent=[x0_b, x0_b + crop_b.shape[1],
                               y0_b + crop_b.shape[0], y0_b],
                      vmin=0, vmax=1)
            ax.scatter(rec_b['x'], rec_b['y'],
                       c='#ff4444', s=60, marker='+', linewidths=1.5, zorder=5)
            ax.set_xlim(x0_b, x0_b + crop_b.shape[1])
            ax.set_ylim(y0_b + crop_b.shape[0], y0_b)
        ax.set_title(f"Slice B (slice {rec_b['slice_idx'] + 1})", fontsize=9)
        ax.tick_params(labelsize=7)

        # Row 2: R/G overlay
        ax = axes[2][col]
        overlay = make_rg_overlay(crop_a, crop_b)
        x0_ov = min(x0_a, x0_b)
        y0_ov = min(y0_a, y0_b)
        ax.imshow(overlay, origin='upper',
                  extent=[x0_ov, x0_ov + overlay.shape[1],
                           y0_ov + overlay.shape[0], y0_ov])
        # Mark both landmarks (Updated to use 1-based slice index)
        ax.scatter(rec_a['x'], rec_a['y'], c='#00ff00', s=60, marker='+',
                   linewidths=1.5, zorder=5, label=f"A (slice {rec_a['slice_idx'] + 1})")
        ax.scatter(rec_b['x'], rec_b['y'], c='#ff4444', s=60, marker='+',
                   linewidths=1.5, zorder=5, label=f"B (slice {rec_b['slice_idx'] + 1})")
        # Arrow from A to B
        dx = rec_b['x'] - rec_a['x']
        dy = rec_b['y'] - rec_a['y']
        dist_px = np.hypot(dx, dy)
        if dist_px > 0:
            ax.annotate('', xy=(rec_b['x'], rec_b['y']),
                        xytext=(rec_a['x'], rec_a['y']),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
        ax.text((rec_a['x'] + rec_b['x']) / 2,
                (rec_a['y'] + rec_b['y']) / 2 - 3,
                f"Δ={dist_px:.1f} px",
                ha='center', va='bottom', fontsize=8, color='yellow',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))
        ax.set_xlim(x0_ov, x0_ov + overlay.shape[1])
        ax.set_ylim(y0_ov + overlay.shape[0], y0_ov)
        ax.set_title(f"R/G overlay  A→B  Δraw={dist_px:.1f} px", fontsize=9)
        ax.legend(fontsize=7, loc='lower right',
                  facecolor='black', labelcolor='white', framealpha=0.6)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR,
                            f"{TARGET_CORE}_landmark_raw_adjacent_mclass{mc}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Adjacent comparison → {out_path}")

logger.info("Done.")