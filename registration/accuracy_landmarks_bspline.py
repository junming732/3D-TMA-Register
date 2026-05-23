"""
registration_accuracy_landmarks.py
───────────────────────────────────
Compute registration accuracy using rough annotations as landmarks.

Each mclass groups annotations that mark the **same anatomical structure**
across z-levels.  After warping every annotation point into registered space,
all points of the same mclass should overlap.  The spread of those warped
points is a direct measure of registration accuracy.

Metrics computed
────────────────
Per mclass (per structure):
  - Pairwise Euclidean distance between all warped points (in pixels and µm)
  - Mean, median, max, std of pairwise distances
  - Centroid + per-point residual from centroid

Per-slice (relative to its neighbours):
  - Distance to the centroid of its mclass group

Global summary:
  - Mean / median / max target registration error (TRE) across all landmarks

Outputs
───────
  registration_accuracy_landmarks.csv   — one row per annotation point
  registration_accuracy_summary.csv     — one row per mclass
  registration_accuracy_plot.png        — scatter + boxplot

All written to:
  <OUTPUT_FOLDER>/annotation_verification/

Usage
─────
    python registration_accuracy_landmarks.py \
        --core_name core_09 \
        --annotation_json /path/to/rough_annotation_core_09.json \
        [--pixel_size_um 0.4961]   # physical pixel size (default from registration script)
        [--mclass all]             # 'all' or comma-separated ints
"""

import os, re, sys, json, glob, argparse, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from collections import defaultdict

# ── config.py ─────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--core_name',       type=str, required=True)
parser.add_argument('--annotation_json', type=str, required=True)
parser.add_argument('--pixel_size_um',   type=float, default=0.4961,
                    help='Pixel size in µm (default 0.4961, from registration script).')
parser.add_argument('--mclass',          type=str, default='all')
args = parser.parse_args()

TARGET_CORE    = args.core_name
PIXEL_SIZE_UM  = args.pixel_size_um

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS — mirrors registration script ──────────────────────────────────────
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Filter_AKAZE_TissueMask_BSpline")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
DEFORM_FOLDER     = os.path.join(OUTPUT_FOLDER, "deformation_maps")
VERIFY_OUTPUT     = os.path.join(OUTPUT_FOLDER, "annotation_verification_bspline")
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
os.makedirs(VERIFY_OUTPUT, exist_ok=True)

logger.info(f"Core         : {TARGET_CORE}")
logger.info(f"Deform folder: {DEFORM_FOLDER}")
logger.info(f"Output       : {VERIFY_OUTPUT}")

# ─── Z MAPPING ────────────────────────────────────────────────────────────────
def z_json_to_slice_idx(z_json):
    return z_json + 10

# ─── FILE LIST — mirrors registration script ──────────────────────────────────
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

FILE_LIST = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")), key=get_slice_number)
if not FILE_LIST:
    logger.error(f"No .ome.tif files in {INPUT_FOLDER}")
    sys.exit(1)
logger.info(f"Found {len(FILE_LIST)} slices")

# ─── FILTERED SLICE LIST + SLICE→VOL_Z MAP ────────────────────────────────────
# The registered volume only contains filtered slices, so Z indices in the
# volume correspond to positions in FILTERED_FILE_LIST, not FILE_LIST.
# Mirrors the working RoMaV2 accuracy script exactly.

def load_slice_filter(yaml_path, core_name):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get(core_name)
    if raw is None:
        return None
    allowed = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            allowed.add(int(part))
    return allowed

allowed_positions = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)

if allowed_positions is not None:
    FILTERED_FILE_LIST = [f for i, f in enumerate(FILE_LIST) if i in allowed_positions]
    logger.info(f"Slice filter active: {len(FILTERED_FILE_LIST)}/{len(FILE_LIST)} slices kept.")
else:
    FILTERED_FILE_LIST = FILE_LIST
    logger.info("No slice filter — using all slices.")

# slice_idx = z_json + 10 is a 0-based position into FILE_LIST.
file_to_orig_pos = {f: i for i, f in enumerate(FILE_LIST)}
SLICE_IDX_TO_VOL_Z = {
    file_to_orig_pos[f]: vol_z
    for vol_z, f in enumerate(FILTERED_FILE_LIST)
}
logger.info(f"Slice→vol_z map: {SLICE_IDX_TO_VOL_Z}")

# ─── DEFORMATION HELPERS ──────────────────────────────────────────────────────

def find_deform_npz(slice_idx):
    for pat in [f"{TARGET_CORE}_Z{slice_idx:03d}_*_deformation.npz",
                f"{TARGET_CORE}_Z{slice_idx}_*_deformation.npz"]:
        m = glob.glob(os.path.join(DEFORM_FOLDER, pat))
        if m:
            return m[0]
    return None


def warp_point(x, y, npz_path):
    """
    Forward-warp (x,y) from original moving-slice space → registered space.
    Mirrors apply_deformation_to_mask() two-step logic:
      Step 1: affine via M_affine
      Step 2: dense remap inversion (nearest-neighbour search)
    Returns (wx, wy) in registered space.
    """
    d        = np.load(npz_path)
    M_affine = d['M_affine'].astype(np.float64)
    map_x    = d['map_x'].astype(np.float32)
    map_y    = d['map_y'].astype(np.float32)

    # Step 1: affine
    pt     = np.array([x, y, 1.0], dtype=np.float64)
    xy_aff = M_affine @ pt
    ax, ay = float(xy_aff[0]), float(xy_aff[1])

    # Step 2: invert remap
    H, W     = map_x.shape
    search_r = 150
    x0 = max(0, int(ax) - search_r);  x1 = min(W, int(ax) + search_r)
    y0 = max(0, int(ay) - search_r);  y1 = min(H, int(ay) + search_r)
    patch_x = map_x[y0:y1, x0:x1]
    patch_y = map_y[y0:y1, x0:x1]
    dist2   = (patch_x - ax)**2 + (patch_y - ay)**2
    ry, rx  = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(x0 + rx), float(y0 + ry)

# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data['annotations']
logger.info(f"Loaded {len(annotations)} annotations.")

if args.mclass.lower() != 'all':
    keep = set(int(m) for m in args.mclass.split(','))
    annotations = [a for a in annotations if a['mclass'] in keep]
    logger.info(f"After mclass filter: {len(annotations)} annotations.")

# ─── WARP EVERY LANDMARK INTO REGISTERED SPACE ───────────────────────────────
records = []

for ann in annotations:
    z_json    = ann['z']
    slice_idx = z_json_to_slice_idx(z_json)
    x_raw     = ann['points'][0]['x']
    y_raw     = ann['points'][0]['y']
    mc        = ann['mclass']
    ann_id    = ann['id']

    npz_path  = find_deform_npz(slice_idx)

    if npz_path is None:
        logger.warning(f"  id={ann_id} z_json={z_json}: no deform .npz — skipping.")
        continue

    try:
        wx, wy = warp_point(x_raw, y_raw, npz_path)
    except Exception as e:
        logger.warning(f"  id={ann_id} z_json={z_json}: warp failed ({e}) — skipping.")
        continue

    records.append({
        'id':        ann_id,
        'mclass':    mc,
        'z_json':    z_json,
        'slice_idx': slice_idx,
        'x_raw':     x_raw,
        'y_raw':     y_raw,
        'x_warped':  wx,
        'y_warped':  wy,
    })

df = pd.DataFrame(records)
logger.info(f"Successfully warped {len(df)} / {len(annotations)} landmarks.")

if df.empty:
    logger.error("No landmarks could be warped. Check deformation .npz files.")
    sys.exit(1)

# ─── COMPUTE ACCURACY METRICS ─────────────────────────────────────────────────
# For each mclass, compute:
#   centroid of warped points → TRE = distance from each point to centroid
#   pairwise distances between all warped points within the class

summary_rows = []
detail_rows  = []

for mc, grp in df.groupby('mclass'):
    pts = grp[['x_warped', 'y_warped']].values   # (N, 2) in pixels

    # Centroid
    cx, cy = pts.mean(axis=0)

    # Per-point residual (TRE) from centroid
    residuals_px = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    residuals_um = residuals_px * PIXEL_SIZE_UM

    # Pairwise distances
    if len(pts) >= 2:
        pairs     = list(combinations(range(len(pts)), 2))
        pair_dist = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        pair_um   = pair_dist * PIXEL_SIZE_UM
    else:
        pair_dist = np.array([0.0])
        pair_um   = np.array([0.0])

    for i, (_, row) in enumerate(grp.iterrows()):
        detail_rows.append({
            'id':               row['id'],
            'mclass':           mc,
            'z_json':           row['z_json'],
            'slice_idx':        row['slice_idx'],
            'x_warped':         round(row['x_warped'], 2),
            'y_warped':         round(row['y_warped'], 2),
            'centroid_x':       round(cx, 2),
            'centroid_y':       round(cy, 2),
            'TRE_px':           round(residuals_px[i], 3),
            'TRE_um':           round(residuals_um[i], 3),
        })

    summary_rows.append({
        'mclass':               mc,
        'n_landmarks':          len(pts),
        'centroid_x':           round(cx, 2),
        'centroid_y':           round(cy, 2),
        'mean_TRE_px':          round(residuals_px.mean(), 3),
        'median_TRE_px':        round(np.median(residuals_px), 3),
        'max_TRE_px':           round(residuals_px.max(), 3),
        'std_TRE_px':           round(residuals_px.std(), 3),
        'mean_TRE_um':          round(residuals_um.mean(), 3),
        'median_TRE_um':        round(np.median(residuals_um), 3),
        'max_TRE_um':           round(residuals_um.max(), 3),
        'mean_pairwise_px':     round(pair_dist.mean(), 3),
        'max_pairwise_px':      round(pair_dist.max(), 3),
        'mean_pairwise_um':     round(pair_um.mean(), 3),
        'max_pairwise_um':      round(pair_um.max(), 3),
    })

df_detail  = pd.DataFrame(detail_rows)
df_summary = pd.DataFrame(summary_rows).sort_values('mclass')

# Global stats across all landmarks
all_tre_px = df_detail['TRE_px'].values
all_tre_um = df_detail['TRE_um'].values
logger.info("─── Global TRE summary ───────────────────────────────────────")
logger.info(f"  n landmarks    : {len(df_detail)}")
logger.info(f"  mean  TRE      : {all_tre_px.mean():.2f} px  = {all_tre_um.mean():.2f} µm")
logger.info(f"  median TRE     : {np.median(all_tre_px):.2f} px  = {np.median(all_tre_um):.2f} µm")
logger.info(f"  max   TRE      : {all_tre_px.max():.2f} px  = {all_tre_um.max():.2f} µm")
logger.info(f"  std   TRE      : {all_tre_px.std():.2f} px  = {all_tre_um.std():.2f} µm")
logger.info("──────────────────────────────────────────────────────────────")

# ─── SAVE CSVs ────────────────────────────────────────────────────────────────
detail_csv  = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_detail.csv")
summary_csv = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_summary.csv")
df_detail.to_csv(detail_csv,   index=False)
df_summary.to_csv(summary_csv, index=False)
logger.info(f"Detail CSV  → {detail_csv}")
logger.info(f"Summary CSV → {summary_csv}")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
from matplotlib.colors import TABLEAU_COLORS
colour_list   = list(TABLEAU_COLORS.values())
all_mclasses  = sorted(df_detail['mclass'].unique())
mclass_colour = {mc: colour_list[i % len(colour_list)] for i, mc in enumerate(all_mclasses)}

# 1 row, 2 columns. Increased figsize to accommodate larger fonts.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"Pipeline A: {TARGET_CORE} — Landmark Registration Accuracy  "
             f"(pixel size = {PIXEL_SIZE_UM} µm)",
             fontsize=18, fontweight='bold')

# ── Panel 1: TRE per landmark vs slice ──────────
ax = axes[0]
for mc in all_mclasses:
    grp = df_detail[df_detail['mclass'] == mc].sort_values('slice_idx')
    ax.plot(grp['slice_idx'], grp['TRE_um'], 'o-',
            color=mclass_colour[mc], label=f"mclass {mc}", markersize=8)

ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=2,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.set_title("TRE per landmark vs slice", fontsize=15)
ax.set_xlabel("slice index", fontsize=13)
ax.set_ylabel("TRE (µm)", fontsize=13)
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 2: boxplot of TRE per mclass ───────────────────────────────────────
ax = axes[1]
data_per_class = [df_detail.loc[df_detail['mclass'] == mc, 'TRE_um'].values
                  for mc in all_mclasses]
bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)
for patch, mc in zip(bp['boxes'], all_mclasses):
    patch.set_facecolor(mclass_colour[mc])
    patch.set_alpha(0.7)

ax.set_xticks(range(1, len(all_mclasses) + 1))
ax.set_xticklabels([f"mclass {mc}" for mc in all_mclasses], rotation=30, ha='right', fontsize=12)
ax.set_ylabel("TRE (µm)", fontsize=13)
ax.set_title("TRE distribution per structure (mclass)", fontsize=15)
ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=2,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.tick_params(axis='y', labelsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Annotate each box with mean value
for i, (mc, vals) in enumerate(zip(all_mclasses, data_per_class)):
    ax.text(i + 1, vals.max() + 0.5, f"{vals.mean():.1f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=mclass_colour[mc])

plt.tight_layout()
# Provide a slight buffer at the top so the main title doesn't overlap the subplots
plt.subplots_adjust(top=0.88)

plot_path = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
logger.info(f"Plot → {plot_path}")

# ─── PRINT SUMMARY TABLE ──────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {TARGET_CORE}  —  Landmark Registration Accuracy")
print("="*70)
print(df_summary.to_string(index=False))
print("="*70)
print(f"  GLOBAL  mean TRE = {all_tre_um.mean():.2f} µm  |  "
      f"median = {np.median(all_tre_um):.2f} µm  |  "
      f"max = {all_tre_um.max():.2f} µm")
print("="*70 + "\n")

# ─── ADJACENT-SLICE OVERLAYS ──────────────────────────────────────────────────
# Layout: 2 rows × N columns per figure.
#   Row 0 — DAPI overlay  (channel 0)
#   Row 1 — CK overlay    (channel CK_CHANNEL_IDX)
# Both rows use the same cyan/magenta blend and identical annotations so the
# error vector is visible against both stains.

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6


def load_slice_channel_from_vol(vol, slice_idx, channel_idx):
    """
    Load and contrast-stretch a single channel from the registered volume.

    Uses SLICE_IDX_TO_VOL_Z to map TMA slice_idx to the correct Z position
    in the registered volume (Z, C, H, W).  Loading from the registered
    volume ensures image content is aligned with the warped landmark coords.
    Previously this read from the raw unregistered FILE_LIST which caused
    the image to be misaligned with the warped landmark coordinates.
    """
    vol_z = SLICE_IDX_TO_VOL_Z.get(slice_idx)
    if vol_z is None:
        logger.warning(f"slice_idx={slice_idx} not found in filtered slice list — skipping.")
        return None
    try:
        ch = vol[vol_z, channel_idx].astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        return ch
    except Exception as e:
        logger.warning(f"Could not load vol_z={vol_z} ch={channel_idx}: {e}")
        return None


def make_two_channel_rgb(img_a, img_b):
    """
    Blend two greyscale images into a cyan (A = lower z) / magenta (B = upper z) overlay.
    White/grey pixels indicate agreement between the two slices.
    """
    h = max(img_a.shape[0] if img_a is not None else 0,
            img_b.shape[0] if img_b is not None else 0)
    w = max(img_a.shape[1] if img_a is not None else 0,
            img_b.shape[1] if img_b is not None else 0)
    a = img_a if img_a is not None else np.zeros((h, w), np.float32)
    b = img_b if img_b is not None else np.zeros((h, w), np.float32)
    if a.shape != b.shape:
        from skimage.transform import resize as sk_resize
        b = sk_resize(b, a.shape, anti_aliasing=True).astype(np.float32)
    r  = np.clip(b, 0, 1)   # red   — upper slice
    g  = np.clip(a, 0, 1)   # green — lower slice
    bl = np.zeros_like(r)   # no blue: overlap → yellow, single → red or green
    return np.stack([r, g, bl], axis=2)


def _annotate_overlay_ax(ax, rgb, wx_a, wy_a, wx_b, wy_b,
                         x0, x1, y0, y1, z_a, z_b,
                         dist_px, dist_um, row_label):
    """Shared helper: imshow + markers + arrow + label for one overlay panel."""
    mid_x = (wx_a + wx_b) / 2
    mid_y = (wy_a + wy_b) / 2

    ax.imshow(rgb, origin='upper', extent=[x0, x1, y1, y0])

    ax.scatter(wx_a, wy_a, c='#00ff00', s=80, zorder=5,
               edgecolors='white', linewidths=0.8, label=f"slice {z_a}")
    ax.scatter(wx_b, wy_b, c='#ff0000', s=80, zorder=5,
               edgecolors='white', linewidths=0.8, label=f"slice {z_b}")

    ax.annotate('', xy=(wx_b, wy_b), xytext=(wx_a, wy_a),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.8))

    ax.text(mid_x, mid_y - 5,
            f"{dist_px:.1f} px / {dist_um:.1f} µm",
            ha='center', va='bottom', fontsize=8,
            color='yellow', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(f"{row_label}  |  slice {z_a} → slice {z_b}  |  Δ = {dist_um:.1f} µm",
                 fontsize=9)
    ax.set_xlabel("x (px)", fontsize=8)
    ax.set_ylabel("y (px)", fontsize=8)
    ax.legend(fontsize=7, loc='lower right',
              facecolor='black', labelcolor='white', framealpha=0.6)


def plot_adjacent_slice_overlays(df_detail,
                                 dapi_crop_half=50,
                                 ck_crop_half=150,
                                 dpi=120):
    """
    For each mclass with ≥2 z-levels, produce one PNG with:
      Row 0 — DAPI (ch 0) overlay — tighter crop (dapi_crop_half=50)
      Row 1 — CK   (ch 6) overlay — wider crop  (ck_crop_half=150)

    Loads from the registered output volume so image content is aligned
    with the warped landmark coordinates.  Uses SLICE_IDX_TO_VOL_Z for
    correct Z-axis indexing into the volume.
    """
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed — skipping adjacent-slice overlays.")
        return

    # Load registered volume once — avoids re-reading the TIFF for every pair
    reg_path = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}_AKAZE_TissueMask_Aligned.ome.tif")
    if not os.path.isfile(reg_path):
        logger.warning(f"Registered volume not found at {reg_path} — skipping overlays.")
        return

    logger.info(f"Loading registered volume from {reg_path} ...")
    reg_vol = tifffile.imread(reg_path)   # expected shape: (Z, C, H, W)
    logger.info(f"Registered volume shape: {reg_vol.shape}")

    for mc, grp in df_detail.groupby('mclass'):
        grp_sorted = grp.sort_values('z_json').reset_index(drop=True)
        if len(grp_sorted) < 2:
            continue

        pairs   = [(i, i + 1) for i in range(len(grp_sorted) - 1)]
        n_pairs = len(pairs)

        # 2 rows (DAPI / CK) × n_pairs columns
        fig, axes = plt.subplots(2, n_pairs,
                                 figsize=(5 * n_pairs, 10),
                                 squeeze=False)

        fig.suptitle(
            f"Pipeline A: {TARGET_CORE}  —  Adjacent-slice overlay  |  mclass {mc}\n"
            f"Green = lower slice, Red = upper slice  |  pixel size = {PIXEL_SIZE_UM} µm",
            fontsize=10, fontweight='bold'
        )

        for col, (ia, ib) in enumerate(pairs):
            row_a  = grp_sorted.iloc[ia]
            row_b  = grp_sorted.iloc[ib]
            sidx_a, sidx_b = int(row_a['slice_idx']), int(row_b['slice_idx'])
            z_a,    z_b    = sidx_a, sidx_b
            wx_a,  wy_a   = float(row_a['x_warped']), float(row_a['y_warped'])
            wx_b,  wy_b   = float(row_b['x_warped']), float(row_b['y_warped'])

            dist_px = np.hypot(wx_b - wx_a, wy_b - wy_a)
            dist_um = dist_px * PIXEL_SIZE_UM
            mid_x   = (wx_a + wx_b) / 2
            mid_y   = (wy_a + wy_b) / 2

            for row_idx, (ch_idx, ch_label, ch_crop) in enumerate([
                    (DAPI_CHANNEL_IDX, "DAPI", dapi_crop_half),
                    (CK_CHANNEL_IDX,   "CK",   ck_crop_half)]):

                ax    = axes[row_idx][col]
                img_a = load_slice_channel_from_vol(reg_vol, sidx_a, ch_idx)
                img_b = load_slice_channel_from_vol(reg_vol, sidx_b, ch_idx)

                if img_a is None and img_b is None:
                    ax.set_title(f"{ch_label}  z {z_a}→{z_b}\n(unavailable)")
                    ax.axis('off')
                    continue

                ref  = img_a if img_a is not None else img_b
                H, W = ref.shape
                x0 = max(0, int(mid_x) - ch_crop)
                x1 = min(W, int(mid_x) + ch_crop)
                y0 = max(0, int(mid_y) - ch_crop)
                y1 = min(H, int(mid_y) + ch_crop)

                def crop(img, _y0=y0, _y1=y1, _x0=x0, _x1=x1):
                    return img[_y0:_y1, _x0:_x1] if img is not None else None

                rgb = make_two_channel_rgb(crop(img_a), crop(img_b))
                _annotate_overlay_ax(ax, rgb, wx_a, wy_a, wx_b, wy_b,
                                     x0, x1, y0, y1, z_a, z_b,
                                     dist_px, dist_um, ch_label)

        plt.tight_layout()
        overlay_path = os.path.join(
            VERIFY_OUTPUT, f"{TARGET_CORE}_adjacent_overlay_mclass{mc}.png"
        )
        fig.savefig(overlay_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Adjacent overlay (mclass {mc}) → {overlay_path}")


plot_adjacent_slice_overlays(df_detail)

logger.info("Done.")