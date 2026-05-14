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

import os, re, sys, json, glob, argparse
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
WORK_OUTPUT    = os.path.join(config.DATASPACE, "Filter_AKAZE_RoMaV2_Linear_Warp_map")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
DEFORM_FOLDER  = os.path.join(OUTPUT_FOLDER, "deformation_maps")
VERIFY_OUTPUT  = os.path.join(OUTPUT_FOLDER, "annotation_verification")
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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"{TARGET_CORE} — Landmark Registration Accuracy  "
             f"(pixel size = {PIXEL_SIZE_UM} µm)",
             fontsize=13, fontweight='bold')

# ── Panel 1: warped landmark positions (x,y scatter coloured by mclass) ──────
ax = axes[0]
for mc in all_mclasses:
    grp = df_detail[df_detail['mclass'] == mc]
    ax.scatter(grp['x_warped'], grp['y_warped'],
               color=mclass_colour[mc], label=f"mclass {mc}", s=60, zorder=3)
    # centroid marker
    cx = df_summary.loc[df_summary['mclass'] == mc, 'centroid_x'].values[0]
    cy = df_summary.loc[df_summary['mclass'] == mc, 'centroid_y'].values[0]
    ax.scatter(cx, cy, color=mclass_colour[mc], marker='+', s=200,
               linewidths=2, zorder=4)
    # lines from centroid to each point
    for _, row in grp.iterrows():
        ax.plot([cx, row['x_warped']], [cy, row['y_warped']],
                color=mclass_colour[mc], alpha=0.4, lw=1)

ax.set_title("Warped landmark positions\n(+ = centroid, lines = TRE)")
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.invert_yaxis()
ax.legend(fontsize=7, ncol=2)
ax.set_aspect('equal')

# ── Panel 2: TRE per landmark coloured by mclass, ordered by z_json ──────────
ax = axes[1]
for mc in all_mclasses:
    grp = df_detail[df_detail['mclass'] == mc].sort_values('z_json')
    ax.plot(grp['z_json'], grp['TRE_um'], 'o-',
            color=mclass_colour[mc], label=f"mclass {mc}", markersize=6)

ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=1.5,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.set_title("TRE per landmark vs z-level")
ax.set_xlabel("z_json")
ax.set_ylabel("TRE (µm)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 3: boxplot of TRE per mclass ───────────────────────────────────────
ax = axes[2]
data_per_class = [df_detail.loc[df_detail['mclass'] == mc, 'TRE_um'].values
                  for mc in all_mclasses]
bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)
for patch, mc in zip(bp['boxes'], all_mclasses):
    patch.set_facecolor(mclass_colour[mc])
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(all_mclasses) + 1))
ax.set_xticklabels([f"mclass {mc}" for mc in all_mclasses], rotation=30, ha='right')
ax.set_ylabel("TRE (µm)")
ax.set_title("TRE distribution per structure (mclass)")
ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=1.5,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Annotate each box with mean value
for i, (mc, vals) in enumerate(zip(all_mclasses, data_per_class)):
    ax.text(i + 1, vals.max() + 0.5, f"{vals.mean():.1f}",
            ha='center', va='bottom', fontsize=7, color=mclass_colour[mc])

plt.tight_layout()
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

logger.info("Done.")