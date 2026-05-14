"""
valis_registration_accuracy_landmarks.py
─────────────────────────────────────────
Compute landmark-based TRE for the VALIS pipeline.

VALIS stores its transforms in a pickled Registrar object — no separate
deformation .npz files needed.  This script:
  1. Reloads the pickled registrar saved by valis_register_core2.py
  2. For each annotation, finds the matching Slide object by slice index
  3. Calls slide_obj.warp_xy() to forward-warp the raw (x, y) point
  4. Computes TRE identically to registration_accuracy_landmarks.py
     so results are directly comparable across pipelines

Coordinate notes
────────────────
- VALIS uses the THUMBNAIL resolution internally for transforms.
  warp_xy() accepts coordinates in the *original full-resolution* image
  space and returns coordinates in registered (full-resolution) space,
  handling the scale internally — so no manual scaling is needed.
- z-index mapping: slice_idx = z_json + 10  (same as all other pipelines)
- VALIS sorts/orders slides by filename; imgs_ordered=True was NOT set in
  valis_register_core2.py, so VALIS may have reordered slices by similarity.
  This script matches annotations to slides by the TMA slice number in the
  filename, not by position, to be robust to reordering.

Usage
─────
    python valis_accuracy_landmarks.py \
        --core_name Core_09 \
        --annotation_json /path/to/rough_annotation_core_09.json \
        [--pixel_size_um 0.4961] \
        [--mclass all]
"""

import os, re, sys, json, glob, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TABLEAU_COLORS
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
parser.add_argument('--pixel_size_um',   type=float, default=0.4961)
parser.add_argument('--mclass',          type=str,   default='all')
args = parser.parse_args()

TARGET_CORE   = args.core_name
PIXEL_SIZE_UM = args.pixel_size_um

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS — mirrors valis_register_core2.py exactly ─────────────────────────
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)

WORK_OUTPUT    = os.path.join(config.DATASPACE, "VALIS_Baseline_Eval")
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
VERIFY_OUTPUT  = os.path.join(OUTPUT_FOLDER, "annotation_verification")
os.makedirs(VERIFY_OUTPUT, exist_ok=True)

# VALIS saves the pickled registrar in <output_dir>/data/<name>.pickle
PICKLE_PATH = os.path.join(OUTPUT_FOLDER, "data", f"{TARGET_CORE}.pickle")

logger.info(f"Core         : {TARGET_CORE}")
logger.info(f"Input folder : {INPUT_FOLDER}")
logger.info(f"Pickle       : {PICKLE_PATH}")
logger.info(f"Output       : {VERIFY_OUTPUT}")

# ─── Z MAPPING ────────────────────────────────────────────────────────────────
def z_json_to_slice_idx(z_json):
    return z_json + 10

# ─── FILE LIST — mirrors valis_register_core2.py ─────────────────────────────
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

raw_files = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "*.tif")) +
    glob.glob(os.path.join(INPUT_FOLDER, "*.tiff")),
    key=get_slice_number
)
raw_files = [f for f in raw_files if "_thumb" not in os.path.basename(f)]

if not raw_files:
    logger.error(f"No TIFF files in {INPUT_FOLDER}")
    sys.exit(1)

# Build slice_idx → filename mapping
idx_to_file = {i: f for i, f in enumerate(raw_files)}
logger.info(f"Found {len(raw_files)} slices")

# ─── LOAD PICKLED REGISTRAR ───────────────────────────────────────────────────
if not os.path.isfile(PICKLE_PATH):
    # VALIS sometimes names the pickle differently — search for it
    candidates = glob.glob(os.path.join(OUTPUT_FOLDER, "data", "*.pickle"))
    if candidates:
        PICKLE_PATH = candidates[0]
        logger.warning(f"Using pickle found at: {PICKLE_PATH}")
    else:
        logger.error(f"No pickle file found in {os.path.join(OUTPUT_FOLDER, 'data')}")
        logger.error("Run valis_register_core2.py first to generate the registrar.")
        sys.exit(1)

logger.info("Loading pickled VALIS registrar …")
try:
    from valis import registration
    registrar = registration.load_registrar(PICKLE_PATH)
    logger.info("  Registrar loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load registrar: {e}")
    sys.exit(1)

# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data['annotations']
logger.info(f"Loaded {len(annotations)} annotations.")

if args.mclass.lower() != 'all':
    keep        = set(int(m) for m in args.mclass.split(','))
    annotations = [a for a in annotations if a['mclass'] in keep]
    logger.info(f"After mclass filter: {len(annotations)} annotations.")

# ─── WARP EVERY LANDMARK USING VALIS ─────────────────────────────────────────
# VALIS warp_xy signature:
#   slide_obj.warp_xy(xy, M=None, non_rigid=True)
#   xy : (N, 2) array of (x, y) coordinates in original image space
#   Returns (N, 2) array in registered space
#
# We get the Slide object by passing the source filename to registrar.get_slide()

records = []

for ann in annotations:
    z_json    = ann['z']
    slice_idx = z_json_to_slice_idx(z_json)
    x_raw     = ann['points'][0]['x']
    y_raw     = ann['points'][0]['y']
    mc        = ann['mclass']
    ann_id    = ann['id']

    if slice_idx not in idx_to_file:
        logger.warning(f"  id={ann_id} z_json={z_json}: slice_idx={slice_idx} "
                       f"out of range — skipping.")
        continue

    src_file = idx_to_file[slice_idx]

    try:
        slide_obj = registrar.get_slide(src_file)
    except Exception as e:
        logger.warning(f"  id={ann_id}: could not get Slide for "
                       f"{os.path.basename(src_file)}: {e} — skipping.")
        continue

    try:
        # warp_xy expects shape (N, 2): [[x, y], ...]
        xy_raw    = np.array([[x_raw, y_raw]], dtype=np.float64)
        xy_warped = slide_obj.warp_xy(xy_raw)
        wx, wy    = float(xy_warped[0, 0]), float(xy_warped[0, 1])
    except Exception as e:
        logger.warning(f"  id={ann_id} z_json={z_json}: warp_xy failed ({e}) — skipping.")
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
    logger.error("No landmarks warped. Check the registrar and file matching.")
    sys.exit(1)

# ─── COMPUTE TRE — identical to registration_accuracy_landmarks.py ────────────
summary_rows = []
detail_rows  = []

for mc, grp in df.groupby('mclass'):
    pts          = grp[['x_warped', 'y_warped']].values
    cx, cy       = pts.mean(axis=0)
    residuals_px = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    residuals_um = residuals_px * PIXEL_SIZE_UM

    if len(pts) >= 2:
        pairs     = list(combinations(range(len(pts)), 2))
        pair_dist = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        pair_um   = pair_dist * PIXEL_SIZE_UM
    else:
        pair_dist = np.array([0.0])
        pair_um   = np.array([0.0])

    for i, (_, row) in enumerate(grp.iterrows()):
        detail_rows.append({
            'id':          row['id'],
            'mclass':      mc,
            'z_json':      row['z_json'],
            'slice_idx':   row['slice_idx'],
            'x_warped':    round(row['x_warped'], 2),
            'y_warped':    round(row['y_warped'], 2),
            'centroid_x':  round(cx, 2),
            'centroid_y':  round(cy, 2),
            'TRE_px':      round(residuals_px[i], 3),
            'TRE_um':      round(residuals_um[i], 3),
        })

    summary_rows.append({
        'mclass':            mc,
        'n_landmarks':       len(pts),
        'centroid_x':        round(cx, 2),
        'centroid_y':        round(cy, 2),
        'mean_TRE_px':       round(residuals_px.mean(), 3),
        'median_TRE_px':     round(np.median(residuals_px), 3),
        'max_TRE_px':        round(residuals_px.max(), 3),
        'std_TRE_px':        round(residuals_px.std(), 3),
        'mean_TRE_um':       round(residuals_um.mean(), 3),
        'median_TRE_um':     round(np.median(residuals_um), 3),
        'max_TRE_um':        round(residuals_um.max(), 3),
        'mean_pairwise_px':  round(pair_dist.mean(), 3),
        'max_pairwise_px':   round(pair_dist.max(), 3),
        'mean_pairwise_um':  round(pair_um.mean(), 3),
        'max_pairwise_um':   round(pair_um.max(), 3),
    })

df_detail  = pd.DataFrame(detail_rows)
df_summary = pd.DataFrame(summary_rows).sort_values('mclass')

all_tre_px = df_detail['TRE_px'].values
all_tre_um = df_detail['TRE_um'].values

logger.info("─── Global TRE summary (VALIS) ───────────────────────────────")
logger.info(f"  n landmarks  : {len(df_detail)}")
logger.info(f"  mean  TRE    : {all_tre_px.mean():.2f} px = {all_tre_um.mean():.2f} µm")
logger.info(f"  median TRE   : {np.median(all_tre_px):.2f} px = {np.median(all_tre_um):.2f} µm")
logger.info(f"  max   TRE    : {all_tre_px.max():.2f} px = {all_tre_um.max():.2f} µm")
logger.info(f"  std   TRE    : {all_tre_px.std():.2f} px = {all_tre_um.std():.2f} µm")
logger.info("──────────────────────────────────────────────────────────────")

# ─── SAVE CSVs ────────────────────────────────────────────────────────────────
detail_csv  = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_landmark_accuracy_detail.csv")
summary_csv = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_landmark_accuracy_summary.csv")
df_detail.to_csv(detail_csv,   index=False)
df_summary.to_csv(summary_csv, index=False)
logger.info(f"Detail CSV  → {detail_csv}")
logger.info(f"Summary CSV → {summary_csv}")

# ─── PLOTS — same layout as registration_accuracy_landmarks.py ───────────────
colour_list   = list(TABLEAU_COLORS.values())
all_mclasses  = sorted(df_detail['mclass'].unique())
mclass_colour = {mc: colour_list[i % len(colour_list)] for i, mc in enumerate(all_mclasses)}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"{TARGET_CORE} — VALIS Landmark Registration Accuracy  "
             f"(pixel size = {PIXEL_SIZE_UM} µm)",
             fontsize=13, fontweight='bold')

# Panel 1: warped positions
ax = axes[0]
for mc in all_mclasses:
    grp = df_detail[df_detail['mclass'] == mc]
    ax.scatter(grp['x_warped'], grp['y_warped'],
               color=mclass_colour[mc], label=f"mclass {mc}", s=60, zorder=3)
    cx = df_summary.loc[df_summary['mclass'] == mc, 'centroid_x'].values[0]
    cy = df_summary.loc[df_summary['mclass'] == mc, 'centroid_y'].values[0]
    ax.scatter(cx, cy, color=mclass_colour[mc], marker='+', s=200, linewidths=2, zorder=4)
    for _, row in grp.iterrows():
        ax.plot([cx, row['x_warped']], [cy, row['y_warped']],
                color=mclass_colour[mc], alpha=0.4, lw=1)
ax.set_title("Warped landmark positions\n(+ = centroid, lines = TRE)")
ax.set_xlabel("x (px)");  ax.set_ylabel("y (px)")
ax.invert_yaxis();  ax.legend(fontsize=7, ncol=2);  ax.set_aspect('equal')

# Panel 2: TRE vs z
ax = axes[1]
for mc in all_mclasses:
    grp = df_detail[df_detail['mclass'] == mc].sort_values('z_json')
    ax.plot(grp['z_json'], grp['TRE_um'], 'o-',
            color=mclass_colour[mc], label=f"mclass {mc}", markersize=6)
ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=1.5,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.set_title("TRE per landmark vs z-level");  ax.set_xlabel("z_json");  ax.set_ylabel("TRE (µm)")
ax.legend(fontsize=7, ncol=2);  ax.grid(True, alpha=0.3)

# Panel 3: boxplot per mclass
ax = axes[2]
data_per_class = [df_detail.loc[df_detail['mclass'] == mc, 'TRE_um'].values
                  for mc in all_mclasses]
bp = ax.boxplot(data_per_class, patch_artist=True)
for patch, mc in zip(bp['boxes'], all_mclasses):
    patch.set_facecolor(mclass_colour[mc]);  patch.set_alpha(0.7)
ax.set_xticks(range(1, len(all_mclasses) + 1))
ax.set_xticklabels([f"mclass {mc}" for mc in all_mclasses], rotation=30, ha='right')
ax.set_ylabel("TRE (µm)");  ax.set_title("TRE distribution per structure (VALIS)")
ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=1.5,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='y')
for i, (mc, vals) in enumerate(zip(all_mclasses, data_per_class)):
    ax.text(i + 1, vals.max() + 0.5, f"{vals.mean():.1f}",
            ha='center', va='bottom', fontsize=7, color=mclass_colour[mc])

plt.tight_layout()
plot_path = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_landmark_accuracy_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
logger.info(f"Plot → {plot_path}")

# ─── PRINT SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {TARGET_CORE}  —  VALIS Landmark Registration Accuracy")
print("="*70)
print(df_summary.to_string(index=False))
print("="*70)
print(f"  GLOBAL  mean TRE = {all_tre_um.mean():.2f} µm  |  "
      f"median = {np.median(all_tre_um):.2f} µm  |  "
      f"max = {all_tre_um.max():.2f} µm")
print("="*70 + "\n")

logger.info("Done.")