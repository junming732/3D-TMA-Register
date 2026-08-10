"""
valis_accuracy_landmarks.py
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
- z-index mapping: slice_idx = z_json - 1  (same as all other pipelines,
  see z_json_to_slice_idx in landmark_accuracy_common.py)
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
        [--landmark_id all]
"""

import os, re, sys, json, glob, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TABLEAU_COLORS
from collections import defaultdict

# ── config.py ─────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config
from landmark_accuracy_common import (
    z_json_to_slice_idx, get_slice_number, make_two_channel_rgb, _annotate_overlay_ax,
)

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--core_name',       type=str, required=True)
parser.add_argument('--annotation_json', type=str, required=True)
parser.add_argument('--pixel_size_um',   type=float, default=0.4961)
parser.add_argument('--landmark_id',          type=str,   default='all')
parser.add_argument('--work_output_dir', type=str, default='VALIS_Baseline_Eval',
                    help='Folder under DATASPACE holding VALIS registration output '
                         '(default: VALIS_Baseline_Eval). Override to point at a '
                         'different experiment run.')
args = parser.parse_args()

TARGET_CORE   = args.core_name
PIXEL_SIZE_UM = args.pixel_size_um

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS — mirrors valis_register_core2.py exactly ─────────────────────────
WORK_OUTPUT   = os.path.join(config.DATASPACE, args.work_output_dir)
# VALIS creates <dst_dir>/<name>/ internally, so the actual output is one level deeper
OUTPUT_FOLDER = os.path.join(WORK_OUTPUT, TARGET_CORE, TARGET_CORE)
VERIFY_OUTPUT = os.path.join(OUTPUT_FOLDER, "annotation_verification_valis")
os.makedirs(VERIFY_OUTPUT, exist_ok=True)

# VALIS saves the pickled registrar in <output_dir>/data/<name>.pickle
PICKLE_PATH = os.path.join(OUTPUT_FOLDER, "data", f"{TARGET_CORE}.pickle")
REG_SLIDES_DIR = os.path.join(OUTPUT_FOLDER, "..", "registered_slides")
REG_SLIDES_DIR = os.path.normpath(REG_SLIDES_DIR)

logger.info(f"Core         : {TARGET_CORE}")
logger.info(f"Pickle       : {PICKLE_PATH}")
logger.info(f"Output       : {VERIFY_OUTPUT}")

# ─── SLICE INDEX → SLIDE NAME MAPPING ────────────────────────────────────────
# Built after loading the registrar from registrar.slide_dict, which maps
# slide name → Slide object for every file VALIS registered.
# We sort slide names by TMA number (same sort as the raw file list) so that
# position in that sorted order == slice_idx.
# get_slice_number imported from landmark_accuracy_common (safe here even
# though this script passes bare slide names, not full paths — it applies
# os.path.basename() internally, which is a no-op on an already-bare name).

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

# Build slice_idx → Slide mapping from the registrar itself — no file scanning needed.
# registrar.slide_dict maps slide name → Slide object.
# Sort by TMA number so position == slice_idx, same as other pipelines.
idx_to_slide = {}
for name, slide_obj in registrar.slide_dict.items():
    tma_num = get_slice_number(name)
    slice_idx = tma_num - 1  # 0-based to match z_json_to_slice_idx
    idx_to_slide[slice_idx] = slide_obj

logger.info(f"Registrar contains {len(idx_to_slide)} slides:")
# Sort by the actual slice_idx keys for clean, ordered logging
for s_idx in sorted(idx_to_slide.keys()):
    logger.info(f"  slice_idx={s_idx:02d}  ->  {idx_to_slide[s_idx].src_f}")


# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data
logger.info(f"Loaded {len(annotations)} annotations.")

if args.landmark_id.lower() != 'all':
    keep        = set(int(m) for m in args.landmark_id.split(','))
    annotations = [a for a in annotations if a['landmark_id'] in keep]
    logger.info(f"After mclass filter: {len(annotations)} annotations.")

# ─── WARP EVERY LANDMARK USING VALIS ─────────────────────────────────────────
# warp_xy(xy): xy is (N,2) float64 in original full-res image space
#              returns (N,2) in registered space

records = []

for ann in annotations:
    z_json    = ann['z']
    slice_idx = z_json_to_slice_idx(z_json)
    x_raw     = ann['x']
    y_raw     = ann['y']
    mc        = ann['landmark_id']
    ann_id    = ann['landmark_id']

    if slice_idx not in idx_to_slide:
        logger.warning(f"  id={ann_id} z_json={z_json}: slice_idx={slice_idx} "
                       f"not in registrar ({len(idx_to_slide)} slides) — skipping.")
        continue

    slide_obj = idx_to_slide[slice_idx]

    try:
        xy_raw    = np.array([[x_raw, y_raw]], dtype=np.float64)
        xy_warped = slide_obj.warp_xy(xy_raw)
        wx, wy    = float(xy_warped[0, 0]), float(xy_warped[0, 1])
    except Exception as e:
        logger.warning(f"  id={ann_id} z_json={z_json}: warp_xy failed ({e}) — skipping.")
        continue

    records.append({
        'id':        ann_id,
        'landmark_id':    mc,
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

# ─── COMPUTE TRE (Fitzpatrick et al. 1998 — consecutive pairwise) ─────────────
# For each mclass, TRE is the Euclidean distance between warped landmark
# positions on consecutive slices: TRE_i = ||T(p_i) - T(p_i+1)||
# (eq. 1 from the paper).  Mean TRE for the mclass = mean over its N-1 pairs.
# This matches our chained rolling registration exactly: each adjacent pair
# was registered directly to each other, so pairwise distance is the direct
# measure of that registration step's error.
summary_rows = []
detail_rows  = []

for mc, grp in df.groupby('landmark_id'):
    grp = grp.sort_values('z_json').reset_index(drop=True)
    pts = grp[['x_warped', 'y_warped']].values   # (N, 2), sorted by z

    # Consecutive pairs only: (0,1), (1,2), ..., (N-2, N-1)
    if len(pts) >= 2:
        pair_indices = [(i, i + 1) for i in range(len(pts) - 1)]
        pair_dist_px = np.array([
            np.linalg.norm(pts[i] - pts[j]) for i, j in pair_indices
        ])
        pair_dist_um = pair_dist_px * PIXEL_SIZE_UM
    else:
        pair_indices = []
        pair_dist_px = np.array([0.0])
        pair_dist_um = np.array([0.0])

    mean_TRE_px = pair_dist_px.mean()
    mean_TRE_um = pair_dist_um.mean()
    logger.info(
        f"mclass {mc}: {len(pts)} slices, {len(pair_indices)} pair(s), "
        f"mean TRE = {mean_TRE_px:.2f} px = {mean_TRE_um:.2f} µm"
    )

    # Detail rows — one per consecutive pair
    for (ia, ib), d_px, d_um in zip(pair_indices, pair_dist_px, pair_dist_um):
        row_a, row_b = grp.iloc[ia], grp.iloc[ib]
        detail_rows.append({
            'landmark_id':        mc,
            'z_json_a':      row_a['z_json'],
            'z_json_b':      row_b['z_json'],
            'slice_idx_a':   row_a['slice_idx'],
            'slice_idx_b':   row_b['slice_idx'],
            'id_a':          row_a['id'],
            'id_b':          row_b['id'],
            'x_warped_a':    round(row_a['x_warped'], 2),
            'y_warped_a':    round(row_a['y_warped'], 2),
            'x_warped_b':    round(row_b['x_warped'], 2),
            'y_warped_b':    round(row_b['y_warped'], 2),
            'TRE_px':        round(d_px, 3),
            'TRE_um':        round(d_um, 3),
        })

    summary_rows.append({
        'landmark_id':        mc,
        'n_slices':      len(pts),
        'n_pairs':       len(pair_indices),
        'mean_TRE_px':   round(mean_TRE_px, 3),
        'median_TRE_px': round(np.median(pair_dist_px), 3),
        'q3_TRE_px':     round(np.percentile(pair_dist_px, 75), 3),
        'p90_TRE_px':    round(np.percentile(pair_dist_px, 90), 3),
        'max_TRE_px':    round(pair_dist_px.max(), 3),
        'std_TRE_px':    round(pair_dist_px.std(), 3) if len(pair_dist_px) > 1 else 0.0,
        'mean_TRE_um':   round(mean_TRE_um, 3),
        'median_TRE_um': round(np.median(pair_dist_um), 3),
        'q3_TRE_um':     round(np.percentile(pair_dist_um, 75), 3),
        'p90_TRE_um':    round(np.percentile(pair_dist_um, 90), 3),
        'max_TRE_um':    round(pair_dist_um.max(), 3),
        'std_TRE_um':    round(pair_dist_um.std(), 3) if len(pair_dist_um) > 1 else 0.0,
    })

df_detail  = pd.DataFrame(detail_rows)
df_summary = pd.DataFrame(summary_rows).sort_values('landmark_id')

# Global TRE across all consecutive pairs
all_tre_px = df_detail['TRE_px'].values
all_tre_um = df_detail['TRE_um'].values
logger.info("─── Global TRE summary ───────────────────────────────────────")
logger.info(f"  n pairs        : {len(df_detail)}")
logger.info(f"  mean  TRE      : {all_tre_px.mean():.2f} px  = {all_tre_um.mean():.2f} µm")
logger.info(f"  median TRE     : {np.median(all_tre_px):.2f} px  = {np.median(all_tre_um):.2f} µm")
logger.info(f"  Q3 (75th pct)  : {np.percentile(all_tre_px, 75):.2f} px  = {np.percentile(all_tre_um, 75):.2f} µm")
logger.info(f"  P90            : {np.percentile(all_tre_px, 90):.2f} px  = {np.percentile(all_tre_um, 90):.2f} µm")
logger.info(f"  max   TRE      : {all_tre_px.max():.2f} px  = {all_tre_um.max():.2f} µm")
logger.info(f"  std   TRE      : {all_tre_px.std():.2f} px  = {all_tre_um.std():.2f} µm")
logger.info("──────────────────────────────────────────────────────────────")

# ─── SAVE CSVs ────────────────────────────────────────────────────────────────
detail_csv  = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_landmark_accuracy_detail.csv")
summary_csv = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_landmark_accuracy_summary.csv")
df_detail.to_csv(detail_csv,   index=False)
df_summary.to_csv(summary_csv, index=False)
logger.info(f"Detail CSV  → {detail_csv}")
logger.info(f"Summary CSV → {summary_csv}")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
colour_list   = list(TABLEAU_COLORS.values())
all_mclasses  = sorted(df_detail['landmark_id'].unique())
mclass_colour = {mc: colour_list[i % len(colour_list)] for i, mc in enumerate(all_mclasses)}

# 1 row, 2 columns. Increased figsize to accommodate larger fonts.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"Pipeline B {TARGET_CORE} — Landmark Registration Accuracy  "
             f"(pixel size = {PIXEL_SIZE_UM} µm)",
             fontsize=18, fontweight='bold')

# ── Panel 1: TRE per consecutive pair vs slice pair ─────────────────────────
ax = axes[0]
for mc in all_mclasses:
    grp = df_detail[df_detail['landmark_id'] == mc].sort_values('z_json_a').reset_index(drop=True)
    # Convert from 0-based slice_idx to 1-based slice index for labels
    x_labels = [f"{int(r.slice_idx_a) + 1}→{int(r.slice_idx_b) + 1}" for _, r in grp.iterrows()]
    x_pos    = np.arange(len(grp))
    ax.plot(x_pos, grp['TRE_um'], 'o-',
            color=mclass_colour[mc], label=f"landmark {mc}", markersize=8)
    for xi, lab in zip(x_pos, x_labels):
        ax.annotate(lab, (xi, grp['TRE_um'].iloc[xi]),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=7, color=mclass_colour[mc])

ax.axhline(all_tre_um.mean(), color='black', linestyle='--', lw=2,
           label=f"global mean {all_tre_um.mean():.1f} µm")
ax.set_title("TRE per consecutive slice pair", fontsize=15)
# Removed the explicit z_json_a -> z_json_b text from the axis label
ax.set_xlabel("slice pair", fontsize=13)
ax.set_ylabel("TRE (µm)", fontsize=13)
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 2: boxplot of TRE per mclass ───────────────────────────────────────
ax = axes[1]
data_per_class = [df_detail.loc[df_detail['landmark_id'] == mc, 'TRE_um'].values
                  for mc in all_mclasses]
bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)
for patch, mc in zip(bp['boxes'], all_mclasses):
    patch.set_facecolor(mclass_colour[mc])
    patch.set_alpha(0.7)

ax.set_xticks(range(1, len(all_mclasses) + 1))
ax.set_xticklabels([f"landmark {mc}" for mc in all_mclasses], rotation=30, ha='right', fontsize=12)
ax.set_ylabel("TRE (µm)", fontsize=13)
ax.set_title("TRE distribution per structure (landmark_id)", fontsize=15)
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

plot_path = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_plot_Valis.png")
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

# ─── ADJACENT-SLICE OVERLAYS ──────────────────────────────────────────────────
# Layout: 2 rows × N columns per figure.
#   Row 0 — DAPI overlay  (channel 0)
#   Row 1 — CK overlay    (channel CK_CHANNEL_IDX)

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6


def load_slice_channel_valis(slice_idx, channel_idx):
    slide_obj = idx_to_slide.get(slice_idx)
    if slide_obj is None:
        return None

    # Find registered output file instead of raw src_f
    src_basename = os.path.splitext(os.path.basename(slide_obj.src_f))[0]
    candidates = glob.glob(os.path.join(REG_SLIDES_DIR, f"*{src_basename}*"))
    if not candidates:
        logger.warning(f"No registered file found for slice_idx={slice_idx} "
                       f"(looking for *{src_basename}* in {REG_SLIDES_DIR})")
        return None
    reg_path = candidates[0]

    try:
        import tifffile
        img = tifffile.imread(reg_path)
        if img.ndim == 2:
            ch = img
        elif img.ndim == 3:
            ch = img[channel_idx]
        else:
            ch = img[0, channel_idx]
        ch = ch.astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        return ch
    except Exception as e:
        logger.warning(f"Could not load registered slice_idx={slice_idx}: {e}")
        return None


# make_two_channel_rgb, _annotate_overlay_ax imported from landmark_accuracy_common.


def plot_adjacent_slice_overlays_valis(df_detail,
                                       dapi_crop_half=50,
                                       ck_crop_half=150,
                                       dpi=120):
    """
    For each mclass with ≥2 z-levels, produce one PNG with a horizontal grid layout:
      Row 0 — DAPI (ch 0) cyan/magenta overlay — tighter crop (dapi_crop_half=50)
      Row 1 — CK   (ch 6) cyan/magenta overlay — wider crop  (ck_crop_half=150)
    Each column corresponds to an adjacent slice pair.
    """
    try:
        import tifffile  # noqa
    except ImportError:
        logger.warning("tifffile not installed — skipping adjacent-slice overlays.")
        return

    max_cols = 3

    for mc, grp in df_detail.groupby('landmark_id'):
        grp_sorted = grp.sort_values('z_json_a').reset_index(drop=True)
        n_pairs = len(grp_sorted)
        if n_pairs < 1:
            continue

        # Split pairs into chunks to prevent overly wide figures
        chunks = [grp_sorted.iloc[i:i + max_cols] for i in range(0, n_pairs, max_cols)]

        for chunk_idx, chunk in enumerate(chunks):
            n_cols = len(chunk)
            
            # ─── GEOMETRY FIX ────────────────────────────────────────────────
            # Maintained original grid direction (2 Rows x n_cols Columns).
            # Expanded baseline canvas width to (10 * n_cols) and height to 21 inches.
            # This allocates a 10x10.5 inch space per square subplot, giving the
            # long figure title natural room to fit without generating white side margins.
            fig, axes = plt.subplots(2, n_cols,
                                     figsize=(10 * n_cols, 21),
                                     squeeze=False)

            part_str = f" (Part {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            
            # Expanded main title font size with newline padding
            fig.suptitle(
                f"Pipeline B: {TARGET_CORE} | landmark {mc}{part_str}\n"
                f"Adjacent-slice overlay",
                fontsize=28, fontweight='bold'
            )

            # Enumerate slice pairs along the horizontal columns
            for col, (_, pair_row) in enumerate(chunk.iterrows()):
                sidx_a = int(pair_row['slice_idx_a'])
                sidx_b = int(pair_row['slice_idx_b'])
                
                z_a    = sidx_a + 1
                z_b    = sidx_b + 1
                
                wx_a   = float(pair_row['x_warped_a'])
                wy_a   = float(pair_row['y_warped_a'])
                wx_b   = float(pair_row['x_warped_b'])
                wy_b   = float(pair_row['y_warped_b'])

                dist_px = float(pair_row['TRE_px'])
                dist_um = float(pair_row['TRE_um'])
                mid_x   = (wx_a + wx_b) / 2
                mid_y   = (wy_a + wy_b) / 2

                # Enumerate channel stains along the vertical rows
                for row_idx, (ch_idx, ch_label, ch_crop) in enumerate([
                        (DAPI_CHANNEL_IDX, "DAPI", dapi_crop_half),
                        (CK_CHANNEL_IDX,   "CK",   ck_crop_half)]):

                    # Maintained original matrix array indexing layout
                    ax    = axes[row_idx][col]
                    img_a = load_slice_channel_valis(sidx_a, ch_idx)
                    img_b = load_slice_channel_valis(sidx_b, ch_idx)

                    if img_a is None and img_b is None:
                        ax.set_title(f"{ch_label}  z {z_a}→{z_b}\n(unavailable)", fontsize=20)
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

            # Restrict subplots to 95% height to prevent overlapping with the main title text
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            file_suffix  = f"_pt{chunk_idx + 1}" if len(chunks) > 1 else ""
            overlay_path = os.path.join(
                VERIFY_OUTPUT, f"{TARGET_CORE}_VALIS_adjacent_overlay_mclass{mc}{file_suffix}.png"
            )
            fig.savefig(overlay_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Adjacent overlay (mclass {mc}{part_str}) → {overlay_path}")
plot_adjacent_slice_overlays_valis(df_detail)

logger.info("Done.")