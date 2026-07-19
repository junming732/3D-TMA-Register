"""
accuracy_landmarks_deform.py
───────────────────────────────────
Compute registration accuracy using rough annotations as landmarks, for
either deformation-map-based registration pipeline: BSpline (AKAZE +
TissueMask BSpline) or RoMaV2 (AKAZE + RoMaV2 dense warp), selected via
--pipeline.

This merges what were previously two near-identical scripts
(accuracy_landmarks_bspline.py, accuracy_landmarks_roma.py) — a diff between
them showed zero remaining logic difference after landmark_accuracy_common.py
/ landmark_accuracy_deform_common.py were extracted, only path/label strings.
See PIPELINE_CONFIGS below for exactly what differs between the two.

NOT used for VALIS — that pipeline stores its transform in a pickled
Registrar object and loads overlay images per-slice rather than from a
merged volume, so it isn't a config-only difference; see
valis_accuracy_landmarks.py.

Each mclass groups annotations that mark the **same anatomical structure**
across z-levels.  After warping every annotation point into registered space,
all points of the same mclass should overlap.  The spread of those warped
points is a direct measure of registration accuracy.

Metrics computed
────────────────
Per mclass (per structure):
  - Pairwise Euclidean distance between all warped points (in pixels and µm)
  - Mean, median, max, std of pairwise distances

Per-slice (relative to its neighbours):

Global summary:
  - Mean / median / max target registration error (TRE) across all landmarks

Outputs
───────
  registration_accuracy_landmarks.csv   — one row per annotation point
  registration_accuracy_summary.csv     — one row per mclass
  registration_accuracy_plot.png        — scatter + boxplot

All written to:
  <OUTPUT_FOLDER>/<verify_subdir>/   (verify_subdir is pipeline-specific, see below)

Usage
─────
    python accuracy_landmarks_deform.py \
        --core_name core_09 --pipeline bspline \
        --annotation_json /path/to/rough_annotation_core_09.json \
        [--pixel_size_um 0.4961]   # physical pixel size (default from registration script)
        [--landmark_id all]             # 'all' or comma-separated ints

    python accuracy_landmarks_deform.py --core_name core_09 --pipeline roma \
        --annotation_json /path/to/rough_annotation_core_09.json
"""

import os, re, sys, json, glob, argparse, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── config.py ─────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config
from landmark_accuracy_common import (
    z_json_to_slice_idx, get_slice_number, make_two_channel_rgb, _annotate_overlay_ax,
)
from landmark_accuracy_deform_common import (
    load_slice_filter, find_deform_npz, warp_point, load_slice_channel_from_vol,
)

# ─── PIPELINE CONFIG — the only real difference between BSpline and RoMaV2 ────
PIPELINE_CONFIGS = {
    'bspline': dict(
        label               = 'Pipeline A',
        work_output_dir     = 'Filter_AKAZE_TissueMask_BSpline',
        verify_subdir       = 'annotation_verification_bspline',
        vol_filename_suffix = '_AKAZE_TissueMask_Aligned.ome.tif',
        plot_filename_suffix = '',
    ),
    'roma': dict(
        label               = 'Pipeline C',
        work_output_dir     = 'Filter_AKAZE_RoMaV2_Linear_Warp_map_hr_isolated',
        verify_subdir       = 'annotation_verification_Romav2',
        vol_filename_suffix = '_AKAZE_RoMaV2_Linear_Aligned.ome.tif',
        plot_filename_suffix = '_Romav2',
    ),
}

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--core_name',       type=str, required=True)
parser.add_argument('--annotation_json', type=str, required=True)
parser.add_argument('--pipeline',        type=str, required=True,
                    choices=sorted(PIPELINE_CONFIGS.keys()),
                    help='Which deformation-map-based registration pipeline to evaluate.')
parser.add_argument('--pixel_size_um',   type=float, default=0.4961,
                    help='Pixel size in µm (default 0.4961, from registration script).')
parser.add_argument('--landmark_id',          type=str, default='all')
parser.add_argument('--work_output_dir', type=str, default=None,
                    help='Override the folder under DATASPACE holding this pipeline\'s '
                         'registration output (default: PIPELINE_CONFIGS[--pipeline]'
                         '[\'work_output_dir\']). Use this to point at a different '
                         'experiment run without editing PIPELINE_CONFIGS.')
parser.add_argument('--verify_subdir', type=str, default=None,
                    help='Override the annotation_verification subfolder name '
                         '(default: PIPELINE_CONFIGS[--pipeline][\'verify_subdir\']).')
args = parser.parse_args()

PCFG = PIPELINE_CONFIGS[args.pipeline]
if args.work_output_dir is not None:
    PCFG = {**PCFG, 'work_output_dir': args.work_output_dir}
if args.verify_subdir is not None:
    PCFG = {**PCFG, 'verify_subdir': args.verify_subdir}

TARGET_CORE    = args.core_name
PIXEL_SIZE_UM  = args.pixel_size_um

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── PATHS — mirrors registration script ──────────────────────────────────────
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER   = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT    = os.path.join(config.DATASPACE, PCFG['work_output_dir'])
OUTPUT_FOLDER  = os.path.join(WORK_OUTPUT, TARGET_CORE)
DEFORM_FOLDER     = os.path.join(OUTPUT_FOLDER, "deformation_maps")
VERIFY_OUTPUT     = os.path.join(OUTPUT_FOLDER, PCFG['verify_subdir'])
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
os.makedirs(VERIFY_OUTPUT, exist_ok=True)

logger.info(f"Core         : {TARGET_CORE}")
logger.info(f"Deform folder: {DEFORM_FOLDER}")
logger.info(f"Output       : {VERIFY_OUTPUT}")

# ─── FILE LIST — mirrors registration script ──────────────────────────────────
FILE_LIST = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")), key=get_slice_number)
if not FILE_LIST:
    logger.error(f"No .ome.tif files in {INPUT_FOLDER}")
    sys.exit(1)
logger.info(f"Found {len(FILE_LIST)} slices")

# ─── FILTERED SLICE LIST + SLICE→VOL_Z MAP ────────────────────────────────────
# The registered volume only contains filtered slices, so Z indices in the
# volume correspond to positions in FILTERED_FILE_LIST, not FILE_LIST.
# Mirrors the working RoMaV2 accuracy script exactly.

allowed_positions = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)

if allowed_positions is not None:
    FILTERED_FILE_LIST = [f for i, f in enumerate(FILE_LIST) if i in allowed_positions]
    logger.info(f"Slice filter active: {len(FILTERED_FILE_LIST)}/{len(FILE_LIST)} slices kept.")
else:
    FILTERED_FILE_LIST = FILE_LIST
    logger.info("No slice filter — using all slices.")

# slice_idx (0-based position into FILE_LIST) → vol_z (position in the
# registered volume, which only contains filtered slices). See
# z_json_to_slice_idx in landmark_accuracy_common.py for the annotation-JSON
# to slice_idx conversion (z_json - 1).
file_to_orig_pos = {f: i for i, f in enumerate(FILE_LIST)}
SLICE_IDX_TO_VOL_Z = {
    file_to_orig_pos[f]: vol_z
    for vol_z, f in enumerate(FILTERED_FILE_LIST)
}
logger.info(f"Slice→vol_z map: {SLICE_IDX_TO_VOL_Z}")


# ─── DEFORMATION HELPERS ──────────────────────────────────────────────────────
# find_deform_npz / warp_point imported from landmark_accuracy_deform_common.

# ─── LOAD ANNOTATIONS ─────────────────────────────────────────────────────────
with open(args.annotation_json) as fh:
    ann_data = json.load(fh)

annotations = ann_data
logger.info(f"Loaded {len(annotations)} annotations.")

if args.landmark_id.lower() != 'all':
    keep = set(int(m) for m in args.landmark_id.split(','))
    annotations = [a for a in annotations if a['landmark_id'] in keep]
    logger.info(f"After mclass filter: {len(annotations)} annotations.")

# ─── WARP EVERY LANDMARK INTO REGISTERED SPACE ───────────────────────────────
records = []

for ann in annotations:
    z_json    = ann['z']
    slice_idx = z_json_to_slice_idx(z_json)
    x_raw     = ann['x']
    y_raw     = ann['y']
    mc        = ann['landmark_id']
    ann_id    = ann['landmark_id']

    npz_path  = find_deform_npz(slice_idx, TARGET_CORE, DEFORM_FOLDER)

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
        'landmark_id': mc,
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
detail_csv  = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_detail.csv")
summary_csv = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_summary.csv")
df_detail.to_csv(detail_csv,   index=False)
df_summary.to_csv(summary_csv, index=False)
logger.info(f"Detail CSV  → {detail_csv}")
logger.info(f"Summary CSV → {summary_csv}")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
from matplotlib.colors import TABLEAU_COLORS
colour_list   = list(TABLEAU_COLORS.values())
all_mclasses  = sorted(df_detail['landmark_id'].unique())
mclass_colour = {mc: colour_list[i % len(colour_list)] for i, mc in enumerate(all_mclasses)}

# 1 row, 2 columns. Increased figsize to accommodate larger fonts.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"{PCFG['label']}: {TARGET_CORE} — Landmark Registration Accuracy  "
             f"(pixel size = {PIXEL_SIZE_UM} µm)",
             fontsize=18, fontweight='bold')

# ── Panel 1: TRE per consecutive pair vs slice pair ─────────────────────────
ax = axes[0]
for mc in all_mclasses:
    grp = df_detail[df_detail['landmark_id'] == mc].sort_values('z_json_a').reset_index(drop=True)
    # Convert from 0-based slice_idx to 1-based slice index
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
ax.set_xlabel("slice pair", fontsize=13)  # Removed z_json_a -> z_json_b reference
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

plot_path = os.path.join(VERIFY_OUTPUT, f"{TARGET_CORE}_landmark_accuracy_plot{PCFG['plot_filename_suffix']}.png")
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
# Both rows use the same red/green blend and identical annotations so the
# error vector is visible against both stains.

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6

# load_slice_channel_from_vol, make_two_channel_rgb, _annotate_overlay_ax
# imported from landmark_accuracy_common / landmark_accuracy_deform_common.


def plot_adjacent_slice_overlays(df_detail,
                                 dapi_crop_half=50,
                                 ck_crop_half=150,
                                 dpi=120):
    """
    For each mclass with ≥2 z-levels, produce one PNG with:
      Row 0 — DAPI (ch 0) overlay — tighter crop (dapi_crop_half=50)
      Row 1 — CK   (ch 6) overlay — wider crop  (ck_crop_half=150)
    Generates multiple files (max 3 columns) if the number of pairs is large.
    """
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed — skipping adjacent-slice overlays.")
        return

    reg_path = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}{PCFG['vol_filename_suffix']}")
    if not os.path.isfile(reg_path):
        logger.warning(f"Registered volume not found at {reg_path} — skipping overlays.")
        return

    logger.info(f"Loading registered volume from {reg_path} ...")
    reg_vol = tifffile.imread(reg_path)  
    logger.info(f"Registered volume shape: {reg_vol.shape}")

    max_cols = 3

    for mc, grp in df_detail.groupby('landmark_id'):
        grp_sorted = grp.sort_values('z_json_a').reset_index(drop=True)
        n_pairs = len(grp_sorted)
        if n_pairs < 1:
            continue

        chunks = [grp_sorted.iloc[i:i + max_cols] for i in range(0, n_pairs, max_cols)]

        for chunk_idx, chunk in enumerate(chunks):
            n_cols = len(chunk)
            
            # Expanded canvas size to 10x21 inches (10x10 per square subplot + title clearance).
            # This balances the layout grid's aspect ratio and completely eliminates side padding.
            fig, axes = plt.subplots(2, n_cols,
                                     figsize=(10 * n_cols, 21),
                                     squeeze=False)

            part_str = f" (Part {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            fig.suptitle(
                f"{PCFG['label']}: {TARGET_CORE} | landmark {mc}{part_str}\n"
                f"Adjacent-slice overlay",
                fontsize=28, fontweight='bold'
            )

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

                for row_idx, (ch_idx, ch_label, ch_crop) in enumerate([
                        (DAPI_CHANNEL_IDX, "DAPI", dapi_crop_half),
                        (CK_CHANNEL_IDX,   "CK",   ck_crop_half)]):

                    ax    = axes[row_idx][col]
                    img_a = load_slice_channel_from_vol(reg_vol, sidx_a, ch_idx, SLICE_IDX_TO_VOL_Z)
                    img_b = load_slice_channel_from_vol(reg_vol, sidx_b, ch_idx, SLICE_IDX_TO_VOL_Z)

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

            # Restrict subplots to 95% height to prevent overlapping with the large fig title
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            file_suffix  = f"_pt{chunk_idx + 1}" if len(chunks) > 1 else ""
            overlay_path = os.path.join(
                VERIFY_OUTPUT, f"{TARGET_CORE}_adjacent_overlay_mclass{mc}{file_suffix}.png"
            )
            fig.savefig(overlay_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Adjacent overlay (mclass {mc}{part_str}) → {overlay_path}")


plot_adjacent_slice_overlays(df_detail)

logger.info("Done.")