"""
phenotype_cells.py
==================
Cell phenotyping based on DAPI-segmented nuclei and registered multi-channel images.

Pipeline per core:
  1. Load the registered volume  (<CORE>_AKAZE_RoMaV2_Linear_Aligned.ome.tif)  — ZCYX uint16
  2. For each slice Z:
       a. Load the warped DAPI mask  (CellPose_DAPI_Warped/<CORE>/TMA_<ID>_DAPI_cp_masks.tif)
       b. For each marker channel (CD31, GAP43, NFP, CD3, CD163, CK, AF):
            - Extract per-nucleus mean intensity using the DAPI mask labels
            - Compute Mean+2SD threshold on the per-cell mean intensity distribution
            - Assign binary positivity (0/1) per cell per marker
  3. Concatenate all slices → one CSV per core

Output columns
--------------
    core, slice_z, slice_id, cell_id,
    mean_CD31, mean_GAP43, mean_NFP, mean_CD3, mean_CD163, mean_CK, mean_AF,
    pos_CD31,  pos_GAP43,  pos_NFP,  pos_CD3,  pos_CD163,  pos_CK,  pos_AF,
    area_px, area_um2, centroid_x, centroid_y,
    thresh_CD31, thresh_GAP43, thresh_NFP, thresh_CD3, thresh_CD163, thresh_CK, thresh_AF

Usage
-----
    python phenotype_cells.py --core_name Core_01 [--plot_qc] [--min_area_px 200]

Input
-----
    Registered volume :
        <DATASPACE>/Filter_AKAZE_RoMaV2_Linear_Warp_map/<CORE>/<CORE>_AKAZE_RoMaV2_Linear_Aligned.ome.tif
    Warped DAPI masks :
        <DATASPACE>/CellPose_DAPI_Warped/<CORE>/TMA_<ID>_DAPI_cp_masks.tif

Output
------
    <DATASPACE>/Phenotypes/<CORE>/<CORE>_phenotypes.csv
    <DATASPACE>/Phenotypes/<CORE>/qc_plots/  (if --plot_qc)
"""

import os
import sys
import re
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy import ndimage as ndi

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match akaze_linear_romav2_warp_map.py)
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
MARKER_CHANNELS  = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']   # DAPI excluded
CHANNEL_IDX      = {name: i for i, name in enumerate(CHANNEL_NAMES)}
PIXEL_SIZE_XY_UM = 0.4961

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Phenotype cells: measure marker expression in DAPI-segmented nuclei.'
)
parser.add_argument('--core_name',    type=str, required=True)
parser.add_argument('--min_area_px',  type=int, default=200,
                    help='Minimum nucleus area in pixels to include (default: 200).')
parser.add_argument('--plot_qc',      action='store_true',
                    help='Save per-slice QC overlay images.')
args = parser.parse_args()

TARGET_CORE = args.core_name

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REGISTERED_VOL = os.path.join(
    config.DATASPACE,
    'Filter_AKAZE_RoMaV2_Linear_Warp_map',
    TARGET_CORE,
    f'{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif',
)

DAPI_MASK_DIR = os.path.join(
    config.DATASPACE,
    'CellPose_DAPI_Warped',
    TARGET_CORE,
)

OUTPUT_DIR = os.path.join(config.DATASPACE, 'Phenotypes', TARGET_CORE)
QC_DIR     = os.path.join(OUTPUT_DIR, 'qc_plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
if args.plot_qc:
    os.makedirs(QC_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id(path):
    """Extract integer slice ID from filename TMA_<ID>_..."""
    m = re.search(r'TMA_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def otsu_threshold(values: np.ndarray) -> float:
    """Compute Otsu threshold on a 1-D array of per-cell mean intensities."""
    if len(values) < 2:
        return float(values[0]) if len(values) == 1 else 0.0
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return float(vmax)
    counts, bin_edges = np.histogram(values, bins=256)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return float(np.median(values))
    prob     = counts / total
    best_t   = bin_centres[0]
    best_var = -1.0
    w0 = 0.0; mu0_sum = 0.0
    mu_total = float(np.sum(prob * bin_centres))
    for i in range(len(prob)):
        w0 += prob[i]; mu0_sum += prob[i] * bin_centres[i]
        w1 = 1.0 - w0
        if w0 == 0.0 or w1 == 0.0:
            continue
        mu0 = mu0_sum / w0
        mu1 = (mu_total - mu0_sum) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var; best_t = bin_centres[i]
    return float(best_t)


def background_floor(ch_img: np.ndarray, k: float = 3.0) -> float:
    """
    Estimate per-channel background noise floor as median + k * MAD over all
    pixels. MAD is robust to bright signal outliers — it characterises the
    noise level regardless of how many positive cells are present.
    Default k=3.0 covers ~99.7% of Gaussian noise.
    """
    flat = ch_img.ravel().astype(np.float32)
    med  = float(np.median(flat))
    mad  = float(np.median(np.abs(flat - med)))
    return med + k * mad


def compute_threshold(values: np.ndarray, ch_img: np.ndarray,
                      k_mad: float = 3.0) -> tuple:
    """
    Final threshold = max(Otsu on per-cell means, background floor).
    Prevents Otsu from collapsing to near-zero on sparse/dim channels (e.g. NFP).
    Returns (threshold, otsu_val, floor_val) for CSV logging and QC plots.
    """
    o_val = otsu_threshold(values)
    f_val = background_floor(ch_img, k=k_mad)
    return float(max(o_val, f_val)), float(o_val), float(f_val)


def measure_slice(mask: np.ndarray, volume_slice: np.ndarray, min_area: int):
    """
    Measure per-cell marker intensities for one slice.

    Parameters
    ----------
    mask         : (H, W) uint32  — DAPI label mask (0 = background)
    volume_slice : (C, H, W) uint16 — registered multi-channel image for this slice
    min_area     : int — minimum nucleus area in pixels

    Returns
    -------
    pd.DataFrame with one row per cell, columns: cell_id, area_px, centroid_x,
    centroid_y, mean_<MARKER>, pos_<MARKER>, thresh_<MARKER>, otsu_<MARKER>, floor_<MARKER>
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]   # remove background

    if len(labels) == 0:
        return pd.DataFrame()

    # ── Fast label statistics via scipy ─────────────────────────────────────
    # Pre-compute label index array for ndimage (faster than iterating per label)
    label_list = labels.tolist()

    # Area per label
    areas = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int32)

    # Filter by min area
    keep      = areas >= min_area
    label_list = [l for l, k in zip(label_list, keep) if k]
    areas      = areas[keep]

    if len(label_list) == 0:
        return pd.DataFrame()

    # Centroids
    cy = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape),
        mask, label_list))
    cx = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[1])[None, :], mask.shape),
        mask, label_list))

    # Per-marker mean intensity
    marker_means = {}
    for ch_name in MARKER_CHANNELS:
        ch_idx = CHANNEL_IDX[ch_name]
        ch_img = volume_slice[ch_idx].astype(np.float32)
        means  = np.array(ndi.mean(ch_img, mask, label_list), dtype=np.float32)
        marker_means[ch_name] = means

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        'cell_id':    label_list,
        'area_px':    areas,
        'area_um2':   np.round(areas * PIXEL_SIZE_XY_UM ** 2, 3),
        'centroid_x': np.round(cx, 1),
        'centroid_y': np.round(cy, 1),
    })

    # ── Otsu + background floor thresholding ────────────────────────────────
    # threshold = max(Otsu on per-cell means, median+3*MAD on full channel image)
    # Prevents Otsu from collapsing near zero on sparse/dim channels (e.g. NFP).
    for ch_name in MARKER_CHANNELS:
        means  = marker_means[ch_name]
        ch_img = volume_slice[CHANNEL_IDX[ch_name]]
        df[f'mean_{ch_name}'] = np.round(means, 2)

        threshold, otsu_val, floor_val = compute_threshold(means, ch_img)
        df[f'thresh_{ch_name}'] = round(threshold,  2)
        df[f'otsu_{ch_name}']   = round(otsu_val,   2)
        df[f'floor_{ch_name}']  = round(floor_val,  2)
        df[f'pos_{ch_name}']    = (means >= threshold).astype(np.uint8)

    return df


def build_label_map(mask, cell_ids, values, dtype=np.float32):
    """Paint a per-cell scalar value onto a label mask image (fast vectorised)."""
    out = np.zeros(mask.shape, dtype=dtype)
    # Build lookup table: label → value  (labels can be non-contiguous)
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=dtype)
    for cid, val in zip(cell_ids, values):
        if cid <= max_label:
            lut[cid] = val
    out = lut[mask]   # fancy indexing — much faster than per-cell loop
    return out


def save_qc_plot(dapi_img, mask, df_slice, volume_slice, slice_id, out_path):
    """
    Save a per-marker QC panel showing for each marker:
      - raw channel image (percentile-stretched)
      - nucleus positivity map (green=positive, black=negative)
      - Mean+2SD threshold marked on the intensity histogram
    Layout: rows = markers, cols = (raw image | positivity map | histogram)
    Plus a top row for DAPI overview.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_markers = len(MARKER_CHANNELS)
        n_rows    = 1 + n_markers        # DAPI overview row + one row per marker
        n_cols    = 3                    # raw | positivity | histogram
        fig = plt.figure(figsize=(18, 4 * n_rows))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.35, wspace=0.15)

        n_cells = len(df_slice)
        fig.suptitle(
            f'{TARGET_CORE}  |  Slice {slice_id}  |  {n_cells} nuclei',
            fontsize=14, fontweight='bold', y=1.002,
        )

        def stretch(img):
            """Percentile stretch to [0,1] for display."""
            tissue = img[mask > 0]
            if tissue.size == 0 or tissue.max() == tissue.min():
                return img.astype(np.float32)
            lo, hi = np.percentile(tissue, [1, 99])
            if hi == lo:
                return np.zeros_like(img, dtype=np.float32)
            return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)

        # ── Row 0: DAPI overview + total positivity score ────────────────────
        ax_dapi = fig.add_subplot(gs[0, 0])
        ax_dapi.imshow(stretch(dapi_img), cmap='gray', interpolation='nearest')
        ax_dapi.set_title('DAPI', fontsize=10)
        ax_dapi.axis('off')

        ax_score = fig.add_subplot(gs[0, 1])
        if n_cells > 0:
            pos_cols  = [f'pos_{ch}' for ch in MARKER_CHANNELS]
            scores    = df_slice[pos_cols].sum(axis=1).values.astype(np.float32)
            score_map = build_label_map(mask, df_slice['cell_id'].values, scores)
        else:
            score_map = np.zeros(mask.shape, dtype=np.float32)
        im = ax_score.imshow(score_map, cmap='hot',
                             vmin=0, vmax=n_markers, interpolation='nearest')
        plt.colorbar(im, ax=ax_score, fraction=0.03, pad=0.02)
        ax_score.set_title(f'Total marker score (0–{n_markers})', fontsize=10)
        ax_score.axis('off')

        # Positivity bar chart in col 2
        ax_bar = fig.add_subplot(gs[0, 2])
        if n_cells > 0:
            pct_pos = [100.0 * df_slice[f'pos_{ch}'].sum() / n_cells for ch in MARKER_CHANNELS]
        else:
            pct_pos = [0] * n_markers
        colors = ['#e74c3c' if p > 50 else '#3498db' for p in pct_pos]
        bars = ax_bar.barh(MARKER_CHANNELS, pct_pos, color=colors)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('% positive cells', fontsize=9)
        ax_bar.set_title('Positivity summary', fontsize=10)
        for bar, pct in zip(bars, pct_pos):
            ax_bar.text(min(pct + 1, 98), bar.get_y() + bar.get_height() / 2,
                        f'{pct:.0f}%', va='center', fontsize=8)
        ax_bar.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # ── Rows 1+: one row per marker ──────────────────────────────────────
        for row, ch in enumerate(MARKER_CHANNELS, start=1):
            ch_idx  = CHANNEL_IDX[ch]
            ch_img  = volume_slice[ch_idx]
            thresh_t  = float(df_slice[f'thresh_{ch}'].iloc[0]) if n_cells > 0 else 0.0
            means   = df_slice[f'mean_{ch}'].values if n_cells > 0 else np.array([])
            pos_arr = df_slice[f'pos_{ch}'].values  if n_cells > 0 else np.array([])

            # Col 0: raw channel image (percentile stretched)
            ax_raw = fig.add_subplot(gs[row, 0])
            ax_raw.imshow(stretch(ch_img), cmap='gray', interpolation='nearest')
            ax_raw.set_title(f'{ch}  raw', fontsize=9)
            ax_raw.axis('off')

            # Col 1: positivity map  (green = positive, dark = negative)
            ax_pos = fig.add_subplot(gs[row, 1])
            if n_cells > 0:
                pos_map = build_label_map(mask, df_slice['cell_id'].values,
                                          pos_arr.astype(np.float32))
            else:
                pos_map = np.zeros(mask.shape, dtype=np.float32)
            # Background black, negative dark-red, positive green
            rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
            rgb[pos_map == 1] = [0.2, 0.9, 0.2]   # green  = positive
            rgb[(mask > 0) & (pos_map == 0)] = [0.4, 0.05, 0.05]  # dark-red = negative
            ax_pos.imshow(rgb, interpolation='nearest')
            n_pos = int(pos_arr.sum()) if n_cells > 0 else 0
            pct   = 100.0 * n_pos / n_cells if n_cells > 0 else 0
            ax_pos.set_title(f'{ch}  positivity  ({n_pos}/{n_cells}, {pct:.0f}%)', fontsize=9)
            ax_pos.axis('off')

            # Col 2: histogram of per-cell mean intensities + Mean+2SD threshold line
            ax_hist = fig.add_subplot(gs[row, 2])
            if len(means) > 1:
                ax_hist.hist(means, bins=80, color='steelblue', alpha=0.75,
                             edgecolor='none')
                otsu_v  = float(df_slice[f'otsu_{ch}'].iloc[0])
                floor_v = float(df_slice[f'floor_{ch}'].iloc[0])
                ax_hist.axvline(thresh_t, color='red',    linewidth=2.0,
                                label=f'Threshold = {thresh_t:.0f}')
                ax_hist.axvline(otsu_v,   color='orange', linewidth=1.2,
                                linestyle='--', label=f'Otsu = {otsu_v:.0f}')
                ax_hist.axvline(floor_v,  color='green',  linewidth=1.2,
                                linestyle='--', label=f'Floor = {floor_v:.0f}')
                ax_hist.legend(fontsize=8)
            ax_hist.set_xlabel('Mean intensity (per nucleus)', fontsize=8)
            ax_hist.set_ylabel('Cell count', fontsize=8)
            ax_hist.set_title(f'{ch}  intensity distribution', fontsize=9)
            ax_hist.tick_params(labelsize=7)

        plt.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  QC plot saved: {os.path.basename(out_path)}')
    except Exception as exc:
        logger.warning(f'  QC plot failed for slice {slice_id}: {exc}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f'Phenotyping — core: {TARGET_CORE}')
    logger.info(f'Registered volume : {REGISTERED_VOL}')
    logger.info(f'DAPI mask dir     : {DAPI_MASK_DIR}')

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(REGISTERED_VOL):
        logger.error(f'Registered volume not found: {REGISTERED_VOL}')
        sys.exit(1)

    if not os.path.isdir(DAPI_MASK_DIR):
        logger.error(f'DAPI mask directory not found: {DAPI_MASK_DIR}')
        sys.exit(1)

    # ── Load registered volume (ZCYX) ────────────────────────────────────────
    logger.info('Loading registered volume ...')
    vol = tifffile.imread(REGISTERED_VOL)  # expected shape: (Z, C, H, W)

    # Normalise axis order to (Z, C, H, W)
    if vol.ndim == 3:
        # Single-slice single-channel edge case: (H, W) → (1, 1, H, W)
        vol = vol[np.newaxis, np.newaxis]
    elif vol.ndim == 4:
        # Could be (Z, C, H, W) or (C, Z, H, W) — check C dimension
        # CHANNEL_NAMES has 8 entries; if axis-0 == 8 it's (C, Z, H, W)
        if vol.shape[0] == len(CHANNEL_NAMES) and vol.shape[1] != len(CHANNEL_NAMES):
            vol = np.moveaxis(vol, 0, 1)   # → (Z, C, H, W)

    n_slices, n_channels, H, W = vol.shape
    logger.info(f'Volume shape: Z={n_slices}  C={n_channels}  H={H}  W={W}')

    if n_channels != len(CHANNEL_NAMES):
        logger.warning(
            f'Expected {len(CHANNEL_NAMES)} channels, got {n_channels}. '
            f'Proceeding — verify CHANNEL_NAMES matches your data.'
        )

    # ── Build slice-ID → volume z-index mapping from registration stats CSV ─────
    # The registered volume Z axis follows the order slices were processed
    # (centre-out), NOT necessarily ascending slice ID order.
    # The registration stats CSV has Slice_Z (0-based volume index) and Slice_ID.
    reg_stats_csv = os.path.join(
        config.DATASPACE,
        'Filter_AKAZE_RoMaV2_Linear_Warp_map',
        TARGET_CORE,
        'registration_stats_AKAZE_RoMaV2_Linear.csv',
    )
    slice_id_to_z = {}   # {slice_id (int) → z_idx in volume (int)}
    if os.path.exists(reg_stats_csv):
        reg_df = pd.read_csv(reg_stats_csv)
        for _, row in reg_df.iterrows():
            slice_id_to_z[int(row['Slice_ID'])] = int(row['Slice_Z'])
        # Also add the anchor slice (centre) which has no row in the stats CSV
        # because it was never registered — it IS the reference.
        # Its z-index is simply whichever z is missing from the mapping.
        all_z = set(range(n_slices))
        mapped_z = set(slice_id_to_z.values())
        missing_z = all_z - mapped_z
        if len(missing_z) == 1:
            # Infer anchor slice_id from the mask files not yet in the map
            mask_ids = {get_slice_id(p) for p in glob.glob(
                os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif'))}
            unmapped_ids = mask_ids - set(slice_id_to_z.keys())
            if len(unmapped_ids) == 1:
                anchor_id = unmapped_ids.pop()
                anchor_z  = missing_z.pop()
                slice_id_to_z[anchor_id] = anchor_z
                logger.info(f'Anchor slice inferred: ID={anchor_id} → Z={anchor_z}')
        logger.info(f'Slice ID→Z mapping loaded from CSV: {len(slice_id_to_z)} entries')
    else:
        logger.warning(
            f'Registration stats CSV not found at {reg_stats_csv}. '
            f'Falling back to sorted-order matching (may be incorrect if a slice was skipped).'
        )

    # ── Find warped DAPI masks ────────────────────────────────────────────────
    mask_files = sorted(
        glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif')),
        key=get_slice_id,
    )

    if not mask_files:
        logger.error(f'No DAPI mask files found in {DAPI_MASK_DIR}')
        sys.exit(1)

    logger.info(f'Found {len(mask_files)} DAPI mask files.')

    # ── Process each slice ────────────────────────────────────────────────────
    all_records = []

    for enum_idx, mask_path in enumerate(mask_files):
        slice_id = get_slice_id(mask_path)

        # Resolve volume z-index: prefer CSV mapping, fall back to enumerate order
        if slice_id_to_z and slice_id in slice_id_to_z:
            z_idx = slice_id_to_z[slice_id]
        else:
            z_idx = enum_idx
            if slice_id_to_z:
                logger.warning(f'  Slice ID {slice_id} not in CSV mapping — using position {enum_idx}.')

        logger.info(f'  Slice Z={z_idx:03d}  ID={slice_id:03d}  '
                    f'({enum_idx + 1}/{len(mask_files)})')

        # Load DAPI mask
        mask = tifffile.imread(mask_path).astype(np.uint32)
        if mask.ndim != 2:
            logger.warning(f'  Unexpected mask shape {mask.shape} — squeezing.')
            mask = mask.squeeze()

        # Sanity check spatial dimensions match
        if mask.shape != (H, W):
            logger.warning(
                f'  Mask shape {mask.shape} != volume ({H}, {W}) — skipping slice.'
            )
            continue

        # Get corresponding registered volume slice
        if z_idx >= n_slices:
            logger.warning(f'  z_idx {z_idx} out of range for volume ({n_slices} slices) — skipping.')
            continue
        volume_slice = vol[z_idx]   # (C, H, W)

        # Measure
        df_slice = measure_slice(mask, volume_slice, min_area=args.min_area_px)

        if df_slice.empty:
            logger.warning(f'  No cells found in slice {slice_id} (after min_area filter).')
            continue

        # Add slice/core metadata
        df_slice.insert(0, 'slice_id', slice_id)
        df_slice.insert(0, 'slice_z',  z_idx)
        df_slice.insert(0, 'core',     TARGET_CORE)

        n_pos_summary = {ch: int(df_slice[f'pos_{ch}'].sum()) for ch in MARKER_CHANNELS}
        logger.info(
            f'  {len(df_slice)} nuclei | '
            + '  '.join(f'{ch}+={n}' for ch, n in n_pos_summary.items())
        )

        all_records.append(df_slice)

        # QC plot
        if args.plot_qc:
            qc_path = os.path.join(QC_DIR, f'TMA_{slice_id:03d}_DAPI_phenotype_qc.png')
            dapi_img = volume_slice[CHANNEL_IDX['DAPI']]
            save_qc_plot(dapi_img, mask, df_slice, volume_slice, slice_id, qc_path)

    # ── Write output CSV ──────────────────────────────────────────────────────
    if not all_records:
        logger.error('No cells phenotyped across any slice — no CSV written.')
        sys.exit(1)

    df_all = pd.concat(all_records, ignore_index=True)

    # Column order: metadata | areas | centroids | means | positivity | thresholds
    meta_cols   = ['core', 'slice_z', 'slice_id', 'cell_id',
                   'area_px', 'area_um2', 'centroid_x', 'centroid_y']
    mean_cols   = [f'mean_{ch}' for ch in MARKER_CHANNELS]
    pos_cols    = [f'pos_{ch}'  for ch in MARKER_CHANNELS]
    thresh_cols = [f'thresh_{ch}' for ch in MARKER_CHANNELS]
    otsu_cols   = [f'otsu_{ch}'  for ch in MARKER_CHANNELS]
    floor_cols  = [f'floor_{ch}' for ch in MARKER_CHANNELS]
    df_all      = df_all[meta_cols + mean_cols + pos_cols + thresh_cols + otsu_cols + floor_cols]

    csv_path = os.path.join(OUTPUT_DIR, f'{TARGET_CORE}_phenotypes.csv')
    df_all.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_cells = len(df_all)
    logger.info('=' * 60)
    logger.info(f'Done.  Core: {TARGET_CORE}  |  Total nuclei: {total_cells}')
    logger.info(f'Output CSV: {csv_path}')
    for ch in MARKER_CHANNELS:
        n_pos  = int(df_all[f'pos_{ch}'].sum())
        pct    = 100.0 * n_pos / total_cells if total_cells > 0 else 0.0
        logger.info(f'  {ch:8s}: {n_pos:6d} positive  ({pct:.1f}%)')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()