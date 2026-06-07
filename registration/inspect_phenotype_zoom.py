"""
inspect_phenotype_zoom.py
=========================
Zoomed-in phenotype QC viewer for a single core / slice / channel.

Instead of fixed quadrants, the default mode finds a dense cluster of
positive cells and crops a small window around it — so you always see
real signal at a scale where individual cells are visible (~500–1000 px).

The histograms (panels 2 and 3) are always computed from the FULL slice so
that the threshold and distribution are not affected by the crop choice.

Usage examples
--------------
# Default: auto-crop around densest positive cluster, CD31, slice 5
python inspect_phenotype_zoom.py --core_name Core_01 --slice_id 5 --channel CD31

# Save N crops around the top-N positive clusters (default N=3)
python inspect_phenotype_zoom.py --core_name Core_01 --slice_id 5 --channel CD3 --n_crops 5

# Fix the crop size in pixels (default 800)
python inspect_phenotype_zoom.py --core_name Core_01 --slice_id 5 --channel CK --crop_size 600

# Manual pixel crop  (row_start row_end col_start col_end)
python inspect_phenotype_zoom.py --core_name Core_01 --slice_id 5 --channel CD163 \\
    --crop 1000 1800 2000 2800

# Override threshold (e.g. consensus from a previous full run)
python inspect_phenotype_zoom.py --core_name Core_01 --slice_id 5 --channel CD31 \\
    --threshold 200.0
"""

import os
import sys
import re
import glob
import argparse
import logging

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (keep in sync with phenotype_cells.py)
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
MARKER_CHANNELS  = ['CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK']
CHANNEL_IDX      = {name: i for i, name in enumerate(CHANNEL_NAMES)}
PIXEL_SIZE_XY_UM = 0.4961
DEFAULT_BIC_THRESHOLD = 6.0
PREFER_OTSU = {'CK'}
MANUAL_THRESHOLDS: dict[str, float] = {
    'CD31': 200.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Zoomed phenotype QC: crop around positive cell clusters.'
)
parser.add_argument('--core_name',     type=str, required=True)
parser.add_argument('--slice_id',      type=int, required=True,
                    help='TMA slice ID (the number in the mask filename)')
parser.add_argument('--channel',       type=str, required=True,
                    help='Marker channel: CD31 GAP43 NFP CD3 CD163 CK')
parser.add_argument('--crop_size',     type=int, default=800,
                    help='Side length of the square crop in pixels (default: 800). '
                         'At 0.5 µm/px this is ~400 µm, enough to show ~20–50 cells.')
parser.add_argument('--n_crops',       type=int, default=3,
                    help='Number of positive-cluster crops to save (default: 3). '
                         'Each crop is centred on a different dense cluster.')
parser.add_argument('--crop',          type=int, nargs=4,
                    metavar=('R0', 'R1', 'C0', 'C1'),
                    help='Manual pixel crop (overrides auto-clustering).')
parser.add_argument('--threshold',     type=float, default=None,
                    help='Override the auto-derived threshold.')
parser.add_argument('--min_area_px',   type=int, default=200)
parser.add_argument('--bic_threshold', type=float, default=DEFAULT_BIC_THRESHOLD)
parser.add_argument('--out_dir',       type=str, default=None)
args = parser.parse_args()

TARGET_CORE   = args.core_name
BIC_THRESHOLD = args.bic_threshold

DENOISED_VOL  = os.path.join(config.DATASPACE, 'Denoised', TARGET_CORE,
                              f'{TARGET_CORE}_denoised.ome.tif')
DAPI_MASK_DIR = os.path.join(config.DATASPACE, 'CellPose_DAPI_Warped', TARGET_CORE)
OUT_DIR = args.out_dir or os.path.join(
    config.DATASPACE, 'Phenotypes', TARGET_CORE, 'qc_zoomed'
)
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD HELPERS  (identical to phenotype_cells.py)
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_intersection(gmm):
    from scipy.optimize import brentq
    means   = gmm.means_.flatten()
    sigmas  = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    neg_idx = int(np.argmin(means)); pos_idx = int(np.argmax(means))
    mu_neg, sig_neg, w_neg = means[neg_idx], sigmas[neg_idx], weights[neg_idx]
    mu_pos, sig_pos, w_pos = means[pos_idx], sigmas[pos_idx], weights[pos_idx]
    if (mu_pos - mu_neg) / max(min(sig_neg, sig_pos), 1e-9) < 0.5:
        return None
    def delta_pdf(x):
        return (w_neg * np.exp(-0.5*((x-mu_neg)/sig_neg)**2)/sig_neg
              - w_pos * np.exp(-0.5*((x-mu_pos)/sig_pos)**2)/sig_pos)
    try:
        root = float(brentq(delta_pdf, mu_neg, mu_pos, xtol=1e-6, maxiter=200))
    except (ValueError, RuntimeError):
        root = float((mu_neg + mu_pos) / 2.0)
    return None if root <= mu_neg + 0.5 * sig_neg else root


def threshold_from_cell_medians(cell_medians, marker_name='Marker'):
    from sklearn.mixture import GaussianMixture
    cell_medians = np.asarray(cell_medians, dtype=np.float64)
    if marker_name in MANUAL_THRESHOLDS:
        t = float(MANUAL_THRESHOLDS[marker_name])
        logger.info(f'  [{marker_name}] Manual threshold: {t:.1f}')
        return t, 'manual'
    if len(cell_medians) < 10:
        return float(np.percentile(cell_medians, 75)), 'p75_fallback_too_few_cells'
    if marker_name not in PREFER_OTSU:
        X = cell_medians.reshape(-1, 1)
        try:
            gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
            gmm2 = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(X)
            delta_bic = gmm1.bic(X) - gmm2.bic(X)
        except Exception as exc:
            logger.warning(f'  [{marker_name}] GMM failed: {exc}')
            delta_bic = 0.0; gmm2 = None
        if delta_bic > BIC_THRESHOLD and gmm2 is not None:
            ix = _gmm_intersection(gmm2)
            mu_pos = float(gmm2.means_.max()); mu_neg = float(gmm2.means_.min())
            if ix is not None and ix > max(mu_pos*0.05, (mu_pos-mu_neg)*0.10):
                return ix, 'gmm_bimodal'
    nz = cell_medians[cell_medians > 0]
    if len(nz) >= 10:
        n_bins = min(256, len(nz)//2)
        counts, edges = np.histogram(nz, bins=n_bins)
        centres = (edges[:-1] + edges[1:]) / 2.0
        prob = counts / counts.sum()
        w0 = np.cumsum(prob); mu0s = np.cumsum(prob * centres); w1 = 1.0 - w0
        mu_t = float(np.sum(prob * centres))
        with np.errstate(invalid='ignore', divide='ignore'):
            mu0 = np.where(w0 > 0, mu0s / w0, 0.0)
            mu1 = np.where(w1 > 0, (mu_t - mu0s) / w1, 0.0)
        otsu_val = float(centres[np.argmax(w0 * w1 * (mu0 - mu1)**2)])
        if otsu_val > float(np.percentile(nz, 10)):
            return otsu_val, 'otsu_nz'
    t = float(np.percentile(cell_medians, 75))
    logger.warning(f'  [{marker_name}] p75 fallback ({t:.1f}).')
    return t, 'p75_fallback'


# ─────────────────────────────────────────────────────────────────────────────
# MEASUREMENT  (identical to phenotype_cells.py)
# ─────────────────────────────────────────────────────────────────────────────

def measure_slice(mask, volume_slice, min_area=200):
    labels = np.unique(mask); labels = labels[labels != 0]
    if len(labels) == 0:
        return pd.DataFrame(), {}
    label_list = labels.tolist()
    areas = np.array(ndi.sum(np.ones_like(mask), mask, label_list), dtype=np.int32)
    keep  = areas >= min_area
    label_list = [l for l, k in zip(label_list, keep) if k]
    areas      = areas[keep]
    if not label_list:
        return pd.DataFrame(), {}
    cell_mask = expand_labels(mask, distance=4)
    cy = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[0])[:,None], mask.shape), mask, label_list))
    cx = np.array(ndi.mean(
        np.broadcast_to(np.arange(mask.shape[1])[None,:], mask.shape), mask, label_list))
    df = pd.DataFrame({'cell_id': label_list, 'area_px': areas,
                       'area_um2': np.round(areas * PIXEL_SIZE_XY_UM**2, 3),
                       'centroid_x': np.round(cx, 1), 'centroid_y': np.round(cy, 1)})
    for ch in MARKER_CHANNELS:
        ch_img = volume_slice[CHANNEL_IDX[ch]].astype(np.float32)
        fl = cell_mask.ravel(); fp = ch_img.ravel()
        si = np.argsort(fl, kind='stable')
        sl = fl[si]; sp = fp[si]
        b  = np.searchsorted(sl, label_list)
        e  = np.searchsorted(sl, label_list, side='right')
        meds = np.array([float(np.median(sp[s:v])) if v>s else 0.0
                         for s, v in zip(b, e)], dtype=np.float32)
        df[f'median_{ch}'] = np.round(meds, 2)
        t, tt = threshold_from_cell_medians(meds, marker_name=ch)
        df[f'thresh_{ch}']      = round(t, 2)
        df[f'thresh_type_{ch}'] = tt
        df[f'pos_{ch}']         = (meds >= t).astype(np.uint8)
    ch_imgs = {ch: volume_slice[CHANNEL_IDX[ch]].astype(np.float32) for ch in MARKER_CHANNELS}
    return df, ch_imgs


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-CROP: find dense positive clusters
# ─────────────────────────────────────────────────────────────────────────────

def find_positive_cluster_crops(df_slice, ch_name, thresh_t, H, W,
                                  crop_size, n_crops):
    """
    Divide the image into a grid of tiles and rank tiles by the number of
    positive cells whose centroid falls inside them.  Return the top-n tile
    centres as (r0, r1, c0, c1) tuples, guaranteed non-overlapping.

    Falls back to the image centre if no positive cells exist.
    """
    pos_cells = df_slice[df_slice[f'median_{ch_name}'] >= thresh_t]

    if pos_cells.empty:
        logger.warning(f'No positive cells found for {ch_name} — using image centre.')
        cy, cx = H // 2, W // 2
        return [_safe_crop(cy, cx, H, W, crop_size)]

    # Grid of tiles, each the same size as the crop window.
    # Count positive cells per tile and return top-n by density.
    step   = crop_size // 2          # 50 % overlap between adjacent tiles
    rows   = np.arange(0, H - crop_size + 1, step)
    cols   = np.arange(0, W - crop_size + 1, step)

    cy_arr = pos_cells['centroid_y'].values
    cx_arr = pos_cells['centroid_x'].values

    tile_scores = []
    for r in rows:
        for c in cols:
            in_tile = ((cy_arr >= r) & (cy_arr < r + crop_size) &
                       (cx_arr >= c) & (cx_arr < c + crop_size))
            n = int(in_tile.sum())
            if n > 0:
                tile_scores.append((n, r + crop_size//2, c + crop_size//2))

    if not tile_scores:
        # Positive cells exist but none fall in a full tile (edge case)
        cy = int(pos_cells['centroid_y'].median())
        cx = int(pos_cells['centroid_x'].median())
        return [_safe_crop(cy, cx, H, W, crop_size)]

    tile_scores.sort(key=lambda x: x[0], reverse=True)

    # Pick top-n tiles that don't overlap with already-selected ones
    selected = []
    for n, cy, cx in tile_scores:
        # Check for overlap with already-selected crops
        overlap = False
        for r0s, r1s, c0s, c1s in selected:
            cy_s = (r0s + r1s) // 2; cx_s = (c0s + c1s) // 2
            if abs(cy - cy_s) < crop_size and abs(cx - cx_s) < crop_size:
                overlap = True
                break
        if not overlap:
            selected.append(_safe_crop(cy, cx, H, W, crop_size))
        if len(selected) == n_crops:
            break

    # If we still need more, relax the non-overlap requirement
    if len(selected) < n_crops:
        for n, cy, cx in tile_scores:
            crop = _safe_crop(cy, cx, H, W, crop_size)
            if crop not in selected:
                selected.append(crop)
            if len(selected) == n_crops:
                break

    logger.info(f'  Auto-selected {len(selected)} crop(s) for {ch_name}')
    for i, (r0, r1, c0, c1) in enumerate(selected):
        n_in = int(((cy_arr >= r0) & (cy_arr < r1) &
                    (cx_arr >= c0) & (cx_arr < c1)).sum())
        logger.info(f'    Crop {i+1}: rows {r0}–{r1}, cols {c0}–{c1}  '
                    f'({n_in} positive cells inside)')

    return selected


def _safe_crop(cy, cx, H, W, size):
    """Return (r0, r1, c0, c1) centred at (cy, cx), clamped to image bounds."""
    half = size // 2
    r0 = max(0, cy - half);  r1 = min(H, r0 + size)
    c0 = max(0, cx - half);  c1 = min(W, c0 + size)
    # Push back from the edge if we hit a boundary
    if r1 - r0 < size: r0 = max(0, r1 - size)
    if c1 - c0 < size: c0 = max(0, c1 - size)
    return (r0, r1, c0, c1)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stretch(img):
    lo, hi = np.percentile(img.ravel(), [1, 99])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


def build_label_map(mask, cell_ids, values, dtype=np.float32):
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=dtype)
    cell_ids = np.asarray(cell_ids); values = np.asarray(values)
    valid = cell_ids <= max_label
    lut[cell_ids[valid]] = values[valid]
    return lut[mask]


def _make_overlay(raw_img, mask, cell_ids, cell_meds, threshold):
    gray = _stretch(raw_img)
    rgb  = np.stack([gray, gray, gray], axis=-1).copy()
    if len(cell_ids) == 0:
        return rgb
    pos_map = build_label_map(mask, cell_ids, (cell_meds >= threshold).astype(np.float32))
    neg_px  = (mask > 0) & (pos_map == 0)
    rgb[neg_px, 0] = np.clip(rgb[neg_px, 0] * 0.5 + 0.3, 0, 1)
    rgb[neg_px, 1] = rgb[neg_px, 1] * 0.2
    rgb[neg_px, 2] = rgb[neg_px, 2] * 0.2
    pos_px = pos_map == 1
    rgb[pos_px, 0] = rgb[pos_px, 0] * 0.2
    rgb[pos_px, 1] = np.clip(rgb[pos_px, 1] * 0.4 + 0.5, 0, 1)
    rgb[pos_px, 2] = rgb[pos_px, 2] * 0.2
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def save_zoomed_panel(raw_img_full, mask_full, df_slice, ch_name,
                      slice_id, r0, r1, c0, c1, crop_label, out_path,
                      threshold_override=None):
    """
    4-panel QC plot (2x2 Layout):
      Top-Left     — raw image, cropped
      Top-Right    — positivity overlay, cropped
      Bottom-Left  — cell-median histogram, full slice
      Bottom-Right — pixel histogram, full slice
    """
    n_cells   = len(df_slice)
    thresh_t  = float(df_slice[f'thresh_{ch_name}'].iloc[0]) if n_cells > 0 else 0.0
    t_type    = df_slice[f'thresh_type_{ch_name}'].iloc[0]   if n_cells > 0 else ''
    cell_meds = df_slice[f'median_{ch_name}'].values          if n_cells > 0 else np.array([])

    if threshold_override is not None:
        thresh_t = threshold_override
        t_type   = t_type + '_overridden'

    n_pos = int((cell_meds >= thresh_t).sum()) if n_cells > 0 else 0
    pct   = 100.0 * n_pos / n_cells            if n_cells > 0 else 0.0

    # ── Crop ──────────────────────────────────────────────────────────────────
    raw_crop  = raw_img_full[r0:r1, c0:c1]
    mask_crop = mask_full[r0:r1, c0:c1]

    # Only cells whose centroid is inside the crop window
    if n_cells > 0:
        in_crop = (
            (df_slice['centroid_y'] >= r0) & (df_slice['centroid_y'] < r1) &
            (df_slice['centroid_x'] >= c0) & (df_slice['centroid_x'] < c1)
        )
        df_crop    = df_slice[in_crop]
        meds_crop  = df_crop[f'median_{ch_name}'].values
        ids_crop   = df_crop['cell_id'].values
        n_crop     = len(df_crop)
        n_pos_crop = int((meds_crop >= thresh_t).sum())
        pct_crop   = 100.0 * n_pos_crop / n_crop if n_crop > 0 else 0.0
    else:
        ids_crop = np.array([]); meds_crop = np.array([])
        n_crop = 0; n_pos_crop = 0; pct_crop = 0.0

    overlay_crop = _make_overlay(raw_crop, mask_crop, ids_crop, meds_crop, thresh_t)

    # ── Figure ────────────────────────────────────────────────────────────────
    # 2x2 grid with simplified titles and increased top margin
    fig = plt.figure(figsize=(24, 24))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.20, hspace=0.25,
                            left=0.05, right=0.95, top=0.88, bottom=0.05)

    # Simplified suptitle
    fig.suptitle(
        f'{TARGET_CORE}  |  Slice {slice_id:03d}  |  {ch_name}  |  {crop_label}\n'
        f'Threshold: {thresh_t:.1f} [{t_type}]  |  Crop Positivity: {pct_crop:.0f}%  |  Slice Positivity: {pct:.0f}%',
        fontsize=36, fontweight='bold', y=0.96,
    )

    # Panel 1 (Top-Left) — raw (cropped)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(_stretch(raw_crop), cmap='gray', interpolation='nearest')
    ax.set_title(f'Raw Image (Y[{r0}:{r1}], X[{c0}:{c1}])', fontsize=30, pad=15)
    ax.axis('off')

    # Panel 2 (Top-Right) — positivity overlay (cropped)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(overlay_crop, interpolation='nearest')
    ax.set_title('Positivity Overlay', fontsize=30, pad=15)
    ax.axis('off')

    # Panel 3 (Bottom-Left) — cell-median histogram (full slice)
    ax = fig.add_subplot(gs[1, 0])
    if n_cells > 0 and len(cell_meds) > 0:
        ax.hist(cell_meds, bins=100, color='#e67e22', alpha=0.80,
                edgecolor='none', log=True)
        col = 'magenta' if 'manual' in t_type else 'red'
        ax.axvline(thresh_t, color=col, linewidth=3.5,
                   label=f'threshold = {thresh_t:.1f}\n[{t_type}]')
        ax.legend(fontsize=26)
    ax.set_xlabel('Per-cell median intensity', fontsize=28)
    ax.set_ylabel('Cell count (log scale)', fontsize=28)
    ax.set_title('Per-cell Median Distribution', fontsize=30, pad=15)
    ax.tick_params(labelsize=26)

    # Panel 4 (Bottom-Right) — pixel histogram (full slice)
    ax = fig.add_subplot(gs[1, 1])
    flat = raw_img_full.ravel().astype(np.float32); flat = flat[flat > 0]
    if len(flat) > 0:
        counts, edges = np.histogram(flat, bins=512)
        centres = (edges[:-1] + edges[1:]) / 2.0
        ax.bar(centres, counts, width=centres[1]-centres[0],
               color='steelblue', alpha=0.75, edgecolor='none')
        ax.set_yscale('log')
        ax.axvline(thresh_t, color='red', linewidth=3.0, linestyle='--',
                   label=f'threshold = {thresh_t:.1f}')
        ax.legend(fontsize=26)
    ax.set_xlabel('Pixel intensity', fontsize=28)
    ax.set_ylabel('Pixel count (log scale)', fontsize=28)
    ax.set_title('Full-image Pixel Distribution', fontsize=30, pad=15)
    ax.tick_params(labelsize=26)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'  Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_id_from_path(path):
    m = re.search(r'TMA_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ch_name = args.channel
    if ch_name not in MARKER_CHANNELS:
        logger.error(f'Channel {ch_name!r} not in {MARKER_CHANNELS}'); sys.exit(1)

    logger.info(f'inspect_phenotype_zoom.py  core={TARGET_CORE}  '
                f'slice_id={args.slice_id}  channel={ch_name}  '
                f'crop_size={args.crop_size}  n_crops={args.n_crops}')

    if not os.path.exists(DENOISED_VOL):
        logger.error(f'Not found: {DENOISED_VOL}'); sys.exit(1)
    if not os.path.isdir(DAPI_MASK_DIR):
        logger.error(f'DAPI mask dir not found: {DAPI_MASK_DIR}'); sys.exit(1)

    mask_files = glob.glob(os.path.join(DAPI_MASK_DIR, '*_DAPI_cp_masks_warped.tif'))
    mask_map   = {get_slice_id_from_path(p): p for p in mask_files}
    if args.slice_id not in mask_map:
        logger.error(f'No mask for slice_id={args.slice_id}. '
                     f'Available: {sorted(mask_map.keys())}'); sys.exit(1)

    reg_stats_csv = os.path.join(
        config.DATASPACE, 'Filter_AKAZE_RoMaV2_Linear_Warp_map',
        TARGET_CORE, 'registration_stats_AKAZE_RoMaV2_Linear.csv')
    z_idx = None
    if os.path.exists(reg_stats_csv):
        reg_df  = pd.read_csv(reg_stats_csv)
        mapping = dict(zip(reg_df['Slice_ID'].astype(int), reg_df['Slice_Z'].astype(int)))
        z_idx   = mapping.get(args.slice_id)
    if z_idx is None:
        z_idx = sorted(mask_map.keys()).index(args.slice_id)
        logger.warning(f'slice_id {args.slice_id} not in CSV — using sorted position z={z_idx}')
    logger.info(f'slice_id={args.slice_id} → z_idx={z_idx}')

    logger.info('Loading denoised volume ...')
    vol = tifffile.imread(DENOISED_VOL)
    if vol.ndim == 4:
        if vol.shape[0] == len(CHANNEL_NAMES) and vol.shape[1] != len(CHANNEL_NAMES):
            vol = np.moveaxis(vol, 0, 1)
    elif vol.ndim == 3:
        vol = vol[np.newaxis]
    n_slices, n_channels, H, W = vol.shape
    logger.info(f'Volume: Z={n_slices}  C={n_channels}  H={H}  W={W}')

    if z_idx >= n_slices:
        logger.error(f'z_idx={z_idx} out of range'); sys.exit(1)

    logger.info('Loading DAPI mask ...')
    mask = tifffile.imread(mask_map[args.slice_id]).astype(np.uint32)
    if mask.ndim != 2: mask = mask.squeeze()
    if mask.shape != (H, W):
        logger.error(f'Mask {mask.shape} != volume ({H},{W})'); sys.exit(1)

    logger.info('Measuring cells ...')
    df_slice, ch_imgs = measure_slice(mask, vol[z_idx], min_area=args.min_area_px)
    if df_slice.empty:
        logger.error('No cells found.'); sys.exit(1)
    logger.info(f'  {len(df_slice)} cells measured.')

    raw_img_full = ch_imgs[ch_name]

    # Apply threshold override before crop selection so the cluster finder
    # uses the same threshold that will appear in the plot
    thresh_t = float(df_slice[f'thresh_{ch_name}'].iloc[0])
    if args.threshold is not None:
        thresh_t = args.threshold
        df_slice[f'thresh_{ch_name}']      = thresh_t
        df_slice[f'thresh_type_{ch_name}'] = df_slice[f'thresh_type_{ch_name}'] + '_overridden'

    # ── Determine crops ───────────────────────────────────────────────────────
    if args.crop:
        r0, r1, c0, c1 = args.crop
        crops = [(r0, r1, c0, c1)]
        labels = [f'manual_crop']
    else:
        crops  = find_positive_cluster_crops(
            df_slice, ch_name, thresh_t, H, W, args.crop_size, args.n_crops)
        labels = [f'cluster_{i+1}_of_{len(crops)}' for i in range(len(crops))]

    for (r0, r1, c0, c1), label in zip(crops, labels):
        out_png = os.path.join(
            OUT_DIR,
            f'{TARGET_CORE}_Slice{args.slice_id:03d}_{ch_name}_{label}.png'
        )
        save_zoomed_panel(
            raw_img_full=raw_img_full, mask_full=mask,
            df_slice=df_slice, ch_name=ch_name,
            slice_id=args.slice_id,
            r0=r0, r1=r1, c0=c0, c1=c1,
            crop_label=label, out_path=out_png,
            threshold_override=args.threshold,
        )

    logger.info(f'Done. Output in: {OUT_DIR}')


if __name__ == '__main__':
    main()