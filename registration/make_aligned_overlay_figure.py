"""
make_aligned_overlay_figure.py
===============================
Builds one figure showing the SAME cropped region across DAPI, CK, CD163,
and CD31, addressing the reviewer's request to compare overlays directly.

Row 0 : DAPI (raw + segmentation outline)
Row 1 : CK    (raw + positivity overlay)
Row 2 : CD163 (raw + positivity overlay)
Row 3 : CD31  (raw + positivity overlay)

All rows use the SAME crop box (y0:y1, x0:x1), so a structure visible in
DAPI/segmentation can be visually traced to its positivity call in each
marker channel.

Usage
-----
    python make_aligned_overlay_figure.py \
        --core_name Core_01 --slice_id 5 \
        --y0 1500 --y1 2000 --x0 1500 --x1 2000

Adjust --y0/--y1/--x0/--x1 to match the crop already used in your existing
Figures 14-16 (CK/CD163/CD31), so the new figure is directly comparable.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import expand_labels, find_boundaries

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
import config  # same config module used by phenotype_cells.py

CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
CHANNEL_IDX   = {name: i for i, name in enumerate(CHANNEL_NAMES)}
ROW_CHANNELS  = ['CK', 'CD163', 'CD31']  # marker rows below DAPI


def _stretch(img: np.ndarray) -> np.ndarray:
    flat = img.ravel()
    lo, hi = np.percentile(flat, [1, 99])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


def build_label_map(mask, cell_ids, values, dtype=np.float32):
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=dtype)
    cell_ids = np.asarray(cell_ids)
    values = np.asarray(values)
    valid = cell_ids <= max_label
    lut[cell_ids[valid]] = values[valid]
    return lut[mask]


def make_overlay(raw_img, mask, cell_ids, cell_meds, threshold):
    """Same tinting logic as phenotype_cells.py: green = positive, dark red = negative."""
    gray = _stretch(raw_img)
    rgb = np.stack([gray, gray, gray], axis=-1).copy()
    if len(cell_ids) == 0:
        return rgb
    pos_arr = (cell_meds >= threshold).astype(np.float32)
    pos_map = build_label_map(mask, cell_ids, pos_arr)
    neg_px = (mask > 0) & (pos_map == 0)
    rgb[neg_px, 0] = np.clip(rgb[neg_px, 0] * 0.5 + 0.3, 0, 1)
    rgb[neg_px, 1] = rgb[neg_px, 1] * 0.2
    rgb[neg_px, 2] = rgb[neg_px, 2] * 0.2
    pos_px = pos_map == 1
    rgb[pos_px, 0] = rgb[pos_px, 0] * 0.2
    rgb[pos_px, 1] = np.clip(rgb[pos_px, 1] * 0.4 + 0.5, 0, 1)
    rgb[pos_px, 2] = rgb[pos_px, 2] * 0.2
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--core_name', required=True)
    ap.add_argument('--slice_id', type=int, required=True)
    ap.add_argument('--y0', type=int, required=True)
    ap.add_argument('--y1', type=int, required=True)
    ap.add_argument('--x0', type=int, required=True)
    ap.add_argument('--x1', type=int, required=True)
    ap.add_argument('--outdir', default='aligned_panels')
    args = ap.parse_args()

    core = args.core_name
    y0, y1, x0, x1 = args.y0, args.y1, args.x0, args.x1

    # ── Load denoised volume ──────────────────────────────────────────────
    vol_path = os.path.join(config.DATASPACE, 'Denoised_bspline', core,
                             f'{core}_denoised.ome.tif')
    vol = tifffile.imread(vol_path)
    if vol.ndim == 4 and vol.shape[0] == len(CHANNEL_NAMES) and vol.shape[1] != len(CHANNEL_NAMES):
        vol = np.moveaxis(vol, 0, 1)

    # ── Load DAPI mask + phenotype table for the same slice ──────────────
    mask_dir = os.path.join(config.DATASPACE, 'CellPose_DAPI_Warped_Bspline', core)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_DAPI_cp_masks_warped.tif')]
    # crude match — adjust if your slice/Z indexing differs
    z_idx = args.slice_id
    mask_path = sorted(mask_files)[z_idx]
    mask = tifffile.imread(os.path.join(mask_dir, mask_path)).astype(np.uint32)

    pheno_csv = os.path.join(config.DATASPACE, 'Phenotypes_Bspline', core,
                              f'{core}_phenotypes.csv')
    df = pd.read_csv(pheno_csv)
    df_slice = df[df['slice_z'] == z_idx]

    # ── Crop everything to the same window ────────────────────────────────
    dapi_img = vol[z_idx][CHANNEL_IDX['DAPI']][y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    boundaries = find_boundaries(mask_crop, mode='outer')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    tag = f'{core}_slice{args.slice_id:03d}_y{y0}-{y1}_x{x0}-{x1}'

    dapi_gray = _stretch(dapi_img)
    dapi_rgb = np.stack([dapi_gray] * 3, axis=-1)
    dapi_rgb[boundaries] = [1, 1, 0]  # yellow outline = detected cells

    # ── Figures: one per marker channel, 2x2 layout ───────────────────────
    # (0,0) raw  |  (0,1) cell-median histogram
    # (1,0) DAPI + segmentation outline  |  (1,1) positivity overlay
    for ch in ROW_CHANNELS:
        ch_img = vol[z_idx][CHANNEL_IDX[ch]][y0:y1, x0:x1]
        thresh_t = float(df_slice[f'thresh_{ch}'].iloc[0]) if len(df_slice) else 0.0
        t_type = df_slice[f'thresh_type_{ch}'].iloc[0] if len(df_slice) else ''
        cell_ids = df_slice['cell_id'].values
        cell_meds = df_slice[f'median_{ch}'].values
        n_cells = len(df_slice)
        n_pos = int((cell_meds >= thresh_t).sum()) if n_cells else 0
        pct = 100.0 * n_pos / n_cells if n_cells else 0.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 13))

        # (0,0) raw
        axes[0, 0].imshow(_stretch(ch_img), cmap='gray', interpolation='nearest')
        axes[0, 0].set_title(f'{ch}\nBackground-corrected', fontsize=21)
        axes[0, 0].axis('off')

        # (0,1) cell-median histogram
        if n_cells > 0 and len(cell_meds) > 0:
            axes[0, 1].hist(cell_meds, bins=80, color='#e67e22',
                            alpha=0.80, edgecolor='none', log=True)
            line_col = 'magenta' if 'manual' in t_type else 'red'
            axes[0, 1].axvline(thresh_t, color=line_col, linewidth=2.0,
                               label=f'threshold = {thresh_t:.1f}\n[{t_type}]')
            axes[0, 1].legend(fontsize=20)
        axes[0, 1].set_xlabel('Per-cell median intensity', fontsize=20)
        axes[0, 1].set_ylabel('Cell count (log scale)', fontsize=20)
        axes[0, 1].set_title('Per-cell median distribution', fontsize=21)

        # (1,0) DAPI + segmentation outline
        axes[1, 0].imshow(dapi_rgb, interpolation='nearest')
        axes[1, 0].set_title('DAPI + CellPose segmentation outline', fontsize=21)
        axes[1, 0].axis('off')

        # (1,1) positivity overlay
        rgb = make_overlay(ch_img, mask_crop, cell_ids, cell_meds, thresh_t)
        axes[1, 1].imshow(rgb, interpolation='nearest')
        axes[1, 1].set_title(f'Positivity overlay\n{n_pos}/{n_cells} positive ({pct:.0f}%)',
                             fontsize=21)
        axes[1, 1].axis('off')

        fig.suptitle(f'{ch}: threshold = {thresh_t:.1f} [{t_type}]', fontsize=23, y=1.0)
        fig.tight_layout()
        out_path = os.path.join(args.outdir, f'{tag}_{ch}_figure.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    print(f'\nAll figures saved to: {args.outdir}')
    print('3 files total — CK_figure, CD163_figure, CD31_figure — each a '
          '2x2 grid (raw, cell-median histogram, DAPI+segmentation, overlay), '
          'all using the same crop window.')


if __name__ == '__main__':
    main()