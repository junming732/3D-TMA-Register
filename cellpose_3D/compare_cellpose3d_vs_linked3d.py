"""Compares true 3D Cellpose reconstructions against 2D-linked reconstructions.

Evaluates two independently produced 3D cell segmentations (Pipeline A: native 3D 
Cellpose vs. Pipeline B: 2D Cellpose with overlap-linking) using existing disk assets. 
Derives a statistics table from Pipeline A's mask volume to match Pipeline B's format, 
enabling distribution-level and cell-by-cell centroid matching. Outputs include 
comparative distributions, matched pair statistics, and an interactive 3D HTML report.

Example:
    $ python compare_cellpose3d_vs_linked3d.py \
        --core-name Core_19 \
        --cellpose3d-mask /data3/.../Core_19/Core_19_3D_Cellpose_Masks.tif \
        --linked3d-stats /data3/.../Core_19/Core_19_DAPI_3d_stats.csv \
        --linked3d-mask /data3/.../Core_19/Core_19_DAPI_3d_labels.tif \
        --dapi-path /data3/.../Core_19/Core_19_denoised.ome.tif \
        --xy-pixel-um 0.4961 --z-thickness-um 4.0 \
        --out-dir ./compare_Core_19

Notes:
    - Cell-level matching requires both label volumes to share the exact same 
      registered coordinate frame. If registration variants differ, use 
      `--skip-matching` to rely solely on distribution-level comparisons.
    - Physical constants (`--z-thickness-um` and `--xy-pixel-um`) must be passed 
      explicitly to ensure accurate volume and aspect-ratio comparisons.
    - The `--dapi-path` argument is optional; if provided, it generates per-cell 
      DAPI + segmentation montages.

Outputs:
    Files are generated in the specified `--out-dir`:
    - `<core>_cellpose3d_stats.csv`: Pipeline A per-cell stats (includes `has_z_gap` 
      and `below_min_volume` flags).
    - `<core>_matched_pairs.csv`: Cell-to-cell matches with drift and volume ratios.
    - `<core>_volume_hist.png`: Overlaid volume distributions.
    - `<core>_aspect_ratio_hist.png`: Overlaid aspect-ratio distributions.
    - `<core>_matched_volume_scatter.png`: Linked3D vs. Cellpose3D matched volumes.
    - `<core>_centroid_drift_hist.png`: Matched-pair centroid drift distribution.
    - `<core>_summary.csv` / `.png`: Headline comparison metrics.
    - `<core>_3d_qc.html`: Interactive shared-crop 3D mesh report.
    - `visual_tile_qc_cellpose3d/cell_XXXXXX_dapi_seg.png`: Per-cell montages across 
      spanned Z-slices (yellow = all boundaries, green = target cell). Generated 
      only if `--dapi-path` is provided.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tifffile
import zarr
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.segmentation import find_boundaries

# --- 3D mesh QC panel settings (mirrors test_3d_cellpose.py's render_qc_html) ---
QC_N_SAMPLES_PER_PIPELINE = 15
QC_MIN_VOXELS = 20
QC_PAD_VOXELS = 2

# --- DAPI+segmentation tile montage settings (mirrors link_3d_cells.py's
# visual_tile_qc_aligned panels, but reads straight off the native 3D
# Cellpose label volume instead of a linked chain of 2D masks) ---
DAPI_SEG_N_SAMPLES = 20
DAPI_SEG_PATCH_PX = 80

# --- minimum-plausible-nucleus volume floor ---
# Real mammalian nuclei run roughly 5-20 um diameter depending on cell type
# (small lymphocytes at the low end, hepatocytes/large secretory cells at
# the high end). This default is set well below even the smallest of those
# (a 5 um-diameter sphere is ~65 um3) specifically so it only strips
# obvious segmentation debris/fragments - a handful of stray voxels do_3D
# occasionally emits at tile/chunk seams or from over-eager flow-splitting -
# not small-but-real nuclei. It is a floor against noise, not a biological
# size filter; tune --min-nucleus-volume-um3 for your tissue if needed.
MIN_NUCLEUS_VOLUME_UM3 = 65.0


# -----------------------------------------------------------------------------
# Stats extraction for the true-3D Cellpose mask volume
# -----------------------------------------------------------------------------

class MultiSeriesZStack:
    """Present a possibly-multi-series label TIFF as one seamless (Z, Y, X)
    array, indexable like a lazy zarr array (single z, or a z-slice, plus
    optional trailing Y/X indexing).

    test_3d_cellpose.py writes one Z-chunk per `writer.write()` call inside a
    single TiffWriter context. tifffile can only append a write as a page of
    the PREVIOUS series if it has the same leading (Z) length; the moment a
    chunk's length differs (e.g. a final, uneven chunk), tifffile silently
    starts a new series instead. So a volume written in N Z-chunks of
    unequal size ends up as N separate series in one file, and naive
    `tifffile.imread(path, aszarr=True)` (or `zarr.open` on that store) only
    exposes series 0 - i.e. only the FIRST chunk's planes. This class reads
    the file's series list once, and routes any Z index/slice to the right
    underlying series, so the rest of this script never needs to know the
    file is fragmented like this.
    """

    def __init__(self, path: str):
        tf = tifffile.TiffFile(path)
        self._arrays = [zarr.open(s.aszarr(), mode="r") for s in tf.series]
        shapes = [tuple(a.shape) for a in self._arrays]
        yx_shapes = {shp[-2:] for shp in shapes}
        if len(yx_shapes) != 1:
            raise ValueError(f"{path}: series have mismatched Y/X shapes: {shapes}")
        y, x = next(iter(yx_shapes))
        self._z_sizes = [shp[0] for shp in shapes]
        self._offsets = np.cumsum([0] + self._z_sizes)
        self.shape = (int(self._offsets[-1]), y, x)
        if len(self._arrays) > 1:
            print(f"  NOTE: {path} is split into {len(self._arrays)} separate "
                  f"TIFF series (Z sizes {self._z_sizes}) because "
                  f"test_3d_cellpose.py's Z-chunks weren't all the same "
                  f"length - stitching them into one Z={self.shape[0]} stack "
                  f"here so nothing downstream sees the fragmentation.")

    def __getitem__(self, key):
        zkey, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(zkey, slice):
            z0, z1, step = zkey.indices(self.shape[0])
            if step != 1:
                raise ValueError("Only contiguous (step=1) Z-slices are supported")
            planes = [self[(z,) + rest] if rest else self[z] for z in range(z0, z1)]
            return np.stack(planes, axis=0)
        z = zkey if zkey >= 0 else zkey + self.shape[0]
        si = int(np.searchsorted(self._offsets, z, side="right") - 1)
        local_z = z - self._offsets[si]
        arr = self._arrays[si][local_z]
        return arr[rest] if rest else arr


def open_lazy_labels(path: str):
    """Lazy (Z, Y, X) view of a label TIFF, correctly stitched across
    Z-chunks even if test_3d_cellpose.py wrote them as separate series.
    """
    return MultiSeriesZStack(path)


def open_dapi_lazy(path: str, channel_idx: int = 0):
    """Lazy (Z, Y, X) view of the raw DAPI volume that fed test_3d_cellpose.py.
    Mirrors that script's own open_dapi_lazy: handles a plain 3D (Z,Y,X) stack
    or a 4D (Z,C,Y,X) stack, in which case `channel_idx` picks out DAPI.
    """
    store = tifffile.imread(path, aszarr=True)
    z = zarr.open(store, mode="r")
    if z.ndim == 4:
        return z[:, channel_idx, :, :]
    elif z.ndim == 3:
        return z
    else:
        raise ValueError(f"Expected 3D or 4D array in {path}, got shape {z.shape}")


def extract_stats_from_mask_volume(mask_path: str, xy_pixel_um: float,
                                    z_thickness_um: float) -> pd.DataFrame:
    """Derive a per-cell stats table from a 3D label volume, matching the
    column names/units link_3d_cells.py writes to *_3d_stats.csv. Single
    pass over Z-planes, all per-plane work is vectorized bincount - no
    per-label Python loop, so this scales to millions of labels the same
    way link_3d_cells.py's own 2D centroid computation does.
    """
    vol = open_lazy_labels(mask_path)
    n_z, n_y, n_x = vol.shape
    print(f"  Volume shape: Z={n_z}, Y={n_y}, X={n_x}")

    # Pass 1: global max label id (needed to size the accumulator arrays once).
    max_label = 0
    for z in range(n_z):
        plane_max = int(np.asarray(vol[z]).max())
        max_label = max(max_label, plane_max)
    print(f"  Max label id: {max_label:,}")

    count_total = np.zeros(max_label + 1, dtype=np.int64)
    sum_y = np.zeros(max_label + 1, dtype=np.float64)
    sum_x = np.zeros(max_label + 1, dtype=np.float64)
    sum_z = np.zeros(max_label + 1, dtype=np.float64)
    z_min = np.full(max_label + 1, np.iinfo(np.int32).max, dtype=np.int64)
    z_max = np.full(max_label + 1, -1, dtype=np.int64)
    # Number of Z-PLANES a label actually has >=1 voxel on - distinct from
    # z_span_slices (= z_max - z_min + 1), which only looks at the endpoints.
    # A label present at Z11, Z12, Z14 but with zero voxels at Z13 gets
    # z_span_slices=4 but z_present_count=3: a real gap inside do_3D's
    # reconstruction, not visible from z_min/z_max alone.
    z_present_count = np.zeros(max_label + 1, dtype=np.int64)

    yy, xx = np.indices((n_y, n_x))
    yy_f = yy.ravel().astype(np.float64)
    xx_f = xx.ravel().astype(np.float64)
    del yy, xx

    # Pass 2: accumulate per-label sums, one Z-plane at a time.
    for z in range(n_z):
        plane = np.asarray(vol[z])
        flat = plane.ravel().astype(np.int64)

        c = np.bincount(flat, minlength=max_label + 1)
        count_total += c

        present = np.nonzero(c)[0]
        present = present[present > 0]
        if present.size:
            z_min[present] = np.minimum(z_min[present], z)
            z_max[present] = np.maximum(z_max[present], z)
            z_present_count[present] += 1

        sum_y += np.bincount(flat, weights=yy_f, minlength=max_label + 1)
        sum_x += np.bincount(flat, weights=xx_f, minlength=max_label + 1)
        sum_z += c.astype(np.float64) * z

        if (z + 1) % 5 == 0 or z == n_z - 1:
            print(f"    plane {z + 1}/{n_z} accumulated")

    labels = np.nonzero(count_total)[0]
    labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError(f"No labeled voxels found in {mask_path}")

    volume_px = count_total[labels]
    z_span_slices = (z_max[labels] - z_min[labels] + 1)
    z_present_slices = z_present_count[labels]
    n_missing_z_planes = z_span_slices - z_present_slices
    has_z_gap = n_missing_z_planes > 0
    centroid_y_px = sum_y[labels] / count_total[labels]
    centroid_x_px = sum_x[labels] / count_total[labels]
    centroid_z_idx = sum_z[labels] / count_total[labels]

    volume_um3 = volume_px * (xy_pixel_um ** 2) * z_thickness_um
    centroid_y_um = centroid_y_px * xy_pixel_um
    centroid_x_um = centroid_x_px * xy_pixel_um
    centroid_z_um = centroid_z_idx * z_thickness_um

    # Average cross-sectional area per Z-plane the cell appears in, used the
    # same way link_3d_cells.py uses it: as an isotropy/aspect-ratio sanity
    # check, not a precise shape measurement. Divide by z_present_slices
    # (planes that actually have voxels), not z_span_slices - a label with a
    # mid-span gap (see has_z_gap above) would otherwise get its area diluted
    # by an empty plane it doesn't really occupy.
    mean_area_px = volume_px / np.maximum(z_present_slices, 1)
    mean_diameter_um = 2.0 * np.sqrt(mean_area_px / np.pi) * xy_pixel_um
    z_extent_um = z_span_slices * z_thickness_um
    aspect_ratio = np.where(mean_diameter_um > 0, z_extent_um / mean_diameter_um, 0.0)

    df = pd.DataFrame({
        "cell_id_3d": labels,
        "z_min": z_min[labels],
        "z_max": z_max[labels],
        "z_span_slices": z_span_slices,
        "z_present_slices": z_present_slices,
        "n_missing_z_planes": n_missing_z_planes,
        "has_z_gap": has_z_gap,
        "volume_px": volume_px,
        "volume_um3": np.round(volume_um3, 3),
        "centroid_x_px": np.round(centroid_x_px, 2),
        "centroid_y_px": np.round(centroid_y_px, 2),
        "centroid_z_idx": np.round(centroid_z_idx, 2),
        "centroid_x_um": np.round(centroid_x_um, 3),
        "centroid_y_um": np.round(centroid_y_um, 3),
        "centroid_z_um": np.round(centroid_z_um, 3),
        "mean_diameter_um": np.round(mean_diameter_um, 3),
        "aspect_ratio": np.round(aspect_ratio, 3),
    })
    df = df.sort_values("cell_id_3d").reset_index(drop=True)

    n_gapped = int(df["has_z_gap"].sum())
    multi_slice = df["z_span_slices"] > 1
    n_multi_slice = int(multi_slice.sum())
    if n_multi_slice:
        print(f"  {n_gapped:,} / {n_multi_slice:,} multi-slice labels "
              f"({n_gapped / n_multi_slice:.1%}) have a mid-span Z-gap "
              f"(z_present_slices < z_span_slices) - see has_z_gap / "
              f"n_missing_z_planes columns.")
    return df


# -----------------------------------------------------------------------------
# Cell-to-cell matching by 3D centroid (mutual nearest neighbor)
# -----------------------------------------------------------------------------

def match_by_centroid(df_a: pd.DataFrame, df_b: pd.DataFrame,
                       max_dist_um: float) -> pd.DataFrame:
    """Mutual-nearest-neighbor match between two centroid sets, thresholded
    by max_dist_um. Returns one row per accepted match with both cells'
    stats side by side plus derived drift/volume-ratio columns.
    """
    coords_a = df_a[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].values
    coords_b = df_b[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].values

    tree_b = cKDTree(coords_b)
    tree_a = cKDTree(coords_a)

    dist_ab, idx_ab = tree_b.query(coords_a, k=1)
    dist_ba, idx_ba = tree_a.query(coords_b, k=1)

    rows = []
    for i_a, (d, j_b) in enumerate(zip(dist_ab, idx_ab)):
        if d > max_dist_um:
            continue
        # mutual: b's nearest neighbor must point back to this a
        if idx_ba[j_b] != i_a:
            continue
        row_a = df_a.iloc[i_a]
        row_b = df_b.iloc[j_b]
        rows.append({
            "cellpose3d_id": row_a["cell_id_3d"],
            "linked3d_id": row_b["cell_id_3d"],
            "centroid_drift_um": float(d),
            "cellpose3d_volume_um3": row_a["volume_um3"],
            "linked3d_volume_um3": row_b["volume_um3"],
            "volume_ratio_cp3d_over_linked": (
                row_a["volume_um3"] / row_b["volume_um3"] if row_b["volume_um3"] > 0 else np.nan
            ),
            "cellpose3d_aspect_ratio": row_a["aspect_ratio"],
            "linked3d_aspect_ratio": row_b["aspect_ratio"] if "aspect_ratio" in df_b.columns else np.nan,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Shared-crop 3D mesh QC panel - reads only a small crop from each mask TIFF,
# no re-segmentation. Both pipelines' nuclei are meshed into the SAME scene
# so you can see where they agree/disagree spatially.
# -----------------------------------------------------------------------------

def load_mask_crop(mask_path: str, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    try:
        vol = open_lazy_labels(mask_path)
        return np.asarray(vol[z0:z1, y0:y1, x0:x1])
    except Exception as e:
        print(f"  (zarr lazy-load unavailable ({type(e).__name__}: {e}), falling back to page reads)")
        pages = np.asarray(tifffile.imread(mask_path, key=range(z0, z1)))
        return pages[:, y0:y1, x0:x1]


def _mesh_traces_for_crop(label_vol: np.ndarray, color: str, name_prefix: str,
                           xy_pixel_um: float, z_thickness_um: float,
                           n_samples: int, seed: int) -> list:
    ids, counts = np.unique(label_vol, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    ids = ids[counts >= QC_MIN_VOXELS]
    if len(ids) == 0:
        return []

    random.seed(seed)
    selected = sorted(random.sample(ids.tolist(), min(n_samples, len(ids))))

    traces = []
    for cid in selected:
        cell_mask = label_vol == cid
        zs, ys, xs = np.where(cell_mask)
        zc0, zc1 = max(zs.min() - QC_PAD_VOXELS, 0), min(zs.max() + QC_PAD_VOXELS + 1, label_vol.shape[0])
        yc0, yc1 = max(ys.min() - QC_PAD_VOXELS, 0), min(ys.max() + QC_PAD_VOXELS + 1, label_vol.shape[1])
        xc0, xc1 = max(xs.min() - QC_PAD_VOXELS, 0), min(xs.max() + QC_PAD_VOXELS + 1, label_vol.shape[2])
        sub = cell_mask[zc0:zc1, yc0:yc1, xc0:xc1]
        if sub.shape[0] < 2:
            continue
        try:
            verts, faces, _, _ = marching_cubes(
                sub.astype(np.float32), level=0.5,
                spacing=(z_thickness_um, xy_pixel_um, xy_pixel_um),
            )
        except (ValueError, RuntimeError):
            continue
        offset = np.array([zc0, yc0, xc0]) * np.array([z_thickness_um, xy_pixel_um, xy_pixel_um])
        verts_um = verts + offset
        traces.append(go.Mesh3d(
            x=verts_um[:, 2], y=verts_um[:, 1], z=verts_um[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.55, name=f"{name_prefix} {int(cid)}",
            showlegend=False,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2, roughness=0.6),
        ))
    return traces


# -----------------------------------------------------------------------------
# Static PNG plots (volume/aspect histograms, matched-pair scatter, drift, summary)
# -----------------------------------------------------------------------------

def save_static_plots(core_name: str, df_cp3d: pd.DataFrame, df_linked: pd.DataFrame,
                       df_matched: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

    # --- volume histogram ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df_cp3d["volume_um3"], bins=60, alpha=0.6, color="royalblue", label="Cellpose3D")
    ax.hist(df_linked["volume_um3"], bins=60, alpha=0.6, color="darkorange", label="Linked3D")
    ax.set_xlabel("volume (um3)"); ax.set_ylabel("count")
    ax.set_title(f"{core_name}: volume distribution")
    ax.legend()
    fig.savefig(out_dir / f"{core_name}_volume_hist.png"); plt.close(fig)

    # --- aspect ratio histogram (Linked3D skipped if column absent) ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df_cp3d["aspect_ratio"], bins=40, alpha=0.6, color="royalblue", label="Cellpose3D")
    if "aspect_ratio" in df_linked.columns:
        ax.hist(df_linked["aspect_ratio"], bins=40, alpha=0.6, color="darkorange", label="Linked3D")
    ax.set_xlabel("aspect ratio (z-extent / diameter)"); ax.set_ylabel("count")
    ax.set_title(f"{core_name}: aspect ratio distribution")
    ax.legend()
    fig.savefig(out_dir / f"{core_name}_aspect_ratio_hist.png"); plt.close(fig)

    # --- matched-pair volume scatter ---
    if len(df_matched):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(df_matched["linked3d_volume_um3"], df_matched["cellpose3d_volume_um3"],
                   s=6, alpha=0.35, color="seagreen")
        max_v = float(max(df_matched["cellpose3d_volume_um3"].max(),
                           df_matched["linked3d_volume_um3"].max()))
        ax.plot([0, max_v], [0, max_v], "--", color="gray", label="y = x")
        ax.set_xlabel("Linked3D volume (um3)"); ax.set_ylabel("Cellpose3D volume (um3)")
        ax.set_title(f"{core_name}: matched-pair volume ({len(df_matched):,} pairs)")
        ax.legend()
        fig.savefig(out_dir / f"{core_name}_matched_volume_scatter.png"); plt.close(fig)

        # --- centroid drift histogram ---
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hist(df_matched["centroid_drift_um"], bins=40, color="seagreen")
        ax.set_xlabel("centroid drift (um)"); ax.set_ylabel("count")
        ax.set_title(f"{core_name}: matched-pair centroid drift")
        fig.savefig(out_dir / f"{core_name}_centroid_drift_hist.png"); plt.close(fig)

    # --- summary table as CSV (and a quick PNG render for at-a-glance viewing) ---
    match_rate_cp3d = len(df_matched) / len(df_cp3d) if len(df_cp3d) else 0.0
    match_rate_linked = len(df_matched) / len(df_linked) if len(df_linked) else 0.0
    summary_rows = [
        ["Cellpose3D cell count", f"{len(df_cp3d):,}"],
        ["Linked3D cell count", f"{len(df_linked):,}"],
        ["Matched pairs", f"{len(df_matched):,}"],
        ["Match rate (of Cellpose3D)", f"{match_rate_cp3d:.1%}"],
        ["Match rate (of Linked3D)", f"{match_rate_linked:.1%}"],
        ["Median centroid drift (um)", f"{df_matched['centroid_drift_um'].median():.2f}" if len(df_matched) else "n/a"],
        ["Median volume ratio (CP3D/Linked)", f"{df_matched['volume_ratio_cp3d_over_linked'].median():.2f}" if len(df_matched) else "n/a"],
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_df.to_csv(out_dir / f"{core_name}_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")
    table = ax.table(cellText=summary_rows, colLabels=["Metric", "Value"],
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    ax.set_title(f"{core_name}: summary", pad=14)
    fig.savefig(out_dir / f"{core_name}_summary.png"); plt.close(fig)

    print(f"  Static PNGs + summary.csv saved -> {out_dir}")


# -----------------------------------------------------------------------------
# DAPI+segmentation tile montages for the true-3D Cellpose pipeline.
#
# Same visual language as link_3d_cells.py's visual_tile_qc_aligned montages:
# yellow = every CellPose boundary touching the crop, green = the sampled
# cell's own boundary drawn on top. The difference is that here there's no
# linked chain of 2D masks to walk - a "3D cell" is just one label already
# native to Pipeline A's volume, so for each sampled label id this walks its
# own z_min..z_max range straight out of the label volume and crops the same
# XY window (around the cell's global centroid) at every Z it appears in.
# Mismatches between the green boundary and the yellow region it sits inside
# are the same kind of visible red flag as in link_3d_cells.py: they mean the
# do_3D stitching merged/split something it shouldn't have between slices.
# -----------------------------------------------------------------------------

def save_dapi_segmentation_montages(cellpose3d_mask_path: str, dapi_path: str | None,
                                     df_cp3d: pd.DataFrame, out_dir: Path,
                                     n_samples: int = DAPI_SEG_N_SAMPLES,
                                     patch_size_px: int = DAPI_SEG_PATCH_PX,
                                     dapi_channel_idx: int = 0,
                                     seed: int = 0) -> None:
    if dapi_path is None:
        print("  --dapi-path not given; skipping DAPI+segmentation montages "
              "(pass the same raw DAPI file test_3d_cellpose.py used as --input).")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tile_dir = out_dir / "visual_tile_qc_cellpose3d"
    tile_dir.mkdir(parents=True, exist_ok=True)

    mask_vol = open_lazy_labels(cellpose3d_mask_path)
    dapi_vol = open_dapi_lazy(dapi_path, dapi_channel_idx)
    n_z, n_y, n_x = mask_vol.shape

    n_samples = min(n_samples, len(df_cp3d))
    if n_samples == 0:
        print("  No Cellpose3D cells available for DAPI+segmentation montages.")
        return
    sample_rows = df_cp3d.sample(n=n_samples, random_state=seed).sort_values("cell_id_3d")

    PATCH = patch_size_px
    TARGET_CELL_COLOUR = np.array([0.15, 1.0, 0.15])  # solid green = the sampled cell
    OUTLINE_COLOUR = np.array([1.0, 1.0, 0.0])         # yellow = every boundary in the crop

    def _stretch_patch(img: np.ndarray) -> np.ndarray:
        fg = img[img > 0]
        if fg.size > 10:
            lo, hi = np.percentile(fg, (2, 98))
        else:
            lo, hi = 0.0, max(float(img.max()), 1e-6)
        return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

    print(f"Generating DAPI+segmentation montages for Cellpose3D (n={n_samples})...")
    for _, row in sample_rows.iterrows():
        cid = int(row["cell_id_3d"])
        z0, z1 = int(row["z_min"]), int(row["z_max"])
        cy, cx = float(row["centroid_y_px"]), float(row["centroid_x_px"])

        y0 = max(0, int(cy) - PATCH); y1 = min(n_y, int(cy) + PATCH)
        x0 = max(0, int(cx) - PATCH); x1 = min(n_x, int(cx) + PATCH)
        pad_t = max(0, PATCH - int(cy)); pad_b = max(0, int(cy) + PATCH - n_y)
        pad_l = max(0, PATCH - int(cx)); pad_r = max(0, int(cx) + PATCH - n_x)

        z_list = list(range(z0, z1 + 1))
        fig, axes = plt.subplots(1, len(z_list), figsize=(len(z_list) * 2.5, 3.0))
        if len(z_list) == 1:
            axes = [axes]

        gap_note = "  |  \u26a0 has_z_gap" if bool(row.get("has_z_gap", False)) else ""
        fig.suptitle(
            f"Cellpose3D cell {cid}  |  span={row['z_span_slices']} slices  |  "
            f"vol={row['volume_um3']:.0f} \u00b5m\u00b3{gap_note}\n"
            f"yellow = every Cellpose3D boundary in crop  |  green outline = this cell",
            fontsize=8, y=1.05,
        )

        for ax, z in zip(axes, z_list):
            mask_crop = np.asarray(mask_vol[z, y0:y1, x0:x1])
            raw_crop = np.asarray(dapi_vol[z, y0:y1, x0:x1]).astype(np.float32)
            gray = _stretch_patch(raw_crop)
            rgb = np.stack([gray, gray, gray], axis=-1)

            # every Cellpose3D label boundary present in the crop, this Z
            boundaries = find_boundaries(mask_crop, mode="outer")
            rgb[boundaries] = OUTLINE_COLOUR

            # the sampled cell's own boundary, drawn on top in green
            target_mask = (mask_crop == cid)
            if target_mask.any():
                target_boundary = find_boundaries(target_mask, mode="outer")
                rgb[target_boundary] = TARGET_CELL_COLOUR

            rgb = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")

            ax.imshow(rgb, interpolation="nearest")
            area = int(target_mask.sum())
            title_colour = "red" if area == 0 else "black"
            ax.set_title(f"Z{z}\narea={area}px", fontsize=6, color=title_colour)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(tile_dir / f"cell_{cid:06d}_dapi_seg.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  DAPI+segmentation montages -> {tile_dir}/")


# -----------------------------------------------------------------------------
# Interactive 3D mesh QC panel (the only piece that stays HTML, since it's the
# one thing worth rotating/zooming - static histograms don't need that)
# -----------------------------------------------------------------------------

def build_3d_qc_html(core_name: str, cellpose3d_mask_path: str, linked3d_mask_path: str,
                      qc_region, xy_pixel_um: float, z_thickness_um: float,
                      out_html: str) -> None:
    (z0, z1), (y0, y1), (x0, x1) = qc_region
    print(f"Loading shared QC crop Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] from both volumes...")
    crop_cp3d = load_mask_crop(cellpose3d_mask_path, z0, z1, y0, y1, x0, x1)
    crop_linked = load_mask_crop(linked3d_mask_path, z0, z1, y0, y1, x0, x1)

    fig = go.Figure()
    for tr in _mesh_traces_for_crop(crop_cp3d, "royalblue", "CP3D", xy_pixel_um,
                                     z_thickness_um, QC_N_SAMPLES_PER_PIPELINE, seed=0):
        fig.add_trace(tr)
    for tr in _mesh_traces_for_crop(crop_linked, "darkorange", "Linked3D", xy_pixel_um,
                                     z_thickness_um, QC_N_SAMPLES_PER_PIPELINE, seed=1):
        fig.add_trace(tr)

    fig.update_layout(
        title=f"{core_name}: shared-crop 3D QC (blue=Cellpose3D, orange=Linked3D)",
        height=800, width=1000,
        scene=dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="z (um)",
                   aspectmode="data"),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"3D QC report saved -> {out_html}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-name", required=True)
    ap.add_argument("--cellpose3d-mask", required=True,
                     help="<CORE>_3D_Cellpose_Masks.tif from test_3d_cellpose.py")
    ap.add_argument("--linked3d-stats", required=True,
                     help="<CORE>_DAPI_3d_stats.csv from link_3d_cells.py")
    ap.add_argument("--linked3d-mask", required=True,
                     help="<CORE>_DAPI_3d_labels.tif from link_3d_cells.py")
    ap.add_argument("--dapi-path", default=None,
                     help="Raw DAPI volume (the same file passed as --input to "
                          "test_3d_cellpose.py). Optional - only needed for the "
                          "DAPI+segmentation tile montages; everything else in "
                          "this script works from the mask/stats files alone.")
    ap.add_argument("--dapi-channel-idx", type=int, default=0,
                     help="Channel index if --dapi-path is a multi-channel "
                          "(Z,C,Y,X) stack (default: 0).")
    ap.add_argument("--n-dapi-seg-samples", type=int, default=DAPI_SEG_N_SAMPLES,
                     help=f"Number of Cellpose3D cells to render as DAPI+segmentation "
                          f"montages (default: {DAPI_SEG_N_SAMPLES}).")
    ap.add_argument("--dapi-seg-patch-px", type=int, default=DAPI_SEG_PATCH_PX,
                     help=f"Half-size of the crop around each cell's centroid, in "
                          f"pixels (default: {DAPI_SEG_PATCH_PX}).")
    ap.add_argument("--min-nucleus-volume-um3", type=float, default=MIN_NUCLEUS_VOLUME_UM3,
                     help=f"Cellpose3D labels smaller than this are flagged as "
                          f"implausible-for-a-nucleus (default: {MIN_NUCLEUS_VOLUME_UM3} "
                          f"um3, ~5 um-diameter sphere - a floor against segmentation "
                          f"debris, not a biological size filter). They're kept in the "
                          f"raw stats CSV but excluded from matching/plots/montages "
                          f"unless --keep-small-volumes is passed.")
    ap.add_argument("--keep-small-volumes", action="store_true",
                     help="Don't exclude labels below --min-nucleus-volume-um3 from "
                          "downstream matching/plots/montages.")
    ap.add_argument("--xy-pixel-um", type=float, required=True)
    ap.add_argument("--z-thickness-um", type=float, required=True)
    ap.add_argument("--max-match-dist-um", type=float, default=None,
                     help="Mutual-NN matching distance cutoff (um). "
                          "Default: 1.5x the median nucleus diameter in the data.")
    ap.add_argument("--qc-region", default=None,
                     help="Shared crop for the 3D mesh panel: 'z0,z1,y0,y1,x0,x1'. "
                          "Default: a small region centered in the volume.")
    ap.add_argument("--skip-matching", action="store_true",
                     help="Skip cell-to-cell centroid matching (use if the two "
                          "volumes are not in the same registered coordinate frame).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Extracting stats from Cellpose3D mask volume: {args.cellpose3d_mask}")
    df_cp3d = extract_stats_from_mask_volume(
        args.cellpose3d_mask, args.xy_pixel_um, args.z_thickness_um
    )

    df_cp3d["below_min_volume"] = df_cp3d["volume_um3"] < args.min_nucleus_volume_um3
    n_small = int(df_cp3d["below_min_volume"].sum())
    print(f"  {n_small:,} / {len(df_cp3d):,} labels ({n_small / len(df_cp3d):.1%}) are "
          f"below --min-nucleus-volume-um3={args.min_nucleus_volume_um3} um3 "
          f"(flagged in 'below_min_volume' column).")

    cp3d_csv = out_dir / f"{args.core_name}_cellpose3d_stats.csv"
    df_cp3d.to_csv(cp3d_csv, index=False)
    print(f"  -> {cp3d_csv}  ({len(df_cp3d):,} cells, all - including flagged small ones)")

    if args.keep_small_volumes or n_small == 0:
        df_cp3d_used = df_cp3d
    else:
        df_cp3d_used = df_cp3d[~df_cp3d["below_min_volume"]].reset_index(drop=True)
        print(f"  Excluding {n_small:,} flagged labels from matching/plots/montages below "
              f"(pass --keep-small-volumes to include them).")

    print(f"[2/5] Loading Linked3D stats: {args.linked3d_stats}")
    df_linked = pd.read_csv(args.linked3d_stats)
    print(f"  -> {len(df_linked):,} cells, columns: {list(df_linked.columns)}")

    required = {"volume_um3", "centroid_x_um", "centroid_y_um", "centroid_z_um"}
    missing = required - set(df_linked.columns)
    if missing:
        raise ValueError(
            f"{args.linked3d_stats} is missing required column(s) {sorted(missing)}. "
            f"This looks like it was written by an older/different version of "
            f"link_3d_cells.py than the one this comparison script assumes. "
            f"Re-run link_3d_cells.py for this core with the current script, "
            f"or point --linked3d-stats at a stats.csv that has these columns."
        )
    has_aspect = "aspect_ratio" in df_linked.columns
    if not has_aspect:
        print("  WARNING: 'aspect_ratio' column not found in Linked3D stats - "
              "skipping aspect-ratio comparison (older link_3d_cells.py output?).")

    if args.skip_matching:
        df_matched = pd.DataFrame()
    else:
        max_dist = args.max_match_dist_um
        if max_dist is None:
            combined_diam = pd.concat([df_cp3d_used["mean_diameter_um"], df_linked.get(
                "mean_diameter_um", pd.Series(dtype=float))])
            max_dist = 1.5 * float(combined_diam.median()) if len(combined_diam) else 10.0
        print(f"[3/5] Matching cells by 3D centroid (max_dist={max_dist:.2f} um)...")
        df_matched = match_by_centroid(df_cp3d_used, df_linked, max_dist)
        matched_csv = out_dir / f"{args.core_name}_matched_pairs.csv"
        df_matched.to_csv(matched_csv, index=False)
        print(f"  -> {matched_csv}  ({len(df_matched):,} matched pairs)")

    if args.qc_region:
        z0, z1, y0, y1, x0, x1 = (int(v) for v in args.qc_region.split(","))
        qc_region = ((z0, z1), (y0, y1), (x0, x1))
    else:
        n_z = open_lazy_labels(args.cellpose3d_mask).shape[0]
        n_y, n_x = open_lazy_labels(args.cellpose3d_mask).shape[1:]
        half = 150
        yc, xc = n_y // 2, n_x // 2
        qc_region = ((0, min(n_z, 16)),
                     (max(0, yc - half), min(n_y, yc + half)),
                     (max(0, xc - half), min(n_x, xc + half)))

    print("[4/5] Writing outputs...")
    save_static_plots(args.core_name, df_cp3d_used, df_linked, df_matched, out_dir)

    print("[5/5] DAPI+segmentation tile montages (Cellpose3D)...")
    save_dapi_segmentation_montages(
        args.cellpose3d_mask, args.dapi_path, df_cp3d_used, out_dir,
        n_samples=args.n_dapi_seg_samples, patch_size_px=args.dapi_seg_patch_px,
        dapi_channel_idx=args.dapi_channel_idx,
    )

    out_html = out_dir / f"{args.core_name}_3d_qc.html"
    build_3d_qc_html(args.core_name, args.cellpose3d_mask, args.linked3d_mask, qc_region,
                     args.xy_pixel_um, args.z_thickness_um, str(out_html))


if __name__ == "__main__":
    main()