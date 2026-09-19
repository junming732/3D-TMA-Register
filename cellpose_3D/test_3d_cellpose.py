"""Executes memory-efficient 3D segmentation using Cellpose v4 via Z, Y, and X tiling.

Cellpose's `do_3D=True` parameter internally resamples the Z-axis by `anisotropy` 
to create an isotropic volume, executing flow passes across three orthogonal views 
(XY, XZ, YZ). Because memory usage scales aggressively with `(Z_chunk * anisotropy) * Y * X`, 
this module mitigates out-of-memory errors on high-resolution volumes by chunking 
the data across all three spatial dimensions. Each `do_3D` invocation processes only 
a single tile (Z_CHUNK native planes x XY_TILE x XY_TILE, plus overlap padding), 
maintaining a stable memory footprint.

Tiling and Stitching Methodology:
    * Z-Axis: Chunks are generated with overlap and stitched across chunk boundaries 
      using Intersection over Union (IoU) on the overlap planes.
    * Y/X-Axes: Tiles are generated with overlap and stitched against the left (X-1) 
      and top (Y-1) neighbors using IoU on the overlap bands, referencing labels 
      already written to the active output buffer for the current Z-chunk.

Notes:
    Label matching utilizes a greedy heuristic (evaluating Z-tail, then left-neighbor, 
    then top-neighbor). If a single object spans two previously stitched neighbors 
    with conflicting identities, it adopts the higher-priority match's ID without 
    merging both. This streaming architecture trade-off intentionally avoids the 
    severe memory overhead of a global union-find pass, though it may occasionally 
    result in duplicate IDs at tile corners.
"""

from __future__ import annotations

import argparse
import colorsys
import inspect
import random
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
import zarr
from cellpose import models
from skimage.measure import marching_cubes

DAPI_CHANNEL_IDX = 0
Z_THICKNESS_UM = 4.0
XY_RESOLUTION_UM = 0.4961
ANISOTROPY = Z_THICKNESS_UM / XY_RESOLUTION_UM  # ~8.063

DIAMETER = None        # CP4/Cellpose-SAM doesn't require this; leave None unless quality needs it
MIN_SIZE = 50
NORMALIZE = True
BATCH_SIZE = 1
# flow_threshold / cellprob_threshold: same semantics as cellpose_segmentation.py
# (2D) - kept as separate module-level defaults here so the two scripts can be
# overridden independently, but pass --flow-threshold/--cellprob-threshold with
# the same values on both scripts for an apples-to-apples 2D vs 3D comparison.
FLOW_THRESHOLD = 0.6
CELLPROB_THRESHOLD = 0.0
# The SAM-based cpsam model (the default for CellposeModel) has a FIXED
# position embedding grid built for 256x256 tiles - it cannot be changed.
# BSIZE=128 (inherited from the original script) silently produces a wrong
# patch grid and crashes inside vit_sam's pos_embed addition. 256 is the
# only valid value for this model; only DINO-based models (cpdino) support
# a configurable bsize.
BSIZE = 256
TILE_OVERLAP = 0.1      # Cellpose's own internal tile overlap fraction

# --- our manual 3D tiling, to bound the isotropic-resample memory blowup ---
Z_CHUNK = 16            # native Z-planes per chunk (before anisotropy resampling)
Z_OVERLAP = 6           # native Z-planes of overlap between chunks, for boundary stitching
XY_TILE = 768           # tile size in Y and X
XY_OVERLAP = 96         # overlap in Y and X, for neighbor stitching
IOU_THRESH = 0.25       # match threshold for all stitching decisions

# --- automatic post-run QC render (mesh nuclei + raw DAPI overlay, as HTML) ---
QC_N_SAMPLES = 20       # nuclei to mesh in the QC region
QC_MIN_VOXELS = 20      # skip labels smaller than this (fragments/noise)
QC_VOLUME_DOWNSAMPLE = 2   # minimum XY downsample for the raw-intensity overlay (auto-increased if needed, see below)
QC_MAX_VOLUME_POINTS = 400_000  # hard cap on raw-overlay point count - keeps the HTML small no matter the region/density
QC_PAD_VOXELS = 2       # padding around each cell's local bounding box before marching_cubes


def open_dapi_lazy(path: str):
    """Return a lazily-readable (Z, Y, X) array-like backed by the file on disk."""
    print(f"Opening {path} lazily (no full-volume read)...")
    store = tifffile.imread(path, aszarr=True)
    z = zarr.open(store, mode="r")
    if z.ndim == 4:
        print(f"Detected 4D array shape {z.shape} (assuming Z,C,Y,X)")
        return z[:, DAPI_CHANNEL_IDX, :, :]
    elif z.ndim == 3:
        print(f"Detected 3D array shape {z.shape} (Z,Y,X)")
        return z
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {z.shape}")


def make_eval_kwargs(model, flow_threshold: float = FLOW_THRESHOLD,
                      cellprob_threshold: float = CELLPROB_THRESHOLD) -> dict:
    supported = set(inspect.signature(model.eval).parameters)
    requested = {
        "batch_size": BATCH_SIZE,
        "bsize": BSIZE,
        "diameter": DIAMETER,
        "do_3D": True,
        "z_axis": 0,
        "anisotropy": ANISOTROPY,
        "min_size": MIN_SIZE,
        "normalize": NORMALIZE,
        "tile_overlap": TILE_OVERLAP,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
    }
    return {k: v for k, v in requested.items() if k in supported}


def tile_ranges(total: int, core: int, overlap: int):
    """Yield (core_start, core_end, read_start, read_end) tiles covering [0, total)."""
    ranges = []
    start = 0
    while start < total:
        core_end = min(start + core, total)
        read_start = max(start - overlap, 0)
        read_end = min(core_end + overlap, total)
        ranges.append((start, core_end, read_start, read_end))
        start = core_end
    return ranges


def _vectorized_best_matches(local_view: np.ndarray, ref_view: np.ndarray, iou_thresh: float) -> dict:
    """For each nonzero id in local_view, find its best-IoU nonzero id in ref_view
    (both same-shaped overlap-band views), using bincount/histogram tricks instead
    of a per-label Python loop with full-array boolean comparisons. Returns
    {local_id: matched_ref_id} only for matches >= iou_thresh.
    """
    local_flat = local_view.ravel().astype(np.int64)
    ref_flat = ref_view.ravel().astype(np.int64)

    local_nonzero = local_flat != 0
    if not local_nonzero.any():
        return {}

    local_counts = np.bincount(local_flat[local_nonzero])
    ref_nonzero = ref_flat != 0
    if not ref_nonzero.any():
        return {}
    ref_counts = np.bincount(ref_flat[ref_nonzero])

    pair_mask = local_nonzero & ref_nonzero
    if not pair_mask.any():
        return {}
    local_p = local_flat[pair_mask]
    ref_p = ref_flat[pair_mask]
    max_ref = int(ref_p.max())
    codes, inter_counts = np.unique(local_p * (max_ref + 1) + ref_p, return_counts=True)
    pair_local = codes // (max_ref + 1)
    pair_ref = codes % (max_ref + 1)

    ious = inter_counts / (local_counts[pair_local] + ref_counts[pair_ref] - inter_counts)

    order = np.argsort(pair_local, kind="stable")
    pl_sorted, pr_sorted, iou_sorted = pair_local[order], pair_ref[order], ious[order]
    unique_pl, start_idx = np.unique(pl_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(pl_sorted))

    result = {}
    for u, s, e in zip(unique_pl, start_idx, end_idx):
        group = iou_sorted[s:e]
        best_i = int(np.argmax(group))
        if group[best_i] >= iou_thresh:
            result[int(u)] = int(pr_sorted[s:e][best_i])
    return result


def match_and_relabel(local_arr: np.ndarray, next_id: int, matches, iou_thresh=IOU_THRESH):
    """Relabel local_arr's ids to global ids using a priority list of neighbor matches.

    matches: list of (local_view, ref_view) pairs, both views into arrays of the
    SAME shape as (a region of) local_arr and its corresponding already-resolved
    neighbor, in that priority order. First match with IoU >= iou_thresh wins.

    Vectorized: no per-label full-array boolean scans. Matching uses a joint-
    histogram over just the (small) overlap band; applying the final mapping
    uses a lookup table indexed by id, not a loop over labels.
    """
    local_ids = np.unique(local_arr)
    local_ids = local_ids[local_ids != 0]
    remaining = set(local_ids.tolist())
    mapping = {}

    for local_view, ref_view in matches:
        if not remaining:
            break
        found = _vectorized_best_matches(local_view, ref_view, iou_thresh)
        for lid, gid in found.items():
            if lid in remaining:
                mapping[lid] = gid
                remaining.discard(lid)

    for lid in remaining:
        mapping[lid] = next_id
        next_id += 1

    if mapping:
        max_local = int(local_arr.max())
        lut = np.zeros(max_local + 1, dtype=np.uint32)
        for lid, gid in mapping.items():
            lut[lid] = gid
        relabeled = lut[local_arr]
    else:
        relabeled = np.zeros_like(local_arr)
    return relabeled, next_id


def auto_qc_region(n_z: int, n_y: int, n_x: int, z_chunk: int, xy_tile: int):
    """Pick a modest crop straddling the first Z-chunk and XY-tile boundary by
    default, since that's the most diagnostically useful place to check for
    stitching artifacts. Kept deliberately small - this is a QC thumbnail, not
    a full-region viewer - so the HTML stays a reasonable size regardless of
    cell density. Use --qc-region to pick a specific area/size instead.
    """
    z0, z1 = 0, min(n_z, z_chunk)
    half = min(xy_tile // 4, 100)
    y_center = min(xy_tile, n_y) if n_y > xy_tile else n_y // 2
    x_center = min(xy_tile, n_x) if n_x > xy_tile else n_x // 2
    y0, y1 = max(0, y_center - half), min(n_y, y_center + half)
    x0, x1 = max(0, x_center - half), min(n_x, x_center + half)
    return (z0, z1), (y0, y1), (x0, x1)


def distinct_color(i: int, n: int) -> str:
    h = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def load_mask_crop(mask_path: str, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Load a (z0:z1, y0:y1, x0:x1) crop from a label TIFF, preferring true 3-axis
    lazy zarr slicing (reads only the crop off disk). Falls back to tifffile's
    page-based `key=` reading (loads full XY pages, crops in numpy - more RAM,
    but has no zarr version dependency) if the zarr path fails for any reason -
    zarr/tifffile version compatibility for aszarr= has been known to break
    across environments.
    """
    try:
        store = tifffile.imread(mask_path, aszarr=True)
        z = zarr.open(store, mode="r")
        return np.asarray(z[z0:z1, y0:y1, x0:x1])
    except Exception as e:
        print(f"  (zarr lazy-load unavailable ({type(e).__name__}: {e}), "
              f"falling back to page-based reading)")
        pages = np.asarray(tifffile.imread(mask_path, key=range(z0, z1)))
        return pages[:, y0:y1, x0:x1]


def render_qc_html(dapi_lazy, mask_path: str, region, out_html: str,
                    n_samples: int = QC_N_SAMPLES, min_voxels: int = QC_MIN_VOXELS,
                    seed: int = 0, volume_downsample: int = QC_VOLUME_DOWNSAMPLE) -> None:
    """Render a sample of segmented nuclei as meshes, with the raw DAPI signal
    for the same crop shown as a semi-transparent volume behind them, so mask
    quality can be checked against real signal - not just viewed in isolation.
    All coordinates are crop-local (origin at the crop corner); this is a
    standalone QC image, not meant to align with any other rendering.
    """
    (z0, z1), (y0, y1), (x0, x1) = region
    print(f"\nQC render: cropping Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}]...")

    raw = np.asarray(dapi_lazy[z0:z1, y0:y1, x0:x1]).astype(np.float32)
    label_vol = load_mask_crop(mask_path, z0, z1, y0, y1, x0, x1)

    ids, counts = np.unique(label_vol, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    ids = ids[counts >= min_voxels]
    if len(ids) == 0:
        print("QC render: no nuclei found in this region, skipping.")
        return

    random.seed(seed)
    n = min(n_samples, len(ids))
    selected = sorted(random.sample(ids.tolist(), n))
    print(f"QC render: meshing {n} nuclei...")

    traces = []
    for i, cid in enumerate(selected):
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
                spacing=(Z_THICKNESS_UM, XY_RESOLUTION_UM, XY_RESOLUTION_UM),
            )
        except (ValueError, RuntimeError):
            continue
        voxel_offset = np.array([zc0, yc0, xc0])
        verts_um = verts + voxel_offset * np.array([Z_THICKNESS_UM, XY_RESOLUTION_UM, XY_RESOLUTION_UM])
        color = distinct_color(i, n)
        traces.append(go.Mesh3d(
            x=verts_um[:, 2], y=verts_um[:, 1], z=verts_um[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.85, name=f"nucleus {int(cid)}", showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2, roughness=0.6),
        ))

    # raw DAPI intensity as a semi-transparent volume, same crop-local coords.
    # Downsample factor is auto-increased (never decreased below the requested
    # minimum) so the point count stays under QC_MAX_VOLUME_POINTS no matter
    # how large a region was passed in - this is what keeps the HTML small.
    ds = max(volume_downsample, 1)
    z_n, y_n, x_n = raw.shape
    while z_n * max(y_n // ds, 1) * max(x_n // ds, 1) > QC_MAX_VOLUME_POINTS and ds < 64:
        ds += 1
    if ds > volume_downsample:
        print(f"QC render: auto-increased volume downsample to {ds}x to keep the HTML small "
              f"({z_n * (y_n//ds) * (x_n//ds):,} points)")
    raw_ds = raw[:, ::ds, ::ds]
    zz, yy, xx = np.mgrid[0:raw_ds.shape[0], 0:raw_ds.shape[1], 0:raw_ds.shape[2]]
    zz_um = zz * Z_THICKNESS_UM
    yy_um = (yy * ds) * XY_RESOLUTION_UM
    xx_um = (xx * ds) * XY_RESOLUTION_UM
    lo, hi = np.percentile(raw_ds, [50, 99.5])
    traces.insert(0, go.Volume(
        x=xx_um.flatten(), y=yy_um.flatten(), z=zz_um.flatten(), value=raw_ds.flatten(),
        isomin=float(lo), isomax=float(hi), opacity=0.08, surface_count=12,
        colorscale="Greys", showscale=False, name="DAPI signal",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f"QC: {len(traces) - 1} nuclei + raw DAPI overlay "
               f"(Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}])"),
        scene=dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="z (um)",
                   aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"QC render saved -> {out_html}")


def main(input_path: str, output_path: str, z_chunk: int, z_overlap: int,
         xy_tile: int, xy_overlap: int, qc_region=None, skip_qc: bool = False,
         qc_output: str | None = None, flow_threshold: float = FLOW_THRESHOLD,
         cellprob_threshold: float = CELLPROB_THRESHOLD) -> None:
    t0 = time.time()
    dapi_lazy = open_dapi_lazy(input_path)
    n_z, n_y, n_x = dapi_lazy.shape
    print(f"Full volume: Z={n_z}, Y={n_y}, X={n_x}")
    print(f"Anisotropy: {ANISOTROPY:.3f} (Z={Z_THICKNESS_UM} um, XY={XY_RESOLUTION_UM} um)")

    print("Initializing Cellpose model...")
    model = models.CellposeModel(gpu=True)
    eval_kwargs = make_eval_kwargs(model, flow_threshold=flow_threshold,
                                    cellprob_threshold=cellprob_threshold)
    print("Cellpose eval parameters:")
    for key, value in eval_kwargs.items():
        print(f"  {key}={value}")

    y_ranges = tile_ranges(n_y, xy_tile, xy_overlap)
    x_ranges = tile_ranges(n_x, xy_tile, xy_overlap)
    print(f"XY tile grid: {len(y_ranges)} x {len(x_ranges)} tiles "
          f"(tile={xy_tile}, overlap={xy_overlap})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    next_id = 1
    prev_z_tail = {}  # (yi, xi) -> full-read-extent array, last z_overlap planes, global ids

    zstart = 0
    z_chunk_idx = 0

    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        while zstart < n_z:
            zcore0, zcore1 = zstart, min(zstart + z_chunk, n_z)
            zread0 = max(zcore0 - z_overlap, 0) if z_chunk_idx > 0 else zcore0
            zread1 = zcore1  # tail overlap only needed looking backward, not forward
            z_core_len = zcore1 - zcore0
            print(f"\nZ-chunk {z_chunk_idx}: core Z[{zcore0}:{zcore1}), read Z[{zread0}:{zread1})")

            chunk_result = np.zeros((z_core_len, n_y, n_x), dtype=np.uint32)

            for yi, (ycore0, ycore1, yread0, yread1) in enumerate(y_ranges):
                for xi, (xcore0, xcore1, xread0, xread1) in enumerate(x_ranges):
                    tile = np.asarray(
                        dapi_lazy[zread0:zread1, yread0:yread1, xread0:xread1]
                    )
                    masks, flows, styles = model.eval(tile, **eval_kwargs)
                    local_masks = np.asarray(masks).astype(np.uint32, copy=False)
                    del flows, styles, tile

                    # --- Z stitching against previous chunk's tail for this tile ---
                    key = (yi, xi)
                    if z_chunk_idx > 0 and key in prev_z_tail:
                        z_overlap_actual = min(z_overlap, local_masks.shape[0])
                        z_matches = [(local_masks[:z_overlap_actual], prev_z_tail[key])]
                        local_masks, next_id = match_and_relabel(local_masks, next_id, z_matches)
                    else:
                        local_masks, next_id = match_and_relabel(local_masks, next_id, [])

                    # trim to core Z range (drop backward-looking overlap planes)
                    z_off = zcore0 - zread0
                    local_core_z = local_masks[z_off:z_off + z_core_len]

                    # --- XY stitching against already-placed left/top neighbors ---
                    xy_matches = []
                    left_overlap_w = xcore0 - xread0
                    if xi > 0 and left_overlap_w > 0:
                        y_off = ycore0 - yread0
                        y_len = ycore1 - ycore0
                        local_view = local_core_z[:, y_off:y_off + y_len, 0:left_overlap_w]
                        ref_view = chunk_result[:, ycore0:ycore1, xread0:xcore0]
                        xy_matches.append((local_view, ref_view))
                    top_overlap_h = ycore0 - yread0
                    if yi > 0 and top_overlap_h > 0:
                        x_off = xcore0 - xread0
                        x_len = xcore1 - xcore0
                        local_view = local_core_z[:, 0:top_overlap_h, x_off:x_off + x_len]
                        ref_view = chunk_result[:, yread0:ycore0, xcore0:xcore1]
                        xy_matches.append((local_view, ref_view))

                    relabeled_full, next_id = match_and_relabel(local_core_z, next_id, xy_matches)

                    # save tail (full XY read extent) for the NEXT z-chunk's comparison
                    tail_len = min(z_overlap, relabeled_full.shape[0])
                    prev_z_tail[key] = relabeled_full[-tail_len:].copy()

                    # write only the XY core into the chunk buffer
                    x_off_core = xcore0 - xread0
                    x_len = xcore1 - xcore0
                    y_off_core = ycore0 - yread0
                    y_len = ycore1 - ycore0
                    chunk_result[:, ycore0:ycore1, xcore0:xcore1] = relabeled_full[
                        :, y_off_core:y_off_core + y_len, x_off_core:x_off_core + x_len
                    ]

                print(f"  row {yi + 1}/{len(y_ranges)} done, running max id: {next_id - 1:,}")

            # photometric="minisblack" is required here: without it, tifffile
            # guesses from array shape whether a small leading axis (1-4) means
            # "color channels" rather than "Z-planes" - a chunk with exactly
            # 1-4 planes (e.g. the last, uneven chunk) gets silently stored as
            # one RGB-style sample-interleaved image instead of separate
            # grayscale label pages, corrupting the file structure downstream
            # code (including render_qc_html) assumes.
            writer.write(chunk_result, compression="deflate", metadata={"axes": "ZYX"},
                         photometric="minisblack")
            del chunk_result

            print(f"Z-chunk {z_chunk_idx} written. Elapsed: {time.time() - t0:.1f}s")
            zstart = zcore1
            z_chunk_idx += 1

    print(f"\nSegmentation complete in {time.time() - t0:.1f}s.")
    print(f"~{next_id - 1:,} unique 3D cells (approx, post-stitch).")
    print(f"Saved uint32 output to {output_path}")

    if not skip_qc:
        region = qc_region or auto_qc_region(n_z, n_y, n_x, z_chunk, xy_tile)
        out_html = qc_output or str(Path(output_path).with_suffix("").with_suffix(".qc_3d.html"))
        try:
            render_qc_html(dapi_lazy, output_path, region, out_html)
        except Exception as e:
            # QC rendering is a nice-to-have; never let it fail the segmentation run
            print(f"QC render failed (segmentation output is still valid): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cellpose v4 true 3D segmentation, tiled in Z/Y/X to bound RAM"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flow-threshold", type=float, default=FLOW_THRESHOLD,
                         help=f"Cellpose flow error threshold. Default: {FLOW_THRESHOLD}")
    parser.add_argument("--cellprob-threshold", type=float, default=CELLPROB_THRESHOLD,
                         help=f"Cellpose cell probability threshold. Default: {CELLPROB_THRESHOLD}")
    parser.add_argument("--z-chunk", type=int, default=Z_CHUNK)
    parser.add_argument("--z-overlap", type=int, default=Z_OVERLAP)
    parser.add_argument("--xy-tile", type=int, default=XY_TILE)
    parser.add_argument("--xy-overlap", type=int, default=XY_OVERLAP)
    parser.add_argument("--qc-region", default=None,
                         help="QC render crop 'z0,z1,y0,y1,x0,x1'. Default: auto-picked "
                              "to straddle the first Z-chunk/tile boundary.")
    parser.add_argument("--qc-output", default=None,
                         help="Path for the QC HTML. Default: <output>.qc_3d.html")
    parser.add_argument("--skip-qc", action="store_true", help="Don't render the QC HTML")
    args = parser.parse_args()

    qc_region = None
    if args.qc_region:
        z0, z1, y0, y1, x0, x1 = (int(v) for v in args.qc_region.split(","))
        qc_region = ((z0, z1), (y0, y1), (x0, x1))

    main(args.input, args.output, args.z_chunk, args.z_overlap, args.xy_tile, args.xy_overlap,
         qc_region=qc_region, skip_qc=args.skip_qc, qc_output=args.qc_output,
         flow_threshold=args.flow_threshold, cellprob_threshold=args.cellprob_threshold)