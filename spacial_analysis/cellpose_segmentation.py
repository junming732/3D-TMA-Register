"""
cellpose_segmentation.py
===============
Run CellPose nuclear segmentation on original (unregistered) mIF slices
using the DAPI channel only.

  DAPI (channel index 0) — nuclear segmentation
       Model  : nuclei
       Input  : single channel (DAPI only); CellPose normalises internally
       Output : integer label masks, one per slice

Usage
-----
    python cellpose_segmentation.py \\
        --core_name  <CORE_NAME>           \\
        [--diameter  25]                   \\
        [--flow_threshold 0.4]             \\
        [--cellprob_threshold -1.0]        \\
        [--use_gpu]                        \\
        [--batch_size 1]                   \\
        [--plot_qc]                        \\
        [--save_flows]

Input
-----
Original (unregistered) ome.tif slices from:
    <DATASPACE>/TMA_Cores_Grouped_Rotate_Conformed/<CORE_NAME>/

Output (per slice)
------------------
    <DATASPACE>/CellPose_DAPI/<CORE_NAME>/
        TMA_<ID>_DAPI_cp_masks.tif         — uint32 label mask
        TMA_<ID>_DAPI_cp_flows.tif         — (optional) flow magnitude map
        <CORE_NAME>_cellpose_DAPI_stats.csv
        qc_plots/  (if --plot_qc)
            TMA_<ID>_DAPI_qc.png           — 3-panel: raw | mask overlay | centroid overlay

Downstream
----------
Pass the label masks and the deformation_maps/ folder to warp_cellpose_masks.py
to bring the segmentation into the registered image space.
"""

import os
import sys
import re
import glob
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
import yaml
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# Redirect CellPose weight cache to the shared model_weights directory in
# DATASPACE — the same location used by RoMaV2.  Must be set BEFORE cellpose
# is imported, because cellpose reads this variable at import time.
os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = os.path.join(
    config.DATASPACE, 'model_weights', 'cellpose'
)
os.makedirs(os.environ['CELLPOSE_LOCAL_MODELS_PATH'], exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DAPI CHANNEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# idx            : channel index in the ome.tif (C dimension)
# model          : CellPose model name
# diameter       : expected nucleus diameter in pixels (None = auto-estimate)
# flow_threshold : CellPose flow error threshold (higher → more permissive)
# cellprob_threshold : cell probability threshold (lower → more cells detected)
# notes          : free-text explanation
#
DAPI_CONFIG = dict(
    idx                = 0,
    model              = 'nuclei',
    diameter           = 14,     # 0.4961 µm/px → 25 px ≈ 12.4 µm; mid-range for typical nuclei (10–15 µm)
                                 # Override with --diameter 20 for small nuclei or --diameter 30 for large
    flow_threshold     = 0.8,    # standard for nuclei; lower to 0.3 if over-segmenting
    cellprob_threshold = -3.0,   # slightly permissive to catch dim/peripheral nuclei;
                                 # raise to 0.0 if getting too much background
    notes = (
        "Nuclear segmentation at 0.4961 µm/px. Uses the 'nuclei' model (single-channel). "
        "diameter=25 px ≈ 12.4 µm — mid-range for typical nuclei (10–15 µm). "
        "Run with --min_size 200 to filter debris. "
        "Tune: --diameter 20 (small nuclei) or --diameter 30 (large nuclei). "
        "Lower cellprob_threshold to -2.0 if missing dim nuclei; raise to 0.0 to reduce background."
    ),
)

CH_NAME = 'DAPI'
CFG     = DAPI_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='CellPose nuclear segmentation on DAPI channel of mIF slices.'
)
parser.add_argument('--core_name',          type=str,   required=True)
parser.add_argument('--diameter',           type=float, default=None,
                    help="Override nucleus diameter (px). Default: 25 px ≈ 12.4 µm at 0.4961 µm/px.")
parser.add_argument('--flow_threshold',     type=float, default=None,
                    help="Override flow_threshold. Default: 0.4")
parser.add_argument('--cellprob_threshold', type=float, default=None,
                    help="Override cellprob_threshold. Default: -1.0")
parser.add_argument('--use_gpu',            action='store_true',
                    help="Use GPU if available.")
parser.add_argument('--batch_size',         type=int,   default=1,
                    help="Number of slices to process per CellPose call (default: 1).")
parser.add_argument('--plot_qc',            action='store_true',
                    help="Save QC overlay images (mask overlay + centroid overlay).")
parser.add_argument('--save_flows',         action='store_true',
                    help="Save flow magnitude TIFF alongside label masks.")
parser.add_argument('--min_size',           type=int,   default=50,
                    help="Minimum nucleus size in pixels (default: 200; ~π*(25/2)² * 0.5 at 0.4961 µm/px).")
parser.add_argument('--slice_filter_yaml',  type=str,   default=None,
                    help="Path to slice_filter.yaml (default: <DATASPACE>/slice_filter.yaml).")
args = parser.parse_args()

TARGET_CORE = args.core_name

# Allow CLI overrides
DIAMETER           = args.diameter           if args.diameter           is not None else CFG['diameter']
FLOW_THRESHOLD     = args.flow_threshold     if args.flow_threshold     is not None else CFG['flow_threshold']
CELLPROB_THRESHOLD = args.cellprob_threshold if args.cellprob_threshold is not None else CFG['cellprob_threshold']

logger.info(f"Channel: {CH_NAME}  |  {CFG['notes']}")
logger.info(
    f"Model={CFG['model']}  diameter={DIAMETER}  "
    f"flow_thresh={FLOW_THRESHOLD}  cellprob_thresh={CELLPROB_THRESHOLD}"
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
OUTPUT_FOLDER     = os.path.join(config.DATASPACE, "CellPose_DAPI", TARGET_CORE)
SLICE_FILTER_YAML = args.slice_filter_yaml or os.path.join(config.DATASPACE, "slice_filter.yaml")

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if args.plot_qc:
    QC_DIR = os.path.join(OUTPUT_FOLDER, "qc_plots")
    os.makedirs(QC_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL METADATA
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_XY_UM = 0.4961

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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


def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def load_slice(filepath):
    arr = tifffile.imread(filepath)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    return arr   # (C, H, W)


def extract_channel(arr: np.ndarray, ch_idx: int) -> np.ndarray:
    """
    Extract a single channel as float32 for CellPose v4.

    We pass normalize=True to model.eval() so CellPose handles normalisation
    internally on a per-tile basis.  This avoids the checkerboard artifact
    that appears in the probability map when a globally pre-normalised image
    is tiled — adjacent tiles see slightly different effective contrast at
    their seams when normalisation is done externally.
    """
    return arr[ch_idx].astype(np.float32)   # (H, W) float32


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_cp_model = None

def get_model(model_name: str, use_gpu: bool):
    global _cp_model
    if _cp_model is None:
        from cellpose import models
        gpu = use_gpu
        try:
            import torch
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but CUDA unavailable — running on CPU.")
                gpu = False
        except ImportError:
            gpu = False
        logger.info(f"Loading CellPose model '{model_name}' (gpu={gpu})...")
        _cp_model = models.CellposeModel(model_type=model_name, gpu=gpu)
        logger.info("CellPose model loaded.")
    return _cp_model


# ─────────────────────────────────────────────────────────────────────────────
# QC PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_stretch(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Percentile-stretch a 2-D float array to [0, 1], ignoring zero-padding."""
    fg = img[img > 0]
    p_lo, p_hi = np.percentile(fg, (lo, hi)) if fg.size > 0 else (0, 1)
    return np.clip((img.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)


def _compute_centroids(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cy, cx) arrays of nucleus centroid coordinates.
    Uses label bounding-box moments — fast even for 100 k+ nuclei.
    """
    from scipy import ndimage as ndi
    labels = np.arange(1, masks.max() + 1)
    
    # Capture the list of coordinate tuples: [(y1, x1), (y2, x2), ...]
    centroids = ndi.center_of_mass(masks > 0, labels=masks, index=labels)
    
    # Convert to a 2D NumPy array for vectorized slicing
    centroids_arr = np.array(centroids)
    
    # Safety check: Ensure the array has the expected 2D shape (N, 2)
    # This prevents IndexError if the mask contains labels but no valid volumes
    if centroids_arr.ndim != 2:
        return np.array([]), np.array([])
        
    # Slice the array by column to separate Y and X coordinates
    cy = centroids_arr[:, 0]
    cx = centroids_arr[:, 1]
    
    return cy, cx


def save_qc_plot(raw_dapi: np.ndarray, masks: np.ndarray,
                 slice_id: int, n_cells: int, out_path: str):
    """
    3-panel QC figure for DAPI nuclear segmentation:

      Panel 1 — raw DAPI (percentile-stretched, grayscale)
      Panel 2 — mask overlay  : coloured label masks on raw DAPI
      Panel 3 — centroid overlay : nucleus centroids (dots) on raw DAPI

    Both overlay panels are plotted at the full image resolution so that
    even small/touching nuclei can be inspected by zooming the saved PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    display = _percentile_stretch(raw_dapi)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 1: raw DAPI ────────────────────────────────────────────────────
    axes[0].imshow(display, cmap='gray', interpolation='nearest')
    axes[0].set_title(f"Raw DAPI  (slice {slice_id})", fontsize=11)
    axes[0].axis('off')

    # ── Panel 2: mask overlay ────────────────────────────────────────────────
    axes[1].imshow(display, cmap='gray', interpolation='nearest')
    if n_cells > 0:
        # Cycle tab20 colours so adjacent labels are visually distinct.
        # Shift by 1 so background (0) and any label that is a multiple of
        # 19 don't both render as the same tab20 bin-0 blue.
        cycled = np.where(masks > 0, (masks % 19) + 1, 0).astype(np.uint8)
        axes[1].imshow(cycled, cmap='tab20', vmin=0, vmax=20,
                       interpolation='nearest', alpha=0.55)
    axes[1].set_title(f"Mask overlay — {n_cells} nuclei", fontsize=11)
    axes[1].axis('off')

    # ── Panel 3: centroid overlay ────────────────────────────────────────────
    axes[2].imshow(display, cmap='gray', interpolation='nearest')
    if n_cells > 0:
        cy, cx = _compute_centroids(masks)
        # Dot size: scale inversely with nucleus count so the plot stays
        # readable from a handful of nuclei up to >100 k.
        dot_size = max(1.0, min(8.0, 3000.0 / max(n_cells, 1)))
        axes[2].scatter(cx, cy, s=dot_size, c='#00e5ff',
                        linewidths=0, alpha=0.8, rasterized=True)
    axes[2].set_title(f"Centroid overlay — {n_cells} nuclei", fontsize=11)
    axes[2].axis('off')

    fig.suptitle(
        f"CellPose DAPI QC — {TARGET_CORE}  slice {slice_id}  "
        f"diameter={DIAMETER}  flow_thresh={FLOW_THRESHOLD}  "
        f"cellprob_thresh={CELLPROB_THRESHOLD}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  QC plot saved → {out_path}")


def save_full_res_overlay_tiff(raw_dapi: np.ndarray, masks: np.ndarray, out_path: str):
    """
    Generates a 1:1 native resolution RGB overlay of the entire slice
    and saves it directly via tifffile, bypassing Matplotlib's DPI limits.
    """
    # 1. Normalize DAPI to 8-bit
    fg_pixels = raw_dapi[raw_dapi > 0]
    p_lo, p_hi = np.percentile(fg_pixels, (0.5, 99.5)) if fg_pixels.size > 0 else (0, 1)
    
    dapi_norm = np.clip((raw_dapi.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    dapi_8bit = (dapi_norm * 255).astype(np.uint8)
    
    # 2. Broadcast to 3-channel RGB (H, W, 3)
    rgb_img = np.stack([dapi_8bit] * 3, axis=-1)
    
    # 3. Generate a deterministic color lookup table (LUT)
    np.random.seed(42)
    lut = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)
    lut[0] = [0, 0, 0]  # Ensure background (0) maps strictly to black
    
    # Map uint32 masks into the 8-bit LUT space. 
    # Shift by 1 to prevent valid labels from hashing to 0.
    mask_mod = np.where(masks > 0, (masks % 255) + 1, 0).astype(np.uint8)
    color_mask = lut[mask_mod]
    
    # 4. Blend images via NumPy vectorization
    fg_idx = masks > 0
    alpha = 0.4  # Opacity of the segmentation mask
    
    # Apply standard alpha blending: Background * (1 - alpha) + Foreground * alpha
    rgb_img[fg_idx] = (rgb_img[fg_idx] * (1.0 - alpha) + color_mask[fg_idx] * alpha).astype(np.uint8)
    
    # 5. Write directly to disk as a compressed BigTIFF
    tifffile.imwrite(
        out_path,
        rgb_img,
        photometric='rgb',
        compression='deflate',
        bigtiff=True
    )
    logger.info(f"  Full-res RGB overlay saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT ONE SLICE
# ─────────────────────────────────────────────────────────────────────────────

def segment_slice(arr: np.ndarray, slice_id: int) -> dict:
    """
    Run CellPose on the DAPI channel of one slice and write outputs.

    Returns a stats dict for the summary CSV.
    """
    t0 = time.time()

    model = get_model(CFG['model'], args.use_gpu)

    # Extract DAPI channel — (H, W) float32, single channel.
    # normalize=True is passed to model.eval() so CellPose handles
    # normalisation per-tile internally, avoiding checkerboard artifacts.
    dapi_img = extract_channel(arr, CFG['idx'])   # (H, W) float32

    # ── Run CellPose ─────────────────────────────────────────────────────────
    # normalize=True: CellPose normalises each tile independently and
    # consistently, which avoids the checkerboard artifact in the probability
    # map that appears when a globally pre-normalised image is tiled.
    #
    # Tiling is critical for 6000+ px images: without it CellPose downsizes
    # the entire image to its internal resolution and loses all small nuclei.
    # tile=True (default) with tile_overlap=0.1 avoids seam artefacts.
    # ─────────────────────────────────────────────────────────────────────────
    try:
        masks, flows, styles = model.eval(
            dapi_img,
            diameter           = DIAMETER,
            flow_threshold     = FLOW_THRESHOLD,
            cellprob_threshold = CELLPROB_THRESHOLD,
            min_size           = args.min_size,
            do_3D              = False,
            batch_size         = args.batch_size,
            normalize          = True,    # CellPose normalises per-tile — no checkerboard
            tile_overlap       = 0.1,     # 10% overlap avoids seam artefacts
        )
    except Exception as exc:
        logger.error(f"CellPose eval failed for slice {slice_id}: {exc}")
        return dict(
            slice_id=slice_id, n_cells=0, mean_area_px=0.0,
            median_area_px=0.0, total_area_px=0, runtime_s=0.0, status="FAILED",
        )

    # Use uint32 — large tissue sections routinely exceed 65535 nuclei
    masks   = masks.astype(np.uint32)
    n_cells  = int(masks.max())
    runtime  = time.time() - t0

    # ── Nucleus size statistics ───────────────────────────────────────────────
    if n_cells > 0:
        # bincount is orders of magnitude faster than iterating per label
        areas       = np.bincount(masks.ravel())[1:]   # index 0 = background
        mean_area   = float(np.mean(areas))
        median_area = float(np.median(areas))
        total_area  = int(areas.sum())
    else:
        mean_area = median_area = total_area = 0.0

    logger.info(
        f"  Slice ID {slice_id:03d}: {n_cells} nuclei | "
        f"mean area={mean_area:.0f}px  median={median_area:.0f}px | "
        f"{runtime:.1f}s"
    )

    # ── Save label mask ───────────────────────────────────────────────────────
    base      = f"TMA_{slice_id:03d}_DAPI"
    mask_path = os.path.join(OUTPUT_FOLDER, f"{base}_cp_masks.tif")
    tifffile.imwrite(mask_path, masks.astype(np.uint32), compression='deflate')

    # ── Save flow magnitude (optional) ────────────────────────────────────────
    if args.save_flows and len(flows) > 1 and flows[1] is not None:
        xy = np.array(flows[1])   # (2, H, W) or (H, W, 2)
        if xy.ndim == 3 and xy.shape[0] == 2:
            xy = np.moveaxis(xy, 0, -1)  # → (H, W, 2)
        flows_mag = np.sqrt(xy[..., 0]**2 + xy[..., 1]**2).astype(np.float32)
        flow_path = os.path.join(OUTPUT_FOLDER, f"{base}_cp_flows.tif")
        tifffile.imwrite(flow_path, flows_mag, compression='deflate')

    # ── QC plot ───────────────────────────────────────────────────────────────
    if args.plot_qc:
        qc_path = os.path.join(QC_DIR, f"{base}_qc.png")
        overlay_path = os.path.join(QC_DIR, f"{base}_full_overlay.tif")
        try:
            save_qc_plot(
                raw_dapi = dapi_img,
                masks    = masks,
                slice_id = slice_id,
                n_cells  = n_cells,
                out_path = qc_path,
            )

            save_full_res_overlay_tiff(
                raw_dapi = dapi_img,
                masks    = masks,
                out_path = overlay_path
            )
        except Exception as exc:
            logger.warning(f"QC plot failed for slice {slice_id}: {exc}")

    return dict(
        slice_id       = slice_id,
        n_cells        = n_cells,
        mean_area_px   = round(mean_area, 2),
        median_area_px = round(median_area, 2),
        total_area_px  = total_area,
        mean_area_um2  = round(mean_area   * PIXEL_SIZE_XY_UM**2, 3),
        median_area_um2= round(median_area * PIXEL_SIZE_XY_UM**2, 3),
        total_area_um2 = round(total_area  * PIXEL_SIZE_XY_UM**2, 3),
        runtime_s      = round(runtime, 2),
        status         = "OK" if n_cells > 0 else "EMPTY",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"CellPose DAPI nuclear segmentation — core={TARGET_CORE}")
    logger.info(f"Input : {INPUT_FOLDER}")
    logger.info(f"Output: {OUTPUT_FOLDER}")

    raw_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")),
                       key=get_slice_number)
    if not raw_files:
        logger.error(f"No .ome.tif files in {INPUT_FOLDER}")
        sys.exit(1)

    allowed = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)
    if allowed is not None:
        raw_files = [f for i, f in enumerate(raw_files) if i in allowed]
        logger.info(f"Slice filter active: {len(raw_files)} slices retained.")
    logger.info(f"Slices to process: {len(raw_files)}")

    # Preload model once
    get_model(CFG['model'], args.use_gpu)

    all_stats = []
    for fpath in raw_files:
        sid = get_slice_number(fpath)
        logger.info(f"Processing slice ID {sid:03d}: {os.path.basename(fpath)}")
        try:
            arr  = load_slice(fpath)
            stat = segment_slice(arr, sid)
        except Exception as exc:
            logger.error(f"Slice {sid} crashed: {exc}")
            stat = dict(
                slice_id=sid, n_cells=0, mean_area_px=0, median_area_px=0,
                total_area_px=0, mean_area_um2=0, median_area_um2=0,
                total_area_um2=0, runtime_s=0, status="CRASHED",
            )
        all_stats.append(stat)
        del arr

    # Summary CSV
    df = pd.DataFrame(all_stats)
    csv_path = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}_cellpose_DAPI_stats.csv")
    df.to_csv(csv_path, index=False)

    n_ok    = int((df['status'] == 'OK').sum())
    n_empty = int((df['status'] == 'EMPTY').sum())
    n_fail  = int(df['status'].isin(['FAILED', 'CRASHED']).sum())
    logger.info(
        f"Done. OK={n_ok}  EMPTY={n_empty}  FAILED/CRASHED={n_fail} | "
        f"Stats → {csv_path}"
    )


if __name__ == "__main__":
    main()