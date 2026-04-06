"""
cellposes.py
===============
Run CellPose segmentation on original (unregistered) mIF slices.

Supports two segmentation modes depending on the target channel:

  CD3  (channel index 4) — compact T-cell segmentation
       Model  : cyto3
       Pairing: CD3 (cyto) + DAPI (nuclear)
       Output : integer label masks, one per slice

  CD31 (channel index 1) — endothelial / vessel segmentation
       Model  : cyto3  (vessels are elongated — lower flow_threshold)
       Pairing: CD31 only (no nuclear channel; vessels lack DAPI signal)
       Output : integer label masks + optional binary vessel-area mask

Additional channels can be added in CHANNEL_CONFIGS below.

Usage
-----
    python run_cellpose.py \\
        --core_name  <CORE_NAME>           \\
        --channel    CD3                   \\
        [--diameter  30]                   \\
        [--flow_threshold 0.4]             \\
        [--cellprob_threshold 0.0]         \\
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
    <DATASPACE>/CellPose_<CHANNEL>/<CORE_NAME>/
        TMA_<ID>_<CHANNEL>_cp_masks.tif    — uint32 label mask
        TMA_<ID>_<CHANNEL>_cp_flows.tif    — (optional) flow magnitude map
        TMA_<ID>_<CHANNEL>_cp_probs.tif    — cell probability map
        <CORE_NAME>_cellpose_<CHANNEL>_stats.csv
        qc_plots/  (if --plot_qc)

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
# CHANNEL CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each entry describes how CellPose should handle that marker.
#
#   idx            : channel index in the ome.tif (C dimension)
#   nuclear_idx    : index of the nuclear (DAPI) channel, or None
#   model          : CellPose model name
#   diameter       : expected cell/object diameter in pixels (None = auto-estimate)
#   flow_threshold : CellPose flow error threshold (higher → more permissive)
#   cellprob_threshold : cell probability threshold (lower → more cells detected)
#   notes          : free-text explanation
#
CHANNEL_CONFIGS = {
    'CD3': dict(
        idx                = 4,
        nuclear_idx        = None,   # single-channel only — CD3 is membranous,
                                     # DAPI adds noise more than signal in mIF
        model              = 'cyto3',
        diameter           = 30,     # ~15 µm / 0.5 µm per px; tune per dataset
        flow_threshold     = 0.4,
        cellprob_threshold = 0.0,
        notes = (
            "T-cells: compact round morphology. Single-channel cyto3 on CD3 only. "
            "Diameter ~30 px at 0.5 µm/px ≈ 15 µm."
        ),
    ),
    'CD31': dict(
        idx                = 1,
        nuclear_idx        = None,   # endothelial cells lack reliable DAPI signal
        model              = 'cyto3',
        diameter           = 50,     # vessels are larger and elongated
        flow_threshold     = 0.6,    # more permissive — irregular vessel shapes
        cellprob_threshold = -1.0,   # lower to catch dim vessel margins
        notes = (
            "Endothelial / vessel marker. Objects are elongated vessel cross-sections, "
            "not individual round cells. Interpret label count as vessel segments, "
            "not cell count. Consider binary vessel-area analysis as complement."
        ),
    ),
    'DAPI': dict(
        idx                = 0,
        nuclear_idx        = None,
        model              = 'nuclei',
        diameter           = 20,
        flow_threshold     = 0.4,
        cellprob_threshold = 0.0,
        notes              = "Nuclear segmentation using the nuclei model.",
    ),
    'CD163': dict(
        idx                = 5,
        nuclear_idx        = None,   # single-channel only — consistent with CD3 approach
        model              = 'cyto3',
        diameter           = 30,
        flow_threshold     = 0.4,
        cellprob_threshold = 0.0,
        notes              = "Macrophage marker; single-channel cyto3 on CD163 only.",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='CellPose segmentation on original mIF slices.'
)
parser.add_argument('--core_name',          type=str,   required=True)
parser.add_argument('--channel',            type=str,   required=True,
                    choices=list(CHANNEL_CONFIGS.keys()),
                    help=f"Target channel. One of: {list(CHANNEL_CONFIGS.keys())}")
parser.add_argument('--diameter',           type=float, default=None,
                    help="Override diameter (px). Default: from CHANNEL_CONFIGS.")
parser.add_argument('--flow_threshold',     type=float, default=None,
                    help="Override flow_threshold. Default: from CHANNEL_CONFIGS.")
parser.add_argument('--cellprob_threshold', type=float, default=None,
                    help="Override cellprob_threshold. Default: from CHANNEL_CONFIGS.")
parser.add_argument('--use_gpu',            action='store_true',
                    help="Use GPU if available.")
parser.add_argument('--batch_size',         type=int,   default=1,
                    help="Number of slices to process per CellPose call (default: 1).")
parser.add_argument('--plot_qc',            action='store_true',
                    help="Save QC overlay images.")
parser.add_argument('--save_flows',         action='store_true',
                    help="Save flow magnitude TIFF alongside label masks.")
parser.add_argument('--min_size',           type=int,   default=15,
                    help="Minimum cell size in pixels (default: 15).")
parser.add_argument('--slice_filter_yaml',  type=str,   default=None,
                    help="Path to slice_filter.yaml (default: <DATASPACE>/slice_filter.yaml).")
args = parser.parse_args()

TARGET_CORE = args.core_name
CH_NAME     = args.channel
CFG         = CHANNEL_CONFIGS[CH_NAME]

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
OUTPUT_FOLDER     = os.path.join(config.DATASPACE, f"CellPose_{CH_NAME}", TARGET_CORE)
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


def normalise_channel(img: np.ndarray) -> np.ndarray:
    """
    Percentile stretch to float32 [0, 1] for CellPose v4 input.
    We do our own robust stretch (normalize=False passed to eval) so dim
    mIF channels are not dominated by background.

    Percentiles are computed on foreground pixels only (img > 0), which
    excludes zero-padded canvas borders produced by conform_slice / stitching.
    Using subsampled pixels (e.g. [::2,::2]) or including background zeros
    skews p_lo/p_hi and causes CellPose's internal tiles to be normalised
    inconsistently, producing a visible checkerboard in the probability map.
    """
    img_f = img.astype(np.float32)
    fg    = img_f[img_f > 0]
    if fg.size == 0:
        return np.zeros_like(img_f)
    p_lo, p_hi = np.percentile(fg, (0.5, 99.5))
    if p_hi <= p_lo:
        return np.zeros_like(img_f)
    return np.clip((img_f - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def build_cellpose_input(arr: np.ndarray, ch_idx: int,
                          nuclear_idx: int | None) -> np.ndarray:
    """
    Build input array for CellPose v4.

    Always returns a single-channel (H, W) float32 array.  nuclear_idx is
    intentionally ignored — all channels in CHANNEL_CONFIGS have nuclear_idx=None.

    Rationale: mIF DAPI signal is inconsistent across slices and adds noise
    rather than signal when segmenting membranous markers like CD3/CD163.
    Single-channel input also avoids the (H, W, 2) path where a dim DAPI
    channel can confuse cyto3's cyto/nuclear boundary estimation.
    """
    return normalise_channel(arr[ch_idx])   # (H, W) float32


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

def save_qc_plot(raw_img: np.ndarray, masks: np.ndarray,
                 flows_mag: np.ndarray | None,
                 slice_id: int, n_cells: int, out_path: str):
    """
    3-panel QC plot:
      Panel 1 — raw channel (percentile-stretched)
      Panel 2 — label mask overlay (random colours)
      Panel 3 — flow magnitude (if available)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = 3 if flows_mag is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Panel 1: raw
    p_lo, p_hi = np.percentile(raw_img, (0.5, 99.5))
    display    = np.clip((raw_img.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1), 0, 1)
    axes[0].imshow(display, cmap='gray', interpolation='nearest')
    axes[0].set_title(f"Raw {CH_NAME}  (ID {slice_id})", fontsize=11)
    axes[0].axis('off')

    # Panel 2: mask overlay — cycle through tab20 so even sparse detections
    # are clearly visible regardless of total object count.
    axes[1].imshow(display, cmap='gray', interpolation='nearest')
    if n_cells > 0:
        # Modulo-20 cycling: label 0 (background) stays 0, labels 1-N cycle
        cycled = np.where(masks > 0, (masks % 19) + 1, 0).astype(np.uint8)
        axes[1].imshow(cycled, cmap='tab20', vmin=0, vmax=20,
                       interpolation='nearest', alpha=0.6)
    axes[1].set_title(f"Masks — {n_cells} objects", fontsize=11)
    axes[1].axis('off')

    # Panel 3: flows
    if flows_mag is not None and n_panels == 3:
        im = axes[2].imshow(flows_mag, cmap='hot', interpolation='nearest')
        axes[2].set_title("Flow magnitude", fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)

    fig.suptitle(
        f"CellPose QC — {TARGET_CORE}  {CH_NAME}  slice ID {slice_id}  "
        f"diameter={DIAMETER}  flow_thresh={FLOW_THRESHOLD}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT ONE SLICE
# ─────────────────────────────────────────────────────────────────────────────

def segment_slice(arr: np.ndarray, slice_id: int) -> dict:
    """
    Run CellPose on one slice and write outputs.

    Returns a stats dict for the summary CSV.
    """
    t0 = time.time()

    ch_idx      = CFG['idx']
    nuclear_idx = CFG['nuclear_idx']
    model       = get_model(CFG['model'], args.use_gpu)

    # Build input image — (H, W) or (H, W, 2) float32 [0,1] for CellPose v4
    cp_input = build_cellpose_input(arr, ch_idx, nuclear_idx)

    # ── Run CellPose ─────────────────────────────────────────────────────────
    # v4 API changes vs v3:
    #   - `channels` argument removed — shape encodes channel assignment
    #   - `tile` / `tile_overlap` control tiling for large images
    #   - masks may exceed uint16 range → save as uint32
    #
    # Tiling is critical for 6000+ px images: without it CellPose downsizes
    # the entire image to its internal resolution and loses all small cells.
    # tile=True splits into overlapping tiles at the native resolution,
    # segments each, then stitches — giving correct cell diameters throughout.
    # ─────────────────────────────────────────────────────────────────────────
    try:
        masks, flows, styles = model.eval(
            cp_input,
            diameter           = DIAMETER,
            flow_threshold     = FLOW_THRESHOLD,
            cellprob_threshold = CELLPROB_THRESHOLD,
            min_size           = args.min_size,
            do_3D              = False,
            batch_size         = args.batch_size,
            normalize          = False,   # we normalised above
            tile_overlap       = 0.1,     # 10% overlap avoids seam artefacts
        )
    except Exception as exc:
        logger.error(f"CellPose eval failed for slice {slice_id}: {exc}")
        return dict(
            slice_id=slice_id, n_cells=0, mean_area_px=0.0,
            median_area_px=0.0, total_area_px=0, runtime_s=0.0, status="FAILED",
        )

    # Use uint32 — large tissue sections routinely exceed 65535 cells
    masks   = masks.astype(np.uint32)
    n_cells  = int(masks.max())
    runtime  = time.time() - t0

    # ── Cell size statistics ──────────────────────────────────────────────────
    if n_cells > 0:
        # bincount is orders of magnitude faster than iterating per label
        areas       = np.bincount(masks.ravel())[1:]   # index 0 = background
        mean_area   = float(np.mean(areas))
        median_area = float(np.median(areas))
        total_area  = int(areas.sum())
    else:
        mean_area = median_area = total_area = 0.0

    logger.info(
        f"  Slice ID {slice_id:03d}: {n_cells} objects | "
        f"mean area={mean_area:.0f}px  median={median_area:.0f}px | "
        f"{runtime:.1f}s"
    )

    # ── Save label mask ───────────────────────────────────────────────────────
    base      = f"TMA_{slice_id:03d}_{CH_NAME}"
    mask_path = os.path.join(OUTPUT_FOLDER, f"{base}_cp_masks.tif")
    tifffile.imwrite(mask_path, masks.astype(np.uint32), compression='deflate')

    # ── Save probability map ──────────────────────────────────────────────────
    # flows[2] is the cell probability map in CellPose v3/v4 — but only when
    # it is a 2-D spatial array matching the input image shape.  In some v4
    # builds flows[2] can be a 1-D style embedding; guard against that.
    for _fi in range(2, min(len(flows), 5)):
        if flows[_fi] is None:
            continue
        _arr = np.array(flows[_fi])
        if _arr.ndim == 2 and _arr.shape == masks.shape:
            prob_map  = _arr.astype(np.float32)
            prob_path = os.path.join(OUTPUT_FOLDER, f"{base}_cp_probs.tif")
            tifffile.imwrite(prob_path, prob_map, compression='deflate')
            logger.info(f"  Prob map saved from flows[{_fi}] shape={prob_map.shape}")
            break
    else:
        logger.warning(f"  No valid 2-D probability map found in flows — skipping probs.tif")

    # ── Save flow magnitude (optional) ────────────────────────────────────────
    flows_mag = None
    if args.save_flows and len(flows) > 1 and flows[1] is not None:
        xy        = np.array(flows[1])   # (2, H, W) or (H, W, 2)
        if xy.ndim == 3 and xy.shape[0] == 2:
            xy = np.moveaxis(xy, 0, -1)  # → (H, W, 2)
        flows_mag = np.sqrt(xy[..., 0]**2 + xy[..., 1]**2).astype(np.float32)
        flow_path = os.path.join(OUTPUT_FOLDER, f"{base}_cp_flows.tif")
        tifffile.imwrite(flow_path, flows_mag, compression='deflate')

    # ── QC plot ───────────────────────────────────────────────────────────────
    if args.plot_qc:
        qc_path = os.path.join(QC_DIR, f"{base}_qc.png")
        try:
            save_qc_plot(
                raw_img   = arr[ch_idx],
                masks     = masks,
                flows_mag = flows_mag,
                slice_id  = slice_id,
                n_cells   = n_cells,
                out_path  = qc_path,
            )
        except Exception as exc:
            logger.warning(f"QC plot failed for slice {slice_id}: {exc}")

    # ── CD31-specific: binary vessel-area mask ────────────────────────────────
    if CH_NAME == 'CD31' and n_cells > 0:
        vessel_binary = (masks > 0).astype(np.uint8) * 255
        vessel_path   = os.path.join(OUTPUT_FOLDER, f"{base}_vessel_binary.tif")
        tifffile.imwrite(vessel_path, vessel_binary, compression='deflate')
        vessel_frac   = float(vessel_binary.mean()) / 255.0
        logger.info(f"  CD31 vessel area: {vessel_frac*100:.2f}% of image")

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
    logger.info(f"CellPose segmentation — core={TARGET_CORE}  channel={CH_NAME}")
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
                            f"{TARGET_CORE}_cellpose_{CH_NAME}_stats.csv")
    df.to_csv(csv_path, index=False)

    n_ok    = int((df['status'] == 'OK').sum())
    n_empty = int((df['status'] == 'EMPTY').sum())
    n_fail  = int(df['status'].isin(['FAILED', 'CRASHED']).sum())
    logger.info(
        f"Done. OK={n_ok}  EMPTY={n_empty}  FAILED/CRASHED={n_fail} | "
        f"Stats → {csv_path}"
    )

    if CH_NAME == 'CD31':
        logger.info(
            "CD31 note: label count = vessel segments, not individual cells. "
            "Consider using *_vessel_binary.tif for area-based vessel quantification."
        )


if __name__ == "__main__":
    main()