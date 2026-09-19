"""Fast, crop-based parameter sweep for CellPose DAPI nuclear segmentation.

Executes CellPose on a defined image sub-region (crop) across a grid of
`flow_threshold` and `cellprob_threshold` values. Optionally sweeps `model`,
`diameter`, and `min_size`. The module renders an image grid for each
(model, diameter, min_size) combination, where rows represent `cellprob_threshold`
and columns represent `flow_threshold`. Each panel displays the raw DAPI crop
with a yellow outline traced around predicted masks.

Example:
    Standard execution with an auto-centered crop:
    $ python cellpose_param_sweep.py \
        --core_name Core_19 \
        --slice_id 5 \
        --crop_size 400 \
        --models cpsam \
        --diameters 14 20 \
        --min_sizes 15 30 50 \
        --flow_thresholds 0.3 0.4 0.6 0.8 \
        --cellprob_thresholds -3 -1.5 0 1.5 \
        --use_gpu

    Execution targeting specific region coordinates:
    $ python cellpose_param_sweep.py \
        --core_name Core_19 \
        --slice_id 5 \
        --crop 1200,1800,2400,3000 \
        --use_gpu

Notes:
    Output assets are generated in the target dataspace directory:
    <DATASPACE>/CellPose_ParamSweep/<CORE_NAME>/
    |-- <core>_slice<NNN>_model-<model>_diam-<diam>_min<N>_sweep.png
    |-- <core>_slice<NNN>_model-<model>_diam-<diam>_min<N>_panels/
    |   |-- cellprob+0.00_flow0.40_n340.png
    |   |-- cellprob+0.00_flow0.80_n424.png
    |-- <core>_slice<NNN>_sweep_summary.csv

    Use the `--no_save_panels` flag to skip individual panel PNG generation.

 """

import os
import sys
import re
import glob
import time
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import tifffile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = os.path.join(
    config.DATASPACE, 'model_weights', 'cellpose'
)
os.makedirs(os.environ['CELLPOSE_LOCAL_MODELS_PATH'], exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DAPI_CHANNEL_IDX = 0
PIXEL_SIZE_XY_UM = 0.4961
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Crop-based CellPose flow_threshold / cellprob_threshold sweep with DAPI+boundary overlay grids.'
)
parser.add_argument('--core_name', type=str, required=True)
parser.add_argument('--slice_id', type=int, default=None,
                    help="TMA slice number to use. Default: first slice found for this core.")
parser.add_argument('--crop', type=str, default=None,
                    help="Explicit crop 'y0,y1,x0,x1' in full-slice pixel coords. "
                         "Overrides --crop_size / --center if given.")
parser.add_argument('--crop_size', type=int, default=400,
                    help="Size (px) of an auto-centered square crop, if --crop not given. "
                         "Default: 400 (chosen for fast iteration; not tied to any production "
                         "tile size). Pass --crop_size 768 to match test_3d_cellpose.py's "
                         "XY_TILE if you specifically want to check tile-boundary artifacts.")
parser.add_argument('--center', type=str, default=None,
                    help="'y,x' center for the auto crop. Default: center of the slice.")
parser.add_argument('--models', type=str, nargs='+', default=['cpsam'],
                    help="CellPose pretrained_model(s) to sweep, e.g. --models cpsam. "
                         "Default: cpsam. 'nuclei' is NOT a working default here — confirmed "
                         "on this cellpose install that it fails to load and silently falls "
                         "back to cpsam anyway (see get_model()'s mismatch check). Only pass "
                         "'nuclei' if you want to re-confirm that on a new environment.")
parser.add_argument('--diameters', type=float, nargs='+', default=[14.0],
                    help="Diameter(s) in px to sweep. Pass 0 to mean 'None' (auto-estimate). Default: 14.")
parser.add_argument('--flow_thresholds', type=float, nargs='+',
                    default=[0.3, 0.4, 0.6, 0.8],
                    help="flow_threshold values to sweep (columns of the grid).")
parser.add_argument('--cellprob_thresholds', type=float, nargs='+',
                    default=[-3.0, -1.5, 0.0, 1.5],
                    help="cellprob_threshold values to sweep (rows of the grid).")
parser.add_argument('--min_sizes', type=int, nargs='+', default=[15],
                    help="min_size value(s) in px to sweep — same-priority axis as "
                         "flow_threshold/cellprob_threshold, not a fixed setting. "
                         "Same nominal number means very different things between your "
                         "2D and 3D scripts (see conversation); worth sweeping explicitly "
                         "rather than trusting either script's current default. "
                         "Try e.g. --min_sizes 15 30 50 80 and watch the histogram's left "
                         "tail — too high silently truncates real small nuclei, too low "
                         "leaves a pile-up of fragments near the low end.")
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--save_panels', dest='save_panels', action='store_true', default=True,
                    help="Also save each grid cell as its own full-res PNG, for side-by-side "
                         "comparison outside the grid. Default: on.")
parser.add_argument('--no_save_panels', dest='save_panels', action='store_false',
                    help="Skip individual panel PNGs, only produce the combined grid.")
parser.add_argument('--out_dir', type=str, default=None,
                    help="Default: <DATASPACE>/CellPose_ParamSweep/<core_name>/")
args = parser.parse_args()

OUT_DIR = args.out_dir or os.path.join(config.DATASPACE, "CellPose_ParamSweep", args.core_name)
os.makedirs(OUT_DIR, exist_ok=True)

N_COMBOS_PER_GRID = len(args.flow_thresholds) * len(args.cellprob_thresholds)
N_GRIDS = len(args.models) * len(args.diameters)
logger.info(
    f"Sweep plan: {len(args.models)} model(s) x {len(args.diameters)} diameter(s) "
    f"= {N_GRIDS} grid(s), each {len(args.cellprob_thresholds)} rows x "
    f"{len(args.flow_thresholds)} cols = {N_COMBOS_PER_GRID} panels "
    f"({N_GRIDS * N_COMBOS_PER_GRID} CellPose calls total)."
)
if N_COMBOS_PER_GRID > 30:
    logger.warning(
        f"{N_COMBOS_PER_GRID} panels per grid is a lot — the PNG may be large/slow to "
        f"inspect. Consider narrowing --flow_thresholds / --cellprob_thresholds."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SLICE LOADING (self-contained; mirrors cellpose_segmentation.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename: str) -> int:
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def find_slice_file(core_name: str, slice_id: int | None) -> str:
    input_folder = os.path.join(DATA_BASE_PATH, core_name)
    if not os.path.exists(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        sys.exit(1)
    raw_files = sorted(glob.glob(os.path.join(input_folder, "*.ome.tif")), key=get_slice_number)
    if not raw_files:
        logger.error(f"No .ome.tif files in {input_folder}")
        sys.exit(1)
    if slice_id is None:
        chosen = raw_files[0]
        logger.info(f"No --slice_id given; using first slice found: {os.path.basename(chosen)}")
        return chosen
    for f in raw_files:
        if get_slice_number(f) == slice_id:
            return f
    logger.error(f"Slice ID {slice_id} not found in {input_folder}")
    sys.exit(1)


def load_slice(filepath: str) -> np.ndarray:
    arr = tifffile.imread(filepath)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    return arr  # (C, H, W)


def extract_channel(arr: np.ndarray, ch_idx: int) -> np.ndarray:
    return arr[ch_idx].astype(np.float32)  # (H, W) float32


def get_crop_bounds(img_shape: tuple[int, int], crop_arg: str | None,
                    crop_size: int, center_arg: str | None) -> tuple[int, int, int, int]:
    h, w = img_shape
    if crop_arg:
        y0, y1, x0, x1 = (int(v) for v in crop_arg.split(","))
    else:
        if center_arg:
            cy, cx = (int(v) for v in center_arg.split(","))
        else:
            cy, cx = h // 2, w // 2
        half = crop_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        logger.error(f"Invalid/empty crop bounds after clamping to image shape {img_shape}: "
                     f"y[{y0}:{y1}] x[{x0}:{x1}]")
        sys.exit(1)
    return y0, y1, x0, x1


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (cached per model_type so we don't reload between combos)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}

def get_model(model_name: str, use_gpu: bool):
    """
    Loads a CellPose model and VERIFIES what actually got loaded.

    cellpose >=4.0.1 ignores the `model_type=` kwarg entirely (silently falls
    back to cpsam) and ViT-based CellposeModel likely cannot even load the
    old ResUNet 'nuclei' checkpoint. Rather than trust the requested name,
    this checks model.pretrained_model after load and loudly warns if it
    doesn't match what was asked for, so a silent fallback (like the one
    that was happening before this fix) can't go unnoticed again.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    from cellpose import models
    gpu = use_gpu
    try:
        import torch
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA unavailable — running on CPU.")
            gpu = False
    except ImportError:
        gpu = False

    logger.info(f"Requesting CellPose model '{model_name}' (gpu={gpu})...")
    model = models.CellposeModel(pretrained_model=model_name, gpu=gpu)

    actual_path = getattr(model, 'pretrained_model', None)
    actual_name = os.path.basename(str(actual_path)) if actual_path else "unknown"
    if model_name.lower() not in actual_name.lower():
        logger.warning(
            f"*** MODEL MISMATCH *** requested '{model_name}' but CellPose actually "
            f"loaded '{actual_name}'. Your installed cellpose version likely doesn't "
            f"support '{model_name}' (either the kwarg is ignored, or the checkpoint "
            f"isn't compatible with the current architecture). All results below/for "
            f"this combo reflect '{actual_name}', not '{model_name}' — panel titles, "
            f"filenames, and the summary CSV are labelled with the ACTUAL model."
        )
    else:
        logger.info(f"Confirmed loaded model: '{actual_name}'")

    _model_cache[model_name] = (model, actual_name)
    return _model_cache[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# ONE COMBO
# ─────────────────────────────────────────────────────────────────────────────

def run_one(model, img_crop: np.ndarray, diameter: float | None,
           flow_threshold: float, cellprob_threshold: float, min_size: int) -> dict:
    t0 = time.time()
    try:
        masks, flows, styles = model.eval(
            img_crop,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            do_3D=False,
            normalize=True,
            tile_overlap=0.1,
        )
    except Exception as exc:
        logger.error(f"CellPose eval failed (diam={diameter}, flow={flow_threshold}, "
                     f"cellprob={cellprob_threshold}): {exc}")
        return dict(masks=np.zeros_like(img_crop, dtype=np.uint32), n_cells=0,
                    mean_diam_px=0.0, runtime_s=0.0, status="FAILED")

    masks = masks.astype(np.uint32)
    n_cells = int(masks.max())
    if n_cells > 0:
        areas = np.bincount(masks.ravel())[1:]
        diam_px = 2.0 * np.sqrt(areas / np.pi)
        mean_diam_px = float(np.mean(diam_px))
    else:
        diam_px = np.array([])
        mean_diam_px = 0.0
    return dict(masks=masks, n_cells=n_cells, mean_diam_px=mean_diam_px, diam_px=diam_px,
               runtime_s=round(time.time() - t0, 2), status="OK" if n_cells > 0 else "EMPTY")


# ─────────────────────────────────────────────────────────────────────────────
# GRID PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_stretch(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    fg = img[img > 0]
    p_lo, p_hi = np.percentile(fg, (lo, hi)) if fg.size > 0 else (0, 1)
    return np.clip((img.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)


YELLOW = (1.0, 0.92, 0.0, 1.0)  # classic CellPose GUI outline colour


def _draw_yellow_outline(ax, display: np.ndarray, masks: np.ndarray, n_cells: int) -> None:
    """Shared rendering: raw DAPI + single-colour dilated yellow boundary.
    Used identically by both the combined grid and the individual panel PNGs
    so the two views are visually consistent with each other."""
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation

    ax.imshow(display, cmap='gray', interpolation='nearest')
    if n_cells > 0:
        boundary = find_boundaries(masks, mode='inner')
        boundary = binary_dilation(boundary, iterations=1)
        overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
        overlay[boundary] = YELLOW
        ax.imshow(overlay, interpolation='nearest')


def save_individual_panel(display: np.ndarray, result: dict, core_name: str, slice_id: int,
                          actual_model_name: str, diam_label: str, min_size: int,
                          flow: float, cellprob: float, out_path: str) -> None:
    """One combo, one full-resolution PNG — image + diameter distribution side by side,
    for flipping through / side-by-side comparison outside the grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={'width_ratios': [1, 1]})

    # ── Left: segmentation ──────────────────────────────────────────────────
    _draw_yellow_outline(ax_img, display, result['masks'], result['n_cells'])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"n={result['n_cells']}  mean_d={result['mean_diam_px']:.1f}px", fontsize=10)

    # ── Right: per-nucleus diameter distribution (µm) ───────────────────────
    # Shaded band = typical mammalian nucleus diameter range (~5-10 um), for a
    # quick visual sanity check on whether the population is centred where
    # real nuclei should be, or skewed by merged/fragmented masks.
    if result['n_cells'] > 0:
        diam_um = result['diam_px'] * PIXEL_SIZE_XY_UM
        ax_hist.axvspan(5, 10, color='#2ecc71', alpha=0.12, label='typical mammalian\nnucleus (5-10 um)')
        ax_hist.hist(diam_um, bins=30, color='#3498db', edgecolor='white', linewidth=0.5)
        mean_um = float(np.mean(diam_um))
        ax_hist.axvline(mean_um, color='#e74c3c', linestyle='--', linewidth=1.5,
                        label=f'mean = {mean_um:.1f} um')
        ax_hist.set_xlabel('Nucleus diameter (um)', fontsize=9)
        ax_hist.set_ylabel('Count', fontsize=9)
        ax_hist.legend(fontsize=8, loc='upper right')
    else:
        ax_hist.text(0.5, 0.5, 'No nuclei detected', ha='center', va='center',
                     transform=ax_hist.transAxes, fontsize=11, color='gray')
        ax_hist.set_xticks([]); ax_hist.set_yticks([])
    ax_hist.set_title('Diameter distribution', fontsize=10)

    fig.suptitle(
        f"{core_name}  slice {slice_id}  |  model={actual_model_name}  diameter={diam_label}  min_size={min_size}px  |  "
        f"flow_threshold={flow:g}  cellprob_threshold={cellprob:g}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_sweep_grid(core_name: str, slice_id: int, requested_model_name: str, actual_model_name: str,
                     diameter: float | None, min_size: int, dapi_crop: np.ndarray, flow_thresholds: list,
                     cellprob_thresholds: list, model, out_path: str,
                     summary_rows: list, save_panels: bool, panels_dir: str | None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    display = _percentile_stretch(dapi_crop)
    n_rows, n_cols = len(cellprob_thresholds), len(flow_thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows), squeeze=False)

    diam_label = "auto" if diameter is None else f"{diameter:g}px"

    if save_panels:
        os.makedirs(panels_dir, exist_ok=True)

    for ri, cellprob in enumerate(cellprob_thresholds):
        for ci, flow in enumerate(flow_thresholds):
            ax = axes[ri][ci]
            result = run_one(model, dapi_crop, diameter, flow, cellprob, min_size)
            _draw_yellow_outline(ax, display, result['masks'], result['n_cells'])

            ax.set_title(
                f"n={result['n_cells']}  mean_d={result['mean_diam_px']:.1f}px",
                fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([])

            if save_panels:
                # Sign-explicit, zero-padded filename so a plain alphabetical file
                # listing already sorts in cellprob-then-flow order, matching the grid.
                panel_name = (
                    f"cellprob{cellprob:+05.2f}_flow{flow:04.2f}_n{result['n_cells']}.png"
                )
                save_individual_panel(
                    display=display, result=result, core_name=core_name, slice_id=slice_id,
                    actual_model_name=actual_model_name, diam_label=diam_label, min_size=min_size,
                    flow=flow, cellprob=cellprob,
                    out_path=os.path.join(panels_dir, panel_name),
                )

            if ri == 0:
                ax.annotate(f"flow_thresh={flow:g}", xy=(0.5, 1.18), xycoords='axes fraction',
                           ha='center', fontsize=10, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f"cellprob_thresh\n={cellprob:g}", fontsize=10, fontweight='bold')

            summary_rows.append(dict(
                core_name=core_name, slice_id=slice_id,
                requested_model=requested_model_name, actual_model=actual_model_name,
                diameter_px=diam_label, min_size=min_size,
                flow_threshold=flow, cellprob_threshold=cellprob,
                n_cells=result['n_cells'], mean_diameter_px=round(result['mean_diam_px'], 2),
                runtime_s=result['runtime_s'], status=result['status'],
            ))

    model_label = actual_model_name
    if requested_model_name.lower() not in actual_model_name.lower():
        model_label = f"{actual_model_name}  [requested '{requested_model_name}' — NOT what loaded]"

    fig.suptitle(
        f"{core_name}  slice {slice_id}  |  model={model_label}  diameter={diam_label}  min_size={min_size}px\n"
        f"yellow outline = predicted nucleus boundary — judge tightness against the underlying DAPI signal",
        fontsize=12, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved sweep grid -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fpath = find_slice_file(args.core_name, args.slice_id)
    slice_id = get_slice_number(fpath)
    logger.info(f"Using slice {slice_id}: {fpath}")

    arr = load_slice(fpath)
    dapi_full = extract_channel(arr, DAPI_CHANNEL_IDX)
    del arr

    y0, y1, x0, x1 = get_crop_bounds(dapi_full.shape, args.crop, args.crop_size, args.center)
    dapi_crop = dapi_full[y0:y1, x0:x1]
    logger.info(f"Crop: y[{y0}:{y1}] x[{x0}:{x1}]  ({dapi_crop.shape[1]}x{dapi_crop.shape[0]} px, "
               f"~{dapi_crop.shape[1] * PIXEL_SIZE_XY_UM:.0f}x{dapi_crop.shape[0] * PIXEL_SIZE_XY_UM:.0f} um)")

    summary_rows: list = []
    seen_actual_models: dict = {}  # actual_model_name -> requested_model_name (first one seen)

    for model_name, diameter, min_size in itertools.product(args.models, args.diameters, args.min_sizes):
        diameter_val = None if diameter == 0 else diameter
        model, actual_name = get_model(model_name, args.use_gpu)

        # If two different --models resolve to the same actual checkpoint (e.g. both
        # 'nuclei' and 'cyto' silently fall back to cpsam), skip the redundant grid
        # rather than burning time rendering identical results twice under different names.
        dedupe_key = (actual_name, diameter_val, min_size)
        if dedupe_key in seen_actual_models:
            logger.warning(
                f"Skipping grid for requested='{model_name}': it resolved to the same "
                f"actual model ('{actual_name}') as requested='{seen_actual_models[dedupe_key]}', "
                f"already rendered at this diameter/min_size. No new information to show."
            )
            continue
        seen_actual_models[dedupe_key] = model_name

        diam_tag = "auto" if diameter_val is None else f"{diameter_val:g}"
        out_path = os.path.join(
            OUT_DIR,
            f"{args.core_name}_slice{slice_id:03d}_model-{actual_name}_diam-{diam_tag}_min{min_size}_sweep.png"
        )
        panels_dir = os.path.join(
            OUT_DIR,
            f"{args.core_name}_slice{slice_id:03d}_model-{actual_name}_diam-{diam_tag}_min{min_size}_panels"
        )
        build_sweep_grid(
            core_name=args.core_name, slice_id=slice_id,
            requested_model_name=model_name, actual_model_name=actual_name,
            diameter=diameter_val, min_size=min_size, dapi_crop=dapi_crop,
            flow_thresholds=args.flow_thresholds, cellprob_thresholds=args.cellprob_thresholds,
            model=model, out_path=out_path, summary_rows=summary_rows,
            save_panels=args.save_panels, panels_dir=panels_dir if args.save_panels else None,
        )
        if args.save_panels:
            logger.info(f"  Individual panels -> {panels_dir}/")

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUT_DIR, f"{args.core_name}_slice{slice_id:03d}_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Sweep complete. Summary -> {csv_path}")
    logger.info(
        "Next: pick the (model, diameter, min_size, flow_threshold, cellprob_threshold) row "
        "whose boundary overlay tracks the DAPI signal most tightly AND whose diameter "
        "histogram sits cleanly in the 5-10um band, then hardcode those values into "
        "cellpose_segmentation.py's DAPI_CONFIG and test_3d_cellpose.py's make_eval_kwargs()."
    )


if __name__ == "__main__":
    main()