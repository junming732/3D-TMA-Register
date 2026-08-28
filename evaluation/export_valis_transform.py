"""
export_valis_transform.py
====================
One-time-per-core export: pulls the per-slide transform data (M, bk_dxdy,
and the shapes needed to correctly scale it) out of a Valis registrar
pickle, and saves ONE small .npz per z-slice — the same "boring, load with
plain numpy" shape RomaV2's deformation .npz files already have.

Why this exists
----------------
raw_space_transform.py needs each pipeline's transform data as small,
dependency-light .npz files it can just np.load(). It deliberately never
imports `valis` itself, since that pulls in torch/kornia/pyvips/a JVM
bridge, none of which are needed for the actual coordinate math. This
script is the one place that DOES import valis, run ONCE per core in your
real environment, so nothing downstream (qc_reference.py, or anything else
in the matching pipeline) has to.

This does NOT re-run registration or touch the registrar in any way — it
only reads the already-computed Slide objects out of the pickle
valis_register_core2.py already produced.

Usage
-----
    python export_valis_transform.py \\
        --core_name Core_09 \\
        --work_output_dir VALIS_Filter_Eval
"""

import os
import sys
import glob
import argparse
import logging

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config
from landmark_accuracy_common import get_slice_number

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--core_name', type=str, required=True)
parser.add_argument('--work_output_dir', type=str, required=True,
                    help='Folder under DATASPACE holding VALIS registration output '
                         '(e.g. VALIS_Filter_Eval — must match registration.valis.'
                         'output_dir_name in config.yaml). No default on purpose: '
                         'this repo has multiple similarly-named Valis experiment '
                         'folders (VALIS_Baseline_Eval, VALIS_Filter_Eval, '
                         'VALIS_Registered) and a wrong default here would silently '
                         'export the wrong experiment\'s transforms.')
parser.add_argument('--output_subdir', type=str, default='point_transforms',
                    help='Subfolder (under <core>/<core>/) to save the exported .npz '
                         'files into (default: point_transforms).')
args = parser.parse_args()

TARGET_CORE = args.core_name

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — mirrors valis_accuracy_landmarks.py exactly
# ─────────────────────────────────────────────────────────────────────────────
WORK_OUTPUT      = os.path.join(config.DATASPACE, args.work_output_dir)
OUTPUT_FOLDER    = os.path.join(WORK_OUTPUT, TARGET_CORE, TARGET_CORE)  # Valis nests by core twice
TRANSFORM_OUTPUT = os.path.join(OUTPUT_FOLDER, args.output_subdir)
os.makedirs(TRANSFORM_OUTPUT, exist_ok=True)

PICKLE_PATH = os.path.join(OUTPUT_FOLDER, "data", f"{TARGET_CORE}.pickle")
if not os.path.isfile(PICKLE_PATH):
    candidates = glob.glob(os.path.join(OUTPUT_FOLDER, "data", "*.pickle"))
    if candidates:
        PICKLE_PATH = candidates[0]
        logger.warning(f"Using pickle found at: {PICKLE_PATH}")
    else:
        logger.error(f"No pickle file found in {os.path.join(OUTPUT_FOLDER, 'data')}")
        sys.exit(1)

logger.info(f"Core    : {TARGET_CORE}")
logger.info(f"Pickle  : {PICKLE_PATH}")
logger.info(f"Output  : {TRANSFORM_OUTPUT}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD REGISTRAR
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading pickled VALIS registrar…")
from valis import registration
registrar = registration.load_registrar(PICKLE_PATH)
logger.info("  Registrar loaded successfully.")

# Same slice_idx <-> Slide mapping as valis_accuracy_landmarks.py, so exported
# files line up with everything else that already reads this registrar.
idx_to_slide = {}
for name, slide_obj in registrar.slide_dict.items():
    tma_num = get_slice_number(name)
    slice_idx = tma_num - 1  # 0-based
    idx_to_slide[slice_idx] = slide_obj

logger.info(f"Registrar contains {len(idx_to_slide)} slides.")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT ONE .npz PER SLIDE
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_ATTRS = ('M', 'bk_dxdy', 'processed_img_shape_rc', 'reg_img_shape_rc',
                   'aligned_slide_shape_rc', 'slide_dimensions_wh')

n_ok, n_skip = 0, 0
for slice_idx in sorted(idx_to_slide.keys()):
    slide_obj = idx_to_slide[slice_idx]
    slide_name = getattr(slide_obj, 'src_f', f'slice_idx={slice_idx}')

    missing = [a for a in REQUIRED_ATTRS if not hasattr(slide_obj, a)]
    if missing:
        logger.error(f"  slice_idx={slice_idx:02d} ({slide_name}): missing attribute(s) "
                     f"{missing} on this Slide object — your installed valis version's "
                     f"API differs from what this script expects. Run the attribute "
                     f"self-check in this script's docstring to find the right names, "
                     f"then update REQUIRED_ATTRS / the extraction below. Skipping.")
        n_skip += 1
        continue

    try:
        M = np.asarray(slide_obj.M, dtype=np.float64)
        bk_dxdy = np.asarray(slide_obj.bk_dxdy, dtype=np.float64)   # (2, H, W)
        processed_shape_rc = tuple(int(v) for v in slide_obj.processed_img_shape_rc)
        registered_shape_rc = tuple(int(v) for v in slide_obj.reg_img_shape_rc)
        raw_full_res_shape_rc = tuple(int(v) for v in np.array(slide_obj.slide_dimensions_wh[0])[::-1])
        registered_full_res_shape_rc = tuple(int(v) for v in slide_obj.aligned_slide_shape_rc)
    except Exception as e:
        logger.error(f"  slice_idx={slice_idx:02d} ({slide_name}): failed while reading "
                     f"attributes ({type(e).__name__}: {e}). Skipping.")
        n_skip += 1
        continue

    if bk_dxdy.ndim != 3 or bk_dxdy.shape[0] != 2:
        logger.error(f"  slice_idx={slice_idx:02d} ({slide_name}): bk_dxdy has unexpected "
                     f"shape {bk_dxdy.shape} (expected (2, H, W)). Skipping.")
        n_skip += 1
        continue

    out_path = os.path.join(TRANSFORM_OUTPUT, f"{TARGET_CORE}_Z{slice_idx:03d}_valis_transform.npz")
    np.savez(
        out_path,
        M=M,
        bk_dx=bk_dxdy[0],
        bk_dy=bk_dxdy[1],
        processed_shape_rc=np.array(processed_shape_rc),
        registered_shape_rc=np.array(registered_shape_rc),
        raw_full_res_shape_rc=np.array(raw_full_res_shape_rc),
        registered_full_res_shape_rc=np.array(registered_full_res_shape_rc),
    )
    logger.info(f"  slice_idx={slice_idx:02d}  ({os.path.basename(str(slide_name))})  -> {out_path}")
    n_ok += 1

logger.info("=" * 60)
logger.info(f"Done. Exported: {n_ok}  Skipped: {n_skip}")
logger.info(f"Output: {TRANSFORM_OUTPUT}/")
if n_skip:
    logger.info("Some slides were skipped — see errors above. Cell matching for those "
                "z-slices will fall back to being unavailable until re-exported.")
logger.info("=" * 60)