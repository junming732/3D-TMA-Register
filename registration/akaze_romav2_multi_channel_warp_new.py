"""
Feature registration — AKAZE affine pre-alignment → RoMaV2 dense warp residual.

Pipeline per slice pair:
  L0  AKAZE tissue-masked detection → RANSAC affine pre-alignment
      Reduces global translation/rotation so RoMaV2 only sees residual deformation.
  L1  RoMaV2 dense warp on affine-prealigned images
      Input image controlled by ROMA_MODE (default: CK linear).
      Handles local non-rigid tissue deformation that the affine cannot model.

Why this order:
  RoMaV2 runs its coarse match at a fixed internal resolution (ROMAV2_H/W,
  currently 800×800, plus an 1280×1280 HR refinement pass — see ROMAV2_H_HR/
  W_HR below), so large global shifts (e.g. 387px) push correspondences far
  outside any cell's receptive field, causing mass displacement capping and poor
  confidence maps. Affine pre-alignment collapses the global offset so RoMaV2
  only needs to recover the small residual deformation it is actually designed for.

Fallback hierarchy:
  L0 fails  → run RoMaV2 on raw images (better than identity for moderate shifts)
  L0 succeeds but affine NCC ≤ raw NCC → revert affine to identity; RoMaV2 on raw
  L1 fails  → use affine-only result (L0 output)
  L1 does not improve NCC by WARP_NCC_MIN_IMPROVEMENT over affine → revert to affine
  Both fail → identity (raw moving slice)

NCC is measured at three points for comparison:
  ncc_raw    — raw moving vs fixed (before any alignment)
  ncc_affine — after AKAZE affine (L0 output vs fixed)
  ncc_warp   — after RoMaV2 warp  (L1 output vs fixed)
  All three NCC measurements always use the log-normalised CK channel for
  comparability across ROMA_MODE settings, regardless of what image RoMaV2 consumed.

Deformation maps:
  For each registered slice, a .npz file is saved containing:
    M_affine : (2, 3) float64 — affine matrix applied at L0
    map_x    : (H, W) float32 — full-resolution X remap (L1, or None if warp failed)
    map_y    : (H, W) float32 — full-resolution Y remap (L1, or None if warp failed)
    warp_ok  : bool — whether the RoMaV2 warp succeeded
    akaze_ok : bool — whether AKAZE affine succeeded
    orig_h   : int  — image height
    orig_w   : int  — image width

  These maps can be applied to CellPose segmentation masks (run on the original,
  unregistered images) using the companion script warp_cellpose_masks.py, which
  applies the identical two-step (affine, then remap) transform documented below.

  Warping semantics — the maps describe a FORWARD warp (moving→fixed):
    map_x[y, x] = x-coordinate in the source (moving) image that maps to pixel (x, y)
    map_y[y, x] = y-coordinate in the source (moving) image that maps to pixel (x, y)
  Apply with cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST) using INTER_NEAREST
  for label masks to avoid interpolating integer cell IDs across boundaries.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
import glob
import re
import cv2
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

# Torch setup — must happen before any torch import
os.environ['TORCH_HOME']           = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Mask the GPU completely from this process before PyTorch initializes
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch._dynamo.config.disable = True

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Registration: AKAZE affine pre-alignment → RoMaV2 dense warp.'
)
parser.add_argument('--core_name', type=str, required=True)
parser.add_argument('--registration_mode',  type=str, default='clahe_visual_additive_2ch',
                    choices=['dapi_clahe', 'clahe_visual_additive_2ch',
                             'clahe_visual_additive_3ch', 'clahe_visual_additive_7ch',
                             'clahe_direct_rgb_2ch'],
                    help="Which image is fed to RomaV2. Per the full 20-config parametric "
                         "ROMA_MODE ablation (test_roma_mode_ablation_2.py) — kept 'dapi_clahe' "
                         "(still solid, cheap, VALIS-matching baseline) and added the top-3 by "
                         "real TRE, all CLAHE (skimage.equalize_adapthist) normalized:\n"
                         "  'clahe_visual_additive_2ch' — best real TRE (5.32px). DAPI (white,"
                         " half-weight) + CK (violet, quarter-weight) additively blended with"
                         " hard clipping. Also the highest-coverage option (76.4%%).\n"
                         "  'clahe_visual_additive_3ch' — DAPI + CK + AF, same additive-blend"
                         " recipe. Colors copied verbatim from test_roma_mode_ablation_3.py's"
                         " 'clahe_visual_additive_rgb_3ch_fusion' config (VISUAL_COLORS[0]/[6]/[7]"
                         " there — white/half, violet/quarter, orange/quarter).\n"
                         "  'clahe_visual_additive_7ch' — ties for best real TRE (5.34px) using"
                         " all 7 non-AF markers (DAPI, CD31, GAP43, NFP, CD3, CD163, CK), same"
                         " additive-blend recipe. NOTE: coverage is notably lower (58.6%% vs"
                         " 76.4%% for the 2ch version) — RoMaV2's per-cell confidence gate is"
                         " passing less of the image, so more of the frame falls back to the"
                         " affine-only result here even though TRE at matched landmarks is"
                         " nearly identical to the 2ch option.\n"
                         "  'clahe_direct_rgb_2ch' — DAPI in R, CK in G, B=0, no blending"
                         " (5.37px, 59.1%% coverage). Cheapest option in the top tier.")

args = parser.parse_args()

TARGET_CORE = args.core_name
# ROMA_MODE: controls which image is fed to RoMaV2 only.
# AKAZE (L0) always uses CK regardless of this setting. Tissue mask is
# loaded from the precomputed sibling file and is independent of both.
#   'dapi_clahe'                — RoMaV2 uses DAPI + CLAHE, matching VALIS's default
#                                 fluorescence preprocessing (preprocessing.ChannelGetter)
#                                 exactly. Kept as-is from the original 5-mode set.
#   'clahe_visual_additive_2ch' — DAPI + CK only, CLAHE-normalized, additively blended
#                                 with hard clipping (see prepare_visual_additive_2ch).
#                                 Best real-TRE mode in the ablation; highest coverage too.
#   'clahe_visual_additive_3ch' — DAPI + CK + AF, same additive-blend recipe (see
#                                 prepare_visual_additive_3ch). Colors match
#                                 test_roma_mode_ablation_3.py's VISUAL_COLORS dict exactly
#                                 (white/half, violet/quarter, orange/quarter).
#   'clahe_visual_additive_7ch' — all 7 non-AF markers, same additive-blend recipe (see
#                                 prepare_visual_additive_7ch). Ties on real TRE but has
#                                 notably lower coverage than the 2ch version — use with
#                                 that tradeoff in mind.
#   'clahe_direct_rgb_2ch'      — DAPI in R, CK in G, B=0, no blending (see
#                                 prepare_direct_rgb_2ch).
ROMA_MODE = args.registration_mode

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, f"Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_{ROMA_MODE}")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# Sub-folder where deformation .npz files are written, one per slice pair.
# Filename convention:  <core>_Z<idx>_ID<slice_id>_deformation.npz
DEFORM_FOLDER = os.path.join(OUTPUT_FOLDER, "deformation_maps")

# ─────────────────────────────────────────────────────────────────────────────
# L0: AKAZE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AKAZE_THRESHOLD     = 0.0001  # low threshold — tissue mask is the quality gate
AKAZE_MAX_KEYPOINTS = 20_000  # hard cap after tissue-masked detection to avoid
                               # BFMatcher IMGIDX_ONE assertion on large descriptor sets
LOWE_RATIO          = 0.80
MIN_MATCHES         = 20
MIN_INLIERS         = 6
RANSAC_CONFIDENCE   = 0.995
RANSAC_MAX_ITERS    = 5000
RANSAC_THRESH       = 8.0     # pixels at full resolution
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0

# Adopted from ablation: RANSAC -> USAC_MAGSAC (small, consistent, significant
# NCC gain, ~free) and full-res -> downsampled AKAZE detection (NCC-neutral,
# ~20x faster detection). Downsampled detection + RANSAC/MAGSAC fit happens at
# AKAZE_DOWNSAMPLE_MAX_DIM; the surviving inlier correspondences are then
# rescaled to full resolution and refit via ordinary least squares there, so
# the final affine is still full-resolution-precise despite detecting on a
# coarser image. No-ops (scale=1.0) if the image is already <= this size.
RANSAC_METHOD            = cv2.USAC_MAGSAC
AKAZE_DOWNSAMPLE_MAX_DIM = 1200

# AKAZE detection normally always runs on CK log (prepare_ck), independent of
# ROMA_MODE — see prepare_akaze_gray() below. Core_16 was flagged as
# "unsuitable" by analyse_unsuitable_cores.py: CK log-space AKAZE detection
# didn't find enough inlier matches there, while the 7-channel color-LUT
# composite (prepare_color_lut_fusion, collapsed to gray) did. Cores listed
# here get their L0 AKAZE detection switched to that color-LUT gray image
# instead. This affects ONLY what AKAZE detects keypoints on — RoMaV2's input
# (ROMA_MODE) and every NCC measurement still always use CK log, so results
# stay directly comparable across cores. Add more core names here if the same
# diagnostic turns up other cores where color_lut outperforms CK for AKAZE.
AKAZE_COLOR_LUT_CORES = {"Core_16"}

# ─────────────────────────────────────────────────────────────────────────────
# L1: ROMAV2 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# After affine pre-alignment the residual shift is small, so the displacement
# cap can be much tighter than in the raw-image RoMaV2 script.
# 100px is generous for a residual deformation after affine alignment.
ROMAV2_DEVICE            = 'cpu'  # falls back to 'cpu' automatically if unavailable (see get_romav2_model)
ROMAV2_H                 = 800     # raised from 448 — matches the model's own
ROMAV2_W                 = 800     # "precise" preset LR default
ROMAV2_H_HR              = 1280    # re-enabled — was None, which skipped the HR
ROMAV2_W_HR              = 1280    # refinement pass entirely (see romav2.py match());
                                    # 1280 matches the model's own "precise" preset default.
                                    # Must stay >= ROMAV2_H/W — HR gives the finest
                                    # (patch_size=4) refiner more detail than the coarse
                                    # pass, not less.
WARP_CONFIDENCE_THRESH   = 0.75    # raised from 0.5 — stricter per-cell confidence gate
WARP_MAX_DISPLACEMENT_PX = 80.0    # tightened from 200px — residual after affine should be small
# RoMaV2 warp acceptance criterion — mirrors BSPLINE_NCC_MIN_IMPROVEMENT in the B-spline
# script.  The warp result is accepted only if it improves NCC (relative to the affine
# baseline) by at least this fraction.
# NCC is negative — improvement means becoming more negative.
# Relative improvement = (ncc_affine − ncc_warp) / |ncc_affine|
# e.g. ncc_affine=−0.72, ncc_warp=−0.76 → gain=(−0.72−−0.76)/0.72 = +0.056 → accepted.
# Set to 0.0 to accept any non-negative improvement; raise to be stricter.
WARP_NCC_MIN_IMPROVEMENT = 0.03

# Physical pixel size at full resolution — used only for the NIfTI
# displacement-field export (save_deformation_nifti). Does not affect any
# pixel-space computation elsewhere in the pipeline. Assumes a single
# isotropic value for every core; if different scans/cores have different
# pixel sizes this needs to become per-core rather than a global constant.
PIXEL_SIZE_UM             = 0.4961

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE MASK
# ─────────────────────────────────────────────────────────────────────────────
MASK_MIN_FRAC    = 0.05  # skip RoMaV2 if tissue covers < 5% of canvas

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL / METADATA
# ─────────────────────────────────────────────────────────────────────────────
CK_CHANNEL_IDX       = 6
CHANNEL_NAMES        = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5
MIN_CK_NONZERO_FRAC  = 0.01


if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEFORM_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_slice_filter(yaml_path: str, core_name: str) -> set[int] | None:
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


def get_slice_number(filename: str) -> int:
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def prepare_ck(img_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (norm_lin, norm_log) — both uint8, derived from a single CK channel array.

    norm_lin : linear percentile stretch (0.1–99.9%).
               Used as RoMaV2 input only in 'ck_only' mode.
    norm_log : log-stretch then percentile normalise.
               Used for AKAZE detection and all NCC measurements.
               Always CK-derived regardless of ROMA_MODE.
    """
    img_float = img_arr.astype(np.float32)

    # Linear normalisation — for RoMaV2
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Log normalisation — for AKAZE, NCC
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return norm_lin, norm_log


def clahe_normalize(img_arr: np.ndarray) -> np.ndarray:
    """
    Matches VALIS's default fluorescence preprocessing exactly —
    preprocessing.ChannelGetter.process_image(adaptive_eq=True):
        rescale to [0,1] -> skimage.exposure.equalize_adapthist -> rescale to uint8
    No log/linear percentile stretch first; CLAHE's local contrast
    normalization replaces that step entirely, same as VALIS. Used for
    ROMA_MODE 'dapi_clahe' / 'ck_clahe' — testing whether matching VALIS's
    own preprocessing (not just channel choice) closes the accuracy gap.
    """
    from skimage import exposure
    img_float = img_arr.astype(np.float32)
    img01 = exposure.rescale_intensity(img_float, out_range=(0, 1))
    eq    = exposure.equalize_adapthist(img01)
    return exposure.rescale_intensity(eq, out_range=(0, 255)).astype(np.uint8)

DAPI_CHANNEL_IDX = 0
AF_CHANNEL_IDX   = 7

COLOR_LUT = {
    0: (0,   128, 255),
    1: (51,  255,  51),
    2: (255,  51,  51),
    3: (0,   255, 255),
    4: (255,   0, 255),
    5: (255, 255,   0),
    6: (255, 128,   0),
}

def _prepare_single(img_arr, lo=0.1, hi=99.5):
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (lo, hi))
    return cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

def prepare_color_lut_fusion(img_arr_vol):
    """
    Returns (H, W, 3) uint8 RGB composite for RoMaV2:
    7-channel (AF excluded) weighted-average color-LUT blend.
    """
    h, w = img_arr_vol.shape[1], img_arr_vol.shape[2]
    acc  = np.zeros((h, w, 3), dtype=np.float32)
    n    = len(COLOR_LUT)
    for idx, color in COLOR_LUT.items():
        norm      = _prepare_single(img_arr_vol[idx].astype(np.float32)).astype(np.float32) / 255.0
        color_arr = np.array(color, dtype=np.float32) / 255.0
        acc      += norm[..., None] * color_arr[None, None, :]
    return np.clip(acc / n * 255.0, 0, 255).astype(np.uint8)   # (H, W, 3)


# Modes below all use CLAHE (equalize_adapthist) per-channel normalization.
# Colors are copied verbatim from test_roma_mode_ablation_3.py's VISUAL_COLORS
# dict throughout, so any given channel's tint/weight is identical across
# VISUAL_COLORS_2CH / _3CH / _7CH below.

VISUAL_COLORS_2CH = {
    DAPI_CHANNEL_IDX: np.array([1, 1, 1],   dtype=np.float32) / 2,  # white, half-weight
    CK_CHANNEL_IDX:   np.array([0.5, 0, 1], dtype=np.float32) / 4,  # violet, quarter-weight
}

# DAPI + CK + AF tint set — matches the ablation's 'clahe_visual_additive_rgb_3ch_fusion'
# config. Used only by prepare_visual_additive_3ch.
VISUAL_COLORS_3CH = {
    DAPI_CHANNEL_IDX: np.array([1, 1, 1],   dtype=np.float32) / 2,  # white, half-weight
    CK_CHANNEL_IDX:   np.array([0.5, 0, 1], dtype=np.float32) / 4,  # violet, quarter-weight
    AF_CHANNEL_IDX:   np.array([1, 0.5, 0], dtype=np.float32) / 4,  # orange, quarter-weight
}

# Full 7-marker tint set (all channels except AF), matching the ablation's
# VISUAL_COLORS dict — used only by prepare_visual_additive_7ch.
VISUAL_COLORS_7CH = {
    0: np.array([1, 1, 1],   dtype=np.float32) / 2,
    1: np.array([0, 1, 0],   dtype=np.float32) / 4,
    2: np.array([1, 1, 0],   dtype=np.float32) / 4,
    3: np.array([1, 0, 1],   dtype=np.float32) / 4,
    4: np.array([0, 1, 1],   dtype=np.float32) / 4,
    5: np.array([1, 0, 0],   dtype=np.float32) / 4,
    6: np.array([0.5, 0, 1], dtype=np.float32) / 4,
}


def prepare_visual_additive_2ch(img_arr_vol):
    """
    Returns (H, W, 3) uint8 RGB image for RoMaV2 — ablation's
    'clahe_visual_additive_rgb_2ch_fusion' (best real-TRE mode overall,
    5.32px, and highest coverage of any top-tier mode, 76.4%%).
    DAPI and CK are each CLAHE-normalized, tinted per VISUAL_COLORS_2CH, and
    summed with a hard clip at 1.0.
    """
    dapi = clahe_normalize(img_arr_vol[DAPI_CHANNEL_IDX].astype(np.float32)).astype(np.float32) / 255.0
    ck   = clahe_normalize(img_arr_vol[CK_CHANNEL_IDX].astype(np.float32)).astype(np.float32) / 255.0
    rgb  = dapi[..., None] * VISUAL_COLORS_2CH[DAPI_CHANNEL_IDX][None, None, :] \
         + ck[..., None]   * VISUAL_COLORS_2CH[CK_CHANNEL_IDX][None, None, :]
    rgb  = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def prepare_visual_additive_7ch(img_arr_vol):
    """
    Returns (H, W, 3) uint8 RGB image for RoMaV2 — ablation's
    'clahe_visual_additive_rgb_7ch_fusion' (5.34px real TRE, ties for best,
    but only 58.6%% coverage vs 76.4%% for the 2ch version — RoMaV2's
    per-cell confidence gate passes less of the image here, so more of the
    frame falls back to affine-only even though matched-landmark TRE is
    nearly identical). All 7 non-AF markers, CLAHE-normalized, tinted per
    VISUAL_COLORS_7CH, and summed with a hard clip at 1.0.
    """
    rgb = np.zeros((*img_arr_vol.shape[1:], 3), dtype=np.float32)
    for ch_idx, color in VISUAL_COLORS_7CH.items():
        norm_img = clahe_normalize(img_arr_vol[ch_idx].astype(np.float32)).astype(np.float32) / 255.0
        rgb += norm_img[..., None] * color[None, None, :]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def prepare_visual_additive_3ch(img_arr_vol):
    """
    Returns (H, W, 3) uint8 RGB image for RoMaV2 — ablation's
    'clahe_visual_additive_rgb_3ch_fusion' (DAPI + CK + AF).
    Same additive-blend recipe as prepare_visual_additive_2ch/_7ch: each of
    DAPI, CK, AF is CLAHE-normalized, tinted per VISUAL_COLORS_3CH (colors
    copied verbatim from the ablation — white/half, violet/quarter,
    orange/quarter respectively), and summed with a hard clip at 1.0.
    """
    rgb = np.zeros((*img_arr_vol.shape[1:], 3), dtype=np.float32)
    for ch_idx, color in VISUAL_COLORS_3CH.items():
        norm_img = clahe_normalize(img_arr_vol[ch_idx].astype(np.float32)).astype(np.float32) / 255.0
        rgb += norm_img[..., None] * color[None, None, :]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def prepare_direct_rgb_2ch(img_arr_vol):
    """
    Returns (H, W, 3) uint8 RGB image for RoMaV2 — ablation's
    'clahe_direct_rgb_2ch_fusion' (5.37px real TRE, 59.1%% coverage).
    DAPI CLAHE in the R channel, CK CLAHE in the G channel, B left at 0.
    No blending — cheapest option in the top tier.
    """
    dapi = clahe_normalize(img_arr_vol[DAPI_CHANNEL_IDX].astype(np.float32))
    ck   = clahe_normalize(img_arr_vol[CK_CHANNEL_IDX].astype(np.float32))
    rgb = np.zeros((*dapi.shape, 3), dtype=np.uint8)
    rgb[..., 0] = dapi
    rgb[..., 1] = ck
    return rgb


def prepare_akaze_gray(multichannel_np: np.ndarray, core_name: str) -> np.ndarray:
    """
    Returns the grayscale uint8 image AKAZE (L0) detects keypoints on.

    Defaults to CK log-normalised (prepare_ck's norm_log), matching AKAZE's
    original always-CK behaviour. For cores listed in AKAZE_COLOR_LUT_CORES,
    returns the 7-channel color-LUT composite (prepare_color_lut_fusion)
    collapsed to gray via cv2.cvtColor's RGB2GRAY luma weights instead —
    identical recipe to get_display_and_gray('color_lut', ...) in
    analyse_unsuitable_cores.py, so this is the exact image that was verified
    to succeed there. No further log-stretch is applied on top: each channel
    going into the composite is already log-normalised individually inside
    prepare_color_lut_fusion.
    """
    if core_name in AKAZE_COLOR_LUT_CORES:
        rgb = prepare_color_lut_fusion(multichannel_np)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        _, log = prepare_ck(multichannel_np[CK_CHANNEL_IDX].astype(np.float32))
        return log


def prepare_roma_input(multichannel_np):
    """
    Returns the RoMaV2-input-equivalent image (grayscale (H, W) or RGB
    (H, W, 3) uint8) for an arbitrary multi-channel volume, using whatever
    channel combination ROMA_MODE selects — identical logic to how
    fixed_lin_roma / moving_lin_roma are built from fixed_np / moving_np in
    register_slice. Used to re-derive the RoMaV2-equivalent view of
    affine_np / warp_candidate (already-warped multi-channel volumes) so
    NCC gating can measure the same channel(s) RoMaV2 actually optimizes at
    every stage, not just the raw pre-alignment pair.
    """
    if ROMA_MODE == 'clahe_visual_additive_2ch':
        return prepare_visual_additive_2ch(multichannel_np)
    elif ROMA_MODE == 'clahe_visual_additive_3ch':
        return prepare_visual_additive_3ch(multichannel_np)
    elif ROMA_MODE == 'clahe_visual_additive_7ch':
        return prepare_visual_additive_7ch(multichannel_np)
    elif ROMA_MODE == 'clahe_direct_rgb_2ch':
        return prepare_direct_rgb_2ch(multichannel_np)
    else:  # 'dapi_clahe'
        return clahe_normalize(multichannel_np[DAPI_CHANNEL_IDX].astype(np.float32))


def prepare_ncc_gray_log(img_lin):
    """
    Collapse a RoMaV2-input image (grayscale or RGB uint8, from
    prepare_roma_input or fixed_lin_roma/moving_lin_roma directly) to a
    single-channel, log-stretched, percentile-normalised uint8 image for
    NCC measurement — the same log-space treatment prepare_ck applies to
    CK, run instead on whichever channel(s) RoMaV2 actually saw this run
    (per ROMA_MODE). RGB inputs are collapsed via standard luminance
    weights (cv2.COLOR_RGB2GRAY) — this is a scalar-summary simplification,
    not a per-channel metric, so a fusion mode with one badly-behaved
    channel and two good ones can still average out to a healthy-looking
    NCC. WARP_NCC_MIN_IMPROVEMENT was tuned against CK-log NCC and will
    likely need re-checking against this metric's distribution before
    trusting the acceptance gate at scale.
    """
    if img_lin.ndim == 3:
        gray = cv2.cvtColor(img_lin, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = img_lin.astype(np.float32)
    log_img    = np.log1p(gray)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    return cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

def normalize_pct(img: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """
    Percentile-clip normalise an image to [0, 1] float32 for display.
    Shared by save_registration_plot and generate_qc_montage so both use
    identical normalisation and can't silently drift apart.
    """
    p = np.percentile(img, pct)
    return np.clip(img.astype(np.float32) / (p if p > 0 else 1), 0, 1)


def to_rgb_pil(img):
    """
    Convert a uint8 image to a PIL RGB image for RoMaV2.
    Accepts either:
      - (H, W)    grayscale — duplicated to 3 channels (ck_only mode)
      - (H, W, 3) already RGB — passed through directly (color_lut mode)
    """
    from PIL import Image
    if img.ndim == 2:
        return Image.fromarray(np.stack([img, img, img], axis=-1))
    elif img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img)
    else:
        raise ValueError(f"Unexpected image shape for RoMaV2: {img.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# TISSUE MASK
# ─────────────────────────────────────────────────────────────────────────────
# Masks are no longer derived on-the-fly from the CK channel (see removed
# build_tissue_mask). Instead we load the same precomputed
# '<stem>_tissue_mask.png' sibling files that analyse_unsuitable_cores.py
# uses (see crop_conform_mask_tma.py for how those are produced), so the
# diagnostics and the production pipeline are guaranteed to be scoring
# against the identical mask geometry rather than two independently-tuned
# heuristics that can silently drift apart.
#
# One thing this change requires that the analysis script didn't have to
# deal with: in register_slice, "fixed" is not always a raw slice loaded
# from disk. Only the anchor slice is raw; every other "fixed" reference is
# aligned_vol[i + fixed_offset], which has already been carried through one
# or more affine+warp steps as the pass walks outward from the anchor. A
# precomputed mask file reflects the *raw*, unwarped geometry, so it can
# only be loaded directly for the moving slice (always freshly loaded raw).
# For the fixed slice, the correct mask is the raw mask *warped forward*
# through the same transform chain already baked into aligned_vol — so we
# carry a mask forward step-by-step in lockstep with the pixel data (see
# mask_vol in main()/process_pass), rather than reloading from disk each
# time.


def get_mask_sibling_path(tif_path: str) -> str:
    """
    Given the path to a raw '<stem>.ome.tif' slice, return the path to its
    precomputed tissue mask: '<stem>_tissue_mask.png' (see
    crop_conform_mask_tma.py, which produces these). Handles the double
    '.ome.tif' extension explicitly so the stem isn't truncated to
    '..._tissue_mask.png' vs '...ome_tissue_mask.png' by mistake.
    """
    if tif_path.endswith(".ome.tif"):
        stem = tif_path[:-len(".ome.tif")]
    else:
        stem = os.path.splitext(tif_path)[0]
    return stem + "_tissue_mask.png"


def load_mask_or_none(tif_path: str, shape_hw):
    """
    Load the precomputed tissue mask sibling for a raw slice, if present.

    Returns a uint8 (H, W) array (255 = tissue, 0 = background) whose shape
    matches shape_hw, or None if no mask file exists for this slice, the
    file failed to load, or its shape doesn't match shape_hw (e.g. a stale
    mask left over from before a re-crop/re-conform pass). Callers must
    treat None as "match/measure unmasked" — every downstream consumer here
    (akaze_affine, measure_ncc, romav2_dense_warp) already accepts
    mask=None and behaves as if the whole canvas is tissue.
    """
    mask_path = get_mask_sibling_path(tif_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape_hw:
            return mask
        logger.warning(f"Mask at {mask_path} missing/shape-mismatched — matching unmasked.")
    else:
        logger.warning(f"No precomputed mask for {os.path.basename(tif_path)} — matching unmasked.")
    return None


def warp_mask_forward(mask, M_affine, map_x, map_y, h, w):
    """
    Carry a raw-space tissue mask forward through the same transform chain
    just applied to the pixel data, so it stays valid as the 'fixed' mask
    for the next slice out from the anchor. Returns None if mask is None
    (propagates 'unmasked' status onward rather than fabricating a mask).

    INTER_NEAREST is used throughout (unlike the INTER_LINEAR used for
    pixel data) so the mask stays strictly binary instead of picking up
    fractional edge values.
    """
    if mask is None:
        return None
    warped = cv2.warpAffine(
        mask, M_affine, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if map_x is not None and map_y is not None:
        warped = cv2.remap(
            warped, map_x, map_y,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# NCC MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_ncc(fixed_f32: np.ndarray,
                moving_f32: np.ndarray,
                mask_uint8: np.ndarray) -> float:
    """
    Masked NCC via 0-iteration SimpleITK LBFGSB (evaluation only).
    NCC is negative — more negative = better alignment.
    Returns 0.0 on failure.
    """
    try:
        sitk_f = sitk.GetImageFromArray(fixed_f32)
        sitk_m = sitk.GetImageFromArray(moving_f32)
        reg    = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.10)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(numberOfIterations=0)
        reg.SetInitialTransform(sitk.TranslationTransform(2), inPlace=False)
        if mask_uint8 is not None and mask_uint8.max() > 0:
            reg.SetMetricFixedMask(sitk.GetImageFromArray(mask_uint8))
        reg.Execute(sitk_f, sitk_m)
        return reg.GetMetricValue()
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DEFORMATION MAP I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_deformation_maps(slice_id: str,
                          M_affine: np.ndarray,
                          map_x: np.ndarray | None,
                          map_y: np.ndarray | None,
                          akaze_ok: bool,
                          warp_ok: bool,
                          orig_h: int,
                          orig_w: int) -> str:
    """
    Save the composite deformation for one slice as a compressed .npz.

    The file holds everything needed to warp a CellPose mask (or any label
    image) from the original moving-slice space into the registered fixed space:

        M_affine : (2, 3) float64  — AKAZE affine (identity if AKAZE failed)
        map_x    : (H, W) float32  — full-res X remap from RoMaV2 stage
                                     (identity grid if warp failed)
        map_y    : (H, W) float32  — full-res Y remap from RoMaV2 stage
        akaze_ok : bool
        warp_ok  : bool
        orig_h   : int
        orig_w   : int

    Applying to a mask (see warp_cellpose_masks.py, which implements this):
        Step 1 — affine:  mask_affine = cv2.warpAffine(mask, M_affine, (W, H),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Step 2 — remap:   mask_final  = cv2.remap(mask_affine, map_x, map_y,
                              cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    Returns the path of the saved .npz file.
    """
    # If warp failed, store identity remap so the loading code never has to
    # branch — it always applies both steps in sequence.
    if map_x is None or map_y is None:
        grid_x = np.arange(orig_w, dtype=np.float32)[None, :]
        grid_y = np.arange(orig_h, dtype=np.float32)[:, None]
        map_x, map_y = np.broadcast_to(grid_x, (orig_h, orig_w)).copy(), \
                       np.broadcast_to(grid_y, (orig_h, orig_w)).copy()

    out_path = os.path.join(DEFORM_FOLDER, f"{TARGET_CORE}_{slice_id}_deformation.npz")
    np.savez_compressed(
        out_path,
        M_affine = M_affine.astype(np.float64),
        map_x    = map_x.astype(np.float32),
        map_y    = map_y.astype(np.float32),
        akaze_ok = np.bool_(akaze_ok),
        warp_ok  = np.bool_(warp_ok),
        orig_h   = np.int32(orig_h),
        orig_w   = np.int32(orig_w),
    )
    logger.info(f"[{slice_id}] Deformation map saved → {out_path}")
    return out_path


def save_deformation_nifti(slice_id: str,
                            M_affine: np.ndarray,
                            map_x: np.ndarray | None,
                            map_y: np.ndarray | None,
                            orig_h: int,
                            orig_w: int,
                            pixel_size_um: float = PIXEL_SIZE_UM) -> str:
    """
    Export the same total deformation as save_deformation_maps, but as a
    single composite displacement field NIfTI (.nii.gz) for downstream
    tools that expect ITK/ANTs-style displacement fields rather than a
    two-step (affine + remap) representation.

    Two conventions this locks in — flag if your group's tooling expects
    the opposite:
      1. Physical units (mm), using pixel_size_um — NOT raw pixel deltas.
         ITK/ANTs displacement fields are conventionally physical-space.
      2. "Pull" / backward field: for each voxel in the FIXED image grid,
         the vector points to the corresponding location in the RAW MOVING
         image — i.e. exactly what ITK's own resampling expects when you
         hand it this field to warp the moving image into fixed space.
         This is the composition of the same two steps documented in
         save_deformation_maps (affine, then remap — see also
         warp_cellpose_masks.py), collapsed into one field rather than
         kept as two stages.

    Returns the path of the saved .nii.gz file.
    """
    # Same identity-fallback as save_deformation_maps, for a failed/rejected warp.
    if map_x is None or map_y is None:
        grid_x = np.arange(orig_w, dtype=np.float64)[None, :]
        grid_y = np.arange(orig_h, dtype=np.float64)[:, None]
        map_x = np.broadcast_to(grid_x, (orig_h, orig_w)).copy()
        map_y = np.broadcast_to(grid_y, (orig_h, orig_w)).copy()
    else:
        map_x = map_x.astype(np.float64)
        map_y = map_y.astype(np.float64)

    # Compose M_affine^-1 with (map_x, map_y): cv2.warpAffine samples
    # dst(x,y) = src(M^-1 . [x,y,1]), and cv2.remap then samples
    # moving_affine(map_x, map_y) — so the raw-moving-space coordinate for
    # output pixel (x,y) is M^-1 . [map_x(x,y), map_y(x,y), 1]. See the
    # exact two-step pixel warp in register_slice for the forward version
    # this is inverting.
    M_inv = cv2.invertAffineTransform(M_affine.astype(np.float64))  # (2, 3)
    total_x = M_inv[0, 0] * map_x + M_inv[0, 1] * map_y + M_inv[0, 2]
    total_y = M_inv[1, 0] * map_x + M_inv[1, 1] * map_y + M_inv[1, 2]

    grid_x      = np.arange(orig_w, dtype=np.float64)[None, :]
    grid_y      = np.arange(orig_h, dtype=np.float64)[:, None]
    identity_x  = np.broadcast_to(grid_x, (orig_h, orig_w))
    identity_y  = np.broadcast_to(grid_y, (orig_h, orig_w))

    pixel_size_mm = pixel_size_um / 1000.0
    disp_x_mm     = (total_x - identity_x) * pixel_size_mm
    disp_y_mm     = (total_y - identity_y) * pixel_size_mm

    # (H, W, 2) vector field -> SimpleITK vector image (already a project
    # dependency, no new import needed).
    field    = np.stack([disp_x_mm, disp_y_mm], axis=-1).astype(np.float64)
    sitk_img = sitk.GetImageFromArray(field, isVector=True)
    sitk_img.SetSpacing((pixel_size_mm, pixel_size_mm))
    sitk_img.SetOrigin((0.0, 0.0))

    out_path = os.path.join(DEFORM_FOLDER, f"{TARGET_CORE}_{slice_id}_displacement.nii.gz")
    sitk.WriteImage(sitk_img, out_path)
    logger.info(f"[{slice_id}] Displacement field NIfTI saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# L0: AKAZE AFFINE
# ─────────────────────────────────────────────────────────────────────────────

def constrain_affine(M: np.ndarray) -> np.ndarray:
    if M is None:
        return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S        = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out


def transform_is_sane(M: np.ndarray) -> bool:
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG


def downsample_for_detection(img, mask, max_dim):
    """
    Resize img (and its mask, nearest-neighbor to stay binary) so its
    longer side is at most max_dim. Returns (img_ds, mask_ds, scale), where
    scale = downsampled_size / original_size (so full_res_coord =
    ds_coord / scale). scale == 1.0 (no-op) if already <= max_dim.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img, mask, 1.0
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_ds  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_ds = None
    if mask is not None:
        mask_ds = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_ds, mask_ds, scale


def fit_affine_lstsq(from_pts, to_pts):
    """Ordinary (non-robust) least-squares affine fit: to ~= M @ from. Used
    for the full-res refit after downsampled detection — the input set has
    already been RANSAC/MAGSAC-filtered, so an unweighted LS fit here just
    recovers full-resolution precision from the coarser detection."""
    from_flat = from_pts.reshape(-1, 2).astype(np.float64)
    to_flat   = to_pts.reshape(-1, 2).astype(np.float64)
    n = from_flat.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    A[0::2, 0] = from_flat[:, 0]
    A[0::2, 1] = from_flat[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 3] = from_flat[:, 0]
    A[1::2, 4] = from_flat[:, 1]
    A[1::2, 5] = 1.0
    b[0::2] = to_flat[:, 0]
    b[1::2] = to_flat[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[sol[0], sol[1], sol[2]],
                     [sol[3], sol[4], sol[5]]], dtype=np.float64)


def _rescale_keypoints(kps, scale_factor):
    """
    Return a new tuple of cv2.KeyPoint with .pt (and .size) scaled by
    scale_factor, all other attributes preserved. Used to convert AKAZE
    keypoints detected on a downsampled image back into full-resolution
    pixel coordinates, so downstream consumers (save_inlier_plot, which
    overlays them on the full-res log images) don't need to know or care
    that detection happened at a different scale — match indices
    (m.queryIdx/m.trainIdx) and the inlier mask stay valid unchanged since
    rescaling doesn't reorder anything.
    """
    return tuple(
        cv2.KeyPoint(kp.pt[0] * scale_factor, kp.pt[1] * scale_factor,
                    kp.size * scale_factor, kp.angle, kp.response,
                    kp.octave, kp.class_id)
        for kp in kps
    )


def akaze_affine(fixed_log: np.ndarray, moving_log: np.ndarray, slice_id: str,
                 fixed_mask: np.ndarray | None = None,
                 moving_mask: np.ndarray | None = None):
    """
    Tissue-masked AKAZE detection (on a downsampled image) -> BFMatcher ->
    USAC_MAGSAC affine on the downsampled correspondences -> full-resolution
    least-squares refit on the surviving inliers.
    Returns (M, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask).
    M is None on failure. kp1/kp2 are returned in FULL-RESOLUTION pixel
    coordinates (rescaled from the downsampled detection) so callers that
    overlay them on the full-res log images (save_inlier_plot) need no
    changes.
    """
    fixed_ds,  fixed_mask_ds,  scale_f = downsample_for_detection(
        fixed_log, fixed_mask, AKAZE_DOWNSAMPLE_MAX_DIM)
    moving_ds, moving_mask_ds, scale_m = downsample_for_detection(
        moving_log, moving_mask, AKAZE_DOWNSAMPLE_MAX_DIM)
    scale = scale_f
    if abs(scale_f - scale_m) > 1e-6:
        logger.warning(f"[{slice_id}] Downsample scale mismatch fixed={scale_f} "
                       f"moving={scale_m} (unequal canvas shapes?) — using fixed-side scale.")
    inv_scale = 1.0 / scale

    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    # fixed_mask/moving_mask=None is a valid, intentional state here (no
    # precomputed tissue mask for this slice) — cv2's detectAndCompute
    # already treats mask=None as "detect over the whole image", so no
    # fallback mask needs to be synthesized.
    kp1_raw, des1 = detector.detectAndCompute(fixed_ds,  fixed_mask_ds)
    kp2_raw, des2 = detector.detectAndCompute(moving_ds, moving_mask_ds)

    n1, n2 = len(kp1_raw) if kp1_raw else 0, len(kp2_raw) if kp2_raw else 0
    if n1 > 0:
        coords = np.array([kp.pt for kp in kp1_raw])
        logger.info(f"[{slice_id}] AKAZE (tissue-masked, downsampled scale={scale:.3f}, "
                    f"{fixed_ds.shape[1]}x{fixed_ds.shape[0]}): n={n1}, "
                    f"y_range=[{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]")
    else:
        logger.info(f"[{slice_id}] AKAZE (tissue-masked, downsampled): n=0")

    if des1 is None or des2 is None or n1 < 4 or n2 < 4:
        logger.warning(f"[{slice_id}] Feature starvation (fixed={n1}, moving={n2}).")
        return None, 0, 0, [], [], [], np.array([])

    # Cap by response to avoid BFMatcher IMGIDX_ONE overflow
    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1_ds, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2_ds, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)
    logger.info(f"[{slice_id}] After cap: fixed={len(kp1_ds)}, moving={len(kp2_ds)}")

    # Rescale to full-res coordinates now — matches (below) are index-based
    # (m.queryIdx/m.trainIdx), so rescaling the keypoint list in place here
    # doesn't affect matching at all, and every caller downstream (including
    # the RANSAC/MAGSAC fit that follows) can just work in full-res pixel
    # space directly rather than needing to remember a scale factor.
    kp1 = _rescale_keypoints(kp1_ds, inv_scale)
    kp2 = _rescale_keypoints(kp2_ds, inv_scale)

    matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw      = matcher.knnMatch(des1, des2, k=2)
    good     = [m for m, n in raw
                if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    # Full-res coordinates now (kp1/kp2 already rescaled above).
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=RANSAC_METHOD,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] {RANSAC_METHOD} diverged.")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    # Full-resolution refit: the RANSAC/MAGSAC fit above ran on points that
    # came from a downsampled detection (even though they're expressed in
    # full-res coordinates now). Refitting via ordinary least squares on
    # just the accepted inliers recovers full-resolution precision — this
    # is the step that makes downsampled detection quality-neutral rather
    # than a resolution downgrade.
    inlier_idx = np.where(mask.ravel() == 1)[0]
    src_in = src_pts[inlier_idx]
    dst_in = dst_pts[inlier_idx]
    M = fit_affine_lstsq(dst_in, src_in)

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected (rot={rot:.1f}°).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, kp1, kp2, good, mask


# ─────────────────────────────────────────────────────────────────────────────
# L1: ROMAV2 MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        
        logger.info("Loading RoMaV2 on strict CPU (GPU is masked)...")
        _romav2_model = RoMaV2().to('cpu')
        _romav2_model.eval()
        
        # Disable dynamo compilation for CPU stability
        _romav2_model = torch._dynamo.disable(_romav2_model)
        
        _romav2_model.H_lr = ROMAV2_H
        _romav2_model.W_lr = ROMAV2_W
        _romav2_model.H_hr = ROMAV2_H_HR
        _romav2_model.W_hr = ROMAV2_W_HR
        
        logger.info("RoMaV2 model loaded.")
    return _romav2_model


def romav2_dense_warp(fixed_lin: np.ndarray, moving_lin: np.ndarray, slice_id: str,
                      orig_h: int, orig_w: int,
                      tissue_mask_full: np.ndarray | None = None):
    """
    Run RoMaV2 on a (fixed_img, moving_img) pair and return remap maps.

    Inputs are uint8, either (H, W) grayscale or (H, W, 3) RGB depending on
    ROMA_MODE. to_rgb_pil() handles the conversion to PIL RGB for RoMaV2.
    Inputs are expected to be affine-prealigned so the residual displacement
    is small — WARP_MAX_DISPLACEMENT_PX and WARP_CONFIDENCE_THRESH should be
    set accordingly.

    tissue_mask_full : uint8 (orig_h, orig_w) — 255=tissue, 0=background.
                       Background cells are forced to identity BEFORE upsampling
                       so featureless-canvas vectors cannot bleed into tissue
                       edges and create hard seam artefacts.

    Returns (map_x, map_y, n_confident, coverage_pct, mean_confidence)
    or      (None,  None,  0,           0.0,          0.0) on failure.
    """
    try:
        model = get_romav2_model()
        img_A = to_rgb_pil(fixed_lin)
        img_B = to_rgb_pil(moving_lin)

        with torch.no_grad():
            preds = model.match(img_A, img_B)

        warp_AB    = preds['warp_AB'].squeeze(0).cpu().numpy()    # (H_lr, W_lr, 2)
        overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy()  # (H_lr, W_lr)

        H_lr, W_lr = warp_AB.shape[:2]

        # Convert B-side coords from [-1,1] to full-resolution pixel space
        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

        confident_2d    = overlap_AB.reshape(H_lr, W_lr) >= WARP_CONFIDENCE_THRESH
        n_confident     = int(confident_2d.sum())
        coverage_pct    = n_confident / (H_lr * W_lr) * 100
        mean_confidence = float(overlap_AB.mean())

        # Identity coordinates at lr resolution
        grid_x_lr  = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
        grid_y_lr  = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
        identity_x, identity_y = np.meshgrid(grid_x_lr, grid_y_lr)

        # Apply confidence mask
        map_x_lr = np.where(confident_2d, b_coords_x, identity_x).astype(np.float32)
        map_y_lr = np.where(confident_2d, b_coords_y, identity_y).astype(np.float32)

        # Cap displacement magnitude
        disp_x = map_x_lr - identity_x
        disp_y = map_y_lr - identity_y
        mag    = np.sqrt(disp_x**2 + disp_y**2)
        excess = mag > WARP_MAX_DISPLACEMENT_PX
        if np.any(excess):
            scale    = np.where(excess, WARP_MAX_DISPLACEMENT_PX / (mag + 1e-8), 1.0)
            disp_x  *= scale
            disp_y  *= scale
            map_x_lr = (identity_x + disp_x).astype(np.float32)
            map_y_lr = (identity_y + disp_y).astype(np.float32)
            logger.info(
                f"[{slice_id}] Clipped {int(excess.sum())} warp vectors "
                f"> {WARP_MAX_DISPLACEMENT_PX}px."
            )

        # Zero out displacement in background regions BEFORE upsampling.
        if tissue_mask_full is not None:
            mask_lr    = cv2.resize(
                tissue_mask_full, (W_lr, H_lr), interpolation=cv2.INTER_NEAREST
            ).astype(bool)   # True = tissue
            background = ~mask_lr
            if np.any(background):
                map_x_lr[background] = identity_x[background]
                map_y_lr[background] = identity_y[background]
                logger.info(
                    f"[{slice_id}] Tissue mask: zeroed {int(background.sum())} "
                    f"background warp cells ({background.sum()/(H_lr*W_lr)*100:.1f}%)."
                )

        # Upsample to full resolution — INTER_CUBIC reduces ringing at the
        # hard tissue/background boundary compared to bilinear.
        map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        logger.info(
            f"[{slice_id}] RoMaV2 warp: {H_lr}×{W_lr} grid, "
            f"{coverage_pct:.1f}% confident, mean confidence={mean_confidence:.3f}"
        )
        return map_x, map_y, n_confident, coverage_pct, mean_confidence

    except Exception as exc:
        logger.error(f"[{slice_id}] RoMaV2 dense warp failed: {exc}")
        return None, None, 0, 0.0, 0.0
    

def visualize_displacement_field(map_x, map_y, tissue_mask=None,
                                 ck_log=None, slice_id="", save_path=None):
    """
    Four-panel displacement field visualization for small post-affine residuals.

    map_x, map_y : full-resolution remap arrays from romav2_dense_warp (float32)
    tissue_mask  : uint8 (H,W), 255=tissue — used to mask background in all panels
    ck_log       : uint8 (H,W) CK log image — used as underlay for context
    """
    h, w = map_x.shape

    # ── Identity grid → displacement field ───────────────────────────────────
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    disp_x = map_x - grid_x   # positive = rightward shift
    disp_y = map_y - grid_y   # positive = downward shift
    mag    = np.sqrt(disp_x**2 + disp_y**2)

    # Mask background
    if tissue_mask is not None:
        tissue = tissue_mask > 0
        disp_x = np.where(tissue, disp_x, 0.0)
        disp_y = np.where(tissue, disp_y, 0.0)
        mag    = np.where(tissue, mag,    0.0)

    mag_max = np.percentile(mag[mag > 0], 99) if np.any(mag > 0) else 1.0

    # ── Panel 1: HSV color-wheel encoding (standard optical flow vis) ─────────
    # Hue = direction (angle), Value = magnitude (scaled to 99th percentile)
    angle = np.arctan2(disp_y, disp_x)                    # -π to π
    hue   = (angle / (2 * np.pi) + 0.5) % 1.0             # 0-1
    val   = np.clip(mag / (mag_max + 1e-8), 0, 1)
    sat   = np.ones_like(hue)
    if tissue_mask is not None:
        val[~tissue] = 0.0
        sat[~tissue] = 0.0
    hsv_img = np.stack([hue, sat, val], axis=-1)
    rgb_flow = mcolors.hsv_to_rgb(hsv_img)

    # ── Panel 2: Magnitude heatmap ────────────────────────────────────────────
    mag_norm = np.clip(mag / (mag_max + 1e-8), 0, 1)

    # ── Panel 3 & 4: dx / dy component maps (diverging, centered at 0) ───────
    dx_norm = np.clip(disp_x / (mag_max + 1e-8), -1, 1)   # -1 to 1
    dy_norm = np.clip(disp_y / (mag_max + 1e-8), -1, 1)

    # ── Panel 5: Amplified quiver on coarse grid ──────────────────────────────
    GRID_STEP   = max(h, w) // 60     # ~60 arrows per axis
    ARROW_SCALE = 10.0                 # visual amplification factor — stated in title
    ys = np.arange(GRID_STEP // 2, h, GRID_STEP)
    xs = np.arange(GRID_STEP // 2, w, GRID_STEP)
    qx, qy = np.meshgrid(xs, ys)
    qdx = disp_x[qy, qx] * ARROW_SCALE
    qdy = disp_y[qy, qx] * ARROW_SCALE
    if tissue_mask is not None:
        valid = tissue_mask[qy, qx] > 0
    else:
        valid = np.ones(qx.shape, dtype=bool)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    fig.suptitle(f"Displacement field — {slice_id}  (99th pct magnitude: {mag_max:.2f}px)",
                 fontsize=12)

    # 1. HSV flow
    axes[0].imshow(rgb_flow)
    axes[0].set_title('HSV color wheel\n(hue=direction, brightness=magnitude)')
    axes[0].axis('off')
    # draw a small color-wheel legend in the corner
    _draw_colorwheel_legend(axes[0])

    # 2. Magnitude heatmap
    underlay = ck_log if ck_log is not None else np.zeros((h, w), dtype=np.uint8)
    axes[1].imshow(underlay, cmap='gray', alpha=0.4)
    im2 = axes[1].imshow(np.ma.masked_where(mag == 0, mag_norm),
                         cmap='hot', alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title(f'Magnitude\n(max shown = {mag_max:.1f}px)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                 label=f'Displacement (0–{mag_max:.1f}px)')

    # 3. dx component
    im3 = axes[2].imshow(np.ma.masked_where(~tissue if tissue_mask is not None
                                             else np.zeros((h,w), bool), dx_norm),
                         cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title(f'dx component\n(blue=left, red=right, ±{mag_max:.1f}px)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. dy component
    im4 = axes[3].imshow(np.ma.masked_where(~tissue if tissue_mask is not None
                                             else np.zeros((h,w), bool), dy_norm),
                         cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3].set_title(f'dy component\n(blue=up, red=down, ±{mag_max:.1f}px)')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Amplified quiver
    axes[4].imshow(underlay, cmap='gray')
    axes[4].quiver(qx[valid], qy[valid], qdx[valid], -qdy[valid],
                   mag[qy, qx][valid],
                   cmap='hot', scale=GRID_STEP * 4, scale_units='xy',
                   angles='xy', width=0.002, headwidth=4)
    axes[4].set_title(f'Amplified quiver\n(arrows ×{ARROW_SCALE:.0f}, color=magnitude)')
    axes[4].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _draw_colorwheel_legend(ax):
    """Small HSV color-wheel inset in the bottom-right corner."""
    size = 60
    cx, cy = size / 2, size / 2
    y, x   = np.ogrid[:size, :size]
    dx, dy = x - cx, y - cy
    r      = np.sqrt(dx**2 + dy**2)
    wheel  = np.zeros((size, size, 3))
    mask   = r <= cx
    h_     = (np.arctan2(dy, dx) / (2 * np.pi) + 0.5) % 1.0
    wheel[..., 0] = h_
    wheel[..., 1] = mask.astype(float)
    wheel[..., 2] = mask.astype(float)
    wheel_rgb = mcolors.hsv_to_rgb(wheel)
    # place in bottom-right using an inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="15%", height="15%", loc='lower right')
    axins.imshow(wheel_rgb)
    axins.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# CORE: REGISTER ONE SLICE PAIR
# ─────────────────────────────────────────────────────────────────────────────

def register_slice(fixed_np: np.ndarray, moving_np: np.ndarray,
                   fixed_mask: np.ndarray | None, moving_mask: np.ndarray | None,
                   slice_id: str | None = None):
    """
    Full pipeline: AKAZE affine → RoMaV2 dense warp.

    fixed_mask / moving_mask: uint8 (H, W) tissue masks (255=tissue, 0=bg),
    or None if no precomputed mask is available for that slice — see
    load_mask_or_none / warp_mask_forward above. fixed_mask must already be
    in the same geometry as fixed_np (i.e. already warped forward through
    whatever transform chain produced fixed_np — the caller in
    main()/process_pass is responsible for this, since register_slice has
    no way to know how many prior steps fixed_np has already been through).
    moving_mask is expected to be in raw/unwarped geometry, matching
    moving_np, which is always freshly loaded from disk.

    Returns (aligned_np, affine_np, elapsed, stats, success, M_affine,
             map_x, map_y, aligned_mask).
    affine_np is the L0-only result (useful for montage comparison).
    map_x / map_y are the full-resolution RoMaV2 remap arrays (or None).
    aligned_mask is moving_mask carried forward through the same transform
    chain applied to aligned_np — pass this back in as fixed_mask for
    whichever slice is registered next in the same pass.
    """
    start = time.time()
    sid   = slice_id or "unknown"

    # AKAZE always uses CK — EXCEPT for cores in AKAZE_COLOR_LUT_CORES, where
    # fixed_akaze_img/moving_akaze_img (below) use the color-LUT composite
    # instead. fixed_log/moving_log (CK) are still computed unconditionally:
    # they remain the basis for RoMaV2's 'ck_only' input, ROMA_MODE lin/log
    # needs, and every NCC measurement, regardless of the AKAZE override.
    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)
    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # Image AKAZE (L0) actually detects keypoints on — CK log by default,
    # color-LUT gray for cores in AKAZE_COLOR_LUT_CORES (see that config and
    # prepare_akaze_gray's docstring above).
    fixed_akaze_img  = prepare_akaze_gray(fixed_np, TARGET_CORE)
    moving_akaze_img = prepare_akaze_gray(moving_np, TARGET_CORE)

    # RoMaV2 input image — determined by ROMA_MODE
    if ROMA_MODE == 'clahe_visual_additive_2ch':
        fixed_roma_input  = prepare_visual_additive_2ch(fixed_np)
        moving_roma_input = prepare_visual_additive_2ch(moving_np)
    elif ROMA_MODE == 'clahe_visual_additive_3ch':
        fixed_roma_input  = prepare_visual_additive_3ch(fixed_np)
        moving_roma_input = prepare_visual_additive_3ch(moving_np)
    elif ROMA_MODE == 'clahe_visual_additive_7ch':
        fixed_roma_input  = prepare_visual_additive_7ch(fixed_np)
        moving_roma_input = prepare_visual_additive_7ch(moving_np)
    elif ROMA_MODE == 'clahe_direct_rgb_2ch':
        fixed_roma_input  = prepare_direct_rgb_2ch(fixed_np)
        moving_roma_input = prepare_direct_rgb_2ch(moving_np)
    else:  # 'dapi_clahe'
        fixed_roma_input  = clahe_normalize(fixed_np[DAPI_CHANNEL_IDX].astype(np.float32))
        moving_roma_input = clahe_normalize(moving_np[DAPI_CHANNEL_IDX].astype(np.float32))
    c    = fixed_np.shape[0]

    # NCC input — always CK log, matching the linear script
    # (akaze_linear_romav2_warp_map.py), regardless of what ROMA_MODE feeds
    # RoMaV2. This keeps the acceptance gate judging the same channel AKAZE
    # detected on, so a geometrically-correct affine can't get reverted just
    # because a fused/alternate channel (DAPI, color-LUT blend, CLAHE, etc.)
    # happens to correlate poorly in log-NCC space.
    fixed_ncc_img  = fixed_log
    moving_ncc_img = moving_log

    # Tissue mask now comes from the caller (precomputed sibling file,
    # carried forward through prior transform steps for the fixed side —
    # see docstring above). None means "no mask available", which every
    # downstream consumer here already treats as "match/measure unmasked".
    if fixed_mask is None:
        tissue_frac = 1.0
        logger.info(f"[{sid}] No fixed-side tissue mask — treating canvas as fully covered.")
    else:
        tissue_frac = float(np.count_nonzero(fixed_mask)) / fixed_mask.size
        logger.info(f"[{sid}] Tissue mask: {tissue_frac*100:.1f}% of canvas covered.")

    # ── NCC raw (before any alignment) — log space, masked ───────────────────
    ncc_raw = measure_ncc(
        fixed_ncc_img.astype(np.float32),
        moving_ncc_img.astype(np.float32),
        fixed_mask,
    )
    logger.info(f"[{sid}] NCC raw (before alignment): {ncc_raw:.4f}")

    # ── L0: AKAZE affine — detects on fixed_akaze_img/moving_akaze_img ───────
    # (CK log, or color-LUT gray for AKAZE_COLOR_LUT_CORES — see above).
    M_affine, n_matches, n_inliers, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_akaze_img, moving_akaze_img, sid,
                     fixed_mask=fixed_mask,
                     moving_mask=moving_mask)

    akaze_ok = M_affine is not None
    if not akaze_ok:
        logger.warning(f"[{sid}] AKAZE failed — RoMaV2 will run on raw images.")
        M_affine = np.eye(2, 3, dtype=np.float64)

    # Save AKAZE inlier plot — same image AKAZE actually detected on.
    if len(kp1) > 0:
        try:
            save_inlier_plot(fixed_akaze_img, moving_akaze_img, kp1, kp2,
                             good_matches, inlier_mask, sid, akaze_ok)
        except Exception as exc:
            logger.warning(f"[{sid}] Inlier plot failed: {exc}")

    # Apply affine to all channels
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    affine_np = np.clip(moving_affine_vol, 0, 65535).astype(np.uint16)

    # Affine-prealigned RoMaV2 input — required for both the dense warp and QC plots
    moving_roma_input_affine = cv2.warpAffine(
        moving_roma_input, M_affine, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Affine-prealigned CK log — used only for the NCC acceptance gate, kept
    # in lockstep with the linear script so the gate always judges CK
    # regardless of ROMA_MODE.
    moving_log_affine = cv2.warpAffine(
        moving_log, M_affine, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # ── NCC affine — log space, masked (CK only, matches linear script) ──────
    ncc_affine = measure_ncc(
        fixed_ncc_img.astype(np.float32),
        moving_log_affine.astype(np.float32),
        fixed_mask,
    )
    logger.info(f"[{sid}] NCC after affine: {ncc_affine:.4f}")

    # ── Affine NCC acceptance — revert to identity if affine made things worse ─
    # AKAZE succeeded geometrically (enough inliers, sane matrix) but the affine
    # transform could still hurt NCC — e.g. a slightly wrong rotation on a low-
    # texture core.  If the affine NCC is worse than raw, fall back to identity
    # so RoMaV2 runs on the unmodified images rather than a degraded baseline.
    # "Worse" = ncc_affine > ncc_raw (less negative), i.e. no improvement at all.
    # We use a strict threshold of 0.0 (any regression reverts) because even a
    # small affine degradation compounds once RoMaV2's acceptance is relative to
    # ncc_affine — a worse baseline makes the warp easier to accept spuriously.
    if akaze_ok and ncc_affine > ncc_raw:
        logger.warning(
            f"[{sid}] Affine NCC ({ncc_affine:.4f}) is worse than raw "
            f"({ncc_raw:.4f}) — reverting affine to identity so RoMaV2 "
            "sees the unmodified images."
        )
        akaze_ok          = False
        M_affine          = np.eye(2, 3, dtype=np.float64)
        moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
        for ch in range(moving_np.shape[0]):
            moving_affine_vol[ch] = moving_np[ch].astype(np.float32)
        affine_np         = moving_np.copy()
        
        # Reset the RoMaV2 input to raw (removing the legacy log_affine line)
        moving_roma_input_affine = moving_roma_input.copy()
        moving_log_affine       = moving_log.copy()
        ncc_affine               = ncc_raw

    # ── L1: RoMaV2 warp on affine-prealigned images ──────────────────────────
    # Input image is determined by ROMA_MODE — not necessarily linear.
    # NCC is always measured on CK log regardless of mode.
    warp_ok         = False
    ncc_warp        = 0.0
    n_confident     = 0
    coverage_pct    = 0.0
    mean_confidence = 0.0
    aligned_np      = affine_np.copy()   # default: affine-only
    map_x = map_y = None                 # returned for deformation saving

    if tissue_frac >= MASK_MIN_FRAC:
        # Pass the pre-computed arrays directly
        map_x, map_y, n_confident, coverage_pct, mean_confidence = romav2_dense_warp(
            fixed_roma_input, moving_roma_input_affine, sid, h, w,
            tissue_mask_full=fixed_mask,
        )

        if map_x is not None:
            visualize_displacement_field(
                map_x, map_y,
                tissue_mask=fixed_mask,
                ck_log=fixed_log,
                slice_id=sid,
                save_path=os.path.join(OUTPUT_FOLDER, f"{sid}_displacement.png"),
            )
            # Apply warp to affine-prealigned volume
            warped_channels = []
            for ch in range(c):
                warped_channels.append(cv2.remap(
                    moving_affine_vol[ch],
                    map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ))
            warp_candidate = np.stack(warped_channels, axis=0).astype(np.uint16)

            # Blank-output sanity check
            ck_out = warp_candidate[CK_CHANNEL_IDX]
            if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
                logger.warning(
                    f"[{sid}] RoMaV2 output nearly blank — reverting to affine."
                )
                map_x = map_y = None   # mark as failed so identity is stored
            else:
                # NCC after warp — log space, CK only (matches linear script):
                # re-derive log-normalized CK from the warped CK channel so
                # the acceptance gate stays on the same channel as ncc_raw /
                # ncc_affine, regardless of ROMA_MODE.
                _, warped_log = prepare_ck(ck_out.astype(np.float32))
                ncc_warp = measure_ncc(
                    fixed_ncc_img.astype(np.float32),
                    warped_log.astype(np.float32),
                    fixed_mask,
                )
                logger.info(f"[{sid}] NCC after RoMaV2 warp: {ncc_warp:.4f}")

                # ── NCC monotonic acceptance — mirrors bspline strategy ───────
                # Accept the warp only if it improves NCC over the affine
                # baseline by at least WARP_NCC_MIN_IMPROVEMENT (relative).
                # NCC is negative — improvement means becoming more negative.
                # This guards against RoMaV2 producing a plausible-looking warp
                # that nonetheless degrades alignment (e.g. on low-texture
                # regions where the confidence filter alone is insufficient).
                if abs(ncc_affine) > 1e-9:
                    relative_improvement = (ncc_affine - ncc_warp) / abs(ncc_affine)
                else:
                    relative_improvement = 0.0

                if relative_improvement < WARP_NCC_MIN_IMPROVEMENT:
                    logger.warning(
                        f"[{sid}] RoMaV2 NCC ({ncc_warp:.4f}) did not improve enough "
                        f"over affine baseline ({ncc_affine:.4f}): "
                        f"relative gain={relative_improvement*100:.2f}% < required "
                        f"{WARP_NCC_MIN_IMPROVEMENT*100:.0f}% "
                        "— reverting to affine-only result."
                    )
                    map_x = map_y = None   # store identity so downstream never branches
                else:
                    aligned_np = warp_candidate
                    warp_ok    = True
                    logger.info(
                        f"[{sid}] RoMaV2 warp accepted "
                        f"(NCC gain={relative_improvement*100:.2f}%)."
                    )
        else:
            logger.warning(f"[{sid}] RoMaV2 warp failed — using affine-only result.")
    else:
        logger.warning(
            f"[{sid}] Tissue fraction {tissue_frac*100:.1f}% < "
            f"{MASK_MIN_FRAC*100:.0f}% — skipping RoMaV2."
        )

    # ── Save deformation maps ─────────────────────────────────────────────────
    # Always saved — even on failure an identity warp is stored so downstream
    # CellPose mask warping code can load without branching.
    try:
        save_deformation_maps(
            slice_id  = sid,
            M_affine  = M_affine,
            map_x     = map_x,
            map_y     = map_y,
            akaze_ok  = akaze_ok,
            warp_ok   = warp_ok,
            orig_h    = h,
            orig_w    = w,
        )
    except Exception as exc:
        logger.error(f"[{sid}] Failed to save deformation maps: {exc}")

    try:
        save_deformation_nifti(
            slice_id = sid,
            M_affine = M_affine,
            map_x    = map_x,
            map_y    = map_y,
            orig_h   = h,
            orig_w   = w,
        )
    except Exception as exc:
        logger.error(f"[{sid}] Failed to save displacement NIfTI: {exc}")

    # Overall success: at minimum affine must have worked or warp improved things
    success = akaze_ok or warp_ok

    # NCC improvement at each stage
    def pct_improvement(before, after):
        if abs(before) > 1e-9:
            return (before - after) / abs(before) * 100.0
        return 0.0

    ncc_affine_improvement = pct_improvement(ncc_raw,    ncc_affine)
    ncc_warp_improvement   = pct_improvement(ncc_affine, ncc_warp) if warp_ok else 0.0
    ncc_total_improvement  = pct_improvement(ncc_raw,    ncc_warp if warp_ok else ncc_affine)

    logger.info(
        f"[{sid}] NCC summary: raw={ncc_raw:.4f} → "
        f"affine={ncc_affine:.4f} ({ncc_affine_improvement:+.1f}%) → "
        f"warp={ncc_warp:.4f} ({ncc_warp_improvement:+.1f}%) | "
        f"total={ncc_total_improvement:+.1f}%"
    )

    # Decompose affine for stats
    U, S, Vt  = np.linalg.svd(M_affine[:2, :2])
    R         = U @ Vt
    rot       = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale_pct = (float(np.mean(S)) - 1.0) * 100.0
    shear_pct = (float(S[0] / S[1]) - 1.0) * 100.0 if S[1] > 1e-6 else 0.0

    stats = dict(
        akaze_ok            = akaze_ok,
        warp_ok             = warp_ok,
        n_matches           = n_matches,
        n_inliers           = n_inliers,
        rotation_deg        = round(rot, 3),
        tx                  = round(float(M_affine[0, 2]), 3),
        ty                  = round(float(M_affine[1, 2]), 3),
        scale_pct           = round(scale_pct, 3),
        shear_pct           = round(shear_pct, 3),
        n_confident         = n_confident,
        coverage_pct        = round(coverage_pct, 2),
        mean_confidence     = round(mean_confidence, 4),
        ncc_raw             = round(float(ncc_raw),    6),
        ncc_affine          = round(float(ncc_affine), 6),
        ncc_warp            = round(float(ncc_warp),   6),
        ncc_affine_improv   = round(float(ncc_affine_improvement), 2),
        ncc_warp_improv     = round(float(ncc_warp_improvement),   2),
        ncc_total_improv    = round(float(ncc_total_improvement),  2),
    )

    # Save interim plot
    try:
        save_registration_plot(
            fixed_roma_input, moving_roma_input, moving_roma_input_affine,
            map_x, map_y,
            ncc_raw, ncc_affine, ncc_warp,
            akaze_ok, warp_ok, sid,
        )
    except Exception as exc:
        logger.warning(f"[{sid}] Registration plot failed: {exc}")

    # Save vector field plot if warp succeeded
    try:
        if warp_ok and map_x is not None and map_y is not None:
            save_deformation_quiver_plot(map_x, map_y, sid, step=150)
    except Exception as exc:
        logger.warning(f"[{sid}] Vector field plot failed: {exc}")

    # Carry the moving-side mask through the exact same transform chain
    # just applied to the pixels (M_affine always; map_x/map_y only when
    # the dense warp was actually accepted — they're None otherwise, and
    # warp_mask_forward treats that as "affine-only", matching aligned_np).
    aligned_mask = warp_mask_forward(moving_mask, M_affine, map_x, map_y, h, w)

    return (aligned_np, affine_np, time.time() - start, stats, success,
            M_affine, map_x, map_y, aligned_mask)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def save_inlier_plot(fixed_log, moving_log, kp1, kp2,
                     good_matches, inlier_mask, slice_id, akaze_ok):
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    h, w    = fixed_log.shape
    gap     = 6
    canvas  = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_log,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_log, cv2.COLOR_GRAY2BGR)

    inlier_matches = ([m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep]
                      if len(inlier_mask) > 0 else [])

    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in
                          cv2.cvtColor(np.uint8([[[hue, 220, 220]]]),
                                       cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), color_bgr, 1, cv2.LINE_AA)

    status      = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)
    title       = f"{slice_id}  inliers={len(inlier_matches)}  [{status}]"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    scale       = max(0.8, canvas.shape[1] / 3000)
    thickness   = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    cv2.putText(canvas, title, ((canvas.shape[1] - tw) // 2, th + 10),
                font, scale, title_color, thickness, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_inliers.png"), canvas)
    logger.info(f"[{slice_id}] Inlier plot saved ({len(inlier_matches)} inliers).")


def save_registration_plot(fixed_img, moving_img, moving_img_affine,
                           map_x, map_y,
                           ncc_raw, ncc_affine, ncc_warp,
                           akaze_ok, warp_ok, slice_id):
    """
    Plots the registration progress using the exact inputs fed to RoMaV2.
    Dynamically handles 2D (grayscale) or 3D (RGB) array inputs via alpha blending.
    """
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    f         = normalize_pct(fixed_img)
    m_raw     = normalize_pct(moving_img)
    m_affine  = normalize_pct(moving_img_affine)
    src_remap = moving_img_affine.astype(np.float32)

    is_rgb = fixed_img.ndim == 3

    def make_overlay(img_fixed, img_moving):
        if is_rgb:
            # 3D RGB: 50/50 Alpha blend to preserve mapped colors without distortion
            return img_fixed * 0.5 + img_moving * 0.5
        else:
            # 2D Grayscale: Standard Red/Green overlay
            return np.dstack((img_fixed, img_moving, np.zeros_like(img_fixed)))

    lbl_base  = "Blended Overlay" if is_rgb else "Fixed (R) vs Moving (G)"
    label     = f"{ROMA_MODE} input"

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(make_overlay(f, m_raw))
    axes[0].set_title(f"Raw: {lbl_base} [{label}]\nNCC={ncc_raw:.4f}", fontsize=10)
    axes[0].axis('off')

    affine_status = "AKAZE OK" if akaze_ok else "identity"
    axes[1].imshow(make_overlay(f, m_affine))
    axes[1].set_title(
        f"Affine: {lbl_base} [{affine_status}]\nNCC={ncc_affine:.4f}", fontsize=10
    )
    axes[1].axis('off')

    if map_x is not None and warp_ok:
        warped = cv2.remap(
            src_remap, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        axes[2].imshow(make_overlay(f, normalize_pct(warped)))
        axes[2].set_title(
            f"RoMaV2 Warp: {lbl_base} [{label}]\nNCC={ncc_warp:.4f}", fontsize=10
        )
    else:
        axes[2].imshow(make_overlay(f, m_affine))
        axes[2].set_title("RoMaV2 FAILED\n(affine shown)", fontsize=10)
    axes[2].axis('off')

    if map_x is not None and warp_ok:
        h_img, w_img = map_x.shape
        id_x     = np.arange(w_img, dtype=np.float32)[None, :]
        id_y     = np.arange(h_img, dtype=np.float32)[:, None]
        disp_mag = np.sqrt((map_x - id_x)**2 + (map_y - id_y)**2)
        disp_display = cv2.resize(disp_mag, (512, 512), interpolation=cv2.INTER_AREA)
        im = axes[3].imshow(disp_display, cmap='hot', vmin=0,
                            vmax=np.percentile(disp_mag, 99))
        axes[3].set_title("Displacement magnitude (px)\n(residual after affine)", fontsize=10)
        plt.colorbar(im, ax=axes[3], fraction=0.03, pad=0.02)
    else:
        axes[3].axis('off')
        axes[3].set_title("No warp field", fontsize=10)
    axes[3].axis('off')

    status = "SUCCESS" if (akaze_ok or warp_ok) else "FAILED"
    if abs(ncc_raw) > 1e-9:
        total_improv = (ncc_raw - (ncc_warp if warp_ok else ncc_affine)) / abs(ncc_raw) * 100
    else:
        total_improv = 0.0
    fig.suptitle(
        f"{slice_id}  "
        f"NCC: raw={ncc_raw:.4f} -> affine={ncc_affine:.4f} -> warp={ncc_warp:.4f}  "
        f"total delta={total_improv:+.1f}%  [{status}]",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{slice_id}_registration.png"),
        dpi=100, bbox_inches='tight',
    )
    plt.close(fig)


def save_deformation_quiver_plot(map_x, map_y, slice_id, step=150):
    """
    Renders a vector field of the dense displacement map.
    Subsampled to prevent memory exhaustion on gigapixel mIF images.
    """
    out_dir = os.path.join(OUTPUT_FOLDER, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    h, w = map_x.shape
    Y, X = np.mgrid[0:h, 0:w]
    
    # Subsample the grids and maps
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    map_x_sub = map_x[::step, ::step]
    map_y_sub = map_y[::step, ::step]
    
    # Calculate relative displacement
    dX = map_x_sub - X_sub
    dY = map_y_sub - Y_sub
    
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"{slice_id} Deformation Vector Field\n(Subsampled 1:{step} px)")
    
    plt.quiver(
        X_sub, Y_sub, dX, dY, 
        angles='xy', scale_units='xy', scale=1, 
        color='red', alpha=0.6, width=0.002
    )
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"{slice_id}_vector_field.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[{slice_id}] Vector field plot saved.")


def generate_qc_montage(vol, output_folder, slice_ids=None,
                        channel_idx=6, channel_name="CK",
                        title_suffix="AKAZE_RoMaV2_Linear"):
    n_slices = vol.shape[0]
    if n_slices < 2:
        return
    logger.info(f"Generating QC montage [{title_suffix}]...")
    all_pairs = [(i, i + 1) for i in range(n_slices - 1)]
    n_cols    = 5
    n_rows    = (len(all_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1:               axes = axes.reshape(1, -1)
    elif n_cols == 1:               axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()
    for idx, (z1, z2) in enumerate(all_pairs):
        s1 = vol[z1, channel_idx].astype(np.float32)
        s2 = vol[z2, channel_idx].astype(np.float32)
        overlay = np.dstack((normalize_pct(s1), normalize_pct(s2), np.zeros_like(s1)))
        axes_flat[idx].imshow(overlay)
        lbl1 = slice_ids[z1] if slice_ids else z1
        lbl2 = slice_ids[z2] if slice_ids else z2
        axes_flat[idx].set_title(f"ID{lbl1} to ID{lbl2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')
    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')
    core = os.path.basename(output_folder)
    fig.suptitle(f"Registration QC {title_suffix}: {core}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f"{core}_QC_Montage_{channel_name}_{title_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"AKAZE → RoMaV2 (Linear) Registration — {TARGET_CORE}")
    logger.info(
        f"AKAZE threshold={AKAZE_THRESHOLD} | Max keypoints={AKAZE_MAX_KEYPOINTS} | "
        f"RoMaV2 {ROMAV2_H}×{ROMAV2_W} | "
        f"Warp cap={WARP_MAX_DISPLACEMENT_PX}px | "
        f"Confidence thresh={WARP_CONFIDENCE_THRESH}"
    )

    raw_files = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    n_slices  = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    allowed_positions = load_slice_filter(SLICE_FILTER_YAML, TARGET_CORE)
    if allowed_positions is not None:
        original_count = len(file_list)
        file_list      = [f for i, f in enumerate(file_list) if i in allowed_positions]
        n_slices       = len(file_list)
        excluded       = original_count - n_slices
        logger.info(
            f"Slice filter active: keeping {n_slices}/{original_count} slices "
            f"(positions {sorted(allowed_positions)}), {excluded} excluded."
        )
        if n_slices == 0:
            logger.error("Slice filter excluded all slices.")
            sys.exit(1)
    else:
        logger.info(f"No slice filter — using all {n_slices} slices.")

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in = tifffile.imread(file_list[0])
        tifffile.imwrite(
            os.path.join(OUTPUT_FOLDER,
                         f"{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif"),
            vol_in[np.newaxis], photometric='minisblack',
            metadata={
                'axes': 'ZCYX', 'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        sys.exit(0)

    get_romav2_model()  # preload

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)
    c, target_h, target_w = _center_arr.shape
    logger.info(f"Shape: C={c}, H={target_h}, W={target_w}")

    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
        arr = tifffile.imread(file_list[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            # Previously silently pad/cropped to (target_h, target_w) via
            # conform_slice. Removed: a pad/crop here shifts the raw pixel
            # geometry without touching the precomputed tissue mask file on
            # disk, so the mask would silently stop lining up with the
            # slice it's supposed to describe. Failing loudly instead so a
            # real shape problem (wrong crop, stale file) gets fixed at the
            # source rather than papered over here.
            raise ValueError(
                f"{os.path.basename(file_list[idx])} is {arr.shape[1:]}, "
                f"expected ({target_h}, {target_w}) — re-crop/re-conform "
                "this slice (and its mask) upstream rather than reshaping it here."
            )
        return arr

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    affine_vol              = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    affine_vol[center_idx]  = center_raw
    del center_raw
    logger.info(f"Anchor: slice index {center_idx} (ID {slice_ids[center_idx]})")

    # mask_vol[i] holds the tissue mask for aligned_vol[i]'s *current*
    # geometry — i.e. already carried through whatever transform chain got
    # that slice to where it is. Only the anchor is raw, so only the anchor
    # loads its mask directly; every other position gets its mask from
    # register_slice's aligned_mask return value as the pass walks outward
    # (see process_pass below and warp_mask_forward above).
    mask_vol             = [None] * n_slices
    mask_vol[center_idx] = load_mask_or_none(file_list[center_idx], (target_h, target_w))

    # Save identity deformation for the anchor slice so warp_cellpose_masks.py
    # never skips it — applying a no-op transform is the correct behaviour.
    center_sid = f"Z{center_idx:03d}_ID{slice_ids[center_idx]:03d}"
    try:
        save_deformation_maps(
            slice_id = center_sid,
            M_affine = np.eye(2, 3, dtype=np.float64),
            map_x    = None,   # triggers identity grid inside save_deformation_maps
            map_y    = None,
            akaze_ok = False,
            warp_ok  = False,
            orig_h   = target_h,
            orig_w   = target_w,
        )
        logger.info(f"Anchor [{center_sid}] identity deformation map saved.")
    except Exception as exc:
        logger.warning(f"Anchor [{center_sid}] Failed to save identity map: {exc}")

    try:
        save_deformation_nifti(
            slice_id = center_sid,
            M_affine = np.eye(2, 3, dtype=np.float64),
            map_x    = None,
            map_y    = None,
            orig_h   = target_h,
            orig_w   = target_w,
        )
    except Exception as exc:
        logger.warning(f"Anchor [{center_sid}] Failed to save identity displacement NIfTI: {exc}")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")
        for i in indices:
            real_id     = slice_ids[i]
            fixed_np    = aligned_vol[i + fixed_offset]
            moving_np   = load_slice(i)
            sid         = f"Z{i:03d}_ID{real_id:03d}"

            fixed_mask_i  = mask_vol[i + fixed_offset]
            moving_mask_i = load_mask_or_none(file_list[i], (target_h, target_w))

            try:
                aligned_np, affine_np, runtime, stats, success, M_final, map_x, map_y, aligned_mask = \
                    register_slice(fixed_np, moving_np, fixed_mask_i, moving_mask_i, slice_id=sid)
            except Exception as exc:
                logger.error(f"[{sid}] register_slice crashed: {exc} — raw fallback.")
                aligned_np   = moving_np.copy()
                affine_np    = moving_np.copy()
                aligned_mask = moving_mask_i   # identity transform — mask stays in raw geometry
                runtime      = 0.0
                stats      = dict(
                    akaze_ok=False, warp_ok=False,
                    n_matches=0, n_inliers=0,
                    rotation_deg=0, tx=0, ty=0, scale_pct=0, shear_pct=0,
                    n_confident=0, coverage_pct=0.0, mean_confidence=0.0,
                    ncc_raw=0, ncc_affine=0, ncc_warp=0,
                    ncc_affine_improv=0, ncc_warp_improv=0, ncc_total_improv=0,
                )
                success = False
                # Save identity deformation so downstream code can still load
                try:
                    save_deformation_maps(
                        slice_id=sid,
                        M_affine=np.eye(2, 3, dtype=np.float64),
                        map_x=None, map_y=None,
                        akaze_ok=False, warp_ok=False,
                        orig_h=target_h, orig_w=target_w,
                    )
                except Exception as save_exc:
                    logger.warning(f"[{sid}] Failed to save identity map: {save_exc}")
                try:
                    save_deformation_nifti(
                        slice_id=sid,
                        M_affine=np.eye(2, 3, dtype=np.float64),
                        map_x=None, map_y=None,
                        orig_h=target_h, orig_w=target_w,
                    )
                except Exception as save_exc:
                    logger.warning(f"[{sid}] Failed to save identity displacement NIfTI: {save_exc}")


            aligned_vol[i] = aligned_np
            affine_vol[i]  = affine_np
            mask_vol[i]    = aligned_mask
            del moving_np

            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK_RAW"
            if not success:
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str}")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | "
                f"AKAZE: {stats['n_inliers']} inliers | "
                f"NCC: raw={stats['ncc_raw']:.4f} "
                f"affine={stats['ncc_affine']:.4f} ({stats['ncc_affine_improv']:+.1f}%) "
                f"warp={stats['ncc_warp']:.4f} ({stats['ncc_warp_improv']:+.1f}%) | "
                f"Conf: {stats['coverage_pct']:.1f}% cells | "
                f"t: {runtime:.2f}s | {status_str}"
            )

            registration_stats.append({
                "Direction":          direction,
                "Slice_Z":            i,
                "Slice_ID":           real_id,
                "AKAZE_OK":           stats["akaze_ok"],
                "Warp_OK":            stats["warp_ok"],
                "N_Matches":          stats["n_matches"],
                "N_Inliers":          stats["n_inliers"],
                "Rotation_Deg":       stats["rotation_deg"],
                "Shift_X_px":         stats["tx"],
                "Shift_Y_px":         stats["ty"],
                "Scale_Pct":          stats["scale_pct"],
                "Shear_Pct":          stats["shear_pct"],
                "N_Confident":        stats["n_confident"],
                "Coverage_Pct":       stats["coverage_pct"],
                "Mean_Confidence":    stats["mean_confidence"],
                "NCC_Raw":            stats["ncc_raw"],
                "NCC_Affine":         stats["ncc_affine"],
                "NCC_Warp":           stats["ncc_warp"],
                "NCC_Affine_Improv":  stats["ncc_affine_improv"],
                "NCC_Warp_Improv":    stats["ncc_warp_improv"],
                "NCC_Total_Improv":   stats["ncc_total_improv"],
                "Success":            success,
                "Status":             status_str,
                "Runtime_s":          round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    df.to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_RoMaV2_Linear.csv"),
        index=False,
    )
    n_ok = int((df["Status"] == "SUCCESS").sum())
    n_fb = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(f"Complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fb}")

    # Write final volume
    out_tiff = os.path.join(OUTPUT_FOLDER,
                            f"{TARGET_CORE}_AKAZE_RoMaV2_Linear_Aligned.ome.tif")
    logger.info(f"Writing volume to {out_tiff}")
    try:
        tifffile.imwrite(
            out_tiff, aligned_vol,
            photometric='minisblack',
            metadata={
                'axes': 'ZCYX', 'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM, 'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM, 'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM, 'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        logger.info("Volume written.")
    except Exception as exc:
        logger.error(f"Volume write failed: {exc}")

    # QC montages
    try:
        generate_qc_montage(affine_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="AKAZE_Affine")
    except Exception as exc:
        logger.error(f"Affine montage failed: {exc}")
    del affine_vol

    try:
        generate_qc_montage(aligned_vol, OUTPUT_FOLDER, slice_ids=slice_ids,
                            channel_idx=CK_CHANNEL_IDX, channel_name="CK",
                            title_suffix="AKAZE_RoMaV2_Linear")
    except Exception as exc:
        logger.error(f"Final montage failed: {exc}")
    del aligned_vol

    logger.info("Done.")


if __name__ == "__main__":
    main()